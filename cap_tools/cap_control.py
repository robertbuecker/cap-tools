import os
from cap_tools.finalization import Finalization, FinalizationCollection
import glob
import time
import pandas as pd
from cap_tools.cell_list import CellList
from typing import *
import queue

class CAPControl:
    """Class to control CrysAlisPro by generating macros or listen mode (planned)
    """
    
    def __init__(self, work_folder: str, 
                 cmd_folder: Optional[str] = None,
                 message_func: Optional[Union[Callable[[str], None], queue.Queue]] = None,
                 cmd_func: Optional[Union[Callable[[str], None], queue.Queue]] = None):
        
        self._work_folder = work_folder
        self._cmd_folder = cmd_folder
        self.message_func = message_func
        self.cmd_func = cmd_func
        
    def send_message(self, msg):
        if isinstance(self.message_func, queue.Queue):
            self.message_func.put(msg)
        elif self.message_func is None:
            print(msg)
        else:
            self.message_func(msg)
            
    def send_cmd(self, cmd):
        if isinstance(self.cmd_func, queue.Queue):
            self.cmd_func.put(cmd)
        elif self.cmd_func is None:
            print(f'Discarding CAP command: {cmd}')
        else:
            self.cmd_func(cmd)
        
    def write_macro(self, macro_name: str, macro: Union[List[str], str],
                    append: bool = False):
        
        if not macro_name.endswith('.mac'): macro_name += '.mac'
        if not os.path.split(macro_name)[0]:
            macro_name = os.path.join(self._work_folder, macro_name)
        
        if isinstance(macro, list): macro = '\n'.join(macro)
        
        with open(macro_name, 'a' if append else 'w') as fh:
            fh.write(macro)
            
        self.send_cmd(f'script {macro_name}')
            
        return f'script {macro_name}'     
    
class CAPMergeFinalize(CAPControl):
    
    def __init__(self, path: str, clusters: Dict[int, CellList],
                 message_func: Optional[Callable[[str], None]] = None,
                 cmd_func: Optional[Callable[[str], None]] = None):
        
        super().__init__(work_folder=os.path.split(path)[0], message_func=message_func, cmd_func=cmd_func)
        self.path = path
        self.clusters = clusters
        
    @property
    def node_info_fn(self) -> str:
        #TODO Factor this into clustering
        return self.path + '_nodes.csv'
    
    @property
    def macro_fn(self) -> str:
        return self.path + '.mac'
    
    @property
    def merge_files_found(self) -> bool:
        return os.path.exists(self.macro_fn) and os.path.exists(self.node_info_fn)
    
    def cluster_merge(self, write_mac: bool = True, delete_existing: bool = False):
        
        cmds = []
        old_fns = []
        
        with open(self.node_info_fn, 'w') as ifh:
            # TODO factor this into cluster class. Why is this here?!
            ifh.write('Name,File path,Cluster,Data sets,Merge code\n')
            merged_cids = []
            for ii, (c_id, cluster) in enumerate(self.clusters.items()):
                out_paths, in_paths, out_codes, out_info = cluster.get_merging_paths(prefix=f'C{c_id}', short_form=True)                
                for out, (in1, in2), code, info in zip(out_paths, in_paths, out_codes, out_info):
                    cmds.append(f'xx proffitmerge "{out}" "{in1}" "{in2}"')
                    ifh.write(f'{os.path.basename(out)},{out},{c_id},{info},{code}\n')
                if delete_existing:
                    old_fns += glob.glob(out + '*.*')
                  
                print(f'Full-cluster merge for cluster {c_id}: {out_paths[-1]}')
                merged_cids.append(c_id)

        for fn in old_fns:        
            os.remove(fn)
            print(f'Deleting {fn}')      
        
        if write_mac:
            mac_cmd = self.write_macro(self.macro_fn, cmds)
            self.send_message(f'Wrote merging macro to {self.macro_fn}. Copied CAP command to clipboard.')
            self.send_cmd(mac_cmd)
        else:
            mac_cmd = None
                
        return mac_cmd, cmds

    def cluster_finalize(self, 
                        res_limit: float = 0.8,
                        finalization_timeout: float = 10):    

        _, cmds = self.cluster_merge(write_mac=False, delete_existing=True)

        tmp_folder = os.path.join(os.path.dirname(self.path), 'tmp')
        os.makedirs(tmp_folder, exist_ok=True)        
        for fn in glob.glob(os.path.join(tmp_folder, '*.xml')):  
            os.remove(fn)
            print(f'Deleting {fn}')
                      
        fc = FinalizationCollection.from_csv(self.node_info_fn, allow_missing=True, verbose=False)

        # append commands to interactively create dummy finalization XMLs from top nodes (which are never executed)
        # TODO add ability to use existing files or infer Laue class automatically
        top_node_names = list(fc.meta.sort_values(by=['Cluster', 'Nexp', 'File path']).drop_duplicates(subset='Cluster', keep='last')['name'])
        top_nodes = fc.get_subset(top_node_names)

        template_files = {}
        for name, fin in top_nodes.items():
            folder = os.path.dirname(fin.path)
            the_xml = os.path.join(tmp_folder, f'C{fin.meta["Cluster"]}.xml')
            cmds.append(f'xx selectexpnogui_ignoreerror ' + os.path.join(folder, os.path.split(folder)[-1] + ".par"))
            cmds.append(f'dc xmlrrp {name} ' + the_xml)
            template_files[fin.meta['Cluster']] = the_xml
            
        # append commands to run finalizations based on individual XML for each node (assuming they exist!)
        cmds.append('xx sleep 1000')
        
        prev_folder = ''

        for name, fin in fc.sort_by_meta(by=['Cluster', 'File path']).items():
            # print(node)
            folder = fin.folder
            
            if folder != prev_folder:
                par =  os.path.basename(folder)
                cmds.append(f'xx selectexpnogui_ignoreerror {os.path.join(folder, par) + ".par"}')
                
            cmds.append(f'dc rrpfromxml {fin.pars_xml_path}')
            prev_folder = folder

        mac_cmd = self.write_macro(self.macro_fn, cmds, append=False)
        self.send_cmd(mac_cmd)
        self.send_message('CAP command copied to Clipboard.\nPlease paste into CMD window, run, and set options.')
        
        print('Please run the following command in CAP:\n-----')
        print(mac_cmd)
        print('-----')
        print('...and set finalization parameters (Laue group, most importantly) for each cluster')

        # wait for template XML files and broadcast to XML for each node

        from time import sleep
        print('Waiting for template XML files...')

        _templates = list(template_files.items())
        N_templates = len(_templates)

        while _templates:
            for cluster, template_fn in _templates:
                if os.path.exists(template_fn):
                    print(f'Finalization template for cluster {cluster} found:', template_fn)
                    for _, fin in fc.items():
                        if fin.meta['Cluster'] == cluster:
                            fin.pars_xml.set_parameters(template=template_fn, 
                            autochem=False, gral=False, res_limit=res_limit)
                    
                    _templates.remove((cluster, template_fn))
                    self.send_message(f'Still waiting for {len(_templates)}/{N_templates} template files.')
            sleep(0.05)
                
        print('All finalization parameter files have been created. CAP should start with finalizations now...')

        from time import sleep
        fins_todo = list(fc.sort_by_meta(by=['Cluster', 'File path']).keys())        
        self.send_message(f'0/{len(fc)} finalizations finished.')        

        while fins_todo:
            for fin_name in fins_todo:
                try:
                    fc[fin_name].parse_finalization_results(check_current=True, timeout=finalization_timeout)
                    print(f'Finalization for {fin_name} completed, results found')
                    fins_todo.remove(fin_name)            
                    self.send_message(f'{len(fc)-len(fins_todo)}/{len(fc)} finalizations finished.')                            
                except FileNotFoundError:
                    pass
                    
            sleep(0.5)

        print(f'Finalizations completed for {self.path}')

        return fc

