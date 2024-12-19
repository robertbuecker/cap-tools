import os
from cap_tools.finalization import Finalization, FinalizationCollection
import glob
import time
import pandas as pd
from cap_tools.cell_list import CellList
from typing import *
import queue
import shutil

class CAPListenModeError(RuntimeError):
    pass
class CAPInstance:
    
    def __init__(self, cmd_folder: Optional[str] = 'C:\\Xcalibur\\tmp\\listen_mode_offline', wait_complete: bool = True):

        self.cmd_folder = cmd_folder
        self.cap_handle = None #TODO: start and handle CAP offline process here
        self.start_timeout = 3        
        self.last_command = ''
        if not wait_complete:
            raise NotImplementedError('Non-blocking execution of CAP commands not implemented yet.')
        
    def status(self):
        listen_fn = lambda ext: os.path.join(self.cmd_folder, f'command.{ext}')      
        if os.path.exists(listen_fn('busy')):
            return('Busy')
        elif os.path.exists(listen_fn('error')):
            return('Error')
        else:
            return('Idle')
        
        
    def run_cmd(self, cmd: Union[str, List[str]], use_mac: bool = True, timeout: Optional[float] = None):        
        
        multi_cmd = isinstance(cmd, list) 
        macro = '\n'.join(cmd) if multi_cmd else ''
        listen_fn = lambda ext: os.path.join(self.cmd_folder, f'command.{ext}')      
        
        if self.status == 'Busy':
            raise CAPListenModeError('CAP Instance is busy. Cannot submit new command.')  
                
        for fn in glob.glob(listen_fn('*')): 
            os.remove(fn)
               
        if multi_cmd and use_mac:
            # replace calls by macro call (might be faster for many little calls)
            with open(fn := listen_fn('mac'), 'w') as fh:
                fh.write(macro)
            cmd = f'script {fn}'
        elif multi_cmd:
            for the_cmd in cmd:
                self.run_cmd(the_cmd)
            return
                                
        with open(listen_fn('in'), 'w') as fh:
            fh.write(cmd)        
        self.last_command = cmd
            
        t0 = time.time()
        while not os.path.exists(listen_fn('busy')):
            if (time.time() - t0) < self.start_timeout:
                time.sleep(0.01)
            else:
                raise CAPListenModeError(f'CAP listen mode not reacting in {self.cmd_folder}. If listen mode is not active, start it by running "xx listenmode on" in the CrysAlisPro CMD window.')
            
        t0 = time.time()
        while os.path.exists(listen_fn('busy')):
            if not timeout or ((time.time() - t0) < timeout):
                time.sleep(0.01)
            else:
                with open(listen_fn('stop'), 'w') as fh:
                    pass
                raise CAPListenModeError(f'CAP command {cmd} timed out after {time.time()-t0:.2f} seconds.')
        
        while not (os.path.exists(listen_fn('done'))
                   | os.path.exists(listen_fn('error'))):
            time.sleep(0.01)        
                    
        if os.path.exists(fn := listen_fn('error')):
            with open(fn, 'r') as fh:
                cmd_ret = fh.read().strip()
            os.remove(fn)
            if cmd_ret.startswith('script'):
                # expand error message to macro
                with open(cmd_ret.split(maxsplit=1)[-1], 'r') as fh:
                    cmds_ret = fh.readlines()
                raise CAPListenModeError(f'Failed CAP commands: {"\n".join(cmds_ret)}')
            else:
                raise CAPListenModeError(f'Failed CAP command: {cmd_ret}')
                        
        elif os.path.exists(fn := listen_fn('done')):
            with open(fn, 'r') as fh:
                cmd_ret = fh.read().strip()
            os.remove(fn)
            if cmd == cmd_ret:
                print(f'Command:\n{cmd_ret}\nfinished successfully.')
            else:
                raise CAPListenModeError(f'Returned command:\n{cmd_ret}\ndoes not match request:\n{cmd}')
            
        else:
            # this should never be reached, unless really bad timing conditions surface
            raise CAPListenModeError('Confirmation file not found.')

class CAPControl:
 
    def __init__(self, work_folder: str, cap_instance: CAPInstance,
                 message_func: Optional[Union[Callable[[str], None], queue.Queue]] = None,
                 request_func: Optional[Union[Callable[[str], None], queue.Queue]] = None,                 
                 response_func: Optional[Union[Callable[[], Any], queue.Queue]] = None):
        
        """Base class for Python-controlled CAP workflows, to be executed via macros or listen mode.
        Functions that run command sequences can be blocking, and should communicate to the outside (e.g. GUI) via queues.

        Args:
            work_folder (str): _description_
            cmd_folder (Optional[str], optional): _description_. Defaults to None.
            message_func (Optional[Union[Callable[[str], None], queue.Queue]], optional): Callable or Queue to send status messages and prompts back to the 
            user interface. If None, messages are simply printed. Defaults to None.
            cmd_func (Optional[Union[Callable[[str], None], queue.Queue]], optional): Callable or Queue for commands to be executed by CAP. Defaults to None.
        """
        
        self._cap = cap_instance
        self.work_folder = work_folder
        
        if message_func is None:
            self._message_func = print
        elif isinstance(message_func, queue.Queue):
            self._message_func = message_func.put
        else:
            self._message_func = message_func
            
        if request_func is None:
            self._request_func = print
        elif isinstance(request_func, queue.Queue):
            self._request_func = request_func.put
        else:
            self._request_func = request_func
            
        if response_func is None:
            self._response_func = input
        elif isinstance(response_func, queue.Queue):
            self._response_func = response_func.get
        else:
            self._response_func = response_func                        

    def message(self, msg):
        self._message_func(msg)
            
    def request(self, prompt: Optional[str] = None):
        if prompt is not None:
            self._request_func(prompt)
        return self._response_func()
    
    def run(self, cmd, timeout: Optional[float] = None, **kwargs):
        try:
            self._cap.run_cmd(cmd, timeout=timeout, **kwargs)
            
        except CAPListenModeError as err:
            self.message(str(err))
            raise(err)
        
    def write_macro(self, macro_name: str, macro: Union[List[str], str],
                    append: bool = False):
        # just writes a macro file into the work folder and returns the command to run it
        
        if not macro_name.endswith('.mac'): macro_name += '.mac'
        if not os.path.split(macro_name)[0]:
            macro_name = os.path.join(self.work_folder, macro_name)
        
        if isinstance(macro, list): macro = '\n'.join(macro)
        
        with open(macro_name, 'a' if append else 'w') as fh:
            fh.write(macro)
            
        return f'script {macro_name}'     
    
class CAPMergeFinalize(CAPControl):
    
    def __init__(self, path: str, clusters: Dict[int, CellList],
                 cap_instance: CAPInstance,
                 message_func: Optional[Union[Callable[[str], None], queue.Queue]] = None,
                 request_func: Optional[Union[Callable[[str], None], queue.Queue]] = None,                 
                 response_func: Optional[Union[Callable[[], Any], queue.Queue]] = None):
        """CAP workflow for bulk merging and/or finalization of clustered (or any, in fact) experiments.

        Args:
            path (str): _description_
            clusters (Dict[int, CellList]): _description_
            cap_instance (CAPInstance): _description_
            message_func (Optional[Union[Callable[[str], None], queue.Queue]], optional): _description_. Defaults to None.
            request_func (Optional[Union[Callable[[str], None], queue.Queue]], optional): _description_. Defaults to None.
            response_func (Optional[Union[Callable[[], Any], queue.Queue]], optional): _description_. Defaults to None.
        """
        
        super().__init__(work_folder=os.path.split(path)[0], cap_instance=cap_instance,
                         message_func=message_func,
                         request_func=request_func,
                         response_func=response_func)
        self.path = path
        self.clusters = clusters
        
    @property
    def node_info_fn(self) -> str:
        return self.path + '_nodes.csv'    
    
    @property
    def merge_files_found(self) -> bool:
        return os.path.exists(self.node_info_fn)
    
    def cluster_merge(self, delete_existing: bool = False, top_only: bool = False):
        
        self.message(f'Running full-tree merging for clusters: {[int(k) for k in self.clusters.keys()]}')
        
        cmds = []
        old_fns = []
        redundant_fns = []
        
        with open(self.node_info_fn, 'w') as ifh:
            # TODO factor this into cluster class. Why is this here?!
            ifh.write('Name,File path,Cluster,Data sets,Merge code\n')
            merged_cids = []
            for ii, (c_id, cluster) in enumerate(self.clusters.items()):
                N_merges = len(cluster.merge_tree)
                out_paths, in_paths, out_codes, out_info = cluster.get_merging_paths(prefix=f'C{c_id}', short_form=True)                
                for ii, (out, (in1, in2), code, info) in enumerate(zip(out_paths, in_paths, out_codes, out_info)):
                    cmds.append(f'xx proffitmerge "{out}" "{in1}" "{in2}"')
                    if (not top_only) or ((ii+1) == N_merges):
                        ifh.write(f'{os.path.basename(out)},{out},{c_id},{info},{code}\n')
                    else:
                        redundant_fns += glob.glob(out + '*.*')
                    if delete_existing:
                        old_fns += glob.glob(out + '*.*')
                  
                print(f'Full-cluster merge for cluster {c_id}: {out_paths[-1]}')
                merged_cids.append(c_id)

        for fn in old_fns:        
            os.remove(fn)
            print(f'Deleting {fn}')
        
        self.run(cmds)

        for fn in redundant_fns:        
            os.remove(fn)
            print(f'Deleting {fn}')
        
        self.message(f'Completed full-tree merging for clusters: {[int(k) for k in self.clusters.keys()]}{" (top nodes only)" if top_only else ""}. Cluster data written to {self.node_info_fn}')
                                

    def cluster_finalize(self, 
                        res_limit: float = 0.8,
                        top_only: bool = False,
                        top_gral: bool = False,
                        top_ac: bool = False):    

        if top_ac: top_gral = True

        self.cluster_merge(delete_existing=True, top_only=top_only)

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
        for top_name, top_fin in top_nodes.items():
            
            cluster = top_fin.meta['Cluster']
            folder = os.path.dirname(top_fin.path)
            fn_template = os.path.join(tmp_folder, f'C{cluster}_finalizer_dlg.xml')
            self.run(f'xx selectexpnogui ' + os.path.join(folder, os.path.split(folder)[-1] + ".par"))
            self.message(f'Please choose finalization settings (Laue group!) for cluster {cluster} in CAP.')
            self.run(f'dc xmlrrp {top_name} ' + fn_template)
            while not os.path.exists(fn_template):
                time.sleep(0.1)
            template_files[cluster] = fn_template
            
            self.message(f'Finalization XML template for cluster {cluster} found. Generating finalization parameter files...')       
            for name, fin in fc.items():
                if fin.meta['Cluster'] == cluster:
                    fin.pars_xml.set_parameters(template=fn_template,
                    gral=True if (top_gral and (name == top_name)) else False, 
                    gral_interactive=True if (top_gral and (name == top_name)) else False, 
                    autochem=True if (top_ac and (name == top_name)) else False, 
                    res_limit=res_limit)
                                
            if top_gral:
                self.message(f'Running finalization for top-node merge {top_name} in cluster {cluster}.'
                             + (' Interactive space group determination started' if top_gral else ''))
                self.run(f'dc rrpfromxml {top_fin.pars_xml_path}')
                top_fin.parse_finalization_results(check_current=True)    
                self.message(f'Finalization for top-level node {top_name} completed, results found.')            
                
        prev_folder = ''

        for ii, (name, fin) in enumerate(fc.sort_by_meta(by=['Cluster', 'File path']).items()):
            # print(node)
            folder = fin.folder
            
            if folder != prev_folder:
                par =  os.path.basename(folder)
                self.run(f'xx selectexpnogui {os.path.join(folder, par) + ".par"}')
                
            if not (top_gral and (name in top_node_names)):                    
                self.message(f'Running finalization for merge {name} in cluster {fin.meta["Cluster"]}. [{ii+1}/{len(fc)}]')
                self.run(f'dc rrpfromxml {fin.pars_xml_path}')
                fin.parse_finalization_results(check_current=True)
            else:
                self.message(f'Skipping top-node finalization for merge {name} in cluster {fin.meta["Cluster"]}. [{ii+1}/{len(fc)}]')
                
            self.message(f'Finalization for {name} completed, results found.')
            prev_folder = folder
            
        self.message(f'All requested finalizations completed. Please use the "Merge/Finalize" tab to inspect the results.')

        return fc

