import os
from cap_tools.finalization import Finalization, FinalizationCollection
import glob
import time
from cap_tools.cell_list import CellList
from typing import *
import queue
import shutil
from configparser import ConfigParser
import csv
import subprocess

class CAPListenModeError(RuntimeError):
    pass
class CAPInstance:
    # TODO this should be a context manager, if we'd want to be pythonic
    
    def __init__(self, cmd_folder: str = 'C:\\Xcalibur\\tmp\\listen_mode_offline', 
                 par_file: str = 'C:\\Xcalibur\\CrysAlisPro171.44\\help\\ideal_microed\\MicroED.par', 
                 cap_folder: str = 'C:\\Xcalibur\\CrysAlisPro171.44',
                 wait_complete: bool = True, start_now: bool = False):

        self.cmd_folder = cmd_folder
        self.par_file = par_file
        self.cap_folder = cap_folder
        self.cap_proc = None #TODO: start and handle CAP offline process here
        self.start_timeout = 3        
        self.last_command = ''
        os.makedirs(cmd_folder, exist_ok=True)
        
        try:
            # make sure that any CAP instance listening in this folder stops doing it
            # (if there is any)            
            self.run_cmd('xx listenmode off', timeout=0.2, auto_start=False)
        except CAPListenModeError:
            pass
        
        if start_now: self.start_cap()
        
        if not wait_complete:
            raise NotImplementedError('Non-blocking execution of CAP commands not implemented yet.')
        
    def start_cap(self, timeout=10):    
        if self.running:
            raise CAPListenModeError('CAP instance is already running; cannot start one.')
                
        self.cap_proc = subprocess.Popen(f'{os.path.join(self.cap_folder, "pro.exe")} {self.par_file} -listenmode {self.cmd_folder}')
        t0 = time.time()
        while True:
            try:
                self.run_cmd('xx sleep 1', timeout=0.2)
                break
            except CAPListenModeError as err:
                if time.time() > (t0 + timeout):
                    raise CAPListenModeError(f'CAP not reacting {timeout} seconds after launch. Please check if a CAP window is running and retry.')
        
    def stop_cap(self, allow_stopped: bool = False):

        if (not self.running or (self.cap_proc is None)) and (not allow_stopped):
            raise CAPListenModeError('No CAP instance running.')
        elif (not self.running or (self.cap_proc is None)):
            return
                
        try:
            self.run_cmd('xx listenmode off', timeout=1)
        except CAPListenModeError:
            pass
        finally:
            self.cap_proc.terminate()
            self.cap_proc = None
            
    def __del__(self):
        try:
            self.stop_cap(allow_stopped=True)
        except Exception as err:
            pass
        
    def status(self):
        listen_fn = lambda ext: os.path.join(self.cmd_folder, f'command.{ext}')      
        if os.path.exists(listen_fn('busy')):
            return('Busy')
        elif os.path.exists(listen_fn('error')):
            return('Error')
        else:
            return('Idle')
        
    @property
    def running(self):
        return (self.cap_proc is not None) and (self.cap_proc.poll() is None)
        
    def run_cmd(self, cmd: Union[str, List[str]], 
                use_mac: bool = True, 
                timeout: Optional[float] = None,
                auto_start: bool = True):        
        
        if auto_start and not self.running:
            self.start_cap()
        
        multi_cmd = isinstance(cmd, list) 
        macro = '\n'.join(cmd) if multi_cmd else ''
        listen_fn = lambda ext: os.path.join(self.cmd_folder, f'command.{ext}')      
        
        if self.status == 'Busy':
            raise CAPListenModeError('CAP Instance is busy. Cannot submit new command.')  
                
        for fn in glob.glob(listen_fn('*')): 
            os.remove(fn)
               
        if multi_cmd and use_mac:
            # replace calls by macro call (will be faster for many little calls)
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
            cap_instance (CAPInstance): Listen mode CAP instance
            message_func (Optional[Union[Callable[[str], None], queue.Queue]], optional): Function/queue via which info messages are sent. 
            If None, just prints messages. Defaults to None.
            request_func (Optional[Union[Callable[[str], None], queue.Queue]], optional): Function/queue via which requests are sent that require a response. 
            If None, just prints requests. Defaults to None.
            response_func (Optional[Union[Callable[[], Any], queue.Queue]], optional): Function/queue via which info responses are returned. 
            If None, reads from command line. Defaults to None.
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
    
    def __init__(self, merge_file: str,
                 cap_instance: CAPInstance,
                 message_func: Optional[Union[Callable[[str], None], queue.Queue]] = None,
                 request_func: Optional[Union[Callable[[str], None], queue.Queue]] = None,                 
                 response_func: Optional[Union[Callable[[], Any], queue.Queue]] = None):
        """CAP workflow for bulk merging and/or finalization of clustered (or any, in fact) experiments.

        Args:
            path (str): _description_
            clusters (Dict[int, CellList]): Clusters (dictionary of cell lists)
            cap_instance (CAPInstance): Listen mode CAP instance
            message_func (Optional[Union[Callable[[str], None], queue.Queue]], optional): Function/queue via which info messages are sent. 
            If None, just prints messages. Defaults to None.
            request_func (Optional[Union[Callable[[str], None], queue.Queue]], optional): Function/queue via which requests are sent that require a response. 
            If None, just prints requests. Defaults to None.
            response_func (Optional[Union[Callable[[], Any], queue.Queue]], optional): Function/queue via which info responses are returned. 
            If None, reads from command line. Defaults to None.
        """
        
        super().__init__(work_folder=os.path.split(merge_file)[0], cap_instance=cap_instance,
                         message_func=message_func,
                         request_func=request_func,
                         response_func=response_func)
        self.merge_file = merge_file
        self.merge_data = {}
        self.read_merge_file()
        
    def read_merge_file(self):
        with open(self.merge_file) as fh:
            
            while l := fh.readline():
                if 'Merge code' not in l:
                    continue
                else:
                    fields = [k.strip() for k in l.split(',')]
                    break
                    
            reader = csv.DictReader(fh, fieldnames=fields)
            self.merge_data = list(reader)
            
        for md in self.merge_data:
            md['Data sets'] = [ds.strip() for ds in md['Data sets'].split('|')]
                      
    def merge(self, reintegrate: bool = False):
        
        ini = ConfigParser()
        self.message(f'Starting merging...')
        
        if reintegrate:
            raise NotImplementedError('Reintegration currently not implemented') #TODO fix this
            # for cid, cluster in self.clusters.items():
            #     for the_ds in cluster.ds:
            #         self.message(f'Running proffit (auto) on {the_ds["Experiment name"]}')
            #         self.run(f'xx selectexpnogui ' + os.path.join(the_ds['Dataset path'], the_ds['Experiment name']) + ".par")
            #         self.run('dc proffit auto') # one day, allow XML templates etc.....     
    
        for ii, md in enumerate(self.merge_data):     
            
            # TODO put a skip here for top_only
            
            self.message(f'Running merging [{ii+1}/{len(self.merge_data)}]: ' + md['Name'])
            
            out_path, inp = md['File path'], md['Data sets']
            os.makedirs(os.path.dirname(md['File path']), exist_ok=True)
            ini_fn = md['File path'] + '_merge.ini'
            
            for in_path in inp:
                if not os.path.exists(fn_tabbin := in_path + '_proffitpeak.tabbin'):   
                    print(fn_tabbin, 'not found.')
                    if existing_tabbin := glob.glob(os.path.join(os.path.dirname(in_path), '*_proffitpeak.tabbin')):
                        print('Copying', existing_tabbin[0], 'to', fn_tabbin)
                        shutil.copyfile(existing_tabbin[0], fn_tabbin)
                    else:
                        print('No alternative _proffitmerge.tabbin file found. Please reprocess', in_path)
                        self.message(f'Missing tabbin file found. Please reprocess {in_path}')
            
            print(f'--- Writing INI file {ini_fn}---\nOutput: ', out_path, '\nInputs:', inp)      
            ini = ConfigParser()
            ini['Number of experiments to merge'] = {
                'number of experiments to merge': str(len(inp))       
            }       
            ini['Target Merged experiment'] = {
                'target rrpprof path filename with ext': out_path + '.rrprof'
                }
            for ii, in_name in enumerate(inp):
                ini[f'Source merged experiment {ii+1}'] = {
                    'source rrpprof path filename with ext': in_name + '.rrpprof'
                    }                           
            ini.write(open(ini_fn, 'w'))
                        
            self.run(f"xx selectexpnogui {os.path.join(os.path.dirname(inp[0]), md['Merge code'].split(':', 1)[0])}.par")            
            self.run(f'XX PROFFITMERGE2FROMINI {ini_fn}')
            
        self.message(f'{len(self.merge_data)} merging runs finished')

    def finalize(self, 
                 res_limit: float = 0.8, 
                 top_gral: bool = False,
                 top_ac: bool = False,
                 use_default_settings: bool = True,
                 reintegrate: bool = False) -> FinalizationCollection:    
        
        """Runs finalizations

        Args:
            res_limit (float, optional): Resolution limit to which to finalize. Defaults to 0.8.
            top_only (bool, optional): Only finalize top nodes where all experiments of the cluster are merged. Defaults to False.
            top_gral (bool, optional): Run GRAL (SG determination) on top nodes. Defaults to False.
            top_ac (bool, optional): Run AutoChem on top nodes. Implies that GRAL is run as well. Defaults to False.

        Returns:
            FinalizationCollection: All finalization data
        """

        if top_ac: top_gral = True

        self.merge(reintegrate=reintegrate)

        tmp_folder = os.path.join(os.path.dirname(self.merge_file), 'tmp')
        for fn in glob.glob(os.path.join(tmp_folder, '*.xml')):  
            os.remove(fn)
            print(f'Deleting {fn}')
                      
        # generate finalization collection from CSV file stored by cluster_merge
        fc = FinalizationCollection.from_csv(self.merge_file, allow_missing=True, verbose=False)

        # append commands to interactively create dummy finalization XMLs from top nodes (which are never executed)
        # TODO add ability to use existing files or infer Laue class automatically
        # TODO generalize to cases without clustering
        top_node_names = list(fc.meta.sort_values(by=['Cluster', 'Nexp', 'File path']).drop_duplicates(subset='Cluster', keep='last')['name'])
        top_nodes = fc.get_subset(top_node_names)
        template_files = {}
        
        for top_name, top_fin in top_nodes.items():
            
            cluster = top_fin.meta['Cluster']
            folder = os.path.dirname(top_fin.path)
            
            if use_default_settings:
                # use finalization options (Laue group, mostly) from merged reduction (requires 44.92 or higher)
                fn_template = top_fin.path + '_finalizer_default_merged.xml'
                if not os.path.exists(fn_template):
                    # Finalizer XML missing - likely, top-level merge needs to be opened once.
                    self.run(f'xx selectexpnogui ' + os.path.join(folder, top_name + ".par"))
                    time.sleep(1)
                if not os.path.exists(fn_template):                    
                    # Finalizer XML _still_ missing. Something is wrong.
                    msg = f'Merged finalizer XML not found. Make sure that "Make finalizer xml file during data reduction" is enabled in CAP options.'
                    self.message(msg)
                    raise FileNotFoundError(f'Merged finalizer XML {fn_template} not found.')
            else:
                # dry-run top-level finalization to set proper finalization options
                os.makedirs(tmp_folder, exist_ok=True)                        
                fn_template = os.path.join(tmp_folder, f'C{cluster}_finalizer_dlg.xml')
                self.run(f'xx selectexpnogui ' + os.path.join(folder, top_name + ".par"))
                self.message(f'Please choose finalization settings (Laue group!) for cluster {cluster} in CAP.')
                self.run(f'dc xmlrrp {top_name} ' + fn_template)
                while not os.path.exists(fn_template):
                    time.sleep(0.1) # wait until user has finished setting the finalization options
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
                # if GRAL should be run, start finalization straight away (otherwise order will be confusing to user)
                self.message(f'Running finalization with interactive GRAL for {top_name} in cluster {cluster}.')
                self.run(f'dc rrpfromxml {top_fin.pars_xml_path}')
                top_fin.parse_finalization_results(check_current=False)    # no idea why false is required here (GRAL/AC seems to change XML somehow)
                self.message(f'Finalization for top-level node {top_name} completed, results found.')            
                
        prev_folder = ''

        for ii, (name, fin) in enumerate(fc.sort_by_meta(by=['Cluster', 'File path']).items()):
 
            if top_gral and (name in top_node_names):      
                # If GRAL has been selected for top-level finalization, it is not redone here              
                self.message(f'Skipping top-node finalization for merge {name} in cluster {fin.meta["Cluster"]}. [{ii+1}/{len(fc)}]')
                continue
                                
            folder = fin.folder
            
            if folder != prev_folder:
                par =  os.path.basename(folder)
                self.run(f'xx selectexpnogui {os.path.join(folder, par) + ".par"}')
                            
            self.message(f'Running finalization for merge {name} in cluster {fin.meta["Cluster"]}. [{ii+1}/{len(fc)}]')
            self.run(f'dc rrpfromxml {fin.pars_xml_path}')
            time.sleep(0.5)
            fin.parse_finalization_results(check_current=True)
            self.message(f'Finalization for {name} completed, results found.')
            
            prev_folder = folder
            
        self.message(f'All requested finalizations completed. Please use the "Merge/Finalize" tab to inspect the results.')

        return fc

