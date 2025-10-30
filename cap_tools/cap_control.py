import os
import glob
import time
from typing import *
import queue
import subprocess
from glob import glob
from datetime import datetime
import io
from warnings import warn

class CAPListenModeError(RuntimeError):
    pass

class CAPCommandError(ValueError):
    pass

class CAPRuntimeError(RuntimeError):
    pass

class CAPInstance:
    # TODO this should be a context manager, if we'd want to be pythonic (we don't want to be pythonic here, though)
    
    def __init__(self, 
                 max_cap_version: Union[int, tuple, str] = 100,
                 min_cap_version: Union[int, tuple, str] = 44,
                 cmd_folder: str = 'C:\\Xcalibur\\tmp\\listen_mode_offline', 
                 par_file: Optional[str] = None, 
                 cap_folder: Optional[str] = None,
                 wait_complete: bool = True, start_now: bool = False):
        
        if cap_folder is not None:
            warn('cap_folder is deprecated and will be removed in a future version. Use max_cap_version and min_cap_version instead.', DeprecationWarning)
            ver = os.path.split(cap_folder)[-1].split('.')
            if len(ver) == 2:
                ver = (int(ver[-1]), 1000)  # assume 1000 as minor version if not specified
            elif len(ver) == 3:
                ver = (int(ver[-2]), int(ver[-1]))
            else:
                raise ValueError(f'Invalid CAP folder name {cap_folder}. Expected format CrysAlisPro171.x or CrysAlisPro171.x.y')
            
        else:                
            cap_base_path = 'C:\\Xcalibur\\CrysAlisPro171'

            if isinstance(max_cap_version, str):
                max_cap_version = tuple(int(n) for n in max_cap_version.strip('a').split('.'))
                
            if isinstance(min_cap_version, str):
                min_cap_version = tuple(int(n) for n in min_cap_version.strip('a').split('.'))

            if max_cap_version is None:
                max_cap_version = (1000, 1000)  # effectively no maximum
            elif isinstance(max_cap_version, int):
                max_cap_version = (max_cap_version, 1000)

            if min_cap_version is None:
                min_cap_version = (0, 0)  # effectively no minimum
            elif isinstance(min_cap_version, int):
                min_cap_version = (min_cap_version, 0)

            cap_versions = [tuple(int(n.strip('a')) for n in os.path.split(fn)[0].split('.')[-2:]) 
                            for fn in glob(cap_base_path + f'.*.*\\pro.exe')]

            for ver in sorted(cap_versions, reverse=True):
                if ver >= min_cap_version and ver <= max_cap_version:
                    print(f'Found suitable CAP version {ver[0]}.{ver[1]}')
                    break
            else:
                print(f'No suitable CAP version found in range {min_cap_version} - {max_cap_version}')
                raise RuntimeError('No suitable CAP version found')

            for suffix in ['', 't', 'a', 'aa']:
                if os.path.exists(fn := os.path.join(cap_base_path + f'.{ver[0]}.{ver[1]}{suffix}', 'pro.exe')):
                    cap_folder = cap_base_path + f'.{ver[0]}.{ver[1]}{suffix}'
                    break
            else:
                raise FileNotFoundError(f'No CAP folder found for {f".{ver[0]}.{ver[1]}"}. This is an internal error.')

        self.cmd_folder = cmd_folder
        self.cap_folder = cap_folder  
        self.cap_version = ver       
        self.par_file = os.path.join(cap_folder, 'help', 'ideal_microed', 'MicroED.par') if par_file is None else par_file
        self.cap_proc: Optional[subprocess.Popen] = None #TODO: start and handle CAP offline process here
        self.start_timeout = 3 # seconds to wait for CAP to start
        self.last_command = '' 
        self.log_handle: Optional[io.TextIOWrapper] = None
        # self.history = [] # TODO: implement command history
        # self.log = [] # TODO: implement log window output per command
        # self.raise_command_error = True # TODO: implement command error handling
        # self.raise_runtime_error = True # TODO: implement runtime error handling
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
        
    def update_log_handle(self):
        """Update the log handle to the latest log file. This is used to read the log file in a non-blocking way.
        """        
        # first, determine filename
        import locale
        saved = locale.setlocale(locale.LC_ALL)
        try:
            locale.setlocale(locale.LC_ALL, 'C')
            red_log_fn = sorted(glob(
                os.path.join(os.path.dirname(self.par_file), 'log', 'crysalispro_redLOG*.txt')), 
                                reverse=True, key=os.path.getmtime)[0]
            timestamp = datetime.strptime(os.path.basename(red_log_fn), f'crysalispro_redLOG%a-%b-%d-%H-%M-%S-%Y.txt')
        except Exception as err:
            raise CAPListenModeError(f'Cannot find log file for experiment {self.par_file}')
        finally:
            locale.setlocale(locale.LC_ALL, saved)
            
        if self.log_handle is None:
            self.log_handle = open(red_log_fn, 'r')
        elif self.log_handle.name != red_log_fn:
            self.log_handle.close()
            self.log_handle = open(red_log_fn, 'r')
        
    def start_cap(self, timeout=20):    
        if self.running:
            raise CAPListenModeError('CAP instance is already running; cannot start one.')
                
        self.cap_proc = subprocess.Popen(f'{os.path.join(self.cap_folder, "pro.exe")} "{self.par_file}" -listenmode "{self.cmd_folder}"')
        t0 = time.time()
        while True:
            try:
                self.run_cmd('xx sleep 1', timeout=0.2)
                self.update_log_handle()
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
            self.log_handle = None
            
    def __del__(self):
        try:
            self.stop_cap(allow_stopped=True)
        except Exception as err:
            pass
        
    @property
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
    
    def load_experiment(self, par_file: str):
        """Load experiment in CAP instance. This is required for any command that requires a loaded experiment.

        Args:
            par_file (str): Name of the experiment to load.
        """
        
        self.par_file = par_file
        if not os.path.exists(self.par_file):
            raise FileNotFoundError(f'Experiment {self.par_file} not found.')
        if self.running:
            self.run_cmd(f'xx selectexpnogui \"{self.par_file}\"') 
            self.update_log_handle()
            
    def run_cmd_multi_exp(self, cmd: Union[str, List[str]], 
                          par_files: Union[str, List[str]], use_mac: bool = True, 
                          timeout: Optional[float] = None, auto_start: bool = True):
        """Run a command on multiple experiments in the CAP instance."""
        if isinstance(par_files, str):
            par_files = [par_files] 
        
        for par_file in par_files:
            if not os.path.exists(par_file):
                raise FileNotFoundError(f'Experiment {par_file} not found.')
            self.load_experiment(par_file)
            self.run_cmd(cmd, use_mac=use_mac, timeout=timeout, auto_start=auto_start)
        
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
                
        for fn in glob(listen_fn('*')): 
            try:
                os.remove(fn)
            except FileNotFoundError as err:
                # file might not exist, e.g. if it was already removed by another CAP instance
                pass
               
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
        while not self.status == 'Busy':
            if (time.time() - t0) < self.start_timeout:
                time.sleep(0.01)
            else:
                raise CAPListenModeError(f'CAP listen mode not reacting in {self.cmd_folder}. If listen mode is not active, start it by running "xx listenmode on" in the CrysAlisPro CMD window.')
            
        t0 = time.time()
        while self.status == 'Busy':
            if not timeout or ((time.time() - t0) < timeout):
                time.sleep(0.01)
            else:
                with open(listen_fn('stop'), 'w') as fh:
                    pass
                raise CAPListenModeError(f'CAP command {cmd} timed out after {time.time()-t0:.2f} seconds.')
        
        while not self.status in ['Idle', 'Error']:
            # wait for command to finish
            time.sleep(0.01)        
                    
        if os.path.exists(fn := listen_fn('error')):
            with open(fn, 'r') as fh:
                cmd_ret = fh.read().strip()
            os.remove(fn)
            if cmd_ret.startswith('script'):
                # expand error message to macro
                with open(cmd_ret.split(maxsplit=1)[-1], 'r') as fh:
                    cmds_ret = fh.readlines()
                raise CAPCommandError(f'Failed CAP commands: \n{"".join(cmds_ret)}')
            else:
                raise CAPCommandError(f'Failed CAP command: \n{cmd_ret}')
                        
        elif os.path.exists(fn := listen_fn('done')):
            with open(fn, 'r') as fh:
                cmd_ret = fh.read().strip()
            os.remove(fn)
            if cmd == cmd_ret:
                # print(f'Command:\n{cmd_ret}\nfinished successfully.')
                pass
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
        #TODO check if this class is actually still required, or if it can be replaced by CAPInstance directly
        
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
        

