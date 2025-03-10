from glob import glob
import os
import pandas as pd
import shutil
import xml.etree.ElementTree as ET
import hashlib
from sys import argv
import sys
from argparse import ArgumentParser
from cap_tools.cap_control import CAPInstance, CAPListenModeError
import csv
from zipfile import ZipFile


def main(experiments: list, out_dir: str, include_path: bool = False, 
         interactive: bool=False, log: callable = print, zip_result: bool = True):

    exp_list = []
    
    for exp_entry in experiments:

        if not os.path.exists(exp_entry):
            raise FileNotFoundError(f'Input folder or CSV file {exp_entry} does not exist')

        if exp_entry.endswith('.csv'):
            with open(exp_entry) as fh:
                for _ in range(7):
                    _ = fh.readline()
                ds = list(csv.DictReader(fh))
                new_exp = [os.path.join(d['Dataset path'], d['Experiment name']) for d in ds]
                log(f'Found {len(new_exp)} experiments in {exp_entry}')

        else:            
            new_exp = [os.path.splitext(fn)[0] for fn in glob(os.path.join(exp_entry,'**\\*.par'), recursive=True) if ('_cracker' not in fn)]
            log(f'Found {len(new_exp)} experiments in {exp_entry}')
            
        exp_list.extend(new_exp)
        
    exp_list = [fn for fn in exp_list if  
                (('tomo' not in fn)
                and ('DD_Calib' not in fn) 
                and ('Preset' not in fn) 
                and ('Cluster' not in fn)
                and (not os.path.basename(fn).startswith('m_'))
                )]
    
    exp_list = sorted(list(set(exp_list)))
    
    log(f'Found {len(exp_list)} new experiments of correct type')
        
    root_dir = '' # historical...

    if include_path:
        log('WARNING: experiment path will be included in output list file. Resulting data will not be anonymized.')

    root_dir = ''
    os.makedirs(out_dir, exist_ok=True)

    info = []
    cap_cmds = []

    for ii, exp in enumerate(sorted(exp_list)):
        info_fn = os.path.join(root_dir, os.path.dirname(exp), 'experiment_results.xmlinfo')
        info_str = open(info_fn).read()
        tree = ET.fromstring('<root>\n' + info_str + '\n</root>')
        exp_info = {'path': exp} if include_path else {}
        
        # generate hash digest stable information (not changing with reprocessing or moving)
        if (user := tree.find('__EXPERIMENT_INFO__/__USER__')) is not None:
            user = user.text
        else:
            user = 'anonymous'
            # print('WARNING: No user found for', exp)
            
        if (exp_time := tree.find('__EXPERIMENT_INFO__/__START_TIME__')) is not None:
            exp_time = exp_time.text
        else:
            exp_time = 'unknown time'
            # print('WARNING: No experiment time found for', exp)
            
        if (exp_name := tree.find('__EXPERIMENT_INFO__/__EXPERIMENT_PAR_NAME_WOEXT__')) is not None:
            exp_name = exp_name.text
        else:
            exp_name = os.path.basename(exp)
            # print('WARNING: No experiment name found for', exp)
        
        m = hashlib.md5()
        hash_text = ';'.join([user, exp_time, exp_name])
        m.update(hash_text.encode()) # this line defines what gets hashed
        basename = exp_info['digest'] = m.hexdigest()
        
        if ii == 0:
            # first loop run
            log('Generating anonymous experiment hashes from string of type:')
            log(hash_text)
            log('-->  ', basename)
        
        xml_entries = {
            'scan_range': tree.find('__EXPERIMENT_INFO__/__SCAN_RANGE__'),
            'detector_distance': tree.find('__EXPERIMENT_INFO__/__DETECTOR_DISTANCE__'),
            'indexation': tree.find('__EXPERIMENT_RESULTS__/__INDEXATION__'),
            'e1': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E1__'),
            'e2': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E2__'),
            'e3': tree.find('__EXPERIMENT_RESULTS__/__MOSAICITY__/__MOSAICITY_E3__'),
            'diff_limit': tree.find('__EXPERIMENT_RESULTS__/__DIFFLIMIT__'),
            'r_int': tree.find('__EXPERIMENT_RESULTS__/__RINT__'),
        }
        
        for k, v in xml_entries.items():
            if v is not None:
                exp_info[k] = float(v.text)
                
        if 'scan_range' not in exp_info:
            # print('No scan range found for', exp)
            continue

        fn_in = os.path.join(root_dir, exp) + '_middle_microed_diff_snapshot'
        fn_out = os.path.join(out_dir, basename) + '_diff.tiff'
        
        exp_info['diff_img'] = basename + '_diff.tiff'
        
        if os.path.exists(fn_out):
            pass
        elif os.path.exists(fn_in + '.tiff'):
            shutil.copy(fn_in + '.tiff', fn_out)
        elif os.path.exists(fn_in + '.rodhypix'):
            cap_cmds.append(f'rd i "{fn_in}.rodhypix"')
            cap_cmds.append(f'wd tiffopt {fn_out} 1 0 0 0')
        else:
            # print('No diffraction snapshot found for', exp)
            continue
        
        fn_in = os.path.join(root_dir, exp) + '_microed_grain_snapshot'
        fn_out = os.path.join(out_dir, basename) + '_grain.tiff'
        
        exp_info['grain_img'] = basename + '_grain.tiff'

        if os.path.exists(fn_out):
            pass              
        elif os.path.exists(fn_in + '.tiff'):
            shutil.copy(fn_in + '.tiff', fn_out)
        elif os.path.exists(fn_in + '.rodhypix'):
            cap_cmds.append(f'rd i {fn_in}.rodhypix')
            cap_cmds.append(f'wd tiffopt {fn_out} 1 0 0 0')
        else:
            # print('No grain snapshot found for', exp)
            continue
                
        info.append(exp_info)
        
    info = pd.DataFrame(info)

    log(f'Found {len(info)} new experiments with sufficient metadata')
    
    if cap_cmds:
        log(f'Running image conversions in CAP')
        listen = CAPInstance()
        while True:
            try:
                listen.run_cmd(cap_cmds, use_mac=True)
                break
            except CAPListenModeError as err:
                if not interactive:
                    log(str(err))
                    raise err
                    
                else:
                    log('-----')
                    log(str(err))
                    log('Press Return to Retry or Ctrl-C to quit.')
                    try:
                        input()
                    except KeyboardInterrupt:
                        log('Exiting.')
                        exit()
    
    if os.path.exists(fn := os.path.join(out_dir, 'info.csv')):
        existing = pd.read_csv(fn)
        log(f'Found existing learning set {fn} with {len(existing)} entries. Extending set, dropping duplicates.')
        info = pd.concat([existing, info]).drop_duplicates(subset='digest')
    
    log(f'Writing set with {len(info)} entries into', os.path.join(out_dir, 'info.csv'))
    info.to_csv(os.path.join(out_dir, 'info.csv'), index=False)

    log('Finished writing training data to:', out_dir)
    
    if zip_result:
        log('Zipping data set to', os.path.join(out_dir, 'learning_set.zip'))
        with ZipFile(os.path.join(out_dir, 'learning_set.zip'), 'w') as zip:
            for fn in (glob(os.path.join(out_dir, '*.tiff')) + [os.path.join(out_dir, 'info.csv')]):
                zip.write(fn, os.path.basename(fn))
    
def gui():
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter.filedialog import askdirectory, askopenfilename
    
    root = tk.Tk()
    root.title('Learning set generator')
    
    try:       
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    # root.iconbitmap(os.path.join(base_path, "calibrate_dd_icon.ico"))

    info = tk.Text(root, font='TkFixedFont', height=40, width=100)
    
    def info_write(*string, append=False):
        string = ' '.join(string)
        info.configure(state="normal")
        if append:
            string = '\n' + string
        else:
            info.delete('1.0', tk.END)
        info.insert("end", string)
        info.configure(state="disabled")
        info.see(tk.END)
        info.update()
                
    output_folder = tk.StringVar()
    input_experiments = []
    
    info_write('Please add data folder(s) or CSV files, and select output folder.')
    
    def set_output_folder():
        fn = os.path.normpath(askdirectory(title=f'Select output folder for learning set'))
        output_folder.set(fn)
        
    def add_folder():
        fn = os.path.normpath(askdirectory(title=f'Add experiment folder structure (will be searched recursively!)'))
        if fn.strip() and (fn.strip != '.'):
            input_experiments.append(fn)
            info_write('\n'.join(input_experiments), append=False)        

    def add_csv():
        fn = os.path.normpath(askopenfilename(title=f'Add Results Viewer CSV file', filetypes=[('Cell tool result', '*.csv')]))
        if fn.strip() and (fn.strip != '.'):
            input_experiments.append(fn)
            info_write('\n'.join(input_experiments), append=False)
        
    def run_processing():
        nonlocal input_experiments
        print('Starting processing...')
        
        def config_window(state):
            for w in root.winfo_children():
                try:
                    w.configure(state)
                except tk.TclError:
                    pass
        
        try:
            config_window('disabled')
            main(input_experiments, output_folder.get(), 
                include_path=False, interactive=False, 
                log=lambda *msgs: info_write(*msgs, append=True))
            info_write(f'Processing successful. Results in {output_folder.get()}',
                       append=True)
            output_folder.set('')
            input_experiments = []
            config_window('normal')
            
        except Exception as e:
            config_window('normal')
            info_write(f'Processing failed:\n{str(e)}\n---\nLoaded experiments are:\n'+'\n'.join(input_experiments),
                       append=True)
         
    ttk.Button(root, text='Set output folder', command=set_output_folder).grid(row=5, column=0, sticky=tk.W)
    ttk.Label(root, textvariable=output_folder).grid(row=5, column=1, sticky=tk.W)
    ttk.Separator(root, orient='horizontal').grid(row=9, columnspan=2, sticky=tk.EW)
    ttk.Button(root, text='Add folder', command=add_folder).grid(row=10, column=0, sticky=tk.NW)
    ttk.Button(root, text='Add CSV', command=add_csv).grid(row=11, column=0, sticky=tk.NW)
    ttk.Button(root, text='Start processing', command=run_processing).grid(row=12, column=0, sticky=tk.NW, columnspan=2)
    
    info.grid(row=10, column=1, rowspan=50, sticky=tk.NSEW)

    def _on_closing():
        root.quit()  # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL tstate

    root.protocol("WM_DELETE_WINDOW", _on_closing)  # bind closing routine    
    root.mainloop()
      
if __name__ == '__main__':
    
    if len(argv) == 1:
        gui()    
    
    else:           
        parser = ArgumentParser(description='Generate anonymous ML training sets from CrysAlisPro ED datasets.')
        parser.add_argument('out_dir', help='Directory to store learning data. If not empty, new learning data will be appended (ignoring duplicates), not overwritten.')
        parser.add_argument('experiments', nargs='+', help='One or more (1) directories to recursively search for experiments, or (2) CSV experiment lists generated from the'
                            ' results viewer. Both can be combined arbitrarily.')
        parser.add_argument('--include-path', action='store_true', help='Include dataset path into output file. WARNING: your learning set will not be anonymous anymore.')
        args = parser.parse_args()

        main(experiments=args.experiments, out_dir=args.out_dir, include_path=args.include_path, interactive=True)
        
        
