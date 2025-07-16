from glob import glob
import os
import pandas as pd
import xml.etree.ElementTree as ET
import hashlib
from sys import argv
import sys
from argparse import ArgumentParser
from cap_tools.cap_control import CAPInstance, CAPListenModeError
from cap_tools.process import get_diff_info
from cap_tools.utils import get_version, parse_cap_csv, parse_cap_meta
import csv
from zipfile import ZipFile
import configparser
import matplotlib.pyplot as plt
from typing import *


def main(experiments: list, out_dir: str, include_path: bool = False, 
         cmdline: bool=False, log: Callable = print, zip_result: bool = True,
         grain_images: bool = True, diff_images: bool = True, rodhypix: bool = False, 
         jpg: bool = False):


    make_single_figure = False
    write_figure = False

    meta_res = parse_cap_meta(['C:\\XcaliburData\\first_still_run\\grid 2'], include=['pre_'])

    shelldata = []
    peak_table = []
    powder = []

    cap = CAPInstance(start_now=False)

    for exp_info in meta_res:
        
        try:
            the_shelldata, the_peak_table, the_powder = get_diff_info(exp_info['path'], cap=cap, keep_peak_file=False, keep_powder_file=False)
        except Exception as e:
            print(f"Skipping {exp_info['name']} due to error: {e}")
            continue
        
        the_shelldata['experiment'] = exp_info['name']
        the_peak_table['experiment'] = exp_info['name']
        the_powder['experiment'] = exp_info['name']

        shelldata.append(the_shelldata)
        peak_table.append(the_peak_table)
        powder.append(the_powder)
        
    peak_table = pd.concat(peak_table, ignore_index=True)
    shelldata = pd.concat(shelldata, ignore_index=True)
    powder = pd.concat(powder, ignore_index=True)
    shelldata['d_range'] = shelldata['d_max'].round(2).astype(str) + '-' + shelldata['d_min'].round(2).astype(str) + ' Å'
    shelldata['s'] = 0.5 / shelldata['d_max'] + 0.5 / shelldata['d_min']
    shelldata.sort_values(by=['experiment','s'], inplace=True)
    
    shell_plot_data = shelldata.pivot(index='experiment', columns=['s', 'd_range'], values=['I_tot', 'I_peak', 'peak_ratio', 'N_peaks'])
    shell_plot_data = shell_plot_data.T.reset_index(level='s', drop=True).T

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    shell_plot_data['I_tot'].plot.bar(title='Total intensity', stacked=True, width=0.8, rot=90, ax=axs[0], legend=False)
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    shell_plot_data['I_peak'].plot.bar(title='Spot intensity', stacked=True, width=0.8, rot=90, ax=axs[1], legend=False)
    shell_plot_data['N_peaks'].plot.bar(title='Spot number', stacked=True, width=0.8, rot=90, ax=axs[2], legend=False)
    shell_plot_data['peak_ratio'].plot.bar(title='Spot ratio', stacked=False, width=0.8, rot=90, ax=axs[3], legend=False)
    plt.subplots_adjust(hspace=0.35)
    plt.show(fig)

    
def gui():
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter.filedialog import askdirectory, askopenfilename
    
    root = tk.Tk()
    root.title(f'Screening viewer ({get_version()})')
    
    try:       
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    # root.iconbitmap(os.path.join(base_path, "calibrate_dd_icon.ico"))

    info = tk.Text(root, font='TkFixedFont', height=40, width=100, wrap=tk.WORD)
    
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
    
    info_write('Please add data folder(s) (will be searched recursively) or CSV files (exported from Results Viewer).')
    
    proc_buttons = []
    
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
                include_path=False, cmdline=False, 
                log=lambda *msgs: info_write(*msgs, append=True))
            config_window('normal')
            
        except Exception as e:
            config_window('normal')
            info_write(f'Processing failed:\n{str(e)}\n---\nLoaded experiments are:\n'+'\n'.join(input_experiments),
                       append=True)
         
    # ttk.Button(root, text='Set output folder', command=set_output_folder).grid(row=5, column=0, sticky=tk.W)
    ttk.Label(root, textvariable=output_folder).grid(row=5, column=1, sticky=tk.W)
    ttk.Separator(root, orient='horizontal').grid(row=9, columnspan=2, sticky=tk.EW)
    proc_buttons.append(ttk.Button(root, text='Add folder structure', command=add_folder, state='disabled'))
    proc_buttons.append(ttk.Button(root, text='Add CSV from RV', command=add_csv, state='disabled'))
    proc_buttons.append(ttk.Button(root, text='Start processing', command=run_processing, state='disabled'))
    for ii, button in enumerate(proc_buttons):
        button.grid(row=10+ii, column=0, sticky=tk.NW)
    
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
        parser.add_argument('--no-grain-images', action='store_true', help='Do not include grain images in the learning set.')
        parser.add_argument('--no-diff-images', action='store_true', help='Do not include diffraction images in the learning set.')
        parser.add_argument('--no-zip', action='store_true', help='Do not zip the learning set.')
        parser.add_argument('--screening', action='store_true', help='Use screening mode (no grain images, no zipping, keep filenames).')
        parser.add_argument('--rodhypix', action='store_true', help='Use .rodhypix images instead of .tiff.')
        parser.add_argument('--jpg', action='store_true', help='Use .jpg images instead of .tiff.')
        args = parser.parse_args()

        if args.screening:
            args.no_grain_images = True            
            args.no_zip = True
            args.include_path = True
            
        if args.jpg and args.rodhypix:
            raise ValueError('Please specify either --jpg or --rodhypix, not both.')
        
        main(experiments=args.experiments, out_dir=args.out_dir, include_path=args.include_path, cmdline=True, 
             zip_result=not args.no_zip, grain_images=not args.no_grain_images, diff_images=not args.no_diff_images, 
             rodhypix=args.rodhypix, jpg=args.jpg)
        
        
        
