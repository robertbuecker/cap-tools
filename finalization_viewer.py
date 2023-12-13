import tkinter as tk
import tkinter.ttk as ttk
import ttkwidgets
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename, askopenfilenames
import math
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from cap_tools.cell_list import CellList
import numpy as np
from collections import defaultdict
from cap_tools.interact_figures import distance_from_dendrogram
from typing import *
from time import time
from collections import namedtuple
from cap_tools.finalization import Finalization, FinalizationCollection


class PlotWidget(ttk.Frame):
    
    def __init__(self, parent: tk.BaseWidget):
        
        super().__init__(parent)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()

        # pack_toolbar=False will make it easier to use a layout manager later on.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()

        self.canvas.mpl_connect(
            "key_press_event", lambda event: print(f"you pressed {event.key}"))
        self.canvas.mpl_connect("key_press_event", key_press_handler)

        self.init_figure_controls()
        
        self.rowconfigure(0, weight=100)
        self.columnconfigure(0, weight=100)
        
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
        
        self.toolbar.grid(row=1, column=0, sticky=tk.S)
        
    def init_figure_controls(self):        
        self.controls = ttk.Frame(self)
        

class FinList(ttk.Frame):
    
    def __init__(self, root: tk.BaseWidget, finalizations: FinalizationCollection):
        super().__init__(root)

        fv = self.fin_view = ttkwidgets.CheckboxTreeview(self, height=6, selectmode='browse')
        self._fc = finalizations
        self._entry_ids: List[int] = []

        # fv.heading('Name', text='Name')
         
        fv.bind('<<TreeviewSelect>>', self.show_entry_info)
        fv.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=fv.yview)
        fv.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        
    def update_table(self, finalizations: Optional[FinalizationCollection] = None):
        if finalizations is not None:
            self._fc = finalizations
            
        for iid in self._entry_ids:
            if iid not in self._fc:
                self.fin_view.delete(iid)
                self._entry_ids.remove(iid)
            
        for name, fin in self._fc.items():
            if name not in self._entry_ids:
                self._entry_ids.append(self.fin_view.insert('', tk.END, iid=name, text=name))
                self.fin_view.change_state(name, 'unchecked')
            
    def show_entry_info(self, event):        
        print(f'--- FINALIZATION {self.fin_view.selection()[0]} ---')            
        print(self.selected_finalization)
        print('')
            
        print(f'--- OVERALL STATISTICS ---')       
        print(self.checked_finalizations.overall)
        print('')

        print(f'--- HIGHEST SHELL ---')       
        print(self.checked_finalizations.highest_shell)
        print('')
                        
    @property
    def checked_finalizations(self) -> FinalizationCollection:
        fc_checked = FinalizationCollection()
        for iid in self.fin_view.get_checked():
            fc_checked[iid] = self._fc[iid]
        return fc_checked                
    
    @property
    def selected_finalization(self) -> Finalization:        
        return self._fc[self.fin_view.selection()[0]].shells            

class FOMList(ttk.Frame):
    
    def __init__(self, root: tk.BaseWidget, finalizations: FinalizationCollection):
        super().__init__(root)

        fv = self.fom_view = ttkwidgets.CheckboxTreeview(self, height=6)
        self._fc = finalizations
        self._entry_ids: List[int] = []
         
        #TODO Bind a nice function to show that FOM for all selected sets
        # fv.bind('<<TreeviewSelect>>', self.show_entry_info)
        fv.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=fv.yview)
        fv.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        
    def update_table(self, finalizations: Optional[FinalizationCollection] = None):
        if finalizations is not None:
            self._fc = finalizations
            
        for iid in self._entry_ids:
            if iid not in self._fc.foms:
                self.fom_view.delete(iid)
                self._entry_ids.remove(iid)
            
        for fom in self._fc.foms:
            if fom not in self._entry_ids:
                self._entry_ids.append(self.fom_view.insert('', tk.END, iid=fom, text=fom))
                self.fom_view.change_state(fom, 'unchecked')
            
    @property
    def checked_foms(self) -> List[str]:
        return list(self.fom_view.get_checked())


class FinalizationGUI:

    def __init__(self, folder: Optional[str] = None,
                 subfolders: bool = False,
                 exclude_auto: bool = False,
                 exclude_autored: bool = False,                 
                 **kwargs):
        
        # DEFAULTS FOR METHOD ARE DEFINED WITH CLI ARGUMENTS
        if kwargs:
            print(f'GUI function got unused extra arguments: {kwargs}')
        
        self.fc = FinalizationCollection()
        self.foms: List[str] = []
        
        # self.root = tk.Tk()
        self.root = tk.Tk()
        self.root.geometry('1300x800')
        self.root.title('Merging statistics comparison tool')
        
        self.plot_widget = PlotWidget(self.root)
        self.plot_widget.grid(row=0, column=0, sticky=tk.NSEW)        
        
        cf = self.control_frame = ttk.LabelFrame(self.root, text='Finalizations')
        
        # file management
        ttk.Button(cf, text='Add CSV', command=self.add_csv).grid(row=0, column=0)
        ttk.Button(cf, text='Add folder', command=self.add_folder).grid(row=5, column=0)
        ttk.Button(cf, text='Add subfolders', command=self.add_subfolders).grid(row=10, column=0)
        ttk.Button(cf, text='Add manual', command=self.add_files).grid(row=15, column=0)
        
        self.fin_list = FinList(cf, finalizations=self.fc)
        self.fin_list.grid(row=20, column=0, sticky=tk.NS)
        
        ttk.Button(cf, text='Clear', command=self.clear).grid(row=25, column=0)
 
        self.fom_list = FOMList(cf, finalizations=self.fc)
        self.fom_list.grid(row=30, column=0, sticky=tk.NS)
 
        ttk.Button(cf, text='Plot', command=self.plot).grid(row=35, column=0)
 
        # final assembly of UI
        self.root.columnconfigure(0, weight=100)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=100)
        # self.root.rowconfigure(1, weight=0)
        cf.grid(row=0, column=1, sticky=tk.E)        
        self.plot_widget.grid(column=0, row=0, sticky=tk.NSEW)
        
    def plot(self):

        self.plot_widget.fig.clear()
        axs = self.plot_widget.fig.subplots(nrows=2, ncols=3, sharex=True)

        for lbl, fin in self.fin_list.checked_finalizations.items():
            for ii, (fom, ax) in enumerate(zip(self.fom_list.checked_foms, axs.ravel())):
                if fom not in fin.foms:
                    ov = np.nan
                    ph = ax.plot([fin.shells['1/d'].min(), fin.shells['1/d'].max()], [0, 0])
                    ph[0].set_visible(False)
                    continue
                ov = fin.overall[fom].iloc[0]
                fin.shells.plot(x='1/d', y=fom, ax=ax, label=f'{lbl}: {ov}' if ii==0 else f'{ov}')
                ax.set_title(fom)
                
        self.plot_widget.fig.canvas.draw()
        
    def update_tables(self):
        self.fin_list.update_table()
        self.fom_list.update_table()
        
    def add_csv(self):    
        self.fc.update(
            FinalizationCollection.from_csv(
                askopenfilename(filetypes=[('Cell tool result', '*.csv')]), 
                ignore_parse_errors=True)
            )
        self.update_tables()
    
    def add_folder(self):    
        self.fc.update(
            FinalizationCollection.from_folder(
                askdirectory(), 
                ignore_parse_errors=True, include_subfolders=False)
            )
        self.update_tables()

        
    def add_subfolders(self):    
        self.fc.update(
            FinalizationCollection.from_folder(
                askdirectory(), 
                ignore_parse_errors=True, include_subfolders=True)
            )
        self.update_tables()        
        
    def add_files(self):
        summaries = askopenfilenames(filetypes=[('Finalization summary', '*_red.sum')])
        
        self.fc.update(
            FinalizationCollection.from_files(
                filenames = [fn[:-8] for fn in summaries])
            )          
        self.update_tables()        
        
    def clear(self):
        self.fc.clear()  
        self.update_tables()                

            
            
# def parse_args():
#     import argparse

#     description = "Visualization tool for comparing finalization runs from CrysAlisPro"
    
#     parser = argparse.ArgumentParser(description=description,
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
        
#     parser.add_argument("folder",
#                         type=str, metavar="FILE", nargs='?',
#                         help="Folder to look for finalization summary files")

#     parser.add_argument("-s", "--subfolders",
#                         action="store_true", dest="subfolders",
#                         help="Include subfolders for scanning for finalization files")
    
#     parser.add_argument("--exclude-auto",
#                         action='store_true', dest='exclude_auto',
#                         help='Exclude finalizations from full autoprocessing (_auto)')
    
#     parser.add_argument("--exclude-autored",
#                         action='store_true', dest='exclude_auto',
#                         help='Exclude finalizations from autoreduction (_autored)')

#     parser.set_defaults(folder=None,
#                         subfolders=False,
#                         exclude_auto=False,
#                         exclude_autored=False)
    
#     options = parser.parse_args()

#     return vars(options)

if __name__ == '__main__':
    cli_args = {} # parse_args()    
    window = FinalizationGUI(**cli_args)
    window.root.mainloop()
    # main()
