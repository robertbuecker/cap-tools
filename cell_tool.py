import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename, askopenfilenames
from tkinter.messagebox import showinfo
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from cap_tools.cell_list import CellList
from cap_tools.interact_figures import distance_from_dendrogram
from cap_tools.finalization import FinalizationCollection, Finalization
import numpy as np
from collections import defaultdict
from typing import *
import os
from concurrent.futures import ThreadPoolExecutor
from cap_tools.utils import ClusterOptions, TextRedirector
from cap_tools.widgets import ClusterTableWidget
from cap_tools.widgets import FinalizationWidget
from cap_tools.widgets import CellHistogramWidget
from cap_tools.widgets import ClusterWidget
from cap_tools.cap_control import CAPMergeFinalize, CAPInstance, CAPListenModeError
import queue
import sys

cluster_presets = {'Direct': ClusterOptions(preproc='None', metric='Euclidean', method='Ward'),
                   'Whitened': ClusterOptions(preproc='PCA', metric='SEuclidean', method='Average'),
                   'LCV (relative)': ClusterOptions(preproc='None', metric='LCV', method='Ward'),
                   'Diagonals (Å)': ClusterOptions(preproc='Diagonals', metric='Euclidean', method='Ward'),
                   'Standardized': ClusterOptions(preproc='None', metric='SEuclidean', method='Average'),
                   'LCV (Å)': ClusterOptions(preproc='None', metric='aLCV', method='Ward')}

class CellGUI:
    
    def __init__(self, filename: Optional[str] = None, 
                 distance: float = 2.0,
                 method: str = 'average',
                 metric: str = 'euclidian',
                 preproc: str = 'none',
                 use_raw_cell: bool = False,
                 debug: bool = False,
                 **kwargs):
        
        if kwargs:
            print(f'GUI function got unused extra arguments: {kwargs}')
        
        # internal variables
        self.all_cells = CellList(cells=np.empty([0,6]))
        # self.clusters: Dict[int, CellList] = {}
        self.fn: Optional[str] = None
        self.fc: Optional[FinalizationCollection] = None
        self._clustering_disabled: bool = False
        self._click_cid: Optional[int] = None
        
        # initialize master GUI 
        self.root = tk.Tk()
        self.root.geometry('1300x900')
        self.root.title("3D ED/MicroED cell tool")
        
        try:       
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        self.root.iconbitmap(os.path.join(base_path, "cell_tool_icon.ico"))
        
        # tools for multithreading (for long-running tasks) and CAP control        
        self.status_q = queue.Queue()
        self.clipboard_q = queue.Queue()               
        def check_queues():
            if not self.status_q.empty():
                self.set_status_message(self.status_q.get())
            if not self.clipboard_q.empty():
                self.set_clipboard(self.clipboard_q.get())
            self.root.after(100, check_queues)
        check_queues()
        
        self.exec = ThreadPoolExecutor()
        self.cap_instance = CAPInstance()            
        
        ## CONTROL FRAME --
        cf = self.cells_frame = ttk.LabelFrame(self.root, text='Cell Lists')
        self._cf = cf
        
        # file opening
        ttk.Button(cf, text='Open list...', command=self.load_cells).grid(row=0, column=0)
        self.v_use_raw = tk.BooleanVar(cf, value=use_raw_cell)
        self.w_use_raw = ttk.Checkbutton(cf, text='Use raw cells', command=self.reload_cells, variable=self.v_use_raw)
        self.w_use_raw.grid(row=5, column=0)
        self.w_all_fn = ttk.Label(cf, text='(nothing loaded)')
        self.w_all_fn.grid(row=10, column=0)
        
        # Clustering settings
        csf = ttk.LabelFrame(cf, text='Clustering', width=200)
        self._csf = csf
        preset_list = list(cluster_presets.keys()) + ['(none)']
        metric_list = 'Euclidean LCV aLCV SEuclidean Volume'.split()
        method_list = 'Ward Average Single Complete Median Weighted Centroid'.split()        
        preproc_list = 'None PCA Diagonals DiagonalsPCA G6 Standardized Radians Sine'.split()       
        self.v_cluster_setting = {
            'distance': tk.DoubleVar(value=distance),
            'preset': tk.StringVar(value='(none)'),
            'preproc': tk.StringVar(value=[m for m in preproc_list if m.lower() == preproc.lower()][0]), # ugly capitalization workaround
            'metric': tk.StringVar(value=[m for m in metric_list if m.lower() == metric.lower()][0]),
            'method': tk.StringVar(value=[m for m in method_list if m.lower() == method.lower()][0])
        }        
        self.w_cluster_setting = {
            'Distance': ttk.Entry(csf, textvariable=self.v_cluster_setting['distance']),
            'Preset': ttk.OptionMenu(csf, self.v_cluster_setting['preset'], self.v_cluster_setting['preset'].get(), *preset_list, command=self.set_preset),
            'Preprocessing': ttk.OptionMenu(csf, self.v_cluster_setting['preproc'], self.v_cluster_setting['preproc'].get(), *preproc_list), #TODO: add init_clustering as callback?
            'Metric': ttk.OptionMenu(csf, self.v_cluster_setting['metric'], self.v_cluster_setting['metric'].get(), *metric_list),
            'Method': ttk.OptionMenu(csf, self.v_cluster_setting['method'], self.v_cluster_setting['method'].get(), *method_list),
            'Refresh': ttk.Button(csf, text='Refresh', command=self.run_clustering)
        }
        for k in ['Preset', 'Preprocessing', 'Metric', 'Method']:
            self.w_cluster_setting[k].config(w=15)
        for ii, (k, w) in enumerate(self.w_cluster_setting.items()):
            if not (isinstance(w, ttk.Button) or isinstance(w, ttk.Checkbutton)):
                ttk.Label(csf, text=k).grid(row=ii, column=0)
                w.grid(row=ii, column=1)
            else:
                w.grid(row=ii, column=0, columnspan=2)
        csf.grid(row=15, column=0, sticky='EW')
        
        ttk.Button(cf, text='Save selected clusters', command=self.save_clusters).grid(row=20, column=0)
        # ttk.Button(cf, text='Save merging macro', command=self.save_merging_macro).grid(row=25, column=0)
        
        # Merge controls
        mff = ttk.LabelFrame(cf, text='Merging', width=200)
        self._mff = mff
        self.v_merge_fin_setting = {
            'resolution': tk.DoubleVar(mff, value=0.8),
            'top_only': tk.BooleanVar(mff, value=False),
            'top_gral': tk.BooleanVar(mff, value=False),
            'top_ac': tk.BooleanVar(mff, value=False),
            'reintegrate': tk.BooleanVar(mff, value=False)
        }
        self.w_merge_fin_setting = {
            'Resolution': ttk.Entry(mff, textvariable=self.v_merge_fin_setting['resolution']),
            'Top nodes only': ttk.Checkbutton(mff, text='Top nodes only', variable=self.v_merge_fin_setting['top_only']),
            'GRAL on top nodes': ttk.Checkbutton(mff, text='GRAL on top nodes', variable=self.v_merge_fin_setting['top_gral']),
            'AutoChem on top nodes': ttk.Checkbutton(mff, text='AutoChem on top nodes', variable=self.v_merge_fin_setting['top_ac']),
            # 'Reintegrate (proffit)': ttk.Checkbutton(mff, text='Reintegrate (proffit)', variable=self.v_merge_fin_setting['reintegrate']),
        }
        for k in ['Resolution']:#, 'Top nodes only']:
            self.w_merge_fin_setting[k].config(w=15)
        for ii, (k, w) in enumerate(self.w_merge_fin_setting.items()):
            if not (isinstance(w, ttk.Button) or isinstance(w, ttk.Checkbutton)):
                ttk.Label(mff, text=k).grid(row=ii, column=0)
                w.grid(row=ii, column=1)
            else:
                w.grid(row=ii, column=0, columnspan=2)
                
        ttk.Button(mff, text='Merge only', command=lambda *args: self.merge_finalize(
            finalize=False)).grid(
            row=5, column=0, columnspan=2)
        ttk.Button(mff, text='Merge/Finalize', command=lambda *args: self.merge_finalize(
            finalize=True)).grid(
            row=10, column=0, columnspan=2)
        ttk.Button(mff, text='Reset', command=lambda *args: self.reset_clusters()).grid(row=15, column=0, columnspan=2)
        mff.grid_columnconfigure(0, weight=1)
        mff.grid(row=30, column=0)

        # status display        
        self.status = tk.Text(cf, height=5, width=2, font=('Arial', 9), wrap=tk.WORD)        
        self.status.grid(row=90, column=0, columnspan=2, sticky='EW')

        
        # Merge controls
        finf = ttk.LabelFrame(self.root, text='View finalizations from', width=200)
        self._finf = finf
        self.v_fin_view_setting = {

        }
        self.w_fin_view_setting = {

        }
        ttk.Button(finf, text='Files', 
                   command=lambda *args: self.mergefin_widget.update_fc(
                       FinalizationCollection.from_files(
                            filenames = [fn[:-8] for fn in 
                                         askopenfilenames(filetypes=[('Finalization summary', '*_red.sum')])])
                       )).grid(row=15, column=0, columnspan=2)     
        ttk.Button(finf, text='Folder', 
                   command=lambda *args: self.mergefin_widget.update_fc(
                        FinalizationCollection.from_folder(
                            askdirectory(), 
                            ignore_parse_errors=True, include_subfolders=False)
                       )).grid(row=20, column=0, columnspan=2)       
        ttk.Button(finf, text='Subfolders', 
                   command=lambda *args: self.mergefin_widget.update_fc(
                       FinalizationCollection.from_folder(
                            askdirectory(), 
                            ignore_parse_errors=True, include_subfolders=True)
                       )).grid(row=25, column=0, columnspan=2)         
        ttk.Button(finf, text='CSV', 
                   command=lambda *args: self.mergefin_widget.update_fc(
                       FinalizationCollection.from_csv(
                            askopenfilename(filetypes=[('Cell tool result', '*.csv')]), 
                            ignore_parse_errors=True)
                       )).grid(row=30, column=0, columnspan=2)                     
        finf.grid_columnconfigure(0, weight=1)
        finf.grid(row=1, column=1)

        
        # quit button
        button_quit = ttk.Button(self.root, text="Quit", command=self.quit)
        button_quit.grid(row=2, column=1, sticky=tk.N)     
        
        ## DISPLAY TABS --
        
        # initialize tabs
        self.tabs = ttk.Notebook(self.root)
        self.tab_cluster = ttk.Frame(self.tabs)
        self.tab_cluster.columnconfigure(0, weight=100) 
        self.tab_cluster.rowconfigure(0, weight=100)       
        
        # place cluster display tab
        self.cluster_widget = ClusterWidget(self.tab_cluster)
        self.cluster_widget.grid(row=0, column=0, sticky=tk.NSEW)
        self.tabs.add(self.tab_cluster, text='Clustering', sticky=tk.NSEW)
        
        # place cell histogram tab
        tab_cellhist = ttk.Frame(self.tabs)    # for technical reasons, a dummy parent has to be created
        tab_cellhist.columnconfigure(0, weight=100)
        tab_cellhist.rowconfigure(0, weight=100)        
        self.cellhist_widget = CellHistogramWidget(tab_cellhist)
        self.cellhist_widget.grid(row=0, column=0, sticky=tk.NSEW)
        self.tabs.add(tab_cellhist, text='Cell Histogram', sticky=tk.NSEW)        
        def refresh_on_tab(event):
            if self.active_tab == 1:
                self.cellhist_widget.update_histograms(clusters=self.clusters)                
        self.tabs.bind('<<NotebookTabChanged>>', refresh_on_tab)
        
        # place merge/finalize tab
        tab_mergefin = ttk.Frame(self.tabs)    # for technical reasons, a dummy parent has to be created
        tab_mergefin.columnconfigure(0, weight=100)
        tab_mergefin.rowconfigure(0, weight=100)        
        self.mergefin_widget = FinalizationWidget(tab_mergefin)
        self.mergefin_widget.grid(row=0, column=0, sticky='NSEW')
        self.tabs.add(tab_mergefin, text='Merge/Finalize', sticky=tk.NSEW)
        
        # log window
        tab_text_out = ttk.Frame(self.tabs)
        tab_text_out.columnconfigure(0, weight=100)
        tab_text_out.rowconfigure(0, weight=100)             
        self.text_out = tk.Text(tab_text_out, wrap="word")
        self.text_out.grid(row=0, column=0, sticky=tk.NSEW)
        if not debug:
            sys.stdout = TextRedirector(self.text_out, "stdout")
            sys.stderr = TextRedirector(self.text_out, "stderr")        
        self.text_out.tag_configure("stderr", foreground="#b22222")        
        self.tabs.add(tab_text_out, text='Log', sticky=tk.NSEW)       

        ## CLUSTER TABLE --
        self.cluster_table = ClusterTableWidget(self.root, clusters=self.selected_clusters)

        # final assembly of UI
        self.root.columnconfigure(0, weight=100)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=100)
        self.root.rowconfigure(1, weight=0)
        cf.grid(row=0, column=1, sticky=tk.E)        
        self.tabs.grid(column=0, row=0, sticky=tk.NSEW, rowspan=3)
        self.cluster_table.grid(row=10, column=0, columnspan=2, sticky=tk.S)
        
        # if filename is provided on CLI, open it now
        if filename is not None:
            self.fn = filename            
            self.reload_cells()
            
    @property
    def clusters(self) -> Dict[int, CellList]:
        return self.all_cells.clusters
            
    @property
    def selected_clusters(self) -> Dict[int, CellList]:
        return {cl_id: cl for cl_id, cl in self.clusters.items() if cl_id in self.cluster_table.selected_cluster_ids}
            
    def quit(self):
        self.root.destroy()
            
    def set_preset(self, *args):
        preset = self.v_cluster_setting['preset'].get()
        if preset not in cluster_presets:
            print(f'Preset {preset} not found.')
            return
        self.v_cluster_setting['preproc'].set(cluster_presets[preset].preproc)
        self.v_cluster_setting['metric'].set(cluster_presets[preset].metric)
        self.v_cluster_setting['method'].set(cluster_presets[preset].method)
            
    @property
    def active_tab(self):
        return self.tabs.index(self.tabs.select())
        
    def run_clustering(self, distance: Optional[float] = None, tree: Optional[Dict[str,Any]] = None):       
        # invoked directly from clustering recomputation button or clicking into the dendrogram via callback
        # (determined by whether distance parameter is supplied). Gets everything for clustering from the UI,
        # calls CellList.cluster, and writes back the outcome
        
        if self._clustering_disabled:
            # raise RuntimeError('Clustering is disabled. How did you get here?')
            print('Clustering is disabled. Ignoring clustering request.')
            return

        cluster_pars = ClusterOptions(preproc=self.v_cluster_setting['preproc'].get(),
                                    metric=self.v_cluster_setting['metric'].get(),
                                    method=self.v_cluster_setting['method'].get())
        matching_preset = [k for k, v in cluster_presets.items() if v == cluster_pars]
        
        if len(matching_preset) == 1:
            self.v_cluster_setting['preset'].set(matching_preset[0])
        else:
            self.v_cluster_setting['preset'].set('(none)')                 

        if distance is None:
            # no distance as input parameter: get from text field and redraw figure
            distance = self.v_cluster_setting['distance'].get()
            redraw = True           
        else:
            # distance as input parameter: called from figure callback
            self.v_cluster_setting['distance'].set(distance)
            redraw = False
            
        #TODO consider a structure where the plots directly run CellList.cluster and the GUI is updated via a callback. It's very confusing as it is.
        node_cids = self.all_cells.cluster(distance=None if distance==0 else distance, cluster_pars=cluster_pars)     
        
        self.cluster_widget.tree = tree
        self.cluster_table.update_table(clusters=self.clusters)
        
        if self.active_tab == 1:
            self.cellhist_widget.update_histograms(clusters=self.clusters)
            
        if redraw:            
            try:
                labels = [d['Experiment name'] for d in self.all_cells.ds]
            except KeyError as err:
                print('Experiment names not found in CSV list. Consider including them.')
                labels = None          
                        
            # generate the dendrogram, supplying the function itself as callback
            _, self._click_cid = distance_from_dendrogram(self.all_cells._z, ylabel=cluster_pars.metric, initial_distance=distance,
                                labels=labels, fig_handle=self.cluster_widget.fig, callback=lambda distance, tree: self.run_clustering(distance, tree))            
            
        return node_cids, self.cluster_table.apply_cluster_colors

    def reload_cells(self):
        raw = self.v_use_raw.get()        
        print(f'Loading cells from {self.fn}')
        self.all_cells = CellList.from_csv(self.fn, use_raw_cell=raw) #TODO change this to selection of raw cells           
        self.w_all_fn.config(text=os.path.basename(self.fn) + (' (raw)' if raw else ''))       
        self.run_clustering()
            
    def load_cells(self):
        self.fn = os.path.normpath(
            askopenfilename(title='Open cell list file', filetypes=(('CrysAlisPro', '*.csv'),))
        )
        self.reload_cells()
        
    def load_clusters(self, fn: str = None):
        if self.fn is None:
            self.fn = os.path.normpath(
                askopenfilename(title='Open cluster info file', filetypes=(('Cell Tool', '*_cluster_info.csv'),))
            )            
        
    def save_clusters(self, fn_template: Optional[str] = None):
        
        if fn_template is None:
            fn_template = asksaveasfilename(confirmoverwrite=False, title='Select root filename for cluster CSVs', 
                                            initialdir=os.path.dirname(self.fn), initialfile=os.path.basename(self.fn),
                                            filetypes=[('CrysAlisPro CSV', '*.csv')])
            
            if not fn_template:
                print('No filename selected, canceling.')
                return
                
        self.all_cells.save_clusters(fn_template, 
                                     list_fn=self.fn + (' (raw)' if self.v_use_raw.get() else ''),
                                     selection = self.cluster_table.selected_cluster_ids)
                
    def _set_clustering_active(self, active: bool=True):
        # activate/deactive clustering controls
        
        self._clustering_disabled = not active        
        for child in self._csf.winfo_children():
            child.config(state='normal' if active else 'disabled')        
        if active:
            self.cluster_table.cluster_view.enable()
        else:
            self.cluster_table.cluster_view.disable()                        
        if (not active) and (self._click_cid is not None):
            self.cluster_widget.canvas.mpl_disconnect(self._click_cid)
        else:
            self.run_clustering()
            #TODO properly re-activate the plot!
        
    def set_status_message(self, msg):
        self.status.delete(1.0, tk.END)
        self.status.insert(tk.END, msg)
        
    def set_clipboard(self, msg):
        self.root.clipboard_clear()
        self.root.clipboard_append(msg)        
        
    def merge_finalize(self, finalize: bool = True):
        
        if not self.cluster_table.selected_cluster_ids:
            showinfo('No cluster selected', 'Please first select one or more cluster(s).')
            return

        ii = 1
        while True:
            results_folder = os.path.splitext(self.fn)[0] + f'_clusters-run{ii}'
            if os.path.exists(results_folder):
                ii += 1
            else:
                break
        
        merge_fn = self.all_cells.save_clusters(fn_template=os.path.join(results_folder, f'run{ii}'), 
                                     out_dir=results_folder,
                                     list_fn=self.fn + (' (raw)' if self.v_use_raw.get() else ''),
                                     selection=self.cluster_table.selected_cluster_ids,
                                     top_only=self.v_merge_fin_setting['top_only'].get())

        cap_control = CAPMergeFinalize(merge_file=merge_fn,
                                       cap_instance=self.cap_instance,
                                       message_func=self.status_q)
      
        if not finalize:        
            
            merge_future = self.exec.submit(cap_control.merge,
                                            reintegrate=self.v_merge_fin_setting['reintegrate'].get())
            
            def check_proc_running():
                if merge_future.done():
                    if e := merge_future.exception():
                        raise e
                    self.status_q.put(f'Merging completed into {results_folder}')
                else:
                    self.root.after(100, check_proc_running)
                    
            self.root.after(100, check_proc_running)
            
            # cap_control.merge(reintegrate=self.v_merge_fin_setting['reintegrate'].get())                       
                
        else:
            self._set_clustering_active(False)
            # TODO why is the following required?
            for child in self._mff.winfo_children():
                child.config(state='normal')      
            
            fin_future = self.exec.submit(cap_control.finalize, 
                                          res_limit=self.v_merge_fin_setting['resolution'].get(),
                                          top_gral=self.v_merge_fin_setting['top_gral'].get(),
                                          top_ac=self.v_merge_fin_setting['top_ac'].get(),
                                          reintegrate=self.v_merge_fin_setting['reintegrate'].get())            
                
            def check_fin_running():
                if fin_future.done():
                    if e := fin_future.exception():
                        raise e                    
                    self.fc = fin_future.result()
                    print('OVERALL RESULTS TABLE')
                    print('---------------------')
                    print(self.fc.overall_highest)  
                    self.status_q.put(f'Finalization completed into {results_folder}')                        
                    for child in self._mff.winfo_children():
                        child.config(state='normal')     
                    self.mergefin_widget.update_fc(self.fc)
                else: 
                    self.root.after(100, check_fin_running)              
                
            self.root.after(100, check_fin_running)
        
    def reset_clusters(self):
        self._set_clustering_active(True)
        self.mergefin_widget.clear()
        self.set_status_message('Finalization viewer cleared. Clustering unlocked.')
        
            
def parse_args():
    import argparse

    description = "Program for clustering unit cells from crystallography experiments. Contains some clustering algorithm and display "
    "functions from edtools by Stef Smeets (https://github.com/instamatic-dev/edtools)."
    
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        
    parser.add_argument("filename",
                        type=str, metavar="FILE", nargs='?',
                        help="Path to .csv cell parameter file")

    parser.add_argument("-m","--method",
                        action="store", type=str, dest="method",
                        choices="single average complete median weighted centroid ward".split(),
                        help="Linkage algorithm to use (see `scipy.cluster.hierarchy.linkage`)")

    parser.add_argument("-t","--metric",
                        action="store", type=str, dest="metric",
                        choices="euclidean lcv volume".split(),
                        help="Metric for calculating the distance between items (Euclidian distance, cell volume, LCV, and aLCV as in CCP4-BLEND)")

    parser.add_argument("-p", "--preprocessing",
                        action="store", type=str, dest="preproc",
                        choices='none standardized pca radians sine'.split(),
                        help="Options for conditioning (pre-processing) cell data.")
    
    parser.add_argument("-w","--raw-cell",
                       action="store_true", dest="use_raw_cell",
                       help="Use the raw lattice (from Lattice Explorer/IDXREF as opposed to the refined one from GRAL/CORRECT) for unit cell finding and clustering")
    
    parser.add_argument("--debug",
                        action="store_true", dest="debug",
                        help="Debug mode - print errors and log messages to console instead of log window.")

    parser.set_defaults(filename=None,
                        distance=0.0,
                        method="ward",
                        metric="euclidean",
                        use_raw_cell=False,
                        preproc='none')
    
    options = parser.parse_args()

    return vars(options)

if __name__ == '__main__':
    cli_args = parse_args()    
    window = CellGUI(**cli_args)
    window.root.mainloop()
    # main()
