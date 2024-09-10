import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename
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
from cap_tools.utils import ClusterOptions
from cap_tools.widgets import ClusterTableWidget
from cap_tools.widgets import FinalizationWidget
from cap_tools.widgets import CellHistogramWidget
from cap_tools.widgets import ClusterWidget
from cap_tools.cap_control import CAPMergeFinalize
import queue

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
                 **kwargs):
        
        if kwargs:
            print(f'GUI function got unused extra arguments: {kwargs}')
        
        # internal variables
        self.all_cells = CellList(cells=np.empty([0,6]))
        self.clusters: Dict[int, CellList] = {}
        self.fn: Optional[str] = None
        self.fc: Optional[FinalizationCollection] = None
        self._clustering_disabled: bool = False
        self._click_cid: Optional[int] = None
        
        # initialize master GUI 
        self.root = tk.Tk()
        self.root.geometry('1300x800')
        self.root.title("3D ED/MicroED cell tool")
        
        # tools for multithreading (for long-running tasks)
        self.exec = ThreadPoolExecutor()
        self.status_q = queue.Queue()
        self.clipboard_q = queue.Queue()            
        def check_queues():
            if not self.status_q.empty():
                self.set_status_message(self.status_q.get())
            if not self.clipboard_q.empty():
                self.set_clipboard(self.clipboard_q.get())
            self.root.after(100, check_queues)
        check_queues()
        
        ## CONTROL FRAME --
        cf = self.control_frame = ttk.LabelFrame(self.root, text='Cell Lists')
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
        
        # Merge/Finalize controls
        mff = ttk.LabelFrame(cf, text='Merging/Finalization', width=200)
        self._mff = mff
        self.v_merge_fin_setting = {
            'resolution': tk.DoubleVar(mff, value=0.8),
            'top_only': tk.BooleanVar(mff, value=False)
        }
        self.w_merge_fin_setting = {
            'Resolution': ttk.Entry(mff, textvariable=self.v_merge_fin_setting['resolution']),
            #'Top nodes only': ttk.Checkbutton(mff, text='Top nodes only', variable=self.v_merge_fin_setting['top_only'])
        }
        for k in ['Resolution']:#, 'Top nodes only']:
            self.w_merge_fin_setting[k].config(w=15)
        for ii, (k, w) in enumerate(self.w_merge_fin_setting.items()):
            if not (isinstance(w, ttk.Button) or isinstance(w, ttk.Checkbutton)):
                ttk.Label(mff, text=k).grid(row=ii, column=0)
                w.grid(row=ii, column=1)
            else:
                w.grid(row=ii, column=0, columnspan=2)
                
        ttk.Button(mff, text='Merge only', command=lambda *args: self.merge_finalize(finalize=False)).grid(row=5, column=0, columnspan=2)
        ttk.Button(mff, text='Merge/Finalize', command=lambda *args: self.merge_finalize(finalize=True)).grid(row=10, column=0, columnspan=2)
        ttk.Button(mff, text='Reload last', 
                   command=lambda *args: self.mergefin_widget.update_fc(
                       FinalizationCollection.from_csv(os.path.splitext(self.fn)[0] + '_nodes.csv')
                       )).grid(row=15, column=0, columnspan=2)
        mff.grid_columnconfigure(0, weight=1)
        mff.grid(row=30, column=0)

        # status display        
        self.status = tk.Text(cf, height=5, width=2, font=('Arial', 9), wrap=tk.WORD)        
        self.status.grid(row=90, column=0, columnspan=2, sticky='EW')
        
        # quit button
        button_quit = ttk.Button(cf, text="Quit", command=self.quit)
        button_quit.grid(row=100, column=0, sticky=tk.S)
        
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

        ## CLUSTER TABLE --
        self.cluster_table = ClusterTableWidget(self.root, clusters=self.selected_clusters)

        # final assembly of UI
        self.root.columnconfigure(0, weight=100)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=100)
        self.root.rowconfigure(1, weight=0)
        cf.grid(row=0, column=1, sticky=tk.E)        
        self.tabs.grid(column=0, row=0, sticky=tk.NSEW)
        self.cluster_table.grid(row=1, column=0, columnspan=2, sticky=tk.S)
        
        # if filename is provided on CLI, open it now
        if filename is not None:
            self.fn = filename
            self.reload_cells()
            
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
            
        self.clusters = self.all_cells.cluster(distance=None if distance==0 else distance, cluster_pars=cluster_pars)     
        
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
                        
            _, self._click_cid = distance_from_dendrogram(self.all_cells._z, ylabel=cluster_pars.metric, initial_distance=distance,
                                labels=labels, fig_handle=self.cluster_widget.fig, callback=lambda distance, tree: self.run_clustering(distance, tree))                        

    def reload_cells(self):
        raw = self.v_use_raw.get()        
        self.all_cells = CellList.from_csv(self.fn, use_raw_cell=raw) #TODO change this to selection of raw cells           
        self.w_all_fn.config(text=os.path.basename(self.fn) + (' (raw)' if raw else ''))       
        self.run_clustering()
            
    def load_cells(self):
        self.fn = os.path.normpath(
            askopenfilename(title='Open cell list file', filetypes=(('CrysAlisPro', '*.csv'),))
        )
        self.reload_cells()
        
    def save_clusters(self, fn_template: Optional[str] = None):
        # TODO Factor into cell list class
        
        if fn_template is None:
            fn_template = asksaveasfilename(confirmoverwrite=False, title='Select root filename for cluster CSVs', 
                                            initialdir=os.path.dirname(self.fn), initialfile=os.path.basename(self.fn),
                                            filetypes=[('CrysAlisPro CSV', '*.csv')])
            
            if not fn_template:
                print('No filename selected, canceling.')
                return
        
        info_fn = os.path.splitext(self.fn)[0] + '_cluster_info.csv'
        ver = 1
        with open(info_fn, 'w') as ifh:
            ifh.write(
                f'VERSION {ver}\n'
                f'HEADER INFO:\n'
                f'Experiment list: {self.fn}\n' #TODO add (raw) info
                f'Preprocessing: {self.all_cells._cluster_pars.preproc}\n'
                f'Metric: {self.all_cells._cluster_pars.metric}\n'
                f'Method: {self.all_cells._cluster_pars.method}\n'
                f'Distance: {self.all_cells._distance}\n'
            )
            ifh.write('Name,File path,Cluster,Data sets,Merge code\n')
            
            for ii, (c_id, cluster) in enumerate(self.clusters.items()):
                if c_id not in self.cluster_table.selected_cluster_ids:
                    print(f'Skipping Cluster {c_id} (not selected in list)')
                    continue
                out_paths, in_paths, out_codes, out_info = cluster.get_merging_paths(prefix=f'C{c_id}', short_form=True)                
                for out, (in1, in2), code, info in zip(out_paths, in_paths, out_codes, out_info):
                    ifh.write(f'{os.path.basename(out)},{out},{c_id},{info},{code}\n')
                cluster_fn = os.path.splitext(fn_template)[0] + f'-cluster_{ii}_ID{c_id}.csv'
                cluster.to_csv(cluster_fn)
                print(f'Wrote cluster {c_id} with {len(cluster)} crystals to file {cluster_fn}')
                
    def set_clustering_active(self, active: bool=True):
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
        
    def merge_finalize(self, finalize: bool = True, top_only: bool = False):
        
        if not self.cluster_table.selected_cluster_ids:
            showinfo('No cluster selected', 'Please first select one or more cluster(s).')
            return

        cap_control = CAPMergeFinalize(path=os.path.splitext(self.fn)[0],
                                       clusters=self.cluster_table.selected_clusters,
                                       message_func=self.status_q, cmd_func=self.clipboard_q)
        
        cap_control.cluster_merge(write_mac=True)                       
                
        if finalize:

            self.set_clustering_active(False)
            # TODO why is the following required?
            for child in self._mff.winfo_children():
                child.config(state='normal')      
            
            fin_future = self.exec.submit(cap_control.cluster_finalize, 
                                          res_limit=self.v_merge_fin_setting['resolution'].get())            
                
            def check_fin_running():
                if fin_future.done():
                    self.fc = fin_future.result()
                    print('OVERALL RESULTS TABLE')
                    print('---------------------')
                    print(self.fc.overall_highest)      
                    self.status.delete(1.0, tk.END)
                    self.status.insert(tk.END, 'Finalization complete')              
                    for child in self._mff.winfo_children():
                        child.config(state='normal')     
                    self.mergefin_widget.update_fc(self.fc)
                else: 
                    self.root.after(100, check_fin_running)              
                
            self.root.after(100, check_fin_running)

        if top_only:
            raise NotImplementedError('Top-node-only finalization not supported (yet)')
        
            
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

    parser.set_defaults(filename=None,
                        binsize=0.5,
                        cluster=True,
                        distance=0.0,
                        method="ward",
                        metric="euclidean",
                        use_raw_cell=False,
                        raw=False,
                        preproc='none')
    
    options = parser.parse_args()

    return vars(options)

if __name__ == '__main__':
    cli_args = parse_args()    
    window = CellGUI(**cli_args)
    window.root.mainloop()
    # main()
