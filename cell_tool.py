import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename
import math
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from cap_tools.cell_list import CellList
import numpy as np
from collections import defaultdict
from cap_tools.interact_figures import distance_from_dendrogram
from typing import *
from time import time
import os

      
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

class ClusterWidget(PlotWidget):
    
    def __init__(self, parent):
        super().__init__(parent)
        
    def init_figure_controls(self):
        super().init_figure_controls()
        ttk.Label(self.controls, text='Nothing here').grid(row=0, column=1)
        ttk.Button(self.controls, text='Don\'t click!', command=lambda *args: print('nothing')).grid(row=0, column=2)


class CellHistogramWidget(PlotWidget):
    
    def __init__(self, parent):
        super().__init__(parent)
                
        axs = self.fig.subplots(2, 4)
        axs[-1,-1].remove()
        self.axs = {'a': axs[0,0], 'b': axs[0,1], 'c': axs[0,2],
                    'al': axs[1,0], 'be': axs[1,1], 'ga': axs[1,2],
                    'V': axs[0,3]}
        self.fig.subplots_adjust(hspace=0.5)        
        
    def init_figure_controls(self):
        super().init_figure_controls()
        ttk.Label(self.controls, text='Nothing here').grid(row=0, column=1)
        ttk.Button(self.controls, text='Don\'t click!', command=lambda *args: print('nothing')).grid(row=0, column=2)
        
    def update_histograms(self, clusters: Dict[int, CellList]):
        
        print('Updating cell parameter histograms...')
        t0 = time()
        cluster_cells = {c_id: np.concatenate([cluster.cells, cluster.volumes.reshape(-1,1)], axis=1) 
                         for c_id, cluster in sorted(clusters.items())}

        for ii, (lbl, ax) in enumerate(self.axs.items()):
            ax.cla()
            ax.hist([cl[:, ii] for cl in cluster_cells.values()], 
                    histtype='bar', label=list(cluster_cells.keys()))    
            ax.set_title(lbl)
            # ax.set_yticks([])
            if lbl=='V':
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5))
        
        self.fig.canvas.draw()
        
        print(f'Updating histograms took {1000*(time()-t0):.0f} ms')
        


class ClusterTableWidget(ttk.Frame):
    
    def __init__(self, root: tk.BaseWidget, clusters: Dict[int, CellList]):
        super().__init__(root)
        
        ct_columns = ['ID', 'obs', 'a', 'b', 'c', 'al', 'be', 'ga', 'V']

        cv = self.cluster_view = ttk.Treeview(self, columns=ct_columns, show='headings', height=6)
        self._clusters = clusters        
        self._selected_cluster = None
        self._entry_ids = []

        cv.heading('ID', text='ID')
        cv.heading('obs', text='# Cryst')
        cv.heading('a', text='a')
        cv.heading('b', text='b')
        cv.heading('c', text='c')
        cv.heading('al', text='alpha')
        cv.heading('be', text='beta')
        cv.heading('ga', text='gamma')
        cv.heading('V', text='volume')
        
        cv.column('ID',  width=20)
        cv.column('obs', width=30)
        cv.column('a',   width=170)
        cv.column('b',   width=170)
        cv.column('c',   width=170)
        cv.column('al',  width=170)
        cv.column('be',  width=170)
        cv.column('ga',  width=170)
        cv.column('V',   width=170) 
         
        cv.bind('<<TreeviewSelect>>', self.show_entry_info)
         
        cv.grid(row=0, column=0, sticky=tk.NSEW)
        
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=cv.yview)
        cv.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        
    def update_table(self, clusters: Optional[Dict[int, CellList]] = None):
        
        if clusters is not None:
            self._clusters = clusters
            
        for the_id in self._entry_ids:
            self.cluster_view.delete(the_id)
            
        self._entry_ids = []
            
        for c_id, cl in sorted(self._clusters.items()):
            stats = cl.stats
            
            cpar_strs = []
            for avg, std, lo, hi in zip(stats.mean, stats.std, stats.min, stats.max):
                digits = max(0,
                             -int(math.floor(math.log10(std)))+1 if std != 0 else 0, 
                             -int(math.floor(math.log10(hi-lo))) if std != 0 else 0)
                cpar_strs.append('{0:.{4}f} ({1:.{4}f}) [{2:.{4}f}, {3:.{4}f}]'.format(avg, std, lo, hi, digits))
                                         
            self._entry_ids.append(self.cluster_view.insert('', tk.END, values=[c_id, len(cl)] + cpar_strs))
            
    def show_entry_info(self, event):
        for cluster_id in self.selected_clusters:
            print(f'--- CLUSTER {cluster_id} ---')            
            print(self._clusters[cluster_id].table)
            
    @property
    def selected_clusters(self):
        return [self.cluster_view.item(selected)['values'][0] 
                for selected in self.cluster_view.selection()]
        

class CellGUI:
    
    def __init__(self, filename: Optional[str] = None, 
                 distance: float = 2.0,
                 method: str = 'average',
                 metric: str = 'euclidian',
                 preproc: str = 'none',
                 use_raw_cell: bool = False,
                 **kwargs):
        
        # DEFAULTS FOR METHOD ARE DEFINED WITH CLI ARGUMENTS
        if kwargs:
            print(f'GUI function got unused extra arguments: {kwargs}')
        
        # internal variables
        self.all_cells = CellList(cells=np.empty([0,6]))
        self.clusters: Dict[int, CellList] = {}
        self.fn: Optional[str] = None
        
        # initialize master GUI 
        self.root = tk.Tk()
        self.root.geometry('1300x800')
        self.root.title("3D ED/MicroED cell tool")
        
        cf = self.control_frame = ttk.LabelFrame(self.root, text='Cell Lists')
        
        # file opening
        ttk.Button(cf, text='Open list...', command=self.load_cells).grid(row=0, column=0)
        self.v_use_raw = tk.BooleanVar(cf, value=use_raw_cell)
        self.w_use_raw = ttk.Checkbutton(cf, text='Use raw cells', command=self.reload_cells, variable=self.v_use_raw)
        self.w_use_raw.grid(row=5, column=0)
        self.w_all_fn = ttk.Label(cf, text='(nothing loaded)')
        self.w_all_fn.grid(row=10, column=0)

        metric_list = 'Euclidean SEuclidean LCV Volume'.split()
        method_list = 'Average Single Complete Median Weighted Centroid Ward'.split()        
        preproc_list = 'None Standardize PCA Radians Sine'.split()
        
        # clustering (default) settings
        csf = ttk.LabelFrame(cf, text='Clustering')
        self.v_cluster_setting = {
            'distance': tk.DoubleVar(value=distance),
            'method': tk.StringVar(value=[m for m in method_list if m.lower() == method.lower()][0]),
            'metric': tk.StringVar(value=[m for m in metric_list if m.lower() == metric.lower()][0]),
            'preproc': tk.StringVar(value=[m for m in preproc_list if m.lower() == preproc.lower()][0])
        }        
        
        # ugly way to get rid of capitalization issues
        
        self.w_cluster_setting = {
            'Distance': ttk.Entry(csf, textvariable=self.v_cluster_setting['distance']),
            'Preprocessing': ttk.OptionMenu(csf, self.v_cluster_setting['preproc'], self.v_cluster_setting['preproc'].get(), *preproc_list),
            'Metric': ttk.OptionMenu(csf, self.v_cluster_setting['metric'], self.v_cluster_setting['metric'].get(), *metric_list),
            'Method': ttk.OptionMenu(csf, self.v_cluster_setting['method'], self.v_cluster_setting['method'].get(), *method_list),
            'Refresh': ttk.Button(csf, text='Refresh', command=self.init_clustering)
        }        
        for ii, (k, w) in enumerate(self.w_cluster_setting.items()):
            if not (isinstance(w, ttk.Button) or isinstance(w, ttk.Checkbutton)):
                ttk.Label(csf, text=k).grid(row=ii, column=0)
                w.grid(row=ii, column=1)
            else:
                w.grid(row=ii, column=0, columnspan=2)
        csf.grid(row=15, column=0)
        
        ttk.Button(cf, text='Save selected clusters', command=self.save_clusters).grid(row=20, column=0)
        ttk.Button(cf, text='Save merging macro', command=self.save_merging_macro).grid(row=25, column=0)
        
        # quit button
        button_quit = ttk.Button(cf, text="Quit", command=self.root.destroy)
        button_quit.grid(row=100, column=0, sticky=tk.S)
        
        # plots
        self.tabs = ttk.Notebook(self.root)
        self.tab_cluster = ttk.Frame(self.tabs)
        self.tab_cluster.columnconfigure(0, weight=100) 
        self.tab_cluster.rowconfigure(0, weight=100)
        
        self.cluster_widget = ClusterWidget(self.tab_cluster)
        self.cluster_widget.grid(row=0, column=0, sticky=tk.NSEW)
        self.tabs.add(self.tab_cluster, text='Clustering', sticky=tk.NSEW)
        
        self.tab_cellhist = ttk.Frame(self.tabs)
        self.tab_cellhist.columnconfigure(0, weight=100)
        self.tab_cellhist.rowconfigure(0, weight=100)
        
        self.cellhist_widget = CellHistogramWidget(self.tab_cellhist)
        self.cellhist_widget.grid(row=0, column=0, sticky=tk.NSEW)
        self.tabs.add(self.tab_cellhist, text='Cell Histogram', sticky=tk.NSEW)
        
        def refresh_on_tab(event):
            if self.active_tab == 1:
                self.cellhist_widget.update_histograms(clusters=self.clusters)
                
        self.tabs.bind('<<NotebookTabChanged>>', refresh_on_tab)

        # cluster view
        self.cluster_table = ClusterTableWidget(self.root, clusters=self.clusters)

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
    def active_tab(self):
        return self.tabs.index(self.tabs.select())
        
    def init_clustering(self):

        def recluster():        
            cluster_args = {k: v.get() for k, v in self.v_cluster_setting.items()}
            cluster_args = {k: v.lower() if isinstance(v, str) else v for k, v in cluster_args.items()}
            cluster_args['distance'] = None if cluster_args['distance'] == 0 else cluster_args['distance']
            self.clusters, z = self.all_cells.cluster(**cluster_args)
            self.cluster_table.update_table(clusters=self.clusters)
            if self.active_tab == 1:
                self.cellhist_widget.update_histograms(clusters=self.clusters)
            return z, cluster_args
        
        def update_distance(cutoff):
            self.v_cluster_setting['distance'].set(cutoff)
            recluster()
        
        z, cluster_args = recluster()
            
        try:
            labels = [d['Experiment name'] for d in self.all_cells.ds]
        except KeyError as err:
            print('Experiment names not found in CSV list. Consider including them.')
            labels = None
            
        distance_from_dendrogram(z, ylabel=cluster_args['metric'], initial_distance=cluster_args['distance'],
                                 labels=labels, fig_handle=self.cluster_widget.fig, callback=update_distance)
        
    def reload_cells(self):
        raw = self.v_use_raw.get()        
        if self.fn.endswith('.csv'):
            self.all_cells = CellList.from_csv(self.fn, use_raw_cell=raw) #TODO change this to selection of raw cells
        else:
            self.all_cells = CellList.from_yaml(self.fn, use_raw_cell=raw)
            
        self.w_all_fn.config(text=os.path.basename(self.fn) + (' (raw)' if raw else ''))
        
        self.init_clustering()
            
    def load_cells(self):
        self.fn = askopenfilename(title='Open cell list file', filetypes=(('CrysAlisPro', '*.csv'), ('YAML list (edtools)', '*.yaml')))
        self.reload_cells()
        
    def save_clusters(self, fn_template: Optional[str] = None):
        if fn_template is None:
            fn_template = asksaveasfilename(confirmoverwrite=False, title='Select root filename for cluster CSVs', 
                                            initialdir=os.path.dirname(self.fn), initialfile=os.path.basename(self.fn),
                                            filetypes=[('CrysAlisPro CSV', '*.csv')])
            
            if not fn_template:
                print('No filename selected, canceling.')
                return
            
        for ii, (c_id, cluster) in enumerate(self.clusters.items()):
            if c_id not in self.cluster_table.selected_clusters:
                print(f'Skipping Cluster {c_id} (not selected in list)')
                continue
            cluster_fn = os.path.splitext(fn_template)[0] + f'-cluster_{ii}_ID{c_id}.csv'
            cluster.to_csv(cluster_fn)
            print(f'Wrote cluster {c_id} with {len(cluster)} crystals to file {cluster_fn}')
            
    def save_merging_macro(self):
        
        mac_fn = asksaveasfilename(confirmoverwrite=True, title='Select MAC filename',
                                   initialdir=os.path.dirname(self.fn), initialfile=os.path.splitext(os.path.basename(self.fn))[0] + '.mac',
                                   filetypes=[('CrysAlisPro Macro', '*.mac')])
        
        info_fn = os.path.splitext(mac_fn)[0] + '_merge_info.csv'
        
        with open(mac_fn, 'w') as fh, open(info_fn, 'w') as ifh:
            ifh.write('File path,Data sets,Cluster,Merge code\n')
            for ii, (c_id, cluster) in enumerate(self.clusters.items()):
                if c_id not in self.cluster_table.selected_clusters:
                    print(f'Skipping Cluster {c_id} (not selected in list)')
                    continue
                out_paths, in_paths, out_codes, out_info = cluster.get_merging_paths(prefix=f'C{c_id}', short_form=True)
                for out, (in1, in2), code, info in zip(out_paths, in_paths, out_codes, out_info):
                    fh.write(f'xx proffitmerge "{out}" "{in1}" "{in2}"\n')
                    print(f'Writing merge file for: {info}')
                    ifh.write(f'{out},{c_id},{info},{code}\n')
                print(f'Full-cluster merge for cluster {c_id}: {out_paths[-1]}')
            
            
def parse_args():
    import argparse

    description = "Program for clustering unit cells from crystallography experiments. Contains clustering algorithm and display "
    "functions from edtools by Stef Smeets (https://github.com/instamatic-dev/edtools), adding a GUI and cluster import/export "
    "for CrysAlisPro."
    
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        
    parser.add_argument("filename",
                        type=str, metavar="FILE", nargs='?',
                        help="Path to .yaml (edtools) or .csv (CrysAlisPro) file")

    parser.add_argument("-b","--binsize",
                        action="store", type=float, dest="binsize",
                        help="Binsize for the histogram, default=0.5")

    parser.add_argument("-c","--cluster",
                        action="store_true", dest="cluster",
                        help="Apply cluster analysis instead of interactive cell finding")

    parser.add_argument("-d","--distance",
                        action="store", type=float, dest="distance",
                        help="Cutoff distance to use for clustering, bypass dendrogram")

    parser.add_argument("-m","--method",
                        action="store", type=str, dest="method",
                        choices="single average complete median weighted centroid ward".split(),
                        help="Linkage algorithm to use (see `scipy.cluster.hierarchy.linkage`)")

    parser.add_argument("-t","--metric",
                        action="store", type=str, dest="metric",
                        choices="euclidean lcv volume".split(),
                        help="Metric for calculating the distance between items (Euclidian distance, cell volume, LCV as in CCP4-BLEND)")

    parser.add_argument("-l", "--use_bravais_lattice",
                        action="store_false", dest="use_raw_cell",
                        help="Use the bravais lattice (symmetry applied)")

    parser.add_argument("-p", "--preprocessing",
                        action="store", type=str, dest="preproc",
                        choices='none standardize pca radians sine'.split(),
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
