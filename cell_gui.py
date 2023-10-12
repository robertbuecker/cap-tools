import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from utils import parse_cap_csv, put_in_order, volume
from find_cell import cluster_cell
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import yaml
from scipy.cluster.hierarchy import linkage
from interact_figures import distance_from_dendrogram, find_cell
from utils import get_clusters, parse_cap_csv, put_in_order, to_radian, to_sin, unit_cell_lcv_distance, write_cap_csv, volume_difference
from typing import *

      
class CellList:
    
    def __init__(self, cells: np.ndarray, ds: Optional[dict] = None, weights: Optional[np.ndarray] = None):        
        self._cells = put_in_order(cells)
        self._weights = np.array([1]*cells.shape[0]) if weights is None else weights
        if ds is None:
            self.ds = []
            for c in cells:
                self.ds.append({'unit cell': ' '.join(list(c))})
        else:
            self.ds = ds
            
    @property
    def cells(self):
        return self._cells
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def volumes(self):
        return np.array([volume(cell) for cell in self.cells])
        
    @classmethod
    def from_yaml(cls, fn, use_raw_cell=True):
        ds = yaml.load(open(fn, "r"), Loader=yaml.Loader)
        key = "raw_unit_cell" if use_raw_cell else "unit_cell"            
        # prune based on NaNs (missing cells)
        ds = [d for d in ds if not any(np.isnan(d[key]))]
        cells = np.array([d[key] for d in ds])
        weights = np.array([d["weight"] for d in ds])
    
    @classmethod
    def from_csv(cls, fn, use_raw_cell=True):
        ds, cells, weights = parse_cap_csv(fn, use_raw_cell, filter_missing=True)
        return cls(cells=cells, ds=ds)
    
    def cluster(self,
                 distance: float=None, 
                 method: str="average", 
                 metric: str="euclidean", 
                 use_radian: bool=False,
                 use_sine: bool=False,
                 labels: Optional[List[str]] = None) -> Dict[int,'CellList']:
                """Perform hierarchical cluster analysis on a list of cells. 

                method: lcv, volume, euclidean
                distance: cutoff distance, if it is not given, pop up a dendrogram to
                    interactively choose a cutoff distance
                use_radian: Use radian instead of degrees to downweight difference
                use_sine: Use sine for unit cell clustering (to disambiguousize the difference in angles)
                """

                from scipy.spatial.distance import pdist

                if use_sine:
                    _cells = to_sin(self.cells)
                elif use_radian:
                    _cells = to_radian(self.cells)
                else:
                    _cells = self.cells

                if metric.lower() == "lcv":
                    dist = pdist(_cells, metric=unit_cell_lcv_distance)
                    z = linkage(dist,  method=method)                    
                elif metric.lower() == "volume":
                    dist = pdist(_cells, metric=volume_difference)
                    z = linkage(dist,  method=method)
                    distance = 250.0 if distance is None else distance
                else:
                    z = linkage(_cells,  metric=metric, method=method.lower())
                    distance = 2.0 if distance is None else distance

                # if not distance:
                #     distance = distance_from_dendrogram(z, ylabel=metric, initial_distance=initial_distance, labels=labels)

                print(f"Linkage method = {method}")
                print(f"Cutoff distance = {distance}")
                print(f"Distance metric = {metric}")
                print("")

                clusters_idx = get_clusters(z, self.cells, distance=distance)
                
                clusters = {}
                for k, cluster_idx in clusters_idx.items():
                    clusters[k] = CellList(cells = self.cells[cluster_idx],
                                           ds=[d for ii, d in enumerate(self.ds) if ii in cluster_idx],
                                           weights=self.weights[cluster_idx])
                
                return clusters, z
                
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
        self.controls.grid(row=0, column=0, sticky=tk.N)
        
        self.rowconfigure(1, weight=100)
        self.columnconfigure(0, weight=100)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=tk.NSEW)
        
        self.toolbar.grid(row=2, column=0, sticky=tk.S)
        
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
        
    def init_figure_controls(self):
        super().init_figure_controls()
        ttk.Label(self.controls, text='Nothing here').grid(row=0, column=1)
        ttk.Button(self.controls, text='Don\'t click!', command=lambda *args: print('nothing')).grid(row=0, column=2)

class CellGUI:
    
    def __init__(self):
        
        # internal variables
        self.all_cells = CellList(cells=np.empty([0,6]))
        self.fn = None
        
        # initialize master GUI 
        self.root = tk.Tk()
        self.root.title("3D ED/MicroED cell tool")
        
        cf = self.control_frame = ttk.LabelFrame(self.root, text='Cell Lists')
        
        # file opening
        ttk.Button(cf, text='Open list...', command=self.load_cells).grid(row=0, column=0)
        self.v_use_raw = tk.BooleanVar(cf)
        self.w_use_raw = ttk.Checkbutton(cf, text='Use raw cells', command=self.reload_cells, variable=self.v_use_raw)
        self.w_use_raw.grid(row=5, column=0)
        self.w_all_fn = ttk.Label(cf, text='(nothing loaded)')
        self.w_all_fn.grid(row=10, column=0)

        # clustering (default) settings
        csf = ttk.LabelFrame(cf, text='Clustering')
        self.v_cluster_setting = {
            'distance': tk.DoubleVar(value=2.0),
            'method': tk.StringVar(value='average'),
            'metric': tk.StringVar(value='euclidian'),
            'use_radian': tk.BooleanVar(value=False),
            'use_sine': tk.BooleanVar(value=False)
        }        
        metric_list = 'Euclidean Volume LCV'.split()
        method_list = 'Single Average Complete Median Weighted Centroid Ward'.split()        
        self.w_cluster_setting = {
            'Distance': ttk.Entry(csf, textvariable=self.v_cluster_setting['distance']),
            'Method': ttk.OptionMenu(csf, self.v_cluster_setting['method'], 'Average', *method_list),
            'Metric': ttk.OptionMenu(csf, self.v_cluster_setting['metric'], 'Euclidean', *metric_list),
            'Radian': ttk.Checkbutton(csf, text='Radian', variable=self.v_cluster_setting['use_radian']),
            'Sine': ttk.Checkbutton(csf, text='Sine', variable=self.v_cluster_setting['use_sine']),
            'Refresh': ttk.Button(csf, text='Refresh', command=self.run_clustering)
        }        
        for ii, (k, w) in enumerate(self.w_cluster_setting.items()):
            if not (isinstance(w, ttk.Button) or isinstance(w, ttk.Checkbutton)):
                ttk.Label(csf, text=k).grid(row=ii, column=0)
                w.grid(row=ii, column=1)
            else:
                w.grid(row=ii, column=0, columnspan=2)
        csf.grid(row=15, column=0)
        
        # quit button
        button_quit = ttk.Button(cf, text="Quit", command=self.root.destroy)
        button_quit.grid(row=100, column=0, sticky=tk.S)
        
        
        tabs = ttk.Notebook(self.root)
        self.tab_cluster = ttk.Frame(tabs)
        self.tab_cluster.columnconfigure(0, weight=100)
        self.tab_cluster.rowconfigure(0, weight=100)
        self.cluster_widget = ClusterWidget(self.tab_cluster)
        self.cluster_widget.grid(row=0, column=0, sticky=tk.NSEW)
        tabs.add(self.tab_cluster, text='Clustering', sticky=tk.NSEW)
        self.tab_cellhist = ttk.Frame(tabs)
        self.tab_cellhist.columnconfigure(0, weight=100)
        self.tab_cellhist.rowconfigure(0, weight=100)
        self.cellhist_widget = CellHistogramWidget(self.tab_cellhist)
        self.cellhist_widget.grid(row=0, column=0, sticky=tk.NSEW)
        tabs.add(self.tab_cellhist, text='Cell Histogram', sticky=tk.NSEW)   
             
        # final assembly of UI
        self.root.columnconfigure(0, weight=100)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=100)
        cf.grid(row=0, column=1, sticky=tk.E)        
        tabs.grid(column=0, row=0, sticky=tk.NSEW)
        
        
    def run_clustering(self):
        cluster_args = {k: v.get() for k, v in self.v_cluster_setting.items()}
        cluster_args = {k: v.lower() if isinstance(v, str) else v for k, v in cluster_args.items()}
        clusters, z = self.all_cells.cluster(**cluster_args)
        try:
            labels = [d['Experiment name'] for d in self.all_cells.ds]
        except KeyError as err:
            print('Experiment names not found in CSV list. Consider including them.')
            labels = None
        distance_from_dendrogram(z, ylabel=cluster_args['metric'], initial_distance=cluster_args['distance'],
                                 labels=labels, fig_handle=self.cluster_widget.fig)
        
    def reload_cells(self):
        raw = self.v_use_raw.get()        
        if self.fn.endswith('.csv'):
            self.all_cells = CellList.from_csv(self.fn, use_raw_cell=raw) #TODO change this to selection of raw cells
        else:
            self.all_cells = CellList.from_yaml(self.fn, use_raw_cell=raw)
            
        self.w_all_fn.config(text=self.fn.rsplit('/',1)[-1] + (' (raw)' if raw else ''))
            
    def load_cells(self):
        self.fn = askopenfilename(title='Open cell list file', filetypes=(('Result Viewer Export', '*.csv'), ('YAML list', '*.yaml')))
        self.reload_cells()

def main():
    import argparse

    description = "Program for finding the unit cell from a serial crystallography experiment."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        
    parser.add_argument("args",
                        type=str, nargs="*", metavar="FILE",
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

    parser.add_argument("-r", "--use_radian_for_angles",
                        action="store_true", dest="use_radian_for_clustering",
                        help="Use radians for unit cell clustering (to downweight the difference in angles)")

    parser.add_argument("-s", "--use_sine_for_angles",
                        action="store_true", dest="use_sine_for_clustering",
                        help="Use sine for unit cell clustering (to disambiguousize the difference in angles)")
    
    parser.add_argument("-w","--raw-cell",
                       action="store_true", dest="use_raw_cell",
                       help="Use the raw lattice (from Lattice Explorer/IDXREF as opposed to the refined one from GRAL/CORRECT) for unit cell finding and clustering")

    parser.set_defaults(binsize=0.5,
                        cluster=True,
                        distance=None,
                        method="average",
                        metric="euclidean",
                        use_raw_cell=False,
                        raw=False,
                        use_radian_for_clustering=False,
                        use_sine_for_clustering=False)
    
    options = parser.parse_args()

    distance = options.distance
    binsize = options.binsize
    cluster = options.cluster
    method = options.method
    metric = options.metric
    use_raw_cell = options.use_raw_cell
    use_radian = options.use_radian_for_clustering
    use_sine = options.use_sine_for_clustering
    args = options.args

    if args:
        fn = args[0]
    else:
        fn = "cells.yaml"
        fn = 'result-viewer.csv'
        
    if fn.endswith('.yaml') or fn.endswith('.yml'):
        use_yaml = True
        ds = yaml.load(open(fn, "r"), Loader=yaml.Loader)
        key = "raw_unit_cell" if use_raw_cell else "unit_cell"            
        # prune based on NaNs (missing cells)
        ds = [d for d in ds if not any(np.isnan(d[key]))]
        cells = np.array([d[key] for d in ds])
        weights = np.array([d["weight"] for d in ds])
        
    elif fn.endswith('.csv'):
        use_yaml = False
        ds, cells, weights = parse_cap_csv(fn, use_raw_cell)
        
    else:
        raise ValueError('Input file must be .yaml (edtools/XDS) or .csv (CrysAlisPro)')
    
    cells = put_in_order(cells)    
    
    if cluster:
        if not use_yaml:
            try:
                labels = [d['Experiment name'] for d in ds]
            except KeyError as err:
                print('Experiment names not found in CSV list. Consider including them.')
                labels = None
        else:
            labels = None
            
        clusters = cluster_cell(cells, distance=distance, method=method, metric=metric, 
                                use_radian=use_radian, use_sine=use_sine, labels=labels, fig=fig)
        
        tk.mainloop()
        
        for i, idx in clusters.items():
            clustered_ds = [ds[i] for i in idx]
            if use_yaml:
                fout = f"cells_cluster_{i}_{len(idx)}-items.yaml"
                yaml.dump(clustered_ds, open(fout, "w"))
            else:
                fout = f"{fn.rsplit('.', 1)[0]}_cells_cluster_{i}_{len(idx)}-items.csv"
                write_cap_csv(fout, clustered_ds)                                
                      
            print(f"Wrote cluster {i} to file `{fout}`")
    
    else:
        constants, esds = find_cell(cells, weights, binsize=binsize)
        
        print()
        print("Weighted mean of histogram analysis")
        print("---")
        print("Unit cell parameters: ", end="")
        for c in constants:
            print(f"{c:8.3f}", end="")
        print()
        print("Unit cell esds:       ", end="")
        for e in esds:
            print(f"{e:8.3f}", end="")
        print()

        try:
            import uncertainties as u
        except ImportError:
            pass
        else:
            print()
            names = (("a"," Å"), ("b"," Å"), ("c"," Å"),
                     ("α", "°"), ("β", "°"), ("γ", "°"))
            for i, (c, e) in enumerate(zip(constants, esds)):
                name, unit = names[i]
                val = u.ufloat(c, e)
                end = ", " if i < 5 else "\n"
                print(f"{name}={val:.2uS}{unit}", end=end)

        print()
        print("UNIT_CELL_CONSTANTS= " + " ".join(f"{val:.3f}" for val in constants))


if __name__ == '__main__':
    window = CellGUI()
    window.root.mainloop()
    # main()
