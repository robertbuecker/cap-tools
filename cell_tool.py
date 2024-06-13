import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename
from tkinter.messagebox import showinfo
import math
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from cap_tools.cell_list import CellList
from cap_tools.interact_figures import distance_from_dendrogram, fom_radar_plot
from cap_tools.finalization import FinalizationCollection, Finalization
from cap_tools.cluster_finalize import cluster_finalize
import numpy as np
from collections import defaultdict
from typing import *
from time import time
import os
from concurrent.futures import ThreadPoolExecutor
from cap_tools.utils import myTreeView, ClusterPreset


cluster_presets = {'Direct': ClusterPreset(preproc='None', metric='Euclidean', method='Ward'),
                   'Whitened': ClusterPreset(preproc='PCA', metric='SEuclidean', method='Average'),
                   'LCV (relative)': ClusterPreset(preproc='None', metric='LCV', method='Ward'),
                   'Diagonals (Å)': ClusterPreset(preproc='Diagonals', metric='Euclidean', method='Ward'),
                   'Standardized': ClusterPreset(preproc='None', metric='SEuclidean', method='Average'),
                   'LCV (Å)': ClusterPreset(preproc='None', metric='aLCV', method='Ward')}

class OutputWidget(ttk.Frame):
    
    def __init__(self, parent: tk.BaseWidget):
        self.text_widget = tk.Text(self)
        self.text_widget['state'] = 'disabled'
        self.rowconfigure(0, weight=100)
        self.columnconfigure(0, weight=100) 
        self.text_widget().grid(row=0, column=0, sticky=tk.NSEW)

class PlotWidget(ttk.Frame):
    
    def __init__(self, parent: tk.BaseWidget, figsize: Tuple[float]=(5,4), 
                 hide_toolbar: bool = False, fig_row: int = 0):
        
        super().__init__(parent)
        
        self.fig = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()

        # pack_toolbar=False will make it easier to use a layout manager later on.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()

        # self.canvas.mpl_connect(
        #     "key_press_event", lambda event: print(f"you pressed {event.key}"))
        # self.canvas.mpl_connect("key_press_event", key_press_handler)

        # self.init_figure_controls()
        
        self.rowconfigure(fig_row, weight=100)
        self.columnconfigure(0, weight=100)
        
        self.canvas.get_tk_widget().grid(row=fig_row, column=0, sticky=tk.NSEW)
        
        if not hide_toolbar:
            self.toolbar.grid(row=fig_row+1, column=0, sticky=tk.S)
        
    def init_figure_controls(self):        
        self.controls = ttk.Frame(self)

class ClusterWidget(PlotWidget):
    
    def __init__(self, parent):
        super().__init__(parent)
        
    def init_figure_controls(self):
        super().init_figure_controls()
        ttk.Label(self.controls, text='Nothing here').grid(row=0, column=1)
        ttk.Button(self.controls, text='Don\'t click!', command=lambda *args: print('nothing')).grid(row=0, column=2)

class FOMWidget(PlotWidget):
    def __init__(self, parent, change_callback: callable, fom_list: Optional[List[str]] = None, **kwargs):
        super().__init__(parent, fig_row=20, **kwargs)
        self._v_fom = tk.StringVar(self.controls)
        self.fom_selector = ttk.OptionMenu(self.controls, self._v_fom, command=change_callback, *fom_list)        
        self.fom_selector.grid(row=10, column=0)
        self.controls.grid(row=10, column=0)
                
    def init_figure_controls(self):
        super().init_figure_controls()
        
    def set_fom_list(self, fom_list):
        self.fom_selector.set_menu(default='CC1/2' if 'CC1/2' in fom_list else fom_list[-1], *fom_list)
        
    @property
    def selected_fom(self):
        return self._v_fom.get()
        
class FOMWidget2(PlotWidget):
    def __init__(self, parent, change_callback: callable, fom_list: Optional[List[str]] = None, initial: Optional[Tuple[str, str]] = ('CC1/2', 'complete'), **kwargs):
        super().__init__(parent, fig_row=20, **kwargs)
        
        self.controls = ttk.Frame(self)        
        self._v_fom = (tk.StringVar(self.controls), tk.StringVar(self.controls))
        self.fom_selector = tuple([ttk.OptionMenu(self.controls, v, init, *fom_list, command=change_callback) for v, init in zip(self._v_fom, initial)])
        tk.Label(self.controls, text='Upper FOM: ').grid(row=10, column=0)
        self.fom_selector[0].grid(row=10, column=5)
        tk.Label(self.controls, text='Lower FOM: ').grid(row=10, column=10)
        self.fom_selector[1].grid(row=10, column=15)
        self.controls.grid(row=10, column=0)
                
    def set_fom_list(self, fom_list):
        for sel in self.fom_selector:
            sel.set_menu(default='CC1/2' if 'CC1/2' in fom_list else fom_list[-1], *fom_list)
        
    @property
    def selected_fom(self):
        return tuple(v.get() for v in self._v_fom)
        
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
        
class FinalizationWidget(ttk.Frame):
    
    def __init__(self, root: tk.BaseWidget):
        super().__init__(root)
        self.fc = FinalizationCollection()
        self.overall_text = tk.Text(self)
        # self.overall_text.grid(row=0, column=0, sticky='NSEW')
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # columns for overall view
        self.ft_columns = {'name': ('', 90), 
                           'Nexp': ('N', 24),
                           'complete': ('Comp', 80), 
                           'redundancy': ('Red', 80), 
                           'F2/sig(F2)': ('I/sig', 80), 
                           'Rurim': ('Rurim', 80), 
                           'Rpim': ('Rpim', 80), 
                           'CC1/2': ('CC1/2', 80)}

        # columns for per-shell view
        self.sv_columns = {'dmin': ('dmin', 56),
                           'dmax': ('dmax', 56)}
        self.sv_columns.update({k: (v[0], 70) for k, v in self.ft_columns.items() if k not in ['name', 'Nexp']})
        self.sv_columns.update({'deltaCC': ('dCC', 70)})
        
        tbl_frame = ttk.Frame(self, borderwidth=0, relief='flat')
        
        # OVERVIEW OF FINALIZATIONS
        fv = self.fin_view = ttk.Treeview(tbl_frame, columns=list(self.ft_columns.keys()), show='headings', height=6) 
        self._selected_fin = None
        self._fv_entry_ids = []
        self._sv_entry_ids = []
        for k, (lbl, w) in self.ft_columns.items():
            fv.heading(k, text=lbl)
            fv.column(k, width=w, stretch=False)           
        fv.bind('<<TreeviewSelect>>', self.show_fin_info)
        scrollbar = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL, command=fv.yview)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        fv.configure(yscroll=scrollbar.set)
        fv.grid(row=0, column=0, sticky=tk.NSEW)
        
        # INFO FOR SELECTED FINALIZATION
        self.fin_info = tk.Text(tbl_frame,height=4)
        self.fin_info.grid(row=5, column=0, columnspan=1, sticky='EW')
        
        # PER-SHELL FOM FOR SELECTED FINALIZATION        
        sv = self.shell_view = ttk.Treeview(tbl_frame, columns=list(self.sv_columns.keys()), show='headings', height='10')      
        for k, (lbl, w) in self.sv_columns.items():
            sv.heading(k, text=lbl)
            sv.column(k, width=w, stretch=False)
        
        scrollbar = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL, command=sv.yview)
        sv.grid(row=10, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=10, column=1, sticky=tk.NS)
        sv.configure(yscroll=scrollbar.set)

        # RADAR PLOT
        self.radar_plot = PlotWidget(tbl_frame, figsize=(4,2), hide_toolbar=True)
        self.radar_plot.grid(row=15, column=0, columnspan=2, sticky='NSEW')        
        
        tbl_frame.grid(row=0, column=0, sticky='NSEW')
        
        plot_frame = ttk.Frame(self, borderwidth=0, relief='flat')
        
        self.fom_plot = FOMWidget2(plot_frame, figsize=(4,3), change_callback=self.update_shell_plot, 
                                   fom_list=[k for k in self.sv_columns.keys() if k not in ['dmin', 'dmax']])
        self.fom_plot.grid(row=0, column=1, sticky='NSEW')
        plot_frame.columnconfigure(1, weight=100)
        plot_frame.rowconfigure(0, weight=100)

        plot_frame.grid(row=0, column=1, sticky='NSEW')
        self.rowconfigure(0, weight=100)
        
    def update_fc(self, fc: FinalizationCollection):
        self.fc = fc
        self.overall_text.delete(1.0, tk.END)
        self.overall_text.insert(tk.END,
                                self.fc.overall_highest.to_string(index=False))

        for the_id in self._fv_entry_ids:
            self.fin_view.delete(the_id)

        tdata = self.fc.overall_highest.merge(self.fc.meta, on='name')
        
        self._fv_entry_ids = []        
        for _, fin_data in tdata.iterrows():                                                          
            self._fv_entry_ids.append(self.fin_view.insert('', tk.END, values=list(fin_data[list(self.ft_columns.keys())])))            
            
    def show_fin_info(self, event):
        
        for the_id in self.shell_view.get_children():
            self.shell_view.delete(the_id)

        fin = self.selected_fc[self.selected_fin_ids[0]]

        info_str = f'Selected finalization: {fin.name}, merged from {fin.meta["Nexp"]} experiments:\n'
        info_str +=f'{fin.meta["Data sets"]}\n'.replace(':', ', ')
        info_str +=f'Path: {os.path.dirname(fin.path)}\n'
        info_str +=f'proffit file {"found" if fin.have_proffit else "not found"}; '
        info_str +=f'parameter XML file {"found" if fin.have_pars_xml else "not found"}\n'
        self.fin_info.delete(1.0, tk.END)
        self.fin_info.insert(tk.END, info_str)

        tdata = fin.shells[list(self.sv_columns)]
        
        self._sv_entry_ids = []        
        for _, shell_data in tdata.iterrows():                                                          
            self._sv_entry_ids.append(self.shell_view.insert('', tk.END, values=list(shell_data)))          
            
        self.update_radar_plot()
        self.update_shell_plot()
        
    def update_shell_plot(self, event=None):
        
        self.fom_plot.fig.clear()
        
        axs = self.fom_plot.fig.subplots(2, 1, sharex=True, squeeze=True)
        
        for c, (name, fin) in enumerate(self.fc.items()):
            if name not in self.selected_fin_ids:
                continue
            else:
                for ax, fom in zip(axs, self.fom_plot.selected_fom):
                    fin.shells.plot(x='1/d', y=fom, color=f'C{c}', label=name, ax=ax)
                
        self.fom_plot.canvas.draw()
        
    def update_radar_plot(self):
        
        def mangle_plot_data(plot_data: pd.DataFrame):

            cluster_plots = []
            for _, cl_data in plot_data.groupby('Cluster'):
                cl_data = cl_data.copy()
                cl_data['Comp'] = cl_data['complete']/100
                cl_data['I/sig (rel)'] = cl_data['F2/sig(F2)']/cl_data['F2/sig(F2)'].max()
                cl_data['Red. (rel)'] = cl_data['redundancy']/cl_data['redundancy'].max()
                cl_data['1/Rurim (rel)'] = cl_data['Rurim'].min()/cl_data['Rurim']
                cl_data['1/Rpim (rel)'] = cl_data['Rpim'].min()/cl_data['Rpim']
                cluster_plots.append(cl_data)
                
            return pd.concat(cluster_plots, axis=0)
                
        overall_plot_data = mangle_plot_data(
            self.fc.overall.merge(self.fc.meta, on='name')).query('name in @self.selected_fin_ids')

        highest_plot_data = mangle_plot_data(
            self.fc.highest_shell.merge(self.fc.meta, on='name')
        ).query('name in @self.selected_fin_ids')
        
        colors = [f'C{ii}' for ii, k in zip(range(len(self.fc)),self.fc.keys()) if k in self.selected_fin_ids]
        
        fom_radar_plot(overall_plot_data, highest_plot_data,
                   fig_handle=self.radar_plot.fig,
                   foms = ['Comp', 'I/sig (rel)', '1/Rurim (rel)', '1/Rpim (rel)', 'CC1/2', 'Red. (rel)'],
                   colors=colors)
            
    @property
    def selected_fc(self) -> FinalizationCollection:
        return self.fc.get_subset(self.selected_fin_ids) #dummy
            
    @property
    def selected_fin_ids(self) -> List[str]:
        return [self.fin_view.item(selected)['values'][0] 
                for selected in self.fin_view.selection()]

class ClusterTableWidget(ttk.Frame):
    
    def __init__(self, root: tk.BaseWidget, clusters: Dict[int, CellList]):
        super().__init__(root)
        
        ct_columns = ['ID', 'obs', 'a', 'b', 'c', 'al', 'be', 'ga', 'V']

        cv = self.cluster_view = myTreeView(self, columns=ct_columns, show='headings', height=6)
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
        for cluster_id in self.selected_cluster_ids:
            print(f'--- CLUSTER {cluster_id} ---')            
            print(self._clusters[cluster_id].table)
            
    @property
    def selected_cluster_ids(self) -> List[int]:
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
        self.exec = ThreadPoolExecutor()
        
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
            'Top nodes only': ttk.Checkbutton(mff, text='Top nodes only', variable=self.v_merge_fin_setting['top_only'])
        }
        for k in ['Resolution', 'Top nodes only']:
            self.w_merge_fin_setting[k].config(w=15)
        for ii, (k, w) in enumerate(self.w_merge_fin_setting.items()):
            if not (isinstance(w, ttk.Button) or isinstance(w, ttk.Checkbutton)):
                ttk.Label(mff, text=k).grid(row=ii, column=0)
                w.grid(row=ii, column=1)
            else:
                w.grid(row=ii, column=0, columnspan=2)
                
        ttk.Button(mff, text='Merge', command=lambda *args: self.merge_finalize(finalize=False)).grid(row=5, column=0, columnspan=2)
        ttk.Button(mff, text='Finalize', command=lambda *args: self.merge_finalize(finalize=True)).grid(row=10, column=0, columnspan=2)
        ttk.Button(mff, text='Restore', 
                   command=lambda *args: self.mergefin_widget.update_fc(
                       FinalizationCollection.from_csv(os.path.splitext(self.fn)[0] + '_merge_info.csv')
                       )).grid(row=15, column=0, columnspan=2)
        self.merge_fin_status = tk.Text(mff, height=5, width=2)
        self.merge_fin_status.grid(row=20, column=0, columnspan=2, sticky='EW')
        mff.grid_columnconfigure(0, weight=1)
        mff.grid(row=30, column=0)
        
        
        # quit button
        button_quit = ttk.Button(cf, text="Quit", command=self.root.destroy)
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
        
    def run_clustering(self, distance: Optional[float] = None):        
        
        if self._clustering_disabled:
            raise RuntimeError('Reclustering is disabled. How did you get here?')
            print('Reclustering is disabled')
            return

        cluster_pars = ClusterPreset(preproc=self.v_cluster_setting['preproc'].get(),
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
        self.cluster_table.update_table(clusters=self.clusters)           
        if self.active_tab == 1:
            self.cellhist_widget.update_histograms(clusters=self.clusters)
            
        if redraw:            
            try:
                labels = [d['Experiment name'] for d in self.all_cells.ds]
            except KeyError as err:
                print('Experiment names not found in CSV list. Consider including them.')
                labels = None          
                        
            distance_from_dendrogram(self.all_cells._z, ylabel=cluster_pars.metric, initial_distance=distance,
                                labels=labels, fig_handle=self.cluster_widget.fig, callback=lambda distance: self.run_clustering(distance))                        

    def reload_cells(self):
        raw = self.v_use_raw.get()        
        self.all_cells = CellList.from_csv(self.fn, use_raw_cell=raw) #TODO change this to selection of raw cells           
        self.w_all_fn.config(text=os.path.basename(self.fn) + (' (raw)' if raw else ''))
        
        self.run_clustering()
            
    def load_cells(self):
        self.fn = os.path.normpath(
            askopenfilename(title='Open cell list file', filetypes=(('CrysAlisPro', '*.csv'), ('YAML list (edtools)', '*.yaml')))
        )
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
            if c_id not in self.cluster_table.selected_cluster_ids:
                print(f'Skipping Cluster {c_id} (not selected in list)')
                continue
            cluster_fn = os.path.splitext(fn_template)[0] + f'-cluster_{ii}_ID{c_id}.csv'
            cluster.to_csv(cluster_fn)
            print(f'Wrote cluster {c_id} with {len(cluster)} crystals to file {cluster_fn}')
            
    def save_merging_macro(self, mac_fn: Optional[str] = None):
        
        if mac_fn is None:
            mac_fn = asksaveasfilename(confirmoverwrite=True, title='Select MAC filename',
                                   initialdir=os.path.dirname(self.fn), initialfile=os.path.splitext(os.path.basename(self.fn))[0] + '_merge.mac',
                                   filetypes=[('CrysAlisPro Macro', '*.mac')])
        
        info_fn = os.path.splitext(mac_fn)[0] + '_info.csv'
        
        if not self.cluster_table.selected_cluster_ids:
            return []
        
        with open(mac_fn, 'w') as fh, open(info_fn, 'w') as ifh:
            ifh.write('Name,File path,Cluster,Data sets,Merge code\n')
            merged_cids = []
            for ii, (c_id, cluster) in enumerate(self.clusters.items()):
                if c_id not in self.cluster_table.selected_cluster_ids:
                    print(f'Skipping Cluster {c_id} (not selected in list)')
                    continue
                out_paths, in_paths, out_codes, out_info = cluster.get_merging_paths(prefix=f'C{c_id}', short_form=True)
                for out, (in1, in2), code, info in zip(out_paths, in_paths, out_codes, out_info):
                    fh.write(f'xx proffitmerge "{out}" "{in1}" "{in2}"\n')
                    print(f'Adding merge instructions for: {info}')
                    ifh.write(f'{os.path.basename(out)},{out},{c_id},{info},{code}\n')
                print(f'Full-cluster merge for cluster {c_id}: {out_paths[-1]}')
                merged_cids.append(c_id)
                
        return merged_cids
                
    def set_cluster_active(self, active: bool=True):
        
        global click_cid_dendrogram
        
        self._clustering_disabled = not active
        
        for child in self._csf.winfo_children():
            child.config(state='normal' if active else 'disabled')
        
        if active:
            self.cluster_table.cluster_view.enable()
        else:
            self.cluster_table.cluster_view.disable()
                        
        if (not active) and (click_cid_dendrogram is not None):
            self.cluster_widget.canvas.mpl_disconnect(click_cid_dendrogram)
        else:
            self.run_clustering()
            #TODO properly re-activate the plot!
        
    def merge_finalize(self, finalize: bool = True, top_only: bool = False):

        # save a pure merging macro (not yet containing the finalizations)
        cids = self.save_merging_macro(mac_fn = os.path.splitext(self.fn)[0] + '_merge.mac')
               
        if not cids:
            showinfo('No cluster selected', 'Please first select one or more cluster(s).')
                
        if finalize:
            # finalization loop
            
            # disable all window controls in the clustering section
            self.set_cluster_active(False)
            for child in self._mff.winfo_children():
                child.config(state='normal')      
                       
            # TODO: factor cluster_finalize into cluster class                            
            mac_fn = cluster_finalize(cluster_name=os.path.splitext(self.fn)[0],
                                             include_proffitmerge=True,
                                             _write_mac_only=True,
                                             finalization_timeout=30)
            
            cmd = f'script {mac_fn}'
            self.merge_fin_status.delete(1.0, tk.END)
            self.merge_fin_status.insert(tk.END, 'CAP command copied to Clipboard.\nPlease paste into CMD window, run, and set options.')
            self.root.clipboard_clear()
            self.root.clipboard_append(cmd)
      
            fin_future = self.exec.submit(cluster_finalize, cluster_name=os.path.splitext(self.fn)[0],
                                            include_proffitmerge=True, 
                                            res_limit=self.v_merge_fin_setting['resolution'].get(),
                                            _skip_mac_write=True)            
                
            def check_fin_running():
                if fin_future.done():
                    self.fc = fin_future.result()
                    print('OVERALL RESULTS TABLE')
                    print('---------------------')
                    print(self.fc.overall_highest)      
                    self.merge_fin_status.delete(1.0, tk.END)
                    self.merge_fin_status.insert(tk.END, 'Finalization complete')              
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

    description = "Program for clustering unit cells from crystallography experiments. Contains clustering algorithm and display "
    "functions from edtools by Stef Smeets (https://github.com/instamatic-dev/edtools), adding a GUI and cluster import/export "
    "for CrysAlisPro."
    
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
