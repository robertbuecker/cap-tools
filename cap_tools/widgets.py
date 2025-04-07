from time import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


import tkinter as tk
import tkinter.ttk as ttk
from typing import *

import numpy as np
import pandas as pd

from cap_tools.cell_list import CellList
from cap_tools.finalization import FinalizationCollection
from cap_tools.interact_figures import fom_radar_plot
from cap_tools.utils import myTreeView
import math
import os


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

    # def init_figure_controls(self):
    #     self.controls = ttk.Frame(self)


class ClusterWidget(PlotWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.tree: Optional[Dict[str, Any]] = None

    # def init_figure_controls(self):
    #     super().init_figure_controls()
    #     ttk.Label(self.controls, text='Nothing here').grid(row=0, column=1)
    #     ttk.Button(self.controls, text='Don\'t click!', command=lambda *args: print('nothing')).grid(row=0, column=2)

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
                if np.isfinite(avg):
                    digits = max(0,
                             -int(math.floor(math.log10(std)))+1 if std != 0 else 0,
                             -int(math.floor(math.log10(hi-lo))) if std != 0 else 0)
                else:
                    digits = 0
                cpar_strs.append('{0:.{4}f} ({1:.{4}f}) [{2:.{4}f}, {3:.{4}f}]'.format(avg, std, lo, hi, digits))

            self._entry_ids.append(self.cluster_view.insert('', tk.END, values=[c_id, len(cl)] + cpar_strs, tags=(str(c_id),)))
            
            # self.apply_cluster_colors()
            
    def apply_cluster_colors(self, cluster_colors: Dict[int, str] = None):
        
        for c_id in self._clusters.keys():
            self.cluster_view.tag_configure(str(c_id), foreground=cluster_colors.get(int(c_id), 'white'))
        

    def show_entry_info(self, event):
        for cluster_id in self.selected_cluster_ids:
            print(f'--- CLUSTER {cluster_id} ---')
            print(self._clusters[cluster_id].table)

    @property
    def selected_cluster_ids(self) -> List[int]:
        return [self.cluster_view.item(selected)['values'][0]
                for selected in self.cluster_view.selection()]
        
    @property
    def selected_clusters(self) -> Dict[int, CellList]:
        return {cid: cl for cid, cl in self._clusters.items() if cid in self.selected_cluster_ids}


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


class FinalizationWidget(ttk.Frame):

    def __init__(self, root: tk.BaseWidget, cluster_widget: Optional[ClusterWidget] = None):
        super().__init__(root)
        self.fc = FinalizationCollection()
        self.overall_text = tk.Text(self)
        # self.overall_text.grid(row=0, column=0, sticky='NSEW')
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.cluster_widget = cluster_widget # to be used for highlighting the selected cluster in the cluster widget

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
        self.fin_info = tk.Text(tbl_frame, height=4, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL, command=self.fin_info.yview)
        scrollbar.grid(row=5, column=1, sticky=tk.NS)
        self.fin_info.configure(yscrollcommand=scrollbar.set)        
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
        
    def clear(self):
        self.update_fc(fc = FinalizationCollection())
        self.update_radar_plot()
        self.update_shell_plot()

    def update_fc(self, fc: FinalizationCollection):
        self.fc = fc
        self.overall_text.delete(1.0, tk.END)
        self.overall_text.insert(tk.END,
                                self.fc.overall_highest.to_string(index=False))

        for the_id in self._fv_entry_ids:
            self.fin_view.delete(the_id)

        tdata = self.fc.overall_highest.merge(self.fc.meta, on='name').copy()
        cols = list(self.ft_columns.keys())
        for c in cols:
            if c not in tdata:
                tdata[c] = '?'

        self._fv_entry_ids = []
        for _, fin_data in tdata.iterrows():
            self._fv_entry_ids.append(self.fin_view.insert('', tk.END, values=list(fin_data[cols])))

    def show_fin_info(self, event):

        for the_id in self.shell_view.get_children():
            self.shell_view.delete(the_id)
            
        if (not len(self.fc)) or (not len(self.selected_fin_ids)):
            self.fin_info.delete(1.0, tk.END)
            return
        
        fin = self.selected_fc[self.selected_fin_ids[0]]        

        info_str = f'Selected finalization: {fin.name}, merged from {fin.meta["Nexp"] if "Nexp"in fin.meta else "?"} experiments:\n'
        info_str +=(f'{fin.meta["Merge code"]}\n'.replace(':', ', ') if 'Merge code' in fin.meta else '')
        info_str +=f'Path: {os.path.dirname(fin.path)}\n'
        info_str +=f'proffit file {"found" if fin.have_proffit else "not found"}; '
        info_str +=f'parameter XML file {"found" if fin.have_pars_xml else "not found"}\n'
        self.fin_info.delete(1.0, tk.END)
        self.fin_info.insert(tk.END, info_str)

        tdata = fin.shells.copy()

        cols = list(self.sv_columns)
        for col in cols:
            if col not in tdata:
                tdata[col] = '?'                
                
        tdata = tdata[cols]

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
                    if fom in fin.shells:
                        fin.shells.plot(x='1/d', y=fom, color=f'C{c}', label=name, ax=ax)
                    else:
                        print(f'Figure of merit {fom} not specified for {name}')

        self.fom_plot.canvas.draw()

    def update_radar_plot(self):

        def mangle_plot_data(plot_data: pd.DataFrame) -> pd.DataFrame:
            
            if 'Cluster' not in plot_data:
                plot_data['Cluster'] = -1

            cluster_plots = []
            for _, cl_data in plot_data.groupby('Cluster'):
                cl_data = cl_data.copy()
                default = cl_data.iloc[:,0].copy()
                default[:] = np.nan
                cl_data['Comp'] = cl_data.get('complete', default=default)/100
                cl_data['I/sig (rel)'] = cl_data.get('F2/sig(F2)', default=default)/cl_data.get('F2/sig(F2)', default=default).max()
                cl_data['Red. (rel)'] = cl_data.get('redundancy', default=default)/cl_data.get('redundancy', default=default).max()
                cl_data['1/Rurim (rel)'] = cl_data.get('Rurim', default=default).min()/cl_data.get('Rurim', default=default)
                cl_data['1/Rpim (rel)'] = cl_data.get('Rpim', default=default).min()/cl_data.get('Rpim', default=default)
                cluster_plots.append(cl_data)

            return pd.concat(cluster_plots, axis=0) if cluster_plots else pd.DataFrame(columns=['name'])

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

