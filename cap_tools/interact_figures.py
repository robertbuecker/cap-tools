from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from typing import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .utils import node_id_from_link
from matplotlib.colors import to_hex

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

matplotlib.use('TkAgg')


click_cid_dendrogram = None

def radar_factory(num_vars, frame='circle'):
    """
    From: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
    
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def fom_radar_plot(overall_df: pd.DataFrame, 
               highest_df: pd.DataFrame, 
               fig_handle: Optional[plt.Figure] = None,
               foms: Optional[Union[List[str], Tuple[str]]] = None,
               fom_lbl: Optional[Union[List[str], Tuple[str]]] = None,
               colors: Optional[List[str]] = None):
    
    if fig_handle:
        fig = fig_handle
        fig.clear()
    else:
        fig = plt.figure(figsize=(9, 5))

    if not (len(overall_df) and len(highest_df)):
        if fig_handle is None:
            plt.show()
        else:
            fig.canvas.draw()
        return
       
    if foms is None:
        foms = ['Comp', 'I/sig (rel)', '1/Rurim (rel)', '1/Rpim (rel)', 'CC1/2', 'Red. (rel)']
        
    if fom_lbl is None:
        fom_lbl = foms
       
    # filter out missing FOMs
    fom_lbl, foms = zip(*[(fl, fom) for fl, fom in zip(fom_lbl, foms) 
                          if ((fom in overall_df.columns) and (fom in highest_df.columns))])
    fom_lbl, foms = list(fom_lbl), list(foms)
       
    data = []
    data.append(overall_df[foms].to_numpy())
    data.append(highest_df[foms].to_numpy())
    
    N = len(foms)

    theta = radar_factory(N, frame='polygon')

    axs = np.array([fig.add_subplot(1, 2, 1, projection='radar'),
           fig.add_subplot(1, 2, 2, projection='radar')])

    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.2)


    if colors is None:
        colors = [f'C{ii}' for ii in range(len(overall_df))]
        
    for ax, case_data in zip(axs.flat, data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1], fontsize='xx-small')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.05, label='_nolegend_')
        ax.set_varlabels(fom_lbl, fontsize='xx-small')
        ax.set_ylim(0, 1.05)

    labels = list(overall_df['name'])
    legend = axs[0].legend(labels, loc=(0.9, .95),
                                labelspacing=0.05, fontsize='xx-small')

    if fig_handle is None:
        plt.show()
    else:
        fig.canvas.draw()

def distance_from_dendrogram(z, ylabel: str="", initial_distance: float=None, 
                             labels: Optional[List[str]] = None, 
                             fig_handle: Optional[Figure] = None, callback: Optional[callable] = None) -> float:
    """Interactive dendrogram plot. Function mostly taken from edtools by Stef Smeets.
    Invoked when dendrogram is recomputed (e.g. if the method is changed), but _not_ when reclustering by changing the distance
    cutoff.
    
    Takes a linkage object `z` from scipy.cluster.hierarchy.linkage and displays a
    dendrogram. The cutoff distance can be picked interactively, and is returned
    ylabel: sets the label for the y-axis
    initial_distance: initial cutoff distsance to display
    """
    
    global click_cid_dendrogram
    
    if initial_distance == None:
        # corresponding with MATLAB behavior
        distance = round(0.7*max(z[:,2]), 4)
    else:
        distance = initial_distance

    fig = plt.figure() if fig_handle is None else fig_handle
    
    if click_cid_dendrogram is not None:
        fig.canvas.mpl_disconnect(click_cid_dendrogram)
    
    ax = fig.gca()
    ax.cla()
    fig.subplots_adjust(bottom=0.15, top=0.9)

    ax.set_prop_cycle(None)
    
    set_link_color_palette([f'C{ii}' for ii in range(10)])

    # tree = dendrogram(z, color_threshold=distance, ax=ax, labels=labels, leaf_rotation=90 if labels is not None else 0, above_threshold_color='k')
    tree = dendrogram(z, no_plot=True)
            
    # use 1-based indexing for display by incrementing label    
    # _, xlabels = plt.xticks()
    # for l in xlabels:
    #     l.set_text(str(int(l.get_text())+1) if labels is None else labels[int(l.get_text())])
            
    # ax.set_xlabel("Dataset")
    # ax.set_ylabel(f"Distance ({ylabel})")
    # ax.set_title(f"Dendrogram (cutoff={distance:.2f})")
    hline = ax.axhline(y=distance, color='g')
    ax.set_ylim(0, 1.05*max(z[:,2]))

    def get_cutoff(event = None):
        # called via callback when clicking into the dendrogram. Repaints the dendrogram with the new clusters,
        # then calls the actual clustering function `callback`, which would usually be CellGUI.run_clustering
        nonlocal hline
        nonlocal tree
        nonlocal distance

        if event:
            distance = round(event.ydata, 4) if event.ydata is not None else 0.0
            
        ax.set_title(f"Dendrogram (cutoff={distance:.2f})")

        for c in ax.collections:
            c.remove()
            
        for child in ax.get_children():
            if type(child) in [matplotlib.text.Annotation, matplotlib.lines.Line2D]:
                child.remove()

        # hline.remove()
        hline = ax.axhline(y=distance, color='g')
        
        yl = ax.get_ylim() # needs to be protected against 
        tree = dendrogram(z, color_threshold=distance, ax=ax, labels=labels, leaf_rotation=90 if labels is not None else 0, above_threshold_color='k')
        ax.set_ylim(yl)
        
        if callback is not None:
            # callback is usually CellGUI.run_clustering
            node_cids, color_func = callback(distance, tree)
            
            # get cluster IDs of top-level cluster nodes
            found = []
            top_node_cids = []                
            for idx in reversed(node_cids):
                if idx in found:
                    top_node_cids.append(-2)
                else:
                    found.append(idx)
                    top_node_cids.append(idx)
            top_node_cids = np.array(top_node_cids[::-1])
            link_cids = top_node_cids[node_id_from_link(z)]
            cluster_colors = {}
            
            # ...and annotate them
            if node_cids is not None:                                            
                for ii,(cid, i, d, c) in enumerate(zip(link_cids, tree['icoord'], tree['dcoord'], tree['color_list'])):
                    x = 0.5 * sum(i[1:3])
                    y = d[1]
                    if (y < distance) & (cid >= 0):
                        ax.plot(x, y, 'o', c=c)
                        ax.annotate(f'{cid}', (x, y), xytext=(0, 5.3),
                                        textcoords='offset points',
                                        va='bottom', ha='center', 
                                        bbox={'boxstyle': 'square,pad=0.1', 'fc': 'w', 'ec': c})    
                        cluster_colors[int(cid)] = to_hex(c)
                        
            color_func(cluster_colors)                    

        fig.canvas.draw()
        
    get_cutoff()

    click_cid_dendrogram = fig.canvas.mpl_connect('button_press_event', get_cutoff)
    
    if fig_handle is None:
        plt.show()
    else:
        fig.canvas.draw()

    return distance, click_cid_dendrogram
