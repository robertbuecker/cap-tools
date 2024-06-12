from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from typing import *
from .utils import weighted_average
import pandas as pd
from radar_chart import radar_factory

click_cid_dendrogram = None

matplotlib.use('TkAgg')

def distance_from_dendrogram(z, ylabel: str="", initial_distance: float=None, 
                             labels: Optional[List[str]] = None, 
                             fig_handle: Optional[Figure] = None, callback: Optional[callable] = None) -> float:
    """Interactive dendrogram plot.
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

    tree = dendrogram(z, color_threshold=distance, ax=ax, labels=labels, leaf_rotation=90 if labels is not None else 0, above_threshold_color='k')

    # use 1-based indexing for display by incrementing label    
    # _, xlabels = plt.xticks()
    # for l in xlabels:
    #     l.set_text(str(int(l.get_text())+1) if labels is None else labels[int(l.get_text())])
            
    ax.set_xlabel("Index")
    ax.set_ylabel(f"Distance ({ylabel})")
    ax.set_title(f"Dendrogram (cutoff={distance:.2f})")
    hline = ax.axhline(y=distance, color='g')

    def get_cutoff(event):
        nonlocal hline
        nonlocal tree
        nonlocal distance

        if event:
            distance = round(event.ydata, 4)
            ax.set_title(f"Dendrogram (cutoff={distance:.2f})")
            hline.remove()
            hline = ax.axhline(y=distance, color='g')

            for c in ax.collections:
                c.remove()

            yl = ax.get_ylim()
            tree = dendrogram(z, color_threshold=distance, ax=ax, labels=labels, leaf_rotation=90 if labels is not None else 0, above_threshold_color='k')
            ax.set_ylim(yl)
            
            if callback is not None:
                callback(distance)

            fig.canvas.draw()

    click_cid = fig.canvas.mpl_connect('button_press_event', get_cutoff)
    
    if fig_handle is None:
        plt.show()
    else:
        fig.canvas.draw()

    return distance, click_cid

def radar_plot(overall_df: pd.DataFrame, 
               highest_df: pd.DataFrame, 
               fig_handle: Optional[plt.Figure] = None,
               foms: Optional[Union[List[str], Tuple[str]]] = None,
               fom_lbl: Optional[Union[List[str], Tuple[str]]] = None,
               colors: Optional[List[str]] = None):
    
    if foms is None:
        foms = ['Comp', 'I/sig (rel)', '1/Rurim (rel)', '1/Rpim (rel)', 'CC1/2', 'Red. (rel)']
        
    if fom_lbl is None:
        fom_lbl = foms
       
    data = []
    data.append(('Overall', overall_df[foms].to_numpy()))
    data.append(('Highest Shell', highest_df[foms].to_numpy()))

    N = len(foms)

    theta = radar_factory(N, frame='polygon')
    
    if fig_handle:
        fig = fig_handle
        fig.clear()
    else:
        plt.figure(figsize=(9, 5))

    axs = np.array([fig.add_subplot(1, 2, 1, projection='radar'),
           fig.add_subplot(1, 2, 2, projection='radar')])

    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.15)

    if colors is None:
        colors = [f'C{ii}' for ii in range(len(overall_df))]
        
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1], fontsize='xx-small')
        # ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
        #                 horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.05, label='_nolegend_')
        ax.set_varlabels(fom_lbl, fontsize='xx-small')
        ax.set_ylim(0, 1.05)

    # add legend relative to top-left plot
    labels = list(overall_df['name'])
    legend = axs[0].legend(labels, loc=(0.9, .95),
                                labelspacing=0.05, fontsize='xx-small')

    # fig.text(0.5, 0.965, f'Merging statistics for cluster {cluster}',
    #             horizontalalignment='center', color='black', weight='bold',
    #             size='large')

    if fig_handle is None:
        plt.show()
    else:
        fig.canvas.draw()