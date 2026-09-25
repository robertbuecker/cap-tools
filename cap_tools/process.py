import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import patches
from cap_auto.cap_control import CAPInstance
from cap_auto.cap_files import get_diff_info as _cap_auto_get_diff_info
import numpy as np
import pandas as pd
import os
from typing import *


def get_diff_info(path, cap: Optional[CAPInstance] = None,
                  keep_peak_file: bool = False, keep_powder_file: bool = False,
                  redo_peak_hunt: bool = True, wavelength: float = 0.0251, pow_dmin: float = 0.3, pow_dmax: float = 20,
                  recenter_pattern: bool = True,
                  log: Optional[Callable] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Return diffraction analysis as pandas frames using cap-auto parsers.

    `cap_auto.cap_files.get_diff_info` is the canonical parser/generator. This
    wrapper preserves cap-tools' historical DataFrame-shaped return values for
    the screening viewer and report helpers.
    """
    if log is None:
        log = print

    if cap is not None:
        cap.load_experiment(path + '.par')

    shell_stats, reflections, powder_data, diff_img_fn = _cap_auto_get_diff_info(
        path,
        cap=cap,
        keep_peak_file=keep_peak_file,
        keep_powder_file=keep_powder_file,
        redo_peak_hunt=redo_peak_hunt,
        wavelength=wavelength,
        pow_dmin=pow_dmin,
        pow_dmax=pow_dmax,
        recenter_pattern=recenter_pattern,
        log=log,
    )

    shelldata = pd.DataFrame(shell_stats)

    peak_table = pd.DataFrame(reflections)
    if 'inv_d' in peak_table:
        peak_table['1/d'] = peak_table['inv_d']

    powder = pd.DataFrame(powder_data)
    if 'd_value' in powder:
        powder['d-value'] = powder['d_value']
    if 'intensity' in powder:
        powder['intx'] = powder['intensity']
    if 'inv_d' in powder:
        powder['1/d'] = powder['inv_d']

    return shelldata, peak_table, powder, diff_img_fn


def create_report_figure(exp_info: dict, shelldata: pd.DataFrame, peak_table: pd.DataFrame, 
                         fig: Optional[plt.Figure] = None, use_png: Optional[bool] = True) -> plt.Figure: # type: ignore

    # --- Style inside function using context manager to avoid global changes ---
    # Or, rely on global settings if plt.style.use and rcParams are set outside.
    with plt.style.context('seaborn-v0_8-whitegrid'): # Using context for local style
        plt.rcParams.update({
            'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 12,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
            'figure.titlesize': 16
        })

        # TODO consider making this optional
        # shelldata = exp_info['shelldata']
        # peak_table = exp_info['peak_table']
        shells = pd.concat([shelldata.d_min, shelldata.d_max]).unique()

        try:
            info_table = [['Name', exp_info['name']],
                ['Stage position', f'{exp_info["stage_x"]:.2f} {exp_info["stage_y"]:.2f} {exp_info["stage_z"]:.2f}'],
                ['Peak count', len(peak_table)],
                ['Total peak intensity',
                f'{peak_table[peak_table["I"] < peak_table.I.mean() + 5 * peak_table.I.std()]['I'].sum()}'],
                ['Shells [Å]', ' - '.join([f'{1/s:.2f}' for s in shells])],
                ['Dark field intensity', ' | '.join([f'{f:.0f}' for f in list(shelldata['I_tot'])])],
                ['Shell peaks', ' | '.join([f'{f:.0f}' for f in list(shelldata['N_peaks'])])],
                ['Shell peak int', ' | '.join([f'{f:.0f}' for f in list(shelldata['I_peak'])])],
                ['Shell peak ratio', ' | '.join([f'{f:.3f}' for f in list(shelldata['peak_ratio'])])]
                ]
        except KeyError as e:
            print(f"KeyError in creating info table for {exp_info['name']}: {e}")
            info_table = [['Name', exp_info['name']],
                          ['Stage position', f'{exp_info["stage_x"]:.2f} {exp_info["stage_y"]:.2f} {exp_info["stage_z"]:.2f}'],
                          ['Peak count', 'N/A'],
                          ['Total peak intensity', 'N/A'],
                          ['Shells [Å]', 'N/A'],
                          ['Dark field intensity', 'N/A'],
                          ['Shell peaks', 'N/A'],
                          ['Shell peak int', 'N/A'],
                          ['Shell peak ratio', 'N/A']]

        diff_jpg = plt.imread(exp_info['diff-png'] if (('diff-png' in exp_info) and use_png) else exp_info['diff-jpg'])
        grain_jpg = plt.imread(exp_info['grain-jpg'])

        scale = diff_jpg.shape[1] / 775

        grain_coords = np.array([exp_info['grain_x_px'], 385-exp_info['grain_y_px']]) * scale
        grain_aperture=scale*exp_info['aperture_px']

        if fig is None:
            # Create a new figure if not provided
            fig = plt.figure(figsize=(12, 9))
        else:
            fig.clf()        

        # Main grid: 2 rows, 2 columns.
        gs_main = gridspec.GridSpec(2, 2, figure=fig,
                                    height_ratios=[1, 2],
                                    width_ratios=[1, 3],
                                    hspace=0.005,
                                    wspace=0.005)

        ax_sq = fig.add_subplot(gs_main[0, 0])
        tables_container_spec = gs_main[0, 1]
        ax_rect = fig.add_subplot(gs_main[1, :])

        gs_tables_container = gridspec.GridSpecFromSubplotSpec(
                                                2, 1, subplot_spec=tables_container_spec,
                                                height_ratios=[0.1, 0.9],
                                                hspace=0.005)

        ax_tables_title_area = fig.add_subplot(gs_tables_container[0, 0])
        ax_tables_title_area.set_title('Screening Report for ' + exp_info['name'], fontsize=14, loc='center', y=0.4)
        ax_tables_title_area.axis('off')

        gs_two_tables = gridspec.GridSpecFromSubplotSpec(
                                                1, 1, subplot_spec=gs_tables_container[1, 0],
                                                wspace=0.005)
        ax_table1 = fig.add_subplot(gs_two_tables[0, 0])
        # ax_table2 = fig.add_subplot(gs_two_tables[0, 1])

        # --- Data Splitting for Tables ---
        num_rows_total = len(info_table)
        split_point = num_rows_total // 1
        table1_cell_text = info_table[0:split_point]
        table2_cell_text = info_table[split_point:num_rows_total]

        # --- Plot Diffraction Pattern ---
        ax_rect.imshow(diff_jpg, aspect='equal')
        # ax_rect.set_title('Diffraction Pattern')
        ax_rect.axis('off')

        # --- Plot Grain Image ---
        cmap_sq = 'viridis' if grain_jpg.ndim == 2 else None
        ax_sq.imshow(grain_jpg, cmap=cmap_sq, aspect='equal')
        ax_sq.add_artist(patches.Circle(grain_coords, grain_aperture/2,
                                        color='red', fill=False, lw=2))
        ax_sq.set_xlim(grain_coords[0] - grain_aperture*1.1 / 2, grain_coords[0] + grain_aperture*1.1 / 2)
        ax_sq.set_ylim(grain_coords[1] + grain_aperture*1.1 / 2, grain_coords[1] - grain_aperture*1.1 / 2)

        ax_sq.axis('off')

        # --- Nested Helper Function for Table Plotting ---
        def _plot_formatted_table(ax, cell_text, column_labels):
            ax.axis('off')
            if not cell_text: # Handle empty table case
                return

            the_table = ax.table(cellText=cell_text,
                                   colLabels=column_labels,
                                   loc='center',
                                   cellLoc='left',
                                   colWidths=[0.4, 0.5] if len(column_labels) == 2 else None) # Adjust if more columns
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(9)
            the_table.scale(1.0, 1.0)
            for (i, j), cell in the_table.get_celld().items():
                if i == 0: # Header
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4C72B0')
                elif i > 0 and i % 2 != 0: # Odd data rows for striping (1-based index for data rows)
                     cell.set_facecolor('#f2f2f2')

                # Right-align if it's the last column and a data row (often numeric)
                if j == len(column_labels) - 1 and i > 0:
                    cell.set_text_props(ha='right')
                elif i > 0: # Other data cells
                    cell.set_text_props(ha='left')

        # --- Plot the Two Tables ---
        _plot_formatted_table(ax_table1, table1_cell_text, ['Parameter', 'Value'])

    return fig

def create_report_figure_no_table(exp_info: dict, shelldata: Optional[pd.DataFrame] = None, 
                                  peak_table: Optional[pd.DataFrame] = None, use_png: bool = True,
                                  fig: Optional[plt.Figure] = None) -> plt.Figure: # type: ignore

    if fig is None:
        # Create a new figure if not provided
        fig = plt.figure(figsize=(12, 9))
    else:
        fig.clf()     

    if 'diff-png' not in exp_info and 'diff-jpg' not in exp_info:
        return fig

    shells = pd.concat([shelldata.d_min, shelldata.d_max]).unique() if shelldata is not None else []

    diff_jpg = plt.imread(exp_info['diff-png'] if ('diff-png' in exp_info and use_png) else exp_info['diff-jpg'])
    grain_jpg = plt.imread(exp_info['grain-jpg'])

    grain_scale = grain_jpg.shape[1] / 775

    grain_coords = np.array([exp_info['grain_x_px'], 385-exp_info['grain_y_px']]) * grain_scale
    grain_aperture=grain_scale*exp_info['aperture_px']

    # Show diffraction pattern
    ax_diff = fig.add_subplot(111)
    ax_diff.imshow(diff_jpg, aspect='equal')
    ax_diff.axis('off')
    
    # Define the size and position for ax_sq (upper left corner of ax_rect)
    rect_pos = ax_diff.get_position()
    inset_edge = rect_pos.height * 0.3
    ax_grain = fig.add_axes((rect_pos.x0, rect_pos.y0 + rect_pos.height - inset_edge, 
                             inset_edge, inset_edge), aspect='equal', anchor='NW')

    # --- Plot Grain Image ---
    cmap_sq = 'viridis' if grain_jpg.ndim == 2 else None
    ax_grain.imshow(grain_jpg, cmap=cmap_sq)
    ax_grain.add_artist(patches.Circle(grain_coords, grain_aperture/2,
                                    color='red', fill=False, lw=2))
    ax_grain.set_xlim(grain_coords[0] - grain_aperture*1.1 / 2, grain_coords[0] + grain_aperture*1.1 / 2)
    ax_grain.set_ylim(grain_coords[1] + grain_aperture*1.1 / 2, grain_coords[1] - grain_aperture*1.1 / 2)

    ax_grain.axis('off')

    return fig

def create_overall_figure(shelldata: pd.DataFrame, fig: Optional[plt.Figure] = None) -> plt.Figure: # type: ignore
    
    if fig is None:
        fig = plt.figure(figsize=(10, 12))
    else:
        fig.clf()
    
    shell_plot_data = shelldata.pivot(index='experiment', columns=['s', 'd_range'], values=['I_tot', 'I_peak', 'peak_ratio', 'N_peaks'])
    shell_plot_data = shell_plot_data.T.reset_index(level='s', drop=True).T

    axs = []
    for ii in range(4):
        axs.append(fig.add_subplot(4, 1, ii + 1, sharex=axs[0] if ii > 0 else None))
        
    shell_plot_data['I_tot'].plot.bar(title='Total intensity', stacked=True, width=0.8, rot=90, ax=axs[0], legend=False)
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    shell_plot_data['I_peak'].plot.bar(title='Spot intensity', stacked=True, width=0.8, rot=90, ax=axs[1], legend=False)
    shell_plot_data['N_peaks'].plot.bar(title='Spot number', stacked=True, width=0.8, rot=90, ax=axs[2], legend=False)
    shell_plot_data['peak_ratio'].plot.bar(title='Spot ratio', stacked=False, width=0.8, rot=90, ax=axs[3], legend=False)  
