import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import patches
from cap_tools.cap_control import CAPInstance
import numpy as np
import pandas as pd
import os
from time import sleep
from typing import *


def get_diff_info(path, cap: Optional[CAPInstance] = None,
                  keep_peak_file: bool = False, keep_powder_file: bool = False,
                  wavelength: float = 0.0251, pow_dmin: float = 0.3, pow_dmax: float = 20,
                  log: Optional[Callable] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:

    if log is None:
        log = print

    if cap is None:
        cap = CAPInstance(par_file=path + '.par', cap_folder='C:\\Xcalibur\\CrysAlisPro171.45')
    else:
        cap.load_experiment(path + '.par')

    cmds = ['dc microedadjustcenter']

    # Powder data extraction
    powder_fn = os.path.join(os.path.dirname(path), 'radial.dat')
    if not os.path.exists(powder_fn) or not keep_powder_file:
        if os.path.exists(powder_fn): os.remove(powder_fn)
        cmds.append(f'powder radial 128 {wavelength/pow_dmax*180/np.pi} {wavelength/pow_dmin*180/np.pi} 0 360 radial')

    peak_fn = path + '.tab'
    if not os.path.exists(peak_fn) or not keep_peak_file:
        if os.path.exists(peak_fn): os.remove(peak_fn)
        cmds.append('ph snogui_pars 1000 20 1 0 2 2 10 10 1 0 0 0 0.0 1000.0 0 1 1 1')
        cmds.append('wd oldasciit ' + '\"' + path + '\"')

    diff_img_fn = path + '_diff_screen.png'
    cmds.append(f'wd pnggiftiff "{diff_img_fn}"')

    log(f"Running commands for {path}: \n{'\n'.join(cmds)}")
    cap.run_cmd(cmds, use_mac=True)

    try:
        ii = 0
        while not os.path.exists(powder_fn):
            sleep(0.1)
            ii += 1
            if ii > 20:
                raise FileNotFoundError(f"Peak hunt result file {powder_fn} not found after 10 seconds.")        
        powder = pd.read_csv(powder_fn, skiprows=1, sep='\\s+')
        powder['1/d'] = 1/powder['d-value']
        d_min, d_max = powder['d-value'].min(), powder['d-value'].max()
        shells = [1/d_max, 1/10, 1/1.2, 1/0.8, 1/(d_min*1.5), 1/d_min]
        powder['shell'] = np.digitize(powder['1/d'], shells, right=False) - 1

    except Exception as e:
        log(f"Error parsing powder data for {path}: {e}")
        raise e

    try:
        ii = 0
        while not os.path.exists(peak_fn):
            sleep(0.1)
            ii += 1
            if ii > 20:
                raise FileNotFoundError(f"Peak hunt result file {peak_fn} not found after 10 seconds.")

        pk_cols=['x', 'y', 'z', 'R', 'I', 'f', 's', 'm', 'st',
            'centroidx', 'centroidy', 'os', 'ts', 'ks', 'ps', 'op',
            'tp', 'calcstatus', 'runframenumber']
        peak_table = pd.read_csv(peak_fn, sep='\\s+', skiprows=1, header=None).iloc[:,:len(pk_cols)]
        peak_table.columns = pk_cols
        peak_table.dropna(inplace=True)
        peak_table.drop(['os', 'ts', 'f', 's', 'm', 'ks', 'ps', 'op', 'tp', 'st'], axis=1, inplace=True)

        peak_table['1/d'] = peak_table['R']/wavelength
        peak_table['d'] = 1/peak_table['1/d']
        peak_table['shell'] = np.digitize(peak_table['1/d'], shells, right=False) - 1

    except Exception as e:
        log(f"Error processing peak data for {path}: {e}")
        raise e

    # get per-shell data from powder data and peak table
    shelldata = []
    for ii, d_inv in enumerate(shells[:-1]):
        shelldata.append({
            'd_max': 1/d_inv,
            'd_min': 1/shells[ii+1],
            'I_tot': powder[powder['shell'] == ii]['intx'].sum(),
            'I_peak': peak_table[peak_table['shell'] == ii]['I'].sum(),
            'N_peaks': len(peak_table[peak_table['shell'] == ii]),
            'peak_ratio': (peak_table[peak_table['shell'] == ii]['I'].sum() /
                        powder[powder['shell'] == ii]['intx'].sum()
                        if powder[powder['shell'] == ii]['intx'].sum() > 0 else np.nan)
        })

    shelldata = pd.DataFrame(shelldata)

    return shelldata, peak_table, powder, diff_img_fn


def create_report_figure(exp_info: dict, shelldata: pd.DataFrame, peak_table: pd.DataFrame, fig: Optional[plt.Figure] = None) -> plt.Figure: # type: ignore

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


        diff_jpg = plt.imread(exp_info['diff-jpg'])
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

    # --- Style inside function using context manager to avoid global changes ---
    # Or, rely on global settings if plt.style.use and rcParams are set outside.
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
    ax_grain = fig.add_axes([rect_pos.x0, rect_pos.y0 + rect_pos.height - inset_edge, 
                             inset_edge, inset_edge], aspect='equal', anchor='NW')

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