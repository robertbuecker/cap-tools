import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename
from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from cap_tools.cap_control import CAPInstance, CAPListenModeError
from cap_tools.utils import get_version, parse_cap_csv, parse_cap_meta
from cap_tools.process import get_diff_info, create_report_figure_no_table, create_overall_figure

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MicroED Screening Viewer")
        self.geometry("1280x1024")

        # Example experiments DataFrame; replace with your real data
        cols = ['N_peaks', 'I_peak', 'I_tot', 'peak_ratio']
        self.experiments = pd.DataFrame(
            [{col: np.random.rand() for col in cols} for _ in range(5)],
            index=[f"dummy_{i}" for i in range(1,6)]
        )
        self.shelldata = pd.DataFrame(
            columns=['experiment', 'd_max', 'd_min', 'I_tot', 'I_peak', 'N_peaks', 'd_range', 's'])
        self.peak_table = pd.DataFrame(
            columns=['experiment', 'x', 'y', 'I', 'd', '1/d', 'shell'])

        # Define which columns appear in each table
        # Experiments table: subset of important columns
        self.exp_table_columns = ['N_peaks', 'I_peak', 'I_tot', 'I_ratio']
        # Details table: more columns (can be all or a subset)
        self.detail_columns = ['N_peaks', 'I_peak', 'I_tot', 'stage_pos',
                               'Shells [Å]', 'I_tot_shells', 
                               'N_pk_shells', 'I_pk_shells', 'I_ratio_shells', 
                               ]

        # Configure grid weights: allocate extra space only to the figure
        # Row 0 (figure) expands; toolbar, experiments, log remain fixed
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=0)
        # Column 0 (figure + experiments) expands; side panels fixed
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        # Build UI
        self._create_figure_frame()
        self._create_details_table()
        self._create_experiments_frame()
        self._create_control_frame()
        self._create_log_frame()

        # Populate experiments table
        self.update_experiments_table()
        # Optionally select the first experiment by default
        first = self.exp_table.get_children()
        if first:
            self.exp_table.selection_set(first[0])
            
        
        self.cap = CAPInstance(start_now=False, cap_folder='C:\\Xcalibur\\CrysAlisPro171.45',
                               par_file='C:\\Xcalibur\\CrysAlisPro171.45\\help\\ideal_microed\\MicroED.par')
        
        self.output_folder = ''

    # --- Figure + toolbar ---
    def _create_figure_frame(self):
        self.fig = Figure(figsize=(5,4), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        x = np.linspace(0, 10, 100)
        self.line1, = self.ax1.plot(x, np.sin(x), label="sin")
        self.line2, = self.ax2.plot(x, np.cos(x), label="cos")
        self.ax1.legend(); self.ax2.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_container = tk.Frame(self)
        toolbar_container.grid(row=1, column=0, sticky="w")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_container)
        toolbar.update()

    def update_figure(self, x, y1, y2):
        """Swap in new data on the two subplots without full redraw."""
        self.line1.set_data(x, y1)
        self.ax1.relim(); self.ax1.autoscale_view()
        self.line2.set_data(x, y2)
        self.ax2.relim(); self.ax2.autoscale_view()
        self.canvas.draw()

    # --- Details (Key/Value) table ---
    def _create_details_table(self):
        self.details_frame = ttk.LabelFrame(self, text="Details")
        self.details_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

        cols = ("Key", "Value")
        self.details_table = ttk.Treeview(
            self.details_frame, columns=cols, show="headings", height=8
        )
        for c in cols:
            self.details_table.heading(c, text=c)
            self.details_table.column(c, width=80 if c == 'Key' else 140, anchor="w")
        self.details_table.pack(fill="both", expand=True)

    def update_details(self, key):
        """Refresh the details table for a given experiment index."""
        for item in self.details_table.get_children():
            self.details_table.delete(item)
        for col in self.detail_columns:
            val = self.experiments.at[key, col] if col in self.experiments.columns else ''
            self.details_table.insert("", "end", values=(col, val))

    # --- Experiments table ---
    def _create_experiments_frame(self):
        self.exp_frame = ttk.LabelFrame(self, text="Experiments")
        self.exp_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        self.exp_table = ttk.Treeview(self.exp_frame, show="headings")
        vsb = ttk.Scrollbar(
            self.exp_frame, orient="vertical", command=self.exp_table.yview
        )
        self.exp_table.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.exp_table.pack(side="left", fill="both", expand=True)
        self.exp_table.bind("<<TreeviewSelect>>", self.on_experiment_select)

    def update_experiments_table(self):
        """Populate the experiments table with subset columns."""
        cols = ["Experiment"] + self.exp_table_columns
        self.exp_table["columns"] = cols
        for c in cols:
            self.exp_table.heading(c, text=c)
            width = 100 if c == "Experiment" else 80
            self.exp_table.column(c, width=width, anchor="center")
        for item in self.exp_table.get_children():
            self.exp_table.delete(item)
        for idx, row in self.experiments.iterrows():
            vals = [idx] + [(row[c] if c in row else '') for c in self.exp_table_columns]
            self.exp_table.insert("", "end", iid=idx, values=vals)

    def on_experiment_select(self, event):
        """Single handler: update details (and optionally figure)."""
        sel = self.exp_table.selection()
        if not sel:
            return
        key = sel[0]
        self.update_details(key)
        try:
            create_report_figure_no_table(dict(self.experiments.loc[key, :]), 
                                      self.shelldata.query('experiment == @key'), 
                                      self.peak_table.query('experiment == @key'), 
                                      fig=self.fig)
        except:
            self.log(f"Error creating report figure for {key}. Ensure data is available.")
            self.fig.clf()
        finally:
            self.canvas.draw()
                    
        self.log(f"Selected experiment: {key}")

    # --- Control subpanel ---
    def _create_control_frame(self):
        self.ctrl_frame = ttk.LabelFrame(self, text="Controls")
        self.ctrl_frame.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        # Example: a button you can rewire to your computations
        self.input_label = ttk.Label(self.ctrl_frame, text="No data loaded", width=30, wraplength=220, anchor="w")
        self.input_label.pack(pady=(0,5))
        
        self.load_folder_btn = ttk.Button(self.ctrl_frame, text="Load Experiment Folder", command=self.load_folder)
        self.load_folder_btn.pack()
        self.load_csv_btn = ttk.Button(self.ctrl_frame, text="Load Results CSV", command=self.load_csv)
        self.load_csv_btn.pack()
        self.pre_only_var = tk.BooleanVar(value=False)
        self.pre_only_btn = ttk.Checkbutton(self.ctrl_frame, text="Pre-experiments only", variable=self.pre_only_var)
        self.pre_only_btn.pack(pady=(0, 10))        
        
        self.output_folder_label = ttk.Label(self.ctrl_frame, text="No output folder set", width=30, wraplength=220, anchor="w")
        self.output_folder_label.pack()
        self.set_output_folder_btn = ttk.Button(self.ctrl_frame, text="Change Output Folder", command=self.set_output_folder)
        self.set_output_folder_btn.pack(pady=(0,10))
        self.process_btn = ttk.Button(self.ctrl_frame, text="Run Computation", command=self.process)
        self.process_btn.pack()
        self.overall_plot_btn = ttk.Button(self.ctrl_frame, text="Summary Plot", command=self.overall_plot)
        self.overall_plot_btn.pack()

    def _on_run(self):
        self.log("Run button clicked — hook this to your computation")

    # --- Logging panel ---
    def _create_log_frame(self):
        self.log_frame = ttk.LabelFrame(self, text="Log")
        self.log_frame.grid(
            row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5
        )
        # Reduced height to approx. 5 lines
        self.log_text = tk.Text(self.log_frame, height=5)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            self.log_frame, orient="vertical", command=self.log_text.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def log(self, msg: str):
        self.log_text.insert("end", f"{msg}\n")
        self.log_text.see("end")
        self.update_idletasks()  # Ensure the log updates immediately
            
    def set_output_folder(self, folder: Optional[str] = None):
        if folder is None:
            self.output_folder = os.path.normpath(askdirectory(title=f'Change output folder for screening results'))
        else:
            self.output_folder = folder
        self.output_folder_label.config(text=f"{self.output_folder}")
        self.log(f"Output folder set to: {self.output_folder}")
        
    def load_meta(self, fn):
        meta_res = parse_cap_meta([fn], include=['pre_'] if self.pre_only_var.get() else None, log_fun=self.log)
        self.input_label.config(text=f"{os.path.split(fn)[-1]}: {len(meta_res)} exps")
        self.set_output_folder(os.path.join(os.path.commonpath([exp_info['path'] for exp_info in meta_res]), 'screening_results'))
        self.experiments = pd.DataFrame(meta_res)
        self.experiments.set_index('name', inplace=True)
        self.update_experiments_table()
               
    def load_folder(self):
        self.load_meta(
            os.path.normpath(askdirectory(title=f'Add experiment folder structure (will be searched recursively!)'))
        )

    def load_csv(self):
        self.load_meta(
            fn = os.path.normpath(askopenfilename(title=f'Add Results Viewer CSV file', filetypes=[('Cell tool result', '*.csv')]))
        )
            
    def process(self):
        
        sd, pt, pw, info = [], [], [], []
        
        self.experiments['diff-png'] = ''      

        for ii, (name, exp_info) in enumerate(self.experiments.iterrows()):
                        
            self.log(f"Processing experiment: {name} [{ii+1}/{len(self.experiments)}]")
                        
            try:
                the_shelldata, the_peak_table, the_powder, the_diff_img = get_diff_info(exp_info['path'], cap=self.cap, 
                                                                                        keep_peak_file=False, keep_powder_file=False,
                                                                                        log=self.log)
            except Exception as e:
                self.log(f"Skipping {name} due to error: {e}")
                continue
            
            the_shelldata['experiment'] = name
            the_peak_table['experiment'] = name
            the_powder['experiment'] = name            
            self.experiments.at[name, 'diff-png'] = the_diff_img

            shells = sorted(pd.concat([the_shelldata.d_min, the_shelldata.d_max]).unique(), reverse=True)

            info.append({
                'experiment': name,
                'stage_pos': f'{exp_info["stage_x"]:.2f} {exp_info["stage_y"]:.2f} {exp_info["stage_z"]:.2f}',
                'N_peaks': len(the_peak_table),
                'I_tot': the_shelldata['I_tot'].sum(),
                'I_peak': the_peak_table[the_peak_table.I < the_peak_table.I.mean() + 5 * the_peak_table.I.std()].I.sum(),
                'I_ratio': the_peak_table[the_peak_table.I < the_peak_table.I.mean() + 5 * the_peak_table.I.std()].I.sum() \
                    / the_shelldata['I_tot'].sum(),
                'Shells [Å]': '-'.join([f'{d:.2f}' for d in shells]),
                'I_tot_shells': ' | '.join([f'{f:.0f}' for f in list(the_shelldata['I_tot'])]),
                'N_pk_shells': ' | '.join([f'{f:.0f}' for f in list(the_shelldata['N_peaks'])]),
                'I_pk_shells':  ' | '.join([f'{f:.0f}' for f in list(the_shelldata['I_peak'])]),
                'I_ratio_shells':  ' | '.join([f'{f:.3f}' for f in list(the_shelldata['peak_ratio'])]) 
            })

            sd.append(the_shelldata)
            pt.append(the_peak_table)
            pw.append(the_powder)
            
            self.log(f"Found {len(the_peak_table)} diffraction peaks for {name}")
            
        info = pd.DataFrame(info)
        info.set_index('experiment', inplace=True)
        self.experiments = self.experiments.join(info, how='left')
            
        self.peak_table = pd.concat(pt, ignore_index=True)
        self.shelldata = pd.concat(sd, ignore_index=True)
        self.powder = pd.concat(pw, ignore_index=True)
        self.shelldata['d_range'] = self.shelldata['d_max'].round(2).astype(str) + '-' + self.shelldata['d_min'].round(2).astype(str) + ' Å'
        self.shelldata['s'] = 0.5 / self.shelldata['d_max'] + 0.5 / self.shelldata['d_min']
        self.shelldata.sort_values(by=['experiment','s'], inplace=True)
        
        # aggregated = self.shelldata.groupby(['experiment']).agg(
        #     I_tot=('I_tot', 'sum'),
        #     I_peak=('I_peak', 'sum'),
        #     N_peaks=('N_peaks', 'sum')
        # )
        
        # aggregated['peak_ratio'] = aggregated['I_peak'] / aggregated['I_tot']
        # self.experiments = self.experiments.join(aggregated, how='left')
        self.update_experiments_table()
        
        self.log(f"Computed {len(self.peak_table)} peaks and {len(self.powder)} powder points from {len(self.experiments)} experiments.")
        
        self.overall_plot()
        
    def overall_plot(self):
        create_overall_figure(self.shelldata, self.fig)      
        self.canvas.draw()      

if __name__ == "__main__":
    # make the grid cells expand with window resizing
    app = App()
    app.rowconfigure(0, weight=3)
    app.rowconfigure(2, weight=1)
    app.rowconfigure(3, weight=1)
    app.columnconfigure(0, weight=1)
    app.columnconfigure(1, weight=0)
    app.mainloop()
