from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk
import warnings
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from cap_tools.finalization import FinalizationCollection
from cap_tools.finalization_view_model import (
    DISPLAY_LABELS,
    ViewSettings,
    load_view_settings,
    normalized_radar_selection,
    numeric_fields,
    save_view_settings,
    stable_fields,
    view_option_fields,
)
from cap_tools.interact_figures import fom_radar_plot


class PlotWidget(ttk.Frame):
    def __init__(self, parent: tk.Misc, figsize=(5, 4), hide_toolbar: bool = False):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        if not hide_toolbar:
            self.toolbar.grid(row=1, column=0, sticky=tk.EW)


def _field_label(field: str) -> str:
    return DISPLAY_LABELS.get(field, field)


class ViewOptionsDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        settings: ViewSettings,
        available: Mapping[str, Sequence[str]],
        apply_callback: Callable[[], None],
    ):
        super().__init__(parent)
        self.title("Finalization View Options")
        self.transient(parent.winfo_toplevel())
        self.settings = settings
        self.available = available
        self.apply_callback = apply_callback
        self.variables: Dict[str, Dict[str, tk.BooleanVar]] = {}
        self.direction_vars: Dict[str, tk.BooleanVar] = {}

        matrix = ttk.Frame(self)
        matrix.grid(row=0, column=0, columnspan=3, sticky=tk.NSEW, padx=8, pady=8)
        canvas = tk.Canvas(matrix, highlightthickness=0)
        scrollbar = ttk.Scrollbar(matrix, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        content = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=content, anchor=tk.NW)
        canvas.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        matrix.rowconfigure(0, weight=1)
        matrix.columnconfigure(0, weight=1)
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))

        headers = ("Figure of merit", "Overall", "Per-shell", "Spider", "Spider invert")
        for column, title in enumerate(headers):
            ttk.Label(content, text=title).grid(row=0, column=column, sticky=tk.W, padx=8, pady=(4, 8))

        overall = tuple(available.get("overall", ()))
        shell = tuple(available.get("shell", ()))
        radar = tuple(available.get("radar", ()))
        self.variables = {key: {} for key in ("overall", "shell", "radar")}
        fields = view_option_fields(overall, shell, radar)
        for row, field in enumerate(fields, start=1):
            ttk.Label(content, text=_field_label(field)).grid(row=row, column=0, sticky=tk.W, padx=8, pady=2)
            specs = (
                ("overall", overall, settings.overall_columns, {"name"}, 1),
                ("shell", shell, settings.shell_columns, {"dmin", "dmax"}, 2),
                ("radar", radar, settings.radar_metrics, set(), 3),
            )
            for key, available_fields, selected, mandatory, column in specs:
                if field not in available_fields:
                    continue
                variable = tk.BooleanVar(value=field in selected or field in mandatory)
                self.variables[key][field] = variable
                check = ttk.Checkbutton(content, variable=variable)
                check.grid(row=row, column=column, padx=12, pady=2)
                if field in mandatory:
                    check.configure(state=tk.DISABLED)
            if field in radar:
                inverted = tk.BooleanVar(value=settings.radar_directions.get(field, "higher") == "lower")
                self.direction_vars[field] = inverted
                ttk.Checkbutton(content, variable=inverted).grid(row=row, column=4, padx=12, pady=2)
        content.columnconfigure(0, weight=1)

        ttk.Button(self, text="Apply", command=self.apply).grid(row=1, column=0, padx=8, pady=(0, 8))
        ttk.Button(self, text="OK", command=self.ok).grid(row=1, column=1, padx=8, pady=(0, 8))
        ttk.Button(self, text="Cancel", command=self.destroy).grid(row=1, column=2, padx=8, pady=(0, 8))
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.geometry("720x650")
        self.grab_set()

    @staticmethod
    def _merge_hidden(old: Sequence[str], available: Sequence[str], selected: Sequence[str]) -> List[str]:
        return list(dict.fromkeys([*selected, *[value for value in old if value not in available]]))

    def apply(self) -> None:
        selected = {
            key: [field for field, variable in values.items() if variable.get()]
            for key, values in self.variables.items()
        }
        self.settings.overall_columns = self._merge_hidden(
            self.settings.overall_columns, self.available.get("overall", ()), selected["overall"]
        )
        self.settings.shell_columns = self._merge_hidden(
            self.settings.shell_columns, self.available.get("shell", ()), selected["shell"]
        )
        self.settings.radar_metrics = self._merge_hidden(
            self.settings.radar_metrics, self.available.get("radar", ()), selected["radar"]
        )
        self.settings.radar_directions.update(
            {field: "lower" if variable.get() else "higher" for field, variable in self.direction_vars.items()}
        )
        save_view_settings(self.settings)
        self.apply_callback()

    def ok(self) -> None:
        self.apply()
        self.destroy()


class FinalizationWidget(ttk.Frame):
    def __init__(self, root: tk.Misc):
        super().__init__(root)
        self.fc = FinalizationCollection()
        self.settings, self.settings_warning = load_view_settings()
        if self.settings_warning:
            warnings.warn(self.settings_warning, RuntimeWarning, stacklevel=2)
        self._row_keys: Dict[str, str] = {}
        self._available: Dict[str, List[str]] = {key: [] for key in ("overall", "shell", "plots", "radar")}

        self.configure(width=1100, height=880)
        self.grid_propagate(False)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.table_frame = ttk.Frame(self, width=650, height=880)
        self.table_frame.grid_propagate(False)
        self.table_frame.grid(row=0, column=0, sticky=tk.NSEW)
        self.table_frame.columnconfigure(0, weight=1)
        self.table_frame.rowconfigure(0, weight=1)
        self.table_frame.rowconfigure(4, weight=2)
        self.table_frame.rowconfigure(6, weight=1)

        self.fin_view = ttk.Treeview(self.table_frame, show="headings", height=7)
        self.fin_view.grid(row=0, column=0, sticky=tk.NSEW)
        self.fin_view.bind("<<TreeviewSelect>>", self.show_fin_info)
        overall_y = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.fin_view.yview)
        overall_y.grid(row=0, column=1, sticky=tk.NS)
        overall_x = ttk.Scrollbar(self.table_frame, orient=tk.HORIZONTAL, command=self.fin_view.xview)
        overall_x.grid(row=1, column=0, sticky=tk.EW)
        self.fin_view.configure(yscrollcommand=overall_y.set, xscrollcommand=overall_x.set)

        self.fin_info = tk.Text(self.table_frame, height=8, wrap=tk.WORD)
        self.fin_info.grid(row=2, column=0, sticky=tk.EW, pady=(4, 4))
        info_scroll = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.fin_info.yview)
        info_scroll.grid(row=2, column=1, sticky=tk.NS)
        self.fin_info.configure(yscrollcommand=info_scroll.set)

        self.shell_view = ttk.Treeview(self.table_frame, show="headings", height=10)
        self.shell_view.grid(row=4, column=0, sticky=tk.NSEW)
        shell_y = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.shell_view.yview)
        shell_y.grid(row=4, column=1, sticky=tk.NS)
        shell_x = ttk.Scrollbar(self.table_frame, orient=tk.HORIZONTAL, command=self.shell_view.xview)
        shell_x.grid(row=5, column=0, sticky=tk.EW)
        self.shell_view.configure(yscrollcommand=shell_y.set, xscrollcommand=shell_x.set)

        self.radar_plot = PlotWidget(self.table_frame, figsize=(6, 3), hide_toolbar=True)
        self.radar_plot.grid(row=6, column=0, columnspan=2, sticky=tk.NSEW, pady=(4, 0))

        right = ttk.Frame(self, width=450, height=880)
        right.grid_propagate(False)
        right.grid(row=0, column=1, sticky=tk.NSEW)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        self.view_options_button = ttk.Button(right, text="View Options...", command=self.open_view_options)
        self.view_options_button.grid(
            row=0, column=0, sticky=tk.E, padx=6, pady=4
        )
        selectors = ttk.Frame(right)
        selectors.grid(row=1, column=0, sticky=tk.EW, padx=6, pady=(0, 4))
        selectors.columnconfigure(1, weight=1)
        selectors.columnconfigure(3, weight=1)
        self.shell_metric_vars = (tk.StringVar(), tk.StringVar())
        ttk.Label(selectors, text="Upper plot:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        upper = ttk.Combobox(selectors, textvariable=self.shell_metric_vars[0], state="readonly", width=18)
        upper.grid(row=0, column=1, sticky=tk.EW, padx=(0, 10))
        ttk.Label(selectors, text="Lower plot:").grid(row=0, column=2, sticky=tk.W, padx=(0, 4))
        lower = ttk.Combobox(selectors, textvariable=self.shell_metric_vars[1], state="readonly", width=18)
        lower.grid(row=0, column=3, sticky=tk.EW)
        self.shell_metric_selectors = (upper, lower)
        for selector in self.shell_metric_selectors:
            selector.bind("<<ComboboxSelected>>", self._shell_plot_selection_changed)
        self.shell_plots = PlotWidget(right, figsize=(6, 7))
        self.shell_plots.grid(row=2, column=0, sticky=tk.NSEW)
        self.shell_axes: tuple = ()

    @staticmethod
    def _format_value(field: str, value) -> str:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return "—"
        if field in {"N inputs", "N used"}:
            try:
                return str(int(float(value)))
            except (TypeError, ValueError):
                return str(value)
        if field in {"R1_gt", "wR_ref"}:
            return f"{float(value):.4f}"
        if field == "GOOF":
            return f"{float(value):.3f}"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)

    @staticmethod
    def _configure_tree(tree: ttk.Treeview, columns: Sequence[str]) -> None:
        tree["columns"] = list(columns)
        for field in columns:
            label = _field_label(field)
            width = 190 if field == "name" else max(70, min(150, 10 * len(label)))
            tree.heading(field, text=label)
            tree.column(field, width=width, stretch=False, anchor=tk.CENTER)

    def _overall_display_data(self) -> pd.DataFrame:
        if not self.fc:
            return pd.DataFrame(columns=["name"])
        return self.fc.overall_highest.merge(self.fc.meta, on="name", how="left")

    def _refresh_available(self) -> None:
        overall = self._overall_display_data()
        shell = self.fc.shelldata
        overall_fields = stable_fields([overall])
        shell_fields = stable_fields([shell])
        radar_fields = numeric_fields(self.fc.overall_numeric, exclude={"1/d"})
        self._available = {
            "overall": overall_fields,
            "shell": shell_fields,
            "plots": shell_fields,
            "radar": radar_fields,
        }
        self.settings.reconcile(
            overall_fields=overall_fields,
            shell_fields=shell_fields,
            radar_fields=radar_fields,
        )

    def _refresh_shell_selectors(self) -> None:
        available = numeric_fields(
            self.fc.shelldata,
            exclude={"name", "dmin", "dmax", "1/d"},
        )
        self._available["plots"] = available
        choices = self.settings.active_shell_plots(available)
        for preferred in available:
            if preferred not in choices:
                choices.append(preferred)
            if len(choices) == 2:
                break
        if len(choices) == 1:
            choices.append(choices[0])
        for index, (variable, selector) in enumerate(zip(self.shell_metric_vars, self.shell_metric_selectors)):
            selector.configure(values=available)
            variable.set(choices[index] if index < len(choices) else "")

    def _shell_plot_selection_changed(self, _event=None) -> None:
        self.settings.shell_plots = [
            value for value in (variable.get() for variable in self.shell_metric_vars) if value
        ]
        save_view_settings(self.settings)
        self.update_shell_plot()

    def update_fc(self, fc: FinalizationCollection) -> None:
        selected = set(self.selected_fin_ids)
        self.fc = fc
        self._refresh_available()
        self._refresh_shell_selectors()
        columns = self.settings.active_overall(self._available["overall"])
        self._configure_tree(self.fin_view, columns)
        for item in self.fin_view.get_children():
            self.fin_view.delete(item)
        self._row_keys.clear()
        data = self._overall_display_data().set_index("name", drop=False) if self.fc else pd.DataFrame()
        for row_number, key in enumerate(self.fc.keys()):
            iid = f"fin-{row_number}"
            self._row_keys[iid] = key
            row = data.loc[key] if not data.empty and key in data.index else pd.Series(dtype=object)
            values = [self._format_value(field, row.get(field, key if field == "name" else None)) for field in columns]
            self.fin_view.insert("", tk.END, iid=iid, values=values)
            if key in selected:
                self.fin_view.selection_add(iid)
        if not self.fin_view.selection() and self.fin_view.get_children():
            self.fin_view.selection_set(self.fin_view.get_children()[0])
        self.show_fin_info()

    def clear(self) -> None:
        self.update_fc(FinalizationCollection())

    @property
    def selected_fin_ids(self) -> List[str]:
        return [self._row_keys[iid] for iid in self.fin_view.selection() if iid in self._row_keys]

    def _detail_text(self, key: str) -> str:
        fin = self.fc[key]
        membership = fin.merge_membership
        lines = [f"Selected finalization: {fin.name}", f"Path: {fin.path}"]
        if "Experiment" in fin.meta:
            lines.append(f"Experiment: {fin.meta['Experiment']}")
        lines.append(f"Proffit file: {'found' if fin.have_proffit else 'not found'}")
        lines.append(f"Olex2 refinement: {fin.refinement.status}")
        if fin.olex2_res_path:
            lines.append(f"Olex2 source: {fin.olex2_res_path}")
        lines.append(
            "Refinement: "
            + ", ".join(
                f"{_field_label(name)}={self._format_value(name, value)}"
                for name, value in fin.olex2_metrics.items()
            )
        )
        used = membership.used_count if membership.used_count is not None else "unknown"
        lines.append(f"Merged inputs: {membership.total_count}; used: {used}")
        if membership.used_names:
            lines.append("Used experiments: " + ", ".join(membership.used_names))
        if membership.excluded_names:
            lines.append("Excluded experiments: " + ", ".join(membership.excluded_names))
        lines.extend(f"Warning: {message}" for message in membership.warnings)
        return "\n".join(lines)

    def show_fin_info(self, _event=None) -> None:
        self.fin_info.delete("1.0", tk.END)
        selected = self.selected_fin_ids
        if not selected:
            self._update_shell_table(None)
            self.update_shell_plot()
            self.update_radar_plot()
            return
        key = selected[0]
        self.fin_info.insert(tk.END, self._detail_text(key))
        self._update_shell_table(self.fc[key].shells)
        self.update_shell_plot()
        self.update_radar_plot()

    def _update_shell_table(self, data: Optional[pd.DataFrame]) -> None:
        columns = self.settings.active_shell_columns(self._available["shell"])
        self._configure_tree(self.shell_view, columns)
        for item in self.shell_view.get_children():
            self.shell_view.delete(item)
        if data is None:
            return
        for _, row in data.iterrows():
            self.shell_view.insert(
                "", tk.END, values=[self._format_value(field, row.get(field)) for field in columns]
            )

    def update_shell_plot(self) -> None:
        metrics = [variable.get() for variable in self.shell_metric_vars]
        self.shell_plots.fig.clear()
        axes = self.shell_plots.fig.subplots(2, 1, sharex=True, squeeze=False).ravel()
        self.shell_axes = tuple(axes)
        for color, key in enumerate(self.selected_fin_ids):
            fin = self.fc[key]
            for ax, metric in zip(axes, metrics):
                if metric in fin.shells and "1/d" in fin.shells:
                    ax.plot(fin.shells["1/d"], fin.shells[metric], color=f"C{color}", label=key)
        for ax, metric in zip(axes, metrics):
            ax.set_ylabel(_field_label(metric) if metric else "No metric")
        if len(self.selected_fin_ids) > 1:
            axes[0].legend(fontsize="x-small")
        axes[-1].set_xlabel("1/d")
        self.shell_plots.fig.tight_layout()
        self.shell_plots.canvas.draw()

    def update_radar_plot(self) -> None:
        self.radar_plot.fig.clear()
        selected = self.selected_fin_ids
        metrics = self.settings.active_radar(self._available["radar"])
        if len(metrics) < 3 or not selected:
            ax = self.radar_plot.fig.add_subplot(111)
            message = "Select at least three spider axes" if len(metrics) < 3 else "Select finalizations to compare"
            ax.text(0.5, 0.5, message, ha="center", va="center")
            ax.axis("off")
            self.radar_plot.canvas.draw()
            return
        overall = normalized_radar_selection(
            self.fc.overall_numeric, selected, metrics, self.settings.radar_directions
        )
        highest = normalized_radar_selection(
            self.fc.highest_numeric, selected, metrics, self.settings.radar_directions
        )
        fom_radar_plot(
            overall,
            highest,
            fig_handle=self.radar_plot.fig,
            foms=metrics,
            fom_lbl=[_field_label(metric) for metric in metrics],
            colors=[f"C{index}" for index in range(len(overall))],
        )

    def open_view_options(self) -> ViewOptionsDialog:
        self.view_options_dialog = ViewOptionsDialog(
            self, self.settings, self._available, self._apply_view_settings
        )
        return self.view_options_dialog

    def _apply_view_settings(self) -> None:
        selected = set(self.selected_fin_ids)
        self.update_fc(self.fc)
        for iid, key in self._row_keys.items():
            if key in selected:
                self.fin_view.selection_add(iid)
        self.show_fin_info()
