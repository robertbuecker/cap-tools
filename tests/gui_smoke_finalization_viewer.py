"""Interactive-source smoke test for the finalization viewer layout.

Run as ``python -m tests.gui_smoke_finalization_viewer <csv>`` through
with-cap-tools.ps1. This is deliberately not named test_*.py because it
requires a live Windows desktop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
import tkinter.ttk as ttk

from PIL import ImageGrab

from cap_tools.finalization import FinalizationCollection, FinalizationLoadOptions
from cap_tools.finalization_viewer import FinalizationViewer


def descendants(widget: tk.Misc):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def run(csv_path: Path, screenshot_dir: Path | None = None) -> None:
    options = FinalizationLoadOptions()
    collection = FinalizationCollection.from_csv(
        str(csv_path), ignore_parse_errors=True, verbose=False, load_options=options
    )
    app = FinalizationViewer(collection, options)
    root = app.root
    try:
        root.update()
        assert app.csv_button.grid_info()["row"] < app.selection_selector.grid_info()["row"]
        assert app.files_button.grid_info()["row"] < app.selection_selector.grid_info()["row"]
        assert app.include_entry.winfo_viewable()
        assert app.exclude_entry.winfo_viewable()
        assert app.viewer.view_options_button.winfo_viewable()
        assert all(selector.winfo_viewable() for selector in app.viewer.shell_metric_selectors)
        assert all(selector.cget("values") for selector in app.viewer.shell_metric_selectors)
        assert len(app.viewer.shell_axes) == 2
        assert all(axis.get_visible() for axis in app.viewer.shell_axes)
        assert app.viewer.shell_plots.canvas.get_tk_widget().winfo_height() > 100
        alternate = next(
            value for value in app.viewer.shell_metric_selectors[0].cget("values") if value != app.viewer.shell_metric_vars[0].get()
        )
        app.viewer.shell_metric_vars[0].set(alternate)
        app.viewer._shell_plot_selection_changed()
        assert app.viewer.shell_axes[0].get_ylabel() == alternate

        app.load_mode.set("patterns")
        app.include_patterns.set("*dyn*; *auto*")
        app.exclude_patterns.set("*reproc*")
        parsed_options = app.load_options()
        assert parsed_options.include_patterns == ("*dyn*", "*auto*")
        assert parsed_options.exclude_patterns == ("*reproc*",)

        root.lift()
        root.attributes("-topmost", True)
        root.update()
        root.attributes("-topmost", False)
        if screenshot_dir is not None:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            ImageGrab.grab().save(screenshot_dir / "finalization-viewer-main.png")

        dialog = app.viewer.open_view_options()
        root.update()
        children = list(descendants(dialog))
        assert not any(isinstance(child, ttk.Notebook) for child in children)
        labels = {child.cget("text") for child in children if isinstance(child, ttk.Label)}
        assert {"Figure of merit", "Overall", "Per-shell", "Spider", "Spider invert"} <= labels
        assert {"name"} <= set(dialog.variables["overall"])
        assert {"dmin", "dmax"} <= set(dialog.variables["shell"])
        assert dialog.direction_vars
        if screenshot_dir is not None:
            dialog.lift()
            root.update()
            ImageGrab.grab().save(screenshot_dir / "finalization-viewer-options.png")
        dialog.destroy()

        print(
            f"GUI smoke passed: {len(collection)} finalizations, "
            f"plots={tuple(variable.get() for variable in app.viewer.shell_metric_vars)}"
        )
    finally:
        root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--screenshot-dir", type=Path)
    arguments = parser.parse_args()
    run(arguments.csv, arguments.screenshot_dir)
