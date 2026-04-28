"""Standalone finalization comparison viewer.

This is the retained "Merge/Finalize" tab from the historical Cell Tool, without
the merge/finalize automation and without CAP remote control.
"""

from __future__ import annotations

import os
import tkinter as tk
import tkinter.ttk as ttk
from argparse import ArgumentParser
from typing import Iterable, Optional

from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames

from cap_tools.finalization import FinalizationCollection
from cap_tools.utils import get_resource_path, get_version
from cap_tools.widgets import FinalizationWidget


class FinalizationViewer:
    def __init__(self, initial_collection: Optional[FinalizationCollection] = None):
        self.root = tk.Tk()
        self.root.geometry("1300x900")
        self.root.title(f"Finalization viewer ({get_version()})")
        try:
            self.root.iconbitmap(get_resource_path("cell_tool_icon.ico"))
        except tk.TclError:
            pass

        self.fc = FinalizationCollection()
        self._build_ui()

        if initial_collection is not None:
            self.set_collection(initial_collection)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.viewer = FinalizationWidget(self.root)
        self.viewer.grid(row=0, column=0, sticky=tk.NSEW)

        controls = ttk.LabelFrame(self.root, text="Load Finalizations")
        controls.grid(row=0, column=1, sticky=tk.NS, padx=(6, 6), pady=6)
        controls.columnconfigure(0, weight=1)

        ttk.Button(controls, text="CSV", command=self.add_csv).grid(row=0, column=0, sticky=tk.EW, pady=(4, 0))
        ttk.Button(controls, text="Folder", command=self.add_folder).grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))
        ttk.Button(controls, text="Subfolders", command=self.add_subfolders).grid(row=2, column=0, sticky=tk.EW, pady=(4, 0))
        ttk.Button(controls, text="Files", command=self.add_files).grid(row=3, column=0, sticky=tk.EW, pady=(4, 0))
        ttk.Separator(controls, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky=tk.EW, pady=8)
        ttk.Button(controls, text="Clear", command=self.clear).grid(row=5, column=0, sticky=tk.EW)

        self.status = tk.StringVar(value="No finalizations loaded")
        ttk.Label(controls, textvariable=self.status, wraplength=180, justify=tk.LEFT).grid(
            row=6, column=0, sticky=tk.EW, pady=(12, 0)
        )

    def set_collection(self, collection: FinalizationCollection) -> None:
        self.fc = collection
        self.viewer.update_fc(self.fc)
        self.status.set(f"Loaded {len(self.fc)} finalizations")

    def add_collection(self, collection: FinalizationCollection) -> None:
        self.fc.update(collection)
        self.set_collection(self.fc)

    def add_csv(self) -> None:
        filename = askopenfilename(filetypes=[("CSV result list", "*.csv"), ("All files", "*.*")])
        if filename:
            self.add_collection(FinalizationCollection.from_csv(filename, ignore_parse_errors=True))

    def add_folder(self) -> None:
        folder = askdirectory(title="Folder containing *_red.sum files")
        if folder:
            self.add_collection(
                FinalizationCollection.from_folder(folder, ignore_parse_errors=True, include_subfolders=False)
            )

    def add_subfolders(self) -> None:
        folder = askdirectory(title="Folder tree containing *_red.sum files")
        if folder:
            self.add_collection(
                FinalizationCollection.from_folder(folder, ignore_parse_errors=True, include_subfolders=True)
            )

    def add_files(self) -> None:
        summaries = askopenfilenames(filetypes=[("Finalization summary", "*_red.sum"), ("All files", "*.*")])
        if summaries:
            self.add_collection(FinalizationCollection.from_files([filename[:-8] for filename in summaries]))

    def clear(self) -> None:
        self.set_collection(FinalizationCollection())

    def mainloop(self) -> None:
        self.root.mainloop()


def _collection_from_args(paths: Iterable[str], include_subfolders: bool) -> FinalizationCollection:
    collection = FinalizationCollection()
    for path in paths:
        norm = os.path.normpath(path)
        if os.path.isdir(norm):
            collection.update(
                FinalizationCollection.from_folder(
                    norm,
                    ignore_parse_errors=True,
                    include_subfolders=include_subfolders,
                )
            )
        elif norm.lower().endswith(".csv"):
            collection.update(FinalizationCollection.from_csv(norm, ignore_parse_errors=True))
        elif norm.endswith("_red.sum"):
            collection.update(FinalizationCollection.from_files([norm[:-8]]))
        else:
            collection.update(FinalizationCollection.from_files([norm]))
    return collection


def main_cli(argv: Optional[list[str]] = None) -> None:
    parser = ArgumentParser(description="View and compare CrysAlisPro finalization summaries.")
    parser.add_argument("paths", nargs="*", help="Folders, CSV result lists, *_red.sum files, or finalization base paths.")
    parser.add_argument("-s", "--subfolders", action="store_true", help="Search folders recursively.")
    args = parser.parse_args(argv)

    initial = _collection_from_args(args.paths, args.subfolders) if args.paths else None
    FinalizationViewer(initial).mainloop()


if __name__ == "__main__":
    main_cli()
