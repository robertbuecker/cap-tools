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

from cap_tools.finalization import (
    FinalizationCollection,
    FinalizationLoadMode,
    FinalizationLoadOptions,
)
from cap_tools.utils import get_resource_path, get_version
from cap_tools.widgets import FinalizationWidget


class FinalizationViewer:
    def __init__(self, initial_collection: Optional[FinalizationCollection] = None,
                 initial_options: Optional[FinalizationLoadOptions] = None):
        self.root = tk.Tk()
        self.root.geometry("1300x900")
        self.root.title(f"Finalization viewer ({get_version()})")
        try:
            self.root.iconbitmap(get_resource_path("cell_tool_icon.ico"))
        except tk.TclError:
            pass

        options = initial_options or FinalizationLoadOptions()
        self.fc = FinalizationCollection()
        self.load_mode = tk.StringVar(value=options.mode.value)
        self.include_patterns = tk.StringVar(value="; ".join(options.include_patterns))
        self.exclude_patterns = tk.StringVar(value="; ".join(options.exclude_patterns))
        self._build_ui()

        if initial_collection is not None:
            self.set_collection(initial_collection)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.viewer = FinalizationWidget(self.root)
        self.viewer.grid(row=0, column=0, sticky=tk.NSEW)

        controls = self.load_controls = ttk.LabelFrame(self.root, text="Load Finalizations")
        controls.grid(row=0, column=1, sticky=tk.NS, padx=(6, 6), pady=6)
        controls.columnconfigure(0, weight=1)

        self.csv_button = ttk.Button(controls, text="CSV", command=self.add_csv)
        self.csv_button.grid(row=0, column=0, sticky=tk.EW, pady=(4, 0))
        self.folder_button = ttk.Button(controls, text="Folder", command=self.add_folder)
        self.folder_button.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))
        self.subfolders_button = ttk.Button(controls, text="Subfolders", command=self.add_subfolders)
        self.subfolders_button.grid(row=2, column=0, sticky=tk.EW, pady=(4, 0))
        self.files_button = ttk.Button(controls, text="Files", command=self.add_files)
        self.files_button.grid(row=3, column=0, sticky=tk.EW, pady=(4, 0))
        ttk.Separator(controls, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky=tk.EW, pady=8)

        ttk.Label(controls, text="Selection mode").grid(row=5, column=0, sticky=tk.W)
        self.selection_selector = ttk.Combobox(
            controls,
            textvariable=self.load_mode,
            values=tuple(mode.value for mode in FinalizationLoadMode),
            state="readonly",
            width=16,
        )
        self.selection_selector.grid(row=6, column=0, sticky=tk.EW)
        ttk.Label(controls, text="Include patterns").grid(row=7, column=0, sticky=tk.W, pady=(8, 0))
        self.include_entry = ttk.Entry(controls, textvariable=self.include_patterns, width=22)
        self.include_entry.grid(row=8, column=0, sticky=tk.EW)
        ttk.Label(controls, text="Exclude patterns").grid(row=9, column=0, sticky=tk.W, pady=(6, 0))
        self.exclude_entry = ttk.Entry(controls, textvariable=self.exclude_patterns, width=22)
        self.exclude_entry.grid(row=10, column=0, sticky=tk.EW)
        ttk.Separator(controls, orient=tk.HORIZONTAL).grid(row=11, column=0, sticky=tk.EW, pady=8)
        ttk.Button(controls, text="Clear", command=self.clear).grid(row=12, column=0, sticky=tk.EW)

        self.status = tk.StringVar(value="No finalizations loaded")
        ttk.Label(controls, textvariable=self.status, wraplength=180, justify=tk.LEFT).grid(
            row=13, column=0, sticky=tk.EW, pady=(12, 0)
        )

    @staticmethod
    def _split_patterns(value: str) -> tuple[str, ...]:
        normalized = value.replace("\n", ";").replace(",", ";")
        return tuple(item.strip() for item in normalized.split(";") if item.strip())

    def load_options(self) -> FinalizationLoadOptions:
        return FinalizationLoadOptions(
            mode=FinalizationLoadMode(self.load_mode.get()),
            include_patterns=self._split_patterns(self.include_patterns.get()),
            exclude_patterns=self._split_patterns(self.exclude_patterns.get()),
        )

    def set_collection(self, collection: FinalizationCollection) -> None:
        self.fc = collection
        self.viewer.update_fc(self.fc)
        self._update_status()

    def _update_status(self) -> None:
        self.status.set(
            f"Loaded {len(self.fc)}; skipped {self.fc.skipped_count}; "
            f"malformed {self.fc.malformed_count}"
        )

    def add_collection(self, collection: FinalizationCollection) -> None:
        for finalization in collection.values():
            self.fc._add_finalization(finalization)
        self.fc.candidate_count += collection.candidate_count
        self.fc.skipped_count += collection.skipped_count
        self.fc.malformed_count += collection.malformed_count
        self.fc.load_messages.extend(collection.load_messages)
        self.set_collection(self.fc)

    def add_csv(self) -> None:
        filename = askopenfilename(filetypes=[("CSV result list", "*.csv"), ("All files", "*.*")])
        if filename:
            self.add_collection(FinalizationCollection.from_csv(
                filename, ignore_parse_errors=True, load_options=self.load_options()
            ))

    def add_folder(self) -> None:
        folder = askdirectory(title="Folder containing *_red.sum files")
        if folder:
            self.add_collection(
                FinalizationCollection.from_folder(
                    folder,
                    ignore_parse_errors=True,
                    include_subfolders=False,
                    load_options=self.load_options(),
                )
            )

    def add_subfolders(self) -> None:
        folder = askdirectory(title="Folder tree containing *_red.sum files")
        if folder:
            self.add_collection(
                FinalizationCollection.from_folder(
                    folder,
                    ignore_parse_errors=True,
                    include_subfolders=True,
                    load_options=self.load_options(),
                )
            )

    def add_files(self) -> None:
        summaries = askopenfilenames(filetypes=[("Finalization summary", "*_red.sum"), ("All files", "*.*")])
        if summaries:
            self.add_collection(FinalizationCollection.from_files([filename[:-8] for filename in summaries]))

    def clear(self) -> None:
        self.set_collection(FinalizationCollection())

    def mainloop(self) -> None:
        self.root.mainloop()


def _collection_from_args(
    paths: Iterable[str],
    include_subfolders: bool,
    load_options: Optional[FinalizationLoadOptions] = None,
) -> FinalizationCollection:
    collection = FinalizationCollection()
    options = load_options or FinalizationLoadOptions()
    for path in paths:
        norm = os.path.normpath(path)
        if os.path.isdir(norm):
            loaded = FinalizationCollection.from_folder(
                norm,
                ignore_parse_errors=True,
                include_subfolders=include_subfolders,
                load_options=options,
            )
        elif norm.lower().endswith(".csv"):
            loaded = FinalizationCollection.from_csv(norm, ignore_parse_errors=True, load_options=options)
        elif norm.endswith("_red.sum"):
            loaded = FinalizationCollection.from_files([norm[:-8]])
        else:
            loaded = FinalizationCollection.from_files([norm])
        for finalization in loaded.values():
            collection._add_finalization(finalization)
        collection.candidate_count += loaded.candidate_count
        collection.skipped_count += loaded.skipped_count
        collection.malformed_count += loaded.malformed_count
        collection.load_messages.extend(loaded.load_messages)
    return collection


def main_cli(argv: Optional[list[str]] = None) -> None:
    parser = ArgumentParser(description="View and compare CrysAlisPro finalization summaries.")
    parser.add_argument("paths", nargs="*", help="Folders, CSV result lists, *_red.sum files, or finalization base paths.")
    parser.add_argument("-s", "--subfolders", action="store_true", help="Search folders recursively.")
    parser.add_argument(
        "--selection",
        choices=tuple(mode.value for mode in FinalizationLoadMode),
        default=FinalizationLoadMode.CURRENT.value,
        help="Choose current, all, or pattern-filtered finalizations.",
    )
    parser.add_argument("--include", action="append", default=[], metavar="GLOB", help="Include glob (repeatable).")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="Exclude glob (repeatable).")
    args = parser.parse_args(argv)

    options = FinalizationLoadOptions(
        mode=FinalizationLoadMode(args.selection),
        include_patterns=tuple(args.include),
        exclude_patterns=tuple(args.exclude),
    )
    initial = _collection_from_args(args.paths, args.subfolders, options) if args.paths else None
    FinalizationViewer(initial, options).mainloop()


if __name__ == "__main__":
    main_cli()
