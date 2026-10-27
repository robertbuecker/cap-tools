"""Tkinter INI Wrangler: aligned entries, JSON editing, and policy findings."""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from dataclasses import asdict
import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cap_tools._version import __version__
from cap_tools.ini_validation import ValidationFailed, loads_policy, read_policy, validate_document
from cap_tools.ini_wrangler import WranglerError, default_diff_name, loads_diff, write_diff
from cap_tools.ini_wrangler_view_model import (
    RowFilters, WranglerModel, display_value, evaluate_preview, load_document,
)


def background(function, *args):
    """A daemon worker: closing a window never waits for a user custom check."""
    future = Future()

    def run():
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(function(*args))
            except BaseException as exc:
                future.set_exception(exc)
    threading.Thread(target=run, daemon=True).start()
    return future


def text_panel(parent, *, editable=False, height=8):
    frame = ttk.Frame(parent)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    text = tk.Text(frame, wrap="none", font="TkFixedFont", height=height, undo=editable)
    text.grid(row=0, column=0, sticky="nsew")
    vertical = ttk.Scrollbar(frame, command=text.yview)
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
    horizontal.grid(row=1, column=0, sticky="ew")
    text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    if not editable:
        text.configure(state="disabled")
    return frame, text


def set_text(widget, value, editable=False):
    if widget.get("1.0", "end-1c") == value:
        return
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", value)
    widget.edit_modified(False)
    if editable:
        widget.edit_reset()
    if not editable:
        widget.configure(state="disabled")


class IniWranglerApp:
    def __init__(self, root=None, model=None):
        self.root = root if root is not None else tk.Tk()
        self.model = model if model is not None else WranglerModel()
        self.root.title(f"INI Wrangler ({__version__})")
        self.root.geometry("1350x900")
        self.root.minsize(820, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False
        self._syncing_json = False
        self._rows = {}
        self._iid_keys = {}
        self._edit = None
        self._edit_key = None
        self._validation_future = None
        self._validation_requested = None
        self._filter_after = None
        self._actions = {}
        self._menus = []
        self._buttons = {}
        self._build()
        self._bind_keys()
        self.refresh()
        self.root.after_idle(lambda: self.panes.sashpos(0, int(self.panes.winfo_height() * 0.72)))
        self._poll_after = self.root.after(100, self._poll)

    def _action(self, name, callback, enabled):
        self._actions[name] = (callback, enabled)

    def invoke(self, name):
        if self._edit is not None and not self.commit_edit(self._edit_key):
            return
        callback, enabled = self._actions[name]
        if not enabled():
            return
        try:
            callback()
        except (OSError, ValueError) as exc:
            messagebox.showerror("INI Wrangler", str(exc), parent=self.root)

    def _build(self):
        ready = lambda: self.model.current is not None and not self.model.draft_pending
        self._action("Open current…", self.open_current, lambda: True)
        self._action("Open comparison…", self.open_comparison, ready)
        self._action("Clear comparison", self.clear_comparison, lambda: ready() and self.model.comparison is not None)
        self._action("Load policy…", self.load_policy, ready)
        self._action("Clear policy", self.clear_policy, lambda: ready() and self.model.policy is not None)
        self._action("Save", self.save, ready)
        self._action("Save As…", lambda: self.save(True), ready)
        self._action("Undo", self.undo, lambda: self.model.can_undo)
        self._action("Redo", self.redo, lambda: self.model.can_redo)
        self._action("Copy comparison → current", self.copy_selected, lambda: ready() and self._selected_scalars("comparison"))
        self._action("Delete current entries", self.delete_selected, lambda: ready() and self._selected_scalars("current"))
        self._action("Export diff…", self.export_diff, lambda: ready() and self.model.comparison is not None)
        self._action("Apply diff / policy…", self.open_patch, ready)
        self._action("Validate now", self.validate_now, lambda: ready() and self.model.policy is not None)
        menu = tk.Menu(self.root)
        self.root.configure(menu=menu)
        for label, actions in (
            ("File", ["Open current…", "Open comparison…", "Clear comparison", "Save", "Save As…", "Export diff…"]),
            ("Edit", ["Undo", "Redo", "Copy comparison → current", "Delete current entries"]),
            ("Policy", ["Load policy…", "Clear policy", "Validate now", "Apply diff / policy…"]),
        ):
            submenu = tk.Menu(menu, tearoff=False)
            menu.add_cascade(label=label, menu=submenu)
            for name in actions:
                submenu.add_command(label=name, command=lambda n=name: self.invoke(n))
                self._menus.append((submenu, submenu.index("end"), name))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.grid(row=0, column=0, sticky="ew")
        for row, names in enumerate((
            ["Open current…", "Open comparison…", "Load policy…", "Save", "Save As…", "Export diff…"],
            ["Copy comparison → current", "Delete current entries", "Undo", "Redo", "Apply diff / policy…", "Validate now"],
        )):
            for column, name in enumerate(names):
                button = ttk.Button(toolbar, text=name, command=lambda n=name: self.invoke(n))
                button.grid(row=row, column=column, padx=(0, 5), pady=2, sticky="w")
                self._buttons[name] = button
        self._toolbar_width = None

        def layout_toolbar(event):
            if event.width == self._toolbar_width:
                return
            self._toolbar_width = event.width
            x, y, count, height = 6, 4, 0, max(b.winfo_reqheight() for b in self._buttons.values())
            for button in self._buttons.values():
                width = button.winfo_reqwidth()
                if count >= 6 or (count and x + width > event.width - 6):
                    x, y, count = 6, y + height + 4, 0
                button.grid_forget()
                button.place(x=x, y=y, width=width, height=height)
                x, count = x + width + 5, count + 1
            toolbar.configure(height=y + height + 4)
        toolbar.bind("<Configure>", layout_toolbar)

        paths = ttk.Frame(self.root, padding=(6, 0))
        paths.grid(row=1, column=0, sticky="ew")
        paths.columnconfigure(0, weight=1)
        self.path_labels = []
        for row in range(3):
            label = ttk.Label(paths, wraplength=1200)
            label.grid(row=row, column=0, sticky="w")
            self.path_labels.append(label)
        paths.bind("<Configure>", lambda event: [label.configure(wraplength=max(400, event.width - 20)) for label in self.path_labels])

        filters = ttk.Frame(self.root, padding=6)
        filters.grid(row=2, column=0, sticky="ew")
        filters.columnconfigure(3, weight=1)
        self.presence = tk.StringVar(value="Either file (OR)")
        self.comparison_filter = tk.StringVar(value="Any")
        self.findings_filter = tk.StringVar(value="All entries")
        self.search = tk.StringVar()
        for row, column, label, variable, values in (
            (0, 0, "Presence", self.presence, ["Either file (OR)", "Both files (AND)", "Current only", "Comparison only"]),
            (0, 2, "Comparison", self.comparison_filter, ["Any", "Different", "Equal"]),
            (1, 0, "Findings", self.findings_filter, ["All entries", "Entries with findings", "Errors", "Warnings", "Information"]),
        ):
            ttk.Label(filters, text=label).grid(row=row, column=column, sticky="w", padx=(0, 5))
            selector = ttk.Combobox(filters, textvariable=variable, values=values, state="readonly", width=23)
            selector.grid(row=row, column=column + 1, sticky="w", padx=(0, 15), pady=2)
        ttk.Label(filters, text="Search").grid(row=1, column=2, sticky="w")
        ttk.Entry(filters, textvariable=self.search).grid(row=1, column=3, sticky="ew")
        for variable in (self.presence, self.comparison_filter, self.findings_filter, self.search):
            variable.trace_add("write", self._filters_changed)

        self.panes = ttk.Panedwindow(self.root, orient="vertical")
        self.panes.grid(row=3, column=0, sticky="nsew", padx=6)
        self.notebook = ttk.Notebook(self.panes)
        self.panes.add(self.notebook, weight=4)
        tree_frame = ttk.Frame(self.notebook)
        tree_frame.rowconfigure(1, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.notebook.add(tree_frame, text="Tree")
        tree_tools = ttk.Frame(tree_frame)
        tree_tools.grid(row=0, column=0, sticky="ew")
        ttk.Button(tree_tools, text="Expand all", command=lambda: self.expand_all(True)).pack(side="left")
        ttk.Button(tree_tools, text="Collapse all", command=lambda: self.expand_all(False)).pack(side="left", padx=4)
        self.row_count = ttk.Label(tree_tools)
        self.row_count.pack(side="right")
        self.tree = ttk.Treeview(tree_frame, columns=("current", "comparison", "status"), selectmode="extended")
        self.tree.heading("#0", text="Entry")
        self.tree.column("#0", width=510, minwidth=180, stretch=True)
        for name, title, width in (("current", "Current — editable", 170),
                                   ("comparison", "Comparison — read-only", 170),
                                   ("status", "Diff / validation", 250)):
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, minwidth=100, stretch=name == "status")
        self.tree.grid(row=1, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(tree_frame, command=self.tree.yview)
        yscroll.grid(row=1, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        xscroll.grid(row=2, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.tree.bind("<Double-1>", self._double_click)
        self.tree.bind("<F2>", self.begin_edit)

        json_frame = ttk.Frame(self.notebook)
        json_frame.rowconfigure(1, weight=1)
        json_frame.columnconfigure(0, weight=1)
        self.notebook.add(json_frame, text="JSON")
        json_tools = ttk.Frame(json_frame)
        json_tools.grid(row=0, column=0, sticky="ew")
        self.json_apply = ttk.Button(json_tools, text="Apply JSON", command=self.apply_json)
        self.json_apply.pack(side="left")
        self.json_revert = ttk.Button(json_tools, text="Revert", command=self.revert_json)
        self.json_revert.pack(side="left", padx=5)
        self.json_status = ttk.Label(json_tools, text="Complete current document")
        self.json_status.pack(side="left", padx=5)
        panel, self.json_text = text_panel(json_frame, editable=True)
        panel.grid(row=1, column=0, sticky="nsew")
        self.json_text.bind("<<Modified>>", self._json_modified)

        self.lower = ttk.Notebook(self.panes)
        self.panes.add(self.lower, weight=1)
        details_frame, self.details = text_panel(self.lower, height=7)
        self.lower.add(details_frame, text="Details")
        findings_frame = ttk.Frame(self.lower)
        findings_frame.rowconfigure(0, weight=1)
        findings_frame.columnconfigure(0, weight=1)
        self.lower.add(findings_frame, text="Findings")
        self.findings_tree = ttk.Treeview(findings_frame, columns=("severity", "rule", "path", "message"), show="headings")
        for name, width in (("severity", 85), ("rule", 180), ("path", 400), ("message", 400)):
            self.findings_tree.heading(name, text=name.title())
            self.findings_tree.column(name, width=width, minwidth=70)
        self.findings_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(findings_frame, command=self.findings_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(findings_frame, orient="horizontal", command=self.findings_tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.findings_tree.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        self.findings_tree.bind("<<TreeviewSelect>>", self.reveal_finding)
        self.status = ttk.Label(self.root, padding=6)
        self.status.grid(row=4, column=0, sticky="ew")

    def _bind_keys(self):
        for key, action in (("<Control-o>", "Open current…"), ("<Control-s>", "Save")):
            self.root.bind(key, lambda event, name=action: self.invoke(name))
        self.root.bind("<Control-z>", lambda event: self._history_key("Undo"))
        self.root.bind("<Control-y>", lambda event: self._history_key("Redo"))

    def _history_key(self, name):
        if self.root.focus_get() is self.json_text or self._edit is not None:
            return
        self.invoke(name)
        return "break"

    def filters(self):
        return RowFilters(self.presence.get(), self.comparison_filter.get(), self.findings_filter.get(), self.search.get())

    def _filters_changed(self, *args):
        if self._filter_after is not None:
            self.root.after_cancel(self._filter_after)
        self._filter_after = self.root.after(120, self._render_filtered)

    def _render_filtered(self):
        self._filter_after = None
        self.render_tree()

    def _selected_keys(self):
        return [self._iid_keys[iid] for iid in self.tree.selection() if iid in self._iid_keys]

    def _selected_scalars(self, side):
        keys = self._selected_keys()
        return bool(keys) and all(getattr(self._rows[key], f"{side}_present")
                                  and type(getattr(self._rows[key], side)) in (str, int, float) for key in keys)

    def _sync_actions(self):
        for name, (_, enabled) in self._actions.items():
            if name in self._buttons:
                self._buttons[name].configure(state="normal" if enabled() else "disabled")
        for menu, index, name in self._menus:
            menu.entryconfigure(index, state="normal" if self._actions[name][1]() else "disabled")
        self.json_apply.configure(state="normal" if self.model.draft_pending else "disabled")
        self.json_revert.configure(state="normal" if self.model.draft_pending else "disabled")

    def refresh(self):
        self._syncing_json = True
        set_text(self.json_text, self.model.json_draft, editable=self.model.current is not None)
        self._syncing_json = False
        self.json_status.configure(text="Draft pending — Apply JSON or Revert" if self.model.draft_pending else "Complete current document")
        self.path_labels[0].configure(text=f"Current: {self.model.current_path or 'No file'}{'  • Unsaved' if self.model.dirty else ''}")
        self.path_labels[1].configure(text=f"Comparison (read-only): {self.model.comparison_path or ('In memory' if self.model.comparison else 'None')}")
        self.path_labels[2].configure(text=f"Policy: {self.model.policy_path or ('In memory' if self.model.policy else 'None')}")
        self.root.title(f"{'* ' if self.model.dirty else ''}INI Wrangler ({__version__})")
        self._rows = self.model.aligned_rows()
        self.render_tree()
        self.render_findings()
        self._sync_actions()
        self.update_status()

    def render_tree(self):
        self.cancel_edit()
        selected = set(self._selected_keys())
        opened = {self._iid_keys[iid] for iid in self._iid_keys if self.tree.exists(iid) and self.tree.item(iid, "open")}
        initial = not self._iid_keys
        top = self.tree.yview()[0]
        self.tree.delete(*self.tree.get_children())
        self._iid_keys = {}
        display = ("current", "comparison", "status") if self.model.comparison is not None else ("current", "status")
        self.tree.configure(displaycolumns=display)
        visible = self.model.filtered_rows(self.filters(), self._rows)
        active_filter = self.filters() != RowFilters()
        for row in visible:
            iid = json.dumps([row.key[0], list(row.key[1])], ensure_ascii=True)
            parent = json.dumps([row.parent[0], list(row.parent[1])], ensure_ascii=True) if row.parent else ""
            label = row.key[0] if not row.key[1] else (f"[{row.key[1][-1]}]" if isinstance(row.key[1][-1], int) else row.key[1][-1])
            status = row.status
            if not row.scalar and row.differences:
                status = f"{row.differences} differences"
            if row.findings:
                levels = ", ".join(f"{sum(f.severity == level for f in row.findings)} {level}" for level in ("error", "warning", "info") if any(f.severity == level for f in row.findings))
                status = f"{status}; {levels}" if status else levels
            self.tree.insert(parent, "end", iid=iid, text=label,
                             values=(display_value(row.current, row.current_present),
                                     display_value(row.comparison, row.comparison_present), status),
                             open=active_filter or row.key in opened or (initial and not row.key[1]))
            self._iid_keys[iid] = row.key
            if row.key in selected:
                self.tree.selection_add(iid)
        self.tree.update_idletasks()
        self.tree.yview_moveto(top)
        self.row_count.configure(text=f"{sum(r.scalar for r in visible)} displayed scalar entries")
        self._selection_changed()

    def render_findings(self):
        self.findings_tree.delete(*self.findings_tree.get_children())
        if self.model.report:
            for index, finding in enumerate(self.model.report.findings):
                self.findings_tree.insert("", "end", iid=str(index), values=(finding.severity, finding.rule_id,
                                          f"[{finding.section}] {finding.path}", finding.message))

    def update_status(self):
        model = self.model
        state = model.validation_state
        if model.report is not None:
            counts = model.report.to_dict()["counts"]
            state += " — " + ", ".join(f"{counts[level]} {level}" for level in ("error", "warning", "info"))
        if model.validation_error:
            state += f": {model.validation_error}"
        diagnostics = len(model.current.diagnostics) if model.current else 0
        self.status.configure(text=f"Validation: {state}  |  {diagnostics} parser diagnostics" +
                              ("  |  JSON draft pending; document actions suspended" if model.draft_pending else ""), wraplength=max(700, self.root.winfo_width() - 20))

    def _selection_changed(self, event=None):
        keys = self._selected_keys()
        if keys:
            row = self._rows[keys[0]]
            findings = self._findings_for_key(row.key)
            data = {"section": row.key[0], "path": row.path, "comparison_status": row.status,
                    "current_present": row.current_present, "current": row.current,
                    "comparison_present": row.comparison_present, "comparison": row.comparison,
                    "findings": [asdict(f) for f in findings]}
            if self.model.current:
                data["parser_diagnostics"] = [asdict(d) for d in self.model.current.diagnostics
                    if d.section == row.key[0] and (d.path == row.path or not row.path)]
            set_text(self.details, json.dumps(data, indent=2, ensure_ascii=False))
        else:
            set_text(self.details, "Select an entry or a finding to see full paths, values, and messages.")
        self._sync_actions()

    def _findings_for_key(self, key):
        if not self.model.report:
            return []
        # Container selection includes findings below it and rules on its ancestors.
        result = []
        for finding in self.model.report.findings:
            nearest = self.model.nearest_key(finding.section, finding.path, self._rows)
            if nearest and nearest[0] == key[0] and (
                    nearest[1][:len(key[1])] == key[1]
                    or (key[1][:len(nearest[1])] == nearest[1]
                        and finding.path == self._rows[nearest].path)):
                result.append(finding)
        return result

    def reveal_finding(self, event=None):
        selection = self.findings_tree.selection()
        if not selection or not self.model.report:
            return
        finding = self.model.report.findings[int(selection[0])]
        self.presence.set("Either file (OR)")
        self.comparison_filter.set("Any")
        self.findings_filter.set("All entries")
        self.search.set("")
        self.render_tree()
        key = self.model.nearest_key(finding.section, finding.path, self._rows)
        if key is not None:
            iid = json.dumps([key[0], list(key[1])], ensure_ascii=True)
            parent = self.tree.parent(iid)
            while parent:
                self.tree.item(parent, open=True)
                parent = self.tree.parent(parent)
            self.tree.selection_set(iid)
            self.tree.see(iid)
        set_text(self.details, json.dumps(asdict(finding), indent=2, ensure_ascii=False))
        self.lower.select(0)
        self.notebook.select(0)

    def expand_all(self, opened):
        for iid in self._iid_keys:
            self.tree.item(iid, open=opened)

    def _double_click(self, event):
        if self.tree.identify_region(event.x, event.y) == "cell" and self.tree.identify_column(event.x) == "#1":
            return self.begin_edit(event)

    def begin_edit(self, event=None):
        if self.model.draft_pending or not self._selected_scalars("current") or len(self.tree.selection()) != 1:
            return
        iid = self.tree.selection()[0]
        bounds = self.tree.bbox(iid, "current")
        if not bounds:
            return
        self.cancel_edit()
        key = self._iid_keys[iid]
        value = self._rows[key].current
        self._edit = ttk.Entry(self.tree)
        self._edit_key = key
        self._edit.insert(0, str(value))
        self._edit.place(x=bounds[0], y=bounds[1], width=bounds[2], height=bounds[3])
        self._edit.focus_set()
        self._edit.selection_range(0, "end")
        self._edit.bind("<Escape>", lambda event: (self.cancel_edit(), self.refresh()))
        self._edit.bind("<Return>", lambda event: self.commit_edit(key))
        self._edit.bind("<FocusOut>", lambda event: self.commit_edit(key))
        return "break"

    def cancel_edit(self):
        editor, self._edit = self._edit, None
        self._edit_key = None
        if editor is not None:
            editor.destroy()

    def commit_edit(self, key):
        if self._edit is None:
            return True
        value = self._edit.get()
        self.cancel_edit()
        try:
            self.model.edit_value(key, value)
            self.refresh()
            return True
        except ValueError as exc:
            messagebox.showerror("Edit value", str(exc), parent=self.root)
            return False

    def _json_modified(self, event=None):
        if not self.json_text.edit_modified():
            return
        self.json_text.edit_modified(False)
        if self._syncing_json or self.model.current is None:
            return
        was_pending = self.model.draft_pending
        self.model.set_json_draft(self.json_text.get("1.0", "end-1c"))
        if was_pending != self.model.draft_pending:
            self._rows = self.model.aligned_rows()
            self.render_tree()
            self.render_findings()
        self._sync_actions()
        self.json_status.configure(text="Draft pending — Apply JSON or Revert" if self.model.draft_pending else "Complete current document")
        self.path_labels[0].configure(text=f"Current: {self.model.current_path or 'In memory'}{'  • Unsaved' if self.model.dirty else ''}")
        self.root.title(f"{'* ' if self.model.dirty else ''}INI Wrangler ({__version__})")
        self.update_status()

    def apply_json(self):
        try:
            self.model.apply_json_draft()
            self.refresh()
        except ValueError as exc:
            messagebox.showerror("JSON draft", str(exc), parent=self.root)

    def revert_json(self):
        self.model.revert_json_draft()
        self.refresh()

    def confirm_replace(self):
        if not self.model.dirty:
            return True
        answer = messagebox.askyesnocancel("Unsaved changes", "Save changes before continuing?", parent=self.root)
        if answer is None:
            return False
        if not answer:
            return True
        if self.model.draft_pending:
            self.model.apply_json_draft()
            self.refresh()
        return self.save()

    def open_current(self, path=None):
        if not self.confirm_replace():
            return
        path = path or filedialog.askopenfilename(parent=self.root, title="Open current configuration", filetypes=[("Configurations", "*.ini *.json"), ("All files", "*.*")])
        if path:
            self.model.set_current(load_document(path), path)
            self._validation_requested = None
            self.refresh()

    def open_comparison(self, path=None):
        path = path or filedialog.askopenfilename(parent=self.root, title="Open read-only comparison", filetypes=[("Configurations", "*.ini *.json"), ("All files", "*.*")])
        if path:
            self.model.set_comparison(load_document(path), path)
            self.refresh()

    def clear_comparison(self):
        self.model.set_comparison(None)
        self.refresh()

    def load_policy(self, path=None):
        path = path or filedialog.askopenfilename(parent=self.root, title="Load validation policy", filetypes=[("Policies", "*.json")])
        if path:
            self.model.set_policy(read_policy(path), path)
            self.refresh()

    def clear_policy(self):
        self.model.set_policy(None)
        self.refresh()

    def save(self, save_as=False):
        path = self.model.current_path
        if save_as or path is None:
            path = filedialog.asksaveasfilename(parent=self.root, title="Save current configuration", defaultextension=".ini",
                    initialfile=path.name if path else "configuration.ini", filetypes=[("INI", "*.ini"), ("Structured JSON", "*.json")])
            if not path:
                return False
        self.model.save(path)
        self.refresh()
        return True

    def undo(self):
        self.model.undo()
        self.refresh()

    def redo(self):
        self.model.redo()
        self.refresh()

    def copy_selected(self):
        self.model.copy_comparison(self._selected_keys())
        self.refresh()

    def delete_selected(self):
        self.model.delete_entries(self._selected_keys())
        self.refresh()

    def export_diff(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Export complete-document diff")
        dialog.transient(self.root)
        direction = tk.StringVar(value="Current → Comparison")
        ttk.Label(dialog, text="Export all entries; display filters do not limit the diff.").pack(padx=15, pady=10)
        for value in ("Current → Comparison", "Comparison → Current"):
            ttk.Radiobutton(dialog, text=value, variable=direction, value=value).pack(anchor="w", padx=15)

        def export():
            reverse = direction.get() == "Comparison → Current"
            first, second = self.model.current_path or Path("current.ini"), self.model.comparison_path or Path("comparison.ini")
            if reverse:
                first, second = second, first
            path = filedialog.asksaveasfilename(parent=dialog, title="Export diff", defaultextension=".json",
                                              initialfile=default_diff_name(first, second), filetypes=[("JSON diff", "*.json")])
            if path:
                try:
                    protected = [p for p in (self.model.current_path, self.model.comparison_path, self.model.policy_path) if p]
                    if any(Path(path).resolve() == p.resolve() for p in protected):
                        raise WranglerError("Export must not overwrite an input file")
                    write_diff(self.model.export_diff(reverse), path, overwrite=True)
                    dialog.destroy()
                except (OSError, ValueError) as exc:
                    messagebox.showerror("Export diff", str(exc), parent=dialog)
        ttk.Button(dialog, text="Export…", command=export).pack(side="left", padx=15, pady=15)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(side="right", padx=15, pady=15)
        dialog.grab_set()

    def open_patch(self, path=None):
        path = path or filedialog.askopenfilename(parent=self.root, title="Apply diff or policy", filetypes=[("Diff / policy JSON", "*.json")])
        if not path:
            return
        text = Path(path).read_text(encoding="utf-8-sig")
        data = json.loads(text)
        operation = loads_policy(text, source=str(path)) if isinstance(data, dict) and {"checks", "format_version"} & data.keys() else loads_diff(text, source=str(path))
        return PatchDialog(self, operation, str(path))

    def validate_now(self):
        self._validation_requested = None
        self.model._invalidate()
        self.refresh()

    def _poll(self):
        if self._closed:
            return
        if self._validation_future is not None and self._validation_future.done():
            token, future = self._validation_future_token, self._validation_future
            self._validation_future = None
            try:
                accepted = self.model.accept_validation(token, future.result())
            except Exception as exc:
                accepted = self.model.accept_validation(token, None, str(exc))
            if accepted:
                if self._edit is None:
                    self.refresh()
                else:
                    # A validation completion must not destroy an in-progress cell edit.
                    self.render_findings()
                    self.update_status()
        if (self._validation_future is None and self.model.policy is not None and self.model.current is not None
                and not self.model.draft_pending and self.model.validation_state == "Pending"
                and self._validation_requested != self.model.token):
            token, document, policy = self.model.validation_request()
            self._validation_requested = self._validation_future_token = token
            self._validation_future = background(validate_document, document, policy)
            self.update_status()
        self._poll_after = self.root.after(100, self._poll)

    def close(self):
        try:
            if not self.confirm_replace():
                return
        except (OSError, ValueError) as exc:
            messagebox.showerror("Unsaved changes", str(exc), parent=self.root)
            return
        self.destroy()

    def destroy(self):
        self._closed = True
        self.cancel_edit()
        if self._filter_after is not None:
            self.root.after_cancel(self._filter_after)
        self.root.after_cancel(self._poll_after)
        self.root.destroy()


class PatchDialog:
    def __init__(self, app, operation, title):
        self.app, self.operation = app, operation
        self.root = tk.Toplevel(app.root)
        self.root.title("Preview diff / policy application")
        self.root.geometry("1000x650")
        self.root.minsize(650, 450)
        self.root.transient(app.root)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False
        self._future = None
        self.preview = None
        ttk.Label(self.root, text=title, wraplength=850).grid(row=0, column=0, sticky="w", padx=10, pady=8)
        options = ttk.Frame(self.root)
        options.grid(row=1, column=0, sticky="ew", padx=10)
        self.flags = {name: tk.BooleanVar(value=name == "include_all") for name in
                      ("include_additions", "include_deletions", "include_changes", "include_all", "check_old")}
        for column, (name, variable) in enumerate(self.flags.items()):
            label = {"include_all": "All", "check_old": "Check old values"}.get(name, name.removeprefix("include_").title())
            ttk.Checkbutton(options, text=label, variable=variable, command=lambda n=name: self.flag_changed(n)).grid(row=0, column=column, padx=(0, 10))
        panel, self.text = text_panel(self.root, height=15)
        panel.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.status = ttk.Label(self.root, text="Choose operations, then generate a preview.", wraplength=850)
        self.status.grid(row=3, column=0, sticky="w", padx=10)
        buttons = ttk.Frame(self.root)
        buttons.grid(row=4, column=0, sticky="ew", padx=10, pady=10)
        self.generate_button = ttk.Button(buttons, text="Generate preview", command=self.generate)
        self.generate_button.pack(side="left")
        self.accept_button = ttk.Button(buttons, text="Accept into current", command=self.accept, state="disabled")
        self.accept_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Cancel", command=self.close).pack(side="right")
        self.root.grab_set()
        self._poll_after = self.root.after(100, self.poll)

    def invalidate(self):
        self.preview = None
        self.accept_button.configure(state="disabled")
        self.status.configure(text="Options changed; generate a new preview.")

    def flag_changed(self, name):
        if name == "include_all" and self.flags[name].get():
            for specific in ("include_additions", "include_deletions", "include_changes"):
                self.flags[specific].set(False)
        elif name != "check_old" and name != "include_all" and self.flags[name].get():
            self.flags["include_all"].set(False)
        self.invalidate()

    def generate(self):
        try:
            request = self.app.model.preview_request(self.operation, **{name: var.get() for name, var in self.flags.items()})
            self._options = tuple(var.get() for var in self.flags.values())
            self._future = background(evaluate_preview, request)
            self.preview = None
            self.accept_button.configure(state="disabled")
            self.generate_button.configure(state="disabled")
            self.status.configure(text="Evaluating candidate and policies…")
        except (OSError, ValueError) as exc:
            self.status.configure(text=str(exc))

    def poll(self):
        if self._closed:
            return
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            self.generate_button.configure(state="normal")
            try:
                preview = future.result()
                if self._options != tuple(var.get() for var in self.flags.values()) or preview.revision != self.app.model.token:
                    self.invalidate()
                else:
                    self.preview = preview
                    set_text(self.text, json.dumps({"candidate_diff": asdict(preview.diff), "validation": preview.report.to_dict()}, indent=2, ensure_ascii=False))
                    self.accept_button.configure(state="normal")
                    self.status.configure(text=f"{len(preview.diff.changes)} entry changes. Acceptance edits current in memory; Save writes it.")
            except Exception as exc:
                self.preview = None
                report = exc.report.to_dict() if isinstance(exc, ValidationFailed) else {"error": str(exc)}
                set_text(self.text, json.dumps(report, indent=2, ensure_ascii=False))
                self.status.configure(text="Candidate rejected; current is unchanged.")
        self._poll_after = self.root.after(100, self.poll)

    def accept(self):
        if self.preview is None:
            return
        try:
            self.app.model.accept_preview(self.preview)
            self.app.refresh()
            self.close()
        except ValueError as exc:
            self.invalidate()
            self.status.configure(text=str(exc))

    def close(self):
        self._closed = True
        self.root.after_cancel(self._poll_after)
        self.root.destroy()


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", nargs="?", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args(argv)
    if (args.compare or args.policy) and args.current is None:
        parser.error("--compare and --policy require a current file")
    model = WranglerModel()
    try:
        if args.current:
            model.set_current(load_document(args.current), args.current)
        if args.compare:
            model.set_comparison(load_document(args.compare), args.compare)
        if args.policy:
            model.set_policy(read_policy(args.policy), args.policy)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    app = IniWranglerApp(model=model)
    app.root.mainloop()


if __name__ == "__main__":
    main_cli()
