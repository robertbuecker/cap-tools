"""Live Tk smoke/visual check; run with python -m tests.gui_smoke_ini_wrangler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch as mock_patch

from PIL import ImageGrab

from cap_tools.ini_validation import ValidationPolicy, ValidationRule, read_policy
from cap_tools.ini_wrangler import dumps_json, read_diff
from cap_tools.ini_wrangler_gui import IniWranglerApp, PatchDialog
from cap_tools.ini_wrangler_view_model import WranglerModel, load_document


def run(screenshots: Path | None = None):
    examples = Path(__file__).resolve().parents[1] / "ini_files"
    current = examples / "CrysAlisMicroEDConfig_ver3_120kV-RC_261021.ini"
    comparison = examples / "CrysAlisMicroEDConfig_ver3_200kV-RC_261023.ini"
    policy_path = examples / "policies" / "example-structure.json"
    state = WranglerModel()
    state.set_current(load_document(current), current)
    state.set_comparison(load_document(comparison), comparison)
    state.set_policy(read_policy(policy_path), policy_path)
    app = IniWranglerApp(model=state)
    root = app.root

    def wait_until(condition, timeout=15):
        end = time.monotonic() + timeout
        while not condition():
            root.update()
            if time.monotonic() > end:
                raise AssertionError("GUI operation timed out")
            time.sleep(0.02)
        root.update()

    def screenshot(name, window=root):
        if screenshots:
            screenshots.mkdir(parents=True, exist_ok=True)
            previous = window.attributes("-topmost")
            try:
                window.attributes("-topmost", True)
                window.lift()
                window.update()
                time.sleep(0.15)
                bounds = (window.winfo_rootx(), window.winfo_rooty(),
                          window.winfo_rootx() + window.winfo_width(), window.winfo_rooty() + window.winfo_height())
                ImageGrab.grab(bbox=bounds).save(screenshots / f"{name}.png")
            finally:
                window.attributes("-topmost", previous)

    try:
        wait_until(lambda: state.validation_state == "Ready")
        assert not state.report.findings
        assert len(app._iid_keys) > 1402
        assert tuple(app.tree.cget("displaycolumns")) == ("current", "comparison", "status")
        editable = next((iid, key) for iid, key in app._iid_keys.items() if app._rows[key].path.endswith("[0].ddetectordistance"))
        iid, key = editable
        parent = app.tree.parent(iid)
        while parent:
            app.tree.item(parent, open=True)
            parent = app.tree.parent(parent)
        app.tree.selection_set(iid)
        app.tree.see(iid)
        root.update()
        screenshot("comparison")
        original = dumps_json(state.current)
        app.validate_now()
        app.begin_edit()
        assert app._edit is not None
        app._edit.delete(0, "end")
        app._edit.insert(0, "777.5")
        wait_until(lambda: state.validation_state == "Ready")
        if app._edit is not None:
            assert app._edit.get() == "777.5"
            app.commit_edit(key)
        else:
            # A real desktop focus change may legitimately commit on FocusOut.
            assert state.current.entries()[key[0], app._rows[key].path] == 777.5
        assert state.dirty and state.current.entries()[key[0], app._rows[key].path] == 777.5
        app.undo()
        assert dumps_json(state.current) == original and not state.dirty
        wait_until(lambda: state.validation_state == "Ready")
        app.invoke("Copy comparison → current")
        assert state.current.entries()[key[0], app._rows[key].path] == state.comparison.entries()[key[0], app._rows[key].path]
        app.undo()
        app.invoke("Delete current entries")
        assert (key[0], app._rows[key].path) not in state.current.entries()
        app.undo()
        assert dumps_json(state.current) == original

        with TemporaryDirectory() as directory:
            target = Path(directory) / "saved.ini"
            state.current_path = target
            app.begin_edit()
            assert app._edit is not None
            app._edit.delete(0, "end")
            app._edit.insert(0, "777.5")
            app.invoke("Save")
            assert not state.dirty and load_document(target).entries()[key[0], app._rows[key].path] == 777.5
            exported = Path(directory) / "saved.json"
            with mock_patch("cap_tools.ini_wrangler_gui.filedialog.asksaveasfilename", return_value=str(exported)):
                assert app.save(True)
            assert load_document(exported).sections == state.current.sections
            state.set_current(load_document(current), current)
            app.refresh()
            diff_path = Path(directory) / "export.json"
            with mock_patch("cap_tools.ini_wrangler_gui.filedialog.asksaveasfilename", return_value=str(diff_path)):
                app.export_diff()
                export_window = next(w for w in root.winfo_children() if w.winfo_class() == "Toplevel")
                next(w for w in export_window.winfo_children() if w.winfo_class() == "TButton" and w.cget("text") == "Export…").invoke()
            assert read_diff(diff_path) == state.export_diff()

        app.search.set("ddetectordistance")
        root.update()
        app._render_filtered()
        assert all("ddetectordistance" in app._rows[k].path for k in app._iid_keys.values() if app._rows[k].scalar)
        app.search.set("")
        app._render_filtered()
        root.update()

        app.notebook.select(1)
        app.json_text.insert("1.0", "invalid draft\n")
        root.update()
        assert state.draft_pending
        assert app._buttons["Save"].instate(["disabled"])
        assert app._buttons["Apply diff / policy…"].instate(["disabled"])
        assert dumps_json(state.current) == original
        screenshot("json-draft")
        app.revert_json()
        app.notebook.select(0)
        wait_until(lambda: state.validation_state == "Ready")

        patch = state.export_diff()
        dialog = PatchDialog(app, patch, "RC voltage comparison")
        dialog.flags["include_changes"].set(True)
        dialog.flag_changed("include_changes")
        assert not dialog.flags["include_all"].get()
        dialog.flags["include_all"].set(True)
        dialog.flag_changed("include_all")
        assert not dialog.flags["include_changes"].get()
        dialog.generate()
        wait_until(lambda: dialog.preview is not None)
        assert not dialog.accept_button.instate(["disabled"])
        screenshot("patch-preview", dialog.root)
        dialog.close()
        assert dumps_json(state.current) == original
        dialog = PatchDialog(app, patch, "RC voltage comparison")
        dialog.generate()
        wait_until(lambda: dialog.preview is not None)
        dialog.accept()
        assert state.current.sections == state.comparison.sections
        assert state.dirty
        app.undo()
        assert dumps_json(state.current) == original

        state.set_policy(ValidationPolicy([ValidationRule("missing", "present", "Missing section", "i", "error")]))
        app.refresh()
        wait_until(lambda: state.validation_state == "Ready")
        assert len(app.findings_tree.get_children()) == 1
        app.findings_tree.selection_set("0")
        app.reveal_finding()
        assert "Missing section" in app.details.get("1.0", "end")
        assert "Missing section" not in state.current.sections
        root.update()
        app.search.set("ddetectordistance")
        app.refresh()
        root.update()
        assert app.search.get() == "ddetectordistance"
        app.search.set("")

        app.clear_policy()
        app.clear_comparison()
        root.geometry("900x650")
        root.update()
        assert tuple(app.tree.cget("displaycolumns")) == ("current", "status")
        for button in app._buttons.values():
            assert button.winfo_rootx() + button.winfo_width() <= root.winfo_rootx() + root.winfo_width()
        screenshot("single-small")
        root.geometry("820x600")
        root.update()
        for button in app._buttons.values():
            assert button.winfo_rootx() + button.winfo_width() <= root.winfo_rootx() + root.winfo_width()
        screenshot("single-minimum")
        print(json.dumps({"result": "passed", "aligned_rows": len(app._iid_keys),
                          "checks": ["comparison", "cell edit during validation", "copy/delete", "undo", "save/save-as", "diff export", "filter", "draft", "preview cancel/accept", "missing finding", "single/small"]}))
    finally:
        app.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path)
    run(parser.parse_args().screenshots)
