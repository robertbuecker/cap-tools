from datetime import datetime, timezone
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from cap_tools import ini_wrangler_view_model as vm
from cap_tools import ini_wrangler_gui as gui
from cap_tools.ini_wrangler import loads_ini, read_ini


@pytest.fixture
def active(tmp_path, monkeypatch):
    path = tmp_path / "CrysAlisMicroEDConfig_ver3.ini"
    path.write_bytes(b"\xef\xbb\xbf[S]\r\ni=1\r\n; Original comment\r\n")
    monkeypatch.setattr(vm, "ACTIVE_INI_PATH", path)
    return path


def edited_model(path):
    model = vm.WranglerModel()
    model.set_current(vm.load_document(path), path)
    model.edit_value(("S", ("i",)), "2")
    return model


def test_active_path_recognizes_identity_not_just_name(active, tmp_path):
    assert vm.is_active_ini(active)
    assert vm.is_active_ini(active.parent / "subdir" / ".." / active.name)
    if os.name == "nt":
        assert vm.is_active_ini(str(active).upper())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert not vm.is_active_ini(elsewhere / active.name)
    assert not vm.is_active_ini(None)
    alias = tmp_path / "alias.ini"
    os.link(active, alias)
    assert vm.is_active_ini(alias)


def test_active_save_requires_confirmation_then_backs_up_original_bytes(active):
    model = edited_model(active)
    original = active.read_bytes()
    assert model.current_is_active
    with pytest.raises(ValueError, match="explicit confirmation"):
        model.save()
    assert active.read_bytes() == original
    assert not (active.parent / "MicroED_ini_backup").exists()
    backup = model.save(confirm_active=True)
    assert backup.parent == active.parent / "MicroED_ini_backup"
    assert backup.name.startswith(active.stem + "_backup_") and backup.suffix == ".ini"
    assert backup.read_bytes() == original
    assert read_ini(active).sections["S"]["i"] == 2
    assert model.last_backup_path == backup and not model.dirty


def test_timestamp_collision_keeps_each_previous_version(active, monkeypatch):
    class FrozenTime:
        @staticmethod
        def now():
            return datetime(2026, 11, 9, 12, 34, 56, 789012, tzinfo=timezone.utc)
    monkeypatch.setattr(vm, "datetime", FrozenTime)
    model = edited_model(active)
    first = model.save(confirm_active=True)
    previous = active.read_bytes()
    model.edit_value(("S", ("i",)), "3")
    second = model.save(confirm_active=True)
    assert first != second and second.stem.endswith("-1")
    assert first.read_bytes().startswith(b"\xef\xbb\xbf")
    assert second.read_bytes() == previous
    assert read_ini(active).sections["S"]["i"] == 3


def test_backup_failure_aborts_before_write(active, monkeypatch):
    model = edited_model(active)
    original = active.read_bytes()
    monkeypatch.setattr(vm, "backup_active_ini", Mock(side_effect=PermissionError("backup denied")))
    writer = Mock()
    monkeypatch.setattr(vm, "_write", writer)
    with pytest.raises(PermissionError, match="backup denied"):
        model.save(confirm_active=True)
    writer.assert_not_called()
    assert active.read_bytes() == original and model.dirty


def test_backup_subfolder_cannot_be_created_aborts_save(active):
    model = edited_model(active)
    original = active.read_bytes()
    folder = active.parent / "MicroED_ini_backup"
    folder.write_text("Existing unrelated file", encoding="utf-8")
    with pytest.raises(FileExistsError):
        model.save(confirm_active=True)
    assert active.read_bytes() == original and model.dirty
    assert folder.read_text(encoding="utf-8") == "Existing unrelated file"


def test_alias_save_uses_the_active_files_backup_subfolder(active, tmp_path):
    folder = tmp_path / "alias-folder"
    folder.mkdir()
    alias = folder / "alias.ini"
    os.link(active, alias)
    original = active.read_bytes()
    backup = edited_model(alias).save(confirm_active=True)
    assert backup.parent == active.parent / "MicroED_ini_backup"
    assert backup.read_bytes() == original
    assert read_ini(active).sections["S"]["i"] == 2


def test_failed_backup_copy_removes_partial_file_and_preserves_active(active, monkeypatch):
    original = active.read_bytes()

    def fail_copy(source, output):
        output.write(b"partial")
        raise OSError("copy failed")
    monkeypatch.setattr(vm.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        edited_model(active).save(confirm_active=True)
    assert active.read_bytes() == original
    assert not list((active.parent / "MicroED_ini_backup").glob("*_backup_*.ini"))


def test_invalid_document_is_rejected_before_backup(active):
    model = edited_model(active)
    original = active.read_bytes()
    model.current.sections["S"]["i"] = True
    with pytest.raises(ValueError, match="string or number"):
        model.save(confirm_active=True)
    assert active.read_bytes() == original and not (active.parent / "MicroED_ini_backup").exists()


def test_write_failure_leaves_valid_backup_and_unsaved_state(active, monkeypatch):
    model = edited_model(active)
    original = active.read_bytes()
    monkeypatch.setattr(vm, "_write", Mock(side_effect=OSError("write failed")))
    with pytest.raises(OSError, match="write failed"):
        model.save(confirm_active=True)
    assert model.dirty
    backups = list((active.parent / "MicroED_ini_backup").glob("*_backup_*.ini"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def bare_app(model):
    app = object.__new__(gui.IniWranglerApp)
    app.model = model
    app.root = None
    app.refresh = Mock()
    return app


@pytest.mark.parametrize("save_as", [False, True])
@pytest.mark.parametrize("confirm", [False, True])
def test_gui_save_checks_destination_and_honors_confirmation(active, tmp_path, monkeypatch, save_as, confirm):
    model = edited_model(active)
    if save_as:
        model.current_path = tmp_path / "ordinary.ini"
    app = bare_app(model)
    original = active.read_bytes()
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", Mock(return_value=str(active)))
    warning = Mock(return_value=confirm)
    monkeypatch.setattr(app, "confirm_active_save", warning)
    assert app.save(save_as) is confirm
    warning.assert_called_once()
    assert warning.call_args.args[0] == active
    changes = warning.call_args.args[1].changes
    assert [(c.kind, c.path, c.old, c.new) for c in changes] == [("changed", "i", 1, 2)]
    backups = list((active.parent / "MicroED_ini_backup").glob("*_backup_*.ini"))
    if confirm:
        assert len(backups) == 1 and backups[0].read_bytes() == original
        assert read_ini(active).sections["S"]["i"] == 2
        assert model.current_is_active and not model.dirty
    else:
        assert active.read_bytes() == original and not backups and model.dirty


def test_save_as_away_from_active_needs_no_active_warning(active, tmp_path, monkeypatch):
    model = edited_model(active)
    app = bare_app(model)
    original = active.read_bytes()
    destination = tmp_path / "copy.ini"
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", Mock(return_value=str(destination)))
    warning = Mock()
    monkeypatch.setattr(app, "confirm_active_save", warning)
    assert app.save(True)
    warning.assert_not_called()
    assert not model.current_is_active and model.last_backup_path is None
    assert active.read_bytes() == original and not (active.parent / "MicroED_ini_backup").exists()
    assert read_ini(destination).sections["S"]["i"] == 2


def test_open_active_uses_existing_unsaved_change_flow(active):
    app = bare_app(vm.WranglerModel())
    app.open_current = Mock()
    app.open_active()
    app.open_current.assert_called_once_with(active)


def test_indicator_updates_for_active_normal_and_draft(active, tmp_path):
    model = edited_model(active)
    app = bare_app(model)
    label = Mock()
    app.path_labels = [label]
    app._update_current_label()
    assert label.configure.call_args.kwargs["style"] == "ActiveIni.TLabel"
    assert "[ACTIVE INI]" in label.configure.call_args.kwargs["text"]
    model.set_json_draft("pending draft")
    app._update_current_label()
    assert label.configure.call_args.kwargs["style"] == "ActiveIni.TLabel"
    model.revert_json_draft()
    model.save(tmp_path / "copy.ini")
    app._update_current_label()
    assert label.configure.call_args.kwargs["style"] == "TLabel"
    assert "[ACTIVE INI]" not in label.configure.call_args.kwargs["text"]


def test_new_active_destination_requires_confirmation_but_has_nothing_to_back_up(active):
    active.unlink()
    model = vm.WranglerModel()
    model.set_current(loads_ini("[S]\ni=5\n"))
    with pytest.raises(ValueError, match="confirmation"):
        model.save(active)
    assert model.save(active, confirm_active=True) is None
    assert read_ini(active).sections["S"]["i"] == 5


def test_active_warning_uses_requested_operating_message():
    assert gui.ACTIVE_SAVE_WARNING == (
        "The CrysAlisPro online instance MUST be closed before proceeding. "
        "Otherwise, loss of information and inconsistent instrument behavior can result"
    )


def test_active_preview_uses_current_disk_values_and_lists_all_operations(active):
    model = vm.WranglerModel()
    model.set_current(loads_ini("[S]\ni=2\ni_added=3\n[New]\n"), active)
    actual = b"[S]\ni=99\ni_removed=8\n[Gone]\n"
    active.write_bytes(actual)
    preview = model.active_save_preview(active)
    assert preview.original_bytes == actual
    assert preview.diff.added_sections == ["New"]
    assert preview.diff.removed_sections == ["Gone"]
    assert [(c.kind, c.path, c.old, c.new) for c in preview.diff.changes] == [
        ("changed", "i", 99, 2), ("added", "i_added", None, 3), ("removed", "i_removed", 8, None)]
    assert active.read_bytes() == actual and not (active.parent / "MicroED_ini_backup").exists()


def test_active_save_rejects_a_disk_change_during_confirmation(active, monkeypatch):
    model = edited_model(active)
    app = bare_app(model)
    external = b"[S]\ni=99\n"

    def confirm(path, changes):
        active.write_bytes(external)
        return True
    monkeypatch.setattr(app, "confirm_active_save", confirm)
    with pytest.raises(ValueError, match="changed while confirmation was open"):
        app.save()
    assert active.read_bytes() == external and model.dirty
    assert not (active.parent / "MicroED_ini_backup").exists()


def test_active_preview_for_new_destination_lists_additions(active):
    model = edited_model(active)
    active.unlink()
    preview = model.active_save_preview(active)
    assert preview.original_bytes is None
    assert [(c.kind, c.path, c.new) for c in preview.diff.changes] == [("added", "i", 2)]
