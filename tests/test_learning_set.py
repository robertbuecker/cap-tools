import numpy as np
from tifffile import imread


def test_materialize_snapshot_prefers_existing_requested_format(tmp_path, monkeypatch):
    from cap_tools import learning_set

    source_base = tmp_path / "snapshot"
    source = source_base.with_suffix(".tiff")
    output = tmp_path / "output.tiff"
    source.write_bytes(b"existing-tiff")

    def fail_read(_filename):
        raise AssertionError("ROD reader should not be used when a TIFF exists")

    monkeypatch.setattr(learning_set, "read_rod_image", fail_read)

    assert learning_set._materialize_snapshot(str(source_base), str(output), ".tiff", lambda _msg: None)
    assert output.read_bytes() == source.read_bytes()


def test_materialize_snapshot_exports_rod_pixels_as_tiff(tmp_path, monkeypatch):
    from cap_tools import learning_set

    source_base = tmp_path / "snapshot"
    rod_source = source_base.with_suffix(".rodhypix")
    output = tmp_path / "output.tiff"
    rod_source.write_bytes(b"placeholder")
    pixels = np.array([[0, 1], [42, 100_000]], dtype=np.int32)
    messages = []

    monkeypatch.setattr(learning_set, "read_rod_image", lambda filename: pixels)

    assert learning_set._materialize_snapshot(str(source_base), str(output), ".tiff", messages.append)
    np.testing.assert_array_equal(imread(output), pixels)
    assert messages == [f"Exported TIFF snapshot from {rod_source}"]


def test_materialize_snapshot_does_not_convert_rod_to_jpeg(tmp_path, monkeypatch):
    from cap_tools import learning_set

    source_base = tmp_path / "snapshot"
    source_base.with_suffix(".rodhypix").write_bytes(b"placeholder")

    def fail_read(_filename):
        raise AssertionError("ROD reader should not be used for JPEG output")

    monkeypatch.setattr(learning_set, "read_rod_image", fail_read)

    assert not learning_set._materialize_snapshot(
        str(source_base), str(tmp_path / "output.jpg"), ".jpg", lambda _msg: None
    )
