"""Unit tests for unzipper.py: parallel recursive archive extraction."""

from __future__ import annotations

import zipfile

import pytest

import dataset_composer.unzipper as uz
from dataset_composer.unzipper import (
    _extract_one,
    _extract_with_7z,
    _extract_with_system_unzip,
    _is_junk,
    _marker_for,
    _scan_for_new_zips,
    extract_all_zips,
)


def _make_zip(path, files: dict):
    """files: {name-in-archive: content-bytes}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


def test_is_junk_filters_macosx_and_ds_store():
    assert _is_junk("__MACOSX/foo")
    assert _is_junk("some/dir/.DS_Store")
    assert not _is_junk("real_data.dcm")


def test_marker_for_appends_extracted_suffix(tmp_path):
    archive = tmp_path / "a.zip"
    assert _marker_for(archive) == tmp_path / "a.zip.extracted"


def test_extract_one_extracts_members_and_writes_marker(tmp_path):
    archive = _make_zip(tmp_path / "data.zip", {"a.txt": b"hello", "b/c.txt": b"world"})
    result = _extract_one(archive)
    assert result is not None
    assert result.n_members == 2
    assert (result.target / "a.txt").read_bytes() == b"hello"
    assert (result.target / "b" / "c.txt").read_bytes() == b"world"
    assert _marker_for(archive).exists()


def test_extract_one_skips_junk_entries(tmp_path):
    archive = _make_zip(tmp_path / "data.zip", {
        "real.txt": b"data",
        "__MACOSX/real.txt": b"junk",
        ".DS_Store": b"junk",
    })
    result = _extract_one(archive)
    assert result.n_members == 1
    assert (result.target / "real.txt").exists()
    assert not (result.target / "__MACOSX").exists()


def test_extract_one_bad_zip_returns_none_and_no_marker(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not actually a zip file")
    result = _extract_one(archive)
    assert result is None
    assert not _marker_for(archive).exists()


def test_extract_one_delete_after_removes_archive(tmp_path):
    archive = _make_zip(tmp_path / "data.zip", {"a.txt": b"x"})
    _extract_one(archive, delete_after=True)
    assert not archive.exists()


def test_scan_for_new_zips_skips_seen_and_marked(tmp_path):
    z1 = _make_zip(tmp_path / "z1.zip", {"a.txt": b"1"})
    z2 = _make_zip(tmp_path / "z2.zip", {"a.txt": b"2"})
    _marker_for(z2).touch()  # already done

    seen = set()
    found = _scan_for_new_zips(tmp_path, seen)
    assert found == [z1]
    assert z1 in seen and z2 in seen  # both marked seen after the scan

    # A second scan finds nothing new.
    assert _scan_for_new_zips(tmp_path, seen) == []


def test_extract_all_zips_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_all_zips(tmp_path / "does_not_exist")


def test_extract_all_zips_empty_dir_returns_empty_list(tmp_path):
    assert extract_all_zips(tmp_path) == []


def test_extract_all_zips_extracts_nested_archives(tmp_path):
    inner_bytes_path = tmp_path / "_inner_stage.zip"
    _make_zip(inner_bytes_path, {"leaf.txt": b"deep"})
    inner_bytes = inner_bytes_path.read_bytes()
    inner_bytes_path.unlink()

    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("inner.zip", inner_bytes)

    results = extract_all_zips(tmp_path, n_workers=2)

    archives = {r.archive.name for r in results}
    assert "outer.zip" in archives
    assert "inner.zip" in archives
    assert (tmp_path / "outer" / "inner" / "leaf.txt").read_bytes() == b"deep"


def test_extract_all_zips_is_idempotent_on_rerun(tmp_path):
    _make_zip(tmp_path / "a.zip", {"x.txt": b"1"})
    first = extract_all_zips(tmp_path)
    second = extract_all_zips(tmp_path)
    assert len(first) == 1
    assert len(second) == 0  # marker file makes the rerun a no-op


def test_extract_with_system_unzip_returns_none_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(uz, "_SYSTEM_UNZIP", None)
    assert _extract_with_system_unzip(tmp_path / "a.zip", tmp_path / "t") is None


def test_extract_with_7z_returns_none_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(uz, "_SYSTEM_7Z", None)
    assert _extract_with_7z(tmp_path / "a.zip", tmp_path / "t") is None


def test_extract_with_system_unzip_returns_none_on_nonzero_exit(tmp_path):
    if uz._SYSTEM_UNZIP is None:
        pytest.skip("system unzip not available on this host")
    # Missing archive -> real unzip exits nonzero -> None.
    assert _extract_with_system_unzip(tmp_path / "nonexistent.zip", tmp_path / "t") is None


def test_extract_with_7z_counts_files_on_success(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(uz, "_SYSTEM_7Z", "/fake/7zz")
    target = tmp_path / "t"
    target.mkdir()
    (target / "extracted.txt").write_text("x")  # pretend 7z wrote this
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stderr=""))
    assert _extract_with_7z(tmp_path / "a.zip", target) == 1


def test_extract_with_7z_returns_none_on_nonzero_exit(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(uz, "_SYSTEM_7Z", "/fake/7zz")
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: types.SimpleNamespace(returncode=2, stderr="7z error"))
    assert _extract_with_7z(tmp_path / "a.zip", tmp_path / "t") is None


def test_extract_one_falls_back_to_system_unzip(tmp_path, monkeypatch):
    # Force zipfile to raise NotImplementedError, as it would on a
    # compression method stdlib can't handle, and fall through to `unzip`.
    archive = _make_zip(tmp_path / "data.zip", {"real.txt": b"payload"})

    def _boom(self, *a, **k):
        raise NotImplementedError("compression type 99 unsupported")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", _boom)

    if uz._SYSTEM_UNZIP is None:
        pytest.skip("system unzip not available on this host")

    result = _extract_one(archive)
    assert result is not None
    assert (result.target / "real.txt").read_bytes() == b"payload"
    assert _marker_for(archive).exists()


def test_extract_one_returns_none_when_all_fallbacks_fail(tmp_path, monkeypatch):
    archive = _make_zip(tmp_path / "data.zip", {"real.txt": b"x"})
    monkeypatch.setattr(zipfile.ZipFile, "extractall",
                        lambda self, *a, **k: (_ for _ in ()).throw(NotImplementedError()))
    monkeypatch.setattr(uz, "_extract_with_system_unzip", lambda *a, **k: None)
    monkeypatch.setattr(uz, "_extract_with_7z", lambda *a, **k: None)
    assert _extract_one(archive) is None
    assert not _marker_for(archive).exists()


def test_extract_one_delete_after_survives_unlink_error(tmp_path, monkeypatch):
    archive = _make_zip(tmp_path / "data.zip", {"a.txt": b"x"})
    monkeypatch.setattr("pathlib.Path.unlink",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("busy")))
    result = _extract_one(archive, delete_after=True)  # must not raise
    assert result is not None


def test_extract_all_zips_max_archives_cap_stops_early(tmp_path):
    for i in range(4):
        _make_zip(tmp_path / f"z{i}.zip", {"a.txt": b"x"})
    results = extract_all_zips(tmp_path, n_workers=1, max_archives=1)
    assert len(results) < 4  # cap engaged, did not process every archive


def test_extract_all_zips_survives_worker_exception(tmp_path, monkeypatch):
    _make_zip(tmp_path / "a.zip", {"x.txt": b"1"})

    def _raise(*a, **k):
        raise RuntimeError("worker blew up")

    monkeypatch.setattr(uz, "_extract_one", _raise)
    assert extract_all_zips(tmp_path) == []  # exception swallowed, no crash
