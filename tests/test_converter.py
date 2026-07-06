"""Unit tests for converter.py: find_dcm2niix + the dcm2niix subprocess wrapper.

Uses a fake stand-in executable instead of
the real dcm2niix binary, so these run without any external tool installed.
"""

from __future__ import annotations

import pytest

from dataset_composer.converter import convert_series, find_dcm2niix

from conftest import make_fake_dcm2niix


def test_find_dcm2niix_explicit_path(tmp_path):
    fake = make_fake_dcm2niix(tmp_path)
    assert find_dcm2niix(fake) == fake


def test_find_dcm2niix_explicit_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_dcm2niix(tmp_path / "does_not_exist")


def test_find_dcm2niix_searches_path(monkeypatch, tmp_path):
    fake = make_fake_dcm2niix(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: str(fake) if name == "dcm2niix" else None)
    assert find_dcm2niix(None) == fake


def test_find_dcm2niix_not_on_path_raises(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        find_dcm2niix(None)


def test_convert_series_success_returns_primary_nifti(tmp_path):
    dcm2niix = make_fake_dcm2niix(tmp_path, "success")
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    out_dir = tmp_path / "out"

    result = convert_series(series_dir, out_dir, "P1__t0__abcd1234", dcm2niix)

    assert bool(result) is True
    assert result.nifti is not None
    assert result.nifti.name == "P1__t0__abcd1234.nii.gz"
    assert result.nifti.exists()
    assert result.stderr == ""


def test_convert_series_nonzero_exit_reports_failure(tmp_path):
    dcm2niix = make_fake_dcm2niix(tmp_path, "fail")
    series_dir = tmp_path / "series"
    series_dir.mkdir()

    result = convert_series(series_dir, tmp_path / "out", "stem", dcm2niix)

    assert bool(result) is False
    assert result.nifti is None
    assert "simulated failure" in result.stderr


def test_convert_series_no_output_produced_reports_diagnostic(tmp_path):
    dcm2niix = make_fake_dcm2niix(tmp_path, "no_output")
    series_dir = tmp_path / "series"
    series_dir.mkdir()

    result = convert_series(series_dir, tmp_path / "out", "stem", dcm2niix)

    assert bool(result) is False
    assert "no output found" in result.stderr
    assert "localizer" in result.stderr


def test_convert_series_picks_largest_when_multiple_produced(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "stem.nii.gz").write_bytes(b"x" * 10)
    (out_dir / "stem_c2.nii.gz").write_bytes(b"x" * 1000)

    # A no-op fake that succeeds without writing anything new; the wrapper's
    # "already produced" fallback path should find the two files above.
    noop = tmp_path / "fake_noop.py"
    noop.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
    noop.chmod(0o755)

    series_dir = tmp_path / "series"
    series_dir.mkdir()
    result = convert_series(series_dir, out_dir, "stem", noop)

    assert bool(result) is True
    assert result.nifti.name == "stem_c2.nii.gz"


def test_convert_series_timeout_reports_failure(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd="dcm2niix", timeout=1)),
    )
    dcm2niix = make_fake_dcm2niix(tmp_path)
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    result = convert_series(series_dir, tmp_path / "out", "stem", dcm2niix, timeout_s=1)
    assert bool(result) is False
    assert "timeout" in result.stderr
