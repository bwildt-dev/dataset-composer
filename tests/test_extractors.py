"""Unit tests for extractors.py: IdExtractor, TimepointExtractor, read_series_tags."""

from __future__ import annotations

from pathlib import Path

from dataset_composer.extractors import IdExtractor, TimepointExtractor, read_series_tags

from conftest import make_series_dir


def test_id_extractor_no_patterns_returns_none():
    ex = IdExtractor()
    assert ex(Path("/data/SUBJ001/scan")) is None


def test_id_extractor_zero_pads_numeric_capture():
    ex = IdExtractor(patterns=[r"SUBJ(\d+)"], format="SUBJ-{:03d}")
    assert ex(Path("/data/SUBJ7/scan/file.dcm")) == "SUBJ-007"


def test_id_extractor_verbatim_format_for_non_numeric():
    ex = IdExtractor(patterns=[r"(PT-[A-Z]+)"], format="{}")
    assert ex(Path("/data/PT-ABC/scan")) == "PT-ABC"


def test_id_extractor_tries_patterns_in_order():
    ex = IdExtractor(patterns=[r"NOPE(\d+)", r"SUBJ(\d+)"], format="{}")
    assert ex(Path("/data/SUBJ42/scan")) == "42"


def test_id_extractor_no_match_returns_none():
    ex = IdExtractor(patterns=[r"SUBJ(\d+)"])
    assert ex(Path("/data/unrelated/scan")) is None


def test_id_extractor_case_insensitive():
    ex = IdExtractor(patterns=[r"subj(\d+)"], format="{}")
    assert ex(Path("/data/SUBJ42/scan")) == "42"


def test_timepoint_extractor_default_labels():
    tp = TimepointExtractor()
    assert tp(Path("/data/P1/PRERAND/scan")) == "t0"
    assert tp(Path("/data/P1/EVAL/scan")) == "t1"
    assert tp(Path("/data/P1/other/scan")) == "unknown"


def test_timepoint_extractor_disabled_returns_empty_string():
    tp = TimepointExtractor(enabled=False)
    assert tp(Path("/data/P1/PRERAND/scan")) == ""


def test_timepoint_extractor_custom_labels_and_fallback():
    tp = TimepointExtractor(labels={"baseline": ["BL"], "followup": ["FU"]}, fallback="na")
    assert tp(Path("/data/P1/BL/scan")) == "baseline"
    assert tp(Path("/data/P1/FU/scan")) == "followup"
    assert tp(Path("/data/P1/xyz/scan")) == "na"


def test_read_series_tags_returns_expected_fields(tmp_path):
    series_dir = make_series_dir(
        tmp_path, "series1", n_slices=5,
        modality="CT", slice_thickness="2.0",
        series_description="ARTERIAL PHASE", protocol_name="Abdomen",
    )
    tags = read_series_tags(series_dir)
    assert tags["Modality"] == "CT"
    assert tags["SliceThickness"] == "2.0"
    assert tags["SeriesDescription"] == "ARTERIAL PHASE"
    assert tags["_n_slices"] == "5"


def test_read_series_tags_empty_dir_returns_empty_dict(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert read_series_tags(empty) == {}


def test_read_series_tags_unreadable_file_returns_empty_dict(tmp_path, monkeypatch):
    # force=True tolerates garbage bytes, so make dcmread raise directly to
    # exercise the except branch.
    series_dir = tmp_path / "bad_series"
    series_dir.mkdir()
    (series_dir / "notdicom.dcm").write_bytes(b"not a dicom file")

    def _raise(*a, **kw):
        raise ValueError("corrupt")

    monkeypatch.setattr("pydicom.dcmread", _raise)
    assert read_series_tags(series_dir) == {}
