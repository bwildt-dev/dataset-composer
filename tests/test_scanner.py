"""Unit tests for scanner.py: header-based DICOM series pre-filters."""

from __future__ import annotations

from dataset_composer.scanner import (
    compile_filter_patterns,
    filter_series,
    find_series_dirs,
    folder_name_filter,
    is_axial,
    is_ct_modality,
    is_volume,
    thickness_in_range,
    _has_dicom_magic,
    _parse_orientation,
)

from conftest import make_series_dir, write_dicom


def test_compile_filter_patterns_matches_any_fragment():
    rx = compile_filter_patterns(["FOO", "BAR"])
    assert rx.search("some_FOO_thing")
    assert rx.search("bar_case_insensitive".upper())
    assert not rx.search("unrelated")


def test_compile_filter_patterns_empty_never_matches():
    rx = compile_filter_patterns([])
    assert not rx.search("anything at all")


def test_parse_orientation_accepts_comma_separated():
    arr = _parse_orientation("1,0,0,0,1,0")
    assert arr is not None
    assert arr.tolist() == [1, 0, 0, 0, 1, 0]


def test_parse_orientation_rejects_wrong_length():
    assert _parse_orientation("1,0,0") is None
    assert _parse_orientation("") is None


def test_is_axial_true_for_identity_orientation():
    ok, reason = is_axial({"ImageOrientationPatient": "1,0,0,0,1,0"})
    assert ok
    assert reason == ""


def test_is_axial_false_for_sagittal_orientation():
    # Row=Y axis, Col=Z axis -> normal along X, not Z.
    ok, reason = is_axial({"ImageOrientationPatient": "0,1,0,0,0,1"})
    assert not ok
    assert "non-axial" in reason


def test_is_axial_missing_tag_assumed_axial():
    ok, reason = is_axial({})
    assert ok
    assert "assumed axial" in reason


def test_thickness_in_range_bounds():
    assert thickness_in_range({"SliceThickness": "1.5"}) == (True, "")
    ok, reason = thickness_in_range({"SliceThickness": "0.1"})
    assert not ok and "too thin" in reason
    ok, reason = thickness_in_range({"SliceThickness": "10.0"})
    assert not ok and "too thick" in reason


def test_thickness_in_range_missing_or_invalid_defaults_to_ok():
    assert thickness_in_range({}) == (True, "")
    assert thickness_in_range({"SliceThickness": "not-a-number"}) == (True, "")
    assert thickness_in_range({"SliceThickness": "0"}) == (True, "")


def test_is_ct_modality_accepts_ct_rejects_others():
    assert is_ct_modality({"Modality": "CT"}) == (True, "")
    ok, reason = is_ct_modality({"Modality": "MR"})
    assert not ok and "non-CT" in reason


def test_is_ct_modality_rejects_pet_mr_model_name():
    ok, reason = is_ct_modality({"Modality": "CT", "ManufacturerModelName": "PET/CT Scanner"})
    assert not ok
    assert "PET/MR/SPECT" in reason


def test_is_volume_min_slice_count():
    assert is_volume({"_n_slices": "25"}) == (True, "")
    ok, reason = is_volume({"_n_slices": "3"})
    assert not ok and "too few slices" in reason


def test_is_volume_non_numeric_count_treated_as_zero():
    ok, reason = is_volume({"_n_slices": "not-a-number"})
    assert not ok and "too few slices" in reason


def test_folder_name_filter_drops_reformats_and_topograms():
    ok, reason = folder_name_filter("Ax_MPR_recon")
    assert not ok and "reformat" in reason
    ok, reason = folder_name_filter("topogram_scan")
    assert not ok and "topogram" in reason
    assert folder_name_filter("normal_ct_series") == (True, "")


def test_has_dicom_magic_true_for_real_dicom(tmp_path):
    series_dir = make_series_dir(tmp_path, "s1", n_slices=1)
    f = next(series_dir.iterdir())
    assert _has_dicom_magic(f)


def test_has_dicom_magic_false_for_plain_file(tmp_path):
    f = tmp_path / "notdicom.txt"
    f.write_bytes(b"just some bytes, not a dicom file at all padding")
    assert not _has_dicom_magic(f)


def test_find_series_dirs_finds_dcm_and_extensionless(tmp_path):
    make_series_dir(tmp_path, "with_ext", n_slices=3)
    no_ext_dir = tmp_path / "no_ext"
    write_dicom(no_ext_dir / "IMG0001", series_uid="1.2.3.4.9")  # no .dcm suffix

    found = find_series_dirs(tmp_path)
    assert (tmp_path / "with_ext") in found
    assert no_ext_dir in found


def test_find_series_dirs_ignores_non_dicom_dirs(tmp_path):
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "readme.txt").write_text("hello")
    assert find_series_dirs(tmp_path) == []


def test_filter_series_keeps_good_series_and_skips_bad(tmp_path):
    good = make_series_dir(tmp_path, "good_series", n_slices=25, modality="CT",
                            slice_thickness="1.5")
    make_series_dir(tmp_path, "topogram_series", n_slices=25, modality="CT")
    too_few = make_series_dir(tmp_path, "too_few_slices", n_slices=5, modality="CT")

    candidates, skipped = filter_series([good, tmp_path / "topogram_series", too_few])

    kept_dirs = {c.series_dir for c in candidates}
    assert good in kept_dirs
    assert (tmp_path / "topogram_series") not in kept_dirs
    assert too_few not in kept_dirs
    skipped_reasons = {s.series_dir: s.reason for s in skipped}
    assert "topogram" in skipped_reasons[tmp_path / "topogram_series"]
    assert "too few slices" in skipped_reasons[too_few]


def test_filter_series_dedups_by_series_uid_keeping_larger(tmp_path):
    uid = "1.2.3.4.1"
    small = make_series_dir(tmp_path, "dup_small", n_slices=20, series_uid=uid)
    large = make_series_dir(tmp_path, "dup_large", n_slices=30, series_uid=uid)

    candidates, skipped = filter_series([small, large])

    assert len(candidates) == 1
    assert candidates[0].series_dir == large
    dup_reasons = [s.reason for s in skipped if s.series_dir == small]
    assert dup_reasons and "duplicate SeriesInstanceUID" in dup_reasons[0]


def test_filter_series_skips_existing_uids(tmp_path):
    series = make_series_dir(tmp_path, "already_done", n_slices=25, series_uid="1.2.3.4.2")
    candidates, skipped = filter_series([series], skip_existing_uids={"1.2.3.4.2"})
    assert candidates == []
    assert skipped == []  # skip_existing happens after reading tags in filter_series's uid check


def test_filter_series_skips_existing_paths_without_reading_header(tmp_path):
    series = make_series_dir(tmp_path, "already_done_path", n_slices=25)
    candidates, skipped = filter_series([series], skip_existing_paths={str(series)})
    assert candidates == []
    assert skipped == []


def test_filter_series_keeps_candidate_with_no_series_uid(tmp_path):
    # A series whose DICOMs lack SeriesInstanceUID still survives; the dedup
    # step keys it by path instead of UID.
    series_dir = tmp_path / "nouid_series"
    for i in range(25):
        write_dicom(series_dir / f"IMG{i:04d}.dcm", series_uid="", instance_number=i + 1)

    candidates, _ = filter_series([series_dir])

    assert len(candidates) == 1
    assert candidates[0].series_uid == ""
