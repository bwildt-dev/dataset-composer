"""Unit + integration tests for pipeline.py: discover/convert/classify/compose."""

from __future__ import annotations

import json

from dataset_composer.scanner import SeriesCandidate
from dataset_composer.state import ClassificationRecord, ComposerState
from dataset_composer import pipeline as pl

from conftest import make_fake_dcm2niix, make_nifti, make_series_dir


def test_shard_of_is_deterministic_and_covers_all_shards():
    paths = [f"/data/patient{i}/scan.nii.gz" for i in range(200)]
    n = 4
    shards = {p: pl._shard_of(p, n) for p in paths}
    assert all(0 <= k < n for k in shards.values())
    # Same path -> same shard every time.
    assert all(pl._shard_of(p, n) == shards[p] for p in paths)
    assert len(set(shards.values())) == n  # 200 paths across 4 shards -> all used


def test_discover_finds_candidates_and_writes_skipped_log(tmp_path):
    make_series_dir(tmp_path / "src", "good", n_slices=25)
    make_series_dir(tmp_path / "src", "topogram_scan", n_slices=25)
    state = ComposerState(tmp_path / "_state")
    skipped_log = tmp_path / "skipped.csv"

    candidates = pl.discover(tmp_path / "src", state, skipped_log)

    assert len(candidates) == 1
    assert candidates[0].series_dir.name == "good"
    assert skipped_log.exists()
    assert "topogram" in skipped_log.read_text()


def test_convert_produces_nifti_and_records_state(tmp_path):
    dcm2niix = make_fake_dcm2niix(tmp_path, "success")
    series_dir = make_series_dir(tmp_path / "src", "series1", n_slices=25)
    cand = SeriesCandidate(
        series_dir=series_dir, patient_id="P1", timepoint="t0",
        n_slices=25, slice_thickness_mm=1.5,
        series_description="test", series_uid="1.2.3.uid",
    )
    state = ComposerState(tmp_path / "_state")
    conversion_log = tmp_path / "conversion.csv"

    produced = pl.convert([cand], tmp_path / "raw", state, conversion_log,
                          dcm2niix_path=dcm2niix)

    assert len(produced) == 1
    assert produced[0].exists()
    rec = state.get_conversion("1.2.3.uid")
    assert rec is not None
    assert rec.patient_id == "P1"
    assert "converted" in conversion_log.read_text()


def test_convert_records_failure_row_when_dcm2niix_fails(tmp_path):
    dcm2niix = make_fake_dcm2niix(tmp_path, "fail")
    series_dir = make_series_dir(tmp_path / "src", "series1", n_slices=25)
    cand = SeriesCandidate(
        series_dir=series_dir, patient_id="P1", timepoint="t0",
        n_slices=25, slice_thickness_mm=1.5, series_description="d", series_uid="uid_fail",
    )
    state = ComposerState(tmp_path / "_state")
    conversion_log = tmp_path / "conversion.csv"

    produced = pl.convert([cand], tmp_path / "raw", state, conversion_log,
                          dcm2niix_path=dcm2niix)

    assert produced == []
    assert "failed" in conversion_log.read_text()
    assert state.get_conversion("uid_fail") is None  # failures are not recorded as done


def test_convert_adopts_existing_on_disk_nifti_without_reconverting(tmp_path):
    # dcm2niix stand-in that always fails; convert() must still succeed by
    # adopting a pre-existing matching NIfTI instead of calling it.
    dcm2niix = make_fake_dcm2niix(tmp_path, "fail")
    series_dir = make_series_dir(tmp_path / "src", "series1", n_slices=25)
    uid = "abcd12345678"
    cand = SeriesCandidate(
        series_dir=series_dir, patient_id="P1", timepoint="t0",
        n_slices=25, slice_thickness_mm=1.5,
        series_description="test", series_uid=uid,
    )
    stem = f"P1__t0__{uid[-8:]}"
    make_nifti(tmp_path / "raw" / "P1" / "t0" / f"{stem}.nii.gz")

    state = ComposerState(tmp_path / "_state")
    produced = pl.convert([cand], tmp_path / "raw", state, tmp_path / "conversion.csv",
                          dcm2niix_path=dcm2niix)

    assert len(produced) == 1
    assert state.get_conversion(uid) is not None


def test_convert_skips_when_state_already_has_valid_nifti(tmp_path):
    dcm2niix = make_fake_dcm2niix(tmp_path, "fail")  # must not be called
    series_dir = make_series_dir(tmp_path / "src", "series1", n_slices=25)
    nifti = make_nifti(tmp_path / "existing.nii.gz")
    cand = SeriesCandidate(
        series_dir=series_dir, patient_id="P1", timepoint="t0",
        n_slices=25, slice_thickness_mm=1.5, series_description="", series_uid="uid1",
    )
    state = ComposerState(tmp_path / "_state")
    state.record_conversion(series_uid="uid1", series_dir=series_dir,
                             patient_id="P1", timepoint="t0", nifti_path=nifti)

    produced = pl.convert([cand], tmp_path / "raw", state, tmp_path / "conversion.csv",
                          dcm2niix_path=dcm2niix)
    assert produced == [nifti]


def test_place_in_final_copies_to_phase_patient_timepoint_layout(tmp_path):
    nifti = make_nifti(tmp_path / "src.nii.gz")
    rec = ClassificationRecord(
        nifti_path=str(nifti), patient_id="P1", timepoint="t0",
        body_region="abdomen", phase="venous", body_confidence=1.0, phase_confidence=1.0,
    )
    pl._place_in_final(nifti, rec, tmp_path / "out")
    dest = tmp_path / "out" / "venous" / "P1" / "t0" / "src.nii.gz"
    assert dest.exists()


def test_place_in_final_no_timepoint_omits_subdirectory(tmp_path):
    nifti = make_nifti(tmp_path / "src.nii.gz")
    rec = ClassificationRecord(
        nifti_path=str(nifti), patient_id="P1", timepoint="",
        body_region="abdomen", phase="venous", body_confidence=1.0, phase_confidence=1.0,
    )
    pl._place_in_final(nifti, rec, tmp_path / "out")
    assert (tmp_path / "out" / "venous" / "P1" / "src.nii.gz").exists()


def test_place_in_final_skips_without_patient_id(tmp_path):
    nifti = make_nifti(tmp_path / "src.nii.gz")
    rec = ClassificationRecord(
        nifti_path=str(nifti), patient_id=None, timepoint="t0",
        body_region="abdomen", phase="venous", body_confidence=1.0, phase_confidence=1.0,
    )
    pl._place_in_final(nifti, rec, tmp_path / "out")
    assert not (tmp_path / "out").exists()  # returns before creating anything


def test_classify_and_finalise_keeps_matching_phase_drops_others(tmp_path, dummy_classifier):
    state = ComposerState(tmp_path / "_state")
    nifti = make_nifti(tmp_path / "raw" / "n1.nii.gz")
    state.record_conversion(series_uid="u1", series_dir=tmp_path, patient_id="P1",
                             timepoint="t0", nifti_path=nifti)

    pl.classify_and_finalise(dummy_classifier, state, tmp_path / "out",
                              tmp_path / "clf.csv", keep_phases=("venous",))

    # DummyClassifier always returns (abdomen, venous) -> kept.
    assert (tmp_path / "out" / "venous" / "P1" / "t0" / "n1.nii.gz").exists()


def test_classify_and_finalise_drops_when_phase_not_in_keep_list(tmp_path, dummy_classifier):
    state = ComposerState(tmp_path / "_state")
    nifti = make_nifti(tmp_path / "raw" / "n1.nii.gz")
    state.record_conversion(series_uid="u1", series_dir=tmp_path, patient_id="P1",
                             timepoint="t0", nifti_path=nifti)

    pl.classify_and_finalise(dummy_classifier, state, tmp_path / "out",
                              tmp_path / "clf.csv", keep_phases=("arterial",))

    assert not (tmp_path / "out" / "venous").exists()  # dropped, no phase dir placed
    clf = state.get_classification(nifti)
    assert clf is not None  # still recorded, just not copied


def test_classify_and_finalise_keep_all_ignores_keep_phases(tmp_path, dummy_classifier):
    state = ComposerState(tmp_path / "_state")
    nifti = make_nifti(tmp_path / "raw" / "n1.nii.gz")
    state.record_conversion(series_uid="u1", series_dir=tmp_path, patient_id="P1",
                             timepoint="t0", nifti_path=nifti)

    pl.classify_and_finalise(dummy_classifier, state, tmp_path / "out", tmp_path / "clf.csv",
                              keep_only_relevant=False, keep_phases=("arterial",))

    assert (tmp_path / "out" / "venous" / "P1" / "t0" / "n1.nii.gz").exists()


def test_classify_and_finalise_skips_missing_nifti(tmp_path, dummy_classifier):
    state = ComposerState(tmp_path / "_state")
    state.record_conversion(series_uid="u1", series_dir=tmp_path, patient_id="P1",
                             timepoint="t0", nifti_path=tmp_path / "gone.nii.gz")
    pl.classify_and_finalise(dummy_classifier, state, tmp_path / "out", tmp_path / "clf.csv")
    assert state.get_classification(tmp_path / "gone.nii.gz") is None


def test_classify_and_finalise_reuses_prior_classification(tmp_path):
    calls = []

    class CountingClassifier:
        def classify(self, nifti_path):
            calls.append(nifti_path)
            from dataset_composer.classifier import Classification
            return Classification("abdomen", "venous", 1.0, 1.0, {})

    state = ComposerState(tmp_path / "_state")
    nifti = make_nifti(tmp_path / "raw" / "n1.nii.gz")
    state.record_conversion(series_uid="u1", series_dir=tmp_path, patient_id="P1",
                             timepoint="t0", nifti_path=nifti)
    state.record_classification(ClassificationRecord(
        nifti_path=str(nifti), patient_id="P1", timepoint="t0",
        body_region="abdomen", phase="venous", body_confidence=1.0, phase_confidence=1.0,
    ))

    pl.classify_and_finalise(CountingClassifier(), state, tmp_path / "out", tmp_path / "clf.csv")

    assert calls == []  # classifier never called; used the cached record
    assert (tmp_path / "out" / "venous" / "P1" / "t0" / "n1.nii.gz").exists()


def test_classify_and_finalise_oversized_nifti_is_skipped(tmp_path, dummy_classifier, monkeypatch):
    monkeypatch.setattr(pl, "MAX_NIFTI_BYTES_FOR_BOA", 10)  # tiny cap
    state = ComposerState(tmp_path / "_state")
    nifti = make_nifti(tmp_path / "raw" / "n1.nii.gz")  # bigger than 10 bytes
    state.record_conversion(series_uid="u1", series_dir=tmp_path, patient_id="P1",
                             timepoint="t0", nifti_path=nifti)

    pl.classify_and_finalise(dummy_classifier, state, tmp_path / "out", tmp_path / "clf.csv")

    clf = state.get_classification(nifti)
    assert clf is not None
    assert clf.phase == "classify_error"
    assert not (tmp_path / "out" / "venous").exists()  # never placed


def test_classify_and_finalise_records_error_on_classifier_exception(tmp_path):
    class BrokenClassifier:
        def classify(self, nifti_path):
            raise RuntimeError("boom")

    state = ComposerState(tmp_path / "_state")
    nifti = make_nifti(tmp_path / "raw" / "n1.nii.gz")
    state.record_conversion(series_uid="u1", series_dir=tmp_path, patient_id="P1",
                             timepoint="t0", nifti_path=nifti)

    pl.classify_and_finalise(BrokenClassifier(), state, tmp_path / "out", tmp_path / "clf.csv")

    clf = state.get_classification(nifti)
    assert clf.phase == "classify_error"
    assert "boom" in clf.series_description


def test_populate_state_from_nifti_dir_parses_canonical_filenames(tmp_path):
    make_nifti(tmp_path / "niftis" / "P1__t0__abcd1234.nii.gz")
    make_nifti(tmp_path / "niftis" / "P2__t1__deadbeef.nii.gz")
    state = ComposerState(tmp_path / "_state")

    n = pl.populate_state_from_nifti_dir(tmp_path / "niftis", state)

    assert n == 2
    recs = {r.patient_id: r.timepoint for r in state.all_conversions()}
    assert recs == {"P1": "t0", "P2": "t1"}


def test_populate_state_from_nifti_dir_falls_back_to_extractors_for_noncanonical_names(tmp_path):
    from dataset_composer.extractors import IdExtractor, TimepointExtractor
    make_nifti(tmp_path / "niftis" / "some_weird_SUBJ42_name.nii.gz")
    state = ComposerState(tmp_path / "_state")

    n = pl.populate_state_from_nifti_dir(
        tmp_path / "niftis", state,
        id_extractor=IdExtractor(patterns=[r"SUBJ(\d+)"], format="SUBJ-{:03d}"),
        tp_extractor=TimepointExtractor(enabled=False),
    )

    assert n == 1
    rec = state.all_conversions()[0]
    assert rec.patient_id == "SUBJ-042"
    assert rec.timepoint == ""


def test_populate_state_from_nifti_dir_skips_already_represented(tmp_path):
    nifti = make_nifti(tmp_path / "niftis" / "P1__t0__abcd1234.nii.gz")
    state = ComposerState(tmp_path / "_state")
    state.record_conversion(series_uid="existing", series_dir=nifti, patient_id="P1",
                             timepoint="t0", nifti_path=nifti)

    n = pl.populate_state_from_nifti_dir(tmp_path / "niftis", state)
    assert n == 0  # already represented, no new record


def test_populate_state_from_nifti_dir_respects_shard(tmp_path):
    for i in range(20):
        make_nifti(tmp_path / "niftis" / f"P{i}__t0__uid{i:04d}.nii.gz")
    state0 = ComposerState(tmp_path / "_state0")
    state1 = ComposerState(tmp_path / "_state1")

    n0 = pl.populate_state_from_nifti_dir(tmp_path / "niftis", state0, shard=(0, 2))
    n1 = pl.populate_state_from_nifti_dir(tmp_path / "niftis", state1, shard=(1, 2))

    assert n0 + n1 == 20
    ids0 = {r.patient_id for r in state0.all_conversions()}
    ids1 = {r.patient_id for r in state1.all_conversions()}
    assert ids0.isdisjoint(ids1)


def test_compose_from_niftis_end_to_end(tmp_path, dummy_classifier):
    make_nifti(tmp_path / "niftis" / "P1__t0__aaaa1111.nii.gz")
    make_nifti(tmp_path / "niftis" / "P2__t1__bbbb2222.nii.gz")

    pl.compose_from_niftis(
        nifti_dir=tmp_path / "niftis", output_root=tmp_path / "out",
        classifier=dummy_classifier, keep_phases=("venous",),
    )

    assert (tmp_path / "out" / "venous" / "P1" / "t0" / "P1__t0__aaaa1111.nii.gz").exists()
    assert (tmp_path / "out" / "venous" / "P2" / "t1" / "P2__t1__bbbb2222.nii.gz").exists()
    assert (tmp_path / "out" / "_state" / "classification_state.json").exists()


def test_compose_from_niftis_shard_seeds_state_from_canonical(tmp_path, dummy_classifier):
    # Simulate a prior full (unsharded) compose_from_niftis run.
    make_nifti(tmp_path / "niftis" / "P1__t0__aaaa1111.nii.gz")
    pl.compose_from_niftis(nifti_dir=tmp_path / "niftis", output_root=tmp_path / "out",
                           classifier=dummy_classifier)

    # A later sharded rerun should seed its shard state from the canonical one.
    pl.compose_from_niftis(nifti_dir=tmp_path / "niftis", output_root=tmp_path / "out",
                           classifier=dummy_classifier, shard=(0, 1))

    shard_conv = json.loads((tmp_path / "out" / "_state_shard0of1" / "conversion_state.json").read_text())
    assert len(shard_conv) == 1


def test_compose_full_pipeline_end_to_end(tmp_path, dummy_classifier):
    from dataset_composer.extractors import IdExtractor
    dcm2niix = make_fake_dcm2niix(tmp_path, "success")
    make_series_dir(tmp_path / "src" / "P1", "abdomen_series", n_slices=25)

    pl.compose(
        source_dir=tmp_path / "src", output_root=tmp_path / "out",
        raw_nifti_dir=tmp_path / "raw", classifier=dummy_classifier,
        dcm2niix_path=dcm2niix, keep_phases=("venous",), unzip_archives=False,
        id_extractor=IdExtractor(patterns=[r"(P\d+)"]),
    )

    kept = list((tmp_path / "out" / "venous").rglob("*.nii.gz")) if (tmp_path / "out" / "venous").exists() else []
    assert len(kept) == 1


def test_compose_convert_only_stops_before_classification(tmp_path, dummy_classifier):
    dcm2niix = make_fake_dcm2niix(tmp_path, "success")
    make_series_dir(tmp_path / "src", "abdomen_series", n_slices=25)

    pl.compose(
        source_dir=tmp_path / "src", output_root=tmp_path / "out",
        raw_nifti_dir=tmp_path / "raw", classifier=dummy_classifier,
        dcm2niix_path=dcm2niix, unzip_archives=False, convert_only=True,
    )

    assert not (tmp_path / "out" / "_state" / "classification_state.json").exists()
    assert any((tmp_path / "raw").rglob("*.nii.gz"))


def test_compose_unzips_archives_before_processing(tmp_path, dummy_classifier):
    import zipfile
    from dataset_composer.extractors import IdExtractor

    # Zip a DICOM series into source_dir; the unzip stage must extract it
    # before discovery can find it.
    make_series_dir(tmp_path / "stage" / "P1", "abdomen_series", n_slices=25)
    src = tmp_path / "src"
    src.mkdir()
    with zipfile.ZipFile(src / "data.zip", "w") as zf:
        for f in (tmp_path / "stage").rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(tmp_path / "stage"))

    dcm2niix = make_fake_dcm2niix(tmp_path, "success")
    pl.compose(
        source_dir=src, output_root=tmp_path / "out", raw_nifti_dir=tmp_path / "raw",
        classifier=dummy_classifier, dcm2niix_path=dcm2niix, keep_phases=("venous",),
        unzip_archives=True, id_extractor=IdExtractor(patterns=[r"(P\d+)"]),
    )

    kept = list((tmp_path / "out" / "venous").rglob("*.nii.gz")) if (tmp_path / "out" / "venous").exists() else []
    assert len(kept) == 1
    assert (src / "data.zip.extracted").exists()  # unzip marker written
