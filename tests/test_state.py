"""Unit tests for state.py: ComposerState persistence, CSV logging, shard merging."""

from __future__ import annotations

import csv
import json

from dataset_composer.state import (
    ClassificationRecord,
    ComposerState,
    init_csv,
    merge_shard_states,
    safe_write_row,
)


def test_composer_state_record_and_get_conversion(tmp_path):
    state = ComposerState(tmp_path / "_state")
    state.record_conversion(
        series_uid="1.2.3", series_dir=tmp_path / "series",
        patient_id="P1", timepoint="t0", nifti_path=tmp_path / "out.nii.gz",
    )
    rec = state.get_conversion("1.2.3")
    assert rec is not None
    assert rec.patient_id == "P1"
    assert rec.nifti_path == str(tmp_path / "out.nii.gz")


def test_composer_state_get_conversion_missing_returns_none(tmp_path):
    state = ComposerState(tmp_path / "_state")
    assert state.get_conversion("nope") is None
    assert state.get_conversion("") is None


def test_composer_state_persists_across_instances(tmp_path):
    state_dir = tmp_path / "_state"
    s1 = ComposerState(state_dir)
    s1.record_conversion(series_uid="1.2.3", series_dir=tmp_path, patient_id="P1",
                          timepoint="t0", nifti_path=tmp_path / "a.nii.gz")

    s2 = ComposerState(state_dir)  # fresh instance, reloads from disk
    rec = s2.get_conversion("1.2.3")
    assert rec is not None
    assert rec.patient_id == "P1"


def test_composer_state_conversion_writes_valid_json(tmp_path):
    state_dir = tmp_path / "_state"
    state = ComposerState(state_dir)
    state.record_conversion(series_uid="1.2.3", series_dir=tmp_path, patient_id="P1",
                             timepoint="t0", nifti_path=tmp_path / "a.nii.gz")
    data = json.loads((state_dir / "conversion_state.json").read_text())
    assert "1.2.3" in data
    assert data["1.2.3"]["patient_id"] == "P1"


def test_composer_state_record_conversion_falls_back_to_synthetic_key(tmp_path):
    state = ComposerState(tmp_path / "_state")
    state.record_conversion(series_uid="", series_dir=tmp_path / "sd", patient_id="P1",
                             timepoint="t0", nifti_path=tmp_path / "a.nii.gz")
    all_recs = state.all_conversions()
    assert len(all_recs) == 1
    assert all_recs[0].series_uid.startswith("_nouid:")


def test_composer_state_classification_round_trip(tmp_path):
    state = ComposerState(tmp_path / "_state")
    rec = ClassificationRecord(
        nifti_path=str(tmp_path / "a.nii.gz"), patient_id="P1", timepoint="t0",
        body_region="abdomen", phase="venous", body_confidence=0.9, phase_confidence=0.8,
    )
    state.record_classification(rec)
    fetched = state.get_classification(tmp_path / "a.nii.gz")
    assert fetched is not None
    assert fetched.body_region == "abdomen"
    assert len(state.all_classifications()) == 1


def test_safe_write_row_appends_and_creates_valid_csv(tmp_path):
    path = tmp_path / "log.csv"
    init_csv(path, ["a", "b"])
    safe_write_row(path, [1, "x"])
    safe_write_row(path, [2, "y"])
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows == [["a", "b"], ["1", "x"], ["2", "y"]]


def test_init_csv_is_idempotent(tmp_path):
    path = tmp_path / "log.csv"
    init_csv(path, ["a", "b"])
    safe_write_row(path, [1, "x"])
    init_csv(path, ["a", "b"])  # must not truncate existing content
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows == [["a", "b"], ["1", "x"]]


def test_merge_shard_states_unions_across_shards(tmp_path):
    root = tmp_path / "out"
    for k in range(2):
        shard_dir = root / f"_state_shard{k}of2"
        shard_dir.mkdir(parents=True)
        (shard_dir / "classification_state.json").write_text(
            json.dumps({f"nifti_{k}.nii.gz": {"phase": "venous"}})
        )

    merge_shard_states(root, n_shards=2)

    merged = json.loads((root / "_state" / "classification_state.json").read_text())
    assert set(merged) == {"nifti_0.nii.gz", "nifti_1.nii.gz"}


def test_merge_shard_states_merges_into_existing_canonical_state(tmp_path):
    root = tmp_path / "out"
    canon_dir = root / "_state"
    canon_dir.mkdir(parents=True)
    (canon_dir / "classification_state.json").write_text(
        json.dumps({"existing.nii.gz": {"phase": "arterial"}})
    )
    shard_dir = root / "_state_shard0of1"
    shard_dir.mkdir(parents=True)
    (shard_dir / "classification_state.json").write_text(
        json.dumps({"new.nii.gz": {"phase": "venous"}})
    )

    merge_shard_states(root, n_shards=1)

    merged = json.loads((canon_dir / "classification_state.json").read_text())
    assert set(merged) == {"existing.nii.gz", "new.nii.gz"}


def test_merge_shard_states_force_discards_existing_canonical_state(tmp_path):
    root = tmp_path / "out"
    canon_dir = root / "_state"
    canon_dir.mkdir(parents=True)
    (canon_dir / "classification_state.json").write_text(
        json.dumps({"stale.nii.gz": {"phase": "arterial"}})
    )
    shard_dir = root / "_state_shard0of1"
    shard_dir.mkdir(parents=True)
    (shard_dir / "classification_state.json").write_text(
        json.dumps({"fresh.nii.gz": {"phase": "venous"}})
    )

    merge_shard_states(root, n_shards=1, force=True)

    merged = json.loads((canon_dir / "classification_state.json").read_text())
    assert set(merged) == {"fresh.nii.gz"}


def test_merge_shard_states_missing_shard_is_skipped(tmp_path):
    root = tmp_path / "out"
    # No shard directories exist at all.
    merge_shard_states(root, n_shards=3)
    merged = json.loads((root / "_state" / "classification_state.json").read_text())
    assert merged == {}


def test_safe_write_row_retries_then_succeeds(tmp_path, monkeypatch):
    path = tmp_path / "log.csv"
    real_open = open
    calls = {"n": 0}

    def flaky_open(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("locked by another writer")
        return real_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", flaky_open)
    safe_write_row(path, ["a", "b"], retries=3, delay=0.0)
    monkeypatch.undo()

    assert path.read_text().strip() == "a,b"
    assert calls["n"] == 2  # failed once, succeeded on the retry


def test_safe_write_row_gives_up_after_all_retries(tmp_path, monkeypatch):
    path = tmp_path / "log.csv"
    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("permanently locked")),
    )
    safe_write_row(path, ["a"], retries=2, delay=0.0)  # must not raise
    monkeypatch.undo()

    assert not path.exists()
