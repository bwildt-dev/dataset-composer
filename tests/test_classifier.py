"""Unit tests for classifier.py: DummyClassifier, BOAClassifier region matching, build_classifier."""

from __future__ import annotations

import sys
import types

import nibabel as nib
import numpy as np
import pytest

from dataset_composer.classifier import (
    BOAClassifier,
    DummyClassifier,
    RegionSpec,
    build_classifier,
)

from conftest import make_mask_dir


def _install_fake_boa(monkeypatch, liver_voxels: int = 5000, predict_result=None):
    """Register a fake `boa_contrast` module in sys.modules.

    `compute_segmentation` writes a `liver.nii.gz` mask with `liver_voxels`
    foreground voxels into the seg folder it is given.
    """
    def compute_segmentation(ct_path, segmentation_folder, device_id, compute_with_docker):
        from pathlib import Path
        arr = np.zeros(8000, dtype=np.float32)
        arr[:liver_voxels] = 1.0
        nib.save(nib.Nifti1Image(arr.reshape(20, 20, 20), np.eye(4)),
                 str(Path(segmentation_folder) / "liver.nii.gz"))

    def predict(ct_path, segmentation_folder):
        return predict_result if predict_result is not None else {}

    fake = types.ModuleType("boa_contrast")
    fake.compute_segmentation = compute_segmentation
    fake.predict = predict
    monkeypatch.setitem(sys.modules, "boa_contrast", fake)
    return fake


def test_dummy_classifier_always_returns_abdomen_venous(tmp_path):
    clf = DummyClassifier()
    result = clf.classify(tmp_path / "anything.nii.gz")
    assert result.body_region == "abdomen"
    assert result.phase == "venous"
    assert result.body_confidence == 0.0
    assert result.phase_confidence == 0.0


def test_build_classifier_dummy():
    assert isinstance(build_classifier("dummy"), DummyClassifier)
    assert isinstance(build_classifier("DUMMY"), DummyClassifier)  # case-insensitive


def test_build_classifier_boa_returns_boa_instance():
    clf = build_classifier("boa", device="cpu")
    assert isinstance(clf, BOAClassifier)
    assert clf.device == "cpu"


def test_build_classifier_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown classifier backend"):
        build_classifier("not-a-real-backend")


def test_boa_classifier_requires_at_least_one_region():
    with pytest.raises(ValueError, match="at least one RegionSpec"):
        BOAClassifier(regions=[])


def test_boa_classifier_default_region_is_abdomen_liver():
    clf = BOAClassifier()
    assert clf.regions == [RegionSpec("abdomen", {"liver": 5000})]


def test_count_voxels_reads_mask_and_counts_foreground(tmp_path):
    seg_dir = make_mask_dir(tmp_path, {"liver": 600})
    clf = BOAClassifier()
    assert clf._count_voxels(seg_dir, "liver") == 600


def test_count_voxels_missing_mask_returns_zero(tmp_path):
    seg_dir = make_mask_dir(tmp_path, {"liver": 600})
    clf = BOAClassifier()
    assert clf._count_voxels(seg_dir, "pancreas") == 0


def test_count_voxels_corrupt_mask_returns_zero(tmp_path):
    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()
    (seg_dir / "liver.nii.gz").write_bytes(b"not a nifti file")  # unreadable
    clf = BOAClassifier()
    assert clf._count_voxels(seg_dir, "liver") == 0


def test_check_regions_matches_when_threshold_met(tmp_path):
    seg_dir = make_mask_dir(tmp_path, {"liver": 6000})
    clf = BOAClassifier(regions=[RegionSpec("abdomen", {"liver": 5000})])
    label, conf = clf._check_regions(seg_dir)
    assert label == "abdomen"
    assert 0.0 < conf <= 1.0


def test_check_regions_falls_through_to_next_region(tmp_path):
    seg_dir = make_mask_dir(tmp_path, {"lung": 8000})
    clf = BOAClassifier(regions=[
        RegionSpec("abdomen", {"liver": 5000}),
        RegionSpec("chest", {"lung": 5000}),
    ])
    label, conf = clf._check_regions(seg_dir)
    assert label == "chest"
    assert conf > 0.0


def test_check_regions_no_match_returns_other(tmp_path):
    seg_dir = make_mask_dir(tmp_path, {"liver": 10})  # below threshold
    clf = BOAClassifier(regions=[RegionSpec("abdomen", {"liver": 5000})])
    label, conf = clf._check_regions(seg_dir)
    assert label == "other"
    assert conf == 0.0


def test_check_regions_confidence_is_min_across_structures(tmp_path):
    seg_dir = make_mask_dir(tmp_path, {"liver": 10_000, "pancreas": 7_000})
    clf = BOAClassifier(regions=[RegionSpec("abdomen", {"liver": 5000, "pancreas": 5000})])
    label, conf = clf._check_regions(seg_dir)
    assert label == "abdomen"
    assert conf == pytest.approx(0.7, abs=1e-6)  # min(1.0, 7000/10000)


def test_check_regions_confidence_saturates_at_one(tmp_path):
    seg_dir = make_mask_dir(tmp_path, {"liver": 15_000}, shape=(30, 30, 30))
    clf = BOAClassifier(regions=[RegionSpec("abdomen", {"liver": 5000})])
    _, conf = clf._check_regions(seg_dir)
    assert conf == 1.0


def test_classify_returns_region_and_phase_from_boa(tmp_path, monkeypatch):
    _install_fake_boa(monkeypatch, liver_voxels=6000, predict_result={
        "phase_ensemble_predicted_class": "Venous",
        "phase_ensemble_prediction": 2,
        "phase_ensemble_probas": [0.1, 0.2, 0.7],
    })
    clf = BOAClassifier(device="cpu", regions=[RegionSpec("abdomen", {"liver": 5000})])
    result = clf.classify(tmp_path / "scan.nii.gz")
    assert result.body_region == "abdomen"
    assert result.phase == "venous"                 # lowercased
    assert result.body_confidence > 0.0
    assert result.phase_confidence == pytest.approx(0.7)


def test_classify_skips_phase_when_no_region_matches(tmp_path, monkeypatch):
    # Liver below threshold -> "other" -> phase prediction must be skipped.
    predict_called = {"n": 0}

    def _tracking_predict(ct_path, segmentation_folder):
        predict_called["n"] += 1
        return {}

    fake = _install_fake_boa(monkeypatch, liver_voxels=10)
    fake.predict = _tracking_predict

    clf = BOAClassifier(device="cpu", regions=[RegionSpec("abdomen", {"liver": 5000})])
    result = clf.classify(tmp_path / "scan.nii.gz")
    assert result.body_region == "other"
    assert result.phase == "skipped_region"
    assert predict_called["n"] == 0                 # never ran phase inference


def test_classify_reports_boa_failed_on_empty_prediction(tmp_path, monkeypatch):
    _install_fake_boa(monkeypatch, liver_voxels=6000, predict_result={})
    clf = BOAClassifier(device="cpu", regions=[RegionSpec("abdomen", {"liver": 5000})])
    result = clf.classify(tmp_path / "scan.nii.gz")
    assert result.body_region == "abdomen"
    assert result.phase == "boa_failed"
    assert result.phase_confidence == 0.0


def test_classify_handles_malformed_probas_gracefully(tmp_path, monkeypatch):
    _install_fake_boa(monkeypatch, liver_voxels=6000, predict_result={
        "phase_ensemble_predicted_class": "arterial",
        "phase_ensemble_prediction": 5,             # out of range for probas
        "phase_ensemble_probas": [0.5, 0.5],
    })
    clf = BOAClassifier(device="cpu", regions=[RegionSpec("abdomen", {"liver": 5000})])
    result = clf.classify(tmp_path / "scan.nii.gz")
    assert result.phase == "arterial"
    assert result.phase_confidence == 0.0           # bad index -> 0.0, no crash


def test_classify_raises_when_boa_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "boa_contrast", None)  # forces ImportError
    clf = BOAClassifier(device="cpu")
    with pytest.raises(RuntimeError):
        clf.classify(tmp_path / "scan.nii.gz")
