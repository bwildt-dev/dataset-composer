"""Shared fixtures and synthetic-data helpers for the dataset-composer test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def write_dicom(
    path: Path,
    series_uid: str,
    study_uid: Optional[str] = None,
    modality: str = "CT",
    slice_thickness: str = "1.5",
    orientation: tuple = (1, 0, 0, 0, 1, 0),
    series_description: str = "",
    protocol_name: str = "",
    body_part: str = "",
    instance_number: int = 1,
    manufacturer_model: str = "",
) -> None:
    """Write one minimal-but-valid DICOM file with the tags the scanner reads."""
    ds = Dataset()
    ds.PatientID = "P1"
    ds.Modality = modality
    if series_uid:
        ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SliceThickness = slice_thickness
    ds.ImageOrientationPatient = [str(v) for v in orientation]
    ds.SeriesDescription = series_description
    ds.ProtocolName = protocol_name
    ds.BodyPartExamined = body_part
    ds.ManufacturerModelName = manufacturer_model
    ds.InstanceNumber = instance_number

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = file_meta

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def make_series_dir(
    root: Path,
    name: str,
    n_slices: int = 25,
    series_uid: Optional[str] = None,
    **tag_kwargs,
) -> Path:
    """Create a directory of `n_slices` synthetic DICOM files sharing one series UID."""
    series_dir = root / name
    uid = series_uid or generate_uid()
    for i in range(n_slices):
        write_dicom(series_dir / f"IMG{i:04d}.dcm", series_uid=uid,
                    instance_number=i + 1, **tag_kwargs)
    return series_dir


def make_nifti(path: Path, shape=(20, 20, 20), fill: float = 1.0) -> Path:
    """Write a synthetic constant-value NIfTI volume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full(shape, fill, dtype=np.float32)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
    return path


def make_mask_dir(tmp_path: Path, voxel_counts: Dict[str, int], shape=(20, 20, 20)) -> Path:
    """Write one `<name>.nii.gz` binary mask per entry, with the given foreground voxel count."""
    seg_dir = tmp_path / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)
    flat_len = int(np.prod(shape))
    for name, n_vox in voxel_counts.items():
        arr = np.zeros(flat_len, dtype=np.float32)
        arr[:n_vox] = 1.0
        nib.save(nib.Nifti1Image(arr.reshape(shape), np.eye(4)), str(seg_dir / f"{name}.nii.gz"))
    return seg_dir


@pytest.fixture
def dummy_classifier():
    from dataset_composer.classifier import DummyClassifier
    return DummyClassifier()


_FAKE_DCM2NIIX_SUCCESS = '''#!/usr/bin/env python3
import sys
outdir = sys.argv[sys.argv.index("-o") + 1]
stem = sys.argv[sys.argv.index("-f") + 1]
with open(f"{outdir}/{stem}.nii.gz", "wb") as f:
    f.write(b"\\x1f\\x8b" + b"0" * 64)  # gzip magic + padding
sys.exit(0)
'''

_FAKE_DCM2NIIX_FAIL = '''#!/usr/bin/env python3
import sys
sys.stderr.write("fake dcm2niix: simulated failure\\n")
sys.exit(1)
'''

_FAKE_DCM2NIIX_NO_OUTPUT = '''#!/usr/bin/env python3
import sys
sys.stdout.write("Skipping (localizer/scout)\\n")
sys.exit(0)
'''


def make_fake_dcm2niix(tmp_path: Path, behavior: str = "success") -> Path:
    """Write an executable stand-in for dcm2niix that mimics its CLI contract"""
    script = {
        "success": _FAKE_DCM2NIIX_SUCCESS,
        "fail": _FAKE_DCM2NIIX_FAIL,
        "no_output": _FAKE_DCM2NIIX_NO_OUTPUT,
    }[behavior]
    path = tmp_path / "fake_dcm2niix.py"
    path.write_text(script)
    path.chmod(0o755)
    return path
