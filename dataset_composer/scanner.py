"""DICOM tree scanner with header-based pre-filters.

Walks a source directory, groups DICOMs into series, and applies fast
deterministic filters: folder-name regex for reformats/topograms,
`ImageOrientationPatient` cosines for non-axial series, slice thickness
bounds, modality (CT only), and minimum slice count. Anything that survives
is a conversion candidate of unknown body region and contrast phase.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

from .config import DEFAULT_REFORMAT_PATTERNS, DEFAULT_TOPOGRAM_PATTERNS
from .extractors import (
    IdExtractor,
    TimepointExtractor,
    _DEFAULT_ID_EXTRACTOR,
    _DEFAULT_TP_EXTRACTOR,
    read_series_tags,
)

logger = logging.getLogger(__name__)


def compile_filter_patterns(patterns: Iterable[str]) -> "re.Pattern":
    """Join a list of regex fragments with `|` and compile case-insensitively.

    Each fragment is wrapped in a non-capturing group `(?:…)` so anchors
    and alternation inside a fragment behave intuitively when combined.

    Args:
        patterns: Regex fragments to join.

    Returns:
        A compiled regex that matches if any fragment matches.
    """
    parts = [p for p in patterns if p]
    if not parts:
        # Never-match sentinel
        return re.compile(r"(?!)")
    joined = "|".join(f"(?:{p})" for p in parts)
    return re.compile(joined, re.IGNORECASE)


# Module-level defaults, used as a fallback when callers don't supply
# config-driven patterns.
_REFORMAT_RE = compile_filter_patterns(DEFAULT_REFORMAT_PATTERNS)
_TOPOGRAM_RE = compile_filter_patterns(DEFAULT_TOPOGRAM_PATTERNS)

MIN_SLICE_THICKNESS_MM = 0.5
MAX_SLICE_THICKNESS_MM = 5.0
MIN_AXIAL_COSINE       = 0.95
MIN_SLICE_COUNT        = 20


@dataclass(frozen=True)
class SeriesCandidate:
    """A series that survived header-based filtering, ready for conversion.

    Attributes:
        series_dir: DICOM series directory.
        patient_id: Extracted patient ID, or None if unrecognised.
        timepoint: Extracted timepoint label.
        n_slices: Number of DICOM files in the series.
        slice_thickness_mm: SliceThickness tag value, or None if absent.
        series_description: SeriesDescription tag value.
        series_uid: SeriesInstanceUID tag value.
    """

    series_dir:  Path
    patient_id:  Optional[str]
    timepoint:   str
    n_slices:    int
    slice_thickness_mm: Optional[float]
    series_description: str
    series_uid:  str


@dataclass(frozen=True)
class SkippedSeries:
    """A series directory rejected by a pre-filter, with the rejection reason.

    Attributes:
        series_dir: DICOM series directory.
        patient_id: Extracted patient ID, or None if unrecognised.
        timepoint: Extracted timepoint label.
        reason: Rejection reason.
    """

    series_dir:  Path
    patient_id:  Optional[str]
    timepoint:   str
    reason:      str


def _parse_orientation(s: str) -> Optional[np.ndarray]:
    """Parse a 6-component `ImageOrientationPatient` string into an ndarray.

    Accepts comma-separated, bracket-wrapped, and pydicom MultiValue forms.

    Args:
        s: Raw `ImageOrientationPatient` tag value.

    Returns:
        A 6-element float array, or None if `s` can't be parsed.
    """
    if not s:
        return None
    try:
        parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if len(parts) != 6:
            return None
        return np.array([float(p) for p in parts], dtype=float)
    except Exception:
        return None


def is_axial(tags: dict) -> Tuple[bool, str]:
    """Return True if the series has axial orientation (slice normal ≈ ±z).

    Missing orientation tag is treated as axial; the classifier will catch
    non-abdominal scans anyway.

    Args:
        tags: DICOM tags from `read_series_tags`.
    """
    iop = _parse_orientation(tags.get("ImageOrientationPatient", ""))
    if iop is None:
        return True, "no orientation tag (assumed axial)"
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    z_alignment = abs(normal[2])
    if z_alignment < MIN_AXIAL_COSINE:
        return False, (
            f"non-axial orientation (|n_z|={z_alignment:.2f} < "
            f"{MIN_AXIAL_COSINE}); normal={normal.round(2).tolist()}"
        )
    return True, ""


def thickness_in_range(tags: dict) -> Tuple[bool, str]:
    """Return True when the SliceThickness tag is within the configured bounds.

    Args:
        tags: DICOM tags from `read_series_tags`.
    """
    raw = tags.get("SliceThickness", "")
    if not raw:
        return True, ""
    try:
        t = float(raw)
    except ValueError:
        return True, ""
    if t <= 0:
        return True, ""
    if t < MIN_SLICE_THICKNESS_MM:
        return False, f"slice too thin ({t} mm < {MIN_SLICE_THICKNESS_MM})"
    if t > MAX_SLICE_THICKNESS_MM:
        return False, f"slice too thick ({t} mm > {MAX_SLICE_THICKNESS_MM})"
    return True, ""


def is_ct_modality(tags: dict) -> Tuple[bool, str]:
    """Return True when the DICOM tags indicate a CT scan.

    Args:
        tags: DICOM tags from `read_series_tags`.
    """
    mod = tags.get("Modality", "").upper()
    if mod and mod != "CT":
        return False, f"non-CT modality: {mod}"
    model = tags.get("ManufacturerModelName", "")
    if re.search(r"PET|SPECT|MRI", model, re.IGNORECASE):
        return False, f"PET/MR/SPECT scanner: {model}"
    return True, ""


def is_volume(tags: dict) -> Tuple[bool, str]:
    """Reject series with too few slices.

    Args:
        tags: DICOM tags from `read_series_tags`.
    """
    n_str = tags.get("_n_slices", "0")
    try:
        n = int(n_str)
    except ValueError:
        n = 0
    if n < MIN_SLICE_COUNT:
        return False, f"too few slices ({n} < {MIN_SLICE_COUNT}); likely scout/topogram"
    return True, ""


def folder_name_filter(
    folder_name: str,
    reformat_re: Optional["re.Pattern"] = None,
    topogram_re: Optional["re.Pattern"] = None,
) -> Tuple[bool, str]:
    """Drop obvious reformats and admin series by folder-name regex.

    Args:
        folder_name: Series folder name to test.
        reformat_re: Compiled reformat-pattern regex.
        topogram_re: Same as `reformat_re`, for topogram/admin series.
    """
    rfm = reformat_re if reformat_re is not None else _REFORMAT_RE
    tpg = topogram_re if topogram_re is not None else _TOPOGRAM_RE
    if rfm.search(folder_name):
        return False, f"folder name suggests reformat: {folder_name!r}"
    if tpg.search(folder_name):
        return False, f"folder name suggests topogram/admin: {folder_name!r}"
    return True, ""


def _has_dicom_magic(path: Path) -> bool:
    """True if the file starts with a 128-byte preamble + 'DICM'.

    Args:
        path: File to check.
    """
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def find_series_dirs(root: Path) -> List[Path]:
    """Return every directory containing at least one DICOM file.

    A file counts as DICOM if it has a `.dcm` extension, or has no
    extension and passes a DICOM magic-byte check.

    Args:
        root: Directory tree to walk.
    """
    series: set = set()
    pbar = tqdm(
        os.walk(root),
        desc="Walking tree",
        unit=" dir",
        mininterval=30.0,
        smoothing=0.1,
    )
    for dirpath, _dirnames, filenames in pbar:
        dir_path = Path(dirpath)
        for name in filenames:
            if name.endswith(".dcm"):
                series.add(dir_path)
                break
            if "." not in name and _has_dicom_magic(dir_path / name):
                series.add(dir_path)
                break
        if hasattr(pbar, "set_postfix_str"):
            pbar.set_postfix_str(f"{len(series)} series found", refresh=False)
    return sorted(series)


def filter_series(
    series_dirs: Iterable[Path],
    skip_existing_uids:  Optional[set] = None,
    skip_existing_paths: Optional[set] = None,
    id_extractor: IdExtractor = _DEFAULT_ID_EXTRACTOR,
    tp_extractor: TimepointExtractor = _DEFAULT_TP_EXTRACTOR,
    reformat_re: Optional["re.Pattern"] = None,
    topogram_re: Optional["re.Pattern"] = None,
) -> Tuple[List[SeriesCandidate], List[SkippedSeries]]:
    """Apply all cheap filters and return the surviving and rejected series.

    Args:
        series_dirs: Candidate DICOM series directories to filter.
        skip_existing_uids: Authoritative resume set, stable across runs.
        skip_existing_paths: Fast-path hint; skip without reading the DICOM
            header when a series_dir matches a last-known path from state.
        id_extractor: Extracts the patient ID from each series path.
        tp_extractor: Extracts the timepoint label from each series path.
        reformat_re: Overrides the module-level reformat-pattern default.
        topogram_re: Overrides the module-level topogram-pattern default.

    Returns:
        A `(candidates, skipped)` tuple of surviving and rejected series.
    """
    skip_existing_uids  = skip_existing_uids  or set()
    skip_existing_paths = skip_existing_paths or set()

    candidates: List[SeriesCandidate] = []
    skipped:    List[SkippedSeries]   = []

    series_dirs = list(series_dirs)
    for sd in tqdm(
        series_dirs,
        desc="Filtering series",
        unit=" series",
        mininterval=10.0,
        total=len(series_dirs),
    ):
        if str(sd) in skip_existing_paths:
            continue

        pid = id_extractor(sd)
        tp  = tp_extractor(sd)

        ok, reason = folder_name_filter(sd.name, reformat_re, topogram_re)
        if not ok:
            skipped.append(SkippedSeries(sd, pid, tp, reason))
            continue

        tags = read_series_tags(sd)
        if not tags:
            skipped.append(SkippedSeries(sd, pid, tp, "could not read DICOM header"))
            continue

        uid = tags.get("SeriesInstanceUID", "")
        if uid and uid in skip_existing_uids:
            continue

        for fn in (is_ct_modality, is_axial, thickness_in_range, is_volume):
            ok, reason = fn(tags)
            if not ok:
                skipped.append(SkippedSeries(sd, pid, tp, reason))
                break
        else:
            try:
                thickness = float(tags.get("SliceThickness", "") or "nan")
            except ValueError:
                thickness = float("nan")
            candidates.append(SeriesCandidate(
                series_dir=sd,
                patient_id=pid,
                timepoint=tp,
                n_slices=int(tags.get("_n_slices", 0)),
                slice_thickness_mm=thickness if thickness == thickness else None,
                series_description=tags.get("SeriesDescription", ""),
                series_uid=tags.get("SeriesInstanceUID", ""),
            ))

    by_uid: dict = {}
    for c in candidates:
        if not c.series_uid:
            by_uid[("__no_uid__", str(c.series_dir))] = c
            continue
        prev = by_uid.get(c.series_uid)
        if prev is None:
            by_uid[c.series_uid] = c
            continue
        better = (
            c if (c.n_slices, -len(str(c.series_dir)))
                 > (prev.n_slices, -len(str(prev.series_dir)))
            else prev
        )
        worse = prev if better is c else c
        by_uid[c.series_uid] = better
        skipped.append(SkippedSeries(
            worse.series_dir, worse.patient_id, worse.timepoint,
            f"duplicate SeriesInstanceUID (kept {better.series_dir})",
        ))

    deduped = list(by_uid.values())
    if len(deduped) < len(candidates):
        logger.info(
            "Scanner: deduped %d → %d candidates by SeriesInstanceUID "
            "(%d duplicate paths)",
            len(candidates), len(deduped), len(candidates) - len(deduped),
        )
    return deduped, skipped
