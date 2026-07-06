"""Body-region and contrast-phase classifier backed by BOA (boa_contrast)."""

from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Classification:
    """Result of classifying one NIfTI volume.

    Attributes:
        body_region: The matched `RegionSpec` label, or `"other"`.
        phase: A lowercased BOA class (`non_contrast`, `arterial`, `venous`,
            `pulmonary_arterial`, `urographic`); `"skipped_region"` if the
            body-region check rejected the scan before BOA ran; or
            `"boa_failed"` if BOA returned an empty prediction.
        body_confidence: Minimum voxel-count confidence across required
            structures.
        phase_confidence: BOA's probability output for the predicted phase.
        raw: The backend's full prediction dict, for debugging.
    """

    body_region:      str
    phase:            str
    body_confidence:  float
    phase_confidence: float
    raw:              dict


class SeriesClassifier(ABC):
    """Classify NIfTI volumes for body region + contrast phase."""

    @abstractmethod
    def classify(self, nifti_path: Path) -> Classification:
        """Run inference on a single NIfTI volume.

        Args:
            nifti_path: Path to the NIfTI volume to classify.
        """


class DummyClassifier(SeriesClassifier):
    """Always returns `(abdomen, venous)` with zero confidence."""

    def classify(self, nifti_path: Path) -> Classification:
        return Classification(
            body_region="abdomen",
            phase="venous",
            body_confidence=0.0,
            phase_confidence=0.0,
            raw={"backend": "dummy", "path": str(nifti_path)},
        )


@dataclass
class RegionSpec:
    """One body-region acceptance rule.

    A scan is assigned `label` when all structures meet their minimum voxel
    count. Multiple entries are tried in order; the first match wins.

    Attributes:
        label: The body-region label assigned on a match.
        structures: TotalSegmentator mask names (without `.nii.gz`) mapped to
            minimum voxel counts, e.g. `{"liver": 5000, "pancreas": 2000}`.
    """
    label:      str
    structures: Dict[str, int]


_CONF_SATURATION_VOXELS = 10_000


class BOAClassifier(SeriesClassifier):
    """UMEssen boa_contrast phase classifier with multi-region structure check.

    Scans matching no region get `body_region="other"` and are dropped by
    the pipeline.

    Args:
        device: Passed as `device_id` to `compute_segmentation`.
        regions: Ordered list of `RegionSpec` entries; the first whose
            structures all exceed their voxel thresholds wins.
    """

    def __init__(
        self,
        device: str = "cuda",
        regions: Optional[List[RegionSpec]] = None,
    ) -> None:
        self.device = device
        self.regions: List[RegionSpec] = (
            regions if regions is not None
            else [RegionSpec("abdomen", {"liver": 5000})]
        )
        self._boa_imported = False

        if not self.regions:
            raise ValueError("regions must contain at least one RegionSpec.")

    def _ensure_importable(self) -> None:
        if self._boa_imported:
            return
        try:
            import boa_contrast  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Original error: " + str(exc)) from exc
        self._boa_imported = True

    def _count_voxels(self, seg_folder: Path, structure: str) -> int:
        """Count non-zero voxels in `<seg_folder>/<structure>.nii.gz`.

        Args:
            seg_folder: Directory of TotalSegmentator output masks.
            structure: Mask name to count, without the `.nii.gz` suffix.
        """
        import nibabel as nib  # type: ignore

        mask_path = seg_folder / f"{structure}.nii.gz"
        if not mask_path.exists():
            logger.warning("Structure mask not found: %s", mask_path)
            return 0
        try:
            arr = nib.load(mask_path).get_fdata(dtype=np.float32)
            return int((arr > 0).sum())
        except Exception as exc:
            logger.warning("Could not read mask %s: %s", mask_path, exc)
            return 0

    def _check_regions(self, seg_folder: Path) -> tuple[str, float]:
        """Return `(body_region, body_confidence)` for the first matching RegionSpec.

        Args:
            seg_folder: Directory of TotalSegmentator output masks.

        Returns:
            A `(body_region, body_confidence)` pair, where `body_confidence`
            is the minimum confidence across structures in the matched
            region. Returns `("other", 0.0)` if no region matches.
        """
        for region in self.regions:
            confidences: List[float] = []
            matched = True
            for structure, min_voxels in region.structures.items():
                n = self._count_voxels(seg_folder, structure)
                if n < min_voxels:
                    logger.debug(
                        "Region '%s': structure '%s' has %d voxels (< %d) — no match",
                        region.label, structure, n, min_voxels,
                    )
                    matched = False
                    break
                conf = min(1.0, n / _CONF_SATURATION_VOXELS)
                logger.debug("  '%s': %d voxels → conf=%.2f", structure, n, conf)
                confidences.append(conf)

            if matched and confidences:
                body_conf = min(confidences)
                logger.info("Region matched: '%s'  body_confidence=%.2f",
                            region.label, body_conf)
                return region.label, body_conf

        logger.info("No region matched — scan will be dropped")
        return "other", 0.0

    def _infer(self, nifti_path: Path, seg_folder: Path) -> dict:
        from boa_contrast import predict  # type: ignore
        result = predict(ct_path=str(nifti_path), segmentation_folder=str(seg_folder))
        return result if isinstance(result, dict) else {"phase_ensemble_predicted_class": str(result)}

    def classify(self, nifti_path: Path) -> Classification:
        self._ensure_importable()

        with tempfile.TemporaryDirectory(prefix="boa_seg_") as _tmp:
            seg_folder = Path(_tmp)

            from boa_contrast import compute_segmentation  # type: ignore
            logger.info("Running TotalSegmentator for %s …", nifti_path.name)
            compute_segmentation(
                ct_path=str(nifti_path),
                segmentation_folder=str(seg_folder),
                device_id=self.device,
                compute_with_docker=False,
            )

            body_region, body_conf = self._check_regions(seg_folder)

            if body_region == "other":
                logger.info(
                    "%s: no region matched → body_region='other', "
                    "skipping phase prediction", nifti_path.name,
                )
                return Classification(
                    body_region="other",
                    phase="skipped_region",
                    body_confidence=0.0,
                    phase_confidence=0.0,
                    raw={"regions_checked": [r.label for r in self.regions],
                         "skipped_phase_prediction": True},
                )

            raw = self._infer(nifti_path, seg_folder)

        phase_raw = str(raw.get("phase_ensemble_predicted_class", "")).strip()
        phase = phase_raw.lower() if phase_raw else "boa_failed"

        # BOA gives a probas array + prediction index rather than a scalar confidence.
        try:
            pred_idx = int(raw.get("phase_ensemble_prediction", -1))
            probas   = raw.get("phase_ensemble_probas")
            phase_conf = float(probas[pred_idx]) if (probas is not None and pred_idx >= 0) else 0.0
        except (TypeError, IndexError, ValueError):
            phase_conf = 0.0

        return Classification(
            body_region=body_region,
            phase=phase,
            body_confidence=body_conf,
            phase_confidence=phase_conf,
            raw=raw,
        )


def build_classifier(
    name: str,
    device: str = "cuda",
    regions: Optional[List[RegionSpec]] = None,
) -> SeriesClassifier:
    """Instantiate a classifier by name.

    Args:
        name: Backend name, `"boa"` or `"dummy"`.
        device: Device string passed through to `BOAClassifier`.
        regions: Region rules passed through to `BOAClassifier`; defaults to
            a single abdomen spec requiring `liver >= 5000` voxels. Ignored
            for `DummyClassifier`.

    Raises:
        ValueError: If `name` is not `"boa"` or `"dummy"`.
    """
    name = name.lower()
    if name == "boa":
        return BOAClassifier(device=device, regions=regions)
    if name == "dummy":
        return DummyClassifier()
    raise ValueError(f"Unknown classifier backend: {name!r}. Choose 'boa' or 'dummy'.")
