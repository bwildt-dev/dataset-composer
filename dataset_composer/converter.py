"""Wrapper around dcm2niix.

Converts a series directory to a single .nii.gz, returning the path or an
error message.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of one dcm2niix conversion attempt.

    Attributes:
        success: Whether conversion produced a usable NIfTI.
        nifti: Path to the primary output NIfTI, or None on failure.
        stderr: Captured error text, empty on success.
    """

    success:   bool
    nifti:     Optional[Path]
    stderr:    str = ""

    def __bool__(self) -> bool:
        return self.success


def find_dcm2niix(explicit: Optional[Path] = None) -> Path:
    """Return the dcm2niix executable, or raise FileNotFoundError.

    Args:
        explicit: Explicit binary path; if unset, searches PATH.

    Raises:
        FileNotFoundError: If `explicit` doesn't exist, or no dcm2niix is
            found on PATH.
    """
    if explicit:
        explicit = Path(explicit)
        if not explicit.exists():
            raise FileNotFoundError(f"dcm2niix not at {explicit}")
        return explicit
    found = shutil.which("dcm2niix")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "dcm2niix not found on PATH. Pass --dcm2niix or load the appropriate module."
    )


def convert_series(
    series_dir:   Path,
    output_dir:   Path,
    filename_stem: str,
    dcm2niix:     Path,
    timeout_s:    int = 600,
) -> ConversionResult:
    """Convert one DICOM series to NIfTI via dcm2niix.

    Args:
        series_dir: DICOM series directory to convert.
        output_dir: Directory dcm2niix writes into.
        filename_stem: Base filename for the output.
        dcm2niix: Path to the dcm2niix executable.
        timeout_s: Subprocess timeout, in seconds.

    Returns:
        A `ConversionResult`. When dcm2niix splits a series into multiple
        files, the largest `filename_stem*.nii.gz` in
        `output_dir` is returned as the primary output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sanitized_stem = re.sub(r"_+", "_", filename_stem)
    # Subtract a small slack so we don't miss a file whose mtime rounds down.
    start_time = time.time() - 2.0

    cmd = [
        str(dcm2niix),
        "-z", "y",
        "-f", filename_stem,
        "-o", str(output_dir),
        "-w", "1",          # overwrite existing files
        str(series_dir),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ConversionResult(False, None, stderr=f"timeout after {timeout_s}s")

    if proc.returncode != 0:
        return ConversionResult(False, None, stderr=proc.stderr.strip()[:500])

    # Look for files matching our stem (original or sanitized form). Then fall
    # back to anything modified during this run.
    produced: List[Path] = sorted(
        set(output_dir.glob(f"{filename_stem}*.nii.gz"))
        | set(output_dir.glob(f"{sanitized_stem}*.nii.gz"))
    )
    if not produced:
        produced = sorted(
            p for p in output_dir.glob("*.nii.gz")
            if p.stat().st_mtime >= start_time
        )
    if not produced:
        # dcm2niix often exits 0 while skipping a series; surface its own
        # explanation (stdout) so the failure log is actionable.
        diag = (proc.stdout or "").strip().splitlines()
        tail = " | ".join(diag[-4:])[:500] if diag else "(no stdout)"
        return ConversionResult(
            False, None,
            stderr=f"dcm2niix returned 0 but no output found in {output_dir} :: {tail}",
        )
    if len(produced) > 1:
        logger.debug("dcm2niix produced %d files, keeping largest:", len(produced))
        for p in produced:
            logger.debug("  %s (%.1f MB)", p.name, p.stat().st_size / 1e6)
    primary = max(produced, key=lambda p: p.stat().st_size)
    return ConversionResult(True, primary, stderr="")
