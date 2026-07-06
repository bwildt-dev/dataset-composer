"""Parallel recursive archive extraction.

Walks the source tree once to seed an initial work set, then extracts archives
concurrently using a thread pool.

Each extracted archive is marked with a `<name>.zip.extracted` sentinel so
re-runs skip completed work. Optionally deletes the source archive after a
successful extraction.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

_JUNK_PREFIXES = ("__MACOSX/", ".DS_Store")


@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting one archive.

    Attributes:
        archive: Source archive that was extracted.
        target: Directory the archive's contents were extracted into.
        n_members: Number of files extracted.
    """

    archive:   Path
    target:    Path
    n_members: int


def _marker_for(archive: Path) -> Path:
    return archive.with_suffix(archive.suffix + ".extracted")


def _is_junk(name: str) -> bool:
    return name.startswith(_JUNK_PREFIXES) or name.endswith("/.DS_Store")


_SYSTEM_UNZIP: Optional[str] = shutil.which("unzip")

_SYSTEM_7Z: Optional[str] = (
    shutil.which("7zz")
    or shutil.which("7z")
    or shutil.which("7za")
)


def _extract_with_system_unzip(archive: Path, target: Path) -> Optional[int]:
    """Fallback to the system `unzip` binary for compression methods stdlib can't handle.

    Args:
        archive: Archive to extract.
        target: Directory to extract into.

    Returns:
        The number of files extracted, or None on failure.
    """
    if _SYSTEM_UNZIP is None:
        logger.error("System `unzip` binary not on PATH — cannot fall back for %s",
                     archive)
        return None
    try:
        proc = subprocess.run(
            [_SYSTEM_UNZIP, "-q", "-o", str(archive), "-d", str(target)],
            capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        logger.error("system unzip timeout on %s", archive)
        return None
    if proc.returncode not in (0, 1):   # 1 = warnings
        logger.error("system unzip failed on %s (rc=%d): %s",
                     archive, proc.returncode, proc.stderr.strip()[:300])
        return None
    # Count extracted files
    n = sum(1 for _ in target.rglob("*") if _.is_file())
    return n


def _extract_with_7z(archive: Path, target: Path) -> Optional[int]:
    """Last-resort fallback using `7z`/`7zz`.

    Args:
        archive: Archive to extract.
        target: Directory to extract into.

    Returns:
        The number of files extracted, or None on failure.
    """
    if _SYSTEM_7Z is None:
        logger.error(
            "No 7z binary found (tried 7zz, 7z, 7za) — cannot extract %s.",
            archive,
        )
        return None
    try:
        proc = subprocess.run(
            [_SYSTEM_7Z, "x", "-aoa", f"-o{target}", str(archive)],
            capture_output=True, text=True, timeout=7200,
        )
    except subprocess.TimeoutExpired:
        logger.error("7z timeout on %s", archive)
        return None
    if proc.returncode != 0:
        logger.error("7z failed on %s (rc=%d): %s",
                     archive, proc.returncode, proc.stderr.strip()[:300])
        return None
    n = sum(1 for _ in target.rglob("*") if _.is_file())
    return n


def _extract_one(
    archive: Path,
    delete_after: bool = False,
) -> Optional[ExtractionResult]:
    """Extract one archive into `<parent>/<stem>/` and mark it done.

    Tries stdlib `zipfile` first; falls back to the system `unzip` binary.

    Args:
        archive: Archive to extract.
        delete_after: Delete the archive after successful extraction.

    Returns:
        The extraction result, or None on permanent failure.
    """
    target = archive.parent / archive.stem
    target.mkdir(parents=True, exist_ok=True)

    n: Optional[int] = None
    try:
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.infolist() if not _is_junk(m.filename)]
            zf.extractall(target, members=members)
            n = len(members)
    except zipfile.BadZipFile as exc:
        logger.error("Bad zip %s: %s — skipping (no marker written)", archive, exc)
        return None
    except NotImplementedError as exc:
        logger.warning("stdlib zipfile can't handle %s (%s) — falling back to system unzip",
                       archive.name, exc)
        n = _extract_with_system_unzip(archive, target)
        if n is None:
            logger.warning("system unzip failed on %s — trying 7z", archive.name)
            n = _extract_with_7z(archive, target)
        if n is None:
            return None
    except Exception as exc:
        logger.error("Failed to extract %s: %s", archive, exc)
        return None

    _marker_for(archive).touch()
    if delete_after:
        try:
            archive.unlink()
        except OSError as exc:
            logger.warning("Could not delete %s after extraction: %s", archive, exc)

    return ExtractionResult(archive=archive, target=target, n_members=n)


def _scan_for_new_zips(root: Path, seen: Set[Path]) -> List[Path]:
    """Return pending zips under `root` that are not in `seen` and not done.

    Args:
        root: Directory tree to scan for `*.zip` files.
        seen: Archives already accounted for; mutated in place with any
            newly-seen archives (pending or already done).
    """
    out: List[Path] = []
    for z in root.rglob("*.zip"):
        if z in seen:
            continue
        if _marker_for(z).exists():
            seen.add(z)
            continue
        out.append(z)
        seen.add(z)
    return out


def extract_all_zips(
    source_dir:   Path,
    delete_after: bool = False,
    n_workers:    Optional[int] = None,
    max_archives: int = 100_000,
) -> List[ExtractionResult]:
    """Recursively extract all `*.zip` archives under `source_dir` in parallel.

    Args:
        source_dir: Directory tree to extract archives under.
        delete_after: Delete each archive after successful extraction.
        n_workers: Worker count; defaults to `min(os.cpu_count(), 8)`.
        max_archives: Hard cap per call, guarding against zip-bomb nesting.

    Returns:
        One `ExtractionResult` per successfully extracted archive.

    Raises:
        FileNotFoundError: If `source_dir` is not a directory.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {source_dir}")

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    seen: Set[Path] = set()
    initial = _scan_for_new_zips(source_dir, seen)
    if not initial:
        logger.info("Unzip: no archives under %s", source_dir)
        return []

    logger.info("Unzip: starting (%d initial archive(s), %d worker(s))",
                len(initial), n_workers)
    results: List[ExtractionResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending = {
            pool.submit(_extract_one, z, delete_after): z
            for z in initial
        }

        while pending:
            if len(results) >= max_archives:
                logger.warning("Unzip: hit max_archives=%d; stopping.", max_archives)
                for f in pending:
                    f.cancel()
                break

            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                archive = pending.pop(future)
                try:
                    res = future.result()
                except Exception as exc:
                    logger.error("Worker raised on %s: %s", archive, exc)
                    continue
                if res is None:
                    continue

                results.append(res)
                logger.info("  extracted %s → %s (%d members)",
                            archive.name, res.target, res.n_members)

                # Only scan the freshly-extracted subtree for nested zips.
                for nested in _scan_for_new_zips(res.target, seen):
                    pending[pool.submit(_extract_one, nested, delete_after)] = nested

    logger.info("Unzip: %d archive(s) extracted.", len(results))
    return results
