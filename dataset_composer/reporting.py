"""Optional Weights & Biases reporting for the dataset composer.

Authenticate by setting `WANDB_API_KEY` in the environment (or passing
`api_key` to `init_wandb`). To disable, set `enabled=False` or
leave `WANDB_API_KEY` unset.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class _NullReporter:
    """No-op reporter used when wandb is disabled or unavailable."""

    enabled = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def summary(self, data: Dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        yield


class WandBReporter:
    """Wrapper around an active wandb run.

    Args:
        run: The active wandb run object to wrap.
    """

    enabled = True

    def __init__(self, run: Any) -> None:
        import wandb  # noqa: F401  (already imported by init_wandb)
        self._run = run
        self._step = 0

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a metrics dict to the active wandb run.

        Args:
            data: Metric name/value pairs to log.
            step: Explicit step index; defaults to an internal counter.
        """
        if step is not None:
            self._step = step
        try:
            self._run.log(data, step=self._step)
            self._step += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb.log failed: %s", exc)

    def summary(self, data: Dict[str, Any]) -> None:
        """Write run-level summary fields.

        Args:
            data: Summary field name/value pairs to write.
        """
        try:
            for k, v in data.items():
                self._run.summary[k] = v
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb.summary failed: %s", exc)

    def finish(self) -> None:
        """Mark the wandb run as finished."""
        try:
            self._run.finish()
        except Exception:  # noqa: BLE001
            pass

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a stage and write the duration to the run summary.

        Stage timings fire once per run.

        Args:
            name: Stage name, used as the summary key prefix.
        """
        t0 = time.time()
        try:
            yield
        finally:
            self.summary({f"stage/{name}_seconds": round(time.time() - t0, 2)})


def init_wandb(
    enabled: bool = False,
    project: str = "pdac-dataset-composer",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    tags: Optional[list] = None,
):
    """Initialise a wandb run, or return a no-op reporter.

    Args:
        enabled: Whether to attempt wandb initialisation at all.
        project: wandb project name.
        run_name: Optional explicit run name.
        config: Config dict recorded on the run.
        api_key: Explicit API key; defaults to `WANDB_API_KEY`.
        tags: Tags applied to the run.

    Returns:
        A `WandBReporter` when successfully initialised, otherwise a
        no-op `_NullReporter`. The reporter is always returned.
    """
    if not enabled:
        return _NullReporter()

    key = api_key or os.environ.get("WANDB_API_KEY")
    if not key:
        logger.warning("wandb enabled but WANDB_API_KEY not set — disabling.")
        return _NullReporter()

    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning("wandb enabled but the package is not installed — disabling.")
        return _NullReporter()

    try:
        wandb.login(key=key, relogin=False)
        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags,
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb.init failed (%s) — disabling.", exc)
        return _NullReporter()

    logger.info("wandb run initialised: %s", run.url if run is not None else "<unknown>")
    return WandBReporter(run)
