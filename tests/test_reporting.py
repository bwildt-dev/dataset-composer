"""Unit tests for reporting.py: _NullReporter, init_wandb fallback paths."""

from __future__ import annotations

import sys
import types

from dataset_composer.reporting import _NullReporter, init_wandb


def test_null_reporter_is_a_safe_no_op():
    rep = _NullReporter()
    rep.log({"a": 1})
    rep.summary({"b": 2})
    rep.finish()
    with rep.stage("anything"):
        pass
    assert rep.enabled is False


def test_init_wandb_disabled_returns_null_reporter():
    rep = init_wandb(enabled=False)
    assert isinstance(rep, _NullReporter)


def test_init_wandb_enabled_without_api_key_returns_null_reporter(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    rep = init_wandb(enabled=True)
    assert isinstance(rep, _NullReporter)


def test_init_wandb_enabled_but_package_missing_returns_null_reporter(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # forces ImportError on `import wandb`
    rep = init_wandb(enabled=True, api_key="fake-key")
    assert isinstance(rep, _NullReporter)


def test_init_wandb_success_wraps_run_in_wandbreporter(monkeypatch):
    fake_run = types.SimpleNamespace(url="https://wandb.ai/x/y", summary={}, log=lambda *a, **k: None,
                                      finish=lambda: None)
    fake_wandb = types.SimpleNamespace(
        login=lambda **kw: None,
        init=lambda **kw: fake_run,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    rep = init_wandb(enabled=True, api_key="fake-key", project="p", run_name="r")

    assert rep.enabled is True
    rep.log({"x": 1})
    rep.summary({"y": 2})
    assert fake_run.summary["y"] == 2
    rep.finish()


def test_init_wandb_init_failure_falls_back_to_null_reporter(monkeypatch):
    fake_wandb = types.SimpleNamespace(
        login=lambda **kw: None,
        init=lambda **kw: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    rep = init_wandb(enabled=True, api_key="fake-key")
    assert isinstance(rep, _NullReporter)


def test_wandbreporter_stage_records_duration(monkeypatch):
    fake_run = types.SimpleNamespace(summary={}, log=lambda *a, **k: None, finish=lambda: None,
                                      url="https://wandb.ai/x/y")
    fake_wandb = types.SimpleNamespace(login=lambda **kw: None, init=lambda **kw: fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    rep = init_wandb(enabled=True, api_key="fake-key")
    with rep.stage("mystage"):
        pass
    assert "stage/mystage_seconds" in fake_run.summary


def test_wandbreporter_swallows_backend_exceptions(monkeypatch):
    class _RaisingSummary:
        def __setitem__(self, key, value):
            raise RuntimeError("network down")

    class _BadRun:
        url = "https://wandb.ai/x/y"
        summary = _RaisingSummary()

        def log(self, *a, **k):
            raise RuntimeError("network down")

        def finish(self):
            raise RuntimeError("network down")

    fake_wandb = types.SimpleNamespace(login=lambda **kw: None, init=lambda **kw: _BadRun())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    rep = init_wandb(enabled=True, api_key="fake-key")
    # None of these must propagate — reporting is best-effort, never fatal.
    rep.log({"a": 1}, step=3)
    rep.summary({"b": 2})
    rep.finish()
