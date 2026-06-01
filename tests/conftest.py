from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_ml_runtime_files(tmp_path, monkeypatch):
    """Keep tests from appending to live/server ML JSONL datasets."""
    ml_dir = tmp_path / "ml"

    try:
        from daytrading.ml import data_collector
        monkeypatch.setattr(data_collector, "_DATA_DIR", ml_dir)
        monkeypatch.setattr(
            data_collector,
            "_CANDIDATES_FILE",
            ml_dir / "entry_candidates.jsonl",
        )
    except Exception:
        pass

    try:
        from daytrading.ml import monitor
        monkeypatch.setattr(monitor, "_DATA_DIR", ml_dir)
        monkeypatch.setattr(monitor, "_SHADOW_FILE", ml_dir / "shadow_results.jsonl")
    except Exception:
        pass

    try:
        from daytrading.ml import shadow_collector
        monkeypatch.setattr(shadow_collector, "DATA_DIR", ml_dir)
        monkeypatch.setattr(
            shadow_collector,
            "MISSED_FILE",
            ml_dir / "missed_opportunities.jsonl",
        )
        monkeypatch.setattr(
            shadow_collector,
            "PULLBACK_FILE",
            ml_dir / "pullback_candidates.jsonl",
        )
        monkeypatch.setattr(
            shadow_collector,
            "EXIT_FILE",
            ml_dir / "exit_snapshots.jsonl",
        )
        monkeypatch.setattr(
            shadow_collector,
            "EXECUTION_FILE",
            ml_dir / "execution_quality.jsonl",
        )
    except Exception:
        pass
