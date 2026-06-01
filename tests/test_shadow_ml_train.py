from __future__ import annotations

import json

import pytest

from daytrading.ml import shadow_train as st


def test_shadow_train_skips_with_too_few_samples(tmp_path, monkeypatch):
    data_file = tmp_path / "missed.jsonl"
    model_file = tmp_path / "missed_model.json"
    data_file.write_text(json.dumps({"features": {"x": 1}, "label": 1}) + "\n")
    monkeypatch.setitem(st.DATASETS, "missed_opportunity", (data_file, model_file))

    trained = st.train_shadow_model("missed_opportunity", min_samples=5)

    assert trained is False
    assert not model_file.exists()


def test_shadow_train_writes_model_with_enough_samples(tmp_path, monkeypatch):
    try:
        __import__("xgboost")
    except ImportError:
        pytest.skip("xgboost not installed")

    data_file = tmp_path / "pullback.jsonl"
    model_file = tmp_path / "pullback_model.json"
    rows = []
    for i in range(60):
        label = 1 if i % 2 else 0
        rows.append({
            "features": {"momentum_5bar_pct": float(i), "spread_pct": float(i % 3)},
            "price": 5.0 + i / 100,
            "label": label,
        })
    data_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setitem(st.DATASETS, "pullback_entry", (data_file, model_file))
    monkeypatch.setattr(st, "MODEL_DIR", tmp_path)

    trained = st.train_shadow_model("pullback_entry", min_samples=50)

    assert trained is True
    assert model_file.exists()
    meta = json.loads(model_file.with_suffix(".meta.json").read_text())
    assert meta["samples"] == 60
    assert "momentum_5bar_pct" in meta["feature_names"]
