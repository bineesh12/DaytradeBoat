from daytrading.config import Settings, StrategyConfig


def test_strategy_config_loads_tunables_from_env(monkeypatch):
    monkeypatch.setenv("DAYTRADING_HOT_WATCH_MAX_SYMBOLS", "12")
    monkeypatch.setenv("DAYTRADING_CANDIDATE_HYDRATE_QUEUE_MAX", "77")
    monkeypatch.setenv("DAYTRADING_TIMED_ENTRY_ANCHOR_TTL_SEC", "123")

    cfg = StrategyConfig.from_env()

    assert cfg.hot_watch_max_symbols == 12
    assert cfg.candidate_hydrate_queue_max == 77
    assert cfg.timed_entry_anchor_ttl_sec == 123


def test_settings_exposes_strategy_config_without_breaking_legacy_attrs(monkeypatch):
    monkeypatch.setenv("DAYTRADING_HOT_WATCH_MIN_SCORE", "0.42")
    monkeypatch.setenv("DAYTRADING_FAST_SCAN_PROCESS_MAX", "31")

    settings = Settings()

    assert settings.strategy.hot_watch_min_score == 0.42
    assert settings.hot_watch_min_score == 0.42
    assert settings.fast_scan_process_max == 31
