from daytrading.config import Settings, StrategyConfig


def test_strategy_config_loads_tunables_from_env(monkeypatch):
    monkeypatch.setenv("DAYTRADING_HOT_WATCH_MAX_SYMBOLS", "12")
    monkeypatch.setenv("DAYTRADING_CANDIDATE_HYDRATE_QUEUE_MAX", "77")
    monkeypatch.setenv("DAYTRADING_TIMED_ENTRY_ANCHOR_TTL_SEC", "123")
    monkeypatch.setenv("DAYTRADING_MISSED_A_PLUS_CHASE_WINDOW_SEC", "900")
    monkeypatch.setenv("DAYTRADING_MISSED_A_PLUS_CHASE_PCT_SUB5", "0.05")
    monkeypatch.setenv("DAYTRADING_MISSED_A_PLUS_CHASE_PCT_5PLUS", "0.04")
    monkeypatch.setenv("DAYTRADING_LEVEL_BREAKOUT_SCOUT_ENABLED", "false")
    monkeypatch.setenv("DAYTRADING_LEVEL_BREAKOUT_SCOUT_MIN_SESSION_MOVE_PCT", "3.5")
    monkeypatch.setenv("DAYTRADING_HOT_WATCH_SETUP_REFRESH_ENABLED", "false")
    monkeypatch.setenv("DAYTRADING_HOT_WATCH_SETUP_REFRESH_MAX_PULLBACK_PCT", "3.25")
    monkeypatch.setenv("DAYTRADING_HOT_WATCH_SETUP_REFRESH_MIN_RECENT_VOLUME", "150000")

    cfg = StrategyConfig.from_env()

    assert cfg.hot_watch_max_symbols == 12
    assert cfg.candidate_hydrate_queue_max == 77
    assert cfg.timed_entry_anchor_ttl_sec == 123
    assert cfg.missed_a_plus_chase_window_sec == 900
    assert cfg.missed_a_plus_chase_pct_sub5 == 0.05
    assert cfg.missed_a_plus_chase_pct_5plus == 0.04
    assert cfg.level_breakout_scout_enabled is False
    assert cfg.level_breakout_scout_min_session_move_pct == 3.5
    assert cfg.hot_watch_setup_refresh_enabled is False
    assert cfg.hot_watch_setup_refresh_max_pullback_pct == 3.25
    assert cfg.hot_watch_setup_refresh_min_recent_volume == 150_000


def test_settings_exposes_strategy_config_without_breaking_legacy_attrs(monkeypatch):
    monkeypatch.setenv("DAYTRADING_HOT_WATCH_MIN_SCORE", "0.42")
    monkeypatch.setenv("DAYTRADING_FAST_SCAN_PROCESS_MAX", "31")

    settings = Settings()

    assert settings.strategy.hot_watch_min_score == 0.42
    assert settings.hot_watch_min_score == 0.42
    assert settings.fast_scan_process_max == 31
