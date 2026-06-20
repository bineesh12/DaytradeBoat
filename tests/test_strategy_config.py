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
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_CYCLE_ENABLED", "true")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_WINDOW_SEC", "240")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_SCALP_COOLDOWN_SEC", "180")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_ENABLED", "true")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_MAX_ENTRIES", "4")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_WIN_COOLDOWN_SEC", "12")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_LOSS_COOLDOWN_SEC", "75")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_MAX_HOLD_SEC", "40")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_REWARD_RISK", "1.2")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_STOP_AFTER_GIVEBACK", "true")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_MAX_GIVEBACK", "35")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_DAILY_LOSS_STOP", "45")
    monkeypatch.setenv("DAYTRADING_MOMENTUM_BURST_HIT_RUN_END_ET", "10:45")
    monkeypatch.setenv("DAYTRADING_WARRIOR_SQUEEZE_ENABLED", "true")
    monkeypatch.setenv("DAYTRADING_WARRIOR_SQUEEZE_MIN_RECLAIM_PRICE", "3.50")
    monkeypatch.setenv("DAYTRADING_WARRIOR_SQUEEZE_STARTER_SIZE_FACTOR", "0.25")
    monkeypatch.setenv("DAYTRADING_WARRIOR_SQUEEZE_MAX_ENTRIES", "4")
    monkeypatch.setenv("DAYTRADING_WARRIOR_SQUEEZE_WIN_COOLDOWN_SEC", "2.5")
    monkeypatch.setenv("DAYTRADING_WARRIOR_SQUEEZE_REWARD_RISK", "2.25")
    monkeypatch.setenv("DAYTRADING_WARRIOR_SQUEEZE_ADD_REWARD_RISK", "0.75")

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
    assert cfg.momentum_burst_cycle_enabled is True
    assert cfg.momentum_burst_window_sec == 240
    assert cfg.momentum_burst_scalp_cooldown_sec == 180
    assert cfg.momentum_burst_hit_run_enabled is True
    assert cfg.momentum_burst_hit_run_max_entries == 4
    assert cfg.momentum_burst_hit_run_win_cooldown_sec == 12
    assert cfg.momentum_burst_hit_run_loss_cooldown_sec == 75
    assert cfg.momentum_burst_hit_run_max_hold_sec == 40
    assert cfg.momentum_burst_hit_run_reward_risk == 1.2
    assert cfg.momentum_burst_hit_run_stop_after_giveback is True
    assert cfg.momentum_burst_hit_run_max_giveback == 35
    assert cfg.momentum_burst_hit_run_daily_loss_stop == 45
    assert cfg.momentum_burst_hit_run_end_et == "10:45"
    assert cfg.warrior_squeeze_enabled is True
    assert cfg.warrior_squeeze_min_reclaim_price == 3.50
    assert cfg.warrior_squeeze_starter_size_factor == 0.25
    assert cfg.warrior_squeeze_max_entries == 4
    assert cfg.warrior_squeeze_win_cooldown_sec == 2.5
    assert cfg.warrior_squeeze_reward_risk == 2.25
    assert cfg.warrior_squeeze_add_reward_risk == 0.75


def test_momentum_burst_hit_run_defaults_to_one_entry_and_giveback_stop() -> None:
    cfg = StrategyConfig()

    assert cfg.momentum_burst_hit_run_max_entries == 1
    assert cfg.momentum_burst_hit_run_stop_after_giveback is True
    assert cfg.warrior_squeeze_max_entries == 3
    assert cfg.warrior_squeeze_win_cooldown_sec == 10.0
    assert cfg.warrior_squeeze_reward_risk == 3.0
    assert cfg.warrior_squeeze_add_reward_risk == 1.0


def test_settings_exposes_strategy_config_without_breaking_legacy_attrs(monkeypatch):
    monkeypatch.setenv("DAYTRADING_HOT_WATCH_MIN_SCORE", "0.42")
    monkeypatch.setenv("DAYTRADING_FAST_SCAN_PROCESS_MAX", "31")

    settings = Settings()

    assert settings.strategy.hot_watch_min_score == 0.42
    assert settings.hot_watch_min_score == 0.42
    assert settings.fast_scan_process_max == 31
