from daytrading.models import BarSource
"""Regression scenarios for access control, broker uncertainty, replay and data."""

import json
import queue
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from daytrading.backtest.broker import BacktestBroker, FillModel
from daytrading.dashboard.hub import DashboardHub
from daytrading.dashboard.server import create_app, start_dashboard
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.execution.pricing import marketable_limit
from daytrading.ml.artifacts import atomic_json, promote_json
from daytrading.ml.validation import TrainingRow, chronological_split, promotion_reason
from daytrading.models import Bar, Order, OrderStatus, PortfolioState, Side
from daytrading.paper import load_paper_config, preflight
from daytrading.runner import AlpacaRunner, TradeEvent
from daytrading.strategy.warrior_ignition_retrain import _runner_label

NOW = datetime(2026, 10, 14, 14, 30, tzinfo=UTC)


def bar(close=10, volume=10000):
    return Bar("TEST", NOW, close, close + 0.1, close - 0.1, close, volume, provenance=BarSource.CANONICAL_INTERVAL)


def test_dashboard_remote_controls_require_credentials(monkeypatch):
    monkeypatch.delenv("DAYTRADING_DASHBOARD_TOKEN", raising=False)
    hub = DashboardHub()
    client = create_app(hub).test_client()
    response = client.post("/api/pause", environ_base={"REMOTE_ADDR": "192.0.2.5"})
    assert response.status_code == 403
    assert not hub.trading_paused


def test_dashboard_token_and_cross_origin_controls(monkeypatch):
    monkeypatch.setenv("DAYTRADING_DASHBOARD_TOKEN", "test-secret")
    hub = DashboardHub()
    client = create_app(hub).test_client()
    assert client.get("/api/snapshot").status_code == 401
    assert (
        client.post("/api/pause", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert (
        client.post(
            "/api/pause",
            headers={
                "Authorization": "Bearer test-secret",
                "Origin": "https://attacker.example",
            },
        ).status_code
        == 403
    )
    assert not hub.trading_paused
    assert (
        client.post(
            "/api/pause", headers={"Authorization": "Bearer test-secret"}
        ).status_code
        == 200
    )
    assert hub.trading_paused


def test_local_dashboard_rejects_cross_site_browser_commands(monkeypatch):
    monkeypatch.delenv("DAYTRADING_DASHBOARD_TOKEN", raising=False)
    client = create_app(DashboardHub()).test_client()
    assert (
        client.post("/api/pause", headers={"Sec-Fetch-Site": "cross-site"}).status_code
        == 403
    )
    assert (
        client.get("/api/snapshot", base_url="http://attacker.example").status_code
        == 403
    )


def test_dashboard_public_bind_fails_before_starting_server(monkeypatch):
    monkeypatch.delenv("DAYTRADING_DASHBOARD_TOKEN", raising=False)
    with pytest.raises(ValueError, match="non-loopback"):
        start_dashboard(DashboardHub(), host="0.0.0.0")


def broker(client):
    instance = AlpacaBroker.__new__(AlpacaBroker)
    instance._client, instance._slippage_guard = client, None
    instance._now = lambda: NOW
    instance._max_wait, instance._poll_interval, instance._cancel_grace_seconds = (
        0.002,
        0.001,
        0.002,
    )
    instance._safety_state()
    return instance


def stop(status="new", filled=0):
    return SimpleNamespace(
        id="old-stop",
        type="stop",
        symbol="TEST",
        side="sell",
        qty="100",
        stop_price="9",
        status=status,
        filled_qty=filled,
        filled_avg_price=9,
    )


def accepted_stop(qty=100, price=9, oid='restored'):
    row=stop();row.id=oid;row.qty=str(qty);row.stop_price=str(price)
    return row


def filled_order(status="filled", qty=20):
    return SimpleNamespace(
        id="entry", status=status, filled_qty=qty, filled_avg_price=10, filled_at=NOW
    )


class APIError(Exception):
    def __init__(self, status_code):
        super().__init__("mock API failure")
        self.status_code = status_code


def test_failed_add_restores_stop_using_fresh_position_quantity():
    client = Mock()
    client.get_orders.side_effect = [[stop()], [], []]
    client.get_all_positions.return_value = [SimpleNamespace(symbol="TEST",qty=100)]
    client.get_order_by_id.return_value = stop("canceled")
    client.submit_order.side_effect = [APIError(422), accepted_stop()]
    instance = broker(client)
    fill, status = instance.submit(
        Order("TEST", Side.BUY, 20, 10), bar(), PortfolioState(cash=10000)
    )
    assert fill is None and status is OrderStatus.REJECTED
    restored = client.submit_order.call_args.kwargs["order_data"]
    assert restored.qty == 100 and restored.stop_price == 9
    assert client.get_all_positions.call_count == 2
    assert instance.get_protective_stop_id("TEST") == "restored"


def test_stop_partial_fill_during_cancel_is_accounted_for():
    client = Mock()
    client.get_orders.side_effect = [[stop()], [], []]
    client.get_order_by_id.return_value = stop('canceled',filled=10)
    client.get_all_positions.return_value = [SimpleNamespace(symbol='TEST',qty=90)]
    client.submit_order.return_value = accepted_stop(90)
    instance = broker(client)
    handler=Mock();instance.set_execution_callbacks(handler,lambda symbol:False)
    fill, status = instance.submit(Order('TEST',Side.BUY,20,10),bar(),PortfolioState(cash=10000))
    assert fill is None and status is OrderStatus.REJECTED
    handler.assert_called_once()
    assert handler.call_args.args[0].order_id == 'old-stop'
    assert client.submit_order.call_args.kwargs['order_data'].qty == 90
    client.submit_order.assert_called_once()


def test_cancel_error_prevents_replacement_order_until_acknowledged():
    client = Mock()
    client.get_orders.return_value = [stop()]
    client.cancel_order_by_id.side_effect = RuntimeError("network")
    instance = broker(client)
    _, status = instance.submit(
        Order("TEST", Side.BUY, 20, 10), bar(), PortfolioState(cash=10000)
    )
    assert status is OrderStatus.NEW and instance._safety_blocked
    client.submit_order.assert_not_called()
    client.cancel_order_by_id.side_effect = None
    client.get_orders.return_value = []
    client.get_order_by_id.return_value = stop("canceled")
    client.get_all_positions.return_value = [SimpleNamespace(symbol="TEST", qty="100")]
    client.submit_order.return_value = accepted_stop()
    assert instance.recover_order_safety()
    assert client.submit_order.call_args.kwargs["order_data"].qty == 100


def test_unknown_order_response_blocks_duplicate_buy():
    client = Mock()
    client.get_orders.return_value = []
    client.submit_order.side_effect = RuntimeError("response lost")
    instance = broker(client)
    _, status = instance.submit(
        Order("TEST", Side.BUY, 20, 10), bar(), PortfolioState(cash=10000)
    )
    assert status is OrderStatus.NEW and instance._safety_blocked
    instance.submit(Order("TEST", Side.BUY, 20, 10), bar(), PortfolioState(cash=10000))
    assert client.submit_order.call_count == 1
    client.get_order_by_client_id.return_value = filled_order()
    assert instance.recover_order_safety()


def test_late_partial_is_not_terminal_until_residual_cancel_confirmed(monkeypatch):
    # Preserve the original 2ms deadline and every safety assertion. Simulated
    # polling time must not depend on the Mac scheduling a real 1ms sleep.
    from daytrading.execution import alpaca_broker as module
    class Clock:
        value = 0.0
        def monotonic(self): return self.value
        def sleep(self, seconds): self.value += seconds
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)
    client = Mock()
    client.get_orders.return_value = []
    client.submit_order.return_value = SimpleNamespace(id="entry")
    client.get_order_by_id.return_value = filled_order("partially_filled", qty=5)
    instance = broker(client)
    instance._max_wait = 0
    fill, status = instance.submit(
        Order("TEST", Side.BUY, 20, 10), bar(), PortfolioState(cash=10000)
    )
    assert fill is None and status is OrderStatus.NEW and instance._safety_blocked
    client.get_order_by_id.return_value = filled_order("canceled", qty=5)
    assert instance.recover_order_safety()


def test_failed_stop_update_does_not_cancel_old_stop():
    client = Mock()
    client.replace_order_by_id.side_effect = RuntimeError("network")
    client.get_order_by_id.return_value = stop()
    client.get_orders.return_value = [stop()]
    client.get_all_positions.return_value = [SimpleNamespace(symbol="TEST",qty=100)]
    instance = broker(client)
    assert instance.replace_protective_stop("TEST", "old-stop", 100, 9.5) == "old-stop"
    client.cancel_order_by_id.assert_not_called()
    assert instance._safety_blocked
    client.replace_order_by_id.side_effect = None
    client.replace_order_by_id.return_value = accepted_stop(100,9.5,"new-stop")
    assert instance.recover_order_safety()
    assert instance.get_protective_stop_id("TEST") == "new-stop"


def test_stop_response_loss_recovers_by_client_id_without_duplicate():
    client = Mock()
    client.submit_order.side_effect = RuntimeError("response lost")
    instance = broker(client)
    assert instance.place_protective_stop("TEST", 100, 9) is None
    request = client.submit_order.call_args.kwargs["order_data"]
    accepted = stop();accepted.id='accepted-stop'
    client.get_order_by_client_id.return_value = accepted
    client.get_orders.return_value = [accepted]
    client.get_all_positions.return_value = [SimpleNamespace(symbol='TEST',qty=100)]
    assert instance.recover_order_safety()
    client.get_order_by_client_id.assert_called_with(request.client_order_id)
    assert client.submit_order.call_count == 1


def test_bad_price_does_not_cancel_protection():
    client = Mock()
    instance = broker(client)
    _, status = instance.submit(
        Order("TEST", Side.BUY, 20, -10), bar(), PortfolioState(cash=10000)
    )
    assert status is OrderStatus.REJECTED
    client.cancel_order_by_id.assert_not_called()
    client.get_orders.assert_not_called()


def test_stale_fallback_bar_does_not_cancel_protection():
    client = Mock()
    instance = broker(client)
    instance._now = lambda: NOW + timedelta(seconds=100)
    _, status = instance.submit(
        Order("TEST", Side.BUY, 20, 10), bar(), PortfolioState(cash=10000)
    )
    assert status is OrderStatus.REJECTED
    client.get_orders.assert_not_called()


def test_replay_unreachable_limit_and_zero_volume_do_not_fill():
    instance = BacktestBroker()
    portfolio = PortfolioState(cash=10000)
    fill, status = instance.submit(Order("TEST", Side.BUY, 20, 9), bar(), portfolio)
    assert fill is None and status is OrderStatus.CANCELLED
    fill, status = instance.submit(
        Order("TEST", Side.BUY, 20, 10), bar(volume=0), portfolio
    )
    assert fill is None and status is OrderStatus.CANCELLED


def test_replay_partial_fill_and_commission_cash_check():
    instance = BacktestBroker(
        FillModel(max_volume_participation=0.1, commission_per_share=0.1)
    )
    order = Order("TEST", Side.BUY, 100, 10)
    fill, status = instance.submit(order, bar(volume=100), PortfolioState(cash=1000))
    assert status is OrderStatus.FILLED and fill.quantity == 10 and fill.commission == 1
    fill, status = instance.submit(order, bar(volume=100), PortfolioState(cash=100))
    assert fill is None and status is OrderStatus.REJECTED


def test_live_and_replay_share_limit_policy():
    client = Mock()
    client.get_orders.return_value = []
    client.submit_order.return_value = SimpleNamespace(id="entry")
    client.get_order_by_id.return_value = filled_order()
    instance = broker(client)
    order = Order("TEST", Side.BUY, 20, 10)
    instance.submit(order, bar(), PortfolioState(cash=10000))
    assert client.submit_order.call_args.kwargs[
        "order_data"
    ].limit_price == marketable_limit(order, bar())


def dated_rows():
    return [
        TrainingRow(
            f"S{i % 3}",
            NOW + timedelta(days=day, seconds=i),
            (float(i),),
            float(i % 2),
            float(1 if i % 2 else -1),
        )
        for day in range(10)
        for i in range(20)
    ]


def test_forward_split_is_sorted_deduplicated_and_session_disjoint():
    rows = dated_rows()
    train, validation, test = chronological_split(list(reversed(rows)) + rows[:10])
    assert sum(map(len, (train, validation, test))) == 200
    assert max(r.ts for r in train) < min(r.ts for r in validation)
    assert max(r.ts for r in validation) < min(r.ts for r in test)
    assert not {r.session for r in train}.intersection(r.session for r in test)
    assert (train, validation, test) == chronological_split(rows)


def test_bad_promotion_or_reused_holdout_is_refused():
    train, validation, test = chronological_split(dated_rows())
    bad = {"selected": 20, "mean_return_after_cost": -1, "total_return_after_cost": -20}
    assert "not positive" in promotion_reason(train, validation, test, bad)
    good = {"selected": 20, "mean_return_after_cost": 1, "total_return_after_cost": 20}
    assert "provenance" in promotion_reason(train, validation, test, good, {})
    assert "overlap" in promotion_reason(
        train, validation, test, good, {"evaluated_through": "2026-10-23"}
    )


def test_ignition_ambiguous_entry_bar_is_not_labeled_a_win():
    ambiguous = SimpleNamespace(high=12, low=9, close=11)
    assert _runner_label([ambiguous], 0, 10, 9) == 0
    target = SimpleNamespace(high=12, low=9.5, close=11)
    assert _runner_label([target], 0, 10, 9) == 1
    incomplete = SimpleNamespace(high=10.5, low=9.5, close=10)
    assert _runner_label([incomplete], 0, 10, 9) is None


def test_atomic_model_version_retains_previous_weights(tmp_path):
    path = tmp_path / "model.json"
    atomic_json(path, {"weights": [1]})
    promote_json(path, {"weights": [2]})
    assert json.loads(path.read_text())["weights"] == [2]
    assert any(
        json.loads(p.read_text())["weights"] == [1]
        for p in (tmp_path / "versions").glob("*.json")
    )


def test_paper_launcher_refuses_live_config_and_placeholder_keys(tmp_path):
    config = tmp_path / ".env.paper"
    config.write_text("DAYTRADING_ALPACA_PAPER=false\n")
    with pytest.raises(ValueError, match="PAPER=true"):
        load_paper_config(config)
    config.write_text(
        "DAYTRADING_ALPACA_PAPER=true\nDAYTRADING_ALPACA_API_KEY=your_key\n"
    )
    with pytest.raises(ValueError, match="API_KEY"):
        load_paper_config(config)


def test_paper_config_clears_other_project_experiments(tmp_path, monkeypatch):
    monkeypatch.setenv("DAYTRADING_WARRIOR_IGNITION", "true")
    config = tmp_path / ".env.paper"
    config.write_text(
        "DAYTRADING_ALPACA_PAPER=true\nDAYTRADING_ALPACA_API_KEY=sample-key\nDAYTRADING_ALPACA_SECRET_KEY=sample-secret\n"
    )
    # isolate environment mutations made by load_paper_config
    with pytest.MonkeyPatch.context() as isolation:
        isolation.setattr("os.environ", dict(__import__("os").environ))
        load_paper_config(config)
        assert "DAYTRADING_WARRIOR_IGNITION" not in __import__("os").environ
        assert __import__("os").environ["DAYTRADING_ALPACA_PAPER"] == "true"


def test_event_overflow_is_counted_without_retaining_dropped_event():
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._event_queue = queue.Queue(maxsize=1)
    first, second = TradeEvent("TEST", None), TradeEvent("OTHER", None)
    assert runner._enqueue_market_event(first)
    assert not runner._enqueue_market_event(second)
    assert runner._event_drop_counts["TradeEvent"] == 1
    assert len(runner._event_enqueued_at) == 1


def test_model_reload_is_atomic_and_invalid_update_keeps_last_valid(
    tmp_path, monkeypatch
):
    from daytrading.strategy import warrior_ignition as module

    payload = json.loads(Path(module._BUNDLED_MODEL_PATH).read_text())
    path = tmp_path / "ignition.json"
    atomic_json(path, payload)
    monkeypatch.setattr(module, "_MODEL_PATH", str(path))
    monkeypatch.setattr(module, "_MODEL_CACHE", None)
    monkeypatch.setattr(module, "_MODEL_SIGNATURE", None)
    old = module.get_model()
    payload["bias"] += 1
    atomic_json(path, payload)
    new = module.get_model()
    assert new.bias == old.bias + 1
    atomic_json(path, {"invalid": True})
    assert module.get_model() is new


def test_free_feed_streams_trade_ticks_and_enforces_symbol_cap(monkeypatch):
    from daytrading.data import alpaca_feed

    monkeypatch.setattr(alpaca_feed, "_ResilientDataStream", Mock())
    stream = alpaca_feed.AlpacaStreamFeed("sample-key", "sample-secret", feed="iex")
    stream.subscribe([f"S{i}" for i in range(30)], trades=True)
    assert stream._trade_filter_symbols == {f"S{i}" for i in range(30)}
    with pytest.raises(ValueError, match="30 streamed"):
        stream.subscribe(["EXTRA"], trades=True)
    assert stream.unsubscribe(["S0"])
    stream.subscribe(["EXTRA"], trades=True)
    assert (
        "EXTRA" in stream._trade_filter_symbols
        and "S0" not in stream._trade_filter_symbols
    )


def test_preflight_uses_paper_endpoint_and_never_submits_orders(monkeypatch):
    from alpaca.data import historical as data_module
    from alpaca.trading import client as trading_module

    trading = Mock()
    trading.get_account.return_value = SimpleNamespace(
        trading_blocked=False, account_blocked=False
    )
    trading.get_all_positions.return_value = []
    trading.get_orders.return_value = []
    trading.get_clock.return_value = SimpleNamespace(is_open=True)
    constructor = Mock(return_value=trading)
    monkeypatch.setattr(trading_module, "TradingClient", constructor)
    data = Mock()
    data.get_stock_latest_quote.return_value = {
        "SPY": SimpleNamespace(bid_price=100, ask_price=100.01, timestamp=NOW)
    }
    monkeypatch.setattr(
        data_module, "StockHistoricalDataClient", Mock(return_value=data)
    )
    values = {
        "DAYTRADING_ALPACA_API_KEY": "sample",
        "DAYTRADING_ALPACA_SECRET_KEY": "sample",
        "DAYTRADING_ALPACA_FEED": "iex",
    }
    assert preflight(values)["paper"] is True
    constructor.assert_called_once_with("sample", "sample", paper=True)
    trading.submit_order.assert_not_called()
    trading.get_all_positions.return_value = [SimpleNamespace(symbol="OTHER")]
    with pytest.raises(ValueError, match="already has positions"):
        preflight(values)


def test_completed_trade_label_matches_bound_candidate_and_all_exit_legs():
    from datetime import datetime

    from daytrading.exits.manager import TrackedPosition
    from daytrading.ml import data_collector as collector
    from daytrading.pipeline.factory import create_scalping_pipeline

    entry_time = datetime.now(UTC) + timedelta(seconds=1)
    first = {
        "ts": (entry_time - timedelta(seconds=1)).isoformat(),
        "symbol": "TEST",
        "passed": True,
        "outcome_pnl": None,
        "features": {"gap_pct": 1},
    }
    collector._CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    collector._CANDIDATES_FILE.write_text(json.dumps(first) + "\n")
    collector.bind_candidate_to_trade("TEST", entry_time)
    newer = {**first, "ts": entry_time.isoformat(), "features": {"gap_pct": 2}}
    with collector._CANDIDATES_FILE.open("a") as file:
        file.write(json.dumps(newer) + "\n")
    pipeline = create_scalping_pipeline(initial_cash=1000)
    tracked = TrackedPosition(
        symbol="TEST",
        side=Side.BUY,
        quantity=100,
        remaining_qty=80,
        entry_price=5,
        entry_ts=entry_time,
    )
    pipeline.exit_manager.track(tracked)
    pipeline.record_realized_exit("TEST", 5, 5.1, 20)
    assert all(
        row["outcome_pnl"] is None
        for row in map(json.loads, collector._CANDIDATES_FILE.read_text().splitlines())
    )
    tracked.remaining_qty = 0
    pipeline.record_realized_exit("TEST", 5, 4.9, 80)
    rows = list(map(json.loads, collector._CANDIDATES_FILE.read_text().splitlines()))
    assert rows[0]["outcome_pct"] == pytest.approx(-1.2)
    assert rows[0]["trade_complete"] is True
    assert rows[1]["outcome_pnl"] is None


def test_entry_training_is_candidate_only_and_does_not_reuse_holdout(
    tmp_path, monkeypatch
):
    from daytrading.ml import data_collector as collector
    from daytrading.ml import train as trainer
    from daytrading.ml.features import FEATURE_NAMES

    rows = []
    for day in range(10):
        for i in range(20):
            outcome = 2 if i % 2 else -2
            rows.append(
                {
                    "ts": (NOW + timedelta(days=day, seconds=i)).isoformat(),
                    "symbol": f"S{i % 3}",
                    "outcome_source": "real_trade",
                    "feed": "iex",
                    "trade_complete": True,
                    "outcome_pct": outcome,
                    "features": {name: float(outcome) for name in FEATURE_NAMES},
                }
            )
    monkeypatch.setattr(collector, "load_training_data", lambda: rows)
    monkeypatch.setattr(trainer, "MODEL_DIR", tmp_path)
    active = tmp_path / "entry_model.json"
    monkeypatch.setattr(trainer, "MODEL_PATH", active)
    assert trainer.train()
    assert not active.exists()
    report = json.loads((tmp_path / "entry_training_report.json").read_text())
    assert report["trained_through"] < report["validation_start"] < report["test_start"]
    assert report["promotion_allowed"]
    assert trainer.train(deploy=True)
    before = active.read_bytes()
    assert trainer.train(deploy=True)
    assert active.read_bytes() == before
    report = json.loads((tmp_path / "entry_training_report.json").read_text())
    assert not report["deployed"]
    assert (
        "overlap" in report["promotion_reason"]
        or "does not beat" in report["promotion_reason"]
    )


def test_entry_training_excludes_shadow_and_incomplete_outcomes(monkeypatch):
    from daytrading.ml import data_collector as collector
    from daytrading.ml import train as trainer

    monkeypatch.setattr(
        collector,
        "load_training_data",
        lambda: [
            {"outcome_source": "shadow_future_price", "trade_complete": True},
            {"outcome_source": "real_trade", "trade_complete": False},
        ],
    )
    assert trainer._load_rows() == []


def test_dashboard_apps_do_not_share_control_state(monkeypatch):
    monkeypatch.delenv("DAYTRADING_DASHBOARD_TOKEN", raising=False)
    first, second = DashboardHub(), DashboardHub()
    first_client = create_app(first).test_client()
    create_app(second)
    assert first_client.post("/api/pause").status_code == 200
    assert first.trading_paused and not second.trading_paused


def test_ignition_session_clock_handles_daylight_savings():
    from daytrading.strategy.warrior_ignition import _PREMKT_OPEN_UTC_MIN, _utc_minutes

    assert _utc_minutes(datetime(2026, 6, 15, 8, tzinfo=UTC)) == _PREMKT_OPEN_UTC_MIN
    assert _utc_minutes(datetime(2026, 12, 15, 9, tzinfo=UTC)) == _PREMKT_OPEN_UTC_MIN


def test_cache_does_not_mix_feed_and_reuses_complete_sparse_session(tmp_path):
    from daytrading.backtest.data_loader import fetch_alpaca_bars_for_day

    timestamp = datetime(2026, 6, 15, 20, tzinfo=UTC)
    cached = {
        "symbol": "TEST",
        "date": "2026-06-15",
        "feed": "iex",
        "complete": True,
        "bars": [
            {
                "symbol": "TEST",
                "ts": timestamp.isoformat(),
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1000,
            }
        ],
    }
    path = tmp_path / "TEST_2026-06-15_1m.json"
    path.write_text(json.dumps(cached))
    feed = SimpleNamespace(
        _feed=SimpleNamespace(value="sip"),
        get_bars=Mock(
            return_value={"TEST": [Bar("TEST", timestamp, 2, 2, 2, 2, 1000, provenance=BarSource.CANONICAL_INTERVAL)]}
        ),
    )
    settings = SimpleNamespace(alpaca_feed="sip")
    assert (
        fetch_alpaca_bars_for_day(
            "TEST", "2026-06-15", cache_dir=str(tmp_path), feed=feed, settings=settings
        )["TEST"][0].close
        == 2
    )
    assert json.loads(path.read_text())["feed"] == "sip"
    fetch_alpaca_bars_for_day(
        "TEST", "2026-06-15", cache_dir=str(tmp_path), feed=feed, settings=settings
    )
    feed.get_bars.assert_called_once()


def test_stop_already_filled_during_recovery_does_not_open_a_short():
    client=Mock();client.submit_order.side_effect=RuntimeError('response lost')
    instance=broker(client);assert instance.place_protective_stop('TEST',100,9) is None
    completed=stop('filled',100);completed.id='accepted-stop'
    client.get_order_by_client_id.return_value=completed
    client.get_order_by_id.return_value=completed
    client.get_all_positions.return_value=[];client.get_orders.return_value=[]
    handler=Mock();instance.set_execution_callbacks(handler,lambda symbol:False)
    assert not instance.recover_order_safety()  # First resolve its unaccounted fill.
    assert instance.recover_order_safety()
    handler.assert_called_once();assert client.submit_order.call_count==1


def test_corrupt_model_rollback_has_no_side_effect(tmp_path, monkeypatch):
    from daytrading.ml import rollback

    monkeypatch.setattr(rollback, "data_root", lambda: tmp_path)
    active = tmp_path / "models" / "warrior_ignition_model.json"
    atomic_json(active, {"original": True})
    versions = active.parent / "versions"
    versions.mkdir()
    atomic_json(versions / "invalid.json", {"bad": True})
    before = active.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        rollback.restore("ignition", "invalid.json")
    assert active.read_bytes() == before
    with pytest.raises(ValueError):
        rollback.restore("ignition", "../invalid.json")


def test_stop_loss_during_cancel_is_accounted_before_another_buy():
    from daytrading.exits.manager import TrackedPosition
    from daytrading.models import Position
    from daytrading.pipeline.factory import create_scalping_pipeline

    client = Mock()
    client.get_orders.side_effect = [[stop()], [], []]
    client.get_all_positions.return_value = [SimpleNamespace(symbol="TEST",qty=90)]
    client.get_order_by_id.return_value = stop("canceled", filled=10)
    client.submit_order.return_value = accepted_stop()
    instance = broker(client)
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._broker = instance
    runner._hub = SimpleNamespace(trading_paused=False)
    runner._last_synced_order_ids = set()
    pipeline = create_scalping_pipeline(initial_cash=10000)
    pipeline._max_daily_loss = -5
    pipeline.portfolio.positions["TEST"] = Position("TEST", 100, 10)
    pipeline.exit_manager.track(
        TrackedPosition(symbol="TEST", side=Side.BUY, quantity=100, entry_price=10)
    )
    runner._pipeline = pipeline
    def record_fill(fill, entry_price, reason):
        pipeline.exit_manager.commit_coordinated_exit_fill(fill)
        return pipeline.record_realized_exit(fill.symbol, entry_price, fill.price, fill.quantity,
            remaining_quantity=90,fill_key=pipeline.exit_fill_key(fill))
    runner._record_trade_exit=record_fill
    instance.set_execution_callbacks(
        runner._on_conflicting_stop_fill,
        lambda symbol: runner._new_entries_blocked(symbol, "test"),
    )
    fill, status = instance.submit(
        Order("TEST", Side.BUY, 20, 10), bar(), pipeline.portfolio
    )
    assert fill is None and status is OrderStatus.REJECTED
    assert pipeline._daily_pnl == -10
    assert pipeline.portfolio.positions["TEST"].quantity == 90
    assert pipeline.exit_manager.tracked["TEST"].remaining_qty == 90
    assert pipeline._circuit_breaker_tripped
    assert ("TEST","sell","old-stop") in pipeline.portfolio._fill_projections["inventory"]
    # Only the restored stop was submitted; there was no new entry.
    client.submit_order.assert_called_once()
    assert client.submit_order.call_args.kwargs["order_data"].qty == 90


def test_hot_retry_cannot_revive_a_setup_invalidated_by_stop_execution():
    from daytrading.models import SignalAction, TradeSignal

    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._broker = SimpleNamespace(_conflict_fill_during_cancel=True)
    signal = TradeSignal(
        symbol="TEST", action=SignalAction.ENTER_LONG, quantity=20, entry_price=10
    )
    fill, status = runner._retry_hot_hod_timed_entry(
        signal, OrderStatus.REJECTED, bar()
    )
    assert fill is None and status is OrderStatus.REJECTED


def test_sdk_transport_supplies_a_bounded_timeout(monkeypatch):
    from requests.adapters import HTTPAdapter

    from daytrading.execution.transport import BoundedHTTPAdapter

    sent = Mock(return_value=object())
    monkeypatch.setattr(HTTPAdapter, "send", sent)
    request = object()
    BoundedHTTPAdapter().send(request, timeout=None)
    assert sent.call_args.kwargs["timeout"] == (3.0, 5.0)
    BoundedHTTPAdapter().send(request, timeout=2)
    assert sent.call_args.kwargs["timeout"] == 2


def test_position_timeout_reuses_request_and_blocks_stale_reconciliation():
    import concurrent.futures

    client = Mock()
    instance = broker(client)
    pending = Mock()
    pending.result.side_effect = concurrent.futures.TimeoutError()
    pending.done.return_value = False
    instance._position_pool = Mock()
    instance._position_pool.submit.return_value = pending
    instance._position_future = None
    assert instance.get_positions() == {}
    assert instance._position_sync_stale
    assert instance.get_positions() == {}
    instance._position_pool.submit.assert_called_once()
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._broker = instance
    runner._reconciler = Mock()
    runner._push_positions_from_alpaca()
    runner._reconciler.reconcile.assert_not_called()
