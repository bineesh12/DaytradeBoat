from datetime import datetime, timedelta, timezone

from daytrading.data.market_data_service import MarketDataService


def test_market_data_service_dedupes_and_pulls_candidate_batch():
    service = MarketDataService(candidate_queue_max=10, candidate_batch_max=2)

    assert service.enqueue_candidate({"symbol": "BATL", "abs_change_pct": 10}, priority=1, source="scan")
    assert not service.enqueue_candidate({"symbol": "BATL", "abs_change_pct": 11}, priority=1, source="scan")
    assert service.enqueue_candidate({"symbol": "DSY", "abs_change_pct": 8}, priority=2, source="scan")

    batch = service.pull_candidate_batch()

    assert [row["symbol"] for row in batch] == ["BATL", "DSY"]
    assert batch[0]["_source"] == "scan"
    assert service.candidate_pending_size() == 0


def test_market_data_service_hot_watch_snapshot_methods_are_copies():
    service = MarketDataService(hot_watch_max_symbols=2)
    service.hot_watch_set("AAA", {"added_at": datetime.now(timezone.utc), "score": 1})

    items = service.hot_watch_items()
    items[0][1]["score"] = 0

    assert service.hot_watch_contains("AAA")
    assert service.hot_watch_get("AAA")["score"] == 1


def test_market_data_service_tracks_oldest_hot_watch_symbol():
    service = MarketDataService(hot_watch_max_symbols=2)
    now = datetime.now(timezone.utc)
    service.hot_watch_set("OLD", {"added_at": now - timedelta(minutes=5)})
    service.hot_watch_set("NEW", {"added_at": now})

    assert service.hot_watch_oldest_symbol() == "OLD"


def test_candidate_worker_pauses_then_processes_batch():
    service = MarketDataService(candidate_queue_max=10, candidate_batch_max=5)
    service.enqueue_candidate({"symbol": "FOXX"}, priority=0, source="unit")
    statuses = []
    processed = []
    pause_calls = {"n": 0}

    def stop_requested():
        return bool(processed)

    def pause_state():
        pause_calls["n"] += 1
        return (pause_calls["n"] == 1, pause_calls["n"] == 1)

    def process_batch(batch):
        processed.extend(batch)
        return len(batch)

    service.run_candidate_hydration_worker(
        stop_requested=stop_requested,
        pause_state=pause_state,
        process_batch=process_batch,
        publish_status=lambda **payload: statuses.append(payload),
        sleep=lambda _seconds: None,
        max_iterations=3,
    )

    assert [row["symbol"] for row in processed] == ["FOXX"]
    assert statuses[0]["paused_for_entry"] is True
    assert statuses[1]["paused_for_entry"] is False
    assert statuses[-1]["hydrated"] == 1
