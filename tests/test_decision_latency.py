import json
from datetime import datetime, timezone
from unittest.mock import patch

from daytrading.journal.store import TradingJournal
from daytrading.ml import shadow_collector as sc


def test_unrelated_pending_rows_are_not_parsed_twice_or_relabelled(tmp_path):
    path = tmp_path / 'labels.jsonl'
    original = json.dumps(dict(symbol='OTHER', ts='2026-10-22T08:00:00Z', label=None, price=2))
    path.write_text(original + '\n')
    sc._row_headers.clear()
    real_loads = json.loads
    with patch.object(sc.json, 'loads', wraps=real_loads) as loads:
        assert sc._label_pending_file(path, {'BKYI': 3.17}, wait_seconds=180, positive_pct=2, fail_pct=1.5) == 0
        assert loads.call_count == 1
    assert path.read_text() == original + '\n'


def test_immutable_header_cache_does_not_reuse_outcomes_or_corrected_lines(tmp_path):
    path = tmp_path / 'labels.jsonl'
    row = dict(symbol='BKYI', ts='2026-10-22T08:00:00Z', label=None, price=3.0)
    path.write_text(json.dumps(row) + '\n')
    now = datetime(2026, 10, 22, 8, 5, tzinfo=timezone.utc)
    with patch.object(sc, '_utc_now', return_value=now):
        assert sc._label_pending_file(path, {'BKYI': 3.17}, wait_seconds=180, positive_pct=2, fail_pct=1.5) == 1
        assert json.loads(path.read_text())['label'] == 1
        row['price'] = 3.5
        path.write_text(json.dumps(row) + '\n')
        assert sc._label_pending_file(path, {'BKYI': 3.17}, wait_seconds=180, positive_pct=2, fail_pct=1.5) == 1
        assert json.loads(path.read_text())['label'] == 0


def test_recording_timing_is_observational_and_keeps_final_evidence(tmp_path):
    journal = TradingJournal(base_dir=str(tmp_path), skipped_event_types=['entry_submission_checks'])
    payload = {'passed': True, 'source_candles': [{'high': 3.18}], 'setup_id': 'original'}
    with patch('daytrading.reliability_metrics.perf_counter_ns', side_effect=[100, 150]):
        journal.record('entry_submission_checks', payload)
    assert journal._stage_metrics.status()['journal_record:str']['total_ns'] == 50
    saved = journal._conn.execute("select payload_json from events where type='entry_submission_checks'").fetchone()
    assert json.loads(saved[0]) == payload
    journal.close()
