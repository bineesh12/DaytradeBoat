"""Regression for SIP batch histories truncated at the global API limit."""
from datetime import datetime, timezone

from daytrading.data.alpaca_feed import AlpacaHistoricalFeed


def test_sdk_pages_all_symbols_then_retains_recent_bars_per_symbol(monkeypatch):
    feed = AlpacaHistoricalFeed('test', 'test', feed='sip', bar_fetch_batch_delay_sec=0)
    calls = []

    def raw_bar(minute, price):
        return {'t':f'2026-10-14T10:0{minute}:00Z','o':price,'h':price,
                'l':price,'c':price,'v':100,'n':1,'vw':price}

    def page(path, data):
        calls.append(dict(data))
        if data.get('page_token') is None:
            bars = [raw_bar(i, 10+i) for i in range(3)]
            # Model the API's total limit, including a token for remaining bars.
            limit = data.get('limit')
            if limit is not None:
                bars = bars[:int(limit)]
            return {'bars':{'AAA':bars},'next_page_token':'second-symbol'}
        return {'bars':{'ZZZ':[raw_bar(i, 20+i) for i in range(3)]},'next_page_token':None}

    monkeypatch.setattr(feed._client, 'get', page)
    result = feed.get_bars(['AAA','ZZZ'], limit=2,
                           start=datetime(2026,10,14,10,tzinfo=timezone.utc),
                           end=datetime(2026,10,14,10,10,tzinfo=timezone.utc))
    assert set(result) == {'AAA','ZZZ'}
    assert [b.close for b in result['AAA']] == [11,12]
    assert [b.close for b in result['ZZZ']] == [21,22]
    assert calls[1]['page_token'] == 'second-symbol'
    assert calls[0]['adjustment'] == 'split'
    assert feed.last_fetch_failures == 0
