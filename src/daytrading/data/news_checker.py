"""News sentiment checker using Alpaca's News API.

Fetches recent news for a symbol and scores sentiment based on
headline/summary keywords. Returns a score from -1.0 (very negative)
to +1.0 (very positive), plus the headlines for logging.

No external NLP dependencies — uses keyword matching which is fast
and sufficient for day-trading news (earnings, FDA, contracts, etc.).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Strongly positive keywords (catalyst-driven moves)
_POSITIVE = {
    "beats", "exceeds", "surges", "soars", "jumps", "rallies", "spikes",
    "upgrade", "upgraded", "buy rating", "outperform", "strong buy",
    "approval", "approved", "fda approval", "patent", "granted",
    "contract", "awarded", "partnership", "deal", "acquisition",
    "revenue growth", "record revenue", "record earnings", "profit",
    "beat expectations", "beats estimates", "above consensus",
    "breakthrough", "launch", "launches", "expands", "expansion",
    "bullish", "positive", "upside", "raises guidance",
    "insider buying", "buyback", "share repurchase", "dividend",
}

# Strongly negative keywords
_NEGATIVE = {
    "misses", "falls", "drops", "plunges", "crashes", "tanks", "tumbles",
    "downgrade", "downgraded", "sell rating", "underperform",
    "recall", "recalled", "lawsuit", "sued", "investigation",
    "fda rejection", "denied", "delisted", "delisting", "warning",
    "bankruptcy", "default", "loss widens", "revenue decline",
    "misses estimates", "below consensus", "disappoints",
    "bearish", "negative", "downside", "lowers guidance", "cuts guidance",
    "dilution", "offering", "secondary offering", "shelf registration",
    "insider selling", "resignation", "fired", "fraud",
}

# Moderate weights
_MILD_POSITIVE = {"rises", "gains", "up", "higher", "growth", "strong"}
_MILD_NEGATIVE = {"dips", "slips", "lower", "weak", "concern", "risk", "volatility"}


class NewsChecker:
    """Check recent news sentiment for symbols via Alpaca API."""

    def __init__(self, api_key: str, secret_key: str, *, max_age_hours: int = 24) -> None:
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._max_age = timedelta(hours=max_age_hours)
        self._cache: Dict[str, Tuple[float, List[str], datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def get_sentiment(self, symbol: str) -> Tuple[float, List[str]]:
        """Return (score, headlines) for a symbol.

        Score: -1.0 to +1.0 (0.0 = neutral/no news)
        Headlines: list of recent headline strings
        """
        now = datetime.now(timezone.utc)

        # Check cache
        if symbol in self._cache:
            score, headlines, cached_at = self._cache[symbol]
            if now - cached_at < self._cache_ttl:
                return score, headlines

        try:
            score, headlines = self._fetch_and_score(symbol)
            self._cache[symbol] = (score, headlines, now)
            return score, headlines
        except Exception as exc:
            logger.debug("News fetch failed for %s: %s", symbol, exc)
            return 0.0, []

    def _fetch_and_score(self, symbol: str) -> Tuple[float, List[str]]:
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": symbol,
            "limit": 5,
            "include_content": "false",
        }
        resp = requests.get(url, headers=self._headers, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        now = datetime.now(timezone.utc)
        headlines: List[str] = []
        total_score = 0.0
        count = 0

        for article in data.get("news", []):
            created = article.get("created_at", "")
            try:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if now - ts > self._max_age:
                    continue
            except (ValueError, TypeError):
                continue

            headline = article.get("headline", "")
            summary = article.get("summary", "")
            text = (headline + " " + summary).lower()
            headlines.append(headline)

            score = self._score_text(text)
            # More recent news gets higher weight
            age_hours = (now - ts).total_seconds() / 3600
            recency_weight = max(0.2, 1.0 - age_hours / 24.0)
            total_score += score * recency_weight
            count += 1

        if count == 0:
            return 0.0, []

        avg = total_score / count
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, avg)), headlines

    def _score_text(self, text: str) -> float:
        score = 0.0

        for phrase in _POSITIVE:
            if phrase in text:
                score += 1.0
        for phrase in _NEGATIVE:
            if phrase in text:
                score -= 1.0
        for phrase in _MILD_POSITIVE:
            if re.search(r'\b' + phrase + r'\b', text):
                score += 0.3
        for phrase in _MILD_NEGATIVE:
            if re.search(r'\b' + phrase + r'\b', text):
                score -= 0.3

        # Normalize: typically 0-3 keyword matches
        if score > 0:
            return min(1.0, score / 2.0)
        elif score < 0:
            return max(-1.0, score / 2.0)
        return 0.0
