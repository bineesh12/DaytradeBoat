"""Composite scanner — runs multiple scanners and merges results.

Symbols flagged by more scanners get higher aggregate scores.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from daytrading.scanner.base import Scanner
from daytrading.models import Bar, ScanResult


class CompositeScanner:
    """Runs several scanners against the same universe, merges & ranks results."""

    def __init__(self, scanners: Sequence[Scanner]) -> None:
        self._scanners = list(scanners)

    @property
    def name(self) -> str:
        return "composite"

    def scan(self, universe: Dict[str, Sequence[Bar]]) -> List[ScanResult]:
        by_symbol: Dict[str, List[ScanResult]] = {}
        for scanner in self._scanners:
            for hit in scanner.scan(universe):
                by_symbol.setdefault(hit.symbol, []).append(hit)

        merged: List[ScanResult] = []
        for symbol, hits in by_symbol.items():
            best = max(hits, key=lambda h: h.score)
            combined_criteria = {
                "scanners_matched": [h.scanner_name for h in hits],
                "scanner_count": len(hits),
            }
            for h in hits:
                combined_criteria[h.scanner_name] = h.criteria

            merged.append(ScanResult(
                symbol=symbol,
                scanner_name=self.name,
                ts=best.ts,
                score=sum(h.score for h in hits),
                criteria=combined_criteria,
                bars=best.bars,
            ))

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged
