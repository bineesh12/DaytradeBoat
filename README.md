# DaytradeBoat — Phase 2 observation candidate

This branch preserves the reviewed Python Phase 2 application source for development and review. It is **not qualified for automatic PAPER entries**. It does not deploy, start, restart or activate a bot.

Source core: `1ee6c69bf48eb600503ce6cf12f55b597ade332f07133fbba96a93653f6dee43`.

## Open in VS Code

Open this repository folder with **File → Open Folder**, or run `code .` from its root. The existing VS Code settings use `.venv/bin/python` when a local virtual environment exists.

## Development setup

Use Python 3.11 or later in a separate virtual environment:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data]'
python tools/test_offline.py
```

Dependency ranges in `pyproject.toml` are not a lockfile. Optional ML tests need the corresponding optional dependencies. Source hashes are recorded in `docs/phase2-source-manifest.json`.

The offline runner strips trading credentials, disables dotenv loading, denies network access before application imports, and isolates runtime files in a temporary directory. It never starts the trading runner. It excludes the historical-context modules and cases whose four private market-data fixtures are not published, including the wick-routing module that imports historical context. Four existing research modules also require the separate `daytradeboat_studies` project and are excluded from this application-only run. Their assertions remain intact. Native pre-existing optional skips remain separate.

## What is included

- Exact Phase 2 application source, including candle provenance, HOD renewal, data-health and Project A retry/position-refresh repairs.
- The two latest reviewed optimizations: partitioned shadow-header caches and avoiding an unused clock read only for an exactly empty native retry book.
- Deterministic tests and the observation entry-latch module/tests.
- A source fingerprint and a sanitized readiness summary.

Generated egg-info is omitted. It is rebuilt by packaging and is not part of the Python source-core fingerprint.

The equivalence tests use unchanged reviewed baseline files in `tests/reference`, replacing only their old workspace-relative lookups and isolating reference output in temporary storage. One historical-looking order identity in a test is replaced with a synthetic UUID. No application code or test assertions are changed for publication.

## Current readiness

At the October 27, 2026 verification, the cloud source was installed but no cloud trading process or transferred current execution database was present. The existing laptop process remained observation-latched. A short healthy laptop sample does not establish cloud performance qualification.

Automatic entries remain blocked pending current-state handover, actual-market observation and qualification of the separate Phase 3 execution-safety repairs. This Phase 2 snapshot still contains the previously identified incomplete-read/false-flat and uncertain bulk-close paths. Dashboard Resume does not remove the immutable observation process latch.

The approved current-state fee handover, account reconciliation, rollback transaction, single-owner fencing and VM lifecycle checks are operational prerequisites. Private deployment manifests and account evidence are deliberately absent here. Existing generic deployment scripts are inherited repository history, not an approved cutover procedure for this candidate.

Do not use this checkout to bypass the reviewed observation launcher, state handover or activation gates. Preserve PAPER-only operation, the two-second health limit, FIFO evidence, original setup expiry, 1R room, and all existing entry, stop, sizing and daily-risk controls. Live-money trading is not authorized.

## Excluded private material

Credentials, account/order/activity exports, journals, execution databases, recordings, SSH keys, financial repair packets, raw reports, production configuration and historical research datasets are not included. The frozen Phase 2 deployment package, integrated Phase 3 candidate and simulator research remain separate and unchanged.
