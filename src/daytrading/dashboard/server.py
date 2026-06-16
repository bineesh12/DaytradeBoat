"""Flask web server for the trading dashboard.

Serves a single-page app with real-time updates via SSE.
Runs in a background daemon thread so it doesn't block the trading pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import BadRequest

from daytrading.dashboard.hub import DashboardHub

logger = logging.getLogger(__name__)

_hub: Optional[DashboardHub] = None


def _missed_a_plus_spread_summary(rows: list[dict]) -> dict:
    spread_rows = [r for r in rows if r.get("is_spread_reject") or "spread" in str(r.get("reason", "")).lower()]
    false_blocks = [r for r in spread_rows if r.get("outcome") == "missed_opportunity"]
    correct_rejects = [r for r in spread_rows if r.get("outcome") == "correct_reject"]
    pending = [r for r in spread_rows if r.get("outcome") == "pending"]
    symbols: dict[str, int] = {}
    for row in spread_rows:
        sym = str(row.get("symbol") or "")
        if sym:
            symbols[sym] = symbols.get(sym, 0) + 1
    return {
        "spread_blocked_runners": len(spread_rows),
        "spread_false_blocks": len(false_blocks),
        "spread_correct_rejects": len(correct_rejects),
        "spread_pending": len(pending),
        "symbols": symbols,
    }


def _missed_a_plus_risk_summary(rows: list[dict]) -> dict:
    risk_rows = [
        r for r in rows
        if r.get("is_risk_reject")
        or "risk too wide" in str(r.get("reason", "")).lower()
        or "r:r" in str(r.get("reason", "")).lower()
    ]
    false_blocks = [r for r in risk_rows if r.get("outcome") == "missed_opportunity"]
    correct_rejects = [r for r in risk_rows if r.get("outcome") == "correct_reject"]
    pending = [r for r in risk_rows if r.get("outcome") == "pending"]
    survived = [r for r in risk_rows if r.get("tactical_stop_survived") is True]
    failed = [r for r in risk_rows if r.get("tactical_stop_survived") is False]
    clean_survived = [r for r in risk_rows if r.get("tactical_stop_clean_survival") is True]
    clean_failed = [r for r in risk_rows if r.get("tactical_stop_clean_survival") is False]
    choppy_survived = [
        r for r in risk_rows
        if r.get("tactical_stop_survived") is True
        and not r.get("smooth_for_tactical_stop")
    ]
    symbols: dict[str, int] = {}
    for row in risk_rows:
        sym = str(row.get("symbol") or "")
        if sym:
            symbols[sym] = symbols.get(sym, 0) + 1
    return {
        "risk_blocked_runners": len(risk_rows),
        "risk_false_blocks": len(false_blocks),
        "risk_correct_rejects": len(correct_rejects),
        "risk_pending": len(pending),
        "tactical_stop_survived": len(survived),
        "tactical_stop_failed": len(failed),
        "clean_tactical_stop_survived": len(clean_survived),
        "clean_tactical_stop_failed": len(clean_failed),
        "choppy_tactical_stop_survived": len(choppy_survived),
        "symbols": symbols,
    }


def create_app(hub: DashboardHub) -> Flask:
    global _hub
    _hub = hub

    app = Flask(__name__)
    analytics_generation_lock = threading.Lock()
    backtest_lock = threading.Lock()
    app.config["PROPAGATE_EXCEPTIONS"] = True

    @app.after_request
    def add_no_cache(response):
        if (
            response.content_type
            and ("text/html" in response.content_type or request.path.startswith("/api/"))
        ):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.errorhandler(BadRequest)
    def handle_bad_request(exc):
        logger.warning(
            "Dashboard bad request on %s %s: %s",
            request.method,
            request.path,
            getattr(exc, "description", str(exc)),
        )
        return jsonify({
            "ok": False,
            "error": "bad request",
            "path": request.path,
        }), 400

    @app.route("/")
    def index():
        from flask import make_response
        resp = make_response(DASHBOARD_HTML)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.route("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.route("/api/snapshot")
    def snapshot():
        return jsonify(_hub.snapshot())

    @app.route("/api/ml-stats")
    def ml_stats():
        """Return ML model monitoring stats."""
        try:
            from daytrading.strategy.entry_guard import (
                get_ml_monitor,
                is_entry_ml_enabled,
                is_entry_ml_loaded,
            )
            monitor = get_ml_monitor()
            if monitor is None:
                return jsonify({"enabled": False, "reason": "no monitor"})
            stats = monitor.stats.to_dict()
            model_enabled = is_entry_ml_enabled()
            model_loaded = is_entry_ml_loaded()
            stats["model_enabled"] = model_enabled
            stats["model_loaded"] = model_loaded
            stats["model_active"] = model_enabled and model_loaded and monitor.is_model_enabled
            if not model_enabled:
                stats["disable_reason"] = stats.get("disable_reason") or "disabled by config"
            return jsonify(stats)
        except Exception as exc:
            return jsonify({"enabled": False, "reason": str(exc)})

    @app.route("/api/missed-a-plus")
    def missed_a_plus():
        """Return blocked A+ setups with later outcome labels."""
        rows = list(getattr(_hub, "missed_a_plus", []))
        return jsonify({
            "ok": True,
            "rows": rows,
            "spread_summary": _missed_a_plus_spread_summary(rows),
            "risk_summary": _missed_a_plus_risk_summary(rows),
            "scanner_near_miss": dict(getattr(_hub, "scanner_near_miss", {}) or {}),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        })

    @app.route("/api/analytics-report")
    def analytics_report():
        """Return the latest nightly analytics report for dashboard display."""
        try:
            report_dir = None
            if getattr(_hub, "journal", None):
                base_dir = getattr(_hub.journal, "base_dir", None)
                if base_dir:
                    report_dir = os.path.join(os.path.dirname(base_dir), "reports")
            report_dir = report_dir or os.environ.get("DAYTRADING_REPORT_DIR", "data/reports")
            report_dir = os.path.abspath(report_dir)
            requested_day = request.args.get("day")
            explicit_day = bool(requested_day)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            has_journal_db = bool(getattr(getattr(_hub, "journal", None), "db_path", None))

            if not os.path.isdir(report_dir):
                os.makedirs(report_dir, exist_ok=True)

            candidates = []
            for name in os.listdir(report_dir):
                if not name.endswith(".json"):
                    continue
                day = name[:-5]
                if explicit_day and day != requested_day:
                    continue
                candidates.append((day, os.path.join(report_dir, name)))
            if not candidates:
                generation_day = requested_day if explicit_day else today
                if generation_day and has_journal_db:
                    try:
                        if not analytics_generation_lock.acquire(blocking=False):
                            return jsonify({
                                "ok": True,
                                "report": None,
                                "message": f"Analytics report for {generation_day} is generating",
                            })
                        from daytrading.analyst.collector import NightlyAnalyst
                        try:
                            analyst = NightlyAnalyst(
                                db_path=_hub.journal.db_path,
                                report_dir=report_dir,
                            )
                            generated = analyst.run(generation_day)
                            if generated.get("status") not in ("no_trades", "holiday"):
                                path = os.path.join(report_dir, f"{generation_day}.json")
                                candidates.append((generation_day, path))
                        finally:
                            analytics_generation_lock.release()
                    except Exception as exc:
                        logger.debug("On-demand analytics report generation skipped: %s", exc)
            if not candidates:
                msg = "No nightly analytics report found"
                if requested_day:
                    msg += f" for {requested_day}"
                return jsonify({"ok": True, "report": None, "message": msg})

            day, path = sorted(candidates, reverse=True)[0]
            with open(path) as f:
                report = json.load(f)
            if explicit_day and getattr(_hub, "journal", None):
                try:
                    from daytrading.analyst.collector import NightlyAnalyst
                    analyst = NightlyAnalyst(
                        db_path=_hub.journal.db_path,
                        report_dir=report_dir,
                    )
                    # Keep the normal dashboard load fast. Only explicit day requests
                    # refresh ML sections because those can scan large JSONL/model files.
                    report["ml_learning"] = analyst._analyze_ml_learning(day)
                    report["ml_progress"] = analyst._analyze_ml_progress(day)
                    if "setup_performance" not in report:
                        report["setup_performance"] = analyst._analyze_setup_performance(day, [])
                except Exception as exc:
                    logger.debug("Analytics report ML refresh skipped: %s", exc)
            return jsonify({
                "ok": True,
                "day": day,
                "report": report,
                "generated_at": report.get("generated_at"),
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.error("Failed to load analytics report: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/screenshot", methods=["POST"])
    def save_screenshot():
        if not getattr(_hub, "journal", None):
            return jsonify({"ok": False, "error": "journal not configured"}), 503
        payload = request.get_json(silent=True) or {}
        symbol = str(payload.get("symbol", "")).strip().upper()
        image_b64 = payload.get("image_b64")
        source_path = payload.get("source_path")
        context = payload.get("context", {})
        if not symbol:
            return jsonify({"ok": False, "error": "symbol is required"}), 400
        if not image_b64 and not source_path:
            return jsonify({"ok": False, "error": "image_b64 or source_path is required"}), 400
        try:
            meta = _hub.journal.save_screenshot(
                symbol,
                image_b64=image_b64,
                source_path=source_path,
                context=context if isinstance(context, dict) else {},
            )
            return jsonify({"ok": True, "screenshot": meta})
        except Exception as exc:
            logger.error("Failed to save screenshot: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/replay")
    def replay():
        if not getattr(_hub, "journal", None):
            return jsonify({"ok": False, "error": "journal not configured"}), 503
        day = request.args.get("day")
        try:
            limit = int(request.args.get("limit", "0")) or None
        except Exception:
            limit = None
        events = _hub.journal.replay_frames(day=day, limit=limit)
        return jsonify({"ok": True, "count": len(events), "events": events})

    @app.route("/api/backtest", methods=["POST"])
    def backtest():
        payload = request.get_json(silent=True) or {}
        symbol = str(payload.get("symbol") or "").upper().strip()
        day = str(payload.get("date") or "").strip()
        start_time = str(payload.get("start_time") or "").strip()
        flags = payload.get("flags") or {}
        if not symbol:
            return jsonify({"ok": False, "error": "symbol is required"}), 400
        if not day:
            return jsonify({"ok": False, "error": "date is required"}), 400
        if not backtest_lock.acquire(blocking=False):
            return jsonify({
                "ok": False,
                "error": "another backtest is already running",
            }), 429
        try:
            from daytrading.config import Settings
            from daytrading.backtest.service import run_backtest
            result = run_backtest(
                symbol,
                day,
                flags=flags,
                start_time=start_time,
                settings=Settings(),
            )
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:
            logger.exception("Dashboard backtest failed")
            return jsonify({"ok": False, "error": str(exc)}), 500
        finally:
            backtest_lock.release()

    @app.route("/api/entry_scores", methods=["GET"])
    def entry_scores():
        """Live/paper entry-quality scores for a symbol+date (what it scored in paper).

        Surfaces the existing ml/entry_candidates.jsonl log so paper scores can
        be compared against a backtest of the same name/day — e.g. why a name
        passed at 85 live but the backtest scores it under 80.
        """
        symbol = str(request.args.get("symbol") or "").upper().strip()
        day = str(request.args.get("date") or "").strip()
        if not symbol or not day:
            return jsonify({"ok": False, "error": "symbol and date are required"}), 400
        try:
            from daytrading.ml.data_collector import load_candidates_for
            rows = load_candidates_for(symbol, day)
        except Exception as exc:
            logger.exception("entry_scores load failed")
            return jsonify({"ok": False, "error": str(exc)}), 500
        passed = [r for r in rows if r.get("passed")]
        best = max((r.get("score") or 0) for r in rows) if rows else None
        best_passed = max((r.get("score") or 0) for r in passed) if passed else None
        return jsonify({
            "ok": True,
            "symbol": symbol,
            "date": day,
            "total": len(rows),
            "passed": len(passed),
            "rejected": len(rows) - len(passed),
            "best_score": best,
            "best_passed_score": best_passed,
            "candidates": rows,
        })

    @app.route("/api/backtest/sweep", methods=["POST"])
    def backtest_sweep():
        payload = request.get_json(silent=True) or {}
        symbols = payload.get("symbols") or []
        dates = payload.get("dates") or []
        experiments = payload.get("experiments") or None
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
        if isinstance(dates, str):
            dates = [d.strip() for d in dates.split(",") if d.strip()]
        if not symbols:
            return jsonify({"ok": False, "error": "symbols are required"}), 400
        if not dates:
            return jsonify({"ok": False, "error": "dates are required"}), 400
        if not backtest_lock.acquire(blocking=False):
            return jsonify({
                "ok": False,
                "error": "another backtest is already running",
            }), 429
        try:
            from daytrading.config import Settings
            from daytrading.backtest.service import run_backtest_sweep
            result = run_backtest_sweep(
                symbols,
                dates,
                experiments=experiments,
                settings=Settings(),
            )
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:
            logger.exception("Dashboard backtest sweep failed")
            return jsonify({"ok": False, "error": str(exc)}), 500
        finally:
            backtest_lock.release()

    @app.route("/api/pause", methods=["POST"])
    def pause_trading():
        _hub.trading_paused = True
        _hub._broadcast("trading_control", {"paused": True})
        logger.info("Trading PAUSED via dashboard")
        return jsonify({"ok": True, "paused": True})

    @app.route("/api/resume", methods=["POST"])
    def resume_trading():
        _hub.trading_paused = False
        _hub._broadcast("trading_control", {"paused": False})
        logger.info("Trading RESUMED via dashboard")
        return jsonify({"ok": True, "paused": False})

    @app.route("/api/force-close", methods=["POST"])
    def force_close_all():
        broker = getattr(_hub, "_broker", None)
        if not broker:
            return jsonify({"ok": False, "error": "broker not available"}), 503
        try:
            _hub.trading_paused = True
            _hub._broadcast("trading_control", {"paused": True})
            if hasattr(broker, "emergency_close_all_positions"):
                result = broker.emergency_close_all_positions(attempts=4, settle_seconds=1.0)
            else:
                if hasattr(broker, "cancel_all_orders"):
                    broker.cancel_all_orders()
                broker.close_all_positions()
                result = {
                    "ok": True,
                    "flat": True,
                    "cancelled_orders": None,
                    "submitted_orders": [],
                    "remaining_positions": {},
                    "errors": [],
                }
            exit_mgr = getattr(_hub, "_exit_manager", None)
            if exit_mgr and result.get("flat", False):
                for sym in list(exit_mgr.tracked.keys()):
                    exit_mgr.untrack(sym)
            _hub._broadcast("trading_control", {
                "force_closed": result.get("flat", False),
                "paused": True,
            })
            logger.warning("EMERGENCY FORCE CLOSE via dashboard: %s", result)
            status = 200 if result.get("flat", False) else 500
            result.setdefault("message", "All positions closed" if result.get("flat", False) else "Positions remain open")
            return jsonify(result), status
        except Exception as exc:
            logger.error("Force close failed: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/stream")
    def stream():
        q = _hub.subscribe()

        def generate():
            try:
                # Push full state once on connect (replaces removed /api/snapshot polling).
                yield "data: {}\n\n".format(
                    json.dumps({"type": "snapshot", "data": _hub.snapshot()}),
                )
                heartbeat_counter = 0
                while True:
                    while q:
                        msg = q.popleft()
                        yield "data: {}\n\n".format(json.dumps(msg))
                        heartbeat_counter = 0
                    time.sleep(0.3)
                    heartbeat_counter += 1
                    # Send keepalive comment every ~5 seconds to prevent timeout
                    if heartbeat_counter >= 16:
                        yield ": keepalive\n\n"
                        heartbeat_counter = 0
            except GeneratorExit:
                _hub.unsubscribe(q)

        return Response(generate(), mimetype="text/event-stream")

    return app


def start_dashboard(hub: DashboardHub, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the dashboard in a background daemon thread.

    Force-kills any existing process on the port to avoid connection
    limit issues from orphaned servers.
    """
    app = create_app(hub)

    def _run(p: int) -> None:
        import warnings
        warnings.filterwarnings("ignore", message=".*development server.*")
        for attempt in range(3):
            try:
                app.run(host=host, port=p, debug=False, use_reloader=False, threaded=True)
                break
            except Exception as exc:
                logger.error("Dashboard thread crashed (attempt %d/3): %s", attempt + 1, exc)
                if attempt < 2:
                    import time as _t
                    _t.sleep(2)

    import socket
    import subprocess

    # Force-kill any process holding our port
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                try:
                    os.kill(int(pid), 9)
                    logger.info("Killed orphan process %s on port %d", pid, port)
                except (ProcessLookupError, ValueError):
                    pass
        if pids:
            import time
            time.sleep(1)
    except Exception:
        pass

    # Verify port is free (retry up to 5s for TIME_WAIT release)
    import time as _time
    port_free = False
    for _retry in range(5):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.close()
            port_free = True
            break
        except OSError:
            _time.sleep(1)
    if not port_free:
        logger.error("Port %d still in use after cleanup — dashboard NOT started", port)
        return

    t = threading.Thread(target=_run, args=(port,), daemon=True, name="dashboard")
    t.start()
    logger.info("Dashboard started at http://localhost:%d", port)


# ---------------------------------------------------------------------------
# Single-page HTML dashboard (embedded — no external files needed)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Day Trading Dashboard</title>
<style>
:root {
  --bg: #0f1117;
  --surface: #1a1d28;
  --surface2: #232736;
  --border: #2d3148;
  --text: #e1e4ed;
  --text2: #8b8fa3;
  --green: #00d68f;
  --green-bg: rgba(0,214,143,0.1);
  --red: #ff4757;
  --red-bg: rgba(255,71,87,0.1);
  --blue: #3b82f6;
  --blue-bg: rgba(59,130,246,0.1);
  --yellow: #fbbf24;
  --yellow-bg: rgba(251,191,36,0.1);
  --purple: #a78bfa;
  --radius: 12px;
  --shadow: 0 2px 8px rgba(0,0,0,0.3);
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }

/* Top navbar */
.navbar { background:var(--surface); border-bottom:1px solid var(--border); padding:12px 24px; display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:100; }
.navbar h1 { font-size:18px; font-weight:700; letter-spacing:-0.5px; }
.navbar h1 span { color:var(--blue); }
.nav-status { display:flex; gap:16px; align-items:center; font-size:13px; }
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }
.status-dot.on { background:var(--green); box-shadow:0 0 6px var(--green); }
.status-dot.off { background:var(--red); }

/* Tabs */
.tabs { display:flex; gap:0; background:var(--surface); border-bottom:1px solid var(--border); padding:0 24px; }
.tab { padding:12px 20px; cursor:pointer; font-size:13px; font-weight:500; color:var(--text2); border-bottom:2px solid transparent; transition:all 0.2s; }
.tab:hover { color:var(--text); }
.tab.active { color:var(--blue); border-bottom-color:var(--blue); }

/* Layout */
.container { padding:20px 24px; max-width:1440px; margin:0 auto; }
.grid { display:grid; gap:16px; }
.grid-4 { grid-template-columns:repeat(4,1fr); }
.grid-3 { grid-template-columns:repeat(3,1fr); }
.grid-2 { grid-template-columns:1fr 1fr; }
.grid-1 { grid-template-columns:1fr; }

/* Cards */
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:20px; box-shadow:var(--shadow); }
.card-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }
.card-header h3 { font-size:14px; font-weight:600; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px; }
.card-header .badge { font-size:11px; padding:3px 8px; border-radius:6px; font-weight:600; }

/* Stat cards */
.stat-card { text-align:center; }
.stat-value { font-size:28px; font-weight:700; margin:4px 0; letter-spacing:-1px; }
.stat-label { font-size:12px; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px; }
.stat-sub { font-size:12px; margin-top:4px; }
.mini-muted { color:var(--text2); font-size:11px; margin-top:4px; }

/* Tables */
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; padding:10px 12px; color:var(--text2); font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; border-bottom:1px solid var(--border); }
td { padding:10px 12px; border-bottom:1px solid var(--border); }
tr:last-child td { border-bottom:none; }
tr:hover { background:var(--surface2); }

/* Pills */
.pill { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
.pill-green { background:var(--green-bg); color:var(--green); }
.pill-red { background:var(--red-bg); color:var(--red); }
.pill-blue { background:var(--blue-bg); color:var(--blue); }
.pill-yellow { background:var(--yellow-bg); color:var(--yellow); }
.pill-purple { background:rgba(167,139,250,0.1); color:var(--purple); }

.text-green { color:var(--green); }
.text-red { color:var(--red); }
.text-blue { color:var(--blue); }
.text-yellow { color:var(--yellow); }

/* Page sections */
.page { display:none; }
.page.active { display:block; }

/* Log panel */
.log-panel { max-height:300px; overflow-y:auto; font-family:'SF Mono',Monaco,'Consolas',monospace; font-size:12px; line-height:1.8; padding:8px; }
.log-panel .log-line { padding:2px 0; }
.log-line .ts { color:var(--text2); }
.log-line.warn { color:var(--yellow); }
.log-line.error { color:var(--red); }

/* Empty state */
.empty { text-align:center; padding:40px; color:var(--text2); }
.empty .icon { font-size:36px; margin-bottom:12px; }

/* Trading control buttons */
.ctrl-btn { padding:6px 14px; border:none; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer; transition:opacity 0.2s; }
.ctrl-btn:hover { opacity:0.85; }
.ctrl-btn:disabled { opacity:0.4; cursor:not-allowed; }
.ctrl-btn-stop { background:var(--red); color:#fff; }
.ctrl-btn-start { background:var(--green); color:#fff; }
.ctrl-btn-close { background:#dc2626; color:#fff; }

/* Confidence bar */
.conf-bar { width:60px; height:6px; background:var(--surface2); border-radius:3px; overflow:hidden; display:inline-block; vertical-align:middle; margin-left:6px; }
.conf-bar-fill { height:100%; border-radius:3px; transition:width 0.3s; }

/* Scanner activity indicator */
.scanner-activity { display:flex; gap:4px; align-items:center; }
.scanner-dot { width:6px; height:6px; border-radius:50%; }

/* HOD Momentum scanner feed */
.hod-momentum-scanner { overflow-x:auto; }
.hod-momentum-scanner table { font-size:12px; }
.hod-momentum-scanner th { white-space:nowrap; font-size:10px; }
.hod-momentum-scanner td { white-space:nowrap; vertical-align:middle; }
.hod-momentum-scanner tr.hod-low-float { background:rgba(0,214,143,0.22); }
.hod-momentum-scanner tr.hod-low-float:hover { background:rgba(0,214,143,0.32); }
.hod-momentum-scanner tr.hod-breakout { background:rgba(255,107,53,0.18); }
.hod-momentum-scanner tr.hod-breakout:hover { background:rgba(255,107,53,0.28); }
.hod-momentum-scanner tr.hod-reclaim { background:rgba(167,139,250,0.14); }
.hod-momentum-scanner tr.hod-today-breakout { background:rgba(251,191,36,0.12); }
.hod-momentum-scanner tr.hod-former-momo { background:rgba(59,130,246,0.14); }
.hod-momentum-scanner tr.hod-squeeze { background:rgba(251,191,36,0.14); }
.hod-momentum-scanner tr.hod-default { background:rgba(251,191,36,0.06); }
.hod-momentum-scanner tr.hod-default:hover { background:rgba(251,191,36,0.12); }
.hod-alert-name { font-weight:600; font-size:11px; color:var(--text); }
.hod-hot { color:#ff6b35; margin-right:4px; }
.hod-source-tick { font-size:10px; color:#ff6b35; font-weight:600; }
.hod-source-bar { font-size:10px; color:var(--text2); }
.hod-filters { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
.hod-filter-btn { padding:4px 10px; border-radius:6px; border:1px solid var(--border); background:var(--surface2); color:var(--text2); font-size:11px; cursor:pointer; }
.hod-filter-btn.active { background:var(--blue-bg); color:var(--blue); border-color:var(--blue); }

@media(max-width:900px) {
  .grid-4 { grid-template-columns:repeat(2,1fr); }
  .grid-3 { grid-template-columns:1fr; }
  .grid-2 { grid-template-columns:1fr; }
}
</style>
</head>
<body>

<nav class="navbar">
  <h1><span>&#9650;</span> Day Trading Bot</h1>
  <div class="nav-status">
    <span><span class="status-dot" id="dot-market"></span><span id="market-label">Market Closed</span></span>
    <span class="pill" id="phase-pill" style="font-size:11px">CLOSED</span>
    <span><span class="status-dot" id="dot-stream"></span><span id="stream-label">Stream Off</span></span>
    <span style="color:var(--text2)">Cycle: <span id="cycle-count">0</span></span>
    <span><span class="status-dot" id="dot-scanner"></span><span id="scanner-label">Scanner Off</span></span>
    <span id="trade-status-pill" class="pill pill-green" style="font-size:11px">ACTIVE</span>
  </div>
  <div style="display:flex;gap:6px;align-items:center">
    <button class="ctrl-btn ctrl-btn-stop" id="btn-stop" title="Pause trading — no new entries">Stop Trade</button>
    <button class="ctrl-btn ctrl-btn-start" id="btn-start" title="Resume trading" disabled>Start Trade</button>
    <button class="ctrl-btn ctrl-btn-close" id="btn-force-close" title="Cancel all orders + close all positions immediately">Force Close All</button>
  </div>
</nav>

<div class="tabs">
  <div class="tab active" data-page="overview">Overview</div>
  <div class="tab" data-page="scanner">Scanner</div>
  <div class="tab" data-page="trades">Trades</div>
  <div class="tab" data-page="analytics">Analytics</div>
  <div class="tab" data-page="backtest">Backtest</div>
  <div class="tab" data-page="journal">Journal</div>
  <div class="tab" data-page="logs">Logs</div>
</div>

<!-- ================= OVERVIEW ================= -->
<div class="page active" id="page-overview">
<div class="container">
  <div class="grid grid-4" style="margin-bottom:16px">
    <div class="card stat-card">
      <div class="stat-label">Total P&L</div>
      <div class="stat-value" id="stat-pnl">$0.00</div>
      <div class="stat-sub" id="stat-pnl-pct"></div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Total Trades</div>
      <div class="stat-value" id="stat-trades">0</div>
      <div class="stat-sub" id="stat-winrate">0% win rate</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Scanner Hits</div>
      <div class="stat-value text-blue" id="stat-scans">0</div>
      <div class="stat-sub"><span id="stat-signals">0</span> signals / <span id="stat-rejected">0</span> rejected</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Account Equity</div>
      <div class="stat-value" id="stat-equity">$0</div>
      <div class="stat-sub">Cash: <span id="stat-cash">$0</span></div>
    </div>
  </div>

  <div class="grid grid-2">
    <div class="card">
      <div class="card-header"><h3>Open Positions</h3><span class="badge pill-blue" id="pos-count">0</span></div>
      <div id="positions-table-wrap">
        <div class="empty"><div class="icon">&#128200;</div>No open positions</div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><h3>Recent Activity</h3></div>
      <div id="activity-wrap">
        <div class="empty"><div class="icon">&#9889;</div>Waiting for trades...</div>
      </div>
    </div>
  </div>

  <div class="grid grid-2" style="margin-top:16px">
    <div class="card">
      <div class="card-header"><h3>Entry Checks</h3><span class="badge pill-blue" id="ov-scan-count2">0</span></div>
      <div id="ov-scanner-wrap">
        <div class="empty"><div class="icon">&#128269;</div>No entry checks yet</div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><h3>ML Model Stats</h3><span class="badge pill-purple" id="ml-status-badge">--</span></div>
      <div id="ml-stats-wrap">
        <div class="empty"><div class="icon">&#129302;</div>ML stats loading...</div>
      </div>
    </div>
  </div>
</div>
</div>

<!-- ================= SCANNER (HOD Momentum feed) ================= -->
<div class="page" id="page-scanner">
<div class="container">
  <div class="grid grid-4" style="margin-bottom:16px">
    <div class="card stat-card">
      <div class="stat-label">HOD Alerts</div>
      <div class="stat-value text-green" id="hod-momentum-count">0</div>
      <div class="stat-sub">Watchlist follows this board</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Verified Signals</div>
      <div class="stat-value text-green" id="scan-signals">0</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Pattern Hits</div>
      <div class="stat-value text-yellow" id="scan-total">0</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Trading Watchlist</div>
      <div class="stat-value text-blue" id="trading-watchlist-count">0</div>
      <div class="stat-sub">HOD + Hot Watch symbols</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <div class="card-header">
      <h3>Trading Watchlist</h3>
      <span style="font-size:11px;color:var(--text2)">Active scan universe from HOD alerts and Hot Watch — SPY is pinned for market panic</span>
    </div>
    <div id="trading-watchlist-wrap">
      <div class="empty"><div class="icon">&#128203;</div>No symbols on watchlist yet</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <div class="card-header">
      <h3>Hot Watch</h3>
      <span style="font-size:11px;color:var(--text2)">Early fast-scan movers waiting for clean pullback or breakout</span>
      <span class="badge pill-yellow" id="hot-watch-count">0</span>
    </div>
    <div id="candidate-worker-status" style="font-size:12px;color:var(--text2);margin-bottom:10px"></div>
    <div id="hot-watch-wrap">
      <div class="empty"><div class="icon">&#128293;</div>No hot-watch movers yet</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <div class="card-header">
      <h3>HOD Momentum Scanner</h3>
      <span style="font-size:11px;color:var(--text2)">Chg % = from today open (or vs prior close if bars truncated) · vs Close % = like TradingView day change</span>
    </div>
    <div class="hod-filters" id="hod-filters">
      <button class="hod-filter-btn active" data-filter="all">All</button>
      <button class="hod-filter-btn" data-filter="New HOD Breakout">New HOD</button>
      <button class="hod-filter-btn" data-filter="Today HOD Breakout">Today HOD</button>
      <button class="hod-filter-btn" data-filter="HOD Reclaim">HOD Reclaim</button>
      <button class="hod-filter-btn" data-filter="Former Momo Stock">Former Momo</button>
      <button class="hod-filter-btn" data-filter="Low Float - High Rel Vol">Low Float</button>
      <button class="hod-filter-btn" data-filter="Squeeze - Up 5% in 5min">Squeeze 5%</button>
      <button class="hod-filter-btn" data-filter="Squeeze - Up 10% in 10min">Squeeze 10%</button>
    </div>
    <div class="hod-momentum-scanner" id="hod-momentum-scanner-wrap">
      <div class="empty"><div class="icon">&#128293;</div>Waiting for HOD momentum alerts...</div>
    </div>
  </div>
</div>
</div>

<!-- ================= TRADES ================= -->
<div class="page" id="page-trades">
<div class="container">
  <div class="grid grid-4" style="margin-bottom:16px">
    <div class="card stat-card">
      <div class="stat-label">Total P&L</div>
      <div class="stat-value" id="trades-pnl">$0.00</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Winners</div>
      <div class="stat-value text-green" id="trades-wins">0</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Losers</div>
      <div class="stat-value text-red" id="trades-losses">0</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Win Rate</div>
      <div class="stat-value" id="trades-winrate">0%</div>
    </div>
  </div>
  <div class="card" style="margin-bottom:16px">
    <div class="card-header"><h3>Daily P&L by Stock</h3></div>
    <div id="stock-summary-wrap">
      <div class="empty"><div class="icon">&#128202;</div>No trades yet</div>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><h3>Trade History</h3></div>
    <div id="trades-table-wrap">
      <div class="empty"><div class="icon">&#128202;</div>No trades yet</div>
    </div>
  </div>
</div>
</div>

<!-- ================= ANALYTICS ================= -->
<div class="page" id="page-analytics">
<div class="container">
  <div class="card" style="margin-bottom:16px">
    <div class="card-header">
      <h3>Daily Scorecard</h3>
      <span id="scorecard-verdict" class="pill pill-yellow">COLLECTING</span>
    </div>
    <div id="daily-scorecard-wrap">
      <div class="empty"><div class="icon">&#128202;</div>Collecting scorecard data...</div>
    </div>
  </div>
  <div class="grid grid-4" style="margin-bottom:16px">
    <div class="card stat-card">
      <div class="stat-label">Avg Win</div>
      <div class="stat-value text-green" id="an-avg-win">$0.00</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Avg Loss</div>
      <div class="stat-value text-red" id="an-avg-loss">$0.00</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Profit Factor</div>
      <div class="stat-value" id="an-pf">0.00</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Expectancy</div>
      <div class="stat-value" id="an-expect">$0.00</div>
    </div>
  </div>
  <div class="card" style="margin-bottom:16px">
    <div class="card-header">
      <h3>Missed A+ Setups</h3>
      <span style="font-size:11px;color:var(--text2)">Blocked elite setups labeled by later price action</span>
    </div>
    <div id="missed-a-plus-wrap">
      <div class="empty"><div class="icon">&#128269;</div>No blocked A+ setups labeled yet</div>
    </div>
  </div>
  <div class="grid grid-2" style="margin-bottom:16px">
    <div class="card">
      <div class="card-header"><h3>Per-Symbol Breakdown</h3></div>
      <div id="an-symbol-table">
        <div class="empty"><div class="icon">&#128202;</div>No data yet</div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><h3>Exit Type Analysis</h3></div>
      <div id="an-exit-table">
        <div class="empty"><div class="icon">&#128202;</div>No data yet</div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><h3>Equity Curve</h3></div>
    <div id="an-equity-chart" style="height:200px;position:relative;overflow:hidden;">
      <canvas id="equity-canvas" style="width:100%;height:100%"></canvas>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <div class="card-header">
      <h3>ML Learning Report</h3>
      <div style="display:flex;gap:8px;align-items:center">
        <span id="ml-report-day" style="font-size:11px;color:var(--text2)">Not loaded</span>
        <button id="ml-report-refresh-btn" style="padding:6px 10px;border:1px solid var(--border);background:var(--surface2);color:var(--text);border-radius:6px;cursor:pointer">Refresh</button>
      </div>
    </div>
    <div id="ml-learning-wrap">
      <div class="empty"><div class="icon">&#129302;</div>ML learning report loading...</div>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <div class="card-header"><h3>ML Progress</h3></div>
    <div id="ml-progress-wrap">
      <div class="empty"><div class="icon">&#128200;</div>ML progress loading...</div>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <div class="card-header">
      <h3>AI Trade Insights</h3>
      <span id="ai-score" style="font-size:13px;color:var(--text2)"></span>
    </div>
    <div id="ai-insights-wrap">
      <div class="empty"><div class="icon">&#129302;</div>AI analysis will run after 5+ trades</div>
    </div>
  </div>
  <div class="grid grid-2" style="margin-top:16px">
    <div class="card">
      <div class="card-header"><h3>Blocked Symbols</h3></div>
      <div id="ai-blocked-wrap">
        <div class="empty" style="padding:12px;font-size:12px;color:var(--text2)">No symbols blocked</div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><h3>Auto-Adjustments</h3></div>
      <div id="ai-adjustments-wrap">
        <div class="empty" style="padding:12px;font-size:12px;color:var(--text2)">No adjustments yet</div>
      </div>
    </div>
  </div>
</div>
</div>

<!-- ================= BACKTEST ================= -->
<div class="page" id="page-backtest">
<div class="container">
  <div class="card" style="margin-bottom:16px">
    <div class="card-header">
      <h3>Backtest</h3>
      <span style="font-size:11px;color:var(--text2)">Single-symbol historical replay through the real entry pipeline</span>
    </div>
    <div style="display:grid;grid-template-columns:120px 170px 110px 1fr 100px;gap:10px;align-items:end">
      <label style="font-size:12px;color:var(--text2)">Symbol<br>
        <input id="bt-symbol" value="CUPR" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface2);color:var(--text);text-transform:uppercase">
      </label>
      <label style="font-size:12px;color:var(--text2)">Date<br>
        <input id="bt-date" placeholder="YYYY-MM-DD or DD/MM" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface2);color:var(--text)">
      </label>
      <label style="font-size:12px;color:var(--text2)">Start ET<br>
        <input id="bt-start-time" placeholder="optional" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface2);color:var(--text)">
      </label>
      <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--text2);padding-bottom:7px">
        <label><input type="checkbox" id="bt-flag-fresh"> fresh VWAP scout</label>
        <label><input type="checkbox" id="bt-flag-vwap-scout"> VWAP reclaim scout</label>
        <label><input type="checkbox" id="bt-flag-level"> level breakout scout</label>
        <label><input type="checkbox" id="bt-flag-spread"> elite wide spread</label>
        <label><input type="checkbox" id="bt-flag-momentum"> momentum burst live</label>
        <label title="Replay the rapid momentum-burst hit-and-run state machine from real 10s bars"><input type="checkbox" id="bt-flag-mb-hit-run"> momentum hit-run</label>
        <label><input type="checkbox" id="bt-flag-capped"> level-capped entry</label>
        <label><input type="checkbox" id="bt-flag-timer" checked> 10s timer replay</label>
        <label><input type="checkbox" id="bt-flag-10s-scout"> 10s breakout scout</label>
        <label title="Replay the runner HOD-alert quick-scalp path from real 10s bars"><input type="checkbox" id="bt-flag-breakout-scalp"> breakout scalp replay</label>
        <label title="Run scans + exits on a 10s clock with partial 1m bars — closest to paper for fast scalps"><input type="checkbox" id="bt-flag-live-like"> live-like 10s (paper-faithful)</label>
      </div>
      <button id="bt-run" style="padding:9px 12px;border:1px solid var(--blue);background:var(--blue-bg);color:var(--blue);border-radius:6px;cursor:pointer;font-weight:700">Run</button>
    </div>
    <div class="mini-muted" style="margin-top:8px">This uses simulated fills only. It never sends live or paper orders. Date accepts YYYY-MM-DD or DD/MM. Start ET accepts HH:MM, for example 10:10.</div>
  </div>
  <div id="backtest-wrap">
    <div class="empty"><div class="icon">&#128202;</div>Choose a symbol and date, then run a backtest.</div>
  </div>
</div>
</div>

<!-- ================= LOGS ================= -->
<div class="page" id="page-logs">
<div class="container">
  <div class="card">
    <div class="card-header"><h3>Live Logs</h3></div>
    <div class="log-panel" id="log-panel">
      <div class="empty"><div class="icon">&#128196;</div>No log messages yet</div>
    </div>
  </div>
</div>
</div>

<!-- ================= JOURNAL ================= -->
<div class="page" id="page-journal">
<div class="container">
  <div class="grid grid-4" style="margin-bottom:16px">
    <div class="card stat-card">
      <div class="stat-label">Replay Events</div>
      <div class="stat-value text-blue" id="jr-total">0</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Trade Events</div>
      <div class="stat-value text-green" id="jr-trades">0</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Mistakes</div>
      <div class="stat-value text-red" id="jr-mistakes">0</div>
    </div>
    <div class="card stat-card">
      <div class="stat-label">Screenshots</div>
      <div class="stat-value text-yellow" id="jr-shots">0</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <div class="card-header">
      <h3>Journal Replay</h3>
      <div style="display:flex;gap:8px;align-items:center">
        <span id="jr-updated" style="font-size:11px;color:var(--text2)">Not loaded</span>
        <button id="jr-refresh-btn" style="padding:6px 10px;border:1px solid var(--border);background:var(--surface2);color:var(--text1);border-radius:6px;cursor:pointer">Refresh</button>
      </div>
    </div>
    <div id="journal-table-wrap">
      <div class="empty"><div class="icon">&#128221;</div>No journal events yet</div>
    </div>
  </div>
</div>
</div>

<script>
// State
let state = {
  stats: {}, account: {}, positions: {}, symbols: {},
  recent_trades: [], recent_scans: [], pnl_history: [],
  daily_scorecard: {}, rolling_scorecard: {},
  market_open: false, stream_connected: false,
  watchlist_scan: [], rt_movers: [], hod_momentum_alerts: [], hot_watch: [], trading_watchlist: [],
  missed_a_plus: [],
  candidate_hydration: {},
  watchlist_pinned: ['SPY'],
  hod_momentum_filter: 'all', rt_new_total: 0,
  backtest: {loading: false, result: null, error: null, chart: {start: null, end: null, dragX: null}},
  news: {}, journal: {events: [], loaded: false, error: null, last_update: null},
  logs: [],
  ml_report: {report: null, loaded: false, error: null, message: null, last_update: null}
};

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('page-' + tab.dataset.page).classList.add('active');
    if (tab.dataset.page === 'journal') {
      loadJournal(true);
    }
    if (tab.dataset.page === 'analytics') {
      loadMLReport(true);
    }
  });
});

let btDate = document.getElementById('bt-date');
if (btDate && !btDate.value) {
  btDate.value = new Date().toISOString().slice(0, 10);
}
let btRun = document.getElementById('bt-run');
if (btRun) {
  btRun.addEventListener('click', runBacktest);
}

// Formatting helpers
function fmt$(v) { return (v>=0?'':'') + '$' + Math.abs(v).toFixed(2); }
function fmtPnl(v) {
  let s = v >= 0 ? '+$' + v.toFixed(2) : '-$' + Math.abs(v).toFixed(2);
  return '<span class="' + (v>=0?'text-green':'text-red') + '">' + s + '</span>';
}
function chartLink(sym) {
  let url = 'https://www.tradingview.com/chart/?symbol=' + encodeURIComponent(sym);
  return '<a href="' + url + '" target="_blank" rel="noopener" style="text-decoration:none" title="Open chart for ' + sym + '"><strong>' + sym + '</strong> <span style="font-size:11px;opacity:0.6">&#128200;</span></a>';
}
function fmtCompact(v) {
  let n = Number(v || 0);
  let abs = Math.abs(n);
  if (abs >= 1000000000) return (n / 1000000000).toFixed(abs >= 10000000000 ? 0 : 1) + 'B';
  if (abs >= 1000000) return (n / 1000000).toFixed(abs >= 10000000 ? 0 : 1) + 'M';
  if (abs >= 1000) return (n / 1000).toFixed(abs >= 10000 ? 0 : 1) + 'K';
  return n.toFixed(0);
}
function confBar(pct) {
  let color = pct >= 60 ? 'var(--green)' : pct >= 40 ? 'var(--yellow)' : 'var(--red)';
  return pct.toFixed(0) + '%<div class="conf-bar"><div class="conf-bar-fill" style="width:'+pct+'%;background:'+color+'"></div></div>';
}
function typePill(t) {
  let cls = {entry:'pill-blue',exit:'pill-green',scale_up:'pill-purple',reentry:'pill-yellow'}[t]||'pill-blue';
  return '<span class="pill '+cls+'">'+t.replace('_',' ').toUpperCase()+'</span>';
}
function sidePill(s) {
  return '<span class="pill '+(s==='buy'?'pill-green':'pill-red')+'">'+s.toUpperCase()+'</span>';
}
function stylePill(s) {
  let cls = {scalping:'pill-blue',day_trading:'pill-purple',swing:'pill-yellow',not_tradeable:'pill-red'}[s]||'pill-blue';
  return '<span class="pill '+cls+'">'+s.replace('_',' ').toUpperCase()+'</span>';
}
function newsPill(sentiment, score) {
  if (!sentiment) return '';
  let cls = sentiment === 'positive' ? 'pill-green' : sentiment === 'negative' ? 'pill-red' : 'pill-yellow';
  let label = sentiment.toUpperCase() + ' (' + (score >= 0 ? '+' : '') + score.toFixed(1) + ')';
  return '<span class="pill '+cls+'">'+label+'</span>';
}
function shortTime(s) {
  if (!s) return '';
  try { let d=new Date(s); return d.toLocaleTimeString(); } catch(e) { return s; }
}
function escapeHtml(v) {
  return String(v || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Render functions
function renderOverview() {
  let s = state.stats;
  let a = state.account;

  let pnlEl = document.getElementById('stat-pnl');
  pnlEl.innerHTML = fmtPnl(s.total_pnl||0);
  if (a.starting_cash > 0) {
    document.getElementById('stat-pnl-pct').textContent =
      ((s.total_pnl||0)/a.starting_cash*100).toFixed(2) + '% return';
  }

  document.getElementById('stat-trades').textContent = s.total_trades||0;
  document.getElementById('stat-winrate').textContent = (s.win_rate||0) + '% win rate';
  document.getElementById('stat-scans').textContent = s.total_scan_hits||0;
  document.getElementById('stat-signals').textContent = s.total_signals||0;
  document.getElementById('stat-rejected').textContent = s.total_rejected||0;
  document.getElementById('stat-equity').textContent = '$' + (a.equity||0).toLocaleString();
  document.getElementById('stat-cash').textContent = '$' + (a.cash||0).toLocaleString();

  renderOverviewScanner();
  renderPositionsCompact();
  renderActivity();
}

function renderOverviewScanner() {
  let wrap = document.getElementById('ov-scanner-wrap');
  if (!wrap) return;
  let scans = state.recent_scans.slice(-10).reverse();
  let countEl = document.getElementById('ov-scan-count2');
  if (countEl) countEl.textContent = scans.length;

  if (scans.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128269;</div>No entry checks yet</div>';
    return;
  }
  let html = '<table><tr><th>Time</th><th>Symbol</th><th>Price</th><th>Pattern</th><th>Tier</th><th>Score</th><th>Entry Check</th></tr>';
  scans.forEach(h => {
    let status = '';
    if (h.verified) {
      status = '<span class="pill pill-green">A+ SIGNAL</span>';
    } else if (h.action_taken) {
      status = renderHumanScanStatus(h.action_taken);
    } else {
      status = '<span style="color:var(--text2);font-size:11px">pending</span>';
    }
    html += '<tr><td style="color:var(--text2)">'+shortTime(h.time)+'</td>';
    html += '<td>'+chartLink(h.symbol)+'</td>';
    html += '<td>$'+(h.price||0).toFixed(2)+'</td>';
    html += '<td><span class="pill pill-blue">'+h.scanner_name+'</span></td>';
    html += '<td>'+renderSetupTier(h)+'</td>';
    html += '<td>'+h.score.toFixed(2)+'</td>';
    html += '<td>'+status+'</td></tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
}

const A_PLUS_SCANNERS = new Set([
  'vwap_pullback',
  'abc_continuation',
  'first_pullback_reclaim',
  'hod_reclaim',
  'pullback_base',
  'runner_readd'
]);

function setupTier(hit) {
  let explicit = hit && hit.criteria ? String(hit.criteria.setup_tier || '').toLowerCase() : '';
  if (explicit.includes('a+')) return 'A+';
  if (explicit.includes('watch')) return 'Watch';
  return A_PLUS_SCANNERS.has(hit.scanner_name) ? 'A+' : 'Watch';
}

function renderSetupTier(hit) {
  let tier = setupTier(hit);
  let cls = tier === 'A+' ? 'pill-green' : 'pill-yellow';
  let title = tier === 'A+'
    ? 'Live A+ setup: can continue to guard, ML, 10-second confirmation, and risk checks'
    : 'Watch only: monitored for learning/watchlist, not a live trade setup by itself';
  return '<span class="pill '+cls+'" title="'+escapeHtml(title)+'">'+tier+'</span>';
}

function humanScanStatus(reason) {
  let raw = String(reason || '');
  let r = raw.toLowerCase();
  let label = 'Rule rejected';
  let cls = 'pill-red';

  if (r.includes('stale data')) {
    label = r.includes('halt') ? 'Halt/data issue' : 'Data issue';
  } else if (r.includes('watch only')) {
    label = 'Watch only';
    cls = 'pill-yellow';
  } else if (r.includes('tape too slow') || r.includes('recent volume')) {
    label = 'Weak tape';
  } else if (r.includes('thin sub-$5 liquidity') || r.includes('liquidity')) {
    label = 'Thin liquidity';
  } else if (r.includes('not enough movement') || r.includes('movement too small')) {
    label = 'Not enough movement';
  } else if (r.includes('outside range') || r.includes('price band')) {
    label = 'Outside price range';
  } else if (r.includes('late pullback not strong above vwap')) {
    let m = raw.match(/VWAP\\s+([0-9.]+)%/i);
    let pct = m ? Number(m[1]) : 0;
    if (pct >= 0.7) {
      label = 'Borderline VWAP';
      cls = 'pill-yellow';
    } else {
      label = 'Weak VWAP';
    }
  } else if (r.includes('below vwap')) {
    label = 'Below VWAP';
  } else if (r.includes('spread')) {
    label = 'Wide spread';
  } else if (r.includes('cooldown')) {
    label = 'Cooldown';
    cls = 'pill-yellow';
  } else if (r.includes('hod momentum') || r.includes('hod board')) {
    label = 'Waiting for HOD';
    cls = 'pill-yellow';
  } else if (r.includes('ml')) {
    label = 'ML rejected';
  } else if (r.includes('risk') || r.includes('r:r')) {
    label = 'Risk rejected';
  } else if (r.includes('negative news')) {
    label = 'Bad news';
  }

  return { label, cls, raw };
}

function renderHumanScanStatus(reason) {
  let s = humanScanStatus(reason);
  return '<span class="pill '+s.cls+'" title="'+escapeHtml(s.raw)+'">\u2718 '+escapeHtml(s.label)+'</span>';
}

function renderPositionsCompact() {
  let wrap = document.getElementById('positions-table-wrap');
  let keys = Object.keys(state.positions);
  document.getElementById('pos-count').textContent = keys.length;

  if (keys.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128200;</div>No open positions</div>';
    return;
  }
  let html = '<table><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Avg</th><th>Current</th><th>P&L</th></tr>';
  keys.forEach(sym => {
    let p = state.positions[sym];
    html += '<tr><td><strong>'+sym+'</strong></td><td>'+sidePill(p.side.toLowerCase())+'</td>';
    html += '<td>'+p.quantity+'</td><td>$'+p.avg_price.toFixed(2)+'</td>';
    html += '<td>$'+p.current_price.toFixed(2)+'</td><td>'+fmtPnl(p.unrealized_pnl)+'</td></tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
}

function renderActivity() {
  let wrap = document.getElementById('activity-wrap');
  let exits = state.recent_trades.filter(t => t.trade_type === 'exit').slice(-15).reverse();
  let entries = state.recent_trades.filter(t => t.trade_type === 'entry').slice(-5).reverse();
  if (exits.length === 0 && entries.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#9889;</div>Waiting for trades...</div>';
    return;
  }
  let html = '';
  if (exits.length > 0) {
    html += '<table><tr><th>Time</th><th>Symbol</th><th>Qty</th><th>Entry</th><th>Exit</th><th>P&L</th></tr>';
    exits.forEach(t => {
      let pnl = t.pnl !== null && t.pnl !== undefined ? fmtPnl(t.pnl) : '-';
      html += '<tr><td style="color:var(--text2)">'+shortTime(t.exit_time||t.entry_time)+'</td>';
      html += '<td>'+chartLink(t.symbol)+'</td><td>'+t.quantity+'</td>';
      html += '<td>$'+(t.entry_price||0).toFixed(2)+'</td>';
      html += '<td>$'+(t.exit_price||0).toFixed(2)+'</td><td>'+pnl+'</td></tr>';
    });
    html += '</table>';
  }
  if (entries.length > 0) {
    html += '<div style="margin-top:8px;font-size:12px;color:var(--text2)">Recent entries: ';
    html += entries.map(t => chartLink(t.symbol)+' '+t.quantity+' @ $'+t.entry_price.toFixed(2)).join(' &middot; ');
    html += '</div>';
  }
  wrap.innerHTML = html;
}

function renderScanner() {
  let s = state.stats;
  let scanTotal = document.getElementById('scan-total');
  let scanSignals = document.getElementById('scan-signals');
  if (scanTotal) scanTotal.textContent = s.total_scan_hits||0;
  if (scanSignals) scanSignals.textContent = s.total_signals||0;
  renderCandidateWorker();
  renderTradingWatchlist();
  renderHotWatch();
  renderHodMomentumScanner();
}

function renderCandidateWorker() {
  let el = document.getElementById('candidate-worker-status');
  if (!el) return;
  let c = state.candidate_hydration || {};
  let paused = c.paused_for_entry ? '<span class="pill pill-yellow">ENTRY TIMER PRIORITY</span>' : '<span class="pill pill-green">READY</span>';
  el.innerHTML = 'Candidate Worker ' + paused
    + ' &middot; pending ' + (c.pending || 0)
    + ' &middot; batches ' + (c.batches || 0)
    + ' &middot; hydrated ' + (c.hydrated || 0)
    + ' &middot; skipped fresh ' + (c.skipped_fresh || 0)
    + ' &middot; dropped ' + (c.dropped || 0);
}

function renderTradingWatchlist() {
  let wrap = document.getElementById('trading-watchlist-wrap');
  if (!wrap) return;
  let hotSyms = (state.hot_watch || [])
    .map(h => String(h.symbol || '').trim().toUpperCase())
    .filter(Boolean);
  let syms = (state.trading_watchlist || [])
    .map(s => String(s || '').trim().toUpperCase())
    .filter(Boolean);
  hotSyms.forEach(sym => {
    if (!syms.includes(sym)) syms.push(sym);
  });
  let countEl = document.getElementById('trading-watchlist-count');

  if (syms.length === 0) {
    if (countEl) countEl.textContent = '0';
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128203;</div>No symbols on watchlist yet</div>';
    return;
  }

  let pinned = new Set(state.watchlist_pinned || ['SPY']);
  let alertBySym = {};
  (state.hod_momentum_alerts || []).forEach(a => {
    if (!alertBySym[a.symbol]) alertBySym[a.symbol] = a;
  });

  let tradeSyms = syms.filter(s => !pinned.has(s));
  if (countEl) countEl.textContent = tradeSyms.length;

  if (tradeSyms.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128203;</div>Waiting for HOD alerts or Hot Watch movers...</div>';
    return;
  }

  let hotBySym = {};
  (state.hot_watch || []).forEach(h => { hotBySym[h.symbol] = h; });
  let html = '<table><tr><th>Symbol</th><th>Source</th><th>Latest</th></tr>';
  tradeSyms.forEach(sym => {
    let a = alertBySym[sym];
    let h = hotBySym[sym];
    let onBoard = a
      ? '<span class="pill pill-green">YES</span>'
      : (h ? hotWatchModePill(h.mode) : '<span class="pill pill-yellow">waiting</span>');
    let alertCell = a
      ? '<span class="hod-alert-name">' + escapeHtml(a.alert_name) + '</span> @ $' + (a.price||0).toFixed(2)
      : (h ? '<span style="color:var(--text2);font-size:11px">Hot Watch session ' + (h.change_pct||0).toFixed(1) + '% · now ' + hotWatchNowText(h) + ' · vol ' + fmtCompact(h.volume||0) + '</span>'
           : '<span style="color:var(--text2);font-size:11px">TTL active — no new alert yet</span>');
    html += '<tr><td>' + chartLink(sym) + '</td><td>' + onBoard + '</td><td>' + alertCell + '</td></tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
}

function hotWatchModePill(mode) {
  let map = {
    runner_watch: {label: 'Runner', cls: 'pill-green'},
    strong_watch: {label: 'Strong', cls: 'pill-blue'},
    watch: {label: 'Watch', cls: 'pill-yellow'}
  };
  let m = map[mode] || {label: mode || 'Watch', cls: 'pill-yellow'};
  return '<span class="pill '+m.cls+'">'+escapeHtml(m.label)+'</span>';
}

function hotWatchNowText(row) {
  let pull = Number(row.pullback_from_high_pct);
  let short = Number(row.short_change_pct);
  if (Number.isFinite(pull) && pull < -0.1) {
    return pull.toFixed(1) + '% from high';
  }
  if (Number.isFinite(short)) {
    return (short >= 0 ? '+' : '') + short.toFixed(1) + '% short-term';
  }
  return 'checking';
}

function hotWatchNowClass(row) {
  let pull = Number(row.pullback_from_high_pct);
  let short = Number(row.short_change_pct);
  if (Number.isFinite(pull) && pull <= -2.0) return 'text-red';
  if (Number.isFinite(short)) return short >= 0 ? 'text-green' : 'text-red';
  return '';
}

function renderHotWatch() {
  let wrap = document.getElementById('hot-watch-wrap');
  if (!wrap) return;
  let rows = (state.hot_watch || []).slice();
  let countEl = document.getElementById('hot-watch-count');
  if (countEl) countEl.textContent = rows.length;

  if (rows.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128293;</div>No hot-watch movers yet</div>';
    return;
  }

  let html = '<table><tr><th>Symbol</th><th>Mode</th><th>Time Left</th><th>Price</th><th>Session</th><th>Now</th><th>Volume</th><th>Score</th><th>Why</th></tr>';
  rows.forEach(r => {
    let remaining = Math.max(0, Number(r.remaining_seconds || 0));
    let mins = Math.ceil(remaining / 60);
    let chg = Number(r.change_pct || r.abs_change_pct || 0);
    let volume = Number(r.volume || 0);
    let score = Number(r.score || 0);
    let price = Number(r.price || 0);
    html += '<tr>';
    html += '<td>' + chartLink(r.symbol) + '</td>';
    html += '<td>' + hotWatchModePill(r.mode) + '</td>';
    html += '<td>' + mins + 'm</td>';
    html += '<td><strong>$' + price.toFixed(2) + '</strong></td>';
    html += '<td class="' + (chg >= 0 ? 'text-green' : 'text-red') + '">' + (chg >= 0 ? '+' : '') + chg.toFixed(1) + '%</td>';
    html += '<td class="' + hotWatchNowClass(r) + '">' + escapeHtml(hotWatchNowText(r)) + '</td>';
    html += '<td>' + volume.toLocaleString() + '</td>';
    html += '<td>' + score.toFixed(2) + '</td>';
    html += '<td style="color:var(--text2);font-size:11px">' + escapeHtml(r.reason || '') + '</td>';
    html += '</tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
}

function renderHodMomentumScanner() {
  let wrap = document.getElementById('hod-momentum-scanner-wrap');
  if (!wrap) return;

  let alerts = (state.hod_momentum_alerts || []).slice();
  alerts.sort((a, b) => new Date(b.time || 0) - new Date(a.time || 0));
  let filter = state.hod_momentum_filter || 'all';
  if (filter !== 'all') {
    alerts = alerts.filter(a => a.alert_name === filter);
  }

  let countEl = document.getElementById('hod-momentum-count');
  if (countEl) countEl.textContent = (state.hod_momentum_alerts || []).length;

  if (alerts.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128293;</div>No alerts match this filter</div>';
    return;
  }

  let html = '<table><tr>'
    + '<th>Time</th><th>Symbol</th><th></th><th>Price</th><th>Volume</th><th>Float</th>'
    + '<th>Rel Vol</th><th>Bar RV</th><th>Chg %</th><th>Gap %</th><th>vs Close %</th><th>From Low %</th><th>Source</th><th>Alert</th><th>Status</th>'
    + '</tr>';

  alerts.forEach(a => {
    let hot = a.hot ? '<span class="hod-hot" title="Hot mover">&#128293;</span>' : '';
    let chgClass = (a.change_session_pct || 0) >= 0 ? 'text-green' : 'text-red';
    let status = '';
    if (a.verified) {
      status = '<span class="pill pill-green">SIGNAL</span>';
    } else if (a.reject_reason) {
      status = renderHumanScanStatus(a.reject_reason);
    } else {
      status = '<span class="pill pill-yellow">ALERT</span>';
    }
    html += '<tr class="'+(a.row_class||'hod-default')+'">';
    html += '<td style="color:var(--text2)">'+shortTime(a.time)+'</td>';
    html += '<td>'+chartLink(a.symbol)+'</td>';
    html += '<td>'+hot+'</td>';
    html += '<td><strong>$'+(a.price||0).toFixed(2)+'</strong></td>';
    html += '<td>'+(a.day_volume_fmt||'—')+'</td>';
    html += '<td>'+(a.float_fmt||'—')+'</td>';
    html += '<td class="'+chgClass+'">'+(a.rel_vol||0).toFixed(2)+'x</td>';
    html += '<td>'+(a.bar_rvol||0).toFixed(2)+'x</td>';
    html += '<td class="'+chgClass+'">'+((a.change_session_pct||0) >= 0 ? '+' : '')+(a.change_session_pct||0).toFixed(2)+'%</td>';
    let gap = a.gap_pct != null ? (a.gap_pct >= 0 ? '+' : '') + a.gap_pct.toFixed(2) + '%' : '—';
    let vsYday = a.change_from_close_pct != null ? (a.change_from_close_pct >= 0 ? '+' : '') + a.change_from_close_pct.toFixed(2) + '%' : '—';
    html += '<td class="'+chgClass+'">'+gap+'</td>';
    html += '<td class="'+chgClass+'">'+vsYday+'</td>';
    html += '<td class="'+chgClass+'">+'+(a.change_from_low_pct||0).toFixed(2)+'%</td>';
    let src = a.source === 'tick'
      ? '<span class="hod-source-tick">TICK</span>'
      : '<span class="hod-source-bar">BAR</span>';
    let alertLabel = escapeHtml(a.alert_name) + (a.burst_text ? ' '+escapeHtml(a.burst_text) : '');
    html += '<td>'+src+'</td>';
    html += '<td><span class="hod-alert-name">'+alertLabel+'</span></td>';
    html += '<td>'+status+'</td></tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
}

function renderTrades() {
  let s = state.stats;
  document.getElementById('trades-pnl').innerHTML = fmtPnl(s.total_pnl||0);
  document.getElementById('trades-wins').textContent = s.winning_trades||0;
  document.getElementById('trades-losses').textContent = s.losing_trades||0;
  document.getElementById('trades-winrate').textContent = (s.win_rate||0) + '%';

  let wrap = document.getElementById('trades-table-wrap');
  let trades = state.recent_trades.filter(t => t.trade_type !== 'entry' || t.pnl != null).slice(-50).reverse();
  if (state.recent_trades.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128202;</div>No trades yet</div>';
    return;
  }
  let html = '<table><tr><th>Time</th><th>Type</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr>';
  state.recent_trades.slice(-50).reverse().forEach(t => {
    let pnl = t.pnl !== null && t.pnl !== undefined ? fmtPnl(t.pnl) : '-';
    html += '<tr><td style="color:var(--text2)">'+shortTime(t.exit_time||t.entry_time)+'</td>';
    html += '<td>'+typePill(t.trade_type)+'</td><td>'+chartLink(t.symbol)+'</td>';
    html += '<td>'+sidePill(t.side)+'</td><td>'+t.quantity+'</td>';
    html += '<td>$'+t.entry_price.toFixed(2)+'</td>';
    html += '<td>'+(t.exit_price?'$'+t.exit_price.toFixed(2):'-')+'</td>';
    html += '<td>'+pnl+'</td>';
    html += '<td>'+(t.exit_reason||'-')+'</td></tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
  renderStockSummary();
}

function renderStockSummary() {
  let wrap = document.getElementById('stock-summary-wrap');
  let exits = state.recent_trades.filter(t => t.trade_type === 'exit' && t.pnl != null);
  if (exits.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128202;</div>No completed trades yet</div>';
    return;
  }
  let stocks = {};
  exits.forEach(t => {
    if (!stocks[t.symbol]) stocks[t.symbol] = {pnl: 0, wins: 0, losses: 0, trades: 0, totalQty: 0};
    let s = stocks[t.symbol];
    s.pnl += t.pnl;
    s.trades++;
    s.totalQty += t.quantity;
    if (t.pnl >= 0) s.wins++; else s.losses++;
  });
  let rows = Object.entries(stocks).sort((a,b) => b[1].pnl - a[1].pnl);
  let totalPnl = rows.reduce((s,r) => s + r[1].pnl, 0);
  let totalTrades = rows.reduce((s,r) => s + r[1].trades, 0);
  let totalWins = rows.reduce((s,r) => s + r[1].wins, 0);
  let totalLosses = rows.reduce((s,r) => s + r[1].losses, 0);
  let html = '<table><tr><th>Symbol</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Shares</th><th>P&L</th></tr>';
  rows.forEach(([sym, s]) => {
    let wr = s.trades > 0 ? ((s.wins / s.trades) * 100).toFixed(0) : '0';
    let pnlClass = s.pnl >= 0 ? 'text-green' : 'text-red';
    let wrClass = parseInt(wr) >= 50 ? 'text-green' : 'text-red';
    html += '<tr><td>' + chartLink(sym) + '</td>';
    html += '<td>' + s.trades + '</td>';
    html += '<td class="text-green">' + s.wins + '</td>';
    html += '<td class="text-red">' + s.losses + '</td>';
    html += '<td class="' + wrClass + '">' + wr + '%</td>';
    html += '<td>' + s.totalQty.toLocaleString() + '</td>';
    html += '<td class="' + pnlClass + '"><strong>' + fmtPnl(s.pnl) + '</strong></td></tr>';
  });
  let totalWR = totalTrades > 0 ? ((totalWins / totalTrades) * 100).toFixed(0) : '0';
  let totalClass = totalPnl >= 0 ? 'text-green' : 'text-red';
  html += '<tr style="border-top:2px solid var(--border);font-weight:bold"><td>TOTAL</td>';
  html += '<td>' + totalTrades + '</td>';
  html += '<td class="text-green">' + totalWins + '</td>';
  html += '<td class="text-red">' + totalLosses + '</td>';
  html += '<td>' + totalWR + '%</td>';
  html += '<td></td>';
  html += '<td class="' + totalClass + '">' + fmtPnl(totalPnl) + '</td></tr>';
  html += '</table>';
  wrap.innerHTML = html;
}


function renderMovers() { /* RT mover scanner removed — HOD-only */ }


function renderLogs() {
  let logs = state.logs || [];
  let panel = document.getElementById('log-panel');
  if (!panel) return;
  if (logs.length === 0) {
    panel.innerHTML = '<div class="empty"><div class="icon">&#128196;</div>No log messages yet</div>';
    return;
  }
  panel.innerHTML = '';
  logs.slice(-200).forEach(addLogLine);
}

function pushLogLine(msg) {
  if (!msg || !msg.message) return;
  state.logs = state.logs || [];
  let key = (msg.ts || '') + '|' + msg.message;
  if (state.logs.some(l => ((l.ts || '') + '|' + l.message) === key)) return;
  state.logs.push(msg);
  if (state.logs.length > 300) state.logs = state.logs.slice(-300);
  addLogLine(msg);
}

function replayEventToLog(ev) {
  if (!ev) return null;
  let p = ev.payload || {};
  let ts = ev.ts || p.ts || new Date().toISOString();
  if (ev.type === 'cycle') {
    return {
      level: 'INFO',
      ts: ts,
      message: 'CYCLE #' + (p.cycle || '') + ' scanned=' + (p.symbols_scanned || 0)
        + ' hits=' + (p.scan_hits || 0) + ' signals=' + (p.signals || 0)
        + ' fills=' + (p.fills || 0),
    };
  }
  if (ev.type === 'scan_hit') {
    return {
      level: 'INFO',
      ts: ts,
      message: 'SCAN HIT ' + (p.symbol || '') + ' ' + (p.scanner_name || p.pattern || '')
        + (p.price ? ' @ $' + p.price : ''),
    };
  }
  if (ev.type === 'signal') {
    return {
      level: 'INFO',
      ts: ts,
      message: 'SIGNAL ' + (p.symbol || '') + ' ' + (p.action || '')
        + (p.reason ? ' — ' + p.reason : ''),
    };
  }
  if (ev.type === 'trade_fill' || ev.type === 'trade_exit') {
    return {
      level: ev.type === 'trade_exit' && Number(p.pnl || 0) < 0 ? 'WARNING' : 'INFO',
      ts: ts,
      message: ev.type.toUpperCase().replace('_', ' ') + ' ' + (p.symbol || '')
        + (p.price ? ' @ $' + p.price : '') + (p.pnl != null ? ' P&L ' + fmtPnl(p.pnl) : ''),
    };
  }
  if (ev.type === 'mistake') {
    return {
      level: 'WARNING',
      ts: ts,
      message: 'MISTAKE ' + (p.symbol || '') + ': ' + (p.reason || p.kind || ''),
    };
  }
  if (ev.type === 'market_regime') {
    return {
      level: 'INFO',
      ts: ts,
      message: 'MARKET ' + (p.phase || '') + (p.regime_label ? ' — ' + p.regime_label : ''),
    };
  }
  return null;
}

function renderNews() {
  let wrap = document.getElementById('news-wrap');
  let countEl = document.getElementById('news-count');
  if (!wrap) return;
  let newsItems = Object.values(state.news || {});
  if (countEl) countEl.textContent = newsItems.length;

  if (newsItems.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128240;</div>No news data yet</div>';
    return;
  }

  newsItems.sort((a, b) => (b.score || 0) - (a.score || 0));
  let html = '<table><tr><th>Symbol</th><th>Sentiment</th><th>Headlines</th></tr>';
  newsItems.forEach(n => {
    html += '<tr><td><strong>' + n.symbol + '</strong></td>';
    html += '<td>' + newsPill(n.sentiment, n.score) + '</td>';
    html += '<td style="font-size:11px;color:var(--text2);max-width:400px">';
    (n.headlines || []).slice(0, 2).forEach(h => {
      html += '<div style="margin-bottom:2px">' + h + '</div>';
    });
    html += '</td></tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
}

function fallbackDailyScorecard() {
  let exits = (state.recent_trades || []).filter(t => t.trade_type === 'exit' && t.pnl != null);
  let wins = exits.filter(t => Number(t.pnl) >= 0);
  let losses = exits.filter(t => Number(t.pnl) < 0);
  let totalWin = wins.reduce((s, t) => s + Number(t.pnl || 0), 0);
  let totalLoss = Math.abs(losses.reduce((s, t) => s + Number(t.pnl || 0), 0));
  let closed = exits.length;
  let totalTrades = Number((state.stats || {}).total_trades || 0);
  let signals = Number((state.stats || {}).total_signals || 0);
  let rejected = Number((state.stats || {}).total_rejected || 0);
  let scanHits = Number((state.stats || {}).total_scan_hits || 0);
  let winRate = closed ? wins.length / closed * 100 : 0;
  let avgWin = wins.length ? totalWin / wins.length : 0;
  let avgLoss = losses.length ? totalLoss / losses.length : 0;
  let pf = totalLoss > 0 ? totalWin / totalLoss : (totalWin > 0 ? 999 : 0);
  let expectancy = ((winRate / 100) * avgWin) - ((1 - (winRate / 100)) * avgLoss);
  let attempts = signals + rejected;
  let rows = state.missed_a_plus || [];
  let missedOpps = rows.filter(r => r.outcome === 'missed_opportunity').length;
  let correctRejects = rows.filter(r => r.outcome === 'correct_reject').length;
  let best = rows.reduce((acc, r) => Number(r.move_after_pct || 0) > Number((acc || {}).move_after_pct || 0) ? r : acc, null);
  return {
    trades_taken: totalTrades,
    closed_trades: closed,
    wins: wins.length,
    losses: losses.length,
    win_rate: winRate,
    total_pnl: totalWin - totalLoss,
    avg_win: avgWin,
    avg_loss: avgLoss,
    profit_factor: pf,
    expectancy_per_trade: expectancy,
    cycles: Number((state.stats || {}).cycle_count || 0),
    funnel: {
      scan_hits: scanHits,
      signals: signals,
      entries: totalTrades,
      rejected: rejected,
      hit_to_signal_pct: scanHits ? signals / scanHits * 100 : 0,
      signal_to_entry_pct: signals ? totalTrades / signals * 100 : 0,
      reject_rate_pct: attempts ? rejected / attempts * 100 : 0,
      closed_rate_pct: totalTrades ? closed / totalTrades * 100 : 0
    },
    missed_a_plus: {
      rows: rows.length,
      missed_opportunities: missedOpps,
      correct_rejects: correctRejects,
      pending: rows.length - missedOpps - correctRejects,
      best_symbol: best ? best.symbol : '',
      best_move_pct: best ? Number(best.move_after_pct || 0) : 0,
      best_pattern: best ? best.pattern : '',
      best_reason: best ? best.reason : ''
    },
    verdict: closed < 5 ? 'collecting' : (expectancy > 0 && pf >= 1.2 ? 'positive_expectancy' : (expectancy < 0 || pf < 1 ? 'negative_expectancy' : 'mixed'))
  };
}

function renderDailyScorecard() {
  let wrap = document.getElementById('daily-scorecard-wrap');
  if (!wrap) return;
  let sc = state.daily_scorecard && Object.keys(state.daily_scorecard).length ? state.daily_scorecard : fallbackDailyScorecard();
  let rolling = state.rolling_scorecard || {};
  let verdict = String(sc.verdict || 'collecting');
  let verdictMeta = {
    collecting: ['COLLECTING', 'pill-yellow'],
    positive_expectancy: ['POSITIVE', 'pill-green'],
    negative_expectancy: ['NEGATIVE', 'pill-red'],
    mixed: ['MIXED', 'pill-blue']
  }[verdict] || ['COLLECTING', 'pill-yellow'];
  let verdictEl = document.getElementById('scorecard-verdict');
  if (verdictEl) {
    verdictEl.textContent = verdictMeta[0];
    verdictEl.className = 'pill ' + verdictMeta[1];
  }

  let f = sc.funnel || {};
  let m = sc.missed_a_plus || {};
  let pf = Number(sc.profit_factor || 0);
  let bestMiss = m.best_symbol
    ? escapeHtml(m.best_symbol) + ' +' + Number(m.best_move_pct || 0).toFixed(1) + '%'
    : 'none';
  let html = '<div class="grid grid-4" style="margin-bottom:12px">';
  html += '<div class="stat-card"><div class="stat-label">P&L / Exp</div><div class="stat-value">' + fmtPnl(Number(sc.total_pnl || 0)) + '</div><div class="mini-muted">' + fmtPnl(Number(sc.expectancy_per_trade || 0)) + ' per trade</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value">' + Number(sc.win_rate || 0).toFixed(1) + '%</div><div class="mini-muted">' + (sc.wins || 0) + 'W / ' + (sc.losses || 0) + 'L</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Avg W / L</div><div class="stat-value"><span class="text-green">$' + Number(sc.avg_win || 0).toFixed(2) + '</span> <span style="color:var(--text2)">/</span> <span class="text-red">$' + Number(sc.avg_loss || 0).toFixed(2) + '</span></div><div class="mini-muted">PF ' + (pf >= 999 ? '999+' : pf.toFixed(2)) + '</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Trades</div><div class="stat-value">' + (sc.closed_trades || 0) + '/' + (sc.trades_taken || 0) + '</div><div class="mini-muted">closed / entries</div></div>';
  html += '</div>';

  html += '<table><tr><th>Funnel</th><th>Count</th><th>Conversion</th><th>Missed A+</th><th>Count</th></tr>';
  html += '<tr><td>Scan hits -> signals</td><td>' + (f.scan_hits || 0) + ' -> ' + (f.signals || 0) + '</td><td>' + pctText(f.hit_to_signal_pct) + '</td><td>Missed opportunities</td><td class="text-red">' + (m.missed_opportunities || 0) + '</td></tr>';
  html += '<tr><td>Signals -> entries</td><td>' + (f.signals || 0) + ' -> ' + (f.entries || 0) + '</td><td>' + pctText(f.signal_to_entry_pct) + '</td><td>Correct rejects</td><td class="text-green">' + (m.correct_rejects || 0) + '</td></tr>';
  html += '<tr><td>Reject pressure</td><td>' + (f.rejected || 0) + ' rejects</td><td>' + pctText(f.reject_rate_pct) + '</td><td>Pending / best miss</td><td>' + (m.pending || 0) + ' / ' + bestMiss + '</td></tr>';
  html += '</table>';

  let strategies = Object.entries(sc.by_strategy || {})
    .sort((a, b) => Number(b[1].total_pnl || 0) - Number(a[1].total_pnl || 0))
    .slice(0, 6);
  if (strategies.length) {
    html += '<div class="card-header" style="margin:14px 0 8px"><h3>Strategy P&L</h3></div>';
    html += '<table><tr><th>Strategy</th><th>Closed</th><th>Win Rate</th><th>P&L</th></tr>';
    strategies.forEach(([name, row]) => {
      let closed = Number(row.closed_trades || 0);
      let wins = Number(row.wins || 0);
      let wr = closed ? wins / closed * 100 : 0;
      html += '<tr><td>' + escapeHtml(name || 'unknown') + '</td><td>' + closed + '</td><td>' + wr.toFixed(1) + '%</td><td>' + fmtPnl(Number(row.total_pnl || 0)) + '</td></tr>';
    });
    html += '</table>';
  }

  html += '<div class="card-header" style="margin:14px 0 8px"><h3>Rolling Go-Live Gauge</h3>';
  if (rolling.available) {
    let rv = String(rolling.verdict || 'collecting');
    let rm = {
      collecting: ['COLLECTING', 'pill-yellow'],
      positive_expectancy: ['POSITIVE', 'pill-green'],
      negative_expectancy: ['NEGATIVE', 'pill-red'],
      mixed: ['MIXED', 'pill-blue']
    }[rv] || ['COLLECTING', 'pill-yellow'];
    html += '<span class="pill ' + rm[1] + '">' + rm[0] + '</span></div>';
    let rf = rolling.funnel || {};
    html += '<table><tr><th>Window</th><th>Sessions</th><th>Trades</th><th>P&L</th><th>Expectancy</th><th>Win Rate</th><th>PF</th><th>Signal -> Entry</th></tr>';
    html += '<tr><td>' + (rolling.window_days || 20) + ' days</td>';
    html += '<td>' + (rolling.sessions || 0) + '</td>';
    html += '<td>' + (rolling.closed_trades || 0) + '/' + (rolling.trades_taken || 0) + '</td>';
    html += '<td>' + fmtPnl(Number(rolling.total_pnl || 0)) + '</td>';
    html += '<td>' + fmtPnl(Number(rolling.expectancy_per_trade || 0)) + '</td>';
    html += '<td>' + Number(rolling.win_rate || 0).toFixed(1) + '%</td>';
    html += '<td>' + (Number(rolling.profit_factor || 0) >= 999 ? '999+' : Number(rolling.profit_factor || 0).toFixed(2)) + '</td>';
    html += '<td>' + pctText(rf.signal_to_entry_pct) + '</td></tr></table>';
    if (rolling.verdict_reason) {
      html += '<div class="mini-muted">Go-live verdict is collecting until ' + escapeHtml(rolling.verdict_reason) + ' are recorded.</div>';
    }
  } else {
    html += '<span class="pill pill-red">NO JOURNAL</span></div>';
    html += '<div class="mini-muted">Rolling scorecard unavailable: ' + escapeHtml(rolling.reason || 'journal not configured') + '</div>';
  }
  wrap.innerHTML = html;
}

function runBacktest() {
  let btn = document.getElementById('bt-run');
  let symbol = (document.getElementById('bt-symbol') || {}).value || '';
  let date = (document.getElementById('bt-date') || {}).value || '';
  let startTime = (document.getElementById('bt-start-time') || {}).value || '';
  symbol = symbol.trim().toUpperCase();
  startTime = startTime.trim();
  if (!symbol || !date) {
    alert('Symbol and date are required');
    return;
  }
  state.backtest = {loading: true, result: null, error: null};
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Running...';
  }
  renderBacktest();
  fetchJsonWithRetry('/api/backtest', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      symbol: symbol,
      date: date,
      start_time: startTime,
      flags: {
        fresh_vwap_reclaim_scout: !!(document.getElementById('bt-flag-fresh') || {}).checked,
        vwap_reclaim_scout: !!(document.getElementById('bt-flag-vwap-scout') || {}).checked,
        level_breakout_scout: !!(document.getElementById('bt-flag-level') || {}).checked,
        elite_wide_spread: !!(document.getElementById('bt-flag-spread') || {}).checked,
        momentum_burst_live: !!(document.getElementById('bt-flag-momentum') || {}).checked,
        momentum_burst_hit_run: !!(document.getElementById('bt-flag-mb-hit-run') || {}).checked,
        level_capped_entry: !!(document.getElementById('bt-flag-capped') || {}).checked,
        execution_timer_10s: !!(document.getElementById('bt-flag-timer') || {}).checked,
        ten_second_breakout_scout: !!(document.getElementById('bt-flag-10s-scout') || {}).checked,
        level_reclaim_10s_scout: !!(document.getElementById('bt-flag-10s-scout') || {}).checked,
        breakout_scalp_replay: !!(document.getElementById('bt-flag-breakout-scalp') || {}).checked,
        live_like_10s: !!(document.getElementById('bt-flag-live-like') || {}).checked
      }
    })
  }, 1).then(data => {
    if (!data.ok) throw new Error(data.error || 'Backtest failed');
    state.backtest = {loading: false, result: data, error: null, chart: {start: null, end: null, dragX: null}, liveScores: null};
    renderBacktest();
    // Fetch what this name actually scored in paper/live for the same day, so
    // the paper-vs-backtest score gap is visible side by side.
    fetch('/api/entry_scores?symbol=' + encodeURIComponent(symbol) + '&date=' + encodeURIComponent(date))
      .then(r => r.json())
      .then(ls => {
        if (state.backtest && state.backtest.result) {
          state.backtest.liveScores = (ls && ls.ok) ? ls : null;
          renderBacktest();
        }
      }).catch(() => {});
  }).catch(err => {
    state.backtest = {loading: false, result: null, error: err && err.message ? err.message : String(err)};
    renderBacktest();
  }).finally(() => {
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Run';
    }
  });
}

function backtestChartRange(barsLen) {
  let chart = (state.backtest && state.backtest.chart) || {};
  let start = Number.isFinite(Number(chart.start)) ? Number(chart.start) : 0;
  let end = Number.isFinite(Number(chart.end)) ? Number(chart.end) : barsLen - 1;
  start = Math.max(0, Math.min(start, Math.max(0, barsLen - 1)));
  end = Math.max(start, Math.min(end, Math.max(0, barsLen - 1)));
  if (end - start < 10 && barsLen > 10) {
    end = Math.min(barsLen - 1, start + 10);
    start = Math.max(0, end - 10);
  }
  return {start, end};
}

function setBacktestChartRange(start, end) {
  let bars = (((state.backtest || {}).result || {}).bars_data || []);
  if (!bars.length) return;
  let minWindow = Math.min(20, bars.length);
  start = Math.max(0, Math.min(Math.round(start), bars.length - 1));
  end = Math.max(start, Math.min(Math.round(end), bars.length - 1));
  if (end - start + 1 < minWindow) {
    let center = (start + end) / 2;
    start = Math.max(0, Math.round(center - minWindow / 2));
    end = Math.min(bars.length - 1, start + minWindow - 1);
    start = Math.max(0, end - minWindow + 1);
  }
  state.backtest.chart = Object.assign({}, state.backtest.chart || {}, {start, end});
  renderBacktest();
}

function zoomBacktestChart(factor) {
  let bars = (((state.backtest || {}).result || {}).bars_data || []);
  if (!bars.length) return;
  let r = backtestChartRange(bars.length);
  let windowSize = r.end - r.start + 1;
  let next = Math.max(20, Math.min(bars.length, Math.round(windowSize * factor)));
  let center = (r.start + r.end) / 2;
  setBacktestChartRange(center - next / 2, center + next / 2);
}

function resetBacktestChart() {
  let bars = (((state.backtest || {}).result || {}).bars_data || []);
  state.backtest.chart = {start: null, end: null, dragX: null};
  if (bars.length) renderBacktest();
}

function panBacktestChart(barDelta) {
  let bars = (((state.backtest || {}).result || {}).bars_data || []);
  if (!bars.length) return;
  let r = backtestChartRange(bars.length);
  let width = r.end - r.start;
  let start = Math.max(0, Math.min(r.start + barDelta, bars.length - width - 1));
  setBacktestChartRange(start, start + width);
}

function beginBacktestChartDrag(ev) {
  if (!state.backtest.chart) state.backtest.chart = {};
  state.backtest.chart.dragX = ev.clientX;
}

function dragBacktestChart(ev) {
  let chart = (state.backtest || {}).chart || {};
  if (chart.dragX == null) return;
  let bars = (((state.backtest || {}).result || {}).bars_data || []);
  if (!bars.length) return;
  let r = backtestChartRange(bars.length);
  let visible = Math.max(1, r.end - r.start + 1);
  let dx = ev.clientX - chart.dragX;
  let barDelta = Math.round((-dx / 900) * visible);
  if (barDelta !== 0) {
    state.backtest.chart.dragX = ev.clientX;
    panBacktestChart(barDelta);
  }
}

function endBacktestChartDrag() {
  if (!state.backtest.chart) state.backtest.chart = {};
  state.backtest.chart.dragX = null;
}

function wheelBacktestChart(ev) {
  ev.preventDefault();
  zoomBacktestChart(ev.deltaY < 0 ? 0.75 : 1.35);
}

function etMinutes(ts) {
  try {
    let parts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false
    }).formatToParts(new Date(ts));
    let h = Number((parts.find(p => p.type === 'hour') || {}).value || 0);
    let m = Number((parts.find(p => p.type === 'minute') || {}).value || 0);
    return h * 60 + m;
  } catch(e) { return 0; }
}

function shortEtTime(ts) {
  if (!ts) return '';
  try {
    return new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false
    }).format(new Date(ts)) + ' ET';
  } catch(e) { return String(ts || ''); }
}

function renderBacktestChart(r) {
  let bars = r.bars_data || [];
  if (!bars.length) {
    return '<div class="card" style="margin-bottom:16px"><div class="empty"><div class="icon">&#128202;</div>No 1-minute bars returned for chart</div></div>';
  }
  let fullBars = bars;
  let range = backtestChartRange(fullBars.length);
  bars = fullBars.slice(range.start, range.end + 1);
  let w = 1040, h = 380, padL = 46, padR = 12, padT = 18, padB = 34;
  let priceH = 270, volTop = padT + priceH + 18, volH = h - volTop - padB;
  let highs = bars.map(b => Number(b.high || b.close || 0));
  let lows = bars.map(b => Number(b.low || b.close || 0));
  let maxP = Math.max(...highs), minP = Math.min(...lows);
  let pad = Math.max((maxP - minP) * 0.08, maxP * 0.002, 0.01);
  maxP += pad; minP = Math.max(0, minP - pad);
  let maxV = Math.max(...bars.map(b => Number(b.volume || 0)), 1);
  let plotW = w - padL - padR;
  let xAt = i => padL + (bars.length <= 1 ? 0 : (i / (bars.length - 1)) * plotW);
  let yAt = p => padT + ((maxP - Number(p || 0)) / Math.max(maxP - minP, 0.01)) * priceH;
  let bw = Math.max(1, Math.min(5, plotW / Math.max(bars.length, 1) * 0.7));
  let viewLabel = shortEtTime(bars[0].ts) + ' - ' + shortEtTime(bars[bars.length - 1].ts);
  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header"><h3>1m Full Session Chart</h3><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><span class="mini-muted">'+escapeHtml(viewLabel)+' · '+bars.length+'/'+fullBars.length+' bars</span><button onclick="zoomBacktestChart(0.65)" style="padding:5px 9px;border:1px solid var(--border);background:var(--surface2);color:var(--text);border-radius:6px;cursor:pointer">+</button><button onclick="zoomBacktestChart(1.45)" style="padding:5px 9px;border:1px solid var(--border);background:var(--surface2);color:var(--text);border-radius:6px;cursor:pointer">-</button><button onclick="resetBacktestChart()" style="padding:5px 9px;border:1px solid var(--border);background:var(--surface2);color:var(--text);border-radius:6px;cursor:pointer">Reset</button></div></div>';
  html += '<div id="bt-svg-chart" style="overflow-x:auto;margin-top:8px"><svg id="bt-chart-svg" viewBox="0 0 '+w+' '+h+'" onmousedown="beginBacktestChartDrag(event)" onmousemove="dragBacktestChart(event)" onmouseup="endBacktestChartDrag()" onmouseleave="endBacktestChartDrag()" onwheel="wheelBacktestChart(event)" style="width:100%;min-width:860px;height:390px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;cursor:grab;user-select:none">';
  let sessions = [];
  let cur = null;
  bars.forEach((b, i) => {
    let m = etMinutes(b.ts);
    let name = m < 570 ? 'premarket' : (m < 960 ? 'regular' : 'after');
    if (!cur || cur.name !== name) {
      if (cur) { cur.end = i - 1; sessions.push(cur); }
      cur = {name: name, start: i, end: i};
    } else {
      cur.end = i;
    }
  });
  if (cur) sessions.push(cur);
  sessions.forEach(s => {
    let x1 = xAt(s.start), x2 = xAt(s.end);
    let fill = s.name === 'regular' ? 'rgba(34,197,94,0.05)' : 'rgba(59,130,246,0.06)';
    html += '<rect x="'+x1.toFixed(1)+'" y="'+padT+'" width="'+Math.max(1, x2-x1).toFixed(1)+'" height="'+(h-padT-padB)+'" fill="'+fill+'"></rect>';
    html += '<text x="'+(x1+5).toFixed(1)+'" y="14" fill="var(--text2)" font-size="10">'+s.name+'</text>';
  });
  let timeTicks = [240, 570, 720, 960, 1200];
  timeTicks.forEach(minute => {
    let idx = bars.findIndex(b => etMinutes(b.ts) >= minute);
    if (idx < 0) return;
    let x = xAt(idx);
    let strong = minute === 570 || minute === 960;
    html += '<line x1="'+x.toFixed(1)+'" y1="'+padT+'" x2="'+x.toFixed(1)+'" y2="'+(h-padB).toFixed(1)+'" stroke="'+(strong ? 'rgba(250,204,21,0.55)' : 'rgba(148,163,184,0.20)')+'" stroke-dasharray="'+(strong ? '4 4' : '2 6')+'"></line>';
    let label = String(Math.floor(minute/60)).padStart(2, '0') + ':' + String(minute%60).padStart(2, '0');
    html += '<text x="'+(x-18).toFixed(1)+'" y="'+(h-10)+'" fill="var(--text2)" font-size="10">'+label+' ET</text>';
  });
  for (let i=0; i<5; i++) {
    let y = padT + (i/4)*priceH;
    let p = maxP - (i/4)*(maxP-minP);
    html += '<line x1="'+padL+'" y1="'+y.toFixed(1)+'" x2="'+(w-padR)+'" y2="'+y.toFixed(1)+'" stroke="rgba(148,163,184,0.16)"></line>';
    html += '<text x="4" y="'+(y+4).toFixed(1)+'" fill="var(--text2)" font-size="10">$'+p.toFixed(2)+'</text>';
  }
  bars.forEach((b, i) => {
    let x = xAt(i), o = Number(b.open || b.close || 0), c = Number(b.close || 0), hi = Number(b.high || c), lo = Number(b.low || c);
    let col = c >= o ? '#14b8a6' : '#f43f5e';
    let yO = yAt(o), yC = yAt(c), yHi = yAt(hi), yLo = yAt(lo);
    html += '<line x1="'+x.toFixed(1)+'" y1="'+yHi.toFixed(1)+'" x2="'+x.toFixed(1)+'" y2="'+yLo.toFixed(1)+'" stroke="'+col+'" stroke-width="1"></line>';
    html += '<rect x="'+(x-bw/2).toFixed(1)+'" y="'+Math.min(yO,yC).toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(1, Math.abs(yC-yO)).toFixed(1)+'" fill="'+col+'"></rect>';
    let vh = (Number(b.volume || 0) / maxV) * volH;
    html += '<rect x="'+(x-bw/2).toFixed(1)+'" y="'+(volTop + volH - vh).toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(1, vh).toFixed(1)+'" fill="'+col+'" opacity="0.28"></rect>';
  });
  let indexByTs = {};
  bars.forEach((b, i) => { indexByTs[String(b.ts).slice(0,16)] = i; });
  let markAt = (ts, price) => {
    let key = String(ts || '').slice(0,16);
    let idx = indexByTs[key];
    if (idx == null) idx = bars.findIndex(b => Math.abs(new Date(b.ts) - new Date(ts || 0)) < 61000);
    if (idx < 0) return null;
    return {x: xAt(idx), y: yAt(price || bars[idx].close || 0)};
  };
  (r.scan_events || []).filter(e => e.a_plus).slice(0, 160).forEach(e => {
    let pt = markAt(e.ts, e.price);
    if (!pt) return;
    let col = e.status === 'accepted' ? '#22c55e' : (e.reason ? '#ef4444' : '#facc15');
    html += '<circle cx="'+pt.x.toFixed(1)+'" cy="'+pt.y.toFixed(1)+'" r="3.2" fill="'+col+'" opacity="0.85"><title>'+escapeHtml(shortEtTime(e.ts) + ' ' + (e.scanner||'') + ': ' + (e.reason||e.status||''))+'</title></circle>';
    if (col === '#facc15') {
      let labelX = Math.min(w - padR - 42, pt.x + 5);
      let labelY = Math.max(padT + 10, pt.y - 6);
      html += '<text x="'+labelX.toFixed(1)+'" y="'+labelY.toFixed(1)+'" fill="#facc15" font-size="10" font-weight="600">'+escapeHtml(shortEtTime(e.ts).replace(' ET', ''))+'</text>';
    }
  });
  (r.round_trips || []).forEach(t => {
    let ept = markAt(t.entry_time, t.entry_price);
    let xpt = markAt(t.exit_time, t.exit_price);
    if (ept) html += '<path d="M '+ept.x.toFixed(1)+' '+(ept.y-8).toFixed(1)+' l 6 10 h -12 z" fill="#22c55e"><title>Entry $'+Number(t.entry_price||0).toFixed(4)+'</title></path>';
    if (xpt) html += '<path d="M '+xpt.x.toFixed(1)+' '+(xpt.y+8).toFixed(1)+' l 6 -10 h -12 z" fill="#f97316"><title>Exit $'+Number(t.exit_price||0).toFixed(4)+'</title></path>';
  });
  html += '<line x1="'+padL+'" y1="'+volTop+'" x2="'+(w-padR)+'" y2="'+volTop+'" stroke="rgba(148,163,184,0.20)"></line>';
  html += '</svg></div>';
  html += '<div class="mini-muted" style="margin-top:8px">Chart times are Eastern Time. Yellow dashed lines mark regular open/close. Red dots are A+ blocks/rejects, green dots are accepted A+ points, triangles are simulated entries/exits.</div>';
  html += '</div>';
  return html;
}

function backtestDecisionRows(r) {
  let rows = [];
  (r.scan_events || []).forEach(e => {
    if (!e.a_plus) return;
    rows.push({
      ts: e.ts, pattern: e.pattern || e.scanner, stage: e.blocked_layer || 'scanner',
      status: e.status || 'scan_hit', price: e.price, score: e.score,
      reason: e.reason || '', setup_tier: e.setup_tier || '', source: 'scan'
    });
  });
  (r.entry_decisions || []).forEach(d => {
    let tier = String(d.setup_tier || d.entry_tier || '');
    let pattern = String(d.pattern || '');
    if (tier.indexOf('A+') < 0) return;
    rows.push({
      ts: d.ts || d.time || '', pattern: pattern, stage: d.stage || d.blocked_layer || 'entry',
      status: d.passed ? 'accepted' : 'rejected', price: d.price || 0,
      score: (d.metadata || {}).entry_score || '', reason: d.reason || d.reject_reason || '',
      setup_tier: d.setup_tier || '', source: 'decision'
    });
  });
  rows.sort((a,b) => String(a.ts || '').localeCompare(String(b.ts || '')));
  return rows;
}

function decisionPill(status) {
  let s = String(status || '').toLowerCase();
  let cls = s === 'accepted' || s === 'passed' ? 'pill-green' : (s === 'scan_hit' ? 'pill-yellow' : 'pill-red');
  return '<span class="pill '+cls+'">'+escapeHtml(s || 'unknown')+'</span>';
}

function renderBacktestFunnel(r) {
  let rows = backtestDecisionRows(r);
  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header"><h3>A+ Funnel Detail</h3><span class="mini-muted">'+rows.length+' A+ scan/decision rows</span></div>';
  if (!rows.length) {
    html += '<div class="empty"><div class="icon">&#128269;</div>No A+ decisions recorded in this replay</div></div>';
    return html;
  }
  html += '<table><tr><th>Time</th><th>Pattern</th><th>Status</th><th>Blocked At</th><th>Price</th><th>Score</th><th>Reason</th></tr>';
  rows.slice(0, 300).forEach(row => {
    html += '<tr>';
    html += '<td>'+shortEtTime(row.ts)+'</td>';
    html += '<td><strong>'+escapeHtml(row.pattern || '')+'</strong><div class="mini-muted">'+escapeHtml(row.setup_tier || row.source || '')+'</div></td>';
    html += '<td>'+decisionPill(row.status)+'</td>';
    html += '<td>'+escapeHtml(row.stage || '')+'</td>';
    html += '<td>'+(Number(row.price || 0) > 0 ? '$'+Number(row.price || 0).toFixed(4) : '')+'</td>';
    html += '<td>'+escapeHtml(row.score || '')+'</td>';
    html += '<td style="max-width:520px;white-space:normal">'+escapeHtml(row.reason || '')+'</td>';
    html += '</tr>';
  });
  if (rows.length > 300) {
    html += '<tr><td colspan="7" class="mini-muted">Showing first 300 rows. Use the API response for full detail.</td></tr>';
  }
  html += '</table></div>';
  return html;
}

function renderBacktestManifest(r) {
  let m = r.manifest || {};
  let data = m.data || {};
  let settings = (m.settings || {});
  let strat = settings.strategy || {};
  let flags = m.flags || r.flags || {};
  let onFlags = Object.entries(flags).filter(([k,v]) => !!v).map(([k]) => k);
  let cache1 = data.cache_1m || {};
  let cache10 = data.cache_10s || {};
  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header"><h3>Run Manifest</h3><span class="mini-muted">Reproducibility</span></div>';
  html += '<div class="grid grid-4" style="margin-bottom:10px">';
  html += '<div class="stat-card"><div class="stat-label">Mode</div><div class="stat-value" style="font-size:18px">'+escapeHtml(data.source || 'unknown')+'</div><div class="mini-muted">'+(data.bars_1m || 0)+' x 1m / '+(data.bars_10s || 0)+' x 10s</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Code</div><div class="stat-value" style="font-size:18px">'+escapeHtml(m.code_version || '')+'</div><div class="mini-muted">'+escapeHtml((m.generated_at || '').slice(0,19))+'</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Hit-Run Window</div><div class="stat-value" style="font-size:18px">'+escapeHtml(strat.momentum_burst_hit_run_end_et || 'all day')+'</div><div class="mini-muted">end ET</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Runner Trail</div><div class="stat-value" style="font-size:18px">'+(strat.runner_trail_adaptive ? 'adaptive' : String(strat.runner_trail_pct || ''))+'</div><div class="mini-muted">cap '+escapeHtml(String(strat.runner_trail_cap || ''))+'</div></div>';
  html += '</div>';
  html += '<div class="mini-muted"><strong>Active flags:</strong> '+escapeHtml(onFlags.length ? onFlags.join(', ') : 'none')+'</div>';
  html += '<div class="mini-muted"><strong>1m cache:</strong> '+escapeHtml(cache1.path || '')+' '+(cache1.sha1 ? '('+escapeHtml(cache1.sha1)+', '+Number(cache1.bytes || 0)+' bytes)' : '')+'</div>';
  if (cache10.path || data.bars_10s) {
    html += '<div class="mini-muted"><strong>10s cache:</strong> '+escapeHtml(cache10.path || '')+' '+(cache10.sha1 ? '('+escapeHtml(cache10.sha1)+', '+Number(cache10.bytes || 0)+' bytes)' : '')+'</div>';
  }
  html += '</div>';
  return html;
}

function renderBacktestSection(title, renderer) {
  try {
    return renderer();
  } catch (err) {
    return '<div class="card" style="margin-bottom:16px"><div class="card-header"><h3>' +
      escapeHtml(title) + '</h3></div><div class="empty"><div class="icon">&#9888;</div>' +
      escapeHtml(err && err.message ? err.message : String(err)) + '</div></div>';
  }
}

function renderBacktestLayerBreakdown(r) {
  let f = r.funnel || {};
  let layers = f.rejected_by_layer || r.rejected_by_layer || {};
  let reasons = f.top_reject_reasons_by_layer || r.top_reject_reasons_by_layer || {};
  let names = Object.keys(layers).sort((a, b) => Number(layers[b] || 0) - Number(layers[a] || 0));
  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header"><h3>Backtest Gate Breakdown</h3><span class="mini-muted">Deferred: '+Number(f.deferred || (r.deferred_signals || []).length || 0)+'</span></div>';
  if (!names.length) {
    html += '<div class="empty"><div class="icon">&#9989;</div>No layer-level rejects recorded in this replay</div></div>';
    return html;
  }
  html += '<table><tr><th>Layer</th><th>Rejects</th><th>Top Reasons</th></tr>';
  names.forEach(layer => {
    let reasonRows = (reasons[layer] || []).map(row => {
      return '<div><strong>'+Number(row.count || 0)+'</strong> × '+escapeHtml(row.reason || '')+'</div>';
    }).join('');
    html += '<tr>';
    html += '<td><span class="pill pill-red">'+escapeHtml(layer)+'</span></td>';
    html += '<td>'+Number(layers[layer] || 0)+'</td>';
    html += '<td style="max-width:720px;white-space:normal">'+(reasonRows || '<span class="mini-muted">No reason detail</span>')+'</td>';
    html += '</tr>';
  });
  html += '</table></div>';
  return html;
}

function renderBacktestMicroOpportunities(r) {
  let rows = (r.micro_opportunities || []).slice().sort((a, b) => Number(b.move_after_pct || 0) - Number(a.move_after_pct || 0));
  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header"><h3>10s Opportunity Map</h3><span class="mini-muted">'+rows.length+' real 10s candidates</span></div>';
  if (!rows.length) {
    html += '<div class="empty"><div class="icon">&#128269;</div>No real 10s breakout opportunities recorded</div></div>';
    return html;
  }
  html += '<table><tr><th>Time</th><th>Pattern</th><th>Level</th><th>10s Price</th><th>Volume</th><th>Max After</th><th>Move After</th><th>Why</th></tr>';
  rows.slice(0, 80).forEach(row => {
    html += '<tr>';
    html += '<td>'+shortEtTime(row.ts)+'</td>';
    html += '<td><strong>'+escapeHtml(row.pattern || '')+'</strong></td>';
    html += '<td>$'+Number(row.breakout_level || 0).toFixed(4)+'</td>';
    html += '<td>$'+Number(row.price || 0).toFixed(4)+'</td>';
    html += '<td>'+Number(row.volume || 0).toLocaleString()+'</td>';
    html += '<td>$'+Number(row.max_after || 0).toFixed(4)+'</td>';
    html += '<td>'+Number(row.move_after_pct || 0).toFixed(1)+'%</td>';
    html += '<td style="max-width:420px;white-space:normal">'+escapeHtml(row.reason || '')+'</td>';
    html += '</tr>';
  });
  html += '</table></div>';
  return html;
}

function renderLivePaperScores() {
  let ls = state.backtest && state.backtest.liveScores;
  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header"><h3>Live Paper Entry Scores</h3>';
  if (!ls) {
    html += '<span class="mini-muted">loading / none logged</span></div>';
    html += '<div class="empty"><div class="icon">&#128202;</div>No paper entry-score records for this name/day (data/ml/entry_candidates.jsonl). The live bot logs these; run the paper bot on this day to populate.</div></div>';
    return html;
  }
  let best = (ls.best_passed_score != null) ? ls.best_passed_score : (ls.best_score != null ? ls.best_score : '--');
  html += '<span class="mini-muted">' + ls.total + ' checks &middot; ' + ls.passed + ' passed &middot; best passed ' + best + '/100</span></div>';
  html += '<div class="mini-muted" style="padding:0 0 8px">What this name actually scored in paper — compare against the backtest gate rejections above. A pass here that the backtest rejects = a scoring/data gap (often rvol/surge), not a strategy gap.</div>';
  let cands = (ls.candidates || []);
  let passed = cands.filter(c => c.passed);
  let rejected = cands.filter(c => !c.passed).sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
  function rowHtml(c) {
    let h = '<tr>';
    h += '<td>' + shortEtTime(c.ts) + '</td>';
    h += '<td>' + (c.passed ? '<strong style="color:var(--green)">PASS ' : '<span style="color:var(--text2)">rej ') + Number(c.score || 0) + '</strong></td>';
    h += '<td>$' + Number(c.price || 0).toFixed(4) + '</td>';
    h += '<td style="max-width:520px;white-space:normal">' + escapeHtml(c.breakdown || c.reject_reason || '') + '</td>';
    h += '</tr>';
    return h;
  }
  html += '<table><tr><th>Time (ET)</th><th>Score</th><th>Price</th><th>Breakdown</th></tr>';
  if (passed.length) {
    html += '<tr><td colspan="4" class="mini-muted" style="background:var(--bg2)">PASSED entries (' + passed.length + ')</td></tr>';
    passed.slice(0, 40).forEach(c => { html += rowHtml(c); });
  }
  html += '<tr><td colspan="4" class="mini-muted" style="background:var(--bg2)">Top rejected by score (' + rejected.length + ' total)</td></tr>';
  rejected.slice(0, 20).forEach(c => { html += rowHtml(c); });
  html += '</table></div>';
  return html;
}

function renderBacktest() {
  let wrap = document.getElementById('backtest-wrap');
  if (!wrap) return;
  let bt = state.backtest || {};
  if (bt.loading) {
    wrap.innerHTML = '<div class="card"><div class="empty"><div class="icon">&#9203;</div>Running backtest...</div></div>';
    return;
  }
  if (bt.error) {
    wrap.innerHTML = '<div class="card"><div class="empty"><div class="icon">&#9888;</div>' + escapeHtml(bt.error) + '</div></div>';
    return;
  }
  let r = bt.result;
  if (!r) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128202;</div>Choose a symbol and date, then run a backtest.</div>';
    return;
  }
  let sc = r.scorecard || {};
  let f = r.funnel || {};
  let trips = r.round_trips || [];
  let flags = r.flags || {};
  let html = '<div class="card" style="margin-bottom:16px">';
  html += '<div class="card-header"><h3>' + escapeHtml(r.symbol) + ' · ' + escapeHtml(r.date) + '</h3>';
  html += '<span class="pill pill-blue">' + (r.bars || 0) + ' bars · ' + (r.cycles || 0) + ' cycles</span></div>';
  html += '<div class="grid grid-4" style="margin-bottom:12px">';
  html += '<div class="stat-card"><div class="stat-label">P&L / Exp</div><div class="stat-value">' + fmtPnl(Number(sc.total_pnl || 0)) + '</div><div class="mini-muted">' + fmtPnl(Number(sc.expectancy_per_trade || 0)) + ' per trade</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value">' + Number(sc.win_rate || 0).toFixed(1) + '%</div><div class="mini-muted">' + (sc.wins || 0) + 'W / ' + (sc.losses || 0) + 'L</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Trades</div><div class="stat-value">' + (sc.closed_trades || 0) + '/' + (sc.trades_taken || 0) + '</div><div class="mini-muted">closed / entries</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Funnel</div><div class="stat-value">' + (f.scan_hits || 0) + ' -> ' + (f.signals || 0) + ' -> ' + (f.entries || 0) + '</div><div class="mini-muted">' + pctText(f.signal_to_entry_pct) + ' signal to entry</div></div>';
  html += '</div>';
  html += '<div class="mini-muted">Flags: fresh_vwap=' + (flags.fresh_vwap_reclaim_scout ? 'on' : 'off') + ', vwap_reclaim=' + (flags.vwap_reclaim_scout ? 'on' : 'off') + ', level_breakout=' + (flags.level_breakout_scout ? 'on' : 'off') + ', elite_wide_spread=' + (flags.elite_wide_spread ? 'on' : 'off') + ', momentum_burst_live=' + (flags.momentum_burst_live ? 'on' : 'off') + ', momentum_hit_run=' + (flags.momentum_burst_hit_run ? 'on' : 'off') + ', level_capped_entry=' + (flags.level_capped_entry ? 'on' : 'off') + ', 10s_timer=' + (flags.execution_timer_10s ? 'on' : 'off') + ', 10s_scout=' + (flags.ten_second_breakout_scout ? 'on' : 'off') + ', 10s_reclaim=' + (flags.level_reclaim_10s_scout ? 'on' : 'off') + ', breakout_scalp=' + (flags.breakout_scalp_replay ? 'on' : 'off') + ', live_like_10s=' + (flags.live_like_10s ? 'on' : 'off') + '</div>';
  if (r.execution_timer_source) {
    html += '<div class="mini-muted">Execution timer source: ' + escapeHtml(r.execution_timer_source) + '</div>';
  }
  if ((r.unsupported_flags || []).length) {
    html += '<div class="mini-muted">Not simulated here: ' + escapeHtml((r.unsupported_flags || []).join(', ')) + '</div>';
  }
  html += '</div>';

  html += renderBacktestManifest(r);
  html += renderBacktestSection('1m Full Session Chart', () => renderBacktestChart(r));
  html += renderBacktestSection('10s Opportunity Map', () => renderBacktestMicroOpportunities(r));
  html += renderBacktestSection('Backtest Gate Breakdown', () => renderBacktestLayerBreakdown(r));
  html += renderBacktestSection('A+ Funnel Detail', () => renderBacktestFunnel(r));
  html += renderBacktestSection('Live Paper Entry Scores (same name/day)', () => renderLivePaperScores());

  html += '<div class="card" style="margin-bottom:16px"><div class="card-header"><h3>Trades</h3></div>';
  if (!trips.length) {
    html += '<div class="empty"><div class="icon">&#128269;</div>No completed trades in this replay</div>';
  } else {
    html += '<table><tr><th>Entry Time</th><th>Pattern / Mode</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>Reason</th></tr>';
    trips.forEach(t => {
      html += '<tr>';
      html += '<td>' + shortEtTime(t.entry_time) + '</td>';
      html += '<td>' + escapeHtml(t.pattern || t.mode || '') + '</td>';
      html += '<td>$' + Number(t.entry_price || 0).toFixed(4) + '</td>';
      html += '<td>$' + Number(t.exit_price || 0).toFixed(4) + '</td>';
      html += '<td>' + Number(t.quantity || 0).toFixed(0) + '</td>';
      html += '<td>' + fmtPnl(Number(t.pnl || 0)) + '</td>';
      html += '<td>' + escapeHtml(t.exit_reason || '') + '</td>';
      html += '</tr>';
    });
    html += '</table>';
  }
  html += '</div>';

  let modes = Object.entries(sc.by_entry_mode || {});
  if (modes.length) {
    html += '<div class="card"><div class="card-header"><h3>By Entry Mode</h3></div>';
    html += '<table><tr><th>Mode</th><th>Closed</th><th>Wins</th><th>P&L</th></tr>';
    modes.forEach(([name, row]) => {
      html += '<tr><td>' + escapeHtml(name) + '</td><td>' + (row.closed_trades || 0) + '</td><td>' + (row.wins || 0) + '</td><td>' + fmtPnl(Number(row.total_pnl || 0)) + '</td></tr>';
    });
    html += '</table></div>';
  }
  wrap.innerHTML = html;
}

function renderAnalytics() {
  renderMissedAPlus();
  renderDailyScorecard();
  let exits = (state.recent_trades||[]).filter(t => t.trade_type === 'exit' && t.pnl != null);
  if (exits.length === 0) {
    document.getElementById('an-avg-win').innerHTML = fmtPnl(0);
    document.getElementById('an-avg-loss').innerHTML = fmtPnl(0);
    document.getElementById('an-pf').textContent = '0.00';
    document.getElementById('an-expect').innerHTML = fmtPnl(0);
    renderMLReport();
    renderAI();
    return;
  }

  let wins = exits.filter(t => t.pnl >= 0);
  let losses = exits.filter(t => t.pnl < 0);
  let totalWin = wins.reduce((s,t) => s + t.pnl, 0);
  let totalLoss = Math.abs(losses.reduce((s,t) => s + t.pnl, 0));
  let avgWin = wins.length > 0 ? totalWin / wins.length : 0;
  let avgLoss = losses.length > 0 ? totalLoss / losses.length : 0;
  let pf = totalLoss > 0 ? totalWin / totalLoss : wins.length > 0 ? 999 : 0;
  let wr = exits.length > 0 ? wins.length / exits.length : 0;
  let expectancy = (wr * avgWin) - ((1 - wr) * avgLoss);

  document.getElementById('an-avg-win').innerHTML = fmtPnl(avgWin);
  document.getElementById('an-avg-loss').innerHTML = fmtPnl(-avgLoss);
  document.getElementById('an-pf').textContent = pf.toFixed(2);
  document.getElementById('an-pf').style.color = pf >= 1 ? 'var(--green)' : 'var(--red)';
  document.getElementById('an-expect').innerHTML = fmtPnl(expectancy);

  // Per-symbol breakdown
  let bySymbol = {};
  exits.forEach(t => {
    if (!bySymbol[t.symbol]) bySymbol[t.symbol] = {wins:0, losses:0, pnl:0, trades:0};
    bySymbol[t.symbol].trades++;
    bySymbol[t.symbol].pnl += t.pnl;
    if (t.pnl >= 0) bySymbol[t.symbol].wins++; else bySymbol[t.symbol].losses++;
  });
  let symArr = Object.entries(bySymbol).sort((a,b) => b[1].pnl - a[1].pnl);
  let shtml = '<table><tr><th>Symbol</th><th>Trades</th><th>W/L</th><th>Win Rate</th><th>P&L</th></tr>';
  symArr.forEach(([sym, d]) => {
    let wr2 = d.trades > 0 ? (d.wins/d.trades*100).toFixed(0) : 0;
    shtml += '<tr><td><strong>'+sym+'</strong></td>';
    shtml += '<td>'+d.trades+'</td>';
    shtml += '<td><span class="text-green">'+d.wins+'</span>/<span class="text-red">'+d.losses+'</span></td>';
    shtml += '<td>'+wr2+'%</td>';
    shtml += '<td>'+fmtPnl(d.pnl)+'</td></tr>';
  });
  shtml += '</table>';
  document.getElementById('an-symbol-table').innerHTML = shtml;

  // Exit type analysis
  let byExit = {};
  exits.forEach(t => {
    let r = t.exit_reason || 'unknown';
    if (!byExit[r]) byExit[r] = {count:0, pnl:0, wins:0};
    byExit[r].count++;
    byExit[r].pnl += t.pnl;
    if (t.pnl >= 0) byExit[r].wins++;
  });
  let exitArr = Object.entries(byExit).sort((a,b) => b[1].count - a[1].count);
  let ehtml = '<table><tr><th>Exit Type</th><th>Count</th><th>Win Rate</th><th>Avg P&L</th><th>Total P&L</th></tr>';
  exitArr.forEach(([reason, d]) => {
    let wr3 = d.count > 0 ? (d.wins/d.count*100).toFixed(0) : 0;
    let avg = d.count > 0 ? d.pnl / d.count : 0;
    ehtml += '<tr><td>'+reason+'</td>';
    ehtml += '<td>'+d.count+'</td>';
    ehtml += '<td>'+wr3+'%</td>';
    ehtml += '<td>'+fmtPnl(avg)+'</td>';
    ehtml += '<td>'+fmtPnl(d.pnl)+'</td></tr>';
  });
  ehtml += '</table>';
  document.getElementById('an-exit-table').innerHTML = ehtml;

  // Equity curve
  let canvas = document.getElementById('equity-canvas');
  if (canvas && exits.length > 1) {
    let ctx = canvas.getContext('2d');
    let dpr = window.devicePixelRatio || 1;
    let rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    let w = rect.width, h = rect.height;
    ctx.clearRect(0,0,w,h);

    let cumPnl = [0];
    exits.forEach(t => cumPnl.push(cumPnl[cumPnl.length-1] + t.pnl));

    let minP = Math.min(...cumPnl);
    let maxP = Math.max(...cumPnl);
    let range = maxP - minP || 1;
    let pad = 20;

    ctx.strokeStyle = cumPnl[cumPnl.length-1] >= 0 ? '#4ade80' : '#f87171';
    ctx.lineWidth = 2;
    ctx.beginPath();
    cumPnl.forEach((p, i) => {
      let x = pad + (i / (cumPnl.length - 1)) * (w - 2*pad);
      let y = h - pad - ((p - minP) / range) * (h - 2*pad);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // zero line
    let zeroY = h - pad - ((0 - minP) / range) * (h - 2*pad);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4,4]);
    ctx.beginPath();
    ctx.moveTo(pad, zeroY);
    ctx.lineTo(w-pad, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // labels
    ctx.fillStyle = 'var(--text2)';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#888';
    ctx.fillText('$'+maxP.toFixed(0), 2, pad);
    ctx.fillText('$'+minP.toFixed(0), 2, h-5);
    ctx.fillText('$0', 2, zeroY - 3);
  }

  // AI Insights
  renderMLReport();
  renderAI();
}

function renderMissedAPlus() {
  let wrap = document.getElementById('missed-a-plus-wrap');
  if (!wrap) return;
  let rows = (state.missed_a_plus || []).slice(0, 30);
  if (!rows.length) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128269;</div>No blocked A+ setups labeled yet</div>';
    return;
  }
  let spreadRows = rows.filter(r => r.is_spread_reject || String(r.reason || '').toLowerCase().includes('spread'));
  let spreadFalse = spreadRows.filter(r => r.outcome === 'missed_opportunity').length;
  let spreadCorrect = spreadRows.filter(r => r.outcome === 'correct_reject').length;
  let riskRows = rows.filter(r => r.is_risk_reject || String(r.reason || '').toLowerCase().includes('risk too wide') || String(r.reason || '').toLowerCase().includes('r:r'));
  let riskFalse = riskRows.filter(r => r.outcome === 'missed_opportunity').length;
  let riskCorrect = riskRows.filter(r => r.outcome === 'correct_reject').length;
  let riskSurvived = riskRows.filter(r => r.tactical_stop_survived === true).length;
  let riskFailed = riskRows.filter(r => r.tactical_stop_survived === false).length;
  let cleanRiskSurvived = riskRows.filter(r => r.tactical_stop_clean_survival === true).length;
  let cleanRiskFailed = riskRows.filter(r => r.tactical_stop_clean_survival === false).length;
  let choppyRiskSurvived = riskRows.filter(r => r.tactical_stop_survived === true && !r.smooth_for_tactical_stop).length;
  let html = '';
  if (spreadRows.length) {
    html += '<div class="mini-muted" style="margin-bottom:8px">Spread blocks: ' + spreadRows.length +
      ' | false blocks: ' + spreadFalse + ' | correct rejects: ' + spreadCorrect + '</div>';
  }
  if (riskRows.length) {
    html += '<div class="mini-muted" style="margin-bottom:8px">Wide-risk blocks: ' + riskRows.length +
      ' | false blocks: ' + riskFalse + ' | correct rejects: ' + riskCorrect +
      ' | clean tactical survived/failed: ' + cleanRiskSurvived + '/' + cleanRiskFailed +
      ' | choppy survived: ' + choppyRiskSurvived +
      ' | raw survived/failed: ' + riskSurvived + '/' + riskFailed + '</div>';
  }
  html += '<table><tr><th>Symbol</th><th>Pattern</th><th>Blocked At</th><th>Reason</th><th>Spread</th><th>Risk</th><th>Move After</th><th>Correct?</th><th>Fix</th></tr>';
  rows.forEach(r => {
    let move = Number(r.move_after_pct || 0);
    let cls = move >= 3 ? 'text-green' : (Number(r.dump_after_pct || 0) <= -3 ? 'text-red' : '');
    let correct = r.correct === true ? '<span class="text-green">Yes</span>' : (r.correct === false ? '<span class="text-red">No</span>' : '<span class="text-yellow">Pending</span>');
    let spread = Number(r.spread_pct || 0) > 0 ? Number(r.spread_pct || 0).toFixed(2) + '%' : '';
    let risk = '';
    if (Number(r.risk_pct || 0) > 0) {
      risk = Number(r.risk_pct || 0).toFixed(1) + '%';
      if (Number(r.tactical_stop_price || 0) > 0) {
        let survived = r.tactical_stop_survived === true ? 'survived' : (r.tactical_stop_survived === false ? 'failed' : 'pending');
        let smooth = r.smooth_for_tactical_stop ? 'smooth' : 'choppy';
        let range = Number(r.median_bar_range_pct || 0) > 0 ? ' ' + Number(r.median_bar_range_pct || 0).toFixed(1) + '%rng' : '';
        risk += '<div style="font-size:10px;color:var(--text2)">tac ' + Number(r.tactical_stop_price || 0).toFixed(2) + ' ' + survived + ' ' + smooth + range + '</div>';
      }
    }
    html += '<tr>';
    html += '<td><strong>' + escapeHtml(r.symbol || '') + '</strong></td>';
    html += '<td><span class="pill pill-blue">' + escapeHtml(r.pattern || r.scanner || '') + '</span><div style="font-size:10px;color:var(--text2)">' + escapeHtml(r.blocked_layer || '') + '</div></td>';
    html += '<td>' + shortTime(r.blocked_at) + '</td>';
    html += '<td style="max-width:360px;white-space:normal">' + escapeHtml(r.reason || '') + '</td>';
    html += '<td>' + escapeHtml(spread) + '</td>';
    html += '<td>' + risk + '</td>';
    html += '<td class="' + cls + '">' + move.toFixed(1) + '%</td>';
    html += '<td>' + correct + '</td>';
    html += '<td style="max-width:320px;white-space:normal">' + escapeHtml(r.suggested_fix || '') + '</td>';
    html += '</tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
}

function pctText(v) {
  if (v === null || v === undefined || isNaN(Number(v))) return 'n/a';
  return Number(v).toFixed(1).replace(/\.0$/, '') + '%';
}

function renderMLReport() {
  let box = state.ml_report || {};
  let report = box.report || null;
  let learningWrap = document.getElementById('ml-learning-wrap');
  let progressWrap = document.getElementById('ml-progress-wrap');
  let dayEl = document.getElementById('ml-report-day');
  if (!learningWrap || !progressWrap) return;

  if (dayEl) {
    dayEl.textContent = report && report.day
      ? 'Report ' + report.day
      : (box.last_update ? 'Checked ' + shortTime(box.last_update) : 'Not loaded');
  }
  if (box.error) {
    let err = '<div class="empty"><div class="icon">&#9888;&#65039;</div>' + escapeHtml(box.error) + '</div>';
    learningWrap.innerHTML = err;
    progressWrap.innerHTML = err;
    return;
  }
  if (!report) {
    let msg = box.message || 'No nightly report yet. It appears after the bot generates a report after market close.';
    learningWrap.innerHTML = '<div class="empty"><div class="icon">&#129302;</div>' + escapeHtml(msg) + '</div>';
    progressWrap.innerHTML = '<div class="empty"><div class="icon">&#128200;</div>' + escapeHtml(msg) + '</div>';
    return;
  }

  let ml = report.ml_learning || {};
  let missed = ml.missed_opportunities || {};
  let pull = ml.pullback_candidates || {};
  let exit = ml.exit_helper || {};
  let exec = ml.execution_quality || {};
  let entry = ml.entry_model || {};
  let shadow = ml.entry_shadow || {};

  let html = '<table><tr><th>Dataset</th><th>Tracked</th><th>Labeled</th><th>Result</th></tr>';
  html += '<tr><td><strong>Entry ML candidates</strong></td><td>' + (entry.total||0) + '</td><td>' + (entry.labeled||0) + '</td><td><span class="text-green">' + (entry.profitable||0) + '</span> profitable (' + pctText(entry.positive_rate) + '), avg ' + (entry.avg_outcome_pct||0) + '%</td></tr>';
  html += '<tr><td><strong>Entry ML reject shadow</strong></td><td>' + (shadow.total||0) + '</td><td>' + (shadow.labeled||0) + '</td><td><span class="text-green">' + (shadow.correct||0) + '</span> correct / <span class="text-red">' + (shadow.wrong||0) + '</span> wrong (' + pctText(shadow.positive_rate) + ')</td></tr>';
  html += '<tr><td><strong>Missed setups</strong></td><td>' + (missed.total||0) + '</td><td>' + (missed.labeled||0) + '</td><td><span class="text-green">' + (missed.went_up||0) + '</span> went up (' + pctText(missed.positive_rate) + ')</td></tr>';
  html += '<tr><td><strong>Pullback entries</strong></td><td>' + (pull.total||0) + '</td><td>' + (pull.labeled||0) + '</td><td><span class="text-green">' + (pull.worked||0) + '</span> worked (' + pctText(pull.positive_rate) + ')</td></tr>';
  html += '<tr><td><strong>Exit helper</strong></td><td>' + (exit.total||0) + '</td><td>' + (exit.labeled||0) + '</td><td><span class="text-green">' + (exit.hold_helped||0) + '</span> hold helped (' + pctText(exit.positive_rate) + ')</td></tr>';
  html += '<tr><td><strong>Execution quality</strong></td><td>' + (exec.total||0) + '</td><td>' + (exec.labeled||0) + '</td><td><span class="text-green">' + (exec.good_fills||0) + '</span> good fills (' + pctText(exec.good_rate) + '), avg slip ' + (exec.avg_slippage_pct||0) + '%</td></tr>';
  html += '</table>';
  let notes = [];
  if (entry.best) notes.push('Best entry ML sample: <strong>' + escapeHtml(entry.best.symbol) + '</strong> finished ' + entry.best.value + '%.');
  if (shadow.best_missed) notes.push('Biggest wrong ML reject: <strong>' + escapeHtml(shadow.best_missed.symbol) + '</strong> moved ' + shadow.best_missed.value + '% after rejection.');
  if (missed.best) notes.push('Best missed setup: <strong>' + escapeHtml(missed.best.symbol) + '</strong> moved ' + missed.best.value + '% after rejection.');
  if (pull.best) notes.push('Best pullback: <strong>' + escapeHtml(pull.best.symbol) + '</strong> moved ' + pull.best.value + '% after signal.');
  if (exec.worst) notes.push('Worst slippage: <strong>' + escapeHtml(exec.worst.symbol) + '</strong> at ' + exec.worst.value + '%.');
  if (notes.length) {
    html += '<div style="margin-top:10px;font-size:12px;color:var(--text2)">';
    notes.forEach(n => { html += '<div style="margin-top:4px">' + n + '</div>'; });
    html += '</div>';
  }

  let setups = report.setup_performance || [];
  if (setups.length) {
    html += '<div style="margin-top:14px;font-size:13px;font-weight:700">Live Setup Scorecard</div>';
    html += '<table style="margin-top:6px"><tr><th>Setup</th><th>Trades</th><th>Win</th><th>P&L</th><th>Pullbacks</th><th>Missed</th><th>Exit</th></tr>';
    setups.forEach(row => {
      let pnl = Number(row.total_pnl || 0);
      let pnlCls = pnl >= 0 ? 'text-green' : 'text-red';
      html += '<tr>';
      html += '<td><strong>' + escapeHtml(row.label || row.setup || '') + '</strong></td>';
      html += '<td>' + (row.trades || 0) + '</td>';
      html += '<td>' + pctText(row.win_rate) + '</td>';
      html += '<td><span class="' + pnlCls + '">$' + (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '</span></td>';
      html += '<td>' + (row.pullback_worked || 0) + '/' + (row.pullback_labeled || 0) + ' (' + pctText(row.pullback_rate) + ')</td>';
      html += '<td>' + (row.missed_went_up || 0) + '/' + (row.missed_labeled || 0) + ' (' + pctText(row.missed_rate) + ')</td>';
      html += '<td>' + (row.exit_hold_helped || 0) + '/' + (row.exit_labeled || 0) + ' (' + pctText(row.exit_hold_helped_rate) + ')</td>';
      html += '</tr>';
    });
    html += '</table>';
  }
  learningWrap.innerHTML = html;

  let prog = report.ml_progress || {};
  let phtml = '<div style="font-size:13px;color:var(--text2);margin-bottom:10px">';
  if (prog.previous_day) {
    let ch = prog.rows_change || 0;
    phtml += 'Compared with <strong style="color:var(--text)">' + escapeHtml(prog.previous_day) + '</strong>: '
      + (prog.rows_today||0) + ' rows today (' + (ch >= 0 ? '+' : '') + ch + ' vs previous).';
  } else {
    phtml += 'No previous ML report found yet. This report is the baseline.';
  }
  phtml += '<br>All-time shadow data: <strong style="color:var(--text)">' + (prog.all_time_rows||0) + '</strong> rows, <strong style="color:var(--text)">' + (prog.all_time_labeled||0) + '</strong> labeled.</div>';

  let datasets = prog.datasets || [];
  if (datasets.length) {
    phtml += '<table><tr><th>Dataset</th><th>Today Labeled</th><th>Today Rate</th><th>Previous</th><th>Change</th></tr>';
    datasets.forEach(row => {
      let change = Number(row.rate_change || 0);
      let cls = change >= 0 ? 'text-green' : 'text-red';
      phtml += '<tr><td><strong>' + escapeHtml(row.label || row.dataset) + '</strong></td><td>' + (row.labeled_today||0) + '</td><td>' + pctText(row.rate_today) + '</td><td>' + pctText(row.previous_rate) + '</td><td><span class="' + cls + '">' + (change >= 0 ? '+' : '') + change.toFixed(1).replace(/\.0$/, '') + '%</span></td></tr>';
    });
    phtml += '</table>';
  }

  let models = prog.models || [];
  if (models.length) {
    phtml += '<div style="margin-top:12px"><table><tr><th>Shadow Model</th><th>Status</th><th>Samples</th><th>Accuracy</th><th>Positive Rate</th></tr>';
    models.forEach(m => {
      let active = m.status === 'trained';
      phtml += '<tr><td><strong>' + escapeHtml(m.model || '') + '</strong></td><td><span class="pill ' + (active ? 'pill-green' : 'pill-yellow') + '">' + escapeHtml(m.status || 'unknown') + '</span></td><td>' + (m.samples||0) + '</td><td>' + (m.test_accuracy == null ? 'collecting' : pctText(Number(m.test_accuracy) * 100)) + '</td><td>' + (m.positive_rate == null ? 'collecting' : pctText(Number(m.positive_rate) * 100)) + '</td></tr>';
    });
    phtml += '</table></div>';
  }
  progressWrap.innerHTML = phtml;
}

function renderAI() {
  let ai = state.ai_analysis || {};
  let insights = ai.insights || [];
  let blocked = ai.blocked_symbols || {};
  let score = ai.score;

  if (score !== undefined && score !== null) {
    let scoreColor = score >= 60 ? 'var(--green)' : score >= 40 ? 'var(--yellow)' : 'var(--red)';
    document.getElementById('ai-score').innerHTML =
      'Session Score: <strong style="color:'+scoreColor+'">'+score.toFixed(0)+'/100</strong>' +
      (ai.last_analysis ? ' (updated '+ai.last_analysis+')' : '');
  }

  let wrap = document.getElementById('ai-insights-wrap');
  if (insights.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#129302;</div>AI analysis will run after 5+ trades</div>';
  } else {
    let html = '';
    let icons = {critical:'&#9888;&#65039;', warning:'&#9888;', info:'&#8505;&#65039;', positive:'&#9989;'};
    let colors = {critical:'var(--red)', warning:'#f59e0b', info:'var(--blue)', positive:'var(--green)'};
    insights.forEach(ins => {
      let icon = icons[ins.severity] || '&#8226;';
      let color = colors[ins.severity] || 'var(--text2)';
      html += '<div style="padding:8px 12px;border-left:3px solid '+color+';margin-bottom:6px;background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0">';
      html += '<div style="font-size:13px"><span>'+icon+'</span> <strong style="color:'+color+'">'+ins.category.toUpperCase()+'</strong>: '+ins.message+'</div>';
      if (ins.action_taken) {
        html += '<div style="font-size:11px;color:var(--green);margin-top:2px">&#8594; '+ins.action_taken+'</div>';
      }
      html += '</div>';
    });
    wrap.innerHTML = html;
  }

  let blockedWrap = document.getElementById('ai-blocked-wrap');
  let blockedEntries = Object.entries(blocked);
  if (blockedEntries.length === 0) {
    blockedWrap.innerHTML = '<div style="padding:12px;font-size:12px;color:var(--text2)">No symbols blocked</div>';
  } else {
    let bhtml = '<table><tr><th>Symbol</th><th>Reason</th></tr>';
    blockedEntries.forEach(([sym, reason]) => {
      bhtml += '<tr><td><strong class="text-red">'+sym+'</strong></td><td style="font-size:12px">'+reason+'</td></tr>';
    });
    bhtml += '</table>';
    blockedWrap.innerHTML = bhtml;
  }

  let adjWrap = document.getElementById('ai-adjustments-wrap');
  let adjs = ai.session_adjustments || {};
  let adjEntries = Object.entries(adjs);
  if (adjEntries.length === 0) {
    adjWrap.innerHTML = '<div style="padding:12px;font-size:12px;color:var(--text2)">No adjustments yet</div>';
  } else {
    let ahtml = '<table><tr><th>Parameter</th><th>New Value</th></tr>';
    adjEntries.forEach(([param, val]) => {
      ahtml += '<tr><td>'+param+'</td><td><strong>'+val+'</strong></td></tr>';
    });
    ahtml += '</table>';
    adjWrap.innerHTML = ahtml;
  }
}

function renderJournal() {
  let jr = state.journal || {};
  let events = jr.events || [];
  let byType = {};
  events.forEach(e => { byType[e.type] = (byType[e.type] || 0) + 1; });
  document.getElementById('jr-total').textContent = events.length;
  document.getElementById('jr-trades').textContent = (byType.trade_fill || 0) + (byType.trade_exit || 0);
  document.getElementById('jr-mistakes').textContent = byType.mistake || 0;
  document.getElementById('jr-shots').textContent = byType.screenshot || 0;
  document.getElementById('jr-updated').textContent =
    jr.last_update ? ('Updated ' + shortTime(jr.last_update)) : 'Not loaded';

  let wrap = document.getElementById('journal-table-wrap');
  if (jr.error) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#9888;&#65039;</div>' + escapeHtml(jr.error) + '</div>';
    return;
  }
  if (events.length === 0) {
    wrap.innerHTML = '<div class="empty"><div class="icon">&#128221;</div>No journal events yet</div>';
    return;
  }

  let typeColors = {
    trade_fill:'pill-green', trade_exit:'pill-red', classification:'pill-purple',
    scan_hit:'pill-blue', signal:'pill-yellow', mistake:'pill-red',
    cycle:'pill-blue', market_regime:'pill-purple', market_context:'pill-blue',
    screenshot:'pill-yellow'
  };

  let rows = events.slice().reverse().slice(0, 200).map(ev => {
    let p = ev.payload || {};
    let sym = p.symbol || '';
    let pillCls = typeColors[ev.type] || 'pill-blue';
    let label = (ev.type || '').replace(/_/g, ' ').toUpperCase();

    let detail = '';
    if (ev.type === 'trade_fill' || ev.type === 'trade_exit') {
      let side = p.side || p.trade_type || '';
      let price = p.price || p.entry_price || p.exit_price || '';
      let qty = p.quantity || '';
      let pnl = p.pnl != null ? fmtPnl(p.pnl) : '';
      detail = escapeHtml(side.toUpperCase()) + ' ' + escapeHtml(qty) + ' @ $' + escapeHtml(price) + (pnl ? ' ' + pnl : '');
    } else if (ev.type === 'classification') {
      detail = escapeHtml(p.style || '') + (p.confidence != null ? ' (' + (p.confidence * 100).toFixed(0) + '%)' : '');
    } else if (ev.type === 'scan_hit' || ev.type === 'signal') {
      detail = escapeHtml(p.scanner_name || p.pattern || '');
      if (p.price) detail += ' @ $' + escapeHtml(p.price);
    } else if (ev.type === 'mistake') {
      detail = '<span class="text-red">' + escapeHtml(p.kind || '') + '</span>: ' + escapeHtml(p.reason || '');
    } else if (ev.type === 'cycle') {
      detail = '#' + (p.cycle || '') + ' scanned:' + (p.symbols_scanned || 0) + ' hits:' + (p.scan_hits || 0) + ' fills:' + (p.fills || 0);
    } else if (ev.type === 'market_regime') {
      detail = escapeHtml(p.phase || '') + (p.regime_label ? ' — ' + escapeHtml(p.regime_label) : '');
    } else {
      let raw = JSON.stringify(p);
      detail = escapeHtml(raw.length > 120 ? raw.slice(0, 120) + '...' : raw);
    }

    return '<tr>'
      + '<td style="white-space:nowrap">' + shortTime(ev.ts) + '</td>'
      + '<td><span class="pill ' + pillCls + '">' + label + '</span></td>'
      + '<td>' + (sym ? chartLink(sym) : '<span style="color:var(--text2)">—</span>') + '</td>'
      + '<td style="font-size:12px">' + detail + '</td>'
      + '</tr>';
  }).join('');

  wrap.innerHTML = '<div style="max-height:500px;overflow-y:auto"><table>'
    + '<tr><th>Time</th><th>Type</th><th>Symbol</th><th>Details</th></tr>'
    + rows
    + '</table></div>';
}

function loadJournal(force) {
  let jr = state.journal || {};
  let now = Date.now();
  if (!force && jr.last_fetch_ms && (now - jr.last_fetch_ms) < 10000) return;
  fetchJsonWithRetry('/api/replay?limit=300')
    .then(data => {
      if (!data.ok) throw new Error(data.error || 'failed to load journal');
      state.journal = {
        events: data.events || [],
        loaded: true,
        error: null,
        last_update: new Date().toISOString(),
        last_fetch_ms: Date.now()
      };
      (state.journal.events || []).slice(-80).forEach(ev => {
        let log = replayEventToLog(ev);
        if (log) pushLogLine(log);
      });
      renderJournal();
    })
    .catch(err => {
      state.journal = {
        events: (state.journal && state.journal.events) || [],
        loaded: true,
        error: err && err.message ? err.message : 'Failed to load journal',
        last_update: (state.journal && state.journal.last_update) || null,
        last_fetch_ms: Date.now()
      };
      renderJournal();
    });
}

function loadMLReport(force) {
  let current = state.ml_report || {};
  let now = Date.now();
  if (!force && current.last_fetch_ms && (now - current.last_fetch_ms) < 30000) return;
  let url = '/api/analytics-report' + (force ? ('?_=' + now) : '');
  fetchJsonWithRetry(url)
    .then(data => {
      if (!data.ok) throw new Error(data.error || 'failed to load analytics report');
      state.ml_report = {
        report: data.report || null,
        loaded: true,
        error: null,
        message: data.message || null,
        last_update: data.loaded_at || new Date().toISOString(),
        last_fetch_ms: Date.now()
      };
      renderMLReport();
    })
    .catch(err => {
      state.ml_report = {
        report: (state.ml_report && state.ml_report.report) || null,
        loaded: true,
        error: err && err.message ? err.message : 'Failed to load analytics report',
        message: null,
        last_update: (state.ml_report && state.ml_report.last_update) || null,
        last_fetch_ms: Date.now()
      };
      renderMLReport();
    });
}


document.querySelectorAll('.hod-filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.hod-filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.hod_momentum_filter = btn.dataset.filter || 'all';
    renderHodMomentumScanner();
  });
});

function renderAll() {
  renderOverview();
  renderMovers();
  renderScanner();
  renderTrades();
  renderAnalytics();
  renderBacktest();
  renderNews();
  renderJournal();
  updateStatus();
}

function updateStatus() {
  let dm = document.getElementById('dot-market');
  let ds = document.getElementById('dot-stream');
  let phase = state.market_phase || (state.market_open ? 'OPEN' : 'CLOSED');
  let isActive = phase !== 'CLOSED';
  dm.className = 'status-dot ' + (isActive ? 'on' : 'off');
  ds.className = 'status-dot ' + (state.stream_connected ? 'on' : 'off');
  document.getElementById('market-label').textContent = isActive ? 'Market Active' : 'Market Closed';
  document.getElementById('stream-label').textContent = state.stream_connected ? 'Stream Live' : 'Stream Off';
  document.getElementById('cycle-count').textContent = state.stats.cycle_count || 0;

  let pill = document.getElementById('phase-pill');
  let phaseColors = {'OPEN':'pill-green','PRE-MARKET':'pill-yellow','AFTER-HOURS':'pill-purple','CLOSED':'pill-red'};
  pill.className = 'pill ' + (phaseColors[phase]||'pill-red');
  pill.textContent = phase;
  updateTradeControls();
}

function currentCycleCount() {
  return Number((state.stats && state.stats.cycle_count) || 0) || 0;
}

function setCycleCountMonotonic(value) {
  let next = Number(value || 0) || 0;
  state.stats = state.stats || {};
  state.stats.cycle_count = Math.max(currentCycleCount(), next);
}

function fetchJsonWithRetry(url, options, attempts) {
  let tries = attempts || 3;
  function once(left) {
    return fetch(url, options || {}).then(r => {
      if (!r.ok) {
        return r.text().then(text => {
          throw new Error('HTTP ' + r.status + ' ' + (text || r.statusText));
        });
      }
      return r.json();
    }).catch(err => {
      if (left <= 1) throw err;
      return new Promise(resolve => setTimeout(resolve, 500)).then(() => once(left - 1));
    });
  }
  return once(tries);
}

function flashScanner() {
  let dot = document.getElementById('dot-scanner');
  let label = document.getElementById('scanner-label');
  dot.className = 'status-dot on';
  label.textContent = 'Scanning... ' + (state.last_scan_time || '');
  setTimeout(() => {
    dot.className = 'status-dot on';
    label.textContent = 'Last scan ' + (state.last_scan_time || '');
  }, 2000);
}

function addLogLine(msg) {
  let panel = document.getElementById('log-panel');
  if (!panel || !msg) return;
  if (panel.querySelector('.empty')) panel.innerHTML = '';
  let cls = '';
  if (msg.level === 'WARNING') cls = ' warn';
  if (msg.level === 'ERROR') cls = ' error';
  let line = document.createElement('div');
  line.className = 'log-line' + cls;
  line.innerHTML = '<span class="ts">'+shortTime(msg.ts)+'</span> ' + msg.message;
  panel.appendChild(line);
  if (panel.children.length > 200) panel.removeChild(panel.firstChild);
  panel.scrollTop = panel.scrollHeight;
}

// Deduplicate scans: keep only latest per symbol+scanner
function dedupeScans(scans) {
  let map = {};
  scans.forEach(s => { map[s.symbol + '|' + s.scanner_name] = s; });
  return Object.values(map);
}

// Apply full state from server push (SSE snapshot event — no HTTP poll)
function applySnapshot(data) {
  let savedJournal = state.journal;
  let savedMLReport = state.ml_report;
  let savedLogs = state.logs || [];
  let savedCycle = currentCycleCount();
  let savedBotStart = state.bot_start_time || null;
  let incomingBotStart = data.bot_start_time || null;
  let sameBotRun = !savedBotStart || !incomingBotStart || savedBotStart === incomingBotStart;
  let incomingCycle = Number((data.stats && data.stats.cycle_count) || 0) || 0;
    state = data;
  state.journal = savedJournal || {events: [], loaded: false, error: null, last_update: null};
  state.ml_report = savedMLReport || {report: null, loaded: false, error: null, message: null, last_update: null};
  state.stats = state.stats || {};
  state.stats.cycle_count = sameBotRun ? Math.max(savedCycle, incomingCycle) : incomingCycle;
  state.trading_paused = data.trading_paused === true;
    state.recent_scans = dedupeScans(state.recent_scans || []);
  state.rt_movers = data.rt_movers || [];
  state.hod_momentum_alerts = data.hod_momentum_alerts || [];
  state.hot_watch = data.hot_watch || [];
  state.missed_a_plus = data.missed_a_plus || [];
  state.daily_scorecard = data.daily_scorecard || {};
  state.rolling_scorecard = data.rolling_scorecard || {};
  state.trading_watchlist = data.trading_watchlist || [];
  state.candidate_hydration = data.candidate_hydration || {};
  state.watchlist_pinned = data.watchlist_pinned || ['SPY'];
    state.rt_new_total = 0;
    state.news = data.news || {};
  state.logs = data.logs || savedLogs;
  renderAll();
  updateTradeControls();
  loadJournal(true);
  loadMLReport(true);
}

let jrRefreshBtn = document.getElementById('jr-refresh-btn');
if (jrRefreshBtn) {
  jrRefreshBtn.addEventListener('click', function() { loadJournal(true); });
}
let mlReportRefreshBtn = document.getElementById('ml-report-refresh-btn');
if (mlReportRefreshBtn) {
  mlReportRefreshBtn.addEventListener('click', function() { loadMLReport(true); });
}

// SSE stream for real-time updates with auto-reconnect
function connectSSE() {
let es = new EventSource('/api/stream');
es.onmessage = function(e) {
  let msg = JSON.parse(e.data);
  switch(msg.type) {
    case 'snapshot':
      applySnapshot(msg.data);
      break;
    case 'account':
      state.account = msg.data;
      renderOverview();
      renderPositionsCompact();
      break;
    case 'trade':
      state.recent_trades.push(msg.data);
      state.stats.total_trades = (state.stats.total_trades||0) + (msg.data.trade_type==='entry'?1:0);
      pushLogLine({
        level: 'INFO',
        ts: msg.data.ts || new Date().toISOString(),
        message: 'TRADE ' + (msg.data.symbol || '') + ' ' + (msg.data.trade_type || '')
          + (msg.data.price ? ' @ $' + msg.data.price : ''),
      });
      renderOverview();
      renderTrades();
      renderAnalytics();
      break;
    case 'exit':
      state.recent_trades.push(msg.data);
      if (msg.data.pnl != null) {
        state.stats.total_pnl = (state.stats.total_pnl||0) + msg.data.pnl;
        if (msg.data.pnl >= 0) state.stats.winning_trades = (state.stats.winning_trades||0) + 1;
        else state.stats.losing_trades = (state.stats.losing_trades||0) + 1;
        let total = (state.stats.winning_trades||0) + (state.stats.losing_trades||0);
        state.stats.win_rate = total > 0 ? ((state.stats.winning_trades/total)*100).toFixed(1) : 0;
      }
      pushLogLine({
        level: Number(msg.data.pnl || 0) < 0 ? 'WARNING' : 'INFO',
        ts: msg.data.ts || new Date().toISOString(),
        message: 'EXIT ' + (msg.data.symbol || '') + (msg.data.exit_price ? ' @ $' + msg.data.exit_price : '')
          + (msg.data.pnl != null ? ' P&L ' + fmtPnl(msg.data.pnl) : ''),
      });
      renderOverview();
      renderTrades();
      renderAnalytics();
      break;
    case 'classification':
      state.symbols[msg.data.symbol] = msg.data;
      if (msg.data.style === 'not_tradeable') {
        state.rt_movers = (state.rt_movers||[]).filter(m => m.symbol !== msg.data.symbol);
      }
      renderMovers();
      break;
    case 'hod_momentum_alerts':
      state.hod_momentum_alerts = msg.data.alerts || [];
      renderHodMomentumScanner();
      renderTradingWatchlist();
      if ((msg.data.alerts || []).length > 0) flashScanner();
      break;
    case 'trading_watchlist':
      state.trading_watchlist = msg.data.symbols || [];
      state.watchlist_pinned = msg.data.pinned || state.watchlist_pinned || ['SPY'];
      renderTradingWatchlist();
      break;
    case 'hot_watch':
      state.hot_watch = msg.data.symbols || [];
      renderHotWatch();
      renderTradingWatchlist();
      break;
    case 'candidate_hydration':
      state.candidate_hydration = msg.data || {};
      renderCandidateWorker();
      break;
    case 'missed_a_plus':
      state.missed_a_plus = msg.data.rows || [];
      if (state.missed_a_plus.length > 0) {
        let row = state.missed_a_plus[0];
        pushLogLine({
          level: 'WARNING',
          ts: row.blocked_at || new Date().toISOString(),
          message: 'MISSED A+ ' + row.symbol + ' ' + row.pattern + ': ' + row.reason
            + (row.move_after_pct != null ? ' move +' + Number(row.move_after_pct).toFixed(1) + '%' : ''),
        });
      }
      renderAnalytics();
      break;
    case 'scan_hit':
      // Replace existing entry for same symbol+scanner, keep only latest
      let idx = state.recent_scans.findIndex(s => s.symbol === msg.data.symbol && s.scanner_name === msg.data.scanner_name);
      if (idx >= 0) {
        state.recent_scans.splice(idx, 1);
      } else {
        state.stats.total_scan_hits = (state.stats.total_scan_hits||0) + 1;
      }
      state.recent_scans.push(msg.data);
      pushLogLine({
        level: 'INFO',
        ts: msg.data.ts || new Date().toISOString(),
        message: 'SCAN HIT ' + (msg.data.symbol || '') + ' ' + (msg.data.scanner_name || msg.data.pattern || '')
          + (msg.data.price ? ' @ $' + msg.data.price : ''),
      });
      renderScanner();
      renderOverview();
      renderAnalytics();
      break;
    case 'positions':
      state.positions = msg.data;
      renderOverview();
      renderPositionsCompact();
      break;
    case 'cycle':
      setCycleCountMonotonic(msg.data.cycle);
      pushLogLine({
        level: 'INFO',
        ts: msg.data.ts || new Date().toISOString(),
        message: 'CYCLE #' + (msg.data.cycle || '') + ' scanned=' + (msg.data.symbols_scanned || 0)
          + ' hits=' + (msg.data.scan_hits || 0) + ' signals=' + (msg.data.signals || 0)
          + ' fills=' + (msg.data.fills || 0),
      });
      updateStatus();
      renderAnalytics();
      break;
    case 'market_status':
      state.market_open = msg.data.market_open;
      state.stream_connected = msg.data.stream_connected;
      if (msg.data.market_phase) state.market_phase = msg.data.market_phase;
      pushLogLine({
        level: 'INFO',
        ts: new Date().toISOString(),
        message: 'MARKET ' + (state.market_phase || '') + ' stream='
          + (state.stream_connected ? 'connected' : 'off'),
      });
      updateStatus();
      break;
    case 'watchlist_scan':
      state.watchlist_scan = msg.data.stocks;
      state.last_scan_time = new Date().toLocaleTimeString();
      break;
    case 'news':
      state.news[msg.data.symbol] = msg.data;
      renderNews();
      if (msg.data.sentiment === 'positive') {
        pushLogLine({level:'INFO', ts: msg.data.ts, message: 'NEWS BOOST ' + msg.data.symbol + ': +' + msg.data.score.toFixed(1) + ' — ' + (msg.data.headlines[0]||'positive news')});
      } else if (msg.data.sentiment === 'negative') {
        pushLogLine({level:'WARNING', ts: msg.data.ts, message: 'NEWS BLOCK ' + msg.data.symbol + ': ' + msg.data.score.toFixed(1) + ' — ' + (msg.data.headlines[0]||'negative news')});
      }
      break;
    case 'rt_movers':
      state.rt_movers = msg.data.movers;
      state.rt_new_total = (state.rt_new_total||0) + (msg.data.new_symbols||[]).length;
      document.getElementById('mv-last-time').textContent = new Date().toLocaleTimeString();
      renderMovers();
      // Flash new symbols in log
      if (msg.data.new_symbols && msg.data.new_symbols.length > 0) {
        pushLogLine({level:'INFO', ts: msg.data.scan_time, message: 'NEW MOVERS: ' + msg.data.new_symbols.join(', ')});
      }
      break;
    case 'ai_update':
      state.ai_analysis = msg.data;
      renderAI();
      break;
    case 'trading_control':
      if (msg.data.paused !== undefined) {
        state.trading_paused = msg.data.paused;
        updateTradeControls();
      }
      if (msg.data.force_closed) {
        state.positions = {};
        renderOverview();
        pushLogLine({level:'WARNING', ts: new Date().toISOString(), message: 'ALL POSITIONS FORCE CLOSED via dashboard'});
      }
      break;
    case 'log':
      pushLogLine(msg.data);
      break;
  }
};
es.onerror = function() {
  console.log('SSE disconnected, reconnecting in 2s...');
  es.close();
  setTimeout(connectSSE, 2000);
};
}
connectSSE();

// Hydrate immediately from REST (fallback if SSE is slow or blocked)
fetchJsonWithRetry('/api/snapshot').then(data => {
  if (data && data.market_phase) applySnapshot(data);
  }).catch(()=>{});
loadMLReport(true);

// Keep pause/resume controls truthful even if an SSE event is missed.
function syncTradingControlState() {
  fetchJsonWithRetry('/api/snapshot').then(data => {
    if (!data) return;
    let savedBotStart = state.bot_start_time || null;
    let incomingBotStart = data.bot_start_time || null;
    let sameBotRun = !savedBotStart || !incomingBotStart || savedBotStart === incomingBotStart;
    state.trading_paused = data.trading_paused === true;
    state.market_open = data.market_open;
    state.stream_connected = data.stream_connected;
    if (incomingBotStart) state.bot_start_time = incomingBotStart;
    if (data.market_phase) state.market_phase = data.market_phase;
    if (data.stats) {
      let savedCycle = currentCycleCount();
      state.stats = Object.assign(state.stats || {}, data.stats);
      let incomingCycle = Number(data.stats.cycle_count || 0) || 0;
      state.stats.cycle_count = sameBotRun ? Math.max(savedCycle, incomingCycle) : incomingCycle;
    }
    if (data.daily_scorecard) state.daily_scorecard = data.daily_scorecard;
    if (data.rolling_scorecard) state.rolling_scorecard = data.rolling_scorecard;
    if (data.missed_a_plus) state.missed_a_plus = data.missed_a_plus;
    updateStatus();
    renderAnalytics();
  }).catch(()=>{});
}
setInterval(syncTradingControlState, 10000);

// ML Stats polling (every 30s)
function fetchMLStats() {
  fetchJsonWithRetry('/api/ml-stats').then(data => {
    let wrap = document.getElementById('ml-stats-wrap');
    let badge = document.getElementById('ml-status-badge');
    if (!data || data.enabled === false) {
      wrap.innerHTML = '<div style="padding:12px;color:var(--text2);font-size:13px;">ML model not loaded</div>';
      badge.textContent = 'OFF';
      badge.className = 'badge pill-red';
      return;
    }
    badge.textContent = data.model_active ? 'ACTIVE' : 'DISABLED';
    badge.className = 'badge ' + (data.model_active ? 'pill-green' : 'pill-red');
    let html = '<div style="padding:12px;font-size:13px;">';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">';
    html += '<div><span style="color:var(--text2)">ML/Quality Passed:</span> <b style="color:var(--green)">' + data.entries_passed + '</b></div>';
    html += '<div><span style="color:var(--text2)">ML Rejected:</span> <b style="color:var(--red)">' + data.entries_rejected_by_ml + '</b></div>';
    html += '<div><span style="color:var(--text2)">Rule Rejected:</span> <b style="color:var(--yellow)">' + (data.entries_rejected_by_rules || 0) + '</b></div>';
    html += '<div><span style="color:var(--text2)">Reject Rate:</span> <b>' + data.rejection_rate_pct + '%</b></div>';
    html += '<div><span style="color:var(--text2)">Shadow Acc:</span> <b style="color:' + (data.shadow_accuracy_pct >= 50 ? 'var(--green)' : 'var(--red)') + '">' + data.shadow_accuracy_pct + '%</b></div>';
    html += '</div>';
    html += '<div style="color:var(--text2);font-size:11px;">Shadow: ' + data.shadow_correct + ' correct / ' + data.shadow_wrong + ' wrong</div>';
    if (data.model_disabled) {
      html += '<div style="margin-top:6px;padding:4px 8px;background:var(--red-bg);border-radius:6px;font-size:11px;color:var(--red);">Auto-disabled: ' + data.disable_reason + '</div>';
    }
    html += '</div>';
    wrap.innerHTML = html;
  }).catch(()=>{});
}
fetchMLStats();
setInterval(fetchMLStats, 30000);

// Trading control buttons
function updateTradeControls() {
  let paused = state.trading_paused === true;
  let btnStop = document.getElementById('btn-stop');
  let btnStart = document.getElementById('btn-start');
  let pill = document.getElementById('trade-status-pill');
  if (!btnStop || !btnStart || !pill) return;
  btnStop.disabled = paused;
  btnStart.disabled = !paused;
  pill.textContent = paused ? 'PAUSED' : 'ACTIVE';
  pill.className = 'pill ' + (paused ? 'pill-red' : 'pill-green');
  pill.title = paused ? 'Trading is paused: no new entries will be placed' : 'Trading is active';
}

document.getElementById('btn-stop').addEventListener('click', function() {
  if (!confirm('Stop trading? No new entries will be placed.')) return;
  fetchJsonWithRetry('/api/pause', {method:'POST'}).then(data => {
    if (data.ok) {
      state.trading_paused = true;
      updateTradeControls();
      addLogLine({level:'WARNING', ts: new Date().toISOString(), message: 'Trading PAUSED by user'});
    }
  }).catch(()=>{});
});

document.getElementById('btn-start').addEventListener('click', function() {
  fetchJsonWithRetry('/api/resume', {method:'POST'}).then(data => {
    if (data.ok) {
      state.trading_paused = false;
      updateTradeControls();
      addLogLine({level:'INFO', ts: new Date().toISOString(), message: 'Trading RESUMED by user'});
    }
  }).catch(()=>{});
});

document.getElementById('btn-force-close').addEventListener('click', function() {
  if (!confirm('FORCE CLOSE ALL positions? This will cancel all open orders and liquidate everything immediately.')) return;
  let btn = document.getElementById('btn-force-close');
  btn.disabled = true;
  addLogLine({level:'WARNING', ts: new Date().toISOString(), message: 'EMERGENCY CLOSE requested'});
  fetchJsonWithRetry('/api/force-close', {method:'POST'}).then(data => {
    if (data.flat) {
      state.trading_paused = true;
      updateTradeControls();
      addLogLine({level:'WARNING', ts: new Date().toISOString(), message: 'ALL POSITIONS FORCE CLOSED; trading paused'});
      state.positions = {};
      renderOverview();
    } else {
      let remaining = data.remaining_positions ? Object.keys(data.remaining_positions).join(', ') : '';
      alert('Emergency close did not flatten everything. Remaining: ' + (remaining || 'unknown') + '. Check Alpaca now.');
    }
  }).catch(err => {
    alert('Emergency close failed: ' + err.message);
  }).finally(() => {
    btn.disabled = false;
  });
});

// Journal tab only — loaded on demand when that page is visible
setInterval(function() {
  let page = document.getElementById('page-journal');
  if (page && page.classList.contains('active')) {
    loadJournal(false);
  }
}, 10000);
setInterval(function() {
  let page = document.getElementById('page-analytics');
  if (page && page.classList.contains('active')) {
    loadMLReport(false);
  }
}, 30000);
</script>
</body>
</html>
"""
