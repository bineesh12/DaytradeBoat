"""Flask web server for the trading dashboard.

Serves a single-page app with real-time updates via SSE.
Runs in a background daemon thread so it doesn't block the trading pipeline.
"""

from __future__ import annotations

import json
import hmac
import ipaddress
import logging
import math
import os
import threading
import time
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlsplit
from pathlib import Path

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import BadRequest

from daytrading.dashboard.hub import DashboardHub

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Return a browser-safe JSON value.

    Python's JSON encoder allows NaN/Infinity by default, but browser
    JSON.parse rejects them. Analytics reports can legitimately produce
    infinite ratios (for example a profit factor with no losses), so coerce
    non-finite numbers to null at the API boundary.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_safe(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return str(value)
    return value


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
    app = Flask(__name__)
    analytics_generation_lock = threading.Lock()
    backtest_lock = threading.Lock()
    app.config["PROPAGATE_EXCEPTIONS"] = True
    access_token = os.environ.get("DAYTRADING_DASHBOARD_TOKEN", "").strip()

    @app.before_request
    def authorize_dashboard():
        # Basic authentication also works for browser EventSource requests.
        if access_token:
            auth = request.authorization
            supplied = auth.password if auth and auth.type == "basic" else ""
            if request.headers.get("Authorization", "").startswith("Bearer "):
                supplied = request.headers["Authorization"][7:]
            if not supplied or not hmac.compare_digest(supplied.encode(), access_token.encode()):
                return jsonify({"ok": False, "error": "authentication required"}), 401, {
                    "WWW-Authenticate": 'Basic realm="DaytradeBoat"',
                }
        else:
            try:
                local = ipaddress.ip_address(request.remote_addr or "").is_loopback
            except ValueError:
                local = False
            if not local:
                return jsonify({"ok": False, "error": "remote dashboard requires authentication"}), 403
            hostname = urlsplit(request.host_url).hostname
            if hostname not in ("localhost", "127.0.0.1", "::1"):
                return jsonify({"ok": False, "error": "untrusted localhost host header"}), 403
        # Stop a website from operating a localhost dashboard via the browser.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("Origin")
            if request.headers.get("Sec-Fetch-Site") == "cross-site" or (
                origin and urlsplit(origin).netloc != request.host
            ):
                return jsonify({"ok": False, "error": "cross-origin control request rejected"}), 403

    @app.after_request
    def add_no_cache(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
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
        resp = make_response(dashboard_html())
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.route("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.route("/api/snapshot")
    def snapshot():
        return jsonify(_json_safe(hub.snapshot()))

    @app.route('/api/entry-setups')
    def entry_setups():
        return jsonify(_json_safe(hub.opportunities.snapshot(
            query=request.args.get('query','')[:100], status=request.args.get('status','')[:30], limit=200)))

    @app.route("/api/paper-handover", methods=["POST"])
    def paper_handover():
        """Local authenticated maintenance, never an order or entry override."""
        payload=request.get_json(silent=True) or {}
        action=payload.get("action", "checkpoint")
        if action not in ("checkpoint", "release"):
            return jsonify({"ok":False,"error":"unsupported handover action"}),400
        if not hub.trading_paused:
            return jsonify({"ok":False,"error":"pause new entries before checkpointing"}),409
        callback=getattr(hub,"paper_handover_release" if action=="release" else "paper_handover_callback",None)
        if callback is None:
            return jsonify({"ok":False,"error":"paper handover unavailable"}),503
        try:
            result=callback() if action=="release" else callback(freeze=payload.get("freeze") is True)
            return jsonify({"ok":True,"checkpoint":result} if action=="checkpoint" else {"ok":True,**result})
        except ValueError as exc:
            return jsonify({"ok":False,"error":str(exc)}),409

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
        rows = list(getattr(hub, "missed_a_plus", []))
        return jsonify({
            "ok": True,
            "rows": rows,
            "spread_summary": _missed_a_plus_spread_summary(rows),
            "risk_summary": _missed_a_plus_risk_summary(rows),
            "scanner_near_miss": dict(getattr(hub, "scanner_near_miss", {}) or {}),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        })

    @app.route("/api/analytics-report")
    def analytics_report():
        """Return the latest nightly analytics report for dashboard display."""
        try:
            report_dir = None
            if getattr(hub, "journal", None):
                base_dir = getattr(hub.journal, "base_dir", None)
                if base_dir:
                    report_dir = os.path.join(os.path.dirname(base_dir), "reports")
            report_dir = report_dir or os.environ.get("DAYTRADING_REPORT_DIR", "data/reports")
            report_dir = os.path.abspath(report_dir)
            requested_day = request.args.get("day")
            explicit_day = bool(requested_day)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            has_journal_db = bool(getattr(getattr(hub, "journal", None), "db_path", None))

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
                                db_path=hub.journal.db_path,
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
            if explicit_day and getattr(hub, "journal", None):
                try:
                    from daytrading.analyst.collector import NightlyAnalyst
                    analyst = NightlyAnalyst(
                        db_path=hub.journal.db_path,
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
            return jsonify(_json_safe({
                "ok": True,
                "day": day,
                "report": report,
                "generated_at": report.get("generated_at"),
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }))
        except Exception as exc:
            logger.error("Failed to load analytics report: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/screenshot", methods=["POST"])
    def save_screenshot():
        if not getattr(hub, "journal", None):
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
            meta = hub.journal.save_screenshot(
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
        if not getattr(hub, "journal", None):
            return jsonify({"ok": False, "error": "journal not configured"}), 503
        day = request.args.get("day")
        try:
            limit = int(request.args.get("limit", "0")) or None
        except Exception:
            limit = None
        events = hub.journal.replay_frames(day=day, limit=limit)
        return jsonify({"ok": True, "count": len(events), "events": events})

    @app.route("/api/backtest/sessions", methods=["GET"])
    def recorded_backtest_sessions():
        from daytrading.replay.store import list_sessions
        return jsonify({"ok": True, "sessions": list_sessions()})

    @app.route("/api/backtest/jobs/<job_id>", methods=["GET"])
    def recorded_backtest_job(job_id):
        from daytrading.replay.service import job_status
        try:
            result = job_status(job_id)
            return jsonify(result), (200 if result.get("ok") else 404)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

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
            mode = str(payload.get('mode', 'historical_estimate'))
            if mode == 'historical_estimate':
                from daytrading.backtest.historical_worker import run_isolated_historical
                result = run_isolated_historical(symbol, day, flags=flags, start_time=start_time)
            else:
                result = run_backtest(symbol, day, flags=flags, mode=mode,
                    session_id=payload.get('session_id'), start_time=start_time, settings=Settings())
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
            from daytrading.backtest.historical_worker import run_isolated_sweep
            result = run_isolated_sweep(
                symbols,
                dates,
                experiments=experiments,
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
        hub.trading_paused = True
        capture_control=getattr(hub,"recording_control",None)
        if capture_control:capture_control(True)
        hub._broadcast("trading_control", {"paused": True})
        logger.info("Trading PAUSED via dashboard")
        return jsonify({"ok": True, "paused": True})

    @app.route("/api/resume", methods=["POST"])
    def resume_trading():
        hub.trading_paused = False
        capture_control=getattr(hub,"recording_control",None)
        if capture_control:capture_control(False)
        hub._broadcast("trading_control", {"paused": False})
        logger.info("Trading RESUMED via dashboard")
        return jsonify({"ok": True, "paused": False})

    @app.route("/api/force-close", methods=["POST"])
    def force_close_all():
        broker = getattr(hub, "_broker", None)
        if not broker:
            return jsonify({"ok": False, "error": "broker not available"}), 503
        try:
            hub.trading_paused = True
            hub._broadcast("trading_control", {"paused": True})
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
            exit_mgr = getattr(hub, "_exit_manager", None)
            if exit_mgr and result.get("flat", False):
                for sym in list(exit_mgr.tracked.keys()):
                    exit_mgr.untrack(sym)
            hub._broadcast("trading_control", {
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
        q = hub.subscribe()

        def encode_sse(payload: Any) -> Optional[str]:
            try:
                return "data: {}\n\n".format(
                    json.dumps(_json_safe(payload), allow_nan=False)
                )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping non-JSON dashboard stream payload: %s", exc)
                return None

        def generate():
            try:
                # Push full state once on connect (replaces removed /api/snapshot polling).
                payload = encode_sse({"type": "snapshot", "data": hub.snapshot()})
                if payload is not None:
                    yield payload
                heartbeat_counter = 0
                while True:
                    while q:
                        msg = q.popleft()
                        payload = encode_sse(msg)
                        if payload is not None:
                            yield payload
                        heartbeat_counter = 0
                    time.sleep(0.3)
                    heartbeat_counter += 1
                    # Send keepalive comment every ~5 seconds to prevent timeout
                    if heartbeat_counter >= 16:
                        yield ": keepalive\n\n"
                        heartbeat_counter = 0
            except GeneratorExit:
                pass
            finally:
                hub.unsubscribe(q)

        return Response(generate(), mimetype="text/event-stream")

    return app


def start_dashboard(hub: DashboardHub, host: Optional[str] = None, port: int = 8080) -> None:
    """Bind privately by default; never terminate another process using the port."""
    host = host or os.environ.get("DAYTRADING_DASHBOARD_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "::1", "localhost") and not os.environ.get("DAYTRADING_DASHBOARD_TOKEN", "").strip():
        raise ValueError("A non-loopback dashboard requires DAYTRADING_DASHBOARD_TOKEN")
    app = create_app(hub)
    # Binding synchronously surfaces port conflicts before the bot starts.
    from werkzeug.serving import make_server
    server = make_server(host, port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="dashboard")
    thread.start()
    logger.info("Dashboard started on %s:%d", host, port)


# ---------------------------------------------------------------------------
# Single-page HTML dashboard (embedded — no external files needed)
# ---------------------------------------------------------------------------

def dashboard_html() -> str:
    """Read editable UI assets per request; refreshing the page needs no bot restart."""
    assets = Path(__file__).with_name("assets")
    return assets.joinpath("index.html").read_text().replace(
        "__DASHBOARD_CSS__", assets.joinpath("dashboard.css").read_text(),
    ).replace("__DASHBOARD_JS__", assets.joinpath("dashboard.js").read_text())
