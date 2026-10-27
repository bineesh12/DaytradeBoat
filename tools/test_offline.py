"""Run published deterministic tests without credentials, network or live data."""
from pathlib import Path
import os
import sys
import tempfile

from offline_guard import install


def main():
    root = Path(__file__).resolve().parents[1]
    for key in tuple(os.environ):
        if key.upper().startswith("DAYTRADING_"):
            os.environ.pop(key, None)
    install()
    os.environ["DAYTRADING_CONFIG_LOADED"] = "true"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / "src"), str(root / "tests"), str(root / "launch-support")]
    excluded = [
        "--ignore=" + str(root / "tests/test_micro_confirmation_context.py"),
        "--ignore=" + str(root / "tests/test_final_entry_context.py"),
        "--ignore=" + str(root / "tests/test_verifier_session_context.py"),
        "--ignore=" + str(root / "tests/test_micro_wick_routing.py"),
        # These original research tests depend on the separate studies project.
        "--ignore=" + str(root / "tests/test_additive_micro_support.py"),
        "--ignore=" + str(root / "tests/test_breakout_partial_research.py"),
        "--ignore=" + str(root / "tests/test_micro_support_study.py"),
        "--ignore=" + str(root / "tests/test_research_boundary.py"),
    ]
    import pytest

    private_cases = {
        ("test_hod_pending_entry.py", "test_weto_recovered_but_real_shared_guard_still_blocks_at_79"),
        ("test_hod_pending_entry.py", "test_valid_retained_hod_routes_through_verifier_risk_and_timer_without_new_scanner_hit"),
        ("test_pending_entry.py", "test_full_pipeline_retries_sprc_without_new_detector_hit"),
    }

    class PrivateFixtureSelection:
        def pytest_collection_modifyitems(self, config, items):
            selected, unavailable = [], []
            for item in items:
                identity = (item.path.name, item.originalname or item.name)
                (unavailable if identity in private_cases else selected).append(item)
            items[:] = selected
            config.hook.pytest_deselected(items=unavailable)

    with tempfile.TemporaryDirectory(prefix="daytradeboat-offline-") as temporary:
        os.chdir(temporary)
        os.environ["DAYTRADING_JOURNAL_DIR"] = str(Path(temporary) / "journal")
        return pytest.main([
            "-q", "-p", "no:cacheprovider", "-c", str(root / "pyproject.toml"),
            "--basetemp=" + str(Path(temporary) / "pytest"),
            *excluded, *sys.argv[1:], str(root / "tests"),
        ], plugins=[PrivateFixtureSelection()])


if __name__ == "__main__":
    raise SystemExit(main())
