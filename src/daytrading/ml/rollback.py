"""Explicitly restore an archived model, validating its schema before replacement."""

import argparse
import hashlib
import json
from pathlib import Path

from daytrading.ml.artifacts import archive_model, atomic_json, data_root

MODELS = {"entry": "entry_model.json", "ignition": "warrior_ignition_model.json"}


def restore(model: str, version: str) -> Path:
    active = data_root() / "models" / MODELS[model]
    if Path(version).name != version:
        raise ValueError(
            "choose a filename from this project's models/versions directory"
        )
    saved = active.parent / "versions" / version
    if (
        not saved.is_file()
        or saved.parent.resolve() != (active.parent / "versions").resolve()
    ):
        raise ValueError("archived model version not found")
    payload = json.loads(saved.read_text())
    metadata = None
    if model == "ignition":
        from daytrading.strategy.warrior_ignition import IgnitionModel

        IgnitionModel.load(str(saved))
    else:
        import xgboost as xgb

        from daytrading.ml.features import FEATURE_NAMES

        metadata = json.loads(saved.with_suffix(".meta.json").read_text())
        digest = hashlib.sha256(saved.read_bytes()).hexdigest()
        if (
            not metadata.get("promotion_allowed")
            or metadata.get("model_sha256") != digest
        ):
            raise ValueError(
                "archived entry model validation metadata is missing or mismatched"
            )
        checker = xgb.Booster()
        checker.load_model(str(saved))
        if checker.num_features() != len(FEATURE_NAMES):
            raise ValueError("archived model feature schema mismatch")
    if active.exists():
        archive_model(active)
    atomic_json(active, payload)
    if metadata is not None:
        metadata["model_sha256"] = hashlib.sha256(active.read_bytes()).hexdigest()
        atomic_json(active.with_suffix(".meta.json"), metadata)
    return active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--version", help="archived filename; omit to list versions")
    args = parser.parse_args()
    if args.version:
        print(f"Restored {restore(args.model, args.version)}")
    else:
        root = data_root() / "models" / "versions"
        for path in sorted(root.glob(f"{Path(MODELS[args.model]).stem}-*.json")):
            if not path.name.endswith(".meta.json"):
                print(path.name)


if __name__ == "__main__":
    main()
