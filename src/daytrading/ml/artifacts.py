"""Persistent, atomic model storage with immutable versions and rollback."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def data_root() -> Path:
    project = Path(__file__).resolve().parents[3]
    default = (
        project / "data"
        if (project / "pyproject.toml").exists()
        else Path.cwd() / "data"
    )
    return Path(os.getenv("DAYTRADING_DATA_DIR", str(default)))


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".model-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def archive_model(path: Path) -> Path:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    archived = path.parent / "versions" / f"{path.stem}-{digest[:16]}.json"
    archived.parent.mkdir(parents=True, exist_ok=True)
    if not archived.exists():
        shutil.copyfile(path, archived)
        metadata = path.with_suffix(".meta.json")
        if metadata.exists():
            shutil.copyfile(metadata, archived.with_suffix(".meta.json"))
    return archived


def promote_json(path: Path, value: dict) -> Path:
    path = Path(path)
    if path.exists():
        archive_model(path)
    # Validation is performed by the trainer; retain the evaluated weights.
    atomic_json(path, value)
    return archive_model(path)
