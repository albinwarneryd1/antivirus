from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import joblib


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    p = ensure_parent_dir(path)
    p.write_text(json.dumps(data, indent=indent), encoding="utf-8")


def load_model_and_schema(model_path: str | Path, schema_path: str | Path) -> Tuple[Any, Any]:
    model = joblib.load(str(model_path))
    schema = joblib.load(str(schema_path))
    return model, schema


def require_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return p
