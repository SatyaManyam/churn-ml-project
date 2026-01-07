from __future__ import annotations
import json
from pathlib import Path

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
