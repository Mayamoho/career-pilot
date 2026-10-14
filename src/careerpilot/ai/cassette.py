"""Recorded provider responses.

Tests replay these instead of calling a model, which buys three things: zero cost,
byte-identical results on every run, and a regression test that fails the moment a
prompt change makes the model answer differently. The key is a hash over the task,
system prompt, user payload *and* output schema, so editing any of them invalidates
the recording rather than silently passing against stale output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from careerpilot.config import FIXTURES_DIR


class CassetteStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or (FIXTURES_DIR / "cassettes")
        self._memory: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._loaded:
            self._loaded = True
            if self.directory.exists():
                for file in sorted(self.directory.glob("*.json")):
                    try:
                        payload = json.loads(file.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict) and "key" in payload:
                        self._memory[payload["key"]] = {**payload, "_file": file}
        return self._memory

    def get(self, key: str) -> dict[str, Any] | None:
        return self._load().get(key)

    def put(self, key: str, *, text: str, model: str, task: str) -> None:
        store = self._load()
        entry = store.get(key)
        if entry is not None and entry.get("text") == text:
            return
        recorded = entry.get("_file") if entry else None
        path = Path(recorded) if recorded is not None else self.directory / f"{task}__{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"key": key, "task": task, "model": model, "text": text}, indent=2),
            encoding="utf-8",
        )
        self._memory[key] = {"key": key, "task": task, "model": model, "text": text, "_file": path}

    def keys_for_task(self, task: str) -> list[str]:
        return [k for k, v in self._load().items() if v.get("task") == task]

    def __len__(self) -> int:
        return len(self._load())
