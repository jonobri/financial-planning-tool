"""Local persistence of the user's inputs (auto-save).

Inputs are written to ``profiles/<name>.json`` (git-ignored) so each tool can
restore its last inputs on startup. Holds only the numbers you typed — no market
data, no results. Writes are atomic. The default profile is ``autosave`` (the
main planner); the optimiser page uses its own profile name.
"""

from __future__ import annotations

import json
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"


def _path(name: str) -> Path:
    return PROFILE_DIR / f"{name}.json"


def save(inputs: dict, name: str = "autosave") -> None:
    """Atomically write inputs to the named profile file."""
    PROFILE_DIR.mkdir(exist_ok=True)
    target = _path(name)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(inputs, indent=2, default=str))
    tmp.replace(target)


def load(name: str = "autosave") -> dict:
    """Return the saved inputs for the named profile, or {} if none/unreadable."""
    try:
        return json.loads(_path(name).read_text())
    except Exception:
        return {}


def clear(name: str = "autosave") -> None:
    """Delete the named profile file (reset to defaults)."""
    _path(name).unlink(missing_ok=True)
