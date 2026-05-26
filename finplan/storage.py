"""Local persistence of the user's plan inputs (auto-save).

Inputs are written to ``profiles/autosave.json`` (git-ignored) so the app can
restore your last plan on startup. This holds only the numbers you typed into
the sidebar — no market data, no results. Writes are atomic.
"""

from __future__ import annotations

import json
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"
AUTOSAVE = PROFILE_DIR / "autosave.json"


def save(inputs: dict) -> None:
    """Atomically write the plan inputs to the autosave file."""
    PROFILE_DIR.mkdir(exist_ok=True)
    tmp = AUTOSAVE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(inputs, indent=2, default=str))
    tmp.replace(AUTOSAVE)


def load() -> dict:
    """Return the saved inputs, or an empty dict if none/unreadable."""
    try:
        return json.loads(AUTOSAVE.read_text())
    except Exception:
        return {}


def clear() -> None:
    """Delete the autosave file (reset to defaults)."""
    AUTOSAVE.unlink(missing_ok=True)
