"""App-level settings that are not per-user UI state."""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent


def app_version() -> str:
    try:
        return (APP_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "0.0.0"


def bundle_path(required: bool = True) -> Path | None:
    """Location of the validation data bundle, from the GAHAN_BUNDLE environment variable.
    Only developer tools and tests need it; the app itself does not."""
    p = os.environ.get("GAHAN_BUNDLE")
    if not p:
        if required:
            raise RuntimeError("Set GAHAN_BUNDLE to the validation bundle folder (e.g. export GAHAN_BUNDLE=~/validation_tool_bundle)")
        return None
    path = Path(p).expanduser()
    if required and not path.is_dir():
        raise RuntimeError(f"GAHAN_BUNDLE={p} is not a folder")
    return path
