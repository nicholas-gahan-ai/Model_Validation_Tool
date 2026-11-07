"""Build dist/validation_tool_<version>.zip: the app, the packaged model(s), scripts and docs.
Internal notes (PLAN.md), caches, virtual envs, developer-only tools and git metadata are left out.

  python tools/make_release.py [--no-models]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["app.py", "VERSION", "README.md", "requirements.txt", "requirements-optional.txt", "requirements-dev.txt",
           "setup_linux.sh", "run_linux.sh", "setup_windows.bat", "run_windows.bat",
           "core", "ui", "config", "assets", "models", "tools", "tests", "docs"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", "dist", "dev"}
EXCLUDE_NAMES = {"PLAN.md", ".DS_Store"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-models", action="store_true", help="leave models/ out (the zip is then ~1 MB)")
    a = ap.parse_args()
    version = (ROOT / "VERSION").read_text().strip()
    out = ROOT / "dist" / f"validation_tool_{version}.zip"
    out.parent.mkdir(exist_ok=True)
    top = f"validation_tool_{version}"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for item in INCLUDE:
            p = ROOT / item
            if not p.exists():
                continue
            if a.no_models and item == "models":
                continue
            files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
            for f in files:
                rel = f.relative_to(ROOT)
                if set(rel.parts) & EXCLUDE_PARTS or f.name in EXCLUDE_NAMES or f.suffix == ".pyc":
                    continue
                if rel.parts[0] == "models" and f.suffix not in (".onnx", ".pt", ".yaml"):
                    continue
                info = zipfile.ZipInfo.from_file(f, f"{top}/{rel.as_posix()}")
                if f.suffix == ".sh":
                    info.external_attr = 0o100755 << 16   # keep the scripts executable after unzip
                with open(f, "rb") as fh:
                    z.writestr(info, fh.read(), zipfile.ZIP_DEFLATED)
                n += 1
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
