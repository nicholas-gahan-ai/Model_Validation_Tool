import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def bundle():
    p = os.environ.get("GAHAN_BUNDLE")
    if not p or not Path(p).expanduser().is_dir():
        pytest.skip("GAHAN_BUNDLE not set")
    return Path(p).expanduser()


@pytest.fixture
def model_path():
    p = ROOT / "models/open_adas_v13s_single.onnx"
    if not p.is_file():
        pytest.skip("packaged model not built (tools/package_model.py)")
    return p
