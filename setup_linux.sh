#!/usr/bin/env bash
# Gahan AI Validation Viewer - Linux setup: creates .venv, installs dependencies, adds a desktop launcher.
#   ./setup_linux.sh                    GPU build if an NVIDIA GPU is found (nvidia-smi), else CPU build
#   ./setup_linux.sh --cpu              force the CPU build
#   ./setup_linux.sh --with-ultralytics also install Ultralytics (.pt models; large)
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
PY="${PYTHON:-python3}"
FORCE_CPU=0; ULTRA=0
for a in "$@"; do
  case "$a" in
    --cpu) FORCE_CPU=1 ;;
    --with-ultralytics) ULTRA=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || { echo "Python 3.9+ is required"; exit 1; }
if [ ! -d .venv ]; then
  echo "== creating .venv"
  "$PY" -m venv .venv || { echo "python3-venv is missing: sudo apt install python3-venv"; exit 1; }
fi
. .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
if [ "$ULTRA" = 1 ]; then
  echo "== installing Ultralytics (optional; before onnxruntime so the right ONNX Runtime build ends up installed)"
  python -m pip install -r requirements-optional.txt
fi

if [ "$FORCE_CPU" = 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "== NVIDIA GPU found: installing onnxruntime-gpu with the CUDA 12 / cuDNN 9 runtime wheels"
  if python -m pip show onnxruntime >/dev/null 2>&1; then   # the CPU build shares the package folder
    python -m pip uninstall -y onnxruntime; FIX="onnxruntime-gpu>=1.20"
  fi
  python -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.20"
else
  echo "== no NVIDIA GPU (or --cpu): installing the CPU build of onnxruntime"
  if python -m pip show onnxruntime-gpu >/dev/null 2>&1; then
    python -m pip uninstall -y onnxruntime-gpu; FIX="onnxruntime>=1.20"
  fi
  python -m pip install "onnxruntime>=1.20"
fi
if [ -n "${FIX:-}" ]; then python -m pip install --force-reinstall --no-deps "$FIX"; fi

echo "== checking the install"
python - <<'PY'
import onnxruntime as ort
try:
    ort.preload_dlls()
except Exception:
    pass
print("onnxruntime", ort.__version__, "providers:", ", ".join(ort.get_available_providers()))
import PySide6, cv2, onnx  # noqa: F401
print("PySide6 / OpenCV / onnx OK")
PY

# desktop launcher (application menu)
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"
cat > "$APPS/gahan-validation-viewer.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Validation Viewer
GenericName=Model validation viewer
Comment=Gahan AI Pvt. Ltd. - run perception models on video, images, cameras and RTSP
Exec="$HERE/run_linux.sh"
Icon=$HERE/assets/logo_icon.png
Terminal=false
Categories=Development;Science;
StartupWMClass=gahan-validation-viewer
DESK
chmod +x run_linux.sh "$APPS/gahan-validation-viewer.desktop"
echo "== done. Start with ./run_linux.sh (or 'Validation Viewer' in the application menu)"
