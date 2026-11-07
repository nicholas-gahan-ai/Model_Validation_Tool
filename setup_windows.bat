@echo off
rem Gahan AI Validation Viewer - Windows setup: creates .venv and installs dependencies.
rem   setup_windows.bat                    GPU build if an NVIDIA GPU is found (nvidia-smi), else CPU build
rem   setup_windows.bat --cpu              force the CPU build
rem   setup_windows.bat --with-ultralytics also install Ultralytics (.pt models; large)
setlocal enabledelayedexpansion
cd /d "%~dp0"
set FORCE_CPU=0
set ULTRA=0
for %%A in (%*) do (
  if /I "%%A"=="--cpu" set FORCE_CPU=1
  if /I "%%A"=="--with-ultralytics" set ULTRA=1
)
set PY=py -3
%PY% -c "import sys" >nul 2>&1 || set PY=python
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" || (
  echo Python 3.9 or newer is required: https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo == creating .venv
  %PY% -m venv .venv || exit /b 1
)
set VPY=.venv\Scripts\python.exe
%VPY% -m pip install --upgrade pip wheel || exit /b 1
%VPY% -m pip install -r requirements.txt || exit /b 1
if "%ULTRA%"=="1" (
  echo == installing Ultralytics ^(optional^)
  %VPY% -m pip install -r requirements-optional.txt || exit /b 1
)
set GPU=0
if "%FORCE_CPU%"=="0" (
  where nvidia-smi >nul 2>&1 && nvidia-smi -L >nul 2>&1 && set GPU=1
)
if "%GPU%"=="1" (
  echo == NVIDIA GPU found: installing onnxruntime-gpu with the CUDA 12 / cuDNN 9 runtime wheels
  %VPY% -m pip show onnxruntime >nul 2>&1 && ( %VPY% -m pip uninstall -y onnxruntime & set FIX=onnxruntime-gpu^>=1.20 )
  %VPY% -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.20" || exit /b 1
) else (
  echo == no NVIDIA GPU ^(or --cpu^): installing the CPU build of onnxruntime
  %VPY% -m pip show onnxruntime-gpu >nul 2>&1 && ( %VPY% -m pip uninstall -y onnxruntime-gpu & set FIX=onnxruntime^>=1.20 )
  %VPY% -m pip install "onnxruntime>=1.20" || exit /b 1
)
if defined FIX %VPY% -m pip install --force-reinstall --no-deps "%FIX%"
echo == checking the install
%VPY% -c "import onnxruntime as o; getattr(o,'preload_dlls',lambda:None)(); print('onnxruntime', o.__version__, 'providers:', o.get_available_providers())" || exit /b 1
%VPY% -c "import PySide6, cv2, onnx; print('PySide6 / OpenCV / onnx OK')" || exit /b 1
rem desktop shortcut with the company icon
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Validation Viewer.lnk'); $s.TargetPath='%~dp0run_windows.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0assets\logo_icon.ico'; $s.WindowStyle=7; $s.Save()" >nul 2>&1
echo == done. Start with run_windows.bat ^(or the "Validation Viewer" shortcut on the desktop^)
endlocal
