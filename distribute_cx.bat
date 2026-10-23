@echo off
SETLOCAL
pushd "%~dp0"

powershell -ExecutionPolicy Bypass -File "%~dp0with-cap-tools.ps1" python "%~dp0setup_cx.py" build_exe
set BUILD_RESULT=%ERRORLEVEL%
popd

ENDLOCAL & EXIT /B %BUILD_RESULT%
