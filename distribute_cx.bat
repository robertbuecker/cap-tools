@echo off
SETLOCAL

CALL conda activate cap-tools-pip
IF ERRORLEVEL 1 EXIT /B %ERRORLEVEL%

for /f "delims=" %%V in ('python setup_cx.py --print-version-tag') do set VERSION=%%V
IF "%VERSION%"=="" EXIT /B 1
echo %VERSION%

IF EXIST dist-cx\cap_tools (
    rmdir /S /Q dist-cx\cap_tools
)

python setup_cx.py build_exe
IF ERRORLEVEL 1 EXIT /B %ERRORLEVEL%

set ZIPNAME=.\dist-cx\ED_Cell_Tool_%VERSION%_cx.zip
echo %ZIPNAME%
IF EXIST %ZIPNAME% (
    del %ZIPNAME%
)
"C:\Program Files\7-Zip\7z.exe" a -tzip %ZIPNAME% .\dist-cx\cap_tools\*

ENDLOCAL
