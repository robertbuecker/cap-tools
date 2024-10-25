@echo off
SETLOCAL
@REM pyinstaller cell_tool.py -y --windowed --icon cell_tool_icon.ico --add-binary cell_tool_icon.ico:.
pyinstaller cap_tools.spec -y
git describe > version.txt
set /p VERSION=<version.txt
echo %VERSION%
set ZIPNAME=.\dist\ED_Cell_Tool_%VERSION%.zip
echo %ZIPNAME%
IF EXIST %ZIPNAME% (
    del %ZIPNAME%
)
"C:\Program Files\7-Zip\7z.exe" a -tzip %ZIPNAME% .\dist\cap_tools\*
ENDLOCAL