@echo off
SETLOCAL
pyinstaller cell_tool.py -y --windowed --icon cell_tool_icon.ico --add-binary cell_tool_icon.ico:.
git describe > version.txt
set /p VERSION=<version.txt
echo %VERSION%
set ZIPNAME=.\dist\ED_Cell_Tool_%VERSION%.zip
echo %ZIPNAME%
IF EXIST %ZIPNAME% (
    del %ZIPNAME%
)
"C:\Program Files\7-Zip\7z.exe" a -tzip %ZIPNAME% .\dist\cell_tool\*
ENDLOCAL