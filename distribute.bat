pyinstaller cap_tools.spec
tar -acf dist\ED_Cell_Tool.zip -C dist\cap_tools cell_tool.exe _internal
@REM copy /Y dist\cap_tools.zip "C:\Users\robert.buecker\OneDrive - Rigaku Americas Holding\RESE-RAC SynED Application Labs\programs" 
@REM pyinstaller compute_z.py --onefile --distpath "C:\Users\robert.buecker\OneDrive - Rigaku Americas Holding\RESE-RAC SynED Application Labs\programs"
@REM pyinstaller cell_tool.py --onefile
@REM pyinstaller finalization_viewer.py --onefile --distpath "C:\Users\robert.buecker\OneDrive - Rigaku Americas Holding\RESE-RAC SynED Application Labs\programs"