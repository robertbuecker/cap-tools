pyinstaller cap_tools.spec
tar -acf dist\cap_tools.zip -C dist cap_tools  
copy /Y dist\cap_tools.zip "C:\Users\robert.buecker\OneDrive - Rigaku Americas Holding\RESE-RAC SynED Application Labs\programs" 
pyinstaller compute_z.py --onefile --distpath "C:\Users\robert.buecker\OneDrive - Rigaku Americas Holding\RESE-RAC SynED Application Labs\programs"
pyinstaller cell_tool.py --onefile --distpath "C:\Users\robert.buecker\OneDrive - Rigaku Americas Holding\RESE-RAC SynED Application Labs\programs"
pyinstaller finalization_viewer.py --onefile --distpath "C:\Users\robert.buecker\OneDrive - Rigaku Americas Holding\RESE-RAC SynED Application Labs\programs"