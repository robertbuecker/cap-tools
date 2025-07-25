# -*- mode: python ; coding: utf-8 -*-

cell_a = Analysis(
    ['cell_tool.py'],
    pathex=[],
    binaries=[('cell_tool_icon.ico', '.')],
    datas=[('version.txt', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# zscore_a = Analysis(
#     ['compute_z.py'],
#     pathex=[],
#     binaries=[],
#     datas=[('version.txt', '.')],
#     hiddenimports=[],
#     hookspath=[],
#     hooksconfig={},
#     runtime_hooks=[],
#     excludes=[],
#     noarchive=False,
# )

# finalization_a = Analysis(
#     ['finalization_viewer.py'],
#     ...
# )

calibrate_dd_a = Analysis(
    ['calibrate_dd.py'],
    pathex=[],
    binaries=[('calibrate_dd_icon.ico', '.')],
    datas=[('version.txt', '.')],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    hiddenimports=['matplotlib.backends.backend_pdf']
)

generate_learning_set_a = Analysis(
    ['generate_learning_set.py'],
    pathex=[],
    binaries=[],
    datas=[('version.txt', '.')],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    hiddenimports=[]
)

MERGE(
    (cell_a, 'cell_tool', 'cell_tool'),
    # (zscore_a, 'compute_z', 'compute_z'),
    (calibrate_dd_a, 'calibrate_dd', 'calibrate_dd'),
    (generate_learning_set_a, 'generate_learning_set', 'generate_learning_set')
    # (finalization_a, 'finalization_viewer', 'finalization_viewer')
)

cell_pyz = PYZ(cell_a.pure)
# zscore_pyz = PYZ(zscore_a.pure)
# finalization_pyz = PYZ(finalization_a.pure)
calibrate_dd_pyz = PYZ(calibrate_dd_a.pure)
generate_learning_set_pyz = PYZ(generate_learning_set_a.pure)

cell_exe = EXE(
    cell_pyz,
    cell_a.scripts,
    [],
    exclude_binaries=True,
    name=f'cell_tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['cell_tool_icon.ico'],
)

# zscore_exe = EXE(
#     zscore_pyz,
#     zscore_a.scripts,
#     [],
#     exclude_binaries=True,
#     name='compute_z',
#     debug=False,
#     bootloader_ignore_signals=False,
#     strip=False,
#     upx=True,
#     console=False,
#     disable_windowed_traceback=False,
#     argv_emulation=False,
#     target_arch=None,
#     codesign_identity=None,
#     entitlements_file=None,
# )

calibrate_dd_exe = EXE(
    calibrate_dd_pyz,
    calibrate_dd_a.scripts,
    [],
    exclude_binaries=True,
    name='calibrate_dd',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['calibrate_dd_icon.ico']
)

generate_learning_set_exe = EXE(
    generate_learning_set_pyz,
    generate_learning_set_a.scripts,
    [],
    exclude_binaries=True,
    name='generate_learning_set',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)

coll = COLLECT(
    cell_exe,
    cell_a.binaries,
    cell_a.datas,
    # zscore_exe,
    # zscore_a.binaries,
    # zscore_a.datas,
    # finalization_exe,
    # finalization_a.binaries,
    # finalization_a.datas,
    calibrate_dd_exe,
    calibrate_dd_a.binaries,
    calibrate_dd_a.datas,
    generate_learning_set_exe,
    generate_learning_set_a.binaries,
    generate_learning_set_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cap_tools',
)