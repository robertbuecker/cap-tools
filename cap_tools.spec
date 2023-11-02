# -*- mode: python ; coding: utf-8 -*-


cell_a = Analysis(
    ['cell_tool.py'],
    pathex=[],
    binaries=[],
    datas=[('cap_tools/spglib.yaml', 'cap_tools')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

zscore_a = Analysis(
    ['compute_z.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

MERGE( (cell_a, 'cell_tool', 'cell_tool'), (zscore_a, 'compute_z', 'compute_z') )

cell_pyz = PYZ(cell_a.pure)
zscore_pyz = PYZ(zscore_a.pure)

cell_exe = EXE(
    cell_pyz,
    cell_a.scripts,
    [],
    exclude_binaries=True,
    name='cell_tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

zscore_exe = EXE(
    zscore_pyz,
    zscore_a.scripts,
    [],
    exclude_binaries=True,
    name='compute_z',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    cell_exe,
    cell_a.binaries,
    cell_a.datas,
    zscore_exe,
    zscore_a.binaries,
    zscore_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cap_tools',
)