# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

packages = ['matplotlib', 'open3d', 'torch', 'torchvision']
datas, binaries, hiddenimports = [], [], []
for pkg in packages:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ['surgicalai/cli.py'],
    pathex=[],
    binaries=binaries,
    datas=datas + [('surgicalai/assets', 'surgicalai/assets')],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=['surgicalai/_pyinstaller_mpl.py'],
    excludes=['tests'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(pyz, a.scripts, a.binaries, name='SurgicalAI', console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='SurgicalAI')
