# -*- mode: python ; coding: utf-8 -*-
import pathlib
from PyInstaller.utils.hooks import collect_all

block_cipher = None

hiddenimports = []
datas = []
binaries = []
for pkg in ["torch", "torchvision", "open3d", "skimage"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# include model and docs
basedir = pathlib.Path(__file__).parent
for rel in ["data/anatomy", "docs", "models"]:
    p = basedir / rel
    if p.exists():
        datas.append((str(p), rel))

a = Analysis(
    ["surgicalai/cli.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=["surgicalai/_pyinstaller_mpl.py"],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="SurgicalAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
