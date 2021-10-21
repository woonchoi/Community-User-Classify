# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
added_files = [('Classify.ui','.')]

a = Analysis(['gui.py'],
             pathex=['C:\\Users\\Stdio\\Desktop\\capstone\\ClassifyUserWithText'],
             binaries=[],
             datas=added_files,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
to_remove = ["_AES", "_ARC4", "_DES", "_DES3", "_SHA256", "_counter"]
for b in a.binaries:
    found = any(
        f'{crypto}.cp37-win_amd64.pyd' in b[1]
        for crypto in to_remove
    )
    if found:
        print(f"Removing {b[1]}")
        a.binaries.remove(b)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False )
