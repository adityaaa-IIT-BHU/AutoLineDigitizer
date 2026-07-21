# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import certifi
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# The MinerU figure detector needs `import transformers` to WORK at runtime —
# that means the whole stack (metadata, native tokenizers/safetensors libs)
# must ship, not just the python submodules. collect_all grabs modules + data
# + dynamic libs + dist metadata; without this the frozen app silently falls
# back to the worse figure detector.
_ml_datas, _ml_bins, _ml_hidden = [], [], []
for _pkg in ("transformers", "tokenizers", "safetensors", "huggingface_hub"):
    try:
        _d, _b, _h = collect_all(_pkg)
        _ml_datas += _d
        _ml_bins += _b
        _ml_hidden += _h
    except Exception:
        pass

# Get the directory containing this spec file
spec_dir = os.path.dirname(os.path.abspath(SPEC))

# Collect all distutils submodules (including setuptools._distutils)
distutils_imports = collect_submodules('distutils') + collect_submodules('setuptools._distutils')

# Collect all mmcv submodules to ensure runner, parallel, cnn, etc. are included
mmcv_imports = collect_submodules('mmcv')

a = Analysis(
    ['src/desktop_app.py'],
    pathex=[
        spec_dir,
        os.path.join(spec_dir, 'submodules', 'chartdete'),
        os.path.join(spec_dir, 'submodules', 'lineformer'),
        os.path.join(spec_dir, 'submodules', 'lineformer', 'mmdetection'),
        os.path.join(spec_dir, 'src'),
    ],
    binaries=_ml_bins,
    datas=_ml_datas + [
        # Starrydata branding (loaded from SCRIPT_DIR/assets at runtime)
        ('assets', 'assets'),
        # SSL certificates for HTTPS downloads
        (certifi.where(), 'certifi'),
        # Config files
        ('config', 'config'),
        # LineFormer submodule (config files and line_utils)
        ('submodules/lineformer/lineformer_swin_t_config.py', 'submodules/lineformer'),
        ('submodules/lineformer/line_utils.py', 'submodules/lineformer'),
        ('submodules/lineformer/infer.py', 'submodules/lineformer'),
        # LineFormer mmdetection
        ('submodules/lineformer/mmdetection/mmdet', 'submodules/lineformer/mmdetection/mmdet'),
        # ChartDete submodule
        ('submodules/chartdete/mmdet', 'submodules/chartdete/mmdet'),
        ('submodules/chartdete/configs', 'submodules/chartdete/configs'),
        # src module
        ('src/chartdete_infer.py', 'src'),
        # KMDS prompt + schema (loaded at runtime relative to SCRIPT_DIR/_MEIPASS)
        ('src/extraction_prompt.md', '.'),
        ('src/kmds_v15.2.4_nullable.json', '.'),
        # KMDS vocabulary extensions (kmds_vocab.py loads it next to the schema)
        ('src/kmds_vocab_extensions.json', '.'),
        # Starrydata2 uploader (loaded dynamically from SCRIPT_DIR/tools)
        ('tools/starrydata_upload.py', 'tools'),
        # Paper-record HTML viewer template
        ('kmds_paper_viewer_claude.html', '.'),
        # EasyOCR models (bundled for offline use)
        (os.path.join(spec_dir, 'easyocr_models'), 'easyocr_models'),
        # MinerU PP-DocLayoutV2 weights (default PDF figure detector).
        # CI downloads these before building (see release.yml); a missing dir
        # fails the build loudly rather than shipping without the detector.
        (os.path.join(spec_dir, 'pp_doclayoutv2_weights'), 'pp_doclayoutv2_weights'),
    ],
    hiddenimports=[
        'mmdet',
        'mmdet.models',
        'mmdet.models.roi_heads',
        'mmdet.models.roi_heads.cascade_roi_head_LGF',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'requests',   # starrydata_upload.py is loaded dynamically (tools/)
        'easyocr',
        'PIL',
        'skimage',
        'scipy',
        'bresenham',
        'terminaltables',
        'matplotlib',
        'pycocotools',
        'kmds_editor',
        'jsonschema',
        'app_settings',
    ] + distutils_imports + mmcv_imports + collect_submodules('mineru_layout')
      + _ml_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoLineDigitizer',
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoLineDigitizer',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='AutoLineDigitizer.app',
        icon=None,
        bundle_identifier='com.lineformer.autolinedigitizer',
    )
