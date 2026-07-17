# AutoLineDigitizer

A desktop application for automatic line chart data extraction using [LineFormer](https://github.com/TheJaeLal/LineFormer) with automatic axis detection via [ChartDete](https://github.com/pengyu965/ChartDete/) and [EasyOCR](https://github.com/JaidedAI/EasyOCR).

## Demo

### Video

https://github.com/user-attachments/assets/7ecb641e-f939-40a5-ad7b-54b64937fdd4

> **Note:** This demo video shows an earlier Streamlit-based version. The current version uses a Flet-based desktop app with a different UI, but the core functionality is the same.

### Input / Output

| Input | Output |
|-------|--------|
| ![Input](demo/10.3390_nano14040384_4i.png) | ![Output](demo/10.3390_nano14040384_4i_result.png) |

### Export to Digitizer Tools

| [StarryDigitizer](https://starrydigitizer.vercel.app/) | [WebPlotDigitizer](https://apps.automeris.io/wpd4/) |
|------------------|------------------|
| ![StarryDigitizer](demo/10.3390_nano14040384_4i_sd.png) | ![WebPlotDigitizer](demo/10.3390_nano14040384_4i_wpd.png) |

## Features

- **Line Extraction**: Automatic line detection using LineFormer
- **Axis Detection**: Automatic axis label reading via ChartDete + OCR
- **PDF mode**: open a whole paper PDF — every chart figure is found and shown in a
  gallery; digitize figure by figure with per-figure review (Axes OK / Extraction OK)
- **Claude-assisted curation** (optional, needs an Anthropic API key):
  - axis **properties** (name + unit) are read from each figure automatically and are
    hand-editable, with a **✦ Ask Claude** button to re-read on demand
  - curves are **named from the legend** automatically (color/style matching fallback works offline)
  - a per-axis **KMDS verification panel** shows: what the model detected → your edited
    value → the live KMDS vocabulary match (official term ✓ · local extension ✓ · not-in-KMDS ⚠)
- **KMDS records**: a `<paper>_kmds/` folder next to the PDF is loaded automatically;
  approved digitizations merge into the record (curator-verified axis names correct the
  record's graph terms, with provenance). Vocabulary extensions live in
  `src/kmds_vocab_extensions.json`.
- **Uploads**: approved figures go to **Starrydata3** (your local KMDS database,
  [repo](https://github.com/adityaaa-IIT-BHU/starrydata3)) and/or **Starrydata2**
  (official internal API; token auth for unattended use — `tools/starrydata_upload.py`)
- **Export Formats**:
  - [StarryDigitizer](https://starrydigitizer.vercel.app/) ZIP
  - [WebPlotDigitizer](https://apps.automeris.io/wpd4/) TAR

## Download

Get everything from the [latest release](https://github.com/adityaaa-IIT-BHU/AutoLineDigitizer/releases/latest)
page — the app **and** the ML model weights are all there (from v0.7.0), so after
downloading nothing else is fetched from the internet.

| File | What it is |
|------|------------|
| `AutoLineDigitizer-macOS.zip` | The app — macOS (Apple Silicon) |
| `AutoLineDigitizer-Windows.zip` | The app — Windows |
| `iter_3000.pth` | LineFormer weights (curve extraction) |
| `checkpoint.pth` | ChartDete weights (axis detection) |

Download the zip for your platform **plus the two `.pth` files** (they cannot
ship inside the zip — GitHub caps release files at 2 GiB).

> **Note:** Intel Mac is not currently supported. Apple Silicon (M1/M2/M3/M4) only.

### Installation

#### macOS

1. Download and unzip `AutoLineDigitizer-macOS.zip`
2. Move `AutoLineDigitizer.app` to Applications
3. On first launch, macOS will warn it **"could not verify the app is free of malware"** (the app is not signed with an Apple Developer ID; it is open source and built by GitHub CI from this repository). Do ONE of:
   - Double-click the app (it will be blocked, click **Done**, not "Move to Trash") → go to **System Settings → Privacy & Security** → scroll down to *"AutoLineDigitizer was blocked…"* → click **"Open Anyway"**
   - Or in Terminal, clear the download quarantine flag: `xattr -dr com.apple.quarantine /Applications/AutoLineDigitizer.app`

   (On macOS 15 Sequoia the old right-click → Open trick no longer bypasses the check — use one of the two options above.)
4. Click **Import Models** (under the Line Model dropdown) and select the two
   downloaded `.pth` files — fully offline from here.
   (If you skipped the `.pth` downloads, the app offers to fetch them
   automatically on first launch instead — that path needs internet.)

#### Windows

1. Download and unzip `AutoLineDigitizer-Windows.zip`
2. Run `AutoLineDigitizer\AutoLineDigitizer.exe`
3. Click **Import Models** and select the two downloaded `.pth` files

### Manual Model Download (Proxy / Firewall environments)

If auto-download fails (e.g., due to a corporate proxy), you can download the models manually via your browser and import them into the app.

#### Base Models (required)

1. Download `iter_3000.pth` and `checkpoint.pth` from [GitHub Releases](https://github.com/t29mato/AutoLineDigitizer/releases/tag/models)
2. In the app, click **Import Models** button that appears under the Line Model dropdown
3. Select the downloaded `.pth` files

#### Battery Fine-tuned Models (optional)

1. Download the model file from [HuggingFace](https://huggingface.co/t29mato/lineformer-battery-finetuned/tree/main)
   - `lineformer_battery_iter_5000.pth` — Battery (iter_5000)
   - `lineformer_battery_best_iter_1300.pth` — Battery (best)
2. Select the Battery model from the **Line Model** dropdown
3. Click the **Import Model** button that appears
4. Select the downloaded `.pth` file

---

## Run from source — the full pipeline

The packaged app above covers single-image digitization. The full research
pipeline (PDF gallery, KMDS integration, Claude curation, Starrydata uploads,
and serving as [Starrydata3](https://github.com/adityaaa-IIT-BHU/starrydata3)'s
digitization engine) runs from source. Everything — including the ML weights —
lives and runs on your own machine.

```bash
git clone https://github.com/adityaaa-IIT-BHU/AutoLineDigitizer.git
cd AutoLineDigitizer

# 1. environment (Python 3.11; torch first, then mmcv-full which builds against it)
conda create -n alddev python=3.11 -y && conda activate alddev
pip install torch==2.12.0
pip install mmcv-full==1.7.2
pip install -r requirements-dev.txt

# 2. model weights (downloaded once, stored locally)
#    - launch the app once: it fetches LineFormer + ChartDete weights automatically
#    - behind a proxy: download manually (see "Manual Model Download" above)

# 3. run the desktop app
cd src && python desktop_app.py
# optional: auto-open a PDF on launch
ALD_OPEN_PDF=/path/to/paper.pdf python desktop_app.py
```

Optional pieces:

- **Claude curation** — put your Anthropic API key in the app's Settings (or
  `ANTHROPIC_API_KEY`); axis reading and legend naming then run automatically
  per figure.
- **Starrydata3** (local database + web UI + server-side digitize studio) —
  install [starrydata3](https://github.com/adityaaa-IIT-BHU/starrydata3) and run
  its server from this env: `ALD_SRC=$PWD/src uvicorn starrydata3.app:app --port 8300`.
- **Starrydata2 uploads** — save your api-token to `~/.sd2_token` (chmod 600);
  then use the app's *Upload approved → Starrydata2* button, or batch-upload:
  `python tools/starrydata_upload.py export.json --commit` (NIMS network only;
  the client routes through the NIMS proxy automatically).

## Image Attribution

Demo images are used under the following licenses:

- **demo/10.3390_nano14040384_4i.png**: Figure 4(i) from [Wang et al., Nanomaterials, 2024](https://doi.org/10.3390/nano14040384), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Components

- **LineFormer**: [ICDAR 2023 Paper](https://link.springer.com/chapter/10.1007/978-3-031-41734-4_24) by Jay Lal et al.
- **ChartDete**: MIT License, Copyright (c) 2023 Pengyu Yan
- **MMDetection**: Apache License 2.0, Copyright (c) 2018-2023 OpenMMLab
