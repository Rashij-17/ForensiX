# Dataset: DeepSafe benchmark

ForensiX is designed around a single multimodal dataset so the same methodology applies to all three media types.

| Modality | Real | Fake | Total |
|----------|------|------|-------|
| Image | 2,000 | 2,000 | 4,000 |
| Audio | 1,000 | 1,000 | 2,000 |
| Video | 100 | 100 | 200 |

Source: https://huggingface.co/datasets/siddharthksah/DeepSafe-benchmark (companion code: https://github.com/siddharthksah/DeepSafe).

> **Licence: confirm it yourself.** The synopsis requires recording the dataset's source and licence in the final report.
> The licence terms of the dataset were **not verified** while building this repository. Read the dataset card, comply
> with it, and write it into your report.

## Getting it

```bash
pip install huggingface_hub
make dataset                          # download + arrange under data/raw/
python scripts/download_dataset.py --src /path/to/already/downloaded   # or arrange a local copy
```

## Expected layout

`forensix.data.scan_raw` reads labels and sources from folder names:

```
data/raw/
├── image/{real,fake}/<source or generator>/*.png|jpg
├── audio/{real,fake}/<source>/*.wav|mp3|flac
└── video/{real,fake}/<source>/*.mp4|avi
```

A folder whose name contains `real / authentic / bonafide / genuine` (or `fake / spoof / synthetic / deepfake / generated`)
sets the label; the folder directly below it is the **source / generator** used by the generator-wise analysis.

If the dataset does not encode labels in folder names, write `data/raw/metadata.csv`:

```csv
path,label,source
image/a.png,fake,midjourney_v6
image/b.jpg,real,camera
```

(`path` relative to `data/raw`; `label` is `real`/`fake` or `0`/`1`.) The file overrides folder inference.

## Verify after arranging

`make ingest` prints, per modality, the file count, status (ok / invalid / duplicate) and split sizes. Compare against the
table above. The unified metadata table is `data/processed/metadata.csv` with one row per file: `sample_id, modality,
label, source, file_format, width, height, duration_s, fps, md5, status, status_reason, dup_group, split,
preprocessing_version`.

## Caveat about `scripts/download_dataset.py`

The exact structure of the HuggingFace repository could not be inspected when the script was written (no network access
to it from the build environment). It handles both plain media folders and parquet files that contain a media-bytes column
plus a label (and optional source) column; this logic was tested on locally built examples of both layouts, **not on the
real dataset**. Read the summary the script prints, and fall back to `metadata.csv` if anything looks off.

## Demo data

`make demo-data` writes **synthetic** files (clearly marked by `data/raw/DEMO_DATA_README.txt`) into `data/raw`.
Delete that folder before using the real dataset. While the marker file exists the app shows a *Demo mode* banner.
