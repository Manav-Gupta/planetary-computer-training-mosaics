# planetary-computer-training-mosaics

Download Sentinel-2 L2A imagery from the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
over a set of field polygons, sample the pixels inside those fields, and build
per-window mosaics (median or best-quality) with vegetation / soil / burn indices.

Built for the HALO Trust demining work in Ukraine (cropland recultivation
monitoring), but the pipeline is generic: give it any polygon layer and a date
range and it returns tidy per-pixel and per-field time series.

## What it produces

- **STAC inventory** — every matching Sentinel-2 scene, plus a field-to-scene map.
- **Scene samples** — one Parquet file per scene, holding the band values for
  every pixel that falls inside a field polygon (with SCL cloud/quality flags and
  Sentinel-2 baseline harmonization already applied).
- **Mosaics** — pixel-level and field-level composites over rolling date windows,
  with indices (NDVI, BSI, NDTI, NDII, NBR) and summary metrics.

## Install

```bash
conda env create -f environment.yml
conda activate halo-s2
```

Planetary Computer access is **anonymous** — no account or API key is required.
Asset URLs are signed automatically via `planetary_computer.sign_inplace`.

## Configure

Copy the template and edit it for your run:

```bash
cp configs/planetary_config_azure_template.json configs/my_config.json
```

Key fields:

| Field | Meaning |
|---|---|
| `fields_path` | Polygon layer (`.shp`, `.gpkg`, or `.geojson`). A **folder** also works — the single vector file inside is auto-detected. |
| `output_dir` | Where outputs are written. |
| `field_id_col` | Column that uniquely identifies each field. |
| `start_date` / `end_date` / `interval_days` | Date range and window size. |
| `assets` | Sentinel-2 bands to pull, e.g. `["B02","B03","B04","B08","B11","B12","SCL"]`. Must include B04/B03/B08 (red/green/NIR) — OmniCloudMask needs them. |
| `max_cloud_cover` | Scene-level STAC search filter (%) — coarse pre-filter before per-pixel cloud masking. |
| `valid_scl_classes` | **Informational only** — SCL is still recorded on every sample for reference/QA, but pixel validity is decided by OmniCloudMask, not this list. |
| `max_pixels_per_date` / `max_pixels_per_scene` | Random pixel cap per window / scene. |
| `resolution` | Metres per pixel (10). |
| `scene_spatial_batching`, `max_scene_window_pixels` | Split very large scene reads to control memory. |
| `max_workers` | Parallel scene workers for the download step. |

Paths in the committed template are Azure mount paths (`/mnt/azureml/...`); set
your own local paths in `my_config.json` for local runs.

## Run locally

Three steps, sharing one config:

```bash
# 1. Query STAC and cache the scene inventory + field/scene map
python scripts/download_s2_pc.py --config configs/my_config.json --stac-inventory-only

# 2. Sample pixels inside fields, one Parquet per scene
python scripts/download_s2_pc.py --config configs/my_config.json --scene-samples

# 3. Build mosaics from the scene samples
python scripts/mosaic_scene_samples.py \
  --input-dir <output_dir>/scene_samples_batched \
  --output-dir <output_dir>/mosaics \
  --start-date 2023-01-01 --end-date 2026-06-10 \
  --window-days 14 --method median
```

Or run all three through the orchestrator:

```bash
python scripts/run_azure_pipeline.py --config configs/my_config.json \
  --steps inventory download mosaic --mosaic-window-days 14 --mosaic-method median
```

> Without `--stac-inventory-only` / `--scene-samples`, `download_s2_pc.py` runs a
> simpler mode: a **median composite per date window**, written as one CSV per window.

## Run on Azure ML

`azureml/halo_s2_pipeline_job.yml` submits the full pipeline as a command job.
Three values are workspace-specific and marked `# CHANGE ME` in the file — the
compute cluster and the two datastore paths. Override them at submit time without
editing the file:

```bash
az ml job create -f azureml/halo_s2_pipeline_job.yml \
  --set compute=azureml:<your-cluster> \
  --set inputs.fields.path=azureml://datastores/<datastore>/paths/<fields-folder>/ \
  --set outputs.samples.path=azureml://datastores/<datastore>/paths/<output-folder>/
```

The `fields` input is a `uri_folder`; upload the shapefile **and its sidecar files**
(`.shx`, `.dbf`, `.prj`) into that folder. The pipeline resolves the folder to the
single vector file inside it, so you don't pass the `.shp` name explicitly.

### Custom environment (`sentinels_poly_timeseries_extract`)

The job points at a named Azure ML environment, `azureml:sentinels_poly_timeseries_extract@latest`,
defined in `azureml/environment/` instead of installing `environment.yml`
fresh on every job run. It bakes the `halo-s2` conda env (including
`omnicloudmask` + CPU `torch`) into a Docker image at build time.

Register it once (Azure ML builds and pushes the image itself, via ACR
Tasks against the workspace's linked registry — no separate `docker build`/
`docker push` needed):

```bash
az ml environment create -f azureml/environment/environment.yml \
  --resource-group <your-resource-group> \
  --workspace-name <your-workspace>
```

If you'd rather build and push the image locally instead, build from the
repo root using `azureml/environment/Dockerfile`, push it to your ACR, then
set `image: <acr>.azurecr.io/sentinels_poly_timeseries_extract:<tag>` in
`azureml/environment/environment.yml` in place of the `build:` block.

`cluster-rise-d16` (and the placeholder `cpu-cluster` in the job file) are
CPU-only, so OmniCloudMask runs on CPU there today. GPU support is already
wired in (`scripts/cloud_mask.py` auto-detects `cuda`/`mps`/`cpu`) — point
`compute:` at a GPU cluster later and it's used automatically, no code
changes needed.

## Notes

- **Sentinel-2 baseline harmonization** is applied automatically: scenes with
  processing baseline ≥ 04.00 have the +1000 DN offset removed so pre- and
  post-2022-01-25 reflectance are on the same scale.
- **Cloud/shadow masking** uses [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)
  (DPIRD-DMA), run once per loaded scene window on the red/green/NIR bands.
  A pixel is valid (`valid_px`) only if OmniCloudMask classifies it as clear
  (thick cloud, thin cloud, and shadow are all masked out — strict
  clear-sky). SCL is still recorded on every sample for reference but no
  longer decides validity. See `METHODS.md` for the full rationale and
  parameter choices.
- **Not included in the repo:** imagery, sample Parquet, and field data are
  git-ignored (see `.gitignore`). You supply your own polygon layer.

## Layout

```
scripts/
  download_s2_pc.py         # STAC search + pixel sampling (+ composite mode)
  cloud_mask.py              # OmniCloudMask GPU cloud/shadow masking
  mosaic_scene_samples.py   # mosaics + indices from scene samples
  append_mosaic_timeline.py # append later date ranges to an existing mosaic timeline
  run_azure_pipeline.py     # orchestrates inventory -> download -> mosaic
azureml/
  halo_s2_pipeline_job.yml  # Azure ML command job
  environment/              # custom AML environment (Dockerfile + env spec)
configs/
  planetary_config_azure_template.json  # config template
environment.yml
METHODS.md                  # log of methodological choices (cloud masking, etc.)
```
