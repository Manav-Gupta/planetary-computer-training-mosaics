# Methods log

Running record of methodological choices made in this repo, in the order
they were decided. Each entry: what was decided, why, and what alternatives
were considered.

## Cloud masking (branch `fix/addCloudMasking`)

Context: the existing pipeline masked clouds using the Sentinel-2 L2A
Scene Classification Layer (SCL), keeping pixels whose SCL class was in
`valid_scl_classes` (config default `[4, 5, 6, 11]` = vegetation / bare
soil / water / snow). This is fast (no extra compute) but SCL is known to
miss thin cloud and mis-classify cloud shadow in some scenes.

Goal: add a GPU-capable deep-learning cloud/shadow mask using
[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) (DPIRD-DMA),
run on the Red/Green/NIR bands.

- 2026-08-20: Repo scaffolding for this work — new branch
  `fix/addCloudMasking`, `.claude/` git-ignored, `CLAUDE.md` project rules
  added, this file created.

- 2026-08-20: **SCL replaced, not supplemented.** OmniCloudMask (OCM) is
  now the sole source of pixel validity (`valid_px`). SCL is still recorded
  on every scene sample for reference/QA, and `max_cloud_cover` (STAC
  scene-level search filter) is unchanged, but `valid_scl_classes` no
  longer gates per-pixel validity — it's kept in the config as an
  informational field only. Chosen over a hybrid (SCL AND OCM) or a
  config-selectable mode to keep the masking logic single-path and
  because OCM's published accuracy is higher than SCL's; a hybrid was
  considered but rejected as unnecessary complexity for this pipeline.

- 2026-08-20: **Integration point: run OCM once per loaded scene window,
  not per full MGRS tile.** The pipeline already windows each scene read
  to the field-polygon bounding box (+ padding) before loading —
  `load_scene()` (feeds the per-scene Parquet / `sample_scene()` path) and
  `load_composite()` (feeds the on-the-fly median-composite path in
  `main()`). OCM now runs on that already-loaded, already-bounded array
  right after it's read into memory, in both places:
  - `load_scene()`: one OCM run per scene (or per spatial batch, when
    `scene_spatial_batching` splits a large scene into sub-windows —
    each batch is loaded and masked independently).
  - `load_composite()`: one OCM run per STAC item (time step) in the
    loaded stack, looped, before the median composite is computed.
  `mosaic_scene_samples.py` (which builds mosaics from the scene-sample
  Parquet files) does **not** re-run OCM — it just reads the `valid_px`
  column that `sample_scene()` already wrote. This satisfies "one OCM run
  per image": each scene/window is masked exactly once, downstream
  composite/mosaic steps reuse that result rather than recomputing it.

- 2026-08-20: **Masked classes: strict clear-sky.** OmniCloudMask outputs
  4 classes (0=clear, 1=thick cloud, 2=thin cloud, 3=shadow). Only class 0
  counts as valid; thick cloud, thin cloud, and shadow are all masked out.
  Rejected: keeping thin cloud (class 2) as valid to retain more
  pixels/dates — decided against because thin cloud still biases
  reflectance and downstream indices (NDVI etc.).

- 2026-08-20: **Inference parameters** (`scripts/cloud_mask.py`): OCM's
  published defaults — `patch_size=1000`, `patch_overlap=300`,
  `batch_size=1`, `model_version=None` (latest). `inference_dtype` is
  `fp16` on CUDA GPUs (speed) and `fp32` on CPU/MPS (fp16 is slow/
  unsupported on most CPUs, MPS has partial fp16 support). `inference_device`
  is auto-detected (`cuda` → `mps` → `cpu`). `no_data_value=0` matches the
  Sentinel-2 nodata fill value used by `odc.stac.load`.

- 2026-08-20: **Model weight cache location.** OmniCloudMask downloads its
  model weights on first use; `destination_model_dir` is pinned to a
  project-local `.model_cache/omnicloudmask/` (git-ignored) instead of the
  library's default out-of-project cache directory, per the standing rule
  against writing outside the project root without asking.

- 2026-08-20: **Local package install: CPU-only torch.** `omnicloudmask`
  and `torch` (CPU wheels, via
  `--extra-index-url https://download.pytorch.org/whl/cpu`) added to
  `environment.yml`; the `halo-s2` conda env was created fresh from it (no
  such env existed locally beforehand — only `base`, `galileo-pc-embeddings`,
  `geospatial` were present). CPU wheels chosen to avoid a multi-GB CUDA
  download on a dev machine that isn't doing GPU inference; GPU inference is
  intended for Azure ML instead, where the exact CUDA build can be pinned to
  the compute SKU when a GPU cluster is actually used.

- 2026-08-20: **Azure ML compute: CPU for now.** `azml.txt` (git-ignored,
  local infra notes) shows the only documented compute cluster,
  `cluster-rise-d16`, is a D-series VM — CPU-only, no GPU. Rather than block
  on provisioning a GPU cluster, the pipeline runs OCM on CPU on Azure ML
  for now. No code change is needed to add GPU support later:
  `cloud_mask.py`'s device auto-detection means pointing the job's
  `compute:` at a GPU cluster (e.g. an NC-series) is a config-only change.

- 2026-08-20: **Azure ML environment: switched to a named, pre-built
  environment** (`sentinels_poly_timeseries_extract`), replacing the
  previous inline `image: mcr.microsoft.com/...` + `conda_file:
  environment.yml` approach in `azureml/halo_s2_pipeline_job.yml`. Defined
  in `azureml/environment/` (Dockerfile + AML environment asset spec) so
  the `halo-s2` env (now including `omnicloudmask`/`torch`) is baked into
  the image at build time instead of solved/installed on every job run —
  faster job startup. Chosen name is project-specific
  (`sentinels_poly_timeseries_extract`), distinct from
  `galileo-pc-embeddings` (a different project's environment referenced in
  `azml.txt`). The Dockerfile/environment spec were written but **not**
  built, pushed, or registered by Claude Code — building/pushing to
  `acrrisewesteurope` and registering the environment in
  `mlw-rise-westeurope` are actions on shared Azure infra, left for the
  user to run (commands documented in `README.md`).

- 2026-08-20: **Verified locally.** Created the `halo-s2` conda env from
  the updated `environment.yml` (`torch-2.13.0+cpu`, `omnicloudmask-1.7.1`;
  no GPU on this dev machine, confirmed via `torch.cuda.is_available()` →
  `False`, so it exercised the CPU/fp32 code path). Smoke-tested
  `add_ocm_mask()`, `sample_scene()`, and the per-time-step masking logic
  in `load_composite()` against synthetic Sentinel-2-shaped data (not real
  STAC data — no network/PC calls were made). All three produced the
  expected `OCM_CLASS`/`OCM_CLEAR`/`valid_px` output and the model weights
  downloaded correctly into the git-ignored `.model_cache/`.

- 2026-08-20: **Found and fixed (user requested):** `timed_step()`'s log
  message (`download_s2_pc.py`) included a `Δ` character that raised
  `UnicodeEncodeError` on a plain Windows console (cp1252 stdout) — never
  surfaced before because Azure ML's Linux containers default to UTF-8.
  Pre-existing, unrelated to cloud masking; replaced `ΔRAM=` with plain
  ASCII `dRAM=` (log text only, no behavior change). Re-verified the
  `add_ocm_mask()` smoke test runs clean without the earlier
  `PYTHONIOENCODING=utf-8` workaround.

## First real Azure ML test (`configs/test_ebrd10_2025.json`, `azureml/test_ebrd10_job.yml`)

- 2026-08-20: **Test field selection.** `configs/test_data/EBRD_merged_20251212_geoDb.gpkg`
  (git-ignored, 203MB, dropped in locally by the user) is a farm-operations
  log, not a clean field layer: 16,714 rows but only 87 unique fields (each
  repeated across many operation records with identical geometry). Deduped
  on `Field ID` to 87 unique polygons, then randomly sampled 10 with
  `gdf.sample(n=10, random_state=42)` → written to
  `configs/test_data/ebrd_test10_fields.geojson` (git-ignored, small enough
  to not matter but kept alongside the source for reproducibility). Seed 42
  chosen for reproducibility, matching the pipeline's own default
  `random_seed`. Selected Field IDs: 90.18.10.01.01.03, 90.18.10.01.01.05,
  90.18.10.01.01.09, 90.18.13.14.55.13, 90.18.13.14.56.09, 90.18.13.14.58.08,
  90.18.13.29.99.12, 90.18.13.29.99.17, 90.25.04.05.06.06, 90.25.15.05.13.23.

- 2026-08-20: **`.amlignore` added.** The 203MB source geopackage sitting in
  `configs/test_data/` would otherwise get swept into the job's code
  snapshot (`halo_s2_pipeline_job.yml`/`test_ebrd10_job.yml` both use
  `code: ..`, i.e. the whole repo root). Added `.amlignore` mirroring the
  `.gitignore` data/model exclusions, per the standing rule to check every
  ignore-style file (not just `.gitignore`) before letting a bulky local
  file sit in-tree.

- 2026-08-20: **Test date range: March–October 2025** (`start_date`/
  `end_date` in `configs/test_ebrd10_2025.json`), per explicit user
  instruction — a 2025-growing-season window over the 10 selected fields.

- 2026-08-20: **max_cloud_cover loosened to 90%** (vs. the production
  template's 40%) for this test only. Since OmniCloudMask now does the
  real per-pixel filtering, the STAC-level `eo:cloud_cover` pre-filter is
  just a coarse pre-fetch filter; loosening it exercises OCM against a
  wider range of scene conditions rather than relying on the scene-level
  metadata to have already screened out cloud.

- 2026-08-20: **max_workers=8 with `OMP_NUM_THREADS=2`/`MKL_NUM_THREADS=2`.**
  `cluster-rise-d16` has 16 vCPUs. Each `ProcessPoolExecutor` worker in
  `run_scene_sampling()` runs its own PyTorch/OmniCloudMask instance;
  PyTorch's CPU backend defaults to using *all visible cores* per process
  for intra-op parallelism unless told otherwise, so naively setting
  `max_workers=16` would oversubscribe the node (up to 16 processes × up to
  16 threads each). Capped each worker to 2 threads via env vars (which
  PyTorch reads automatically at process start — no code change needed,
  and `ProcessPoolExecutor` workers inherit the parent's environment) and
  set `max_workers=8`, so 8 × 2 = 16 matches the node's core count. More
  processes (rather than more threads per process) was chosen because
  scene loading is largely network-bound (downloading Sentinel-2 COG
  assets from Planetary Computer) while OCM inference is CPU-bound, so
  process-level parallelism lets one scene's network wait overlap with
  another's inference. Set only in `azureml/test_ebrd10_job.yml` (test-only),
  not the production `halo_s2_pipeline_job.yml` — the right max_workers/
  thread split for a production run depends on whatever compute cluster
  is eventually used for that, not decided yet.

- 2026-08-20: **Test data path convention:** uploaded the 10-field GeoJSON
  to `rise_data` datastore at
  `JosefWagner/halo_azml/test/fields/ebrd_test10_fields.geojson`, matching
  the `<username>/<project>_azml/<stage>/...` convention documented in
  `azml.txt`. Output written to
  `rise_data:JosefWagner/halo_azml/test/planetary_computer_samples/`.
  Compute: `cluster-rise-d16` (hardcoded in `test_ebrd10_job.yml`, unlike
  the production job's overridable placeholder — this is a one-off test,
  not meant to be reused as a template for arbitrary clusters).

- 2026-08-20: **Test result: passed.** Job `funny_rhubarb_dzgn92fvr0`
  (`halo-s2-cloudmask-test-ebrd10`) completed successfully — ~27 min total
  (first-time image build ~10 min, cold-start node scaling from 0, then
  the actual inventory→download→mosaic run). Verified by downloading real
  output from `rise_data:JosefWagner/halo_azml/test/planetary_computer_samples/`:
  - Scene-sample Parquet files written across 3 MGRS tiles (36UVA, 36UVB,
    36UWB — matches the 10 fields' geographic spread) and multiple months
    (March onward).
  - Spot-checked one real scene (`S2A_MSIL2A_20250314T085751...`, field
    `90.25.04.05.06.06`, 58,941 pixel rows): `OCM_CLASS` was 0 (clear) for
    6,593 px, 1 (thick cloud) for 4 px, 2 (thin cloud) for 52,344 px —
    `valid_px == (OCM_CLASS == 0)` held for every row. **Concretely
    validates the SCL→OCM switch**: SCL classified most of those same
    thin-cloud pixels as class 5 (bare soil — one of the *old*
    `valid_scl_classes`) or class 10 (thin cirrus), i.e. the retired SCL
    filter would have kept ~24,225 px as "valid" that OmniCloudMask
    correctly flags as thin cloud.
  - Mosaic output (`mosaic_metadata_median_14d.json`): 8,866,575 valid
    scene-sample rows survived the OCM filter → 3,198,171 pixel-mosaic
    rows → 158 field-summary rows (10 fields × ~15.8 fourteen-day windows
    over the ~214-day range, as expected).
  - `field_summary_median_14d.csv`: all 10 field IDs present, NDVI within
    a sensible growing-season trajectory (~0.2 in March rising to ~0.7 by
    late June for field 90.18.10.01.01.03), no out-of-range values.
