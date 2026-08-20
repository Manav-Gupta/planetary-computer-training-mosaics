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

- 2026-08-20: **Local preview export.** Wrote a one-off script
  (`.scratch/export_preview.py`, not committed - scratch only) to fetch one
  real scene locally, run OmniCloudMask, and export true-color +
  cloud-mask-class PNG/GeoTIFF previews for visual inspection, at the
  user's request ("can I download the processed img and cloudmask").
  Hit a `libomp.dll`/`libiomp5md.dll` duplicate-OpenMP-runtime crash
  (common torch-on-Windows issue, unrelated to this project) - worked
  around with `KMP_DUPLICATE_LIB_OK=TRUE`, not applied anywhere in the
  committed pipeline code since it hasn't occurred there (Linux AML jobs
  aren't affected; only hit this in ad hoc local scripting that imports
  rasterio+torch together in a way the earlier local smoke tests didn't).

## Scaling to all 87 fields, full calendar year 2025

- 2026-08-20: **Measured before optimizing.** Ran `--stac-inventory-only`
  locally (free, no imagery download) against all 87 unique EBRD fields
  before deciding how to scale up:
  - Mar 1 - Oct 1 2025: 674 scenes / 6 MGRS tiles / 8,584 field-scene
    intersections (avg ~12.7 fields per scene - confirms the "read each
    scene once" design avoids that much redundant reading).
  - Jan 1 - Dec 31 2025 (full calendar year, per user request): 841
    scenes / same 6 tiles / 10,530 field-scene intersections. Tiles
    balanced 95-167 items each.

- 2026-08-20: **Optimization: bigger single node, not multi-node
  re-architecture.** The pipeline already reads each scene once regardless
  of field overlap, so the lever for a bigger run is per-node parallelism.
  `cluster-rise` (`Standard_E64ds_v4`: 64 vCPUs, 504GB RAM - confirmed via
  `az vm list-sizes`) already exists in the workspace, idle, vs.
  `cluster-rise-d16`'s 16 vCPUs/64GB. Switched compute to `cluster-rise`
  and scaled `max_workers` from 8→32 with `OMP_NUM_THREADS`/
  `MKL_NUM_THREADS` still capped at 2 (32×2=64, same core-matching
  reasoning as the first test - see the `max_workers=8` entry above). No
  pipeline code changes, and the already-built `sentinels_poly_timeseries_extract`
  image needed no rebuild (images aren't tied to a specific compute
  cluster). True multi-node horizontal scaling (sharding fields by MGRS
  tile across cluster-rise's up to-32-node ceiling) was considered and
  explicitly rejected as unnecessary re-architecture for only 6 tiles/
  ~800-900 scenes - noted as a future option if the AOI grows much larger.

- 2026-08-20: **Scope for this run, per explicit user instructions:**
  all 87 fields (not the 10-field sample), Jan 1 - Dec 31 2025 (full
  calendar year, not just the growing season used for the first test),
  `inventory` + `download` steps only - **no `mosaic` step** ("just image
  processing and cloud mask" - per-scene Parquet output only, compositing
  deferred). `max_cloud_cover` kept at 90% (unchanged from the first
  test). Config: `configs/test_ebrd87_2025.json`. Job spec:
  `azureml/test_ebrd87_job.yml`. Output path distinct from the 10-field
  test: `rise_data:JosefWagner/halo_azml/test_full87/...` (still a test
  path, not a production deliverable location - this remains the EBRD
  validation dataset, not HALO's own field data). Job:
  `mango_kale_yb18t1cfhr`.

- 2026-08-20: **Snow, for the record.** User asked how OmniCloudMask
  handles snow, relevant now that the run spans winter months. Could not
  access the peer-reviewed benchmark (paywalled), so no precise accuracy
  number is claimed. What's verifiable: OCM's training data was
  deliberately curated with snow as a "hard negative" cloud-like surface
  (alongside sand/haze), unlike SCL which has a dedicated snow class (11)
  that used to be in `valid_scl_classes`. OCM has no separate snow output
  class - misclassified snow would show up as cloud and get masked out.
  For this pipeline's purpose (crop/vegetation monitoring), that's a
  reasonable failure mode even in the worst case: a snow-covered field
  isn't giving a usable NDVI regardless of the label.

- 2026-08-20: **Bug found and fixed: model-download race condition at high
  parallelism.** `mango_kale_yb18t1cfhr` (87 fields, full year 2025,
  `max_workers=32`) reported `status: Completed` but only wrote 17 of an
  expected ~841 scene Parquet files - a near-total silent data loss that
  the job's exit code did not surface. Diagnosed by downloading *only the
  job's text logs* (`az ml job download` without `--output-name`, which
  fetches the "default" artifacts output - system/user logs - not the
  "samples" data output; never downloaded the actual scene/imagery data
  locally, per explicit user instruction). `std_log.txt` showed hundreds
  of `Scene failed: ... | No such file or directory:
  .../.model_cache/omnicloudmask/PM_model_....safetensors` errors,
  clustered in the job's first ~150s. Root cause: `omnicloudmask`
  downloads its model weights lazily on first use; with 32
  `ProcessPoolExecutor` workers all hitting an *empty* cache simultaneously
  at job start, they raced to write the same weights file, and most lost
  the race and crashed (each scene's failure was caught per-scene by the
  existing try/except in `process_one_scene_item`, logged, and skipped -
  which is why the job "succeeded" overall despite ~98% data loss).
  Cross-checked the earlier 10-field test (`funny_rhubarb_dzgn92fvr0`,
  `max_workers=8`): zero `Scene failed` lines, 286/388 items correctly
  written - that earlier "passed" verdict stands; the race only manifests
  at higher parallelism racing a *cold* cache.
  **Fix**: added `warm_model_cache()` (`scripts/cloud_mask.py`) - runs one
  dummy OCM inference synchronously to force the model download to
  complete - called once in `run_scene_sampling()`
  (`scripts/download_s2_pc.py`) before the `ProcessPoolExecutor` is
  created, so the weights file already exists by the time workers start
  and none of them ever trigger a download. Verified the function runs
  correctly locally post-fix. Re-running `mango_kale_yb18t1cfhr`'s exact
  scope (same config/job spec) with the fix applied before treating that
  data as usable, and before proceeding to any larger run (full-tile test,
  a possible future 2017-2024 backfill) that would hit the same race at
  even larger scale.

- 2026-08-20: **Bug found and fixed: fork-after-torch-init deadlock in the
  race-condition fix itself.** The re-run of `mango_kale_yb18t1cfhr`'s scope
  with the `warm_model_cache()` fix (`quiet_leather_wlz7n45797`, display
  name `...-retry`) showed `status: Running` for ~2h with zero progress.
  Diagnosed without waiting for job completion: read `std_log.txt` and
  `scene_samples_batched/` blobs directly from the datastore/artifact store
  (`az storage blob list`/`download` against the `rise_data` and
  `workspaceartifactstore` datastores - bypasses `az ml job download`'s
  "must be Completed" restriction, and avoids downloading any imagery, only
  small text/log blobs). Found `std_log.txt` stopped growing 53s after job
  start - exactly 32 `START: Computing OmniCloudMask...` lines
  (`max_workers=32`), zero matching `DONE:` lines, zero exceptions logged:
  a deadlock, not a crash, with every single worker hung on its first
  inference call. Root cause: `warm_model_cache()` ran a real torch
  inference in the *main* process (to force the model download
  synchronously, fixing the earlier race), which initializes torch's
  OpenMP thread pool there; `ProcessPoolExecutor` then forks its 32 workers
  from that same main process (Python's default `multiprocessing` start
  method on Linux is `fork`). A process forked after the parent has an
  initialized OpenMP thread pool inherits corrupted thread-pool
  bookkeeping (the child gets only the forking thread, not OpenMP's other
  worker threads, but OpenMP's internal state is copied as-is) - a
  well-documented fork/OpenMP hazard, and it matches the evidence exactly
  (all workers hang on their first torch op, no exceptions, node otherwise
  healthy). Cancelled `quiet_leather_wlz7n45797` (`az ml job cancel`) once
  diagnosed. **Fix**: `warm_model_cache()` (`scripts/cloud_mask.py`) now
  runs its dummy inference in a throwaway subprocess spawned via
  `multiprocessing.get_context("spawn")`, not in-process - the main
  process itself never executes a torch op, so `ProcessPoolExecutor`'s
  later fork starts from a clean, torch-untouched parent. Verified locally
  (Windows, where `spawn` is already the default and can't reproduce the
  Linux fork deadlock, but confirms the subprocess correctly downloads/
  caches weights and leaves the main process's torch state usable
  afterward - `.model_cache/omnicloudmask/` populated, ~4.9s). The
  Linux-specific fork deadlock itself can only be validated by re-running
  on `cluster-rise`. Re-running the 87-field/full-year-2025 test a third
  time with this fix, to a new output path
  (`test_full87_v2/planetary_computer_samples/`, not the original
  `test_full87/` path) so this run's output can't be confused with the
  two earlier partial/failed attempts still sitting at the old path
  (left in place, not deleted). Only proceeding to the 2017-2024 backfill
  once this test completes cleanly (all scenes processed, `Scene failed`
  count consistent with real STAC/asset issues rather than systemic
  worker failure).
