"""Sentinel-1 RTC (VV/VH backscatter) download, mirroring download_s2_pc.py.

Same scene-based design as the Sentinel-2 pipeline (one Parquet per scene,
STAC inventory cached first, then sampled in parallel), but against the
``sentinel-1-rtc`` collection instead of ``sentinel-2-l2a``. RTC is already
radiometrically terrain-corrected and geocoded to a UTM grid (unlike raw
GRD), so it reads the same way S2 does via odc.stac.load - no calibration
or terrain-correction step needed here. See METHODS.md for why RTC was
chosen over GRD.

This file duplicates a handful of small generic helpers from
download_s2_pc.py (log/timed_step/retry_with_backoff/resolve_fields_path/
estimate_utm_crs/split_fields_spatially/load_cached_scene_jobs) rather than
importing them from there, specifically to avoid pulling in
download_s2_pc.py's `cloud_mask` import (torch) - Sentinel-1 needs no
optical cloud masking (SAR isn't affected by cloud cover), and this keeps
the S1 pipeline free of any torch/OpenMP fork-safety concerns entirely.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import planetary_computer
import pystac_client
import odc.stac
import rasterio.features
from shapely.geometry import shape
import argparse
import time
from contextlib import contextmanager
import psutil
import os
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pystac


SCRIPT_START = time.perf_counter()

PROCESS = psutil.Process(os.getpid())

S1_ASSETS = ["vv", "vh"]
S1_NODATA = -32768.0


def _is_mpc(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "planetarycomputer.microsoft.com" or host.endswith(
        ".planetarycomputer.microsoft.com"
    )


def get_client(catalog_url: str, sign: bool = True) -> pystac_client.Client:
    kwargs = {}
    if sign and _is_mpc(catalog_url):
        kwargs["modifier"] = planetary_computer.sign_inplace
    return pystac_client.Client.open(catalog_url, **kwargs)


def memory_mb():
    return PROCESS.memory_info().rss / 1024 / 1024


def log(message):
    elapsed = time.perf_counter() - SCRIPT_START
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


@contextmanager
def timed_step(message):
    start_time = time.perf_counter()
    start_mem = memory_mb()

    log(f"START: {message} | RAM={start_mem:.1f} MB")

    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_mem = memory_mb()

        log(
            f"DONE:  {message} "
            f"({end_time - start_time:.1f}s) | "
            f"RAM={end_mem:.1f} MB | "
            f"dRAM={end_mem - start_mem:+.1f} MB"
        )


def retry_with_backoff(label, func, retries=3, base_sleep=10):
    for attempt in range(1, retries + 1):
        try:
            log(f"Attempt {attempt}/{retries}: {label}")
            return func()

        except Exception as exc:
            log(f"{label} failed on attempt {attempt}/{retries}: {exc}")

            if attempt == retries:
                raise

            sleep_seconds = base_sleep * attempt
            log(f"Retrying {label} in {sleep_seconds}s")
            time.sleep(sleep_seconds)


def resolve_fields_path(path):
    p = Path(path)
    if not p.is_dir():
        return p

    for pattern in ("*.shp", "*.gpkg", "*.geojson", "*.json"):
        matches = sorted(p.glob(pattern))
        if matches:
            if len(matches) > 1:
                log(f"Multiple '{pattern}' files in {p}; using {matches[0].name}")
            return matches[0]

    raise FileNotFoundError(
        f"No vector file (.shp/.gpkg/.geojson) found in fields directory: {p}"
    )


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    config["fields_path"] = resolve_fields_path(config["fields_path"])
    config["output_dir"] = Path(config["output_dir"])

    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--stac-inventory-only",
        action="store_true",
        help="Only query STAC and write item/field mapping files. Does not load or sample imagery.",
    )
    parser.add_argument(
        "--scene-samples",
        action="store_true",
        help="Sample every matching Sentinel-1 scene separately and write one Parquet file per scene.",
    )
    return parser.parse_args()


def safe_filename(value):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def estimate_utm_crs(gdf):
    centroid = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def bbox_pixel_count(bounds, resolution, pad=0):
    minx, miny, maxx, maxy = bounds
    width = max(0, (maxx - minx) + (2 * pad))
    height = max(0, (maxy - miny) + (2 * pad))
    return int(np.ceil(width / resolution) * np.ceil(height / resolution))


def split_fields_spatially(
    fields_proj,
    resolution,
    pad=0,
    max_window_pixels=None,
    batch_size_m=None,
):
    if fields_proj.empty:
        return []

    if max_window_pixels is None:
        return [fields_proj]

    full_pixels = bbox_pixel_count(fields_proj.total_bounds, resolution, pad=pad)

    if full_pixels <= max_window_pixels:
        log(f"Scene bbox pixel count {full_pixels:,}; using one read")
        return [fields_proj]

    if batch_size_m is None:
        batch_size_m = np.sqrt(max_window_pixels) * resolution

    fields_tmp = fields_proj.copy()
    centroids = fields_tmp.geometry.centroid
    minx, miny, _, _ = fields_tmp.total_bounds
    fields_tmp["_batch_x"] = np.floor((centroids.x - minx) / batch_size_m).astype(int)
    fields_tmp["_batch_y"] = np.floor((centroids.y - miny) / batch_size_m).astype(int)

    batches = []

    for _, group in fields_tmp.groupby(["_batch_x", "_batch_y"], sort=True):
        group = group.drop(columns=["_batch_x", "_batch_y"])
        group_pixels = bbox_pixel_count(group.total_bounds, resolution, pad=pad)

        if group_pixels <= max_window_pixels or len(group) == 1:
            batches.append(group)
            continue

        group = group.assign(_centroid_x=group.geometry.centroid.x).sort_values("_centroid_x")
        current = []

        for idx, row in group.iterrows():
            candidate = group.loc[current + [idx]].drop(columns=["_centroid_x"])
            candidate_pixels = bbox_pixel_count(candidate.total_bounds, resolution, pad=pad)

            if current and candidate_pixels > max_window_pixels:
                batches.append(group.loc[current].drop(columns=["_centroid_x"]))
                current = [idx]
            else:
                current.append(idx)

        if current:
            batches.append(group.loc[current].drop(columns=["_centroid_x"]))

    log(
        f"Split large scene bbox {full_pixels:,} pixels into "
        f"{len(batches)} spatial read batches"
    )

    return batches


def search_s1_items(
    aoi_4326,
    start,
    end,
    bbox=None,
    retries=3,
    sign=True,
):
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

    client = get_client(catalog_url, sign=sign)

    search_kwargs = {
        "collections": ["sentinel-1-rtc"],
        "datetime": f"{start.date()}/{end.date()}",
        "limit": 50,
    }

    if bbox is not None:
        search_kwargs["bbox"] = [float(v) for v in bbox]
    else:
        search_kwargs["intersects"] = aoi_4326.__geo_interface__

    for attempt in range(1, retries + 1):
        try:
            search = client.search(**search_kwargs)
            return list(search.items())

        except Exception as exc:
            log(f"STAC search failed on attempt {attempt}/{retries}: {exc}")

            if attempt == retries:
                raise

            sleep_seconds = 5 * attempt
            log(f"Retrying STAC search in {sleep_seconds}s")
            time.sleep(sleep_seconds)


def s1_item_metadata_record(item):
    props = item.properties

    return {
        "item_id": item.id,
        "collection": item.collection_id,
        "datetime": props.get("datetime"),
        "platform": props.get("platform"),
        "constellation": props.get("constellation"),
        "orbit_state": props.get("sat:orbit_state"),
        "relative_orbit": props.get("sat:relative_orbit"),
        "absolute_orbit": props.get("sat:absolute_orbit"),
        "instrument_mode": props.get("sar:instrument_mode"),
        "proj_code": props.get("proj:code"),
        "bbox_minx": item.bbox[0] if item.bbox else np.nan,
        "bbox_miny": item.bbox[1] if item.bbox else np.nan,
        "bbox_maxx": item.bbox[2] if item.bbox else np.nan,
        "bbox_maxy": item.bbox[3] if item.bbox else np.nan,
    }


def build_stac_inventory(
    fields_4326,
    aoi_4326,
    start_date,
    end_date,
    output_dir,
    field_id_col,
    bbox=None,
):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    with timed_step("Searching full STAC inventory"):
        items = search_s1_items(
            aoi_4326,
            start,
            end,
            bbox=bbox,
            sign=False,
        )

    log(f"Found {len(items)} Sentinel-1 items in full inventory search")

    if not items:
        return pd.DataFrame(), pd.DataFrame()

    records = []
    geometries = []

    for item in items:
        records.append(s1_item_metadata_record(item))
        geometries.append(shape(item.geometry))

    inventory_gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")

    item_columns = [
        "item_id",
        "collection",
        "datetime",
        "platform",
        "constellation",
        "orbit_state",
        "relative_orbit",
        "absolute_orbit",
        "instrument_mode",
        "proj_code",
        "bbox_minx",
        "bbox_miny",
        "bbox_maxx",
        "bbox_maxy",
    ]

    inventory_csv = output_dir / "stac_inventory_items.csv"
    inventory_geojson = output_dir / "stac_inventory_items.geojson"
    inventory_jsonl = output_dir / "stac_inventory_items.jsonl"

    with timed_step(f"Writing STAC item inventory: {inventory_csv.name}"):
        inventory_gdf[item_columns].to_csv(inventory_csv, index=False)

    with timed_step(f"Writing STAC item footprints: {inventory_geojson.name}"):
        inventory_gdf.to_file(inventory_geojson, driver="GeoJSON")

    with timed_step(f"Writing full STAC item cache: {inventory_jsonl.name}"):
        with open(inventory_jsonl, "w", encoding="utf-8") as file:
            for item in items:
                file.write(json.dumps(item.to_dict()) + "\n")

    fields_for_join = fields_4326[[field_id_col, "geometry"]].copy()

    with timed_step("Joining fields to intersecting STAC items"):
        field_item_gdf = gpd.sjoin(
            fields_for_join,
            inventory_gdf[["item_id", "datetime", "orbit_state", "relative_orbit", "geometry"]],
            how="inner",
            predicate="intersects",
        )

    field_item_map = (
        field_item_gdf
        .drop(columns=["geometry", "index_right"], errors="ignore")
        .rename(columns={field_id_col: "field_id"})
        .sort_values(["field_id", "datetime", "item_id"])
        .reset_index(drop=True)
    )

    map_csv = output_dir / "stac_field_item_map.csv"

    with timed_step(f"Writing field-to-item map: {map_csv.name}"):
        field_item_map.to_csv(map_csv, index=False)

    orbit_summary = (
        field_item_map
        .groupby(["orbit_state", "relative_orbit"], dropna=False)
        .agg(
            item_count=("item_id", "nunique"),
            field_count=("field_id", "nunique"),
            first_datetime=("datetime", "min"),
            last_datetime=("datetime", "max"),
        )
        .reset_index()
        .sort_values(["item_count", "field_count"], ascending=False)
    )

    orbit_csv = output_dir / "stac_orbit_summary.csv"

    with timed_step(f"Writing orbit summary: {orbit_csv.name}"):
        orbit_summary.to_csv(orbit_csv, index=False)

    log(f"Wrote {len(inventory_gdf):,} STAC items")
    log(f"Wrote {len(field_item_map):,} field/item intersections")
    log(f"Wrote {len(orbit_summary):,} orbit summary rows")

    return inventory_gdf, field_item_map


def load_cached_scene_jobs(output_dir):
    item_cache_path = output_dir / "stac_inventory_items.jsonl"
    field_item_map_path = output_dir / "stac_field_item_map.csv"

    if not item_cache_path.exists() or not field_item_map_path.exists():
        missing = [
            str(path)
            for path in [item_cache_path, field_item_map_path]
            if not path.exists()
        ]
        raise FileNotFoundError(
            "Missing STAC cache files. Run with --stac-inventory-only first. "
            f"Missing: {missing}"
        )

    with timed_step(f"Reading cached STAC items: {item_cache_path.name}"):
        item_dicts = {}
        with open(item_cache_path, "r", encoding="utf-8") as file:
            for line in file:
                item_dict = json.loads(line)
                item_dicts[item_dict["id"]] = item_dict

    with timed_step(f"Reading cached field/item map: {field_item_map_path.name}"):
        field_item_map = pd.read_csv(field_item_map_path)

    scene_jobs = []

    field_item_map["datetime"] = pd.to_datetime(field_item_map["datetime"], errors="coerce")
    field_item_map = field_item_map.sort_values(["datetime", "item_id"])

    for item_id, group in field_item_map.groupby("item_id", sort=False):
        if item_id not in item_dicts:
            log(f"Skipping cached map item missing from item cache: {item_id}")
            continue

        field_ids = sorted(group["field_id"].astype(str).unique())
        scene_jobs.append({
            "item_dict": item_dicts[item_id],
            "field_ids": field_ids,
        })

    log(f"Built {len(scene_jobs):,} scene jobs from cached STAC inventory")

    return scene_jobs


def output_path_for_scene(output_dir, item, scene_samples_subdir="s1_scene_samples"):
    props = item.properties
    dt = pd.to_datetime(props.get("datetime"))
    orbit_state = props.get("sat:orbit_state", "unknown_orbit")
    relative_orbit = props.get("sat:relative_orbit", "unknown")

    scene_dir = (
        output_dir
        / scene_samples_subdir
        / f"orbit_state={safe_filename(orbit_state)}"
        / f"relative_orbit={safe_filename(relative_orbit)}"
        / f"year={dt.year}"
        / f"month={dt.month:02d}"
    )
    scene_dir.mkdir(parents=True, exist_ok=True)

    return scene_dir / f"{safe_filename(item.id)}.parquet"


def load_scene(item, crs, x_bounds, y_bounds, assets, resolution=10):
    log(f"Loading scene item: {item.id}")
    log(f"Loading assets: {assets}")
    log(f"Target CRS: {crs}")
    log(f"x bounds: {x_bounds}")
    log(f"y bounds: {y_bounds}")

    ds = odc.stac.load(
        [item],
        bands=assets,
        crs=crs,
        resolution=resolution,
        x=x_bounds,
        y=y_bounds,
        chunks={},
    )

    if "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)

    with timed_step("Computing scene into memory"):
        ds = ds.load()

    return ds


def sample_scene(
    scene,
    fields_proj,
    item,
    field_id_col,
    max_pixels_per_scene=None,
    random_seed=1,
):
    transform = scene.odc.transform
    height = scene.sizes["y"]
    width = scene.sizes["x"]
    props = item.properties

    log(f"Scene grid size: y={height}, x={width}")
    log(f"Scene CRS: {scene.odc.crs}")

    field_lookup = dict(enumerate(fields_proj[field_id_col].astype(str), start=1))

    shapes = [
        (geom, idx)
        for idx, geom in enumerate(fields_proj.geometry, start=1)
        if geom is not None and not geom.is_empty
    ]

    with timed_step("Rasterizing scene field polygons"):
        field_raster = rasterio.features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="int32",
        )

    ys, xs = np.where(field_raster > 0)
    log(f"Candidate pixels inside scene polygons: {len(xs):,}")

    if len(xs) == 0:
        return pd.DataFrame()

    if max_pixels_per_scene is not None and len(xs) > max_pixels_per_scene:
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(xs), size=max_pixels_per_scene, replace=False)
        ys = ys[keep]
        xs = xs[keep]
        log(f"Sampled down to max_pixels_per_scene={max_pixels_per_scene:,}")

    sample_bands = [band for band in S1_ASSETS if band in scene]

    with timed_step("Materializing scene band arrays for sampling"):
        band_arrays = {
            band: scene[band].values
            for band in sample_bands
        }

    field_codes = field_raster[ys, xs]
    field_ids = [field_lookup[code] for code in field_codes]
    x_coords = scene.x.values[xs]
    y_coords = scene.y.values[ys]

    data = {
        "field_id": field_ids,
        "point_id": [
            f"{field_ids[i]}_{int(xs[i])}_{int(ys[i])}"
            for i in range(len(xs))
        ],
        "item_id": item.id,
        "datetime": props.get("datetime"),
        "date": pd.to_datetime(props.get("datetime")).date().isoformat(),
        "orbit_state": props.get("sat:orbit_state"),
        "relative_orbit": props.get("sat:relative_orbit"),
        "instrument_mode": props.get("sar:instrument_mode"),
        "platform": props.get("platform"),
        "pixel_col": xs.astype(int),
        "pixel_row": ys.astype(int),
        "x": x_coords.astype(float),
        "y": y_coords.astype(float),
    }

    for band in sample_bands:
        data[band] = band_arrays[band][ys, xs]

    samples = pd.DataFrame(data)
    samples = samples.replace([np.inf, -np.inf], np.nan)

    valid = pd.Series(True, index=samples.index)
    for band in sample_bands:
        valid &= samples[band] != S1_NODATA
    samples["valid_px"] = valid

    with timed_step("Converting scene sample coordinates to lon/lat"):
        samples_gdf = gpd.GeoDataFrame(
            samples,
            geometry=gpd.points_from_xy(samples["x"], samples["y"]),
            crs=scene.odc.crs,
        ).to_crs("EPSG:4326")

    samples["lon"] = samples_gdf.geometry.x
    samples["lat"] = samples_gdf.geometry.y

    return samples


def process_one_scene_item(
    item,
    fields_4326,
    output_dir,
    config,
    field_ids=None,
    item_number=None,
    item_count=None,
):
    if item_number is None or item_count is None:
        scene_label = "Scene"
    else:
        scene_label = f"Scene {item_number}/{item_count}"

    try:
        output_path = output_path_for_scene(
            output_dir,
            item,
            scene_samples_subdir=config.get("scene_samples_subdir", "s1_scene_samples"),
        )

        if config.get("skip_existing", True) and output_path.exists():
            return f"Skipped existing scene: {output_path.name}"

        if field_ids is not None:
            fields_for_scene = fields_4326[
                fields_4326[config["field_id_col"]].astype(str).isin(field_ids)
            ].copy()
        else:
            item_geom = shape(item.geometry)
            fields_for_scene = fields_4326[fields_4326.intersects(item_geom)].copy()

        if fields_for_scene.empty:
            return f"Skipped scene with no intersecting fields: {item.id}"

        target_crs = estimate_utm_crs(fields_for_scene)
        fields_proj = fields_for_scene.to_crs(target_crs)

        minx, miny, maxx, maxy = fields_proj.total_bounds
        pad = float(config.get("scene_bounds_padding_m", config.get("resolution", 10) * 2))
        spatial_batching = bool(config.get("scene_spatial_batching", False))

        if spatial_batching:
            field_batches = split_fields_spatially(
                fields_proj,
                resolution=config["resolution"],
                pad=pad,
                max_window_pixels=config.get("max_scene_window_pixels"),
                batch_size_m=config.get("scene_spatial_batch_size_m"),
            )
        else:
            field_batches = [fields_proj]

        sample_parts = []

        for batch_id, fields_batch in enumerate(field_batches, start=1):
            minx, miny, maxx, maxy = fields_batch.total_bounds
            x_bounds = (minx - pad, maxx + pad)
            y_bounds = (miny - pad, maxy + pad)

            log(
                f"{scene_label}: Processing spatial batch {batch_id}/{len(field_batches)} "
                f"with {len(fields_batch):,} fields"
            )

            scene = retry_with_backoff(
                f"Loading Sentinel-1 scene assets batch {batch_id}/{len(field_batches)}",
                lambda: load_scene(
                    item,
                    target_crs,
                    x_bounds,
                    y_bounds,
                    assets=config["assets"],
                    resolution=config["resolution"],
                ),
                retries=config.get("download_retries", 3),
                base_sleep=config.get("retry_base_sleep", 10),
            )

            batch_samples = sample_scene(
                scene,
                fields_batch,
                item,
                field_id_col=config["field_id_col"],
                max_pixels_per_scene=config.get("max_pixels_per_scene"),
                random_seed=config.get("random_seed", 1),
            )

            if not batch_samples.empty:
                batch_samples["spatial_batch_id"] = batch_id
                sample_parts.append(batch_samples)

        if sample_parts:
            samples = pd.concat(sample_parts, ignore_index=True)
        else:
            samples = pd.DataFrame()

        if samples.empty:
            return f"Skipped scene with no samples generated: {item.id}"

        with timed_step(f"Writing scene Parquet: {output_path.name}"):
            samples.to_parquet(output_path, index=False)

        return f"Wrote {len(samples):,} scene sample rows to {output_path}"

    except Exception as exc:
        return f"Scene failed: {item.id} | {exc}"


def process_one_scene_worker(scene_job, config):
    fields = gpd.read_file(resolve_fields_path(config["fields_path"]))

    if fields.crs is None:
        fields = fields.set_crs("EPSG:4326")

    fields_4326 = fields.to_crs("EPSG:4326")
    item = pystac.Item.from_dict(scene_job["item_dict"])
    item = planetary_computer.sign(item)

    return process_one_scene_item(
        item=item,
        fields_4326=fields_4326,
        output_dir=Path(config["output_dir"]),
        config=config,
        field_ids=scene_job["field_ids"],
    )


def run_scene_sampling(
    fields_4326,
    aoi_4326,
    output_dir,
    config,
    bbox=None,
):
    scene_jobs = load_cached_scene_jobs(output_dir)

    if not scene_jobs:
        return

    max_workers = int(config.get("max_workers", 1))

    if max_workers > 1:
        log(f"Running scene sampling with max_workers={max_workers}")

        worker_config = dict(config)
        worker_config["fields_path"] = str(worker_config["fields_path"])
        worker_config["output_dir"] = str(worker_config["output_dir"])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_one_scene_worker,
                    scene_job,
                    worker_config,
                ): scene_job["item_dict"]["id"]
                for scene_job in scene_jobs
            }

            for future in as_completed(futures):
                item_id = futures[future]
                try:
                    result = future.result()
                    log(result)
                except Exception as exc:
                    log(f"Scene failed: {item_id} | {exc}")
        return

    for item_number, scene_job in enumerate(scene_jobs, start=1):
        item = pystac.Item.from_dict(scene_job["item_dict"])
        item = planetary_computer.sign(item)

        result = process_one_scene_item(
            item=item,
            fields_4326=fields_4326,
            output_dir=output_dir,
            config=config,
            field_ids=scene_job["field_ids"],
            item_number=item_number,
            item_count=len(scene_jobs),
        )
        log(result)


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = config["start_date"]
    end_date = config["end_date"]

    fields = gpd.read_file(config["fields_path"])

    if fields.crs is None:
        fields = fields.set_crs("EPSG:4326")

    fields_4326 = fields.to_crs("EPSG:4326")
    aoi_4326 = fields_4326.geometry.union_all()
    search_bbox = tuple(float(v) for v in fields_4326.total_bounds)

    if args.stac_inventory_only:
        build_stac_inventory(
            fields_4326=fields_4326,
            aoi_4326=aoi_4326,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            field_id_col=config["field_id_col"],
            bbox=search_bbox,
        )
        return

    if args.scene_samples:
        run_scene_sampling(
            fields_4326=fields_4326,
            aoi_4326=aoi_4326,
            output_dir=output_dir,
            config=config,
            bbox=search_bbox,
        )
        return

    raise SystemExit("Specify --stac-inventory-only or --scene-samples")


if __name__ == "__main__":
    main()
