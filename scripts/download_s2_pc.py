from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import planetary_computer
import pystac_client
import odc.stac
import rasterio.features
from shapely.geometry import Point, shape
from pyproj import CRS
import argparse
import time
from contextlib import contextmanager
import psutil
import os
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pystac

from cloud_mask import ocm_clear_mask, OCM_CLEAR


SCRIPT_START = time.perf_counter()

PROCESS = psutil.Process(os.getpid())

S2_REFLECTANCE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_BASELINE_HARMONIZATION_THRESHOLD = 4.0
S2_BASELINE_OFFSET = 1000

def _is_mpc(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "planetarycomputer.microsoft.com" or host.endswith(
        ".planetarycomputer.microsoft.com"
    )

def get_client(catalog_url: str, sign: bool = True) -> pystac_client.Client:
    """Get a pystac Client for the given catalog URL.

    For MPC, applies the planetary_computer modifier so that
    returned items have signed asset URLs. Pass ``sign=False`` when you
    only need catalog metadata (e.g. listing collections) - the signer
    eagerly hits MPC's SAS token endpoint per collection and a single
    broken collection (e.g. nex-gddp-cmip6) 404s the whole listing.
    """
    kwargs = {}
    if sign and _is_mpc(catalog_url):
        kwargs["modifier"] = planetary_computer.sign_inplace
    return pystac_client.Client.open(catalog_url, **kwargs)

def memory_mb():
    return PROCESS.memory_info().rss /1024/ 1024

def log(message):
    elapsed = time.perf_counter() - SCRIPT_START
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def get_s2_processing_baseline(item_or_props):
    """Return the Sentinel-2 processing baseline as a string, when present."""
    props = getattr(item_or_props, "properties", item_or_props)
    if props is None:
        return None

    return (
        props.get("s2:processing_baseline")
        or props.get("sentinel:processing_baseline")
        or props.get("processing:baseline")
    )


def parse_s2_processing_baseline(value):
    if value is None or pd.isna(value):
        return np.nan

    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError):
        return np.nan


def should_harmonize_s2_baseline(item_or_props):
    baseline = parse_s2_processing_baseline(get_s2_processing_baseline(item_or_props))
    return bool(np.isfinite(baseline) and baseline >= S2_BASELINE_HARMONIZATION_THRESHOLD)


def harmonize_s2_samples(samples, item):
    """Shift Sentinel-2 PB >= 04.00 DNs back to the pre-2022 range."""
    baseline_raw = get_s2_processing_baseline(item)
    baseline_value = parse_s2_processing_baseline(baseline_raw)
    apply_offset = should_harmonize_s2_baseline(item)

    samples["s2_processing_baseline"] = baseline_raw
    samples["s2_processing_baseline_value"] = baseline_value
    samples["s2_baseline_harmonized"] = apply_offset
    samples["s2_baseline_offset_applied"] = S2_BASELINE_OFFSET if apply_offset else 0

    if not apply_offset:
        return samples

    bands = [band for band in S2_REFLECTANCE_BANDS if band in samples.columns]
    if not bands:
        return samples

    log(
        f"Harmonizing Sentinel-2 baseline {baseline_raw}: "
        f"subtracting {S2_BASELINE_OFFSET} from {bands}"
    )
    samples[bands] = samples[bands].astype("float32") - S2_BASELINE_OFFSET
    return samples


def harmonize_s2_dataset(ds, item):
    """Shift Sentinel-2 PB >= 04.00 DNs back before compositing."""
    if not should_harmonize_s2_baseline(item):
        return ds

    baseline_raw = get_s2_processing_baseline(item)
    bands = [band for band in S2_REFLECTANCE_BANDS if band in ds]
    if not bands:
        return ds

    log(
        f"Harmonizing Sentinel-2 baseline {baseline_raw}: "
        f"subtracting {S2_BASELINE_OFFSET} from dataset bands {bands}"
    )

    ds = ds.copy()
    for band in bands:
        ds[band] = ds[band] - S2_BASELINE_OFFSET

    return ds


def harmonize_s2_timeseries_dataset(ds, items):
    """Apply Sentinel-2 baseline harmonization along a loaded time dimension."""
    if not items:
        return ds

    if "time" not in ds.dims:
        return harmonize_s2_dataset(ds, items[0])

    bands = [band for band in S2_REFLECTANCE_BANDS if band in ds]
    if not bands:
        return ds

    sorted_items = sorted(
        items,
        key=lambda item: pd.to_datetime(item.properties.get("datetime"), errors="coerce"),
    )
    offsets = [
        S2_BASELINE_OFFSET if should_harmonize_s2_baseline(item) else 0
        for item in sorted_items
    ]

    if len(offsets) != ds.sizes["time"]:
        log(
            "Skipping baseline harmonization for composite dataset: "
            f"{len(offsets)} item offsets for {ds.sizes['time']} time steps"
        )
        return ds

    if not any(offsets):
        return ds

    log(
        "Harmonizing Sentinel-2 baseline for composite dataset: "
        f"subtracting per-scene offsets {sorted(set(offsets))} from bands {bands}"
    )

    ds = ds.copy()
    offset_array = np.asarray(offsets, dtype=np.float32)[:, None, None]
    for band in bands:
        ds[band] = ds[band] - offset_array

    return ds

@contextmanager
def timed_step(message):
    
    start_time = time.perf_counter()
    start_mem = memory_mb()
    
    log(f"START: {message } | RAM={start_mem:.1f} MB")
    
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
    """Return the actual vector file to read.

    Azure ML mounts a ``uri_folder`` input as a directory, so ``--fields-path``
    can arrive as a folder rather than the ``.shp`` itself. When given a
    directory, pick the single vector file inside it (shapefile first, then
    GeoPackage / GeoJSON). Passing a file path through is a no-op.
    """
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
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--stac-inventory-only",
        action="store_true",
        help="Only query STAC and write item/field mapping files. Does not load or sample imagery.",
    )
    parser.add_argument(
        "--scene-samples",
        action="store_true",
        help="Sample every matching Sentinel-2 scene separately and write one Parquet file per scene.",
    )
    return parser.parse_args()


def safe_divide(num, den):
    return np.where(den != 0, num / den, np.nan)


def estimate_utm_crs(gdf):
    centroid = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def date_windows(start_date, end_date, interval_days):
    starts = pd.date_range(start_date, end_date, freq=f"{interval_days}D")
    for start in starts:
        end = min(start + pd.Timedelta(days=interval_days), pd.Timestamp(end_date))
        yield start, end


def search_s2_items(
    aoi_4326,
    start,
    end,
    max_cloud_cover=90,
    bbox=None,
    retries=3,
    sign=True,
):
    
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    
    client  = get_client(catalog_url, sign=sign)
 

    search_kwargs = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start.date()}/{end.date()}",
        "query": {"eo:cloud_cover": {"lte": max_cloud_cover}},
        "limit": 50,
    }
     
    #return list(search.items())

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


def item_metadata_record(item):
    props = item.properties

    return {
        "item_id": item.id,
        "collection": item.collection_id,
        "datetime": props.get("datetime"),
        "mgrs_tile": props.get("s2:mgrs_tile"),
        "eo_cloud_cover": props.get("eo:cloud_cover"),
        "platform": props.get("platform"),
        "constellation": props.get("constellation"),
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
    max_cloud_cover=90,
    bbox=None,
):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    with timed_step("Searching full STAC inventory"):
        items = search_s2_items(
            aoi_4326,
            start,
            end,
            max_cloud_cover=max_cloud_cover,
            bbox=bbox,
            sign=False,
        )

    log(f"Found {len(items)} Sentinel-2 items in full inventory search")

    if not items:
        return pd.DataFrame(), pd.DataFrame()

    records = []
    geometries = []

    for item in items:
        records.append(item_metadata_record(item))
        geometries.append(shape(item.geometry))

    inventory_gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")

    item_columns = [
        "item_id",
        "collection",
        "datetime",
        "mgrs_tile",
        "eo_cloud_cover",
        "platform",
        "constellation",
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
            inventory_gdf[["item_id", "datetime", "mgrs_tile", "eo_cloud_cover", "geometry"]],
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

    tile_summary = (
        field_item_map
        .groupby("mgrs_tile", dropna=False)
        .agg(
            item_count=("item_id", "nunique"),
            field_count=("field_id", "nunique"),
            first_datetime=("datetime", "min"),
            last_datetime=("datetime", "max"),
        )
        .reset_index()
        .sort_values(["item_count", "field_count"], ascending=False)
    )

    tile_csv = output_dir / "stac_tile_summary.csv"

    with timed_step(f"Writing tile summary: {tile_csv.name}"):
        tile_summary.to_csv(tile_csv, index=False)

    log(f"Wrote {len(inventory_gdf):,} STAC items")
    log(f"Wrote {len(field_item_map):,} field/item intersections")
    log(f"Wrote {len(tile_summary):,} tile summary rows")

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
    field_item_map = field_item_map.sort_values(["datetime", "mgrs_tile", "item_id"])
    
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


def safe_filename(value):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def output_path_for_scene(output_dir, item, scene_samples_subdir="scene_samples"):
    props = item.properties
    dt = pd.to_datetime(props.get("datetime"))
    mgrs_tile = props.get("s2:mgrs_tile", "unknown_tile")

    scene_dir = (
        output_dir
        / scene_samples_subdir
        / f"mgrs_tile={safe_filename(mgrs_tile)}"
        / f"year={dt.year}"
        / f"month={dt.month:02d}"
    )
    scene_dir.mkdir(parents=True, exist_ok=True)

    return scene_dir / f"{safe_filename(item.id)}.parquet"


def rename_s2_bands(ds):
    rename_map = {
        "B02": "B2",
        "B03": "B3",
        "B04": "B4",
        "B05": "B5",
        "B06": "B6",
        "B07": "B7",
        "B08": "B8",
    }
    rename_map = {old: new for old, new in rename_map.items() if old in ds}
    return ds.rename(rename_map)


def add_ocm_mask(ds):
    """Run OmniCloudMask on one loaded scene and attach OCM_CLASS/OCM_CLEAR.

    Requires the red (B4), green (B3), and NIR (B8) bands to already be
    loaded in ``ds``. Runs once for the whole array already in memory (the
    field-bounded window that was just read), not the full source tile.
    """
    required = ("B4", "B3", "B8")
    missing = [band for band in required if band not in ds]
    if missing:
        raise ValueError(
            f"Cannot compute OmniCloudMask: missing required band(s) {missing}. "
            "Include B04, B03, and B08 in the config 'assets' list."
        )

    with timed_step("Computing OmniCloudMask cloud/shadow mask"):
        clear, ocm_class = ocm_clear_mask(
            ds["B4"].values, ds["B3"].values, ds["B8"].values,
        )

    ds = ds.copy()
    ds["OCM_CLASS"] = (("y", "x"), ocm_class)
    ds["OCM_CLEAR"] = (("y", "x"), clear)
    return ds


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

    ds = rename_s2_bands(ds)

    with timed_step("Computing scene into memory"):
        ds = ds.load()

    ds = add_ocm_mask(ds)

    return ds


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

        # Fallback split for a grid cell that is still too broad.
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

    sample_bands = [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL",
        "OCM_CLASS",
    ]
    sample_bands = [band for band in sample_bands if band in scene]

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
        "mgrs_tile": props.get("s2:mgrs_tile"),
        "eo_cloud_cover": props.get("eo:cloud_cover"),
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
    samples = harmonize_s2_samples(samples, item)

    if "OCM_CLASS" not in samples.columns:
        raise ValueError(
            "OCM_CLASS column missing from scene samples; the OmniCloudMask "
            "mask must be computed before sampling (see add_ocm_mask())."
        )
    samples["valid_px"] = samples["OCM_CLASS"] == OCM_CLEAR

    with timed_step("Converting scene sample coordinates to lon/lat"):
        samples_gdf = gpd.GeoDataFrame(
            samples,
            geometry=gpd.points_from_xy(samples["x"], samples["y"]),
            crs=scene.odc.crs,
        ).to_crs("EPSG:4326")

    samples["lon"] = samples_gdf.geometry.x
    samples["lat"] = samples_gdf.geometry.y

    return samples


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
        props = item.properties
        item_dt = pd.to_datetime(props.get("datetime"))
        mgrs_tile = props.get("s2:mgrs_tile")
        output_path = output_path_for_scene(
            output_dir,
            item,
            scene_samples_subdir=config.get("scene_samples_subdir", "scene_samples"),
        )

        log(
            f"{scene_label} | "
            f"{item_dt} | tile={mgrs_tile} | item={item.id}"
        )

        if config.get("skip_existing", True) and output_path.exists():
            return f"Skipped existing scene output: {output_path}"

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
                f"Processing spatial batch {batch_id}/{len(field_batches)} "
                f"with {len(fields_batch):,} fields"
            )

            scene = retry_with_backoff(
                f"Loading Sentinel-2 scene assets batch {batch_id}/{len(field_batches)}",
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


def load_composite(items, crs, x_bounds, y_bounds, assets, resolution=10):
    if not items:
        return None

    log(f"Loading assets: {assets}")
    log(f"Target CRS: {crs}")
    log(f"Resolution: {resolution}")
    log(f"x bounds: {x_bounds}")
    log(f"y bounds: {y_bounds}")
    
    with timed_step("odc.stac.load"):
        ds = odc.stac.load(
            items,
            bands=assets,
            crs=crs,
            resolution=resolution,
            x=x_bounds,
            y=y_bounds,
            chunks={},
        )
    
    log(f"Loaded dataset dims: {dict(ds.sizes)}")
    ds = rename_s2_bands(ds)
    ds = harmonize_s2_timeseries_dataset(ds, items)

    for band in ("B4", "B3", "B8"):
        if band not in ds:
            raise ValueError(
                f"Cannot compute OmniCloudMask: missing required band '{band}'. "
                "Include B04, B03, and B08 in the config 'assets' list."
            )

    with timed_step("Computing OmniCloudMask cloud/shadow mask per scene"):
        n_time = ds.sizes["time"]
        clear_stack = np.empty((n_time, ds.sizes["y"], ds.sizes["x"]), dtype=bool)

        for t in range(n_time):
            clear_t, _ = ocm_clear_mask(
                ds["B4"].isel(time=t).values,
                ds["B3"].isel(time=t).values,
                ds["B8"].isel(time=t).values,
            )
            clear_stack[t] = clear_t

        clear = xr.DataArray(
            clear_stack,
            dims=("time", "y", "x"),
            coords={"time": ds["time"], "y": ds["y"], "x": ds["x"]},
        )

    with timed_step("Applying OmniCloudMask"):
        band_names = [b for b in S2_REFLECTANCE_BANDS if b in ds]
        ds_masked = ds[band_names].where(clear)
        valid_obs_count = clear.sum(dim="time")
        total_obs_count = clear.count(dim="time")

    with timed_step("Computing median composite"):
        # Median composite across all images in the window.
        comp = ds_masked.median(dim="time", skipna=True)
        comp["valid_obs_count"] = valid_obs_count
        comp["total_obs_count"] = total_obs_count
        comp["valid_obs_fraction"] = valid_obs_count / total_obs_count

    with timed_step("Calculting indices"):
        comp["NDVI"] = (comp["B8"] - comp["B4"]) / (comp["B8"] + comp["B4"])

        comp["NDTI"] = (comp["B11"] - comp["B12"]) / (comp["B11"] + comp["B12"])

        comp["NDII"] = (comp["B8"] - comp["B11"]) / (comp["B8"] + comp["B11"])

        comp["BSI"] = (
            ((comp["B11"] + comp["B4"]) - (comp["B8"] + comp["B2"])) /
            ((comp["B11"] + comp["B4"]) + (comp["B8"] + comp["B2"]))
        )

        #comp["NDRE"] = (comp["B8A"] - comp["B5"]) / (comp["B8A"] + comp["B5"])

        comp["NBR"] = (comp["B8"] - comp["B12"]) / (comp["B8"] + comp["B12"])

    with timed_step("Computing composite into memory"):
        comp = comp.load()
    
    return comp


def sample_composite(
    comp,
    fields_proj,
    timestamp,
    field_id_col,
    max_pixels_per_date=100000,
    random_seed=1,
    ):
    
    transform = comp.odc.transform
    height = comp.sizes["y"]
    width = comp.sizes["x"]
    log(f"Sampling grid size: y={height}, x={width}")
    log(f"Composite CRS: {comp.odc.crs}")
    log(f"Composite transform: {transform}")

    field_lookup = dict(enumerate(fields_proj[field_id_col].astype(str), start=1))

    shapes = [
        (geom, idx)
        for idx, geom in enumerate(fields_proj.geometry, start=1)
        if geom is not None and not geom.is_empty
    ]

    with timed_step("Rasterizing field polygons"):
        field_raster = rasterio.features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="int32",
        )

    ys, xs = np.where(field_raster > 0)
    log(f"Candidate pixels inside polygons: {len(xs):,}")

    if len(xs) == 0:
        return pd.DataFrame()

    if len(xs) > max_pixels_per_date:
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(xs), size=max_pixels_per_date, replace=False)
        ys = ys[keep]
        xs = xs[keep]
        log(f"Sampled down to max_pixels_per_date={max_pixels_per_date:,}")

    rows = []
    sample_bands = ["B2", "B3", "B4", "B8",  "B11", "B12",
                    "NDII", "NDTI", "NDVI", "BSI", "NBR",
                    "valid_obs_count", "total_obs_count", "valid_obs_fraction"]

    sample_bands = [b for b in sample_bands if b in comp]
    
    with timed_step("Materializing band arrays for sampling"):
        band_arrays = {
            band: comp[band].values
            for band in sample_bands
        }
    x_coords = comp.x.values[xs]
    y_coords = comp.y.values[ys]

    with timed_step("Building sample rows"):
        for i in range(len(xs)):
            row = {
                "task_code": field_lookup[field_raster[ys[i], xs[i]]],
                "timestamp": timestamp.date().isoformat(),
                "point_id": f"{field_lookup[field_raster[ys[i], xs[i]]]}_{int(xs[i])}_{int(ys[i])}",
                "x": float(x_coords[i]),
                "y": float(y_coords[i]),
            }

            for band in sample_bands:
                row[band] = band_arrays[band][ys[i], xs[i]]

            rows.append(row)

    samples = pd.DataFrame(rows)
    samples = samples.replace([np.inf, -np.inf], np.nan)

    with timed_step("Converting sample coordinates to lon/lat"):
        samples_gdf = gpd.GeoDataFrame(
            samples,
            geometry=gpd.points_from_xy(samples["x"], samples["y"]),
            crs=comp.odc.crs,
        ).to_crs("EPSG:4326")

    samples["lon"] = samples_gdf.geometry.x
    samples["lat"] = samples_gdf.geometry.y

    return samples


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = config["start_date"]
    end_date = config["end_date"]
    interval_days = config["interval_days"]
    
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
            max_cloud_cover=config["max_cloud_cover"],
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

    target_crs = estimate_utm_crs(fields_4326)
    fields_proj = fields.to_crs(target_crs)

    minx, miny, maxx, maxy = fields_proj.total_bounds
    x_bounds = (minx, maxx)
    y_bounds = (miny,maxy)

    for start, end in date_windows(start_date, end_date, interval_days):
        log(f"Processing window {start.date()} to {end.date()}")
        #print(f"Processing {start.date()} to {end.date()}")

        output_path = output_dir / f"{config['output_prefix']}_{start.date()}.csv"

        if config.get("skip_existing", True) and output_path.exists():
            log(f"Skipping existing output: {output_path.name}")
            continue
        
        with timed_step("Searching Sentinel-2 items"):
            items = search_s2_items(
                aoi_4326, 
                start, 
                end, 
                max_cloud_cover=config["max_cloud_cover"],
                bbox=search_bbox
                )

        log(f"Found {len(items)} Sentinel-2 items")
        
        if not items:
            #print("  No Sentinel-2 items found.")
            log("Skippeing window: no items found")
            continue
        
        with timed_step("Loading S2 bands, applying OmniCloudMask, and building composite"):
            comp = retry_with_backoff(
                "Loading/compositing Sentinel-2 assets",
                lambda: load_composite(
                    items,
                    target_crs,
                    x_bounds,
                    y_bounds,
                    assets=config["assets"],
                    resolution=config["resolution"],
                ),
                retries=config.get("download_retries", 3),
                base_sleep=config.get("retry_base_sleep", 10),
            )
            
        if comp is None:
            log("Skipping window: composite is None")
            continue
        
        log(f"Composite grid: y={comp.sizes.get('y')}, x={comp.sizes.get('x')}")
        
        with timed_step("Sampling composite pixels inside polygons"):
          
            samples = sample_composite(comp, fields_proj, start, 
                                       field_id_col=config["field_id_col"], 
                                       max_pixels_per_date=config["max_pixels_per_date"],
                                       random_seed=config["random_seed"])

        if samples.empty:
            log("Skipping window: no samples generated")
            continue

        #output_path = output_dir / f"{config['output_prefix']}_{start.date()}.csv"
        
        with timed_step(f"Writing CSV: {output_path.name}"):
            samples.to_csv(output_path, index=False)
        
        log(f"Wrote {len(samples):,} rows to {output_path}")


if __name__ == "__main__":
    main()
