from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


DEFAULT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDEX_DEPENDENCIES = {
    "NDVI": ["B8", "B4"],
    "BSI": ["B11", "B4", "B8", "B2"],
    "NDTI": ["B11", "B12"],
    "NDII": ["B8", "B11"],
    "NBR": ["B8", "B12"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build pixel-level and field-level mosaics from scene sample Parquet files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing scene sample Parquet files, e.g. data/planetary_computer_samples/scene_samples_batched",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where mosaic outputs will be written.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument(
        "--method",
        choices=["median", "quality"],
        default="median",
        help="median = median across valid observations; quality = choose observation by quality band.",
    )
    parser.add_argument(
        "--quality-band",
        default="NDVI",
        help="Band/index used when --method quality. Can be raw band or NDVI/BSI/NDTI/NDII/NBR.",
    )
    parser.add_argument(
        "--quality-direction",
        choices=["max", "min"],
        default="max",
        help="For quality mosaic, choose max or min quality-band value.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=DEFAULT_BANDS,
        help="Raw bands to mosaic.",
    )
    parser.add_argument(
        "--keep-coords",
        action="store_true",
        help="Keep lon/lat/x/y in the pixel mosaic using the first available coordinate per point/window.",
    )
    return parser.parse_args()


def safe_divide(num, den):
    return np.where(den != 0, num / den, np.nan)


def add_indices(df):
    if {"B8", "B4"}.issubset(df.columns):
        df["NDVI"] = safe_divide(df["B8"] - df["B4"], df["B8"] + df["B4"])

    if {"B11", "B4", "B8", "B2"}.issubset(df.columns):
        df["BSI"] = safe_divide(
            (df["B11"] + df["B4"]) - (df["B8"] + df["B2"]),
            (df["B11"] + df["B4"]) + (df["B8"] + df["B2"]),
        )

    if {"B11", "B12"}.issubset(df.columns):
        df["NDTI"] = safe_divide(df["B11"] - df["B12"], df["B11"] + df["B12"])

    if {"B8", "B11"}.issubset(df.columns):
        df["NDII"] = safe_divide(df["B8"] - df["B11"], df["B8"] + df["B11"])

    if {"B8", "B12"}.issubset(df.columns):
        df["NBR"] = safe_divide(df["B8"] - df["B12"], df["B8"] + df["B12"])

    return df


def required_columns(bands, method, quality_band, keep_coords):
    cols = {"field_id", "point_id", "datetime", "valid_px", "SCL"}
    cols.update(bands)

    if method == "quality":
        if quality_band in INDEX_DEPENDENCIES:
            cols.update(INDEX_DEPENDENCIES[quality_band])
        else:
            cols.add(quality_band)

    if keep_coords:
        cols.update(["x", "y", "lon", "lat"])

    return list(cols)


def read_scene_samples(input_dir, columns, start_date=None, end_date=None):
    paths = sorted(Path(input_dir).rglob("*.parquet"))

    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {input_dir}")

    parts = []

    for path in paths:
        df = pd.read_parquet(path, columns=columns)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

        if start_date is not None:
            df = df[df["datetime"] >= pd.Timestamp(start_date, tz="UTC")]

        if end_date is not None:
            df = df[df["datetime"] <= pd.Timestamp(end_date, tz="UTC")]

        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=columns)

    return pd.concat(parts, ignore_index=True)


def add_window_start(df, start_date, window_days):
    anchor = pd.Timestamp(start_date, tz="UTC")
    elapsed_days = (df["datetime"] - anchor).dt.total_seconds() / 86400.0
    window_index = np.floor(elapsed_days / window_days).astype("int64")
    df["window_start"] = anchor + pd.to_timedelta(window_index * window_days, unit="D")
    df["window_end"] = df["window_start"] + pd.to_timedelta(window_days, unit="D")
    return df


def median_pixel_mosaic(df, bands, keep_coords=False):
    group_cols = ["field_id", "point_id", "window_start", "window_end"]
    agg = {band: (band, "median") for band in bands if band in df.columns}
    agg["obs_count"] = ("datetime", "nunique")
    agg["valid_sample_count"] = ("valid_px", "sum")

    if keep_coords:
        for col in ["x", "y", "lon", "lat"]:
            if col in df.columns:
                agg[col] = (col, "first")

    return df.groupby(group_cols).agg(**agg).reset_index()


def quality_pixel_mosaic(df, bands, quality_band, quality_direction, keep_coords=False):
    if quality_band not in df.columns:
        raise ValueError(f"Quality band/index '{quality_band}' is not available.")

    sort_ascending = quality_direction == "min"
    group_cols = ["field_id", "point_id", "window_start", "window_end"]
    sorted_df = df.sort_values(group_cols + [quality_band], ascending=[True, True, True, True, sort_ascending])
    best = sorted_df.groupby(group_cols, as_index=False).tail(1).copy()

    keep_cols = group_cols + [band for band in bands if band in best.columns]
    keep_cols += ["datetime", quality_band, "SCL", "valid_px"]

    if keep_coords:
        keep_cols += [col for col in ["x", "y", "lon", "lat"] if col in best.columns]

    best = best[keep_cols].rename(
        columns={
            "datetime": "selected_datetime",
            quality_band: f"{quality_band}_quality_value",
        }
    )

    obs_count = (
        df.groupby(group_cols)
        .agg(obs_count=("datetime", "nunique"), valid_sample_count=("valid_px", "sum"))
        .reset_index()
    )

    return best.merge(obs_count, on=group_cols, how="left")


def summarize_fields(pixel_mosaic, bands):
    pixel_mosaic = add_indices(pixel_mosaic.copy())

    metrics = [band for band in bands if band in pixel_mosaic.columns]
    metrics += [idx for idx in ["NDVI", "BSI", "NDTI", "NDII", "NBR"] if idx in pixel_mosaic.columns]

    group_cols = ["field_id", "window_start", "window_end"]
    agg = {
        "pixel_count": ("point_id", "nunique"),
        "obs_count_median": ("obs_count", "median"),
        "obs_count_min": ("obs_count", "min"),
    }

    for metric in metrics:
        agg[f"{metric}_median"] = (metric, "median")
        agg[f"{metric}_p10"] = (metric, lambda x: x.quantile(0.10))
        agg[f"{metric}_p90"] = (metric, lambda x: x.quantile(0.90))
        agg[f"{metric}_std"] = (metric, "std")

    summary = pixel_mosaic.groupby(group_cols).agg(**agg).reset_index()

    if {"BSI", "NDVI"}.issubset(pixel_mosaic.columns):
        state = pixel_mosaic.copy()
        state["bare_soil_px"] = (state["BSI"] - state["NDVI"] > 0) & (state["BSI"] > 0)
        state["green_px"] = (state["NDVI"] > state["BSI"]) & (state["NDVI"] > 0.2)
        state["mixed_px"] = ~(state["bare_soil_px"] | state["green_px"])

        state_summary = (
            state.groupby(group_cols)
            .agg(
                bare_soil_fraction=("bare_soil_px", "mean"),
                green_fraction=("green_px", "mean"),
                mixed_fraction=("mixed_px", "mean"),
            )
            .reset_index()
        )

        summary = summary.merge(state_summary, on=group_cols, how="left")

    return summary


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = args.start_date
    end_date = args.end_date

    if start_date is None:
        raise ValueError("--start-date is required so windows have a stable anchor.")

    cols = required_columns(
        bands=args.bands,
        method=args.method,
        quality_band=args.quality_band,
        keep_coords=args.keep_coords,
    )

    samples = read_scene_samples(
        input_dir=input_dir,
        columns=cols,
        start_date=start_date,
        end_date=end_date,
    )

    if samples.empty:
        raise ValueError("No scene samples found after date filtering.")

    samples = samples[samples["valid_px"] == True].copy()
    samples = add_indices(samples)
    samples = add_window_start(samples, start_date, args.window_days)

    if args.method == "median":
        pixel_mosaic = median_pixel_mosaic(samples, args.bands, keep_coords=args.keep_coords)
    else:
        pixel_mosaic = quality_pixel_mosaic(
            samples,
            bands=args.bands,
            quality_band=args.quality_band,
            quality_direction=args.quality_direction,
            keep_coords=args.keep_coords,
        )

    field_summary = summarize_fields(pixel_mosaic, args.bands)

    suffix = args.method
    if args.method == "quality":
        suffix = f"quality_{args.quality_direction}_{args.quality_band}"

    pixel_path = output_dir / f"pixel_mosaic_{suffix}_{args.window_days}d.parquet"
    summary_path = output_dir / f"field_summary_{suffix}_{args.window_days}d.parquet"
    summary_csv_path = output_dir / f"field_summary_{suffix}_{args.window_days}d.csv"
    metadata_path = output_dir / f"mosaic_metadata_{suffix}_{args.window_days}d.json"

    pixel_mosaic.to_parquet(pixel_path, index=False)
    field_summary.to_parquet(summary_path, index=False)
    field_summary.to_csv(summary_csv_path, index=False)

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "window_days": args.window_days,
        "method": args.method,
        "quality_band": args.quality_band if args.method == "quality" else None,
        "quality_direction": args.quality_direction if args.method == "quality" else None,
        "bands": args.bands,
        "scene_sample_rows_after_filter": int(len(samples)),
        "pixel_mosaic_rows": int(len(pixel_mosaic)),
        "field_summary_rows": int(len(field_summary)),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Wrote {pixel_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
