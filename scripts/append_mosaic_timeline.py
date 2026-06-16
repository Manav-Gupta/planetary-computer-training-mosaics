from pathlib import Path
import argparse
import json
import pandas as pd


DEFAULT_KEY_COLUMNS = ["field_id", "window_start", "window_end"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Append/merge mosaic timeline outputs from multiple runs. "
            "Use this when extending an existing Sentinel-2 mosaic time series "
            "with a newer date range."
        )
    )
    parser.add_argument(
        "--existing-summary",
        required=True,
        help="Existing field summary CSV or Parquet.",
    )
    parser.add_argument(
        "--new-summary",
        required=True,
        help="New field summary CSV or Parquet to append.",
    )
    parser.add_argument(
        "--output-summary",
        required=True,
        help="Merged output summary path. Extension controls format: .csv or .parquet.",
    )
    parser.add_argument(
        "--existing-pixel-mosaic",
        default=None,
        help="Optional existing pixel mosaic Parquet/CSV.",
    )
    parser.add_argument(
        "--new-pixel-mosaic",
        default=None,
        help="Optional new pixel mosaic Parquet/CSV.",
    )
    parser.add_argument(
        "--output-pixel-mosaic",
        default=None,
        help="Optional merged pixel mosaic output path.",
    )
    parser.add_argument(
        "--key-columns",
        nargs="+",
        default=DEFAULT_KEY_COLUMNS,
        help="Columns used to identify duplicate windows.",
    )
    parser.add_argument(
        "--keep",
        choices=["new", "existing"],
        default="new",
        help="Which row to keep when existing and new files overlap.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Optional JSON metadata path for merge details.",
    )
    return parser.parse_args()


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported input format for {path}. Use .csv or .parquet.")


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False)
        return

    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return

    raise ValueError(f"Unsupported output format for {path}. Use .csv or .parquet.")


def normalize_time_columns(df):
    df = df.copy()

    for col in ["window_start", "window_end", "datetime", "selected_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return df


def merge_timelines(existing, new, key_columns, keep):
    existing = normalize_time_columns(existing)
    new = normalize_time_columns(new)

    missing_existing = [col for col in key_columns if col not in existing.columns]
    missing_new = [col for col in key_columns if col not in new.columns]

    if missing_existing:
        raise ValueError(f"Existing table is missing key columns: {missing_existing}")

    if missing_new:
        raise ValueError(f"New table is missing key columns: {missing_new}")

    existing["_append_source"] = "existing"
    new["_append_source"] = "new"

    if keep == "new":
        combined = pd.concat([existing, new], ignore_index=True, sort=False)
        keep_position = "last"
    else:
        combined = pd.concat([new, existing], ignore_index=True, sort=False)
        keep_position = "last"

    before = len(combined)
    combined = combined.drop_duplicates(key_columns, keep=keep_position)
    dropped_duplicates = before - len(combined)

    sort_cols = [col for col in key_columns if col in combined.columns]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    if "_append_source" in combined.columns:
        combined = combined.drop(columns=["_append_source"])

    return combined, dropped_duplicates


def table_summary(df):
    out = {"rows": int(len(df))}

    if "field_id" in df.columns:
        out["field_count"] = int(df["field_id"].nunique())

    if "window_start" in df.columns:
        dates = pd.to_datetime(df["window_start"], errors="coerce", utc=True)
        out["first_window_start"] = None if dates.dropna().empty else str(dates.min())
        out["last_window_start"] = None if dates.dropna().empty else str(dates.max())

    return out


def main():
    args = parse_args()

    existing_summary = read_table(args.existing_summary)
    new_summary = read_table(args.new_summary)

    merged_summary, summary_duplicates = merge_timelines(
        existing_summary,
        new_summary,
        key_columns=args.key_columns,
        keep=args.keep,
    )
    write_table(merged_summary, args.output_summary)

    metadata = {
        "existing_summary": args.existing_summary,
        "new_summary": args.new_summary,
        "output_summary": args.output_summary,
        "key_columns": args.key_columns,
        "keep": args.keep,
        "summary_duplicates_removed": int(summary_duplicates),
        "existing_summary_stats": table_summary(existing_summary),
        "new_summary_stats": table_summary(new_summary),
        "merged_summary_stats": table_summary(merged_summary),
    }

    has_pixel_inputs = args.existing_pixel_mosaic or args.new_pixel_mosaic or args.output_pixel_mosaic
    if has_pixel_inputs:
        if not (args.existing_pixel_mosaic and args.new_pixel_mosaic and args.output_pixel_mosaic):
            raise ValueError(
                "To merge pixel mosaics, provide --existing-pixel-mosaic, "
                "--new-pixel-mosaic, and --output-pixel-mosaic."
            )

        pixel_key_columns = args.key_columns + ["point_id"]
        existing_pixel = read_table(args.existing_pixel_mosaic)
        new_pixel = read_table(args.new_pixel_mosaic)
        merged_pixel, pixel_duplicates = merge_timelines(
            existing_pixel,
            new_pixel,
            key_columns=pixel_key_columns,
            keep=args.keep,
        )
        write_table(merged_pixel, args.output_pixel_mosaic)

        metadata.update(
            {
                "existing_pixel_mosaic": args.existing_pixel_mosaic,
                "new_pixel_mosaic": args.new_pixel_mosaic,
                "output_pixel_mosaic": args.output_pixel_mosaic,
                "pixel_key_columns": pixel_key_columns,
                "pixel_duplicates_removed": int(pixel_duplicates),
                "existing_pixel_stats": table_summary(existing_pixel),
                "new_pixel_stats": table_summary(new_pixel),
                "merged_pixel_stats": table_summary(merged_pixel),
            }
        )

    metadata_path = (
        Path(args.metadata_path)
        if args.metadata_path is not None
        else Path(args.output_summary).with_suffix(".append_metadata.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Wrote merged summary: {args.output_summary}")
    print(f"Wrote metadata: {metadata_path}")

    if args.output_pixel_mosaic:
        print(f"Wrote merged pixel mosaic: {args.output_pixel_mosaic}")


if __name__ == "__main__":
    main()
