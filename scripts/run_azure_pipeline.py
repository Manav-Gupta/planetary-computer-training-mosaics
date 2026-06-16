from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the HALO Sentinel-2 download and mosaicing workflow."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the Planetary Computer/download JSON config.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["inventory", "download", "mosaic"],
        default=["inventory", "download", "mosaic"],
        help="Pipeline steps to run.",
    )
    parser.add_argument(
        "--mosaic-output-dir",
        default=None,
        help="Output folder for mosaic products. Defaults to <output_dir>/mosaics.",
    )
    parser.add_argument(
        "--fields-path",
        default=None,
        help="Override config fields_path. Useful for Azure ML mounted inputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config output_dir. Useful for Azure ML mounted outputs.",
    )
    parser.add_argument("--mosaic-window-days", type=int, default=14)
    parser.add_argument(
        "--mosaic-window-anchor-date",
        default=None,
        help=(
            "Stable date used to anchor mosaic windows. Defaults to config start_date. "
            "Use the original project start date when appending later runs."
        ),
    )
    parser.add_argument(
        "--mosaic-method",
        choices=["median", "quality"],
        default="median",
    )
    parser.add_argument("--quality-band", default="NDVI")
    parser.add_argument(
        "--quality-direction",
        choices=["max", "min"],
        default="max",
    )
    parser.add_argument(
        "--keep-coords",
        action="store_true",
        help="Keep coordinates in the pixel mosaic.",
    )
    parser.add_argument(
        "--mosaic-indices",
        nargs="+",
        default=None,
        help="Indices to calculate in mosaic summary, e.g. NDVI BSI NDTI NBR.",
    )
    parser.add_argument(
        "--mosaic-summary-metrics",
        nargs="+",
        choices=["median", "p10", "p90", "std", "mean", "min", "max"],
        default=None,
        help="Summary metrics to export, e.g. median p10 p90 std.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_command(command, cwd):
    print("Running:", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    if args.fields_path is not None:
        config["fields_path"] = args.fields_path

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp_file:
        json.dump(config, tmp_file, indent=2)
        runtime_config_path = Path(tmp_file.name)

    output_dir = Path(config["output_dir"])
    scene_samples_subdir = config.get("scene_samples_subdir", "scene_samples")
    scene_samples_dir = output_dir / scene_samples_subdir
    mosaic_output_dir = (
        Path(args.mosaic_output_dir)
        if args.mosaic_output_dir is not None
        else output_dir / "mosaics"
    )

    python = sys.executable

    if "inventory" in args.steps:
        run_command(
            [
                python,
                "download_s2_pc.py",
                "--config",
                str(runtime_config_path),
                "--stac-inventory-only",
            ],
            cwd=scripts_dir,
        )

    if "download" in args.steps:
        run_command(
            [
                python,
                "download_s2_pc.py",
                "--config",
                str(runtime_config_path),
                "--scene-samples",
            ],
            cwd=scripts_dir,
        )

    if "mosaic" in args.steps:
        command = [
            python,
            "mosaic_scene_samples.py",
            "--input-dir",
            str(scene_samples_dir),
            "--output-dir",
            str(mosaic_output_dir),
            "--start-date",
            config["start_date"],
            "--end-date",
            config["end_date"],
            "--window-days",
            str(args.mosaic_window_days),
            "--method",
            args.mosaic_method,
        ]

        if args.mosaic_window_anchor_date is not None:
            command.extend(["--window-anchor-date", args.mosaic_window_anchor_date])

        if args.mosaic_method == "quality":
            command.extend(
                [
                    "--quality-band",
                    args.quality_band,
                    "--quality-direction",
                    args.quality_direction,
                ]
            )

        if args.keep_coords:
            command.append("--keep-coords")

        if args.mosaic_indices:
            command.extend(["--indices", *args.mosaic_indices])

        if args.mosaic_summary_metrics:
            command.extend(["--summary-metrics", *args.mosaic_summary_metrics])

        run_command(command, cwd=scripts_dir)


if __name__ == "__main__":
    main()
