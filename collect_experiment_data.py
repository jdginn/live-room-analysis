import os
import argparse
import json
import pandas as pd


def find_experiment_folders(root="."):
    """Recursively find experiment folders containing both '*results.json' and 'summary.json'."""
    experiment_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        print("Checking:", dirpath)
        has_params_json = any(
            f.endswith(".json") and "summary" not in f for f in filenames
        )
        has_summary_json = "summary.json" in filenames
        if has_params_json and has_summary_json:
            experiment_dirs.append(dirpath)
    return experiment_dirs


def load_experiment_data(expdir):
    # Find parameter json
    param_files = [
        f for f in os.listdir(expdir) if f.endswith(".json") and "summary" not in f
    ]
    if not param_files:
        return None  # No param file found
    param_path = os.path.join(expdir, param_files[0])

    # Load parameters
    with open(param_path, "r") as f:
        params = json.load(f)

    # Load summary (results)
    summary_path = os.path.join(expdir, "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Flatten parameters and extract key performance metrics
    row = params.copy()  # base features (l_*, r_*)

    # Derived parameters
    row["avg_angle"] = (row["l_reflector_angle"] + row["r_reflector_angle"]) / 2
    row["angle_diff"] = abs(row["l_reflector_angle"] - row["r_reflector_angle"])
    row["total_depth"] = row["l_reflector_depth"] + row["r_reflector_depth"]
    row["avg_depth"] = row["total_depth"] / 2
    row["total_reflectors"] = row["l_num_reflectors"] + row["r_num_reflectors"]
    row["num_diff"] = abs(row["l_num_reflectors"] - row["r_num_reflectors"])
    row["spacing_asym"] = abs(row["l_finish_offset"] - row["r_finish_offset"]) + abs(
        row["l_start_offset"] - row["r_start_offset"]
    )

    # Helper for metric extraction (mean across standard frequencies)
    def mean_metric(results, key, metric):
        freq_keys = ["1000", "2000", "4000"]
        try:
            values = [
                results[key]["frequencies"][f][metric]
                for f in freq_keys
                if f in results[key]["frequencies"]
            ]
            return sum(values) / len(values) if values else None
        except Exception:
            return None

    def mean(items: list[float | None]) -> float:
        itemsNotNone = [i for i in items if i is not None]
        return sum(itemsNotNone) / len(items)

    results = summary.get("results", {})
    row["DrumDeadT30"] = mean(
        [
            mean_metric(results, "window_drums_OH_back_cardioid", "t30_ms"),
            mean_metric(results, "window_drums_OH_back_cardioid", "t30_ms"),
            mean_metric(results, "window_drums_OH_omni", "t30_ms"),
        ]
    )
    row["DrumLiveT30"] = mean(
        [
            mean_metric(results, "door_drums_OH_cardioid", "t30_ms"),
            mean_metric(results, "door_drums_OH_cardioid", "t30_ms"),
            mean_metric(results, "window_drums_far_omni", "t30_ms"),
        ]
    )
    row["DrumDiffusion"] = mean(
        [
            mean_metric(results, "window_drums_OH_back_cardioid", "echo_density_score"),
            mean_metric(results, "window_drums_OH_omni", "echo_density_score"),
            mean_metric(results, "door_drums_OH_cardioid", "echo_density_score"),
            mean_metric(results, "door_drums_OH_omni", "echo_density_score"),
            mean_metric(results, "window_drums_far_omni", "echo_density_score"),
        ]
    )
    row["VocalCenterToWindowT30"] = mean_metric(
        results, "vox_center_to_window", "t30_ms"
    )
    row["VocalDoorToWindowT30"] = mean_metric(results, "vox_door_to_window", "t30_ms")
    row["VocalDeadT30"] = mean_metric(results, "vox_window_to_door", "t30_ms")
    row["VocalDiffusion"] = mean_metric(
        results, "vox_center_to_window", "echo_density_score"
    )
    row["experiment"] = os.path.basename(expdir)
    return row


def collect_all_experiments(root="."):
    experiment_dirs = find_experiment_folders(root)
    rows = []
    for expdir in experiment_dirs:
        row = load_experiment_data(expdir)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect experiment parameter and result JSONs into a DataFrame."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to search for experiment JSONs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_metrics.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()

    df = collect_all_experiments(args.root)
    print(f"Loaded {len(df)} experiments from {args.root}.")
    print(df.head())
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
