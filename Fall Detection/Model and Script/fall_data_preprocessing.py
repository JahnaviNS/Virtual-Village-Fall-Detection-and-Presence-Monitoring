"""
fall_data_preprocessing.py

Purpose:
    Convert raw radar JSON fall data into a model-ready CSV using the same
    cleaning, labeling, room mapping, sessionization, and feature engineering
    logic from Fall_data_eda.ipynb, excluding model building.

Example usage:
    python fall_data_preprocessing.py \
        --data fall_data_2.json \
        --labels fall_labels_2.json \
        --output df_model_single_fall.csv

For multiple data/label pairs:
    python fall_data_preprocessing.py \
        --data fall_data_2.json fall_data_3.json \
        --labels fall_labels_2.json fall_labels_3.json \
        --output df_model.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PARSED_COLS = ["x1", "y1", "z1", "x2", "y2", "z2", "SNS", "target_count"]

RADAR_MAP = {
    "12E8FE50000400192600000001": "Bedroom",
    "12E8FE50000400192600000002": "Kitchen",
    "12E8FE50000400192600000003": "Living Room",
    "12E8FE50000400192600000004": "Office",
    "12E8FE50000400192600000005": "Sunroom",
    "12E8FE50000400192600000006": "Hallway",
}

# Final CSV schema. These are the only columns saved to the model-ready output.
# Raw/helper columns such as radarUuid, parsedData, dataType, x2, y2, z2, SNS,
# target_count, and time_gap are intentionally excluded.
MODEL_OUTPUT_COLS = [
    "createTime",
    "room",
    "session_id",
    "fall_label",
    "x1",
    "y1",
    "z1",
    "z1_lag1",
    "z1_lag2",
    "z1_lag3",
    "z1_delta1",
    "z1_delta3",
    "z1_roll3_mean",
    "z1_roll3_std",
    "z1_drop_from_lag3",
    "z1_lead_mean5",
    "z1_lead_std3",
    "z1_lead8",
    "z1_is_low",
]

RAW_COLUMNS_EXCLUDED_FROM_OUTPUT = [
    "radarUuid",
    "data",
    "dataType",
    "parsedData",
    "target_count",
    "SNS",
    "x2",
    "y2",
    "z2",
    "time_gap",
]


def load_radar_json(data_file: Path) -> pd.DataFrame:
    """Load radar JSON and return a dataframe of raw records."""
    with open(data_file, "r", encoding="utf-8") as f:
        fall_data = json.load(f)

    try:
        records = fall_data["data"]["result"]
    except KeyError as exc:
        raise KeyError(
            "Expected JSON structure: fall_data['data']['result']. "
            "Please check the input radar JSON format."
        ) from exc

    return pd.DataFrame(records)


def expand_parsed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Expand parsedData into x1, y1, z1, x2, y2, z2, SNS, target_count."""
    if "parsedData" not in df.columns:
        raise ValueError("Input radar dataframe must contain a 'parsedData' column.")

    df = df.copy()

    # Keep only rows where parsedData exists and has the expected length.
    df = df[df["parsedData"].apply(lambda x: isinstance(x, list) and len(x) == len(PARSED_COLS))].copy()

    # Expand list into separate numeric columns.
    df[PARSED_COLS] = pd.DataFrame(df["parsedData"].tolist(), index=df.index)

    # Keep only rows where parsedData is not all zeros.
    df = df[df["parsedData"].apply(lambda x: any(v != 0 for v in x))].copy()

    return df


def add_fall_labels(df: pd.DataFrame, label_file: Optional[Path]) -> pd.DataFrame:
    """
    Add fall_label using exact match on (radarUuid, createTime) against
    (radarUuid, timestamp_start) from the label JSON.

    If no label_file is provided, fall_label is set to 0 for all rows.
    """
    df = df.copy()
    df["fall_label"] = 0

    df["createTime"] = df["createTime"].astype(str)
    df["radarUuid"] = df["radarUuid"].astype(str)

    if label_file is None:
        return df

    with open(label_file, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    falls = label_data.get("falls", [])
    labels_df = pd.DataFrame(falls)

    if labels_df.empty:
        return df

    required_label_cols = {"radarUuid", "timestamp_start"}
    missing = required_label_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"Label file is missing required columns: {missing}")

    labels_df["timestamp_start"] = labels_df["timestamp_start"].astype(str)
    labels_df["radarUuid"] = labels_df["radarUuid"].astype(str)

    fall_keys = set(zip(labels_df["radarUuid"], labels_df["timestamp_start"]))

    df["fall_label"] = df.apply(
        lambda row: 1 if (row["radarUuid"], row["createTime"]) in fall_keys else 0,
        axis=1,
    )

    return df


def add_room_mapping(df: pd.DataFrame, radar_map: Dict[str, str] = RADAR_MAP) -> pd.DataFrame:
    """Map radarUuid values to room names."""
    df = df.copy()
    df["radarUuid"] = df["radarUuid"].astype(str)
    df["room"] = df["radarUuid"].map(radar_map).fillna("Unknown")
    return df


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the same modeling columns used in the EDA notebook before feature engineering.
    The EDA notebook drops: radarUuid, data, dataType, parsedData, target_count,
    SNS, x2, y2, z2.
    """
    df = df.copy()

    columns_to_drop = [
        "radarUuid",
        "data",
        "dataType",
        "parsedData",
        "target_count",
        "SNS",
        "x2",
        "y2",
        "z2",
    ]

    existing_drop_cols = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    required_cols = {"createTime", "x1", "y1", "z1", "room", "fall_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Prepared dataframe is missing required columns: {missing}")

    return df


def engineer_features(df: pd.DataFrame, session_gap_seconds: int = 60) -> pd.DataFrame:
    """
    Engineer the same z1-based features from Fall_data_eda.ipynb.

    Session logic:
        Within each room, if the time gap between rows is greater than 60 seconds,
        a new session starts. This prevents lag/lead values from bleeding across
        unrelated recordings.
    """
    df = df.copy()

    df["createTime"] = pd.to_datetime(df["createTime"], format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["createTime"])

    # Convert main coordinate columns to numeric.
    for col in ["x1", "y1", "z1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x1", "y1", "z1", "room", "fall_label"])

    # Sort by room and time, same as the EDA notebook.
    df = df.sort_values(["room", "createTime"]).reset_index(drop=True)

    # Session boundaries per room.
    df["time_gap"] = df.groupby("room")["createTime"].diff().dt.total_seconds().fillna(0)
    df["session_id"] = df.groupby("room")["time_gap"].transform(
        lambda s: (s > session_gap_seconds).cumsum()
    )
    df = df.drop(columns=["time_gap"])

    all_sessions = []

    for (room, sid), grp in df.groupby(["room", "session_id"]):
        grp = grp.copy().reset_index(drop=True)
        z = grp["z1"]

        # Lags: previous z1 values.
        grp["z1_lag1"] = z.shift(1)
        grp["z1_lag2"] = z.shift(2)
        grp["z1_lag3"] = z.shift(3)

        # Rate of change / descent speed.
        grp["z1_delta1"] = z.diff(1)
        grp["z1_delta3"] = z.diff(3)

        # Pre-fall context.
        grp["z1_roll3_mean"] = z.shift(1).rolling(3).mean()
        grp["z1_roll3_std"] = z.shift(1).rolling(3).std()

        # Drop magnitude compared with 3 frames ago.
        grp["z1_drop_from_lag3"] = grp["z1_lag3"] - z

        # Lead features: post-event context.
        grp["z1_lead_mean5"] = z.iloc[::-1].rolling(5).mean().iloc[::-1].shift(-1)
        grp["z1_lead_std3"] = z.iloc[::-1].rolling(3).std().iloc[::-1].shift(-1)
        grp["z1_lead8"] = z.shift(-8)

        # Low-height indicator.
        grp["z1_is_low"] = (z < 0.3).astype(int)

        all_sessions.append(grp)

    if not all_sessions:
        raise ValueError("No sessions were created. Check whether the input file has valid non-zero parsedData rows.")

    df_model = pd.concat(all_sessions, ignore_index=True)
    return df_model


def process_file_pair(data_file: Path, label_file: Optional[Path]) -> pd.DataFrame:
    """Run full preprocessing pipeline for one radar JSON and optional label JSON."""
    print(f"Processing radar file: {data_file}")
    if label_file:
        print(f"Using label file:    {label_file}")
    else:
        print("No label file provided. fall_label will be 0 for all rows.")

    df = load_radar_json(data_file)
    raw_rows = df.shape[0]

    df = expand_parsed_data(df)
    nonzero_rows = df.shape[0]

    df = add_fall_labels(df, label_file)
    labeled_falls = int(df["fall_label"].sum())

    df = add_room_mapping(df)
    df = prepare_base_dataframe(df)
    df_model = engineer_features(df)

    print(f"Raw rows:            {raw_rows}")
    print(f"Non-zero rows kept:  {nonzero_rows}")
    print(f"Labeled fall rows:   {labeled_falls}")
    print(f"Model-ready rows:    {df_model.shape[0]}")
    print("-" * 50)

    return df_model


def save_outputs(df_model: pd.DataFrame, output_path: Path, drop_na: bool = False) -> None:
    """Save a clean model-ready dataframe to CSV using the strict final schema."""
    missing_output_cols = [col for col in MODEL_OUTPUT_COLS if col not in df_model.columns]
    if missing_output_cols:
        raise ValueError(f"Cannot save model-ready CSV. Missing columns: {missing_output_cols}")

    # Strictly keep only the final model columns. This removes all raw/helper columns.
    final_df = df_model[MODEL_OUTPUT_COLS].copy()

    if drop_na:
        final_df = final_df.dropna().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print("Preprocessing completed successfully.")
    print(f"CSV saved to:        {output_path}")
    print(f"Final CSV shape:     {final_df.shape}")
    print(f"Total fall labels:   {int(final_df['fall_label'].sum())}")
    print("\nFinal CSV columns:")
    print(list(final_df.columns))

    leaked_cols = [col for col in RAW_COLUMNS_EXCLUDED_FROM_OUTPUT if col in final_df.columns]
    if leaked_cols:
        raise AssertionError(f"Raw columns were not removed from final CSV: {leaked_cols}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess radar fall JSON files into a model-ready CSV."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="One or more radar JSON files, e.g., fall_data_2.json fall_data_3.json",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional matching label JSON files. Must match --data order if provided.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path, e.g., df_model.csv",
    )
    parser.add_argument(
        "--drop-na",
        action="store_true",
        help="Drop rows with NaN values from lag/lead features before saving. Use this if your model script expects complete feature rows only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_files = [Path(p) for p in args.data]
    label_files = [Path(p) for p in args.labels] if args.labels else [None] * len(data_files)

    if len(label_files) != len(data_files):
        raise ValueError("The number of --labels files must match the number of --data files.")

    all_dfs = []
    for data_file, label_file in zip(data_files, label_files):
        df_model = process_file_pair(data_file, label_file)
        all_dfs.append(df_model)

    combined_df_model = pd.concat(all_dfs, ignore_index=True)
    save_outputs(combined_df_model, Path(args.output), drop_na=args.drop_na)


if __name__ == "__main__":
    main()
