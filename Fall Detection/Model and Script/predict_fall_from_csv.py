import argparse
import pickle
import pandas as pd


def load_model_package(model_path):
    with open(model_path, "rb") as f:
        package = pickle.load(f)

    # Supports both:
    # 1. A package dict: {"model": ..., "threshold": ..., "feature_cols": ...}
    # 2. A raw sklearn/xgboost model saved directly
    if isinstance(package, dict):
        model = package.get("model")
        threshold = package.get("threshold", 0.35)
        feature_cols = package.get("feature_cols")
    else:
        model = package
        threshold = 0.35
        feature_cols = None

    if model is None:
        raise ValueError("Could not find model inside the pickle file.")

    return model, threshold, feature_cols


def main():
    parser = argparse.ArgumentParser(
        description="Run fall prediction using a preprocessed CSV and saved XGBoost model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV created by fall_data_preprocessing.py"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to saved model pickle file, for example xgboost_fall_detection_model.pkl"
    )

    parser.add_argument(
        "--output",
        default="fall_prediction_results.csv",
        help="Path to save frame-level prediction results"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional custom threshold. If not provided, script uses threshold saved in model package."
    )

    parser.add_argument(
        "--min-fall-frames",
        type=int,
        default=1,
        help="Minimum number of predicted fall frames required to classify the whole file as Fall Detected."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model, saved_threshold, feature_cols = load_model_package(args.model)

    threshold = args.threshold if args.threshold is not None else saved_threshold

    if feature_cols is None:
        feature_cols = [
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

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required feature columns: {missing_cols}")

    X = df[feature_cols]

    if hasattr(model, "predict_proba"):
        fall_prob = model.predict_proba(X)[:, 1]
    else:
        fall_prob = model.predict(X)

    df["fall_probability"] = fall_prob
    df["predicted_fall"] = (df["fall_probability"] >= threshold).astype(int)

    total_rows = len(df)
    fall_frames = int(df["predicted_fall"].sum())
    max_probability = float(df["fall_probability"].max())
    mean_probability = float(df["fall_probability"].mean())

    fall_detected = fall_frames >= args.min_fall_frames

    df.to_csv(args.output, index=False)

    print("\nFall Prediction Summary")
    print(f"Input CSV:              {args.input}")
    print(f"Model file:             {args.model}")
    print(f"Threshold used:         {threshold}")
    print(f"Total frames checked:   {total_rows}")
    print(f"Predicted fall frames:  {fall_frames}")
    print(f"Max fall probability:   {max_probability:.4f}")
    print(f"Mean fall probability:  {mean_probability:.4f}")
    print(f"Results saved to:       {args.output}")

    print("\nFinal Decision")

    if fall_detected:
        print("FALL DETECTED")
    else:
        print("NO FALL DETECTED")

    top_cols = []
    for col in [
        "createTime",
        "room",
        "session_id",
        "z1",
        "z1_drop_from_lag3",
        "z1_is_low",
        "fall_probability",
        "predicted_fall",
    ]:
        if col in df.columns:
            top_cols.append(col)

    print("\nTop 10 highest-risk frames")
    print("--------------------------------------------------")
    print(
        df.sort_values("fall_probability", ascending=False)
          .head(10)[top_cols]
          .to_string(index=False)
    )


if __name__ == "__main__":
    main()
