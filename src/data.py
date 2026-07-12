"""
SECOM dataset consists of two files:
- secom.data         : feature matrix, whitespace-separated, missing = 'NaN'
- secom_labels.data  : one label + timestamp per row

These functions load and merge both files.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_features(path = None):
    path = path or RAW_DIR / "secom.data"
    features = pd.read_csv(path, sep=r"\s+", header=None, na_values="NaN")
    features.columns = [f"sensor_{i}" for i in range(features.shape[1])]        # make the column names sensor_1 sensor_2...
    return features


def load_labels(path = None):
    # original label: -1 = pass, 1 = fail
    # we recode as 1 = fail, 0 = pass, easier for metrics 
    path = path or RAW_DIR / "secom_labels.data"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            # split on the FIRST space only: everything after is the timestamp
            label_str, ts_str = line.split(" ", 1)
            ts_str = ts_str.strip('"')          # remove the surrounding quotes
            rows.append((int(label_str), ts_str))
    labels = pd.DataFrame(rows, columns=["label", "timestamp"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")
    labels["fail"] = (labels["label"] == 1).astype(int)
    return labels[["timestamp", "fail"]]


def load_secom(raw_dir = None):
    # merge data with labels 
    # return columns: "timestamp, label, sensor_0, sensor_1 ..."
    feat_path = (raw_dir / "secom.data") if raw_dir else None
    label_path = (raw_dir / "secom_labels.data") if raw_dir else None

    features = load_features(feat_path)
    labels = load_labels(label_path)

    if len(features) != len(labels):
        raise ValueError(
            f"Row count mismatch: {len(features)} feature rows vs "
            f"{len(labels)} label rows."
        )

    df = pd.concat(
        [labels.reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    return df