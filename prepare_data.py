"""Prepare train/test events for Checkpoint 3.

Run on its own: python prepare_data.py --split-seed 42
"""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/train_data.csv")
OUT = Path("data/processed")

# Time windows and split settings
INPUT_CUTOFF_DAYS = 2.0      # only use CDMs at least this far before TCA
LABEL_MAX_DAYS = 1.0         # final CDM must be within this window of TCA
HIGH_RISK = -6.0             # risk threshold (log scale) for the positive class
TEST_FRACTION = 0.20

MIRROR_URL = (
    "https://raw.githubusercontent.com/spmorillo/"
    "Satellite-Collision-Avoidance-Challenge/HEAD/data/train_data.zip"
)
ZIP_SHA256 = "68362fe5629cc80f17291f2d73f733bf4e922675e37b91a8ee79afadb46f3edc"


def ensure_data(raw=RAW):
    """Make sure the ESA training CSV exists, downloading and verifying if needed."""
    raw = Path(raw)
    if raw.exists():
        return raw

    raw.parent.mkdir(parents=True, exist_ok=True)

    # Sometimes people drop the CSV straight into the working directory
    local = Path("train_data.csv")
    if local.exists():
        local.rename(raw)
        print(f"Moved train_data.csv to {raw}")
        return raw

    zp = raw.parent / "train_data.zip"
    if not zp.exists():
        print(f"Downloading train_data.zip (~97 MB) from {MIRROR_URL}")
        urllib.request.urlretrieve(MIRROR_URL, zp)

    digest = hashlib.sha256(zp.read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        raise SystemExit(f"Checksum mismatch for {zp}: {digest}")

    with zipfile.ZipFile(zp) as z:
        z.extract("train_data.csv", raw.parent)
    print(f"Extracted {raw}")
    return raw


def build(split_seed, raw=RAW, out=OUT):
    df = pd.read_csv(ensure_data(raw)).sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    )

    # The last CDM for each event is the label source
    final = df.groupby("event_id").tail(1).set_index("event_id")
    earliest = df.groupby("event_id")["time_to_tca"].max()

    eligible = (final["time_to_tca"] <= LABEL_MAX_DAYS) & (earliest >= INPUT_CUTOFF_DAYS)
    labels = pd.DataFrame({
        "risk_final": final.risk[eligible],
        "mission_id": final.mission_id[eligible],
    })
    labels["high_risk"] = (labels.risk_final >= HIGH_RISK).astype(int)
    labels.index.name = "parent_event_id"

    # Inputs: only CDMs that arrive early enough, for eligible events
    inp = df[df.event_id.isin(labels.index) & (df.time_to_tca >= INPUT_CUTOFF_DAYS)].copy()
    inp["c_object_type"] = inp["c_object_type"].astype("category").cat.codes
    inp = inp.rename(columns={"event_id": "parent_event_id"})

    # Stratified split so high-risk events appear in both train and test
    rng = np.random.default_rng(split_seed)
    test_ids = []
    for _, g in labels.groupby("high_risk"):
        ids = g.index.to_numpy()
        n_test = int(round(len(ids) * TEST_FRACTION))
        test_ids.extend(rng.choice(ids, size=n_test, replace=False))
    is_test = labels.index.isin(test_ids)

    out.mkdir(parents=True, exist_ok=True)
    stats = {}

    for name, mask in [("train", ~is_test), ("test", is_test)]:
        lab = labels[mask]
        cdm = inp[inp.parent_event_id.isin(lab.index)].copy()
        cdm["latest_input_tta"] = cdm.groupby("parent_event_id")["time_to_tca"].transform("min")
        assert (cdm.time_to_tca >= INPUT_CUTOFF_DAYS).all()

        lab.to_csv(out / f"{name}_labels.csv")
        cdm.to_csv(out / f"{name}_cdms.csv", index=False)
        stats[name] = (len(lab), int(lab.high_risk.sum()), len(cdm))

    overlap = set(labels.index[~is_test]) & set(labels.index[is_test])
    assert not overlap

    print(f"eligible {eligible.sum()}, high-risk {labels.high_risk.sum()}, overlap {len(overlap)}")
    for k, (ne, nh, nr) in stats.items():
        print(f"{k:>5}: {ne} events ({nh} high-risk), {nr} CDM rows")
    return stats


if __name__ == "__main__":
    print("Preparing the training data...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    build(args.split_seed)
    print("Complete.")
