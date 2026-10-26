"""The float split for the QC paper, fixed once and saved.

Floats with a documented real error are forced into a separate "known-bad" test group, so the
model never learns from them and they can be scored as real cases: every float whose T, S,
P or oxygen exclusion-list (greylist) window reaches our data, plus 2902209 (salinity 14.5 at
2000 dbar under flags 1-2, flagged by the oxygen model's variance and not on the greylist). The remaining
floats are split by float into train / validation / clean test. The injection benchmark uses
only the clean test floats; the known-bad group is never corrupted further.

Writes tables/split.json.  Run from the project root:
    python -m paper_qc.split
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from paper_qc.lib import data as odata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables", "split.json")
GREYLIST_OVERLAP = os.path.join(HERE, "tables", "greylist_overlap.csv")
FOUND_BY_MODEL = ["2902209"]
SEED = 2026
TEST_SIZE, VAL_SIZE = 0.20, 0.15


def known_bad():
    g = pd.read_csv(GREYLIST_OVERLAP, dtype={"float": str})
    return sorted(set(g["float"].str.strip()) | set(FOUND_BY_MODEL))


def make(platforms):
    platforms = np.asarray(sorted(set(platforms)))
    bad = [f for f in known_bad() if f in set(platforms)]
    rest = platforms[~np.isin(platforms, bad)]
    idx = np.arange(len(rest))
    tv, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=SEED)
                  .split(idx, groups=rest))
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED)
                  .split(tv, groups=rest[tv]))
    return {"seed": SEED, "train": sorted(rest[tv][tr]), "val": sorted(rest[tv][va]),
            "test_clean": sorted(rest[te]), "test_known_bad": bad}


def load():
    with open(OUT) as f:
        return json.load(f)


def main():
    df = odata.load(verbose=False)
    df["PLATFORM"] = df.PLATFORM.astype(str).str.strip()
    s = make(df.PLATFORM.unique())
    groups = {f: k for k in ("train", "val", "test_clean", "test_known_bad") for f in s[k]}
    lv = df.PLATFORM.map(groups).value_counts()
    s["levels"] = {k: int(lv.get(k, 0)) for k in ("train", "val", "test_clean", "test_known_bad")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(s, f, indent=1)
    for k in ("train", "val", "test_clean", "test_known_bad"):
        print(f"  {k:<15} {len(s[k]):>4} floats  {s['levels'][k]:>10,} levels")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
