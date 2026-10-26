"""Write data_splits.json: the pre-registered train/test split of RCAEval cases.

    python scripts/make_splits.py           # writes data_splits.json
    python scripts/make_splits.py --check   # verifies the committed file is reproducible

Cases are grouped by (service, fault), so the instances of one fault (near
duplicates of each other) are always on the same side. Within each suite,
system and fault type, groups are ordered by a seeded hash and the first 40% go
to train, so every fault type appears on both sides and nobody hand-picks the
split. Development and fine-tuning use train; reported numbers use test.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_rcaeval import SUITES, case_index, load_labels  # noqa: E402

OUT = ROOT / "data_splits.json"
SEED = "rcaeval-split-v1"
TRAIN_FRACTION = 0.4


def build() -> dict:
    split = {"seed": SEED, "train_fraction": TRAIN_FRACTION,
             "rule": "group by (service, fault); within suite/system/fault, first 40% of groups by seeded hash -> train",
             "suites": {}}
    for suite in SUITES:
        index, labels = case_index(suite), load_labels(suite)
        groups = defaultdict(lambda: defaultdict(list))  # (system, fault) -> service -> case ids
        for case_id, case_dir in index.items():
            system = case_dir.parent.parent.name.split("-")[-1]
            lab = labels[case_id]
            groups[(system, lab.fault)][lab.root_service].append(case_id)
        train, test = [], []
        for (system, fault), services in sorted(groups.items()):
            order = sorted(services, key=lambda s: hashlib.sha256(f"{SEED}|{suite}|{system}|{fault}|{s}".encode()).hexdigest())
            n_train = math.floor(len(order) * TRAIN_FRACTION)
            for i, service in enumerate(order):
                (train if i < n_train else test).extend(services[service])
        split["suites"][suite] = {"train": sorted(train), "test": sorted(test)}
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify data_splits.json matches a rebuild")
    args = parser.parse_args()
    split = build()
    if args.check:
        same = json.loads(OUT.read_text()) == split
        print("data_splits.json is reproducible" if same else "data_splits.json differs from a rebuild")
        sys.exit(0 if same else 1)
    OUT.write_text(json.dumps(split, indent=1) + "\n")
    for suite, parts in split["suites"].items():
        print(f"{suite}: train {len(parts['train'])}, test {len(parts['test'])}")


if __name__ == "__main__":
    main()
