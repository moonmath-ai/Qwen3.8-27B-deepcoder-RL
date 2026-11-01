"""Held-out / validation / training split of DeepCoder by problem, deterministic. Writes data/split_manifest.json (hashes, no texts).

From DeepCoder train (primeintellect + taco + lcbv5), built exactly as the recipe builds tasks
(code_env._normalize_tests / _build_question):
  - held-out A: 500 problems with >= 5 tests (the headline set);
  - validation: 100 further problems, the same rule;
  - training: everything else, minus any problem that also appears in the test split (held-out B) or in
    held-out A / validation, including copies of the same problem in another source.
Duplicates are found on the normalised question text (lowercase, whitespace collapsed): an exact match, or
a match of the first 400 characters (the same statement with different examples or formatting).
  $ZROFORM_CLIENT_VENV/bin/python scripts/split.py [--out data/split_manifest.json]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
from typing import Any

from datasets import load_dataset
from tinker_cookbook.recipes.code_rl import code_env

DATASET = "agentica-org/DeepCoder-Preview-Dataset"
SEED, N_HELDOUT, N_VAL, MIN_TESTS, PREFIX = 20261022, 500, 100, 5, 400


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def task_key(problem: str) -> str:
    """Key of a task as the recipe builds it (DeepcoderTask.problem); the training filter uses the same."""
    return hashlib.sha256(norm(problem).encode()).hexdigest()[:20]


def near_key(question: str) -> str:
    return hashlib.sha256(norm(question)[:PREFIX].encode()).hexdigest()[:20]


def load(split: str) -> list[dict[str, Any]]:
    names = ("primeintellect", "taco", "lcbv5") if split == "train" else ("codeforces", "lcbv5")
    out = []
    for name in names:
        for row in load_dataset(DATASET, name=name, split=split):
            meta = code_env._ensure_dict(row.get("metadata", {}))
            tests = code_env._normalize_tests(row.get("tests") or row.get("ground_truth"), meta)
            problem = code_env._build_question(row)
            question = row.get("question") or row.get("prompt") or row.get("problem")
            if not tests or problem is None:
                continue  # the recipe drops these too
            out.append({"subset": name, "key": task_key(problem), "near": near_key(question), "n_tests": len(tests)})
    return out


def main(a) -> dict:
    train, test = load("train"), load("test")
    test_near = {t["near"] for t in test}
    groups: dict[str, list[dict]] = collections.defaultdict(list)  # one group per distinct problem statement
    for t in train:
        groups[t["near"]].append(t)
    contaminated = {n for n in groups if n in test_near}
    eligible = sorted(n for n, g in groups.items()
                      if n not in contaminated and min(x["n_tests"] for x in g) >= MIN_TESTS)
    rng = random.Random(SEED)
    picked = rng.sample(eligible, N_HELDOUT + N_VAL)
    heldout_near, val_near = picked[:N_HELDOUT], picked[N_HELDOUT:]

    def rep(near):  # the task used for evaluation: the first copy in load order
        g = groups[near][0]
        return {"key": g["key"], "subset": g["subset"], "n_tests": g["n_tests"], "copies": len(groups[near])}

    excluded_near = contaminated | set(heldout_near) | set(val_near)
    exclude = sorted({t["key"] for n in excluded_near for t in groups[n]})
    manifest = {
        "version": 1, "dataset": DATASET, "seed": SEED,
        "rules": {"min_tests": MIN_TESTS, "near_duplicate_prefix_chars": PREFIX,
                  "key": "sha256(normalised DeepcoderTask.problem)[:20]", "near": "sha256(normalised question[:400])[:20]"},
        "counts": {
            "train_tasks_loaded": len(train), "distinct_train_problems": len(groups),
            "train_problems_also_in_test_split": len(contaminated),
            "train_tasks_removed_as_test_overlap": sum(len(groups[n]) for n in contaminated),
            "eligible_for_heldout": len(eligible), "heldout_a": len(heldout_near), "validation": len(val_near),
            # tasks, not keys: byte-identical copies of a problem share one key and are all dropped
            "training_tasks_after_exclusion": sum(t["key"] not in set(exclude) for t in train), "test_split_tasks": len(test),
            "heldout_a_by_subset": dict(collections.Counter(groups[n][0]["subset"] for n in heldout_near)),
            "validation_by_subset": dict(collections.Counter(groups[n][0]["subset"] for n in val_near)),
            "test_split_by_subset": dict(collections.Counter(t["subset"] for t in test)),
        },
        "heldout_a": [rep(n) for n in heldout_near],
        "validation": [rep(n) for n in val_near],
        "heldout_b": [t["key"] for t in test],
        "exclude_from_training": exclude,
    }
    with open(a.out, "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest["counts"], indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/split_manifest.json"))
    sys.exit(main(ap.parse_args()) is None)
