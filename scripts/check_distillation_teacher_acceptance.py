from __future__ import annotations

import argparse
import json
from pathlib import Path


MINIMUMS = {
    "m2": {"internal_clean_auroc": 0.94, "unseen_clean_auroc": 0.84},
    "m3": {"internal_clean_auroc": 0.94, "unseen_clean_auroc": 0.84},
}


def _load(path: str) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate reproduced distillation teachers")
    parser.add_argument("--experiment", choices=sorted(MINIMUMS), required=True)
    parser.add_argument("--internal-summary", required=True)
    parser.add_argument("--unseen-summary", required=True)
    arguments = parser.parse_args()
    internal = _load(arguments.internal_summary)
    unseen = _load(arguments.unseen_summary)
    observed = {
        "internal_clean_auroc": float(internal["clean_auroc"]),
        "unseen_clean_auroc": float(unseen["clean_auroc"]),
    }
    minimums = MINIMUMS[arguments.experiment]
    failures = {
        name: {"observed": observed[name], "minimum": minimum}
        for name, minimum in minimums.items()
        if observed[name] < minimum
    }
    payload = {
        "event": "distillation_teacher_acceptance",
        "experiment": arguments.experiment,
        "observed": observed,
        "minimums": minimums,
        "accepted": not failures,
        "failures": failures,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
