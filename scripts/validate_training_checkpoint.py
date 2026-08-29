from __future__ import annotations

import argparse

from repostguard.checkpoint import load_checkpoint
from repostguard.config import config_digest, load_config


REQUIRED_FIELDS = {
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "epoch",
    "batch_in_epoch",
    "global_step",
    "best_metric",
    "rng_state",
    "sampler_state",
    "config",
    "config_sha256",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a resumable RepostGuard checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    checkpoint = load_checkpoint(arguments.checkpoint)
    missing = REQUIRED_FIELDS.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    expected_digest = config_digest(config)
    if checkpoint["config_sha256"] != expected_digest:
        raise ValueError("Checkpoint config digest does not match")
    if int(checkpoint["epoch"]) < 0 or int(checkpoint["global_step"]) < 0:
        raise ValueError("Checkpoint has negative progress")
    print(
        f"valid checkpoint={arguments.checkpoint} "
        f"epoch={checkpoint['epoch']} global_step={checkpoint['global_step']}"
    )


if __name__ == "__main__":
    main()

