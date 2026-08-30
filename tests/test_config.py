from __future__ import annotations

from pathlib import Path

from repostguard.config import load_config


def test_all_experiment_configs_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    for experiment in ("b0", "b1", "b2", "m2", "m3"):
        config = load_config(root / "configs" / f"{experiment}.yaml")
        assert config["model"]["experiment"] == experiment
        assert config["data"]["image_size"] == 224
        assert config["data"].get("format_debias", {}).get("enabled", False) is False
        assert config["output"]["directory"] == f"outputs/{experiment}"


def test_all_sidset_experiment_configs_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    for experiment in ("b0", "b1", "b2", "m2", "m3"):
        config = load_config(root / "configs" / "sidset" / f"{experiment}.yaml")
        assert config["model"]["experiment"] == experiment
        assert config["data"]["root"] == "data/raw/sid_set"
        assert config["data"]["train_manifest"] == "data/manifests/sidset_train.csv"
        assert config["data"]["val_manifest"] == "data/manifests/sidset_validation.csv"
        assert config["data"]["format_debias"]["enabled"] is True
        assert config["data"]["format_debias"]["train_qualities"] == [70, 80, 90, 95]
        assert config["data"]["format_debias"]["eval_quality"] == 90
        assert config["output"]["directory"] == f"outputs/sidset/{experiment}"


def test_all_community_forensics_configs_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    for experiment in ("b0", "b1", "b2", "m2", "m3"):
        config = load_config(root / "configs" / "community_forensics" / f"{experiment}.yaml")
        assert config["model"]["experiment"] == experiment
        assert config["seed"] == 20260828
        assert config["data"]["root"] == "data/raw/community_forensics"
        assert config["data"]["train_manifest"] == (
            "data/manifests/community_forensics_train_v2.csv"
        )
        assert config["data"]["val_manifest"] == (
            "data/manifests/community_forensics_val_unseen_generator.csv"
        )
        assert config["data"]["validation_slices"]["exact_seen_generator"].endswith(
            "community_forensics_val_external_exact_seen_generator.csv"
        )
        assert "seen_family" not in config["data"]["external_tests"]
        assert config["data"]["generator_protocol"]["retired_seen_family_test"] == (
            "promoted into train-v2 and forbidden for future evaluation"
        )
        assert config["data"]["format_debias"]["enabled"] is True
        assert config["output"]["directory"] == (
            f"outputs/community_forensics_v2/{experiment}"
        )


def test_all_community_forensics_v3_configs_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    for experiment in ("b0", "b1", "b2", "m2", "m3"):
        config = load_config(
            root / "configs" / "community_forensics_v3" / f"{experiment}.yaml"
        )
        assert config["model"]["experiment"] == experiment
        assert config["seed"] == 20260830
        assert config["data"]["root"] == "data/raw/community_forensics"
        assert config["data"]["train_manifest"] == (
            "data/manifests/community_forensics_train_v3.csv"
        )
        assert config["data"]["val_manifest"] == (
            "data/manifests/community_forensics_val_unseen_generator.csv"
        )
        assert "seen_family" not in config["data"]["external_tests"]
        assert config["data"]["external_tests"]["unseen_generator_expanded"].endswith(
            "community_forensics_test_external_unseen_generator_v3_expanded.csv"
        )
        assert config["data"]["generator_protocol"]["retired_seen_family_test"] == (
            "promoted into train-v2 and forbidden for future evaluation"
        )
        assert config["data"]["format_debias"]["enabled"] is True
        assert config["output"]["directory"] == (
            f"outputs/community_forensics_v3/{experiment}"
        )
