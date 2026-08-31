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


def test_dual_teacher_student_config_resolves() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "community_forensics" / "student_mnv3_dual_teacher.yaml"
    )
    assert config["model"]["experiment"] == "student_mnv3"
    assert config["model"]["student_backbone"] == "mobilenet_v3_large"
    assert config["distillation"]["m3_weight"] == 0.7
    assert config["distillation"]["m2_weight"] == 0.3
    assert config["distillation"]["m2_config"].endswith(
        "distillation_teacher_m2.yaml"
    )
    assert config["distillation"]["m3_config"].endswith(
        "distillation_teacher_m3.yaml"
    )
    assert config["distillation"]["views"][0]["id"] == "clean"
    assert len(config["distillation"]["views"]) == 4


def test_dual_teacher_student_smoke_config_resolves() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root
        / "configs"
        / "community_forensics"
        / "student_mnv3_dual_teacher_smoke.yaml"
    )
    assert config["model"]["experiment"] == "student_mnv3"
    assert config["train"]["epochs"] == 1
    assert config["train"]["resume"] == "none"
    assert config["distillation"]["m3_weight"] == 0.7


def test_v3_dual_teacher_student_configs_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    full = load_config(
        root
        / "configs"
        / "community_forensics_v3"
        / "student_mnv3_dual_teacher.yaml"
    )
    smoke = load_config(
        root
        / "configs"
        / "community_forensics_v3"
        / "student_mnv3_dual_teacher_smoke.yaml"
    )
    m3_smoke = load_config(
        root
        / "configs"
        / "community_forensics_v3"
        / "student_mnv3_m3_primary_smoke.yaml"
    )

    assert full["data"]["train_manifest"].endswith("community_forensics_train_v3.csv")
    assert full["distillation"]["m2_checkpoint"].endswith(
        "outputs/community_forensics_v3/m2/best.pt"
    )
    assert full["distillation"]["m3_checkpoint"].endswith(
        "outputs/community_forensics_v3/m3/best.pt"
    )
    assert smoke["train"]["epochs"] == 2
    assert smoke["distillation"]["cache_directory"].endswith(
        "community_forensics_m2_m3_smoke_v3"
    )
    assert m3_smoke["distillation"]["m2_weight"] == 0.0
    assert m3_smoke["distillation"]["m3_weight"] == 1.0
