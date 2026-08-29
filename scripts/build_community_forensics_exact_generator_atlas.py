from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = True


SPLITS = (
    {
        "key": "train",
        "title": "Training set",
        "manifest": "data/manifests/community_forensics_train_v2.csv",
        "prefix": "TR",
        "color": "#2F6BFF",
        "group": "Train exact generators",
    },
    {
        "key": "test_external_unseen_generator",
        "title": "External test · unseen architecture family",
        "manifest": "data/manifests/community_forensics_test_external_unseen_generator.csv",
        "prefix": "UF",
        "color": "#D94B64",
        "group": "Test · family-unseen / exact-unseen",
    },
)

REQUIRED_FIELDS = {
    "sample_id",
    "path",
    "label",
    "project_split",
    "canonical_generator_id",
    "architecture",
    "generator_exposure",
    "sha256",
    "width",
    "height",
    "format",
    "byte_size",
}

BACKGROUND = "#F3F5F9"
CARD = "#FFFFFF"
INK = "#172033"
MUTED = "#667085"
LINE = "#D8DEE9"


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Manifest has no header: {manifest}")
        missing = REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"Manifest missing {sorted(missing)}: {manifest}")
        return list(reader)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)


def _atomic_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty atlas index")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def _atomic_jpeg(path: str | Path, image: Image.Image, quality: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.{os.getpid()}{target.suffix}")
    image.save(
        temporary,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=True,
        progressive=True,
        dpi=(150, 150),
    )
    os.replace(temporary, target)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def _fit_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    text = text.strip()
    if not text:
        return ["UNSPECIFIED"]
    lines: list[str] = []
    remaining = text
    while remaining and len(lines) < max_lines:
        if draw.textlength(remaining, font=font) <= max_width:
            lines.append(remaining)
            remaining = ""
            break
        best = 0
        for index in range(1, len(remaining) + 1):
            if draw.textlength(remaining[:index], font=font) <= max_width:
                best = index
            else:
                break
        if best == 0:
            best = 1
        break_at = max(
            remaining.rfind("/", 0, best + 1),
            remaining.rfind("-", 0, best + 1),
            remaining.rfind("_", 0, best + 1),
        )
        if break_at >= max(1, best // 2):
            best = break_at + 1
        lines.append(remaining[:best].rstrip())
        remaining = remaining[best:].lstrip()
    if remaining:
        suffix = lines[-1]
        while suffix and draw.textlength(f"{suffix}…", font=font) > max_width:
            suffix = suffix[:-1]
        lines[-1] = f"{suffix}…"
    return lines


def _rank_generator(generator: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{generator}".encode("utf-8")).hexdigest()
    return digest, generator


def _proportional_quotas(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    if total < 0 or total > sum(group_sizes.values()):
        raise ValueError(f"Invalid proportional quota total: {total}")
    if not group_sizes:
        return {}
    denominator = sum(group_sizes.values())
    quotas = {
        group: min(size, int(total * size / denominator))
        for group, size in group_sizes.items()
    }
    remaining = total - sum(quotas.values())
    remainders = sorted(
        group_sizes,
        key=lambda group: (
            -(total * group_sizes[group] / denominator - quotas[group]),
            group,
        ),
    )
    while remaining:
        progressed = False
        for group in remainders:
            if quotas[group] >= group_sizes[group]:
                continue
            quotas[group] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Could not satisfy proportional generator quotas")
    return quotas


def _choose_train_generators(
    by_generator: dict[str, list[dict[str, str]]],
    limit: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    if limit <= 0 or limit > len(by_generator):
        raise ValueError(f"train generator limit must be in [1, {len(by_generator)}]")
    by_architecture: dict[str, list[str]] = defaultdict(list)
    for generator, rows in by_generator.items():
        architectures = {row["architecture"].strip() or "UNSPECIFIED" for row in rows}
        if len(architectures) != 1:
            raise RuntimeError(
                f"Generator has inconsistent architectures: {generator} / {architectures}"
            )
        by_architecture[next(iter(architectures))].append(generator)

    # Keep every generator from very small architecture families so the compact
    # atlas cannot erase rare training families. Fill the remaining quota
    # proportionally from larger families using a stable hash rank.
    rare_threshold = 5
    rare_groups = {
        architecture: generators
        for architecture, generators in by_architecture.items()
        if len(generators) <= rare_threshold
    }
    rare_total = sum(len(generators) for generators in rare_groups.values())
    if rare_total > limit:
        raise RuntimeError(
            f"Rare-family preservation needs {rare_total} slots but limit is {limit}"
        )
    common_groups = {
        architecture: generators
        for architecture, generators in by_architecture.items()
        if architecture not in rare_groups
    }
    quotas = _proportional_quotas(
        {architecture: len(generators) for architecture, generators in common_groups.items()},
        limit - rare_total,
    )
    chosen: list[str] = []
    selected_counts: dict[str, int] = {}
    for architecture in sorted(by_architecture):
        generators = by_architecture[architecture]
        if architecture in rare_groups:
            selected = sorted(generators)
        else:
            selected = sorted(generators, key=lambda item: _rank_generator(item, seed))[
                : quotas[architecture]
            ]
        chosen.extend(selected)
        selected_counts[architecture] = len(selected)
    if len(chosen) != limit or len(set(chosen)) != limit:
        raise RuntimeError("Compact train-generator selection did not satisfy its limit")
    return sorted(chosen), {
        "selection_seed": seed,
        "train_generator_limit": limit,
        "rare_family_threshold": rare_threshold,
        "available_by_architecture": {
            architecture: len(generators)
            for architecture, generators in sorted(by_architecture.items())
        },
        "selected_by_architecture": selected_counts,
        "selection_rule": (
            "Keep all generators in architecture families with <=5 generators; "
            "allocate remaining slots proportionally and rank generators by "
            "SHA256(seed:canonical_generator_id)."
        ),
    }


def _representatives(
    data_root: Path,
    train_limit: int,
    selection_seed: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    selected: dict[str, list[dict[str, Any]]] = {}
    index_rows: list[dict[str, Any]] = []
    train_sampling: dict[str, Any] = {}
    atlas_index = 0
    for split in SPLITS:
        rows = _read_manifest(split["manifest"])
        by_generator: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["project_split"] != split["key"]:
                raise RuntimeError(
                    f"project_split drift: {split['manifest']} / {row['project_split']}"
                )
            if int(row["label"]) != 1:
                continue
            generator = row["canonical_generator_id"].strip()
            if not generator:
                raise RuntimeError(f"Blank canonical_generator_id: {row['sample_id']}")
            by_generator[generator].append(row)

        generators = sorted(by_generator)
        if split["key"] == "train":
            generators, train_sampling = _choose_train_generators(
                by_generator,
                train_limit,
                selection_seed,
            )
        entries: list[dict[str, Any]] = []
        for split_index, generator in enumerate(generators, start=1):
            candidates = by_generator[generator]
            chosen = min(candidates, key=lambda row: (row["sha256"], row["sample_id"]))
            absolute_path = data_root / chosen["path"]
            if not absolute_path.is_file():
                raise RuntimeError(f"Missing selected image: {absolute_path}")
            if absolute_path.stat().st_size != int(chosen["byte_size"]):
                raise RuntimeError(f"Selected image byte-size mismatch: {absolute_path}")
            atlas_index += 1
            entry = {
                "atlas_index": atlas_index,
                "tile_id": f"{split['prefix']}{split_index:03d}",
                "section": split["title"],
                "section_group": split["group"],
                "split": split["key"],
                "split_index": split_index,
                "generator_id": generator,
                "architecture": chosen["architecture"].strip() or "UNSPECIFIED",
                "generator_exposure": chosen["generator_exposure"].strip(),
                "sample_id": chosen["sample_id"],
                "relative_path": chosen["path"],
                "absolute_path": str(absolute_path),
                "sha256": chosen["sha256"],
                "width": int(chosen["width"]),
                "height": int(chosen["height"]),
                "format": chosen["format"],
                "byte_size": int(chosen["byte_size"]),
                "candidate_images_for_generator": len(candidates),
                "generator_selection_rule": (
                    train_sampling.get("selection_rule", "all test generators retained")
                    if split["key"] == "train"
                    else "all test generators retained"
                ),
                "selection_rule": "minimum (sha256, sample_id) within split and exact generator",
                "color": split["color"],
            }
            entries.append(entry)
            index_rows.append({key: value for key, value in entry.items() if key not in {"absolute_path", "color"}})
        selected[split["key"]] = entries
    return selected, index_rows, train_sampling


def _draw_header(
    canvas: Image.Image,
    title: str,
    subtitle: str,
    accent: str,
    font_path: Path,
    height: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, height), fill="#111827")
    draw.rectangle((0, height - 9, canvas.width, height), fill=accent)
    draw.text((44, 34), title, font=_font(font_path, 44), fill="#FFFFFF")
    draw.multiline_text(
        (46, 98),
        subtitle,
        font=_font(font_path, 21),
        fill="#CFD6E4",
        spacing=8,
    )


def _load_thumbnail(path: str, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        rgb = source.convert("RGB")
    contained = ImageOps.contain(
        rgb,
        (width, height),
        method=Image.Resampling.LANCZOS,
    )
    background = Image.new("RGB", (width, height), "#E7EBF2")
    offset = ((width - contained.width) // 2, (height - contained.height) // 2)
    background.paste(contained, offset)
    return background


def _render_train_atlas(
    entries: list[dict[str, Any]],
    font_path: Path,
) -> Image.Image:
    columns = 10
    tile_width = 330
    tile_height = 335
    gap = 4
    margin = 18
    header_height = 188
    rows = math.ceil(len(entries) / columns)
    width = margin * 2 + columns * tile_width + (columns - 1) * gap
    height = header_height + margin + rows * tile_height + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    _draw_header(
        canvas,
        "TRAIN · compact exact-generator sample",
        f"{len(entries)} of 900 generators · 10 × {rows} atlas · rare families retained, remaining slots proportionally sampled\n"
        "One deterministic AIGI representative per selected exact generator",
        "#2F6BFF",
        font_path,
        header_height,
    )
    draw = ImageDraw.Draw(canvas)
    id_font = _font(font_path, 17)
    generator_font = _font(font_path, 15)
    image_width = tile_width - 16
    image_height = 238
    for position, entry in enumerate(entries):
        row, column = divmod(position, columns)
        x = margin + column * (tile_width + gap)
        y = header_height + margin + row * (tile_height + gap)
        draw.rounded_rectangle(
            (x, y, x + tile_width - 1, y + tile_height - 1),
            radius=7,
            fill=CARD,
            outline=LINE,
            width=1,
        )
        draw.rectangle((x, y, x + tile_width - 1, y + 5), fill=entry["color"])
        thumbnail = _load_thumbnail(entry["absolute_path"], image_width, image_height)
        canvas.paste(thumbnail, (x + 8, y + 11))
        label_y = y + 254
        draw.text(
            (x + 8, label_y),
            f"{entry['tile_id']} · {entry['architecture']}",
            font=id_font,
            fill=entry["color"],
        )
        for line_index, line in enumerate(
            _fit_lines(draw, entry["generator_id"], generator_font, tile_width - 16, 3)
        ):
            draw.text(
                (x + 8, label_y + 23 + line_index * 18),
                line,
                font=generator_font,
                fill=INK,
            )
    return canvas


def _render_test_atlas(
    entries: list[dict[str, Any]],
    font_path: Path,
) -> Image.Image:
    columns = 7
    tile_width = 470
    tile_height = 430
    gap = 10
    margin = 22
    header_height = 188
    rows = math.ceil(len(entries) / columns)
    width = margin * 2 + columns * tile_width + (columns - 1) * gap
    height = header_height + margin + rows * tile_height + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    _draw_header(
        canvas,
        "EXTERNAL TEST · one AIGI image per exact generator",
        f"{len(entries)} generators: 9 family-seen/exact-unseen + 12 family-unseen/exact-unseen\n"
        "Amber = train-seen architecture family · Rose = train-unseen architecture family",
        "#E39A22",
        font_path,
        header_height,
    )
    draw = ImageDraw.Draw(canvas)
    id_font = _font(font_path, 18)
    generator_font = _font(font_path, 17)
    sample_font = _font(font_path, 11)
    image_width = tile_width - 28
    image_height = 300
    for position, entry in enumerate(entries):
        row, column = divmod(position, columns)
        x = margin + column * (tile_width + gap)
        y = header_height + margin + row * (tile_height + gap)
        draw.rounded_rectangle(
            (x, y, x + tile_width - 1, y + tile_height - 1),
            radius=12,
            fill=CARD,
            outline=LINE,
            width=2,
        )
        draw.rectangle((x, y, x + tile_width - 1, y + 10), fill=entry["color"])
        thumbnail = _load_thumbnail(entry["absolute_path"], image_width, image_height)
        canvas.paste(thumbnail, (x + 14, y + 18))
        label_y = y + 325
        draw.text(
            (x + 16, label_y),
            f"{entry['tile_id']} · {entry['architecture']}",
            font=id_font,
            fill=entry["color"],
        )
        generator_lines = _fit_lines(
            draw, entry["generator_id"], generator_font, tile_width - 32, 2
        )
        for line_index, line in enumerate(generator_lines):
            draw.text(
                (x + 16, label_y + 25 + line_index * 21),
                line,
                font=generator_font,
                fill=INK,
            )
        draw.text(
            (x + 16, y + tile_height - 29),
            entry["sample_id"],
            font=sample_font,
            fill=MUTED,
        )
    return canvas


def _render_master(
    train_atlas: Image.Image,
    test_atlas: Image.Image,
    font_path: Path,
    counts: dict[str, int],
) -> Image.Image:
    width = max(train_atlas.width, test_atlas.width)
    master_header = 250
    section_gap = 28
    height = master_header + train_atlas.height + section_gap + test_atlas.height
    canvas = Image.new("RGB", (width, height), "#E8ECF3")
    _draw_header(
        canvas,
        "COMMUNITY FORENSICS · EXACT GENERATOR SAMPLE ATLAS",
        f"{counts['total']} exact generators / {counts['total']} AIGI representatives · "
        f"Train-v2 {counts['train']} · Strict unseen test {counts['unseen_family']}\n"
        "Representative selection is deterministic and label-safe; full image lineage is stored in the companion CSV.",
        "#78D3B6",
        font_path,
        master_header,
    )
    train_x = (width - train_atlas.width) // 2
    test_x = (width - test_atlas.width) // 2
    canvas.paste(train_atlas, (train_x, master_header))
    canvas.paste(test_atlas, (test_x, master_header + train_atlas.height + section_gap))
    return canvas


def build(arguments: argparse.Namespace) -> None:
    generated_at = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
    font_path = Path(arguments.font)
    if not font_path.is_file():
        raise RuntimeError(f"Font not found: {font_path}")
    data_root = Path(arguments.data_root)
    selected, index_rows, train_sampling = _representatives(
        data_root,
        arguments.train_generator_limit,
        arguments.selection_seed,
    )
    train_entries = selected["train"]
    unseen_entries = selected["test_external_unseen_generator"]
    test_entries = unseen_entries

    expected = {
        "train": arguments.train_generator_limit,
        "unseen_family": 12,
    }
    observed = {
        "train": len(train_entries),
        "unseen_family": len(unseen_entries),
    }
    if observed != expected:
        raise RuntimeError(f"Exact-generator coverage drift: {observed} != {expected}")
    if len({row["generator_id"] for row in index_rows}) != len(index_rows):
        raise RuntimeError("An exact generator appears in more than one selected split")
    if len({row["sample_id"] for row in index_rows}) != len(index_rows):
        raise RuntimeError("A representative sample was selected more than once")
    if any(row["generator_id"] in {"", "UNSPECIFIED"} for row in index_rows):
        raise RuntimeError("Selected representative has no exact generator identity")

    train_atlas = _render_train_atlas(train_entries, font_path)
    test_atlas = _render_test_atlas(test_entries, font_path)
    counts = {**observed, "total": sum(observed.values())}
    master = _render_master(train_atlas, test_atlas, font_path, counts)

    _atomic_jpeg(arguments.train_output, train_atlas, arguments.quality)
    _atomic_jpeg(arguments.test_output, test_atlas, arguments.quality)
    _atomic_jpeg(arguments.output, master, arguments.quality)
    _atomic_csv(arguments.index_csv, index_rows)

    outputs = {}
    for key, path, dimensions in (
        ("master", Path(arguments.output), master.size),
        ("train", Path(arguments.train_output), train_atlas.size),
        ("test", Path(arguments.test_output), test_atlas.size),
        ("index", Path(arguments.index_csv), None),
    ):
        outputs[key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if dimensions is not None:
            outputs[key]["width"] = dimensions[0]
            outputs[key]["height"] = dimensions[1]

    audit = {
        "schema_version": 1,
        "generated_at_asia_singapore": generated_at,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scope": {
            "included": [
                "community_forensics_train_v2.csv label=1",
                "community_forensics_test_external_unseen_generator.csv label=1",
            ],
            "excluded": "Real images and all validation manifests, because exact generator identity applies to AIGI and the request names train/test.",
        },
        "train_generator_sampling": train_sampling,
        "selection_rule": "For each retained (project_split, canonical_generator_id), choose minimum (sha256, sample_id).",
        "counts": counts,
        "checks": {
            "expected_generator_counts_match": True,
            "unique_generator_ids": len({row["generator_id"] for row in index_rows}),
            "unique_sample_ids": len({row["sample_id"] for row in index_rows}),
            "missing_selected_files": 0,
            "byte_size_mismatches": 0,
        },
        "manifests": {
            split["key"]: {
                "path": split["manifest"],
                "sha256": _sha256(split["manifest"]),
            }
            for split in SPLITS
        },
        "outputs": outputs,
    }
    _atomic_text(arguments.audit_json, json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "community_forensics_exact_generator_atlas_complete",
                "counts": counts,
                "master": outputs["master"],
                "train": outputs["train"],
                "test": outputs["test"],
                "index_rows": len(index_rows),
                "audit_json": str(arguments.audit_json),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build annotated Community Forensics exact-generator sample atlases"
    )
    parser.add_argument("--data-root", default="data/raw/community_forensics")
    parser.add_argument(
        "--font", default="/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
    )
    parser.add_argument(
        "--output", default="reports/atlases/train_v2/community_forensics_exact_generators_atlas.jpg"
    )
    parser.add_argument(
        "--train-output",
        default="reports/atlases/train_v2/community_forensics_train_exact_generators_atlas.jpg",
    )
    parser.add_argument(
        "--test-output",
        default="reports/atlases/train_v2/community_forensics_test_exact_generators_atlas.jpg",
    )
    parser.add_argument(
        "--index-csv",
        default="reports/atlases/train_v2/community_forensics_exact_generators_atlas_index.csv",
    )
    parser.add_argument(
        "--audit-json",
        default="reports/atlases/train_v2/community_forensics_exact_generators_atlas_audit.json",
    )
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--train-generator-limit", type=int, default=78)
    parser.add_argument("--selection-seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> None:
    build(_parse_args())


if __name__ == "__main__":
    main()
