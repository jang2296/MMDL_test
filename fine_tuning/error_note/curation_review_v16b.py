"""Render private, deterministic quality-review samples; never training targets."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .curation_policy_v16b import feature_candidate


def sample(rows: list[dict], seed: int = 3407) -> list[dict]:
    """Sample score thirds within each source, with unchanged native records."""
    sources: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("exclusion_reasons"):
            sources[row["source"]].append(row)
    result = []
    for source, items in sorted(sources.items()):
        items = sorted(items, key=lambda row: (-feature_candidate(row)["score"], row["problem_id"]))
        count = min(len(items), 40 if len(items) > 400 else 20)
        for part in range(3):
            bucket = items[part * len(items) // 3:(part + 1) * len(items) // 3]
            quota = count // 3 + int(part < count % 3)
            ranked = sorted(bucket, key=lambda row: hashlib.sha256(
                f"v16b-review:{seed}:{source}:{part}:{row['problem_id']}".encode()).hexdigest())
            for row in ranked[:quota]:
                result.append(row | {"review_stratum": ("high", "middle", "low")[part],
                                     "pre_review_features": feature_candidate(row)})
        assert sum(row["source"] == source for row in result) == count
    assert len({row["problem_id"] for row in result}) == len(result)
    return result


def render(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    packet = []
    for source, items in sorted(groups.items()):
        for start in range(0, len(items), 4):
            page = Image.new("RGB", (2000, 2100), "white")
            draw = ImageDraw.Draw(page)
            page_name = f"{source}-{start // 4:02d}.png"
            for offset, row in enumerate(items[start:start + 4]):
                x, y = (offset % 2) * 1000, (offset // 2) * 1050
                draw.rectangle((x, y, x + 998, y + 1048), outline="#555555", width=2)
                number = start + offset
                heading = f"{source} #{number:02d} [{row['review_stratum']}]"
                draw.text((x + 12, y + 10), heading, font=title_font, fill="black")
                actor, verifier = row["actor"], row["verifier"]
                choices = actor.get("choices") or []
                if isinstance(choices, dict):
                    choices_text = " | ".join(f"{key}: {value}" for key, value in choices.items())
                else:
                    choices_text = " | ".join(f"{chr(65 + n)}: {value}" for n, value in enumerate(choices))
                text = f"Q: {actor['question']}\n{choices_text}\nGOLD: {verifier['gold']}"
                lines = [line for paragraph in text.splitlines()
                         for line in (textwrap.wrap(paragraph, 90) or [""])]
                # Long native text remains complete in the sidecar; the sheet signals overflow.
                display = lines[:17]
                if len(lines) > 17:
                    display[-1] = "[Continues in packet.jsonl; read complete native text.]"
                draw.multiline_text((x + 12, y + 42), "\n".join(display), font=font, fill="black", spacing=3)
                top = y + 48 + len(display) * 25
                height = y + 1025 - top
                paths = row["image_paths"]
                for image_index, path in enumerate(paths):
                    with Image.open(path) as native:
                        tile = ImageOps.contain(native.convert("RGB"), (960 // len(paths), height))
                    left = x + 20 + image_index * (960 // len(paths))
                    page.paste(tile, (left, top))
                packet.append(row | {"review_number": number, "sheet": str(output / page_name),
                                     "sheet_cell": offset, "training_allowed": False,
                                     "review_image_is_derivative": True})
            page.save(output / page_name)
    with (output / "packet.jsonl").open("x", encoding="utf-8") as stream:
        for row in packet:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "sampling.json").write_text(json.dumps({
        "seed": 3407, "count": len(packet), "by_source": {key: len(value) for key, value in groups.items()},
        "policy": "score thirds, SHA256 selection; 40 if eligible source count >400, otherwise20",
        "features_stage": "pre-review source/question heuristics; enrichment availability recorded per row",
        "training_allowed": False, "human_verified": False,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--enrichment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines()]
    if args.enrichment:
        extra = {row["problem_id"]: row for row in map(json.loads, args.enrichment.read_text().splitlines())}
        rows = [row | extra.get(row["problem_id"], {}) for row in rows]
    render(sample(rows), args.output)


if __name__ == "__main__":
    main()
