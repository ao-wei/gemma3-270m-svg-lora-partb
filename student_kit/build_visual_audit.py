#!/usr/bin/env python3
"""Render all final variants, write a manifest, and build review contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import write_json
from render_svg import render, render_failure_card
from reward import reward


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", default="results.json")
    parser.add_argument("--seed123", default="results_seed123.json")
    parser.add_argument("--render-dir", default="output/rendered_final")
    parser.add_argument("--contact-dir", default="output/audit")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    primary = json.loads(Path(args.primary).read_text(encoding="utf-8"))
    seed123 = json.loads(Path(args.seed123).read_text(encoding="utf-8"))
    if [x["prompt"] for x in primary["samples"]] != [x["prompt"] for x in seed123["samples"]]:
        raise ValueError("primary and seed123 prompts differ")
    render_dir, contact_dir = Path(args.render_dir), Path(args.contact_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "variants": ["reference", "base", "seed42", "seed123"], "items": []}
    for left, right in zip(primary["samples"], seed123["samples"]):
        variants = {
            "reference": left["reference_svg"],
            "base": left["base"]["svg"],
            "seed42": left["tuned"]["svg"],
            "seed123": right["tuned"]["svg"],
        }
        for variant, svg in variants.items():
            assessment = reward(left["prompt"], svg)
            destination = render_dir / f"sample_{left['id']:02d}_{variant}.png"
            status = "rendered"
            if assessment["metadata"]["fatal"]:
                render_failure_card(destination, args.size, assessment["violations"])
                status = "failure_card"
            else:
                render(svg, destination, args.size)
            manifest["items"].append({
                "id": left["id"],
                "variant": variant,
                "status": status,
                "fatal": assessment["metadata"]["fatal"],
                "violations": assessment["violations"],
                "path": str(destination),
                "sha256": sha256(destination),
            })
    write_json("render_manifest.json", manifest)

    font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    label_font = ImageFont.truetype(font_path, 22)
    title_font = ImageFont.truetype(font_path, 28)
    cell, header, label_height = 230, 52, 34
    variants = manifest["variants"]
    for page_index, start in enumerate(range(0, len(primary["samples"]), 5), 1):
        page_samples = primary["samples"][start : start + 5]
        sheet = Image.new("RGB", (cell * 4, header + (cell + label_height) * len(page_samples)), "white")
        draw = ImageDraw.Draw(sheet)
        for column, variant in enumerate(variants):
            draw.text((column * cell + cell / 2, header / 2), variant, fill="#17324D", font=title_font, anchor="mm")
        for row, sample in enumerate(page_samples):
            y = header + row * (cell + label_height)
            for column, variant in enumerate(variants):
                image = Image.open(render_dir / f"sample_{sample['id']:02d}_{variant}.png").convert("RGB")
                image.thumbnail((cell - 12, cell - 12))
                x = column * cell + (cell - image.width) // 2
                sheet.paste(image, (x, y + (cell - image.height) // 2))
                draw.rectangle((column * cell, y, (column + 1) * cell - 1, y + cell - 1), outline="#C7D0D6", width=2)
            draw.text((8, y + cell + label_height / 2), f"sample {sample['id']}", fill="#222", font=label_font, anchor="lm")
        sheet.save(contact_dir / f"contact_sheet_{page_index}.png")
    print(f"wrote {len(manifest['items'])} renders and contact sheets to {contact_dir}")


if __name__ == "__main__":
    main()
