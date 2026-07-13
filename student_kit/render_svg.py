#!/usr/bin/env python3
"""Render safe SVG samples from results.json for report inspection."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from reward import reward


def render_failure_card(destination: Path, size: int, violations: list[str]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (size, size), "#FFF7F5")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((18, 18, size - 18, size - 18), radius=22, outline="#C94C4C", width=5)
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    title_font = ImageFont.truetype(font_path, max(20, size // 18))
    body_font = ImageFont.truetype(font_path, max(14, size // 30))
    draw.text((size // 2, size // 3), "INVALID SVG", fill="#9E2F2F", font=title_font, anchor="mm")
    lines = violations[:3] or ["fatal validation failure"]
    y = size // 2
    for line in lines:
        text = line.replace("_", " ")[:38]
        draw.text((size // 2, y), text, fill="#633", font=body_font, anchor="mm")
        y += size // 13
    canvas.save(destination)


def render(svg: str, destination: Path, size: int) -> None:
    """Use CairoSVG when available, otherwise a local headless Chromium browser."""
    try:
        import cairosvg

        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(destination),
            output_width=size,
            output_height=size,
        )
        return
    except (ImportError, OSError):
        pass

    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    browser = next((item for item in candidates if item and Path(item).exists()), None)
    if browser is None:
        raise RuntimeError("SVG rendering requires CairoSVG with libcairo or a local Chrome/Chromium")
    with tempfile.TemporaryDirectory(prefix="svg-render-") as temp_dir:
        source = Path(temp_dir) / "image.svg"
        source.write_text(svg, encoding="utf-8")
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--window-size={size},{size}",
                f"--screenshot={destination.resolve()}",
                source.resolve().as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="results.json")
    parser.add_argument("--output-dir", default="output/rendered")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for sample in data["samples"]:
        prompt = sample["prompt"]
        for variant in ("reference", "base", "tuned"):
            svg = sample["reference_svg"] if variant == "reference" else sample[variant]["svg"]
            assessment = reward(prompt, svg)
            destination = output / f"sample_{sample['id']:02d}_{variant}.png"
            if assessment["metadata"]["fatal"]:
                render_failure_card(destination, args.size, assessment["violations"])
                continue
            if destination.exists():
                continue
            try:
                render(svg, destination, args.size)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                print(f"skipped {destination.name}: browser render failed ({type(error).__name__})")
    print(f"rendered samples to {output}")


if __name__ == "__main__":
    main()
