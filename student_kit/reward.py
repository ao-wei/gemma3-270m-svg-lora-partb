"""Deterministic proxy reward for detailed-prompt to SVG generation.

The reward is deliberately conservative: it scores properties that can be
checked without executing the SVG and exposes every component for analysis.
It is a training/model-selection proxy, not a substitute for visual review.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any


WEIGHTS = {
    "syntax_safety": 0.30,
    "geometry": 0.20,
    "structure_style": 0.15,
    "prompt_fidelity": 0.25,
    "anti_degeneracy": 0.10,
}
REWARD_VERSION = "2.0"

VISIBLE_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line", "text"}
FORBIDDEN_TAGS = {"script", "image", "iframe", "object", "embed", "foreignobject", "audio", "video"}
COLOR_ATTRS = {"fill", "stroke", "stop-color", "color", "flood-color", "lighting-color"}
EXTERNAL_VALUE = re.compile(r"(?:https?:|data:|javascript:|file:|//)", re.I)
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b")
RGB_COLOR = re.compile(r"rgba?\([^)]*\)", re.I)

COLOR_NAMES = {
    "black", "white", "gray", "grey", "red", "orange", "yellow", "green", "teal",
    "blue", "navy", "purple", "violet", "pink", "brown", "cream", "gold", "golden",
    "coral", "cyan", "magenta", "beige", "maroon", "indigo", "turquoise",
}

SHAPE_RULES = {
    "circle": {"circle", "ellipse"},
    "circular": {"circle", "ellipse"},
    "ring": {"circle", "ellipse"},
    "ellipse": {"ellipse"},
    "oval": {"ellipse"},
    "square": {"rect"},
    "rectangle": {"rect"},
    "rectangular": {"rect"},
    "line": {"line", "polyline", "path"},
    "triangle": {"polygon", "path"},
    "polygon": {"polygon"},
    "curve": {"path"},
    "curved": {"path"},
    "arc": {"path"},
    "text": {"text"},
    "letter": {"text", "path"},
    "wordmark": {"text", "path"},
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _clip(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def _extract_svg(text: str) -> tuple[str | None, list[str]]:
    violations: list[str] = []
    if not isinstance(text, str) or not text.strip():
        return None, ["empty_output"]
    if re.search(r"<!DOCTYPE|<!ENTITY", text, re.I):
        violations.append("doctype_or_entity_forbidden")
    starts = list(re.finditer(r"<svg\b", text, re.I))
    ends = list(re.finditer(r"</svg\s*>", text, re.I))
    if len(starts) != 1 or len(ends) != 1:
        violations.append("not_exactly_one_svg")
        return None, violations
    start, end = starts[0].start(), ends[0].end()
    if text[:start].strip() or text[end:].strip():
        violations.append("non_svg_wrapper_text")
    return text[start:end], violations


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = NUMBER.fullmatch(value.strip())
    if not match:
        return None
    number = float(match.group())
    return number if math.isfinite(number) else None


def _syntax_safety(svg: str | None, initial: list[str]) -> tuple[float, ET.Element | None, list[str], bool]:
    violations = list(initial)
    if svg is None:
        return 0.0, None, violations, True
    if len(svg) > 60_000:
        violations.append("svg_too_large")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        violations.append("xml_parse_error")
        return 0.0, None, violations, True
    if _local_name(root.tag) != "svg":
        violations.append("root_is_not_svg")
        return 0.0, root, violations, True
    namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
    if namespace != "http://www.w3.org/2000/svg":
        violations.append("invalid_svg_namespace")
        return 0.0, root, violations, True

    unsafe = False
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag in FORBIDDEN_TAGS:
            violations.append(f"forbidden_tag:{tag}")
            unsafe = True
        for raw_key, raw_value in element.attrib.items():
            key = _local_name(raw_key)
            value = str(raw_value).strip()
            if key.startswith("on"):
                violations.append(f"event_handler:{key}")
                unsafe = True
            if key in {"href", "src"} and (EXTERNAL_VALUE.search(value) or not value.startswith("#")):
                violations.append(f"external_reference:{key}")
                unsafe = True
            if "url(" in value.lower() and EXTERNAL_VALUE.search(value):
                violations.append("external_url")
                unsafe = True

    score = 1.0
    score -= 0.22 * ("non_svg_wrapper_text" in violations)
    score -= 0.15 * ("svg_too_large" in violations)
    if any(v in violations for v in ("doctype_or_entity_forbidden",)) or unsafe:
        score = 0.0
    return _clip(score), root, violations, unsafe


def _geometry(root: ET.Element | None, violations: list[str]) -> float:
    if root is None:
        return 0.0
    viewbox_values = [float(x) for x in NUMBER.findall(root.attrib.get("viewBox", ""))]
    if len(viewbox_values) != 4 or not all(math.isfinite(x) for x in viewbox_values):
        violations.append("invalid_viewbox")
        return 0.0
    min_x, min_y, width, height = viewbox_values
    if width <= 0 or height <= 0:
        violations.append("non_positive_viewbox")
        return 0.0
    score = 1.0
    if any(abs(a - b) > 1e-6 for a, b in zip(viewbox_values, (0.0, 0.0, 256.0, 256.0))):
        violations.append("nonstandard_viewbox")
        score -= 0.12

    coordinate_checks = 0
    outside = 0
    extreme = False
    x_attrs = {"x", "cx", "x1", "x2", "fx"}
    y_attrs = {"y", "cy", "y1", "y2", "fy"}
    positive_attrs = {"width", "height", "r", "rx", "ry", "stroke-width"}
    margin_x, margin_y = width * 0.25, height * 0.25

    for element in root.iter():
        tag = _local_name(element.tag)
        rect_x = _parse_number(element.attrib.get("x")) if tag == "rect" else None
        rect_y = _parse_number(element.attrib.get("y")) if tag == "rect" else None
        rect_w = _parse_number(element.attrib.get("width")) if tag == "rect" else None
        rect_h = _parse_number(element.attrib.get("height")) if tag == "rect" else None
        oversized_background = (
            tag == "rect"
            and None not in (rect_x, rect_y, rect_w, rect_h)
            and rect_x <= min_x
            and rect_y <= min_y
            and rect_x + rect_w >= min_x + width
            and rect_y + rect_h >= min_y + height
        )
        for key, raw in element.attrib.items():
            name = _local_name(key)
            lower = str(raw).lower()
            if re.search(r"(?<![a-z])(?:nan|[-+]?inf(?:inity)?)(?![a-z])", lower):
                violations.append("non_finite_number")
                extreme = True
            if name in positive_attrs:
                value = _parse_number(raw)
                if value is not None and value < 0:
                    violations.append(f"negative_{name}")
                    outside += 1
                coordinate_checks += value is not None
            if name in x_attrs or name in y_attrs:
                value = _parse_number(raw)
                if value is None:
                    continue
                if oversized_background and name in {"x", "y"}:
                    continue
                coordinate_checks += 1
                low, high, margin = (min_x, min_x + width, margin_x) if name in x_attrs else (min_y, min_y + height, margin_y)
                outside += not (low - margin <= value <= high + margin)
                extreme |= abs(value) > 4 * max(width, height)
            if name == "points":
                nums = [float(x) for x in NUMBER.findall(raw)]
                for index, value in enumerate(nums):
                    coordinate_checks += 1
                    low, high, margin = (min_x, min_x + width, margin_x) if index % 2 == 0 else (min_y, min_y + height, margin_y)
                    outside += not (low - margin <= value <= high + margin)
                    extreme |= abs(value) > 4 * max(width, height)
            if name == "d":
                nums = [float(x) for x in NUMBER.findall(raw)]
                extreme |= any(abs(x) > 8 * max(width, height) for x in nums)

    if coordinate_checks:
        ratio = outside / coordinate_checks
        score -= min(0.65, ratio * 2.0)
        if ratio > 0.15:
            violations.append("many_coordinates_outside_viewbox")
    if extreme:
        violations.append("extreme_or_non_finite_geometry")
        score -= 0.5
    return _clip(score)


def _collect_colors(root: ET.Element) -> set[str]:
    colors: set[str] = set()
    for element in root.iter():
        for raw_key, raw_value in element.attrib.items():
            key, value = _local_name(raw_key), str(raw_value).lower().strip()
            if key not in COLOR_ATTRS and key != "style":
                continue
            colors.update(x.lower() for x in HEX_COLOR.findall(value))
            colors.update(x.lower() for x in RGB_COLOR.findall(value))
            words = set(re.findall(r"[a-z]+", value))
            colors.update(words & COLOR_NAMES)
    return {x for x in colors if x not in {"none", "transparent"}}


def _element_box(element: ET.Element) -> tuple[float, float, float, float] | None:
    tag = _local_name(element.tag)
    if tag == "rect":
        values = [_parse_number(element.attrib.get(key)) for key in ("x", "y", "width", "height")]
        if None not in values:
            x, y, width, height = values
            return x, y, x + width, y + height
    if tag == "circle":
        cx, cy, radius = (_parse_number(element.attrib.get(key)) for key in ("cx", "cy", "r"))
        if None not in (cx, cy, radius):
            return cx - radius, cy - radius, cx + radius, cy + radius
    if tag == "ellipse":
        cx, cy, rx, ry = (_parse_number(element.attrib.get(key)) for key in ("cx", "cy", "rx", "ry"))
        if None not in (cx, cy, rx, ry):
            return cx - rx, cy - ry, cx + rx, cy + ry
    if tag == "line":
        values = [_parse_number(element.attrib.get(key)) for key in ("x1", "y1", "x2", "y2")]
        if None not in values:
            x1, y1, x2, y2 = values
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if tag in {"polygon", "polyline", "path"}:
        source = element.attrib.get("d", "") if tag == "path" else element.attrib.get("points", "")
        numbers = [float(value) for value in NUMBER.findall(source)]
        if len(numbers) >= 4:
            xs, ys = numbers[::2], numbers[1::2]
            return min(xs), min(ys), max(xs), max(ys)
    return None


def _structure_style(root: ET.Element | None, violations: list[str]) -> tuple[float, dict[str, Any]]:
    if root is None:
        return 0.0, {"visible_elements": 0, "colors": []}
    elements = [element for element in root.iter() if _local_name(element.tag) in VISIBLE_TAGS]
    visible = [_local_name(element.tag) for element in elements]
    colors = _collect_colors(root)
    viewbox = [float(x) for x in NUMBER.findall(root.attrib.get("viewBox", ""))]
    on_canvas, canvas_cover = 0, 0
    if len(viewbox) == 4 and viewbox[2] > 0 and viewbox[3] > 0:
        min_x, min_y, width, height = viewbox
        canvas_area = width * height
        for element in elements:
            style = str(element.attrib.get("style", "")).lower()
            opacity = _parse_number(element.attrib.get("opacity"))
            if "display:none" in style.replace(" ", "") or element.attrib.get("display") == "none" or opacity == 0:
                continue
            box = _element_box(element)
            if box is None:
                on_canvas += 1
                continue
            x0, y0, x1, y1 = box
            overlap_x = max(0.0, min(x1, min_x + width) - max(x0, min_x))
            overlap_y = max(0.0, min(y1, min_y + height) - max(y0, min_y))
            intersection = overlap_x * overlap_y
            fraction = intersection / canvas_area
            line_like_on_canvas = (
                (overlap_x > 0 or overlap_y > 0)
                and (x1 - x0 >= 1 or y1 - y0 >= 1)
                and x1 >= min_x and x0 <= min_x + width and y1 >= min_y and y0 <= min_y + height
            )
            if fraction >= 0.005 or line_like_on_canvas:
                on_canvas += 1
            if fraction >= 0.72:
                canvas_cover += 1
    serialized = [ET.tostring(element, encoding="unicode") for element in elements]
    duplicate_count = sum(count - 1 for count in Counter(serialized).values() if count > 1)
    meaningful_foreground = max(0, on_canvas - canvas_cover)
    score = 1.0
    if not visible:
        violations.append("no_visible_elements")
        score = 0.0
    elif len(visible) == 1:
        violations.append("single_visible_element")
        score -= 0.45
    elif len(visible) > 80:
        violations.append("excessive_visible_elements")
        score -= min(0.6, (len(visible) - 80) / 100)
    if visible and on_canvas == 0:
        violations.append("no_on_canvas_elements")
        score = 0.0
    elif on_canvas and meaningful_foreground == 0:
        violations.append("background_only_or_blank")
        score = min(score, 0.10)
    if duplicate_count:
        violations.append("repeated_identical_elements")
        score -= min(0.7, 0.25 * duplicate_count)
    if not colors:
        violations.append("no_explicit_colors")
        score -= 0.35
    elif len(colors) > 12:
        violations.append("excessive_palette")
        score -= min(0.45, (len(colors) - 12) / 20)
    return _clip(score), {
        "visible_elements": len(visible),
        "on_canvas_elements": on_canvas,
        "canvas_cover_elements": canvas_cover,
        "meaningful_foreground_elements": meaningful_foreground,
        "duplicate_visible_elements": duplicate_count,
        "colors": sorted(colors),
        "tag_counts": dict(Counter(visible)),
    }


def _prompt_fidelity(prompt: str, root: ET.Element | None, metadata: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if root is None:
        return 0.0, {"prompt_colors": [], "color_coverage": 0.0, "shape_coverage": 0.0}
    prompt_lower = (prompt or "").lower()
    prompt_colors = {x.lower() for x in HEX_COLOR.findall(prompt_lower)}
    prompt_colors |= set(re.findall(r"\b[a-z]+\b", prompt_lower)) & COLOR_NAMES
    svg_colors = set(metadata["colors"])

    if prompt_colors:
        color_hits = 0
        for color in prompt_colors:
            if color in svg_colors:
                color_hits += 1
            elif color in {"gray", "grey"} and ({"gray", "grey"} & svg_colors):
                color_hits += 1
            elif color == "golden" and ({"gold", "golden"} & svg_colors):
                color_hits += 1
        color_coverage = color_hits / len(prompt_colors)
    else:
        color_coverage = 1.0

    tags = set(metadata["tag_counts"])
    requested_shapes = {word: allowed for word, allowed in SHAPE_RULES.items() if re.search(rf"\b{re.escape(word)}\b", prompt_lower)}
    shape_coverage = (
        sum(bool(tags & allowed) for allowed in requested_shapes.values()) / len(requested_shapes)
        if requested_shapes else 1.0
    )

    # A logo should use the canvas rather than collapse to a tiny or empty fragment.
    composition = 1.0 if metadata.get("meaningful_foreground_elements", 0) >= 1 else 0.0
    score = 0.55 * color_coverage + 0.30 * shape_coverage + 0.15 * composition
    details = {
        "prompt_colors": sorted(prompt_colors),
        "svg_colors": sorted(svg_colors),
        "color_coverage": _clip(color_coverage),
        "requested_primitive_terms": sorted(requested_shapes),
        "shape_coverage": _clip(shape_coverage),
        "composition": composition,
    }
    return _clip(score), details


def _anti_degeneracy(svg: str | None, root: ET.Element | None, metadata: dict[str, Any], violations: list[str]) -> float:
    if not svg or root is None:
        return 0.0
    score = 1.0
    length = len(svg)
    if length < 180:
        violations.append("suspiciously_short")
        score -= 0.7
    if length > 30_000:
        violations.append("suspiciously_long")
        score -= 0.35
    if "```" in svg or "<start_of_turn>" in svg or "assistant" in svg[:80].lower():
        violations.append("format_or_template_leakage")
        score -= 0.6
    chunks = re.findall(r"<([a-zA-Z][\w:-]*)\b[^>]*>", svg)
    if chunks:
        most_common = Counter(chunks).most_common(1)[0][1]
        if most_common > 50 and most_common / len(chunks) > 0.75:
            violations.append("repetitive_elements")
            score -= 0.55
    repeated_text = re.search(r"(.{40,200})\1{2,}", svg, re.S)
    if repeated_text:
        violations.append("repeated_long_fragment")
        score -= 0.65
    if "background_only_or_blank" in violations or "no_on_canvas_elements" in violations:
        score = 0.0
    if metadata.get("duplicate_visible_elements", 0):
        score -= min(0.8, 0.3 * metadata["duplicate_visible_elements"])
    if metadata.get("visible_elements", 0) > 120:
        score -= 0.25
    return _clip(score)


def reward(prompt: str, generated_svg: str) -> dict[str, Any]:
    """Score a generated SVG using deterministic, non-executing checks."""
    svg, initial = _extract_svg(generated_svg)
    syntax, root, violations, unsafe = _syntax_safety(svg, initial)
    geometry = _geometry(root, violations)
    structure, metadata = _structure_style(root, violations)
    fidelity, fidelity_details = _prompt_fidelity(prompt, root, metadata)
    anti = _anti_degeneracy(svg, root, metadata, violations)
    components = {
        "syntax_safety": syntax,
        "geometry": geometry,
        "structure_style": structure,
        "prompt_fidelity": fidelity,
        "anti_degeneracy": anti,
    }
    total = sum(WEIGHTS[name] * components[name] for name in WEIGHTS)
    fatal = root is None or unsafe or syntax == 0.0
    if fatal:
        total = min(total, 0.10)
    if "no_visible_elements" in violations:
        total = min(total, 0.25)
    if "background_only_or_blank" in violations or "no_on_canvas_elements" in violations:
        total = min(total, 0.35)
    validity = _clip(0.6 * syntax + 0.4 * geometry)
    return {
        "reward_version": REWARD_VERSION,
        "total": _clip(total),
        "validity": validity,
        "fidelity": fidelity,
        "components": components,
        "weights": WEIGHTS.copy(),
        "violations": sorted(set(violations)),
        "metadata": {**metadata, **fidelity_details, "fatal": fatal},
    }


__all__ = ["WEIGHTS", "reward"]
