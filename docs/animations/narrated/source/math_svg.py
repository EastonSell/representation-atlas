#!/usr/bin/env python3
"""Render Computer Modern math as cleaned SVG paths; optionally import in Manim.

CLI example:
  python math_svg.py '\\frac{dL}{d\\theta}=\\theta-F(\\theta)' \
      --output ./formula.svg --cache-dir ./math_cache

Import math_mobject(source, size=1, color='#F4F4F4', cache_dir=...) for a Manim
SVGMobject with a common baseline and advance_width, or equation(*parts, ...)
for a centered row. A part is a string or (source, color) pair. These use
Matplotlib mathtext, not external TeX: keep expressions within its supported
subset. Imports are lazy so --help needs only Python's standard library.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from xml.etree import ElementTree as ET


CACHE_VERSION = 1


@dataclass(frozen=True)
class MathSVG:
    path: Path
    source: str
    width: float
    height: float
    depth: float
    font_size: float


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _expression(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Math expression must be a nonempty string")
    source = source.strip()
    if source.startswith("$") and source.endswith("$") and not source.startswith("$$"):
        source = source[1:-1].strip()
    if not source:
        raise ValueError("Math expression must contain visible content")
    return source


def _clean_svg(path: Path) -> None:
    """Remove the figure background and zero-outline glyphs, including spaces.

    Empty glyph definitions and their <use> references can make Manim's SVG
    parser fail. This removes both, preserving all visible vector outlines.
    """
    tree = ET.parse(path)
    empty_ids = {element.attrib.get("id") for element in tree.iter()
                 if element.tag.rsplit("}", 1)[-1] == "path"
                 and not element.attrib.get("d", "").strip()}
    for parent in tree.iter():
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1]
            href = child.attrib.get("{http://www.w3.org/1999/xlink}href", child.attrib.get("href", ""))
            if (child.attrib.get("id", "").startswith("patch_")
                or (tag == "path" and not child.attrib.get("d", "").strip())
                or (tag == "use" and href.removeprefix("#") in empty_ids)):
                parent.remove(child)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def render_math_svg(source: str, *, cache_dir: Path | str, font_size: float = 48,
                    color: str = "white") -> MathSVG:
    """Return a cached SVG and mathtext metrics in SVG point units (72 dpi)."""
    source = _expression(source)
    font_size = _positive(font_size, "font_size")
    try:
        import matplotlib
        from matplotlib import rc_context
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import MathTextParser, math_to_image
    except ImportError as exc:
        raise RuntimeError("Install SVG math dependency: python -m pip install matplotlib") from exc
    directory = Path(cache_dir).expanduser().resolve()
    params = {"source": source, "font_size": font_size, "color": color,
              "fontset": "cm", "matplotlib": matplotlib.__version__, "cache_version": CACHE_VERSION}
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    target = directory / f"{key}.svg"
    metrics_file = directory / f"{key}.json"
    if target.is_file() and metrics_file.is_file():
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            if metrics.get("parameters") == params:
                return MathSVG(target, source, metrics["width"], metrics["height"], metrics["depth"], font_size)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    prop = FontProperties(size=font_size, math_fontfamily="cm")
    with rc_context({"mathtext.fontset": "cm", "svg.fonttype": "path", "text.usetex": False,
                     "savefig.transparent": True, "svg.hashsalt": key}):
        width, height, depth, _, _ = MathTextParser("path").parse(f"${source}$", dpi=72, prop=prop)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{key}.", suffix=".svg", delete=False) as handle:
                temporary = Path(handle.name)
            math_to_image(f"${source}$", str(temporary), format="svg", prop=prop, color=color, dpi=72)
            _clean_svg(temporary)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    metrics = {"parameters": params, "width": float(width), "height": float(height), "depth": float(depth)}
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return MathSVG(target, source, float(width), float(height), float(depth), font_size)


def math_mobject(source: str, size: float = 1.0, color="#F4F4F4", *, cache_dir=None):
    """Return an animatable Manim SVG with x=0 origin and a common y=0 baseline.

    The 0.015 point-to-scene scale reproduces the tested 48-point baseline
    helper. advance_width includes mathtext spacing even for slim glyphs.
    """
    size = _positive(size, "size")
    try:
        from manim import DOWN, LEFT, ORIGIN, SVGMobject, config
    except ImportError as exc:
        raise RuntimeError("Install the scene dependency: python -m pip install manim") from exc
    directory = Path(cache_dir) if cache_dir is not None else Path(config.media_dir) / "math_svg"
    asset = render_math_svg(source, cache_dir=directory)
    obj = SVGMobject(str(asset.path), height=None, width=None, should_center=False)
    if not obj.family_members_with_points():
        raise ValueError("Math expression generated no visible vector paths")
    baseline = obj.get_bottom()[1] + obj.get_top()[1] - (asset.height - asset.depth)
    obj.shift(LEFT * obj.get_left()[0] + DOWN * baseline)
    obj.set_color(color).scale(0.015 * size, about_point=ORIGIN)
    obj.advance_width = asset.width * 0.015 * size
    obj.source = asset.source
    return obj


def equation(*parts, size: float = 1.0, buff: float = 0.13, cache_dir=None, max_width=None):
    """Compose baseline-aligned SVG chunks, retaining each chunk for animation."""
    if not parts:
        raise ValueError("An equation needs at least one part")
    if isinstance(buff, bool) or not isinstance(buff, (int, float)) or not math.isfinite(buff) or buff < 0:
        raise ValueError("buff must be a finite nonnegative number")
    if max_width is not None:
        max_width = _positive(max_width, "max_width")
    try:
        from manim import ORIGIN, RIGHT, VGroup
    except ImportError as exc:
        raise RuntimeError("Install the scene dependency: python -m pip install manim") from exc
    group = VGroup()
    cursor = 0.0
    for part in parts:
        if isinstance(part, str):
            chunk = math_mobject(part, size=size, cache_dir=cache_dir)
        elif isinstance(part, (tuple, list)) and len(part) == 2:
            chunk = math_mobject(part[0], size=size, color=part[1], cache_dir=cache_dir)
        else:
            raise ValueError("An equation part must be a source string or (source, color) pair")
        chunk.shift(RIGHT * cursor)
        group.add(chunk)
        cursor += chunk.advance_width + buff
    group.move_to(ORIGIN)
    if max_width is not None and group.width > max_width:
        group.scale_to_fit_width(max_width)
    return group


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("expression", help="Mathtext expression, with or without single dollar delimiters")
    parser.add_argument("--output", type=Path, required=True, help="Destination SVG path")
    parser.add_argument("--cache-dir", type=Path, help="Reusable cache; defaults to .math_svg_cache beside the output")
    parser.add_argument("--font-size", type=float, default=48)
    parser.add_argument("--color", default="white")
    args = parser.parse_args(argv)
    try:
        cache = args.cache_dir if args.cache_dir is not None else args.output.parent / ".math_svg_cache"
        asset = render_math_svg(args.expression, cache_dir=cache, font_size=args.font_size, color=args.color)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.resolve() != asset.path.resolve():
            shutil.copyfile(asset.path, args.output)
        metadata = asdict(asset)
        metadata["path"] = str(args.output.resolve())
        print(json.dumps(metadata, ensure_ascii=True))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
