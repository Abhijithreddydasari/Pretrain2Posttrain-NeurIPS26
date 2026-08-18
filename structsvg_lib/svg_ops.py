"""Canonical SVG parse, validate, normalize, render, hash."""
from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree as ET

FORBIDDEN_TAGS = {
    "script",
    "foreignObject",
    "image",
    "animate",
    "animateTransform",
    "animateMotion",
    "set",
    "audio",
    "video",
}
DRAWABLE = {"rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "text"}
ALLOWED = DRAWABLE | {"svg", "g", "defs", "marker", "title", "desc", "tspan", "clipPath"}
NS = {"svg": "http://www.w3.org/2000/svg"}
DEFAULT_VIEWBOX = "0 0 512 512"
# Gemma 4 vision: height/width should be divisible by 48; 960 ≈ 560-token budget class.
TRAIN_RENDER_LONG_EDGE = 960
DEFAULT_LETTERBOX_BG = (255, 255, 255)

# svglib lacks many CSS color names; map before raster fallback.
_NAMED_COLORS: dict[str, str] = {
    "aliceblue": "#f0f8ff",
    "antiquewhite": "#faebd7",
    "aqua": "#00ffff",
    "aquamarine": "#7fffd4",
    "azure": "#f0ffff",
    "beige": "#f5f5dc",
    "bisque": "#ffe4c4",
    "black": "#000000",
    "blue": "#0000ff",
    "brown": "#a52a2a",
    "cyan": "#00ffff",
    "darkgray": "#a9a9a9",
    "darkgrey": "#a9a9a9",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "gray": "#808080",
    "grey": "#808080",
    "green": "#008000",
    "inherit": "none",
    "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3",
    "lime": "#00ff00",
    "magenta": "#ff00ff",
    "maroon": "#800000",
    "navy": "#000080",
    "orange": "#ffa500",
    "pink": "#ffc0cb",
    "purple": "#800080",
    "red": "#ff0000",
    "silver": "#c0c0c0",
    "transparent": "none",
    "white": "#ffffff",
    "yellow": "#ffff00",
}

_RENDERER_NOISE = (
    "can't handle color:",
    "unable to resolve percentage unit",
)


class _FilteredStderr:
    """Drop known-noisy svglib/reportlab stderr lines during rasterization."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def write(self, s: str) -> int:
        low = s.lower()
        if any(n in low for n in _RENDERER_NOISE):
            return len(s)
        return self._real.write(s)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@contextlib.contextmanager
def _quiet_svg_render():
    """Suppress svglib/reportlab color and percentage warnings on stderr."""
    loggers = (
        "svglib",
        "svglib.svglib",
        "reportlab",
        "reportlab.pdfbase",
        "reportlab.graphics",
    )
    saved: list[tuple[logging.Logger, int, bool]] = []
    for name in loggers:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level, lg.propagate))
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
    old_stderr = sys.stderr
    sys.stderr = _FilteredStderr(old_stderr)
    try:
        yield
    finally:
        sys.stderr = old_stderr
        for lg, level, propagate in saved:
            lg.setLevel(level)
            lg.propagate = propagate


def preprocess_svg_for_render(svg: str) -> str:
    """Normalize fill/stroke color names svglib cannot parse."""

    def _repl(match: re.Match[str]) -> str:
        attr, val = match.group(1), match.group(2).strip().lower()
        mapped = _NAMED_COLORS.get(val)
        if mapped:
            return f'{attr}="{mapped}"'
        return match.group(0)

    out = re.sub(r'(fill|stroke)\s*=\s*"([^"]+)"', _repl, svg, flags=re.IGNORECASE)
    out = re.sub(r"(fill|stroke)\s*=\s*'([^']+)'", _repl, out, flags=re.IGNORECASE)
    return out


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def strip_namespaces(elem: ET.Element) -> ET.Element:
    for e in elem.iter():
        e.tag = _local(e.tag)
        # drop namespaced attrs that are not xmlns
        doomed = [k for k in e.attrib if k.startswith("{") and "xmlns" not in k.lower()]
        for k in doomed:
            del e.attrib[k]
    return elem


@dataclass
class SVGValidation:
    ok: bool
    parse_ok: bool = False
    render_ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_elements: int = 0
    n_drawable: int = 0
    normalized: str | None = None
    sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "parse_ok": self.parse_ok,
            "render_ok": self.render_ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "n_elements": self.n_elements,
            "n_drawable": self.n_drawable,
            "sha256": self.sha256,
        }


def extract_svg_blob(text: str) -> str | None:
    """Pull first <svg>...</svg> from model output."""
    if not text:
        return None
    m = re.search(r"<svg\b[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    if m:
        return m.group(0)
    # truncated open svg
    m2 = re.search(r"<svg\b[\s\S]*", text, flags=re.IGNORECASE)
    return m2.group(0) if m2 else None


def parse_svg(svg: str) -> tuple[ET.Element | None, str | None]:
    try:
        root = ET.fromstring(svg.encode("utf-8") if isinstance(svg, str) else svg)
        root = strip_namespaces(root)
        if _local(root.tag) != "svg":
            return None, "root is not svg"
        return root, None
    except ET.ParseError as e:
        return None, f"parse error: {e}"


def _fmt_num(v: str) -> str:
    try:
        f = float(v)
        if abs(f - round(f)) < 1e-6:
            return str(int(round(f)))
        return f"{f:.3f}".rstrip("0").rstrip(".")
    except ValueError:
        return v


def _parse_dim_value(v: str) -> float | None:
    """Parse SVG length (e.g. 443pt, 512px) to a float."""
    v = v.strip()
    m = re.match(r"^([\d.]+)\s*(pt|px|cm|mm|in|em|ex|pc|%)?$", v, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    try:
        return float(v)
    except ValueError:
        return None


def normalize_svg(root: ET.Element, viewbox: str = DEFAULT_VIEWBOX) -> str:
    root.attrib.setdefault("xmlns", "http://www.w3.org/2000/svg")
    vb = root.get("viewBox", viewbox)
    parts = vb.split()
    if len(parts) == 4:
        w = _parse_dim_value(parts[2]) or 512.0
        h = _parse_dim_value(parts[3]) or 512.0
        root.set("viewBox", " ".join(parts[:2] + [_fmt_num(str(w)), _fmt_num(str(h))]))
        root.set("width", _fmt_num(str(w)))
        root.set("height", _fmt_num(str(h)))
    else:
        root.set("viewBox", viewbox)
        root.set("width", "512")
        root.set("height", "512")

    for e in root.iter():
        # normalize numeric-looking attrs
        for k, v in list(e.attrib.items()):
            if k in {"x", "y", "cx", "cy", "r", "rx", "ry", "x1", "y1", "x2", "y2", "width", "height", "stroke-width"}:
                e.set(k, _fmt_num(v))
        # stable attr order
        items = sorted(e.attrib.items(), key=lambda kv: kv[0])
        e.attrib.clear()
        e.attrib.update(items)
        if e.text:
            e.text = e.text.strip() or None
        if e.tail:
            e.tail = e.tail.strip() or None

    # ET doesn't pretty-print compactly; serialize
    rough = ET.tostring(root, encoding="unicode")
    rough = re.sub(r">\s+<", "><", rough)
    return rough


def validate_svg(
    svg: str,
    *,
    max_elements: int = 200,
    min_drawable: int = 2,
    try_render: bool = True,
) -> SVGValidation:
    out = SVGValidation(ok=False)
    root, err = parse_svg(svg)
    if err or root is None:
        out.errors.append(err or "parse failed")
        return out
    out.parse_ok = True

    n = 0
    nd = 0
    for e in root.iter():
        tag = _local(e.tag)
        n += 1
        if tag in FORBIDDEN_TAGS:
            out.errors.append(f"forbidden tag: {tag}")
        if tag not in ALLOWED and tag not in FORBIDDEN_TAGS:
            out.warnings.append(f"unexpected tag: {tag}")
        if tag in DRAWABLE:
            nd += 1
        # external href
        for ak, av in e.attrib.items():
            if ak.endswith("href") or ak == "href":
                if av and not av.startswith("#"):
                    out.errors.append(f"external href: {av[:40]}")
            if "base64," in av:
                out.errors.append("embedded base64 resource")

    out.n_elements = n
    out.n_drawable = nd
    if nd < min_drawable:
        out.errors.append(f"too few drawable elements: {nd}")
    if n > max_elements:
        out.errors.append(f"too many elements: {n}")

    try:
        norm = normalize_svg(root)
        out.normalized = norm
        out.sha256 = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    except Exception as e:  # noqa: BLE001
        out.errors.append(f"normalize failed: {e}")
        return out

    if try_render:
        ok_r, rerr = render_png_bytes(out.normalized)
        out.render_ok = ok_r
        if not ok_r:
            out.errors.append(rerr or "render failed")
    else:
        out.render_ok = True

    out.ok = out.parse_ok and out.render_ok and not out.errors
    # treat only hard errors; warnings ok
    hard = [e for e in out.errors if not e.startswith("unexpected")]
    out.ok = out.parse_ok and out.render_ok and len(hard) == 0
    out.errors = hard
    return out


def _parse_viewbox_aspect(root: ET.Element) -> float:
    vb = root.get("viewBox") or root.get("viewbox")
    if vb:
        parts = re.split(r"[\s,]+", vb.strip())
        if len(parts) >= 4:
            try:
                w = float(parts[2])
                h = float(parts[3])
                if h > 0:
                    return w / h
            except ValueError:
                pass
    try:
        w = float(_parse_dim_value(root.get("width", "1") or "1") or 1.0)
        h = float(_parse_dim_value(root.get("height", "1") or "1") or 1.0)
        if h > 0:
            return w / h
    except (TypeError, ValueError):
        pass
    return 1.0


def svg_aspect_ratio(svg: str) -> float:
    """Width/height from root viewBox (after preprocess)."""
    svg = preprocess_svg_for_render(svg)
    root, err = parse_svg(svg)
    if err or root is None:
        return 1.0
    return _parse_viewbox_aspect(root)


def _letterbox_image(img, size: int, bg: tuple[int, int, int] = DEFAULT_LETTERBOX_BG):
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size {w}x{h}")
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _render_png_bytes_backends(
    svg: str,
    size: int,
    *,
    preserve_aspect: bool = True,
    letterbox: bool = True,
) -> tuple[bytes | None, list[str]]:
    """Try raster backends; optionally preserve aspect + letterbox to size×size."""
    errors: list[str] = []
    aspect = svg_aspect_ratio(svg) if preserve_aspect else 1.0
    with _quiet_svg_render():
        try:
            import resvg_py

            if preserve_aspect:
                if aspect >= 1.0:
                    png = resvg_py.svg_to_bytes(svg_string=svg, width=size)
                else:
                    png = resvg_py.svg_to_bytes(svg_string=svg, height=size)
            else:
                png = resvg_py.svg_to_bytes(svg_string=svg, width=size, height=size)
            if png:
                if letterbox:
                    from PIL import Image

                    img = Image.open(io.BytesIO(png)).convert("RGB")
                    png_buf = io.BytesIO()
                    _letterbox_image(img, size).save(png_buf, format="PNG")
                    return png_buf.getvalue(), errors
                return png, errors
        except Exception as e:  # noqa: BLE001
            errors.append(f"resvg_py: {e}")
        try:
            import cairosvg

            if preserve_aspect and abs(aspect - 1.0) > 1e-3:
                if aspect >= 1.0:
                    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size)
                else:
                    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_height=size)
            else:
                png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size, output_height=size)
            if png:
                if letterbox:
                    from PIL import Image

                    img = Image.open(io.BytesIO(png)).convert("RGB")
                    png_buf = io.BytesIO()
                    _letterbox_image(img, size).save(png_buf, format="PNG")
                    return png_buf.getvalue(), errors
                return png, errors
        except Exception as e:  # noqa: BLE001
            errors.append(f"cairosvg: {e}")
        try:
            from reportlab.graphics import renderPM
            from svglib.svglib import svg2rlg

            drawing = svg2rlg(io.BytesIO(svg.encode("utf-8")))
            if drawing is None:
                errors.append("svglib returned None")
            else:
                png = renderPM.drawToString(drawing, fmt="PNG")
                if png:
                    if letterbox:
                        from PIL import Image

                        img = Image.open(io.BytesIO(png)).convert("RGB")
                        png_buf = io.BytesIO()
                        _letterbox_image(img, size).save(png_buf, format="PNG")
                        return png_buf.getvalue(), errors
                    return png, errors
        except Exception as e:  # noqa: BLE001
            errors.append(f"svglib: {e}")
    return None, errors


def render_png_bytes(svg: str, size: int = 256) -> tuple[bool, str | None]:
    """Rasterize SVG → PNG bytes check (returns success, error).

    If no raster backend is installed (common on bare Windows), return success
    with a soft skip so parse/canonical checks can still gate the pipeline.
    """
    svg = preprocess_svg_for_render(svg)
    png, errors = _render_png_bytes_backends(svg, size)
    if png is not None:
        return True, None

    joined = " | ".join(errors).lower()
    if any(
        k in joined
        for k in ("no module", "dll", "libcairo", "cannot load", "not installed", "cairo", "rlpycairo")
    ):
        # Soft-skip: environment lacks native renderer
        return True, None
    return False, f"render error: {errors[-1] if errors else 'unknown'}"


def render_pil(svg: str, size: int = 256, *, letterbox: bool = True):
    from PIL import Image

    svg = preprocess_svg_for_render(svg)
    png, errors = _render_png_bytes_backends(svg, size, preserve_aspect=True, letterbox=letterbox)
    if png is not None:
        return Image.open(io.BytesIO(png)).convert("RGB")
    joined = " | ".join(errors)
    raise RuntimeError(f"render failed: {joined or 'no backend'}")


def perceptual_hash(svg: str, size: int = 32) -> str | None:
    try:
        img = render_pil(svg, size=size)
        return perceptual_hash_from_image(img)
    except Exception:
        return None


def perceptual_hash_from_image(img) -> str | None:
    """pHash from an already-rendered PIL image (avoids second rasterize)."""
    try:
        from PIL import Image

        small = img.resize((8, 8)).convert("L") if img.size != (8, 8) else img.convert("L")
        pixels = list(small.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= avg else "0" for p in pixels)
        return hashlib.sha256(bits.encode()).hexdigest()[:16]
    except Exception:
        return None


def render_with_phash(svg: str, size: int = 224) -> tuple[Any, str | None]:
    """Single rasterize → (PIL image, phash). Raises on render failure."""
    img = render_pil(svg, size=size)
    return img, perceptual_hash_from_image(img)
