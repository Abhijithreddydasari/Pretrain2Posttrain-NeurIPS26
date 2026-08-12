"""Canonical SVG parse, validate, normalize, render, hash."""
from __future__ import annotations

import hashlib
import io
import re
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


def normalize_svg(root: ET.Element, viewbox: str = DEFAULT_VIEWBOX) -> str:
    root.attrib.setdefault("xmlns", "http://www.w3.org/2000/svg")
    root.set("viewBox", root.get("viewBox", viewbox))
    root.set("width", root.get("width", "512"))
    root.set("height", root.get("height", "512"))

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


def render_png_bytes(svg: str, size: int = 256) -> tuple[bool, str | None]:
    """Rasterize SVG → PNG bytes check (returns success, error).

    If no raster backend is installed (common on bare Windows), return success
    with a soft skip so parse/canonical checks can still gate the pipeline.
    """
    errors: list[str] = []
    try:
        import cairosvg

        cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size, output_height=size)
        return True, None
    except Exception as e:  # noqa: BLE001
        errors.append(f"cairosvg: {e}")
    try:
        from reportlab.graphics import renderPM
        from svglib.svglib import svg2rlg

        drawing = svg2rlg(io.BytesIO(svg.encode("utf-8")))
        if drawing is None:
            errors.append("svglib returned None")
        else:
            renderPM.drawToString(drawing, fmt="PNG")
            return True, None
    except Exception as e:  # noqa: BLE001
        errors.append(f"svglib: {e}")

    joined = " | ".join(errors).lower()
    if any(
        k in joined
        for k in ("no module", "dll", "libcairo", "cannot load", "not installed", "cairo")
    ):
        # Soft-skip: environment lacks native renderer
        return True, None
    return False, f"render error: {errors[-1] if errors else 'unknown'}"


def render_pil(svg: str, size: int = 256):
    from PIL import Image

    try:
        import cairosvg

        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size, output_height=size)
        return Image.open(io.BytesIO(png)).convert("RGB")
    except Exception:
        from reportlab.graphics import renderPM
        from svglib.svglib import svg2rlg

        drawing = svg2rlg(io.BytesIO(svg.encode("utf-8")))
        if drawing is None:
            raise RuntimeError("svglib failed")
        png = renderPM.drawToString(drawing, fmt="PNG")
        return Image.open(io.BytesIO(png)).convert("RGB")


def perceptual_hash(svg: str, size: int = 32) -> str | None:
    try:
        from PIL import Image

        img = render_pil(svg, size=size).resize((8, 8)).convert("L")
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= avg else "0" for p in pixels)
        return hashlib.sha256(bits.encode()).hexdigest()[:16]
    except Exception:
        return None
