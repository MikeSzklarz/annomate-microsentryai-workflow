"""Pure Python read/write for plain-text calibration ratio files.

Format (Proposal B):
    {px_count}px:{world_value}{unit}

Examples:
    1px:0.05mm      1 pixel = 0.05 mm
    50px:1mm        50 pixels = 1 mm
    1px:2px         pixel-to-pixel ratio (no physical unit)
    1px:1px         default uncalibrated state

The left side must always be in pixels. The right side defines the world unit.
Scale stored internally as world_units_per_pixel.
"""

import re

_RATIO_RE = re.compile(
    r"^\s*([\d.]+)\s*px\s*:\s*([\d.]+)\s*([a-zA-Z]+)\s*$",
    re.IGNORECASE,
)


def parse_ratio_string(s: str) -> tuple[float, str]:
    """Parse a Proposal B ratio string into (scale_world_per_px, unit).

    Accepts '1px:0.05mm', '50px:1mm', '1px:2px', etc.
    Scale = world_value / px_count.
    """
    m = _RATIO_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"Invalid ratio format: {s!r}\n"
            "Expected e.g. '1px:0.05mm' or '50px:1mm'"
        )
    px_count = float(m.group(1))
    world_val = float(m.group(2))
    unit = m.group(3)
    if px_count <= 0:
        raise ValueError("Pixel count must be greater than zero")
    if world_val <= 0:
        raise ValueError("World value must be greater than zero")
    return world_val / px_count, unit


def format_ratio_string(scale: float, unit: str) -> str:
    """Format scale as '1px:{scale:g}{unit}'."""
    return f"1px:{scale:g}{unit}"


def write_calibration_ratio(path: str, scale: float, unit: str) -> None:
    """Write scale to a plain-text .txt calibration ratio file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_ratio_string(scale, unit) + "\n")


def read_calibration_ratio(path: str) -> dict:
    """Read a calibration ratio .txt file.

    Returns dict with keys: scale_world_per_px (float), unit (str).
    """
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    scale, unit = parse_ratio_string(content)
    return {"scale_world_per_px": scale, "unit": unit}
