"""Pure Python read/write for .annocalib calibration ratio files."""

import json


def write_calibration_ratio(
    path: str,
    scale: float,
    unit: str,
    grid_spacing_world: float,
    grid_spacing_auto: bool,
) -> None:
    data = {
        "scale_world_per_px": scale,
        "unit": unit,
        "grid_spacing_world": grid_spacing_world,
        "grid_spacing_auto": grid_spacing_auto,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_calibration_ratio(path: str) -> dict:
    """Return dict with keys: scale_world_per_px, unit, grid_spacing_world, grid_spacing_auto."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "scale_world_per_px": float(data["scale_world_per_px"]),
        "unit": str(data["unit"]),
        "grid_spacing_world": float(data.get("grid_spacing_world", 100.0)),
        "grid_spacing_auto": bool(data.get("grid_spacing_auto", True)),
    }
