"""Full-resolution raster overlays for Mapbox GL JS image sources.

History years are reprojected and PNG-encoded once, then served from an
in-process LRU cache and a small on-disk JSON cache. Forecast and Simulate
can reuse ``mask_to_png_data_uri`` later; this module does not call them.
"""

from __future__ import annotations

import base64
import functools
import json
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import array_bounds
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from scipy.ndimage import label

# Existing History fill / legend swatch (#D97706). Alpha stays 255 so the
# map opacity slider is the only transparency control.
HISTORY_RGBA: tuple[int, int, int, int] = (217, 119, 6, 255)

# Forecast overlay: muted amber baseline under vivid coral expansion.
FORECAST_BASELINE_RGBA: tuple[int, int, int, int] = (217, 119, 6, 145)
FORECAST_EXPANSION_RGBA: tuple[int, int, int, int] = (239, 68, 68, 255)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "history"
_KEY_LOCKS: dict[tuple, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _lock_for(key: tuple) -> threading.Lock:
    """One lock per cache key so different years can build in parallel."""
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


def rgba_to_hex(rgba_color: tuple[int, int, int, int]) -> str:
    red, green, blue, _alpha = rgba_color
    return f"#{red:02X}{green:02X}{blue:02X}"


def mask_to_png_data_uri(
    mask: np.ndarray,
    rgba_color: tuple[int, int, int, int],
) -> str:
    """Encode a 0/1 mask as a transparent RGBA PNG data URI.

    Pixels where ``mask == 1`` receive ``rgba_color``. Every other pixel is
    fully transparent. ``mask`` is the destination grid (already warped).
    """
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[mask == 1] = rgba_color
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def categorical_to_png_data_uri(
    categories: np.ndarray,
    palette: dict[int, tuple[int, int, int, int]],
) -> str:
    """Encode class ids as a transparent RGBA PNG data URI.

    Class ids missing from ``palette`` stay fully transparent.
    """
    height, width = categories.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for value, color in palette.items():
        rgba[categories == value] = color
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _warp_categorical(
    categories: np.ndarray,
    src_transform,
    src_crs,
) -> tuple[np.ndarray, list[list[float]]]:
    """Nearest-neighbor warp of a class raster to Web Mercator image corners."""
    height, width = categories.shape
    left, bottom, right, top = array_bounds(height, width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        "EPSG:3857",
        width,
        height,
        left,
        bottom,
        right,
        top,
    )
    dst = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=categories,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        resampling=Resampling.nearest,
    )
    left, bottom, right, top = array_bounds(dst_height, dst_width, dst_transform)
    west, south, east, north = transform_bounds(
        "EPSG:3857",
        "EPSG:4326",
        left,
        bottom,
        right,
        top,
    )
    coordinates = [
        [float(west), float(north)],
        [float(east), float(north)],
        [float(east), float(south)],
        [float(west), float(south)],
    ]
    return dst, coordinates


def mask_to_mapbox_overlay(
    binary_2030: np.ndarray,
    src_transform,
    src_crs,
    baseline_2025: np.ndarray | None = None,
) -> dict:
    """Build a two-tone Mapbox overlay from in-memory forecast masks.

    ``baseline_2025`` must already sit on the same grid as ``binary_2030``.
    Category 1 is the retained 2025 urban core. Category 2 is 2030 expansion
    that is not already baseline. No polygon filter is applied.
    """
    predicted = np.asarray(binary_2030).astype(bool)
    if baseline_2025 is None:
        baseline = np.zeros(predicted.shape, dtype=bool)
    else:
        baseline = np.asarray(baseline_2025).astype(bool)
        if baseline.shape != predicted.shape:
            raise ValueError(
                f"baseline_2025 shape {baseline.shape} does not match "
                f"forecast grid {predicted.shape}"
            )

    categories = np.zeros(predicted.shape, dtype=np.uint8)
    categories[baseline] = 1
    categories[predicted & ~baseline] = 2
    warped, coordinates = _warp_categorical(categories, src_transform, src_crs)
    return {
        "coordinates": coordinates,
        "image_url": categorical_to_png_data_uri(
            warped,
            {
                1: FORECAST_BASELINE_RGBA,
                2: FORECAST_EXPANSION_RGBA,
            },
        ),
        "baseline_color": rgba_to_hex(FORECAST_BASELINE_RGBA),
        "expansion_color": rgba_to_hex(FORECAST_EXPANSION_RGBA),
    }


def raster_to_mapbox_overlay(
    tif_path: str,
    rgba_color: tuple[int, int, int, int],
    year: int,
) -> dict:
    """Read a binary urban GeoTIFF and return a Mapbox image-source payload.

    Area and patch counts are computed on the native meter grid before the
    nearest-neighbor warp to Web Mercator. No sieve, morphology, or value
    filter other than ``arr == 1`` is applied.
    """
    with rasterio.open(tif_path) as src:
        binary = src.read(1) == 1
        pixel_km2 = abs(src.transform.a * src.transform.e) / 1e6
        area_km2 = round(float(binary.sum()) * pixel_km2, 1)
        _labeled, num_patches = label(binary)
        pixel_count = int(binary.sum())
        resolution_m = int(round(abs(src.transform.a)))

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            "EPSG:3857",
            src.width,
            src.height,
            *src.bounds,
        )
        dst_arr = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=binary.astype(np.uint8),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.nearest,
        )
        left, bottom, right, top = array_bounds(dst_height, dst_width, dst_transform)
        west, south, east, north = transform_bounds(
            "EPSG:3857",
            "EPSG:4326",
            left,
            bottom,
            right,
            top,
        )

    coordinates = [
        [float(west), float(north)],
        [float(east), float(north)],
        [float(east), float(south)],
        [float(west), float(south)],
    ]
    return {
        "year": int(year),
        "area_km2": area_km2,
        "patches": int(num_patches),
        "pixel_count": pixel_count,
        "resolution_m": resolution_m,
        "color": rgba_to_hex(rgba_color),
        "coordinates": coordinates,
        "image_url": mask_to_png_data_uri(dst_arr, rgba_color),
    }


def _disk_cache_path(
    year: int,
    mtime_ns: int,
    rgba_color: tuple[int, int, int, int],
) -> Path:
    color_key = rgba_to_hex(rgba_color).removeprefix("#")
    return _CACHE_DIR / f"{year}-{mtime_ns}-{color_key}.json"


@functools.lru_cache(maxsize=16)
def load_history_overlay(
    year: int,
    tif_path: str,
    mtime_ns: int,
    rgba_color: tuple[int, int, int, int] = HISTORY_RGBA,
) -> dict:
    """Return the History overlay for one year, computing it at most once.

    ``mtime_ns`` is part of the cache key so a replaced GeoTIFF is rebuilt.
    The labeled array from ``scipy.ndimage.label`` is not retained.
    """
    cache_path = _disk_cache_path(year, mtime_ns, rgba_color)
    key = (year, tif_path, mtime_ns, rgba_color)
    with _lock_for(key):
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        payload = raster_to_mapbox_overlay(tif_path, rgba_color, year)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(cache_path)
        return payload
