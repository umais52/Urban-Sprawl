"""
Benchmark Script: CDX Point 9 Simulation
========================================
Simulates Point 9 of Change Detection / Urban Sprawl pipeline:
1. Loads a dummy 256x256 spatial tile array with multi-temporal timestamps and multi-spectral channels.
2. Performs on-the-fly temporal patch extraction using NumPy (both zero-copy slicing and materialized patch copies).
3. Measures and reports runtime latency, extraction throughput, and memory usage footprints.
"""

import argparse
import sys
import time
import tracemalloc
import numpy as np


def generate_dummy_tile(
    timesteps: int = 4,
    channels: int = 4,
    height: int = 256,
    width: int = 256,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Generate a dummy multi-temporal, multi-spectral spatial tile array.
    Shape: (timesteps, height, width, channels)
    """
    rng = np.random.default_rng(42)
    tile = rng.random(size=(timesteps, height, width, channels), dtype=dtype)
    return tile


def extract_temporal_patch(
    tile: np.ndarray,
    top: int,
    left: int,
    patch_size: int,
    make_copy: bool = False,
) -> np.ndarray:
    """
    Extract a temporal patch of size (patch_size x patch_size) across all timesteps.
    
    Args:
        tile: Input spatial tile of shape (T, H, W, C)
        top: Row starting index
        left: Column starting index
        patch_size: Height and width of the patch
        make_copy: If True, returns a contiguous copy (materialized in memory);
                   if False, returns a NumPy strided slice view (zero-copy).
    
    Returns:
        Extracted temporal patch of shape (T, patch_size, patch_size, C)
    """
    patch = tile[:, top : top + patch_size, left : left + patch_size, :]
    if make_copy:
        return np.ascontiguousarray(patch)
    return patch


def run_benchmark(
    timesteps: int = 4,
    channels: int = 4,
    tile_size: int = 256,
    patch_size: int = 64,
    iterations: int = 1000,
    warmup: int = 50,
):
    print("=" * 70)
    print("  CDX POINT 9: TEMPORAL PATCH EXTRACTION BENCHMARK")
    print("=" * 70)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"NumPy Version     : {np.__version__}")
    print(f"Tile Dimensions   : {tile_size}x{tile_size} (H x W)")
    print(f"Temporal Steps (T): {timesteps}")
    print(f"Channels (C)      : {channels} (e.g. R, G, B, NIR)")
    print(f"Patch Size (P)    : {patch_size}x{patch_size}")
    print(f"Iterations        : {iterations} (Warmup: {warmup})")
    print("-" * 70)

    # 1. Tile Generation & Memory Footprint
    tracemalloc.start()
    t0_gen = time.perf_counter()
    tile = generate_dummy_tile(
        timesteps=timesteps,
        channels=channels,
        height=tile_size,
        width=tile_size,
        dtype=np.float32,
    )
    t_gen_ms = (time.perf_counter() - t0_gen) * 1000
    gen_current, gen_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tile_bytes = tile.nbytes
    tile_kb = tile_bytes / 1024
    tile_mb = tile_kb / 1024

    print(f"Tile Shape        : {tile.shape} ({tile.dtype})")
    print(f"Tile Buffer Size  : {tile_bytes:,} bytes ({tile_kb:.2f} KB / {tile_mb:.4f} MB)")
    print(f"Generation Time   : {t_gen_ms:.3f} ms")
    print(f"Peak Memory (Gen) : {gen_peak / 1024:.2f} KB")
    print("-" * 70)

    max_top = tile_size - patch_size
    max_left = tile_size - patch_size

    # Pre-generate random coordinates for repeatable benchmark
    np.random.seed(42)
    coords = [
        (np.random.randint(0, max_top + 1), np.random.randint(0, max_left + 1))
        for _ in range(iterations + warmup)
    ]

    # Benchmark both view slicing and copy extraction
    modes = [
        ("Zero-Copy Slice (View)", False),
        ("Materialized Copy (np.ascontiguousarray)", True),
    ]

    for mode_name, make_copy in modes:
        print(f"\n--- Benchmark Mode: {mode_name} ---")

        # Warmup
        for i in range(warmup):
            top, left = coords[i]
            _ = extract_temporal_patch(tile, top, left, patch_size, make_copy=make_copy)

        # Track memory and runtime
        tracemalloc.start()
        tracemalloc.reset_peak()
        t_start = time.perf_counter()

        patches = []
        for i in range(warmup, warmup + iterations):
            top, left = coords[i]
            p = extract_temporal_patch(tile, top, left, patch_size, make_copy=make_copy)
            # Keep patch reference temporarily to measure in-memory footprint if copied
            patches.append(p)

        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time_sec = t_end - t_start
        avg_time_us = (total_time_sec / iterations) * 1_000_000
        avg_time_ms = avg_time_us / 1000
        throughput = iterations / total_time_sec

        single_patch_bytes = patches[0].nbytes
        single_patch_kb = single_patch_bytes / 1024
        total_extracted_mb = (single_patch_bytes * iterations) / (1024 * 1024)

        print(f"Extracted Patch Shape : {patches[0].shape} ({patches[0].dtype})")
        print(f"Single Patch Buffer   : {single_patch_bytes:,} bytes ({single_patch_kb:.2f} KB)")
        print(f"Total Time ({iterations} iters): {total_time_sec * 1000:.3f} ms")
        print(f"Mean Latency          : {avg_time_us:.2f} us ({avg_time_ms:.4f} ms) per patch")
        print(f"Throughput            : {throughput:,.1f} patches/sec")
        print(f"Tracemalloc Current   : {current_mem / 1024:.2f} KB")
        print(f"Tracemalloc Peak      : {peak_mem / 1024:.2f} KB ({peak_mem / (1024*1024):.2f} MB)")
        if make_copy:
            print(f"Total Extracted Data  : {total_extracted_mb:.2f} MB materialized")

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY (CDX POINT 9)")
    print("=" * 70)
    print("  [PASS] Dummy 256x256 spatial tile loaded successfully.")
    print("  [PASS] On-the-fly temporal patch extraction executed.")
    print("  [PASS] Zero-copy view vs. memory copy latency analyzed.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="CDX Point 9: On-the-fly temporal patch extraction benchmark for 256x256 spatial tiles."
    )
    parser.add_argument("--timesteps", type=int, default=4, help="Number of temporal steps (T). Default: 4")
    parser.add_argument("--channels", type=int, default=4, help="Number of spectral channels (C). Default: 4")
    parser.add_argument("--tile-size", type=int, default=256, help="Spatial tile height/width (H=W). Default: 256")
    parser.add_argument("--patch-size", type=int, default=64, help="Extracted patch height/width. Default: 64")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of benchmark iterations. Default: 1000")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations. Default: 50")

    args = parser.parse_args()

    run_benchmark(
        timesteps=args.timesteps,
        channels=args.channels,
        tile_size=args.tile_size,
        patch_size=args.patch_size,
        iterations=args.iterations,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
