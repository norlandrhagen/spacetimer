"""Benchmark: write and read each layout cell, one JSON line per (cell, var, store).

Local smoke test (synthetic data, temp dir):
    uv run python bench/matrix.py --smoke
Screening pass, in-region only (reads dynamical's GFS store, writes to --out):
    uv run python bench/matrix.py --out s3://bucket/prefix > results.jsonl
"""

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

import icechunk
import numpy as np
import obstore as obs
import xarray as xr
import zarr
from obstore.store import from_url
from zarr.codecs import BloscCodec

import spacetimer
from spacetimer import PRESETS, SizeBasis, UseCase, apply_keepbits, convention_keepbits
from spacetimer.core import DEFAULT_COMPRESSORS

RATIO = 0.2  # the recommend()/report() default, used for target scaling and predictions
REGION = "us-west-2"
TILE = 256
COMPRESSORS = {
    "zstd3": DEFAULT_COMPRESSORS,
    "zstd1": (BloscCodec(cname="zstd", clevel=1, shuffle="shuffle"),),
    "lz4": (BloscCodec(cname="lz4", clevel=5, shuffle="shuffle"),),
    "none": (),
}
STORES = ("zarr", "icechunk")


@dataclass(frozen=True)
class Cell:
    preset: UseCase
    cps: int
    scale: float
    compressor: str
    keepbits: bool

    @property
    def id(self) -> str:
        return f"{self.preset}-cps{self.cps}-x{self.scale}-{self.compressor}-kb{int(self.keepbits)}"


def screening(presets: list[UseCase]) -> list[Cell]:
    """Each preset's default, then one axis varied at a time."""
    axes = {
        "cps": [2, 4, 8, 16],
        "scale": [0.5, 2.0, 4.0],
        "compressor": ["zstd1", "lz4", "none"],
        "keepbits": [True],
    }
    cells = []
    for p in presets:
        base = Cell(p, PRESETS[p].chunks_per_shard, 1.0, "zstd3", False)
        cells.append(base)
        cells += [replace(base, **{a: v}) for a, vs in axes.items() for v in vs]
    return list(dict.fromkeys(cells))  # drops cps == preset default


def raw_target(preset: UseCase, scale: float) -> int:
    p = PRESETS[preset]
    raw = p.chunk_bytes if p.basis is SizeBasis.RAW else p.chunk_bytes / RATIO
    return int(raw * scale)


def synthetic(nt: int = 96, ny: int = 181, nx: int = 360) -> xr.Dataset:
    rng = np.random.default_rng(0)
    t, y, x = np.ogrid[:nt, :ny, :nx]
    field = 280 + 10 * np.sin(y / 20 + t / 10) * np.cos(x / 30)
    noise = rng.normal(0, 0.5, (nt, ny, nx))
    dims = ("time", "latitude", "longitude")
    return xr.Dataset(
        {
            "temperature_2m": (dims, (field + noise).astype("f4"), {"units": "K"}),
            "wind_u_10m": (dims, (field - 280 + noise).astype("f4"), {"units": "m/s"}),
        },
        coords={
            "time": np.arange(nt),
            "latitude": np.linspace(90, -90, ny),
            "longitude": np.linspace(0, 359, nx),
        },
    )


def gfs(names: list[str], steps: int) -> xr.Dataset:
    """Last ``steps`` hours of dynamical's GFS analysis, loaded. Run in-region only."""
    storage = icechunk.s3_storage(
        bucket="dynamical-noaa-gfs",
        prefix="noaa-gfs-analysis/v0.1.0.icechunk",
        region=REGION,
        anonymous=True,
    )
    session = icechunk.Repository.open(storage).readonly_session(branch="main")
    ds = xr.open_zarr(session.store, consolidated=False, chunks=None)[names]
    return ds.isel(time=slice(-steps, None)).drop_encoding().load()


def ic_storage(url: str) -> icechunk.Storage:
    u = urlparse(url)
    if u.scheme == "file":
        return icechunk.local_filesystem_storage(u.path)
    return icechunk.s3_storage(
        bucket=u.netloc, prefix=u.path.lstrip("/"), region=REGION, from_env=True
    )


def obstore_at(url: str) -> obs.store.ObjectStore:
    return (
        from_url(url, region=REGION)
        if url.startswith("s3")
        else from_url(url, mkdir=True)
    )


def write(ds: xr.Dataset, enc: dict, kind: str, url: str) -> zarr.abc.store.Store:
    """Write ``ds`` and return a store to read it back from."""
    if kind == "zarr":
        store = zarr.storage.ObjectStore(obstore_at(url))
        ds.to_zarr(store, encoding=enc, mode="w", zarr_format=3, consolidated=False)
        return store
    repo = icechunk.Repository.create(ic_storage(url))
    session = repo.writable_session("main")
    ds.to_zarr(session.store, encoding=enc, mode="w", zarr_format=3, consolidated=False)
    session.commit("bench cell")
    return repo.readonly_session(branch="main").store


def stored(url: str, prefix: str) -> tuple[int, int]:
    """(bytes, objects) of chunk data under ``url/prefix``."""
    store = obstore_at(url)
    metas = [m for batch in obs.list(store, prefix=prefix) for m in batch]
    return sum(m["size"] for m in metas), len(metas)


def delete(url: str) -> None:
    store = obstore_at(url)
    paths = [m["path"] for batch in obs.list(store) for m in batch]
    if paths:
        obs.delete(store, paths)


def latencies(
    arr: zarr.Array, reps: int, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    nt, ny, nx = arr.shape
    picks = {
        "slice": lambda: (rng.integers(nt), slice(None), slice(None)),
        "series": lambda: (slice(None), rng.integers(ny), rng.integers(nx)),
        "tile": lambda: (
            rng.integers(nt),
            slice(y := rng.integers(max(ny - TILE, 0) + 1), y + TILE),
            slice(x := rng.integers(max(nx - TILE, 0) + 1), x + TILE),
        ),
    }
    out = {}
    for name, pick in picks.items():
        times = []
        for _ in range(reps):
            idx = pick()
            t0 = time.perf_counter()
            arr[idx]
            times.append(time.perf_counter() - t0)
        p50, p95 = np.percentile(times, [50, 95])
        out[name] = {"p50_s": float(p50), "p95_s": float(p95)}
    return out


def run_cell(
    ds: xr.Dataset, var: str, cell: Cell, kind: str, url: str, reps: int
) -> dict:
    one = ds[[var]].drop_vars(list(ds.coords))  # coords would add icechunk objects
    assert one[var].dims[0] == "time" and one[var].ndim == 3, one[var].dims
    kb = convention_keepbits(one) if cell.keepbits else {}
    one = apply_keepbits(one, keepbits=kb)
    enc = spacetimer.recommend(
        one,
        cell.preset,
        chunks_per_shard=cell.cps,
        raw_chunk_bytes=raw_target(cell.preset, cell.scale),
        compressors=COMPRESSORS[cell.compressor],
    )
    # tracemalloc sees numpy + Python allocs only, not Rust (icechunk/obstore) buffers
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()
    store = write(one, enc, kind, url)
    write_s = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] - base

    nbytes, objects = stored(url, f"{var}/c" if kind == "zarr" else "chunks")
    arr = zarr.open_array(store, path=var, mode="r")
    raw = one[var].nbytes
    record = {
        "cell": cell.id,
        **asdict(cell),
        "var": var,
        "store": kind,
        "shape": list(arr.shape),
        "chunks": list(enc[var]["chunks"]),
        "shards": list(enc[var].get("shards") or []),
        "keepbits_used": kb.get(var),
        "write_s": write_s,
        "peak_mib": peak / 2**20,
        "raw_bytes": raw,
        "stored_bytes": nbytes,
        "objects": objects,
        "ratio": nbytes / raw,
        "read": latencies(arr, reps, np.random.default_rng(0)),
        "predicted": asdict(
            spacetimer.report(one, enc, compression_ratio=RATIO).costs[var]
        ),
    }
    delete(url)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic data in a temp dir")
    ap.add_argument("--out", help="s3://bucket/prefix for written cells")
    ap.add_argument("--vars", default="temperature_2m,precipitation_surface,wind_u_10m")
    ap.add_argument("--steps", type=int, default=720, help="hours of GFS to load")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--presets", default="time,space,balanced,web")
    args = ap.parse_args()

    if args.smoke:
        ds, out, reps = synthetic(), Path(tempfile.mkdtemp()).as_uri(), 3
    elif args.out:
        ds, out, reps = (
            gfs(args.vars.split(","), args.steps),
            args.out.rstrip("/"),
            args.reps,
        )
    else:
        ap.error("pass --smoke or --out")

    cells = screening([UseCase(p) for p in args.presets.split(",")])
    total = len(cells) * len(ds.data_vars) * len(STORES)
    print(
        f"{total} runs: {len(cells)} cells x {len(ds.data_vars)} vars x {len(STORES)} stores",
        file=sys.stderr,
    )
    results, lines = obstore_at(out), []
    print(f"results: {out}/results.jsonl", file=sys.stderr)
    tracemalloc.start()
    n = 0
    for cell in cells:
        for var in map(str, ds.data_vars):
            for kind in STORES:
                n += 1
                t0 = time.perf_counter()
                rec = run_cell(
                    ds, var, cell, kind, f"{out}/{cell.id}/{var}.{kind}", reps
                )
                lines.append(json.dumps(rec))
                # rewrites the whole file per record; fine at ~1 KB x a few hundred
                obs.put(results, "results.jsonl", "\n".join(lines).encode() + b"\n")
                print(
                    f"[{n}/{total}] {cell.id} {var} {kind} {time.perf_counter() - t0:.1f}s",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
