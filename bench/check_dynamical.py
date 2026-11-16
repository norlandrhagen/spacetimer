"""Phase 2a: open dynamical's GFS analysis store (metadata only) and score its layout.

uv run python bench/check_dynamical.py
"""

import icechunk
import zarr

import spacetimer
from spacetimer import Encoding

BUCKET = "dynamical-noaa-gfs"
PREFIX = "noaa-gfs-analysis/v0.1.0.icechunk"


def open_group(bucket: str = BUCKET, prefix: str = PREFIX) -> zarr.Group:
    storage = icechunk.s3_storage(
        bucket=bucket, prefix=prefix, region="us-west-2", anonymous=True
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch="main")
    return zarr.open_group(session.store, mode="r")


def their_encoding(group: zarr.Group) -> dict[str, Encoding]:
    """The layout as stored, in spacetimer's Encoding shape."""
    out: dict[str, Encoding] = {}
    for name, arr in group.arrays():
        if arr.ndim < 2:
            continue
        enc = Encoding(chunks=arr.chunks, compressors=tuple(arr.compressors))
        if arr.shards is not None:
            enc["shards"] = arr.shards
        out[name] = enc
    return out


def main() -> None:
    group = open_group()
    theirs = their_encoding(group)
    name = next(iter(theirs))
    arr = group[name]
    print(
        f"{len(theirs)} vars; e.g. {name}: {arr.metadata.dimension_names} {arr.shape} {arr.dtype}"
    )
    print(f"compressors: {arr.compressors}\n")
    ours = spacetimer.recommend(group, "time")
    for label, enc in (("dynamical (as stored)", theirs), ("spacetimer 'time'", ours)):
        print(f"{label}:")
        print(spacetimer.report(group, {name: enc[name]}), "\n")


if __name__ == "__main__":
    main()
