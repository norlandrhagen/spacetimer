import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from zarr.storage import MemoryStore

import spacetimer
from spacetimer import UseCase, apply_keepbits, recommend


def _store(kind: str):
    if kind == "zarr":
        return MemoryStore()
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    return repo.writable_session("main").store


@pytest.mark.parametrize("kind", ["zarr", "icechunk"])
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    shape=st.tuples(st.integers(1, 8), st.integers(1, 80), st.integers(1, 80)),
    use_case=st.sampled_from(UseCase),
    cps=st.one_of(st.none(), st.integers(1, 8)),
    raw_chunk_bytes=st.sampled_from([64, 256, 4096]),
)
def test_roundtrip(kind, shape, use_case, cps, raw_chunk_bytes):
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"v": (("time", "y", "x"), rng.random(shape, dtype="f4"))},
        coords={"time": np.arange(shape[0])},
    )
    kw = {
        "use_case": use_case,
        "chunks_per_shard": cps,
        "sharding": cps is not None,
        "raw_chunk_bytes": raw_chunk_bytes,
    }
    encoding = recommend(data=ds, **kw)
    store = _store(kind)
    ds.to_zarr(store, encoding=encoding, zarr_format=3, consolidated=False)

    arr = zarr.open_array(store, path="v")
    enc = encoding["v"]
    assert arr.chunks == enc["chunks"]
    assert arr.shards == enc.get("shards")
    xr.testing.assert_equal(xr.open_zarr(store, consolidated=False).load(), ds)

    # metadata-only path: the zarr group gives the same answer as the dataset
    group = zarr.open_group(store, mode="r")
    assert recommend(data=group, **kw) == encoding


def test_zarr_v2_group_reads_array_dimensions():
    """v2 arrays carry dims in the _ARRAY_DIMENSIONS attr, not metadata."""
    ds = xr.Dataset(
        {"v": (("time", "y", "x"), np.zeros((4, 50, 60), "f4"))},
        coords={"time": np.arange(4)},
    )
    store = MemoryStore()
    ds.to_zarr(store, zarr_format=2, consolidated=False)
    group = zarr.open_group(store, mode="r", zarr_format=2)
    assert recommend(group, "space") == recommend(ds, "space")


def test_group_skips_2d_coords_and_warns_on_unnamed():
    ds = xr.Dataset(
        {"t": (("time", "y", "x"), np.zeros((2, 4, 5), "f4"))},
        coords={
            "lat": (("y", "x"), np.zeros((4, 5))),
            "lon": (("y", "x"), np.zeros((4, 5))),
        },
    )
    store = MemoryStore()
    ds.to_zarr(store, zarr_format=3, consolidated=False)
    group = zarr.open_group(store, mode="a")
    assert recommend(group, "space") == recommend(ds, "space")
    group.create_array("raw", shape=(4, 5), dtype="f4")  # no dimension_names
    with pytest.warns(UserWarning, match="no dimension names on \\['raw'\\]"):
        recommend(group, "space")


def test_dask_keepbits_roundtrip():
    """dask chunks must be shard-aligned for to_zarr; rounding stays lazy."""
    data = np.random.default_rng(0).random((40, 64, 64), dtype="f4")
    ds = xr.Dataset({"v": (("time", "y", "x"), data)}).chunk({"time": 3})
    encoding = recommend(
        data=ds, use_case=UseCase.TIME, raw_chunk_bytes=4 * 10 * 16 * 16
    )
    shards = encoding["v"]["shards"]
    ds = ds.chunk(dict(zip(ds["v"].dims, shards)))
    rounded = apply_keepbits(ds, keepbits={"v": 7})
    assert rounded["v"].chunks is not None  # still lazy
    store = MemoryStore()
    rounded.to_zarr(store, encoding=encoding, zarr_format=3, consolidated=False)
    np.testing.assert_array_equal(
        xr.open_zarr(store, consolidated=False)["v"], rounded["v"].compute()
    )


def test_cf_coords_find_spatial_and_time_dims():
    """WRF-style names: dims are found from coordinate metadata, not names."""
    import pandas as pd

    ny, nx, nt = 20, 30, 50
    ds = xr.Dataset(
        {"T2": (("Time", "south_north", "west_east"), np.zeros((nt, ny, nx), "f4"))},
        coords={
            "XLAT": (
                ("south_north", "west_east"),
                np.zeros((ny, nx)),
                {"units": "degree_north"},
            ),
            "XLONG": (
                ("south_north", "west_east"),
                np.zeros((ny, nx)),
                {"units": "degree_east"},
            ),
            "XTIME": ("Time", pd.date_range("2000-01-01", periods=nt, freq="h")),
            "lev": ("lev", np.arange(3), {"bounds": "lev_bnds"}),
            "lev_bnds": (("lev", "nv"), np.zeros((3, 2)), {"units": "degrees_north"}),
        },
    )
    enc = recommend(ds, "time")  # no warnings
    # without CF detection: no spatial dim, so no entry; without time: 1 step
    assert enc["T2"]["chunks"] == (nt, ny, nx)
    assert spacetimer.report(ds, enc).costs["T2"].series_chunks == 1

    store = MemoryStore()
    ds.to_zarr(store, zarr_format=3, consolidated=False)
    group = zarr.open_group(store, mode="r")
    assert recommend(group, "time") == enc

    # explicit spatial_dims are used as given; bounds dims never become spatial
    from spacetimer.core import _roles

    spatial, _ = _roles(ds, spacetimer.core.SPATIAL_DIMS)
    assert {"south_north", "west_east"} <= set(spatial) and "nv" not in spatial
    with pytest.warns(UserWarning, match="no spatial dim"):
        assert recommend(ds, "time", spatial_dims=("row",)) == {}
