import inspect
import math

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from spacetimer import (
    PRESETS,
    SizeBasis,
    UseCase,
    VarSpec,
    convention_keepbits,
    recommend,
    report,
)
from spacetimer.core import MAX_INNER_CHUNKS, SPATIAL_DIMS, SpacetimerAccessor

DIM_POOL = ("time", "band", "lat", "y", "x")
ERA5 = VarSpec(
    "t2m", ("time", "latitude", "longitude"), (73000, 721, 1440), np.dtype("f4")
)


@st.composite
def specs(draw):
    dims = draw(
        st.lists(st.sampled_from(DIM_POOL), min_size=1, max_size=5, unique=True)
        .map(lambda ds: tuple(d for d in DIM_POOL if d in ds))
        .filter(lambda ds: SPATIAL_DIMS & set(ds))
    )
    shape = tuple(draw(st.integers(1, 10**6)) for _ in dims)
    dtype = np.dtype(draw(st.sampled_from(["f2", "f4", "f8", "i1", "u1", "i2"])))
    return VarSpec("v", dims, shape, dtype)


cps_st = st.one_of(st.none(), st.integers(1, 32))


@pytest.mark.filterwarnings("ignore:'v'. no time dim", "ignore:'v'. chunks_per_shard")
@given(
    specs(),
    st.sampled_from(UseCase),
    cps_st,
    st.one_of(st.none(), st.integers(4096, 2**28)),
    st.floats(0.05, 1),
)
def test_layout_invariants(spec, use_case, cps, raw, ratio):
    kw = {
        "use_case": use_case,
        "chunks_per_shard": cps,
        "sharding": cps is not None,
        "raw_chunk_bytes": raw,
        "compression_ratio": ratio,
    }
    enc = recommend([spec], **kw)["v"]
    chunks, shards = enc["chunks"], enc.get("shards")
    assert all(1 <= c <= s for c, s in zip(chunks, spec.shape))
    assert (shards is None) == (cps is None)
    if shards is not None:
        # a shard is whole chunks and overhangs its dim by less than one chunk
        assert all(
            sh % c == 0 and sh - c < s for sh, c, s in zip(shards, chunks, spec.shape)
        )
        assert math.prod(sh // c for sh, c in zip(shards, chunks)) <= MAX_INNER_CHUNKS

    p = PRESETS[use_case]
    target = raw or (
        p.chunk_bytes if p.basis is SizeBasis.RAW else int(p.chunk_bytes / ratio)
    )
    chunk_bytes = math.prod(chunks) * spec.dtype.itemsize
    assert chunk_bytes <= target
    spatial = [i for i, d in enumerate(spec.dims) if d in SPATIAL_DIMS]
    growable = list(spatial)
    if p.spatial_edge and "time" in spec.dims:
        growable.append(spec.dims.index("time"))
    saturated = all(chunks[i] == spec.shape[i] for i in growable)
    # even splitting can halve each spatial dim, and time rounds down once more
    assert chunk_bytes * 2 ** (len(spatial) + 2) >= target or saturated

    c = report([spec], {"v": enc}, compression_ratio=ratio).costs["v"]
    assert c.objects >= 1 and c.slice_chunks >= 1 and c.series_chunks >= 1
    assert c.shard_mib is None or c.shard_mib >= c.chunk_mib
    assert c.slice_mib == c.slice_chunks * c.chunk_mib

    assert recommend([spec], **kw)["v"] == enc


def test_era5_time_lands_in_dynamical_band():
    enc = recommend(data=[ERA5], use_case=UseCase.TIME)
    assert enc["t2m"]["chunks"] == (4480, 32, 32)
    assert enc["t2m"]["shards"] == (4480, 256, 256)
    r = report([ERA5], enc)
    c = r.costs["t2m"]
    assert 2.5 <= c.chunk_mib <= 6
    assert c.shard_mib is not None and 100 <= c.shard_mib <= 600
    # the price of fast series: one map reads ~3.5 GiB, one series ~60 MiB
    assert c.slice_chunks == 1035 and c.slice_mib == pytest.approx(1035 * 3.5)
    assert c.series_chunks == 17 and c.series_mib == pytest.approx(17 * 3.5)
    assert repr(r).startswith("| var ")
    assert "| t2m " in repr(r)
    df = r.to_pandas()
    assert df.loc["t2m", "chunks"] == (4480, 32, 32)
    assert df.loc["t2m", "series_chunks"] == 17
    assert "<td>4480x32x32</td>" in r._repr_html_()


def test_repr_html_cubes() -> None:
    from xml.etree import ElementTree

    from spacetimer._svg import MAX_LINES, cube_svg

    specs = [
        VarSpec(n, ("time", "lat", "lon"), (8760, 721, 1440), "float32")
        for n in ("a", "b")
    ] + [VarSpec("flat", ("lat", "lon"), (721, 1440), "float32")]
    out = report(specs, recommend(specs, "time"))._repr_html_()
    assert out.count("<svg") == 3  # one collapsible row per variable
    assert "<td>33.9 GiB</td>" in out and "<td>1 x 8 x 8 = 64</td>" in out
    # 100,000 chunks along every axis still draws a bounded number of lines
    svg = ElementTree.fromstring(
        cube_svg(("t", "y", "x"), (100_000,) * 3, (1,) * 3, (10,) * 3)
    )
    lines = list(svg.iter("{http://www.w3.org/2000/svg}line"))
    assert 0 < len(lines) <= 3 * 2 * 2 * MAX_LINES  # axes, faces, layers
    # every shard line is also a chunk line, and chunks still draw finer when they can
    from spacetimer._svg import _cuts

    cases = [(73000, 1, 282, 48), (721, 32, 256, 56), (1440, 32, 256, 62)]
    for n, chunk, shard, length in cases:
        shard_cuts = _cuts(n, shard, length, inner=chunk)
        chunk_cuts = _cuts(n, chunk, length, outer=shard)
        assert set(shard_cuts) <= set(chunk_cuts)
        assert len(chunk_cuts) > len(shard_cuts)
    for shape in [(5,), (5, 7), (2, 3, 5, 7), (9, 8, 2, 3, 5, 7)]:
        dims = tuple(f"d{i}" for i in range(len(shape)))
        ElementTree.fromstring(cube_svg(dims, shape, (1,) * len(shape)))
    # leading axes are their own figure, linked to the cube by a zoom callout
    linked = cube_svg(("d0", "z", "y", "x"), (10, 3, 5, 7), (5, 1, 1, 1))
    assert "stroke-dasharray" in linked and ">d0<" in linked


@pytest.mark.parametrize(
    ("shape", "dims", "chunks", "shards"),
    [
        ((32768, 32768), ("y", "x"), (361, 361), (1444, 1444)),
        ((12, 4000, 3000), ("time", "y", "x"), (1, 334, 334), (1, 1336, 1336)),
    ],
)
def test_web_matches_topozarr(shape, dims, chunks, shards):
    spec = VarSpec("v", dims, shape, np.dtype("f4"))
    enc = recommend(data=[spec], use_case=UseCase.WEB)["v"]
    assert enc["chunks"] == chunks
    assert enc["shards"] == shards


def test_space_packs_time_into_shards():
    enc = recommend(data=[ERA5], use_case="space")["t2m"]  # plain str works
    assert enc["chunks"] == (1, 721, 1440)
    assert enc["shards"] == (282, 721, 1440)


def test_balanced_sits_between_time_and_space():
    enc = recommend([ERA5], UseCase.BALANCED)
    assert enc["t2m"]["chunks"] == (315, 121, 120)
    c = {u: report([ERA5], recommend([ERA5], u)).costs["t2m"] for u in UseCase}
    assert c["space"].slice_mib < c["balanced"].slice_mib < c["time"].slice_mib
    assert c["time"].series_mib < c["balanced"].series_mib < c["space"].series_mib


def test_shards_overhang_instead_of_leaving_a_sliver():
    enc = recommend([ERA5], UseCase.WEB)["t2m"]
    assert enc["chunks"] == (1, 361, 360)
    assert enc["shards"][1:] == (722, 1440)  # one shard over 721, not 361 + 360


def test_silent_fallbacks_warn():
    wrf = VarSpec(
        "T2", ("Time", "south_north", "west_east"), (10, 50, 50), np.dtype("f4")
    )
    with pytest.warns(UserWarning, match="no spatial dim in \\['T2'\\]"):
        assert recommend(data=[wrf], use_case=UseCase.TIME) == {}
    with pytest.warns(UserWarning, match="chunks_per_shard=32 reduced to 22"):
        recommend([ERA5], UseCase.TIME, chunks_per_shard=32)
    dated = VarSpec("v", ("date", "lat", "lon"), (100, 50, 50), np.dtype("f4"))
    with pytest.warns(UserWarning, match="no time dim"):
        recommend(data=[dated], use_case=UseCase.TIME)
    enc = recommend(data=[dated], use_case=UseCase.TIME, time_dim="date")
    assert enc["v"]["chunks"][0] > 1


def test_keepbits_convention():
    specs = [
        VarSpec(
            "u10",
            ("y", "x"),
            (10, 10),
            np.dtype("f4"),
            {"standard_name": "eastward_wind"},
        ),
        VarSpec(
            "tp",
            ("y", "x"),
            (10, 10),
            np.dtype("f4"),
            {"long_name": "precipitation rate"},
        ),
        VarSpec("sp", ("y", "x"), (10, 10), np.dtype("f4"), {"units": "Pa"}),
        VarSpec("t2m", ("y", "x"), (10, 10), np.dtype("f4"), {"units": "K"}),
        VarSpec("wind_direction", ("y", "x"), (10, 10), np.dtype("f4")),
        VarSpec("mask", ("y", "x"), (10, 10), np.dtype("u1")),
        VarSpec("sp16", ("y", "x"), (10, 10), np.dtype("f2"), {"units": "Pa"}),
    ]
    # unmatched floats are left out: 7 bits would move 288.15 K to 288.0
    with pytest.warns(UserWarning, match="\\['t2m', 'wind_direction'\\]"):
        got = convention_keepbits(specs)
    assert got == {
        "u10": 6,
        "tp": 8,
        "sp": 11,
        "sp16": 10,  # clamped to float16's mantissa
    }


def test_bad_inputs():
    spec = VarSpec("v", ("y", "x"), (10, 10), np.dtype("f4"))
    assert recommend([spec], "web") == recommend(data=[spec], use_case="web")
    with pytest.raises(ValueError):
        recommend(data=[spec], use_case="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        recommend(data=[spec], use_case=UseCase.WEB, chunks_per_shard=0)
    with pytest.raises(ValueError):
        recommend(data=[spec], use_case=UseCase.WEB, chunks_per_shard=4, sharding=False)
    with pytest.raises(ValueError):
        recommend(data=[spec], use_case=UseCase.WEB, compression_ratio=0)
    assert VarSpec("v", ("y", "x"), (10, 10), "f4") == spec  # dtype is coerced
    with pytest.raises(ValueError):
        VarSpec("v", ("y", "x"), (10,), "f4")
    with pytest.raises(ValueError):
        VarSpec("v", ("time", "y", "x"), (0, 10, 10), "f4")
    da = xr.DataArray(np.zeros((5, 20, 20), "f4"), dims=("time", "y", "x"))
    with pytest.raises(TypeError):
        recommend(da, "time")  # type: ignore[arg-type]


def test_accessor_matches_functions():
    ds = xr.Dataset({"v": (("y", "x"), np.zeros((100, 100), "f4"))})
    enc = ds.spacetimer.recommend("web", sharding=False)
    assert enc == recommend(ds, "web", sharding=False)
    assert ds.spacetimer.report(enc).costs == report(ds, enc).costs
    wind = xr.Dataset({"wind_speed": ("x", np.float32([1.2345678]))})
    assert wind.spacetimer.bitround()["wind_speed"].item() == np.float32(1.234375)
    assert wind.spacetimer.bitround(keepbits={})["wind_speed"].item() != 1.234375


def test_accessor_signatures_match():
    for fn in (recommend, report):
        want = list(inspect.signature(fn).parameters.values())[1:]
        got = list(
            inspect.signature(
                getattr(SpacetimerAccessor, fn.__name__)
            ).parameters.values()
        )[1:]
        assert got == want, fn.__name__
        assert (
            inspect.signature(
                getattr(SpacetimerAccessor, fn.__name__)
            ).return_annotation
            == inspect.signature(fn).return_annotation
        )


def test_compressors_override(tmp_path):
    from zarr.codecs import ZstdCodec

    ds = xr.Dataset({"a": (("time", "y", "x"), np.zeros((4, 8, 8), "f4"))})
    enc = recommend(ds, "space", compressors=[ZstdCodec(level=5)])
    assert enc["a"]["compressors"] == (ZstdCodec(level=5),)
    assert enc["a"]["chunks"] == recommend(ds, "space")["a"]["chunks"]
    ds.to_zarr(tmp_path / "a.zarr", encoding=enc, zarr_format=3, consolidated=False)


def test_report_hand_written_encoding():
    import numcodecs

    from spacetimer.core import _codec_row, _fmt_bytes

    spec = VarSpec("v", ("time", "y", "x"), (10, 50, 50), "f4")
    r = report([spec], {"v": {"chunks": (1, 50, 50)}})  # no compressors, no shards
    assert "<td>1x50x50</td>" in r._repr_html_()
    with pytest.warns(UserWarning, match="\\['typo'\\], not found"):
        assert report([spec], {"typo": {"chunks": (1, 1, 1)}}).costs == {}
    assert _codec_row(numcodecs.Zstd(level=3)) == ("zstd", "level=3, checksum=False")
    assert _fmt_bytes(3 * 1024**4) == "3.0 TiB"
