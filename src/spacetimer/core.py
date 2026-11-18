from __future__ import annotations

import math
import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import numpy as np
import xarray as xr
import zarr
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec

if TYPE_CHECKING:
    import pandas as pd

MiB = 1024 * 1024
# caps shard size in chunks; the zarr v3 shard index is 16 B per chunk, so 8 KiB.
# dynamical's worked example (1440x32x32 in 2880x384x384) holds 288
MAX_INNER_CHUNKS = 512
MIN_EDGE = 32  # the time preset's tile, so balanced never reads more per map
SPATIAL_DIMS = frozenset({"x", "y", "lat", "lon", "latitude", "longitude"})
# CF coordinate metadata that marks a coordinate's dims as spatial
CF_SPATIAL_NAMES = frozenset(
    {
        "latitude",
        "longitude",
        "grid_latitude",
        "grid_longitude",
        "projection_x_coordinate",
        "projection_y_coordinate",
    }
)
CF_DEGREES = frozenset(
    f"{deg}{sep}{d}"
    for deg in ("degree", "degrees")
    for sep in ("_", "")
    for d in ("north", "east", "N", "E")
)
# warnings skip our frames so they point at the caller, accessor included
_PKG = (str(Path(__file__).parent),)


class UseCase(StrEnum):
    """The access pattern readers will use most. Plain strings work too."""

    TIME = "time"
    """Time series at a point."""
    SPACE = "space"
    """Full maps at one time step."""
    BALANCED = "balanced"
    """A mix of maps and time series."""
    WEB = "web"
    """Map tiles in a browser."""


class SizeBasis(StrEnum):
    """What a preset's ``chunk_bytes`` measures."""

    RAW = "raw"
    """Uncompressed bytes (web tiles are sized by extent)."""
    COMPRESSED = "compressed"
    """Compressed bytes, converted to raw with ``compression_ratio``."""


@dataclass(frozen=True)
class Preset:
    """Chunk and shard targets for one use case; see ``PRESETS``."""

    chunk_bytes: int
    """Target chunk size, measured as ``basis`` says."""
    basis: SizeBasis
    spatial_edge: int | Literal["auto"] | None
    """Spatial tile edge; time takes the rest of the budget. ``"auto"``: the edge
    where one map and one time series read the same number of chunks. ``None``:
    one time step per chunk."""
    chunks_per_shard: int
    """Default chunks per shard along each spatial dim."""


PRESETS: dict[UseCase, Preset] = {
    # dynamical.org reformatters, analysis mode: ~3.5 MiB compressed, 32x32 tiles
    UseCase.TIME: Preset(int(3.5 * MiB), SizeBasis.COMPRESSED, 32, 8),
    # same byte target, spent on one time step of the largest tile that fits
    UseCase.SPACE: Preset(int(3.5 * MiB), SizeBasis.COMPRESSED, None, 8),
    # same byte target; tile edge picked per variable, at least the time preset's 32
    UseCase.BALANCED: Preset(int(3.5 * MiB), SizeBasis.COMPRESSED, "auto", 8),
    # topozarr: 512 KiB raw tiles, 4x4 chunks per shard
    UseCase.WEB: Preset(512 * 1024, SizeBasis.RAW, None, 4),
}


@dataclass(frozen=True)
class VarSpec:
    """One variable's metadata, to plan a layout before the data exists."""

    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    """Final shape; chunks never exceed it."""
    dtype: np.dtype
    """Anything ``np.dtype`` accepts, e.g. ``"f4"``."""
    attrs: Mapping[str, Any] = field(default_factory=dict)
    """CF attributes; ``convention_keepbits`` reads ``standard_name``,
    ``long_name``, and ``units``."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if len(self.dims) != len(self.shape) or any(n < 1 for n in self.shape):
            raise ValueError(
                f"{self.name!r}: dims {self.dims} and shape {self.shape} must "
                "have the same length, and every size must be >= 1"
            )


Data = xr.Dataset | zarr.Group | Iterable[VarSpec]


class Encoding(TypedDict):
    """One variable's entry in the ``to_zarr`` encoding ``recommend`` returns."""

    chunks: tuple[int, ...]
    shards: NotRequired[tuple[int, ...]]
    """Absent with ``sharding=False``."""
    compressors: tuple[BytesBytesCodec, ...]


@dataclass(frozen=True)
class Costs:
    """Estimates; MiB are compressed at the given compression_ratio."""

    chunk_mib: float
    shard_mib: float | None
    """``None`` when unsharded."""
    objects: int
    """Stored objects: shards, or chunks when unsharded."""
    slice_chunks: int
    """Chunks read for one full spatial slice (one map)."""
    slice_mib: float
    series_chunks: int
    """Chunks read for one point's full time series."""
    series_mib: float


# dynamical.org's production codec
DEFAULT_COMPRESSORS: tuple[BytesBytesCodec, ...] = (
    BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),
)


def recommend(
    data: Data,
    use_case: UseCase,
    *,
    chunks_per_shard: int | None = None,
    sharding: bool = True,
    compression_ratio: float = 0.2,
    raw_chunk_bytes: int | None = None,
    spatial_dims: Collection[str] = SPATIAL_DIMS,
    time_dim: str | None = None,
    compressors: Sequence[BytesBytesCodec] = DEFAULT_COMPRESSORS,
) -> dict[str, Encoding]:
    """Return a ``to_zarr`` encoding for every variable with a spatial dim.

    Reads metadata only. ``chunks_per_shard`` is per spatial dim; ``None`` uses
    the preset's value. ``raw_chunk_bytes`` overrides the preset's target with an
    uncompressed size. Chunks never exceed the current shape, so for a dataset
    you will append to, pass ``VarSpec``s with the final shape. ``compressors``
    replaces the default codec and does not change chunk sizes; pass a matching
    ``compression_ratio`` if yours compresses differently.
    """
    use_case = UseCase(use_case)
    preset = PRESETS[use_case]
    _check_ratio(compression_ratio)
    if not sharding and chunks_per_shard is not None:
        raise ValueError("chunks_per_shard has no effect with sharding=False")
    if chunks_per_shard is not None and chunks_per_shard < 1:
        raise ValueError(f"chunks_per_shard must be >= 1, got {chunks_per_shard}")
    cps = (chunks_per_shard or preset.chunks_per_shard) if sharding else None
    if raw_chunk_bytes is None:
        raw_chunk_bytes = (
            preset.chunk_bytes
            if preset.basis is SizeBasis.RAW
            else int(preset.chunk_bytes / compression_ratio)
        )

    spatial_dims, cf_time = _roles(data, spatial_dims)
    encoding: dict[str, Encoding] = {}
    skipped: list[str] = []
    for spec in _specs(data):
        spatial = [i for i, d in enumerate(spec.dims) if d in spatial_dims]
        if not spatial:
            skipped.append(spec.name)
            continue
        t = _time_axis(spec, spatial, time_dim, cf_time)
        if (
            preset.spatial_edge is not None
            and t is None
            and len(spatial) < len(spec.dims)
        ):
            warnings.warn(
                f"{spec.name!r}: no time dim found in {spec.dims}, so the "
                f"{use_case.value!r} layout falls back to spatial tiles; pass time_dim=",
                skip_file_prefixes=_PKG,
            )
        if (
            chunks_per_shard
            and (k := _per_dim(chunks_per_shard, len(spatial))) < chunks_per_shard
        ):
            warnings.warn(
                f"{spec.name!r}: chunks_per_shard={chunks_per_shard} reduced to {k} "
                f"to keep a shard at or under {MAX_INNER_CHUNKS} chunks",
                skip_file_prefixes=_PKG,
            )
        chunks, shards = _layout(spec, preset, raw_chunk_bytes, cps, spatial, t)
        enc = Encoding(
            chunks=chunks,
            compressors=tuple(compressors),
        )
        if shards is not None:
            enc["shards"] = shards
        encoding[spec.name] = enc
    if skipped:
        warnings.warn(
            f"no spatial dim in {skipped}, left to Xarray defaults; "
            "pass spatial_dims= if their dims have other names",
            skip_file_prefixes=_PKG,
        )
    return encoding


REPORT_HEAD = (
    "var",
    "chunks",
    "shards",
    "objects",
    "slice chunks",
    "series chunks",
)


@dataclass(frozen=True)
class Report:
    """Per-variable ``Costs``.

    A notebook shows one expandable view per variable (chunk-grid cube, sizes,
    read costs, encoding); a terminal shows a Markdown table. The numbers are in
    ``costs`` and ``to_pandas()``.
    """

    costs: dict[str, Costs]
    encoding: Mapping[str, Mapping[str, Any]]
    compression_ratio: float
    specs: Mapping[str, VarSpec]

    def _rows(self) -> list[tuple[str, ...]]:
        rows = []
        for name, c in self.costs.items():
            shards = self.encoding[name].get("shards")
            rows.append(
                (
                    name,
                    "x".join(map(str, self.encoding[name]["chunks"])),
                    "x".join(map(str, shards)) if shards else "-",
                    f"{c.objects:,}",
                    f"{c.slice_chunks:,}",
                    f"{c.series_chunks:,}",
                )
            )
        return rows

    def __repr__(self) -> str:
        """A Markdown table, readable in a terminal and pasteable into GitHub."""
        rows = [REPORT_HEAD, *self._rows()]
        widths = [max(len(r[i]) for r in rows) for i in range(len(REPORT_HEAD))]
        rows.insert(1, tuple("-" * w for w in widths))
        lines = [
            "| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |"
            for r in rows
        ]
        return "\n".join(lines)

    def to_pandas(self) -> pd.DataFrame:
        """One row per variable, unformatted numbers."""
        import pandas as pd

        rows = {
            name: {
                "chunks": self.encoding[name]["chunks"],
                "shards": self.encoding[name].get("shards"),
                **asdict(c),
            }
            for name, c in self.costs.items()
        }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("var")

    def _repr_html_(self) -> str:
        """One collapsible row per variable, like the Xarray repr."""
        from spacetimer._html import REPORT_CSS, _var_html  # lazy: _html imports core

        rows = "".join(
            _var_html(
                self.specs[name],
                self.encoding[name],
                c,
                ratio=self.compression_ratio,
                opened=i == 0,
            )
            for i, (name, c) in enumerate(self.costs.items())
        )
        return f'<div class="st-report">{REPORT_CSS}{rows}</div>'


def report(
    data: Data,
    encoding: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    compression_ratio: float = 0.2,
    spatial_dims: Collection[str] = SPATIAL_DIMS,
    time_dim: str | None = None,
) -> Report:
    """Estimate read and storage costs of ``encoding`` (any, not only ours).

    Each entry needs ``chunks``; ``shards`` and ``compressors`` are optional, so
    a hand-written dict or an array's ``.encoding`` works. ``encoding=None``
    reports the layout already stored in a ``zarr.Group``, or in a Dataset
    opened from Zarr.
    """
    _check_ratio(compression_ratio)
    spatial_dims, cf_time = _roles(data, spatial_dims)
    all_specs = _specs(data)
    if encoding is None:
        encoding = _stored_encoding(data, [s.name for s in all_specs])
    out: dict[str, Costs] = {}
    specs: dict[str, VarSpec] = {}
    for spec in all_specs:
        if spec.name not in encoding:
            continue
        enc = encoding[spec.name]
        for key in ("chunks", "shards"):
            if (grid := enc.get(key)) is not None and len(grid) != len(spec.shape):
                raise ValueError(
                    f"{spec.name!r}: {key} {tuple(grid)} has {len(grid)} dims, "
                    f"data has {len(spec.shape)} {spec.dims}"
                )
        spatial = [i for i, d in enumerate(spec.dims) if d in spatial_dims]
        t = _time_axis(spec, spatial, time_dim, cf_time)
        out[spec.name] = _costs(
            spec, enc["chunks"], enc.get("shards"), spatial, t, compression_ratio
        )
        specs[spec.name] = spec
    if missing := sorted(set(encoding) - set(out)):
        warnings.warn(
            f"encoding names {missing}, not found in data", skip_file_prefixes=_PKG
        )
    return Report(out, encoding, compression_ratio, specs)


def _stored_encoding(data: Data, names: list[str]) -> dict[str, dict[str, Any]]:
    """Each variable's stored layout: Zarr array metadata, or Xarray's ``.encoding``."""
    if isinstance(data, zarr.Group):
        out: dict[str, dict[str, Any]] = {}
        for name in names:
            arr = data[name]
            assert isinstance(arr, zarr.Array)
            out[name] = {"chunks": arr.chunks, "compressors": tuple(arr.compressors)}
            if arr.shards is not None:
                out[name]["shards"] = arr.shards
        return out
    if isinstance(data, xr.Dataset):
        keys = ("chunks", "shards", "compressors")
        out = {
            n: {k: v for k in keys if (v := data[n].encoding.get(k)) is not None}
            for n in names
        }
        if missing := [n for n, e in out.items() if "chunks" not in e]:
            raise ValueError(
                f"no stored chunks for {missing}; open the data from Zarr or pass "
                "an encoding"
            )
        return out
    raise TypeError("encoding=None needs a zarr.Group or a Dataset opened from Zarr")


def _check_ratio(ratio: float) -> None:
    if not 0 < ratio <= 1:
        raise ValueError(f"compression_ratio must be in (0, 1], got {ratio}")


def _specs(obj: Data) -> list[VarSpec]:
    if isinstance(obj, xr.Dataset):
        return [
            VarSpec(str(n), tuple(map(str, v.dims)), v.shape, v.dtype, dict(v.attrs))
            for n, v in obj.data_vars.items()
        ]
    if isinstance(obj, zarr.Group):
        arrays = dict(obj.arrays())
        coords = _coord_names(obj, arrays)
        specs, unnamed = [], []
        for name, arr in arrays.items():
            if (dims := _dims(arr)) is None:
                unnamed.append(name)
                continue
            if dims and dims != (name,) and name not in coords:
                specs.append(VarSpec(name, dims, arr.shape, arr.dtype, dict(arr.attrs)))
        if unnamed:
            warnings.warn(
                f"no dimension names on {unnamed}, skipped", skip_file_prefixes=_PKG
            )
        return specs
    if isinstance(obj, xr.DataArray):  # iterating one would yield its slices
        raise TypeError("pass a Dataset (da.to_dataset()), zarr.Group, or VarSpecs")
    return list(obj)


def _dims(arr: zarr.Array) -> tuple[str, ...] | None:
    """Dimension names from v3 metadata or the v2 ``_ARRAY_DIMENSIONS`` attr."""
    names = getattr(arr.metadata, "dimension_names", None) or arr.attrs.get(
        "_ARRAY_DIMENSIONS"
    )
    return tuple(map(str, names)) if isinstance(names, (list, tuple)) else None


def _coord_names(group: zarr.Group, arrays: Mapping[str, zarr.Array]) -> set[str]:
    """Non-dimension coords (e.g. 2-D lat/lon), as Xarray records them."""
    return {
        c
        for attrs in (group.attrs, *(a.attrs for a in arrays.values()))
        for c in str(attrs.get("coordinates", "")).split()
    }


def _coords(obj: Data) -> list[tuple[str, tuple[str, ...], np.dtype, Mapping]]:
    """(name, dims, dtype, attrs) of each coordinate; none for ``VarSpec``s."""
    if isinstance(obj, xr.Dataset):
        return [
            (str(n), tuple(map(str, c.dims)), c.dtype, c.attrs)
            for n, c in obj.coords.items()
        ]
    if isinstance(obj, zarr.Group):
        arrays = dict(obj.arrays())
        named = _coord_names(obj, arrays)
        return [
            (name, dims, arr.dtype, arr.attrs)
            for name, arr in arrays.items()
            if (dims := _dims(arr)) is not None and (dims == (name,) or name in named)
        ]
    return []


def _roles(
    data: Data, spatial_dims: Collection[str]
) -> tuple[Collection[str], set[str]]:
    """Add CF-marked spatial dims to the default names, and find CF time dims.

    A coordinate is spatial by ``axis`` X/Y, a lat/lon/projection
    ``standard_name``, or degrees north/east ``units``; time by ``axis`` T,
    ``standard_name`` time, "since" ``units``, or a datetime dtype. Bounds
    coordinates are skipped. Explicit ``spatial_dims`` are used as given.
    """
    coords = _coords(data)
    bounds = {str(a["bounds"]) for *_, a in coords if "bounds" in a}
    spatial: set[str] = set()
    time: set[str] = set()
    for name, dims, dtype, attrs in coords:
        if name in bounds:
            continue
        axis, sn, units = (
            str(attrs.get(k, "")) for k in ("axis", "standard_name", "units")
        )
        if axis in ("X", "Y") or sn in CF_SPATIAL_NAMES or units in CF_DEGREES:
            spatial.update(dims)
        elif len(dims) == 1 and (
            axis == "T"
            or sn == "time"
            or " since " in units
            or np.issubdtype(dtype, np.datetime64)
        ):
            time.update(dims)
    if spatial_dims is SPATIAL_DIMS:  # the default: names plus coordinate metadata
        spatial_dims = SPATIAL_DIMS | spatial
    return spatial_dims, time - set(spatial_dims)


def _time_axis(
    spec: VarSpec, spatial: list[int], time_dim: str | None, cf_time: Collection[str]
) -> int | None:
    if time_dim is not None:
        return spec.dims.index(time_dim) if time_dim in spec.dims else None
    other = [i for i in range(len(spec.dims)) if i not in spatial]
    # CF metadata first, then names; forecast (init_time x lead_time) takes init_time
    by_cf = [i for i in other if spec.dims[i] in cf_time]
    return next(iter(by_cf + [i for i in other if "time" in spec.dims[i]]), None)


def _iroot(x: int, n: int) -> int:
    """Floor of the nth root; the float root alone gives 32768 ** (1 / 3) == 31.99."""
    r = round(x ** (1 / n))
    return r if r**n <= x else r - 1


def _per_dim(chunks_per_shard: int, n_spatial: int) -> int:
    k = chunks_per_shard
    while k > 1 and k**n_spatial > MAX_INNER_CHUNKS:
        k -= 1
    return k


def _balanced(dim: int, ideal: int) -> int:
    """Chunk near ``ideal`` that splits ``dim`` evenly (topozarr calculate_chunk_size)."""
    return dim if dim <= ideal else math.ceil(dim / math.ceil(dim / ideal))


def _fill_spatial(
    shape: tuple[int, ...], spatial: list[int], budget: int
) -> dict[int, int]:
    """Square tiles of ``budget`` elements; dims smaller than the tile edge are
    taken whole and their unused budget goes to the remaining dims."""
    out: dict[int, int] = {}
    todo = sorted(spatial, key=lambda i: shape[i])
    while todo:
        edge = _iroot(budget, len(todo))
        if shape[todo[0]] > edge:
            out.update({i: _balanced(shape[i], edge) for i in todo})
            break
        i = todo.pop(0)
        out[i] = shape[i]
        budget = max(1, budget // shape[i])
    return out


def _balanced_edge(
    shape: tuple[int, ...], spatial: list[int], t: int, budget: int
) -> int:
    """Edge where map reads (prod(S) / e^n) equal series reads (T e^n / budget).

    Their product prod(S) T / budget is fixed by the chunk size, so equal is
    also the smallest worst case.
    """
    n = len(spatial)
    maps = math.prod(shape[i] for i in spatial)
    return max(MIN_EDGE, _iroot(maps * budget // shape[t], 2 * n))


def _layout(
    spec: VarSpec,
    preset: Preset,
    raw_chunk_bytes: int,
    chunks_per_shard: int | None,
    spatial: list[int],
    t: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    ndim = len(spec.shape)
    other = [i for i in range(ndim) if i not in spatial]
    chunks = [1] * ndim
    budget = max(1, raw_chunk_bytes // spec.dtype.itemsize)  # elements per chunk

    edge = preset.spatial_edge
    if edge == "auto" and t is not None:
        edge = _balanced_edge(spec.shape, spatial, t, budget)
    if isinstance(edge, int) and t is not None:
        # fixed-edge tiles first, then time takes what they leave
        fixed = edge ** len(spatial)
        tile = _fill_spatial(spec.shape, spatial, fixed)
        chunks[t] = min(spec.shape[t], max(1, budget // math.prod(tile.values())))
        # keep those tiles; they only grow when time is too short to use the budget
        budget = max(budget // chunks[t], min(budget, fixed))
    for i, c in _fill_spatial(spec.shape, spatial, budget).items():
        chunks[i] = c

    if chunks_per_shard is None:
        return tuple(chunks), None

    k = _per_dim(chunks_per_shard, len(spatial))
    shards = list(chunks)
    for i in spatial:  # ceil: the last shard may overhang the array, as dynamical's do
        shards[i] = chunks[i] * min(k, -(-spec.shape[i] // chunks[i]))

    # spend leftover shard budget on non-spatial dims, innermost first (topozarr)
    if k > 1:
        target = raw_chunk_bytes * k ** len(spatial) // spec.dtype.itemsize
        inner = math.prod(s // c for s, c in zip(shards, chunks))
        headroom = min(target // math.prod(shards), MAX_INNER_CHUNKS // inner)
        for i in reversed(other):
            if headroom <= 1:
                break
            mult = min(headroom, -(-spec.shape[i] // chunks[i]))
            shards[i] = chunks[i] * mult
            headroom //= mult
    return tuple(chunks), tuple(shards)


def _costs(
    spec: VarSpec,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    spatial: list[int],
    t: int | None,
    ratio: float,
) -> Costs:
    def count(axes: Iterable[int], grid: tuple[int, ...]) -> int:
        return math.prod(math.ceil(spec.shape[i] / grid[i]) for i in axes)

    chunk_mib = math.prod(chunks) * spec.dtype.itemsize * ratio / MiB
    slice_chunks = count(spatial, chunks)
    series_chunks = count([t], chunks) if t is not None else 1
    return Costs(
        chunk_mib=chunk_mib,
        shard_mib=chunk_mib * (math.prod(shards) / math.prod(chunks))
        if shards
        else None,
        objects=count(range(len(spec.shape)), shards or chunks),
        slice_chunks=slice_chunks,
        slice_mib=slice_chunks * chunk_mib,
        series_chunks=series_chunks,
        series_mib=series_chunks * chunk_mib,
    )


@xr.register_dataset_accessor("spacetimer")
class SpacetimerAccessor:
    """``ds.spacetimer.recommend("time")``.

    Signatures are copied from the functions so editors can autocomplete them;
    ``test_accessor_signatures_match`` fails if they drift.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def recommend(
        self,
        use_case: UseCase,
        *,
        chunks_per_shard: int | None = None,
        sharding: bool = True,
        compression_ratio: float = 0.2,
        raw_chunk_bytes: int | None = None,
        spatial_dims: Collection[str] = SPATIAL_DIMS,
        time_dim: str | None = None,
        compressors: Sequence[BytesBytesCodec] = DEFAULT_COMPRESSORS,
    ) -> dict[str, Encoding]:
        """See ``spacetimer.recommend``."""
        return recommend(
            self._ds,
            use_case,
            chunks_per_shard=chunks_per_shard,
            sharding=sharding,
            compression_ratio=compression_ratio,
            raw_chunk_bytes=raw_chunk_bytes,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
            compressors=compressors,
        )

    def bitround(self, *, keepbits: Mapping[str, int] | None = None) -> xr.Dataset:
        """``apply_keepbits``; ``keepbits=None`` uses ``convention_keepbits``."""
        from spacetimer.bitround import apply_keepbits, convention_keepbits

        if keepbits is None:
            keepbits = convention_keepbits(self._ds)
        return apply_keepbits(self._ds, keepbits=keepbits)

    def report(
        self,
        encoding: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        compression_ratio: float = 0.2,
        spatial_dims: Collection[str] = SPATIAL_DIMS,
        time_dim: str | None = None,
    ) -> Report:
        """See ``spacetimer.report``."""
        return report(
            self._ds,
            encoding,
            compression_ratio=compression_ratio,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )
