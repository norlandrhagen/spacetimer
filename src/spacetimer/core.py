from __future__ import annotations

import base64
import html
import math
import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import numpy as np
import xarray as xr
import zarr
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec

from spacetimer._svg import cube_svg

if TYPE_CHECKING:
    import pandas as pd

MiB = 1024 * 1024
# caps shard size in chunks; the zarr v3 shard index is 16 B per chunk, so 8 KiB.
# dynamical's worked example (1440x32x32 in 2880x384x384) holds 288
MAX_INNER_CHUNKS = 512
SPATIAL_DIMS = frozenset({"x", "y", "lat", "lon", "latitude", "longitude"})


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
    RAW = "raw"  # uncompressed bytes (web tiles are sized by extent)
    COMPRESSED = "compressed"  # converted to raw with compression_ratio


@dataclass(frozen=True)
class Preset:
    chunk_bytes: int
    basis: SizeBasis
    # fixed spatial tile edge, remaining budget goes to time; None = time chunk 1
    spatial_edge: int | None
    chunks_per_shard: int


PRESETS: dict[UseCase, Preset] = {
    # dynamical.org reformatters, analysis mode: ~3.5 MiB compressed, 32x32 tiles
    UseCase.TIME: Preset(int(3.5 * MiB), SizeBasis.COMPRESSED, 32, 8),
    # same byte target, spent on one time step of the largest tile that fits
    UseCase.SPACE: Preset(int(3.5 * MiB), SizeBasis.COMPRESSED, None, 8),
    # fixed 128 edge, replace with a measured value after Phase 2
    UseCase.BALANCED: Preset(int(3.5 * MiB), SizeBasis.COMPRESSED, 128, 8),
    # topozarr: 512 KiB raw tiles, 4x4 chunks per shard
    UseCase.WEB: Preset(512 * 1024, SizeBasis.RAW, None, 4),
}


@dataclass(frozen=True)
class VarSpec:
    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if len(self.dims) != len(self.shape) or any(n < 1 for n in self.shape):
            raise ValueError(
                f"{self.name!r}: dims {self.dims} and shape {self.shape} must "
                "have the same length, and every size must be >= 1"
            )


Data = xr.Dataset | zarr.Group | Iterable[VarSpec]


class Encoding(TypedDict):
    chunks: tuple[int, ...]
    shards: NotRequired[tuple[int, ...]]
    compressors: tuple[BytesBytesCodec, ...]


@dataclass(frozen=True)
class Costs:
    """Estimates; MiB are compressed at the given compression_ratio."""

    chunk_mib: float
    shard_mib: float | None
    objects: int  # shards, or chunks when unsharded
    slice_chunks: int  # chunks read for one full spatial slice
    slice_mib: float
    series_chunks: int  # chunks read for one point's full time series
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

    encoding: dict[str, Encoding] = {}
    skipped: list[str] = []
    for spec in _specs(data):
        spatial = [i for i, d in enumerate(spec.dims) if d in spatial_dims]
        if not spatial:
            skipped.append(spec.name)
            continue
        t = _time_axis(spec, spatial, time_dim)
        if (
            preset.spatial_edge is not None
            and t is None
            and len(spatial) < len(spec.dims)
        ):
            warnings.warn(
                f"{spec.name!r}: no time dim found in {spec.dims}, so the "
                f"{use_case.value!r} layout falls back to spatial tiles; pass time_dim=",
                stacklevel=2,
            )
        if (
            chunks_per_shard
            and (k := _per_dim(chunks_per_shard, len(spatial))) < chunks_per_shard
        ):
            warnings.warn(
                f"{spec.name!r}: chunks_per_shard={chunks_per_shard} reduced to {k} "
                f"to keep a shard at or under {MAX_INNER_CHUNKS} chunks",
                stacklevel=2,
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
            stacklevel=2,
        )
    return encoding


# no asterisks in this CSS: Python-Markdown reads them as emphasis when the docs embed it
# !important: JupyterLab and MkDocs Material both style tables in rendered HTML
REPORT_CSS = """<style>
.st-report{display:block;box-sizing:border-box;width:100%;max-width:900px;background:#14112e;color:#e7e4fa;border:2px solid #2b2557;
box-shadow:6px 6px 0 #07061a;padding:12px;margin:0 6px 6px 0;
font:11px/1.6 "Press Start 2P",ui-monospace,monospace}
.st-var{border-bottom:2px solid #2b2557;margin-bottom:8px;padding-bottom:6px}
.st-var:last-child{border:0;margin:0;padding:0}
.st-tog{display:block;cursor:pointer;color:#c9c4ee}
.st-tog b{color:#7ff0e4;font-weight:400}
.st-tog input{position:absolute;opacity:0}
.st-arrow::before{content:"▸";display:inline-block;width:1.2em;color:#3fc9bc;
font:16px/1 system-ui,sans-serif}
.st-tog input:checked+.st-arrow::before{content:"▾"}
.st-tog input:focus-visible+.st-arrow{outline:2px solid #ffd54a}
.st-tog:has(input:not(:checked))+.st-fold{display:none}
.st-fold{padding-left:1.6em}
.st-body{display:flex;flex-wrap:wrap;gap:16px;align-items:center;
justify-content:space-between;padding:8px 0}
.st-shard{display:none}
.st-var:has(.st-sh input:checked) .st-shard{display:inline}
.st-var:has(.st-sh input:checked) .st-chunk{display:none}
.st-sh{display:inline-block;margin-top:10px;color:#c9c4ee;cursor:pointer}
.st-sh input{accent-color:#3fc9bc;vertical-align:middle;margin:0 6px 0 0}
.st-report td.st-dims{text-align:center;color:#c9c4ee!important}
.st-note{color:#9c96c8;margin:6px 0 8px}
.st-read div{margin:4px 0}
.st-ic{display:inline-block;width:32px;height:32px;margin-right:10px;vertical-align:middle;
background:center no-repeat;image-rendering:pixelated}
.st-fig{flex:0 0 250px;height:240px;display:flex;align-items:center;justify-content:center}
.st-wide{flex-basis:100%}
.st-fig svg{max-width:100%;max-height:100%;width:auto;height:auto}
.st-enc{background:#07061a;border:2px solid #2b2557;padding:4px 8px;margin:6px 0 4px 1.6em}
.st-report .st-enc td{text-align:left}
.st-report .st-enc tr:first-child>th,.st-report .st-enc tr:first-child>td{border-top:0}
.st-report table{display:table;border-collapse:collapse;margin:0;border:0;font:inherit;font-size:10px}
.st-report tr{background:transparent!important}
.st-report th,.st-report td{padding:4px 7px;text-align:right;border:0;
border-top:2px solid #2b2557;color:#e7e4fa!important;background:transparent!important}
.st-report th{color:#3fc9bc!important;border-top:0;font-weight:400}
.st-report th:first-child,.st-report td:first-child{text-align:left}
</style>"""


def _icon_css(path: Path) -> str:
    b = path.read_bytes()
    # PNGs are 4x; bytes 16-24 of a PNG are its width and height
    w, h = int.from_bytes(b[16:20]) // 4, int.from_bytes(b[20:24]) // 4
    return (
        f".st-ic-{path.stem}{{background-size:{w}px {h}px;"
        f'background-image:url("data:image/png;base64,{base64.b64encode(b).decode()}")}}\n'
    )


# the docs' time and space sprites, inlined so a notebook needs no assets
REPORT_CSS = REPORT_CSS.replace(
    "</style>",
    "".join(map(_icon_css, sorted((Path(__file__).parent / "icons").glob("*.png"))))
    + "</style>",
)

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
    encoding: Mapping[str, Encoding]
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


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:,.0f} B" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TiB"


def _codec_row(codec: Any) -> tuple[str, str]:
    """Name and settings of a Zarr v3 codec, or a numcodecs one from a v2 store."""
    if hasattr(codec, "to_dict"):
        conf = codec.to_dict()
        name, params = conf["name"], conf.get("configuration", {})
    else:
        params = dict(codec.get_config())
        name = params.pop("id")
    return str(name), ", ".join(f"{k}={v}" for k, v in params.items())


def _toggle(label: str, *, opened: bool) -> str:
    return (
        f'<label class="st-tog"><input type="checkbox"{" checked" if opened else ""}>'
        f'<span class="st-arrow"></span>{label}</label>'
    )


def _var_html(
    spec: VarSpec, enc: Encoding, costs: Costs, *, ratio: float, opened: bool
) -> str:
    chunks, shards = enc["chunks"], enc.get("shards")
    cols = [("array", spec.shape), ("chunk", chunks)]
    if shards:
        cols.append(("shard", shards))
    raw = [math.prod(g) * spec.dtype.itemsize for _, g in cols]
    rows = [
        ("shape", ["x".join(map(str, g)) for _, g in cols]),
        ("size", [_fmt_bytes(b) for b in raw]),
        (
            "count",
            [
                f"{math.prod(math.ceil(n / k) for n, k in zip(spec.shape, g)):,}"
                for _, g in cols
            ],
        ),
    ]
    th = "".join(f"<th>{h}</th>" for h in ("", *(c for c, _ in cols)))
    trs = (
        f'<tr><th>dims</th><td colspan="{len(cols)}" class="st-dims">'
        f"{html.escape(' x '.join(spec.dims))}</td></tr>"
    ) + "".join(
        f"<tr><th>{label}</th>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
        for label, vals in rows
    )
    reads = "".join(
        f'<div><i class="st-ic st-ic-{icon}"></i>one {what} reads {n:,} chunk{"s" * (n != 1)}, '
        f"{_fmt_bytes(mib * MiB)}</div>"
        for icon, what, n, mib in (
            ("space", "map", costs.slice_chunks, costs.slice_mib),
            ("time", "time series", costs.series_chunks, costs.series_mib),
        )
    )
    note = f"sizes are uncompressed; reads assume compression_ratio={ratio:g}"

    def by_dim(grid: tuple[int, ...]) -> str:
        return ", ".join(f"{d}={n}" for d, n in zip(spec.dims, grid))

    kv = [("chunks", by_dim(chunks))]
    if shards:
        per = [s // c for s, c in zip(shards, chunks)]
        kv += [
            ("shards", by_dim(shards)),
            ("chunks per shard", f"{' x '.join(map(str, per))} = {math.prod(per):,}"),
        ]
    kv += [_codec_row(codec) for codec in enc.get("compressors", ())]
    enc_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in kv
    )
    head = (
        f"<b>{html.escape(spec.name)}</b> ({html.escape(', '.join(spec.dims))}) "
        f"{spec.dtype}"
    )
    swap = (
        '<label class="st-sh"><input type="checkbox">show shards</label>'
        if shards
        else ""
    )
    return (
        f'<div class="st-var">{_toggle(head, opened=opened)}<div class="st-fold">'
        f'<div class="st-body"><div><table class="st-info"><thead><tr>{th}</tr></thead>'
        f"<tbody>{trs}</tbody></table>{swap}</div>"
        f'<div class="st-fig{" st-wide" * (len(spec.dims) > 3)}">{cube_svg(spec.dims, spec.shape, chunks, shards)}</div></div>'
        f'<div class="st-read">{reads}</div><div class="st-note">{note}</div>'
        f"{_toggle('encoding', opened=False)}"
        f'<div class="st-fold st-enc"><table class="st-kv">{enc_rows}</table></div>'
        "</div></div>"
    )


def report(
    data: Data,
    encoding: Mapping[str, Encoding],
    *,
    compression_ratio: float = 0.2,
    spatial_dims: Collection[str] = SPATIAL_DIMS,
    time_dim: str | None = None,
) -> Report:
    """Estimate read and storage costs of ``encoding`` (any, not only ours)."""
    _check_ratio(compression_ratio)
    out: dict[str, Costs] = {}
    specs: dict[str, VarSpec] = {}
    for spec in _specs(data):
        if spec.name not in encoding:
            continue
        enc = encoding[spec.name]
        spatial = [i for i, d in enumerate(spec.dims) if d in spatial_dims]
        t = _time_axis(spec, spatial, time_dim)
        out[spec.name] = _costs(
            spec, enc["chunks"], enc.get("shards"), spatial, t, compression_ratio
        )
        specs[spec.name] = spec
    if missing := sorted(set(encoding) - set(out)):
        warnings.warn(f"encoding names {missing}, not found in data", stacklevel=2)
    return Report(out, encoding, compression_ratio, specs)


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
        # non-dimension coords (e.g. 2-D lat/lon), as Xarray records them
        coords = {
            c
            for attrs in (obj.attrs, *(a.attrs for a in arrays.values()))
            for c in str(attrs.get("coordinates", "")).split()
        }
        specs, unnamed = [], []
        for name, arr in arrays.items():
            names = getattr(arr.metadata, "dimension_names", None) or arr.attrs.get(
                "_ARRAY_DIMENSIONS"
            )
            if not isinstance(names, (list, tuple)):
                unnamed.append(name)
                continue
            dims = tuple(map(str, names))
            if dims and dims != (name,) and name not in coords:
                specs.append(VarSpec(name, dims, arr.shape, arr.dtype, dict(arr.attrs)))
        if unnamed:
            warnings.warn(f"no dimension names on {unnamed}, skipped", stacklevel=3)
        return specs
    if isinstance(obj, xr.DataArray):  # iterating one would yield its slices
        raise TypeError("pass a Dataset (da.to_dataset()), zarr.Group, or VarSpecs")
    return list(obj)


def _time_axis(spec: VarSpec, spatial: list[int], time_dim: str | None) -> int | None:
    if time_dim is not None:
        return spec.dims.index(time_dim) if time_dim in spec.dims else None
    other = [i for i in range(len(spec.dims)) if i not in spatial]
    # name match, forecast (init_time x lead_time) treats init_time as time
    return next((i for i in other if "time" in spec.dims[i]), None)


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

    if preset.spatial_edge is not None and t is not None:
        # fixed-edge tiles first, then time takes what they leave
        fixed = preset.spatial_edge ** len(spatial)
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
        encoding: Mapping[str, Encoding],
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
