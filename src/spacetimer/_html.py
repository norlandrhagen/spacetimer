"""Notebook HTML for ``Report._repr_html_``."""

from __future__ import annotations

import base64
import html
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spacetimer._svg import cube_svg
from spacetimer.core import Costs, MiB, VarSpec

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
    spec: VarSpec, enc: Mapping[str, Any], costs: Costs, *, ratio: float, opened: bool
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
