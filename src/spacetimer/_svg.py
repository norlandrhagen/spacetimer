"""Pixel-art chunk cube for ``Report._repr_html_``.

Same idea as ``dask.array.svg`` (oblique cube, capped line count), plus shards.
"""

import html
import math
from typing import NamedTuple

# docs/stylesheets/spacetimer.css tokens; notebooks do not load that CSS
FRONT, TOP, SIDE = "#145b5c", "#1e7f7a", "#0f3b44"
LINE, LABEL, ZOOM = "#3fc9bc", "#c9c4ee", "#ffd54a"

EDGE = 96  # units for the longest axis
MAX_LINES = 16  # per axis, per layer
SCALE = 2  # screen px per unit
GAP = 10  # units between figures


def _edge(n: int, biggest: int) -> int:
    """Log-scaled so 4480x32x32 stays readable."""
    return max(4, round(EDGE * math.log(n + 1) / math.log(biggest + 1)))


def _cuts(n: int, step: int, length: int) -> list[int]:
    """Unit offsets of block boundaries, every ``stride``-th when there are too many."""
    count = -(-n // step)
    stride = -(-count // (min(MAX_LINES, length // 6) + 1))
    cuts = {round(k * step / n * length) for k in range(stride, count, stride)}
    return sorted(cuts - {0, length})


def _poly(points: list[tuple[int, int]], fill: str) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}"/>'


class _Fig(NamedTuple):
    body: str
    w: int
    h: int
    d: int
    cell: tuple[int, int, int, int]  # x, y, width, height of one front-face chunk


def cube_svg(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None = None,
) -> str:
    """Up to three axes per figure, split like dask: the last three are the cube.

    Each leading figure is linked to the next by a zoom callout: one of its
    chunks (yellow) holds the whole figure to its right.
    """
    groups, start = [], 0
    while start < len(shape):
        stop = start + ((len(shape) - start) % 3 or 3)
        groups.append(range(start, stop))
        start = stop
    figs = [
        _cube(
            _pick(dims, g),
            _pick(shape, g),
            _pick(chunks, g),
            _pick(shards, g) if shards else None,
        )
        for g in groups
    ]
    height = max(f.h + f.d + 12 for f in figs)
    parts, spots, x = [], [], 0
    for f in figs:
        ox, oy = x + 10, height - (f.h + f.d + 10)
        spots.append((ox, oy))
        parts.append(f'<g transform="translate({ox} {oy})">{f.body}</g>')
        x += f.w + f.d + 12 + GAP
    for (ox, oy), f, (nx, ny), nxt in zip(spots, figs, spots[1:], figs[1:]):
        cx, cy, cw, ch = (f.cell[0] + ox, f.cell[1] + oy, *f.cell[2:])
        parts.append(
            f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="{ZOOM}"/>'
        )
        parts += [
            f'<line x1="{cx + cw}" y1="{cy + dy}" x2="{nx}" y2="{ny + nxt.d + ty}" '
            f'stroke="{ZOOM}" stroke-width="0.5" stroke-dasharray="2 2"/>'
            for dy, ty in ((0, 0), (ch, nxt.h))
        ]
    width = x - GAP
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width * SCALE}" height="{height * SCALE}" shape-rendering="crispEdges" '
        f'overflow="visible" stroke="{LINE}" stroke-width="1">{"".join(parts)}</svg>'
    )


def _pick[T](values: tuple[T, ...], axes: range) -> tuple[T, ...]:
    return tuple(values[i] for i in axes)


def _labels(items: list[tuple[str, int, int, int]], anchor: str = "middle") -> str:
    """(text, x, y, rotation) in units."""
    texts = "".join(
        f'<text x="{x}" y="{y}" transform="rotate({r} {x} {y})">{html.escape(t)}</text>'
        for t, x, y, r in items
    )
    return f'<g stroke="none" fill="{LABEL}" font-size="5" text-anchor="{anchor}">{texts}</g>'


def _cube(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> _Fig:
    """Axes are (depth, down, across); fewer axes draw flatter."""
    nd = len(shape)

    def last3(v: tuple[int, ...]) -> tuple[int, int, int]:
        z, y, x = (1,) * (3 - nd) + tuple(v[-nd:])
        return z, y, x

    nz, ny, nx = n3 = last3(shape)
    biggest = max(n3)
    w = _edge(nx, biggest)
    h = _edge(ny, biggest) if nd >= 2 else 6
    d = _edge(nz, biggest) // 2 if nd == 3 else 0

    def grid(steps: tuple[int, int, int], cls: str) -> str:
        sz, sy, sx = steps
        lines = []
        for x in _cuts(nx, sx, w):
            lines += [(x, d, x, d + h), (x, d, x + d, 0)]
        for y in _cuts(ny, sy, h):
            lines += [(0, d + y, w, d + y), (w, d + y, w + d, y)]
        for z in _cuts(nz, sz, d):
            lines += [(z, d - z, w + z, d - z), (w + z, d - z, w + z, d - z + h)]
        marks = "".join(
            f'<line x1="{a}" y1="{b}" x2="{c}" y2="{e}"/>' for a, b, c, e in lines
        )
        return f'<g class="{cls}" stroke-width="0.5">{marks}</g>'

    front = [(0, d), (w, d), (w, d + h), (0, d + h)]
    top = [(0, d), (w, d), (w + d, 0), (d, 0)]
    side = [(w, d), (w + d, 0), (w + d, h), (w, d + h)]
    body = [
        _poly(front, FRONT),
        _poly(top, TOP),
        _poly(side, SIDE),
        # REPORT_CSS shows one of the two; "show shards" swaps them
        grid(last3(chunks), "st-chunk"),
        grid(last3(shards), "st-shard") if shards else "",
        _poly(front, "none"),
        _poly(top, "none"),
        _poly(side, "none"),
    ]
    # axis names: across under the front face, down beside it, depth along the top edge
    spots = [
        (w // 2, d + h + 7, 0),
        (-3, d + h // 2, -90),
        (d // 2 - 3, d // 2 - 3, -45),
    ]
    body.append(_labels([(name, *at) for name, at in zip(reversed(dims), spots)]))
    _, cy, cx = last3(chunks)
    xs, ys = _cuts(nx, cx, w), _cuts(ny, cy, h)
    # rightmost full-width chunk in the top row; the last one can be a sliver
    edges = [0, *xs, w]
    if len(edges) > 2 and edges[-1] - edges[-2] < edges[1]:
        edges.pop()
    cell = (edges[-2], d, edges[-1] - edges[-2], ys[0] if ys else h)
    return _Fig("".join(body), w, h, d, cell)
