"""Generate spacetimer pixel-art sprites as crisp SVGs (and optional PNGs).

Every sprite is defined on a pixel grid, so you can tweak a row and rerun:

    python tools/make_sprites.py            # writes docs/assets/sprites/*.svg
    python tools/make_sprites.py --png      # also writes PNGs (needs Pillow)

Output paths are relative to the repo root (run from the root).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

OUT = Path("docs/assets/sprites")

# ---------------------------------------------------------------- palette
VOID = "#0B0A1F"
INK = "#16132E"
NEBULA = "#B3A6F5"
NEBULA_HI = "#D6CEFF"
DEEP = "#3B2F8F"
PURPLE = "#6A5AD8"
TEAL = "#3FC9BC"
TEAL_HI = "#7FF0E4"
TEAL_MID = "#2A9C92"
TEAL_DK = "#1E7F7A"
TEAL_LINE = "#0F3B44"
ORANGE = "#FF8A3D"
ORANGE_HI = "#FFB062"
ORANGE_DK = "#C85A1E"
YELLOW = "#FFD54A"
HULL = "#D4D8E2"
HULL_MID = "#9AA1B2"
HULL_DK = "#565D72"
WHITE = "#FFFFFF"
NAVY_TILE = "#24204A"

SHIP_PAL = {
    "K": INK,
    "H": HULL,
    "h": HULL_MID,
    "d": HULL_DK,
    "G": "#46E0D2",
    "g": "#1F8F8A",
    "W": WHITE,
    "O": ORANGE,
    "Y": YELLOW,
    "P": NEBULA,
    "p": PURPLE,
}


# ---------------------------------------------------------------- sprites
def d(n: int) -> str:
    return "." * n


SHIP_ROWS = [
    d(17) + "KKKKKK" + d(17),
    d(15) + "KKGWWGGGKK" + d(15),
    d(14) + "KGWGGYOOGGgK" + d(14),
    d(8) + "KKKKK" + "KGGGGGOOOGGggK" + "KKKKK" + d(8),
    d(5) + "KKK" + "WHHhHHHHHhHHHHHhHHHHHhHH" + "KKK" + d(5),
    d(3) + "KK" + "HWWHHHhHHHHHhHHHHHhHHHHHhHHHHH" + "KK" + d(3),
    ".G" + "K" + "HHHHHHHHhHHHHHhHHHHHhHHHHHhHHHHYWH" + "K" + d(2),
    ".G" + "K" + "h" * 34 + "K" + d(2),
    d(2) + "K" + "HHHHHHGHhHHHHHhHHGHHhHHHHHhHHGHHHH" + "K" + d(2),
    d(2) + "K" + "h" * 34 + "K" + d(2),
    d(2) + "K" + "HHHHHHHHhHHHGHhHHHHHhHHGHHhHHHHHHH" + "K" + d(2),
    d(3) + "KK" + "ddddddKdddddKdddddKdddddKddddd" + "KK" + d(3),
    d(5) + "KKK" + "d" * 24 + "KKK" + d(5),
    d(8) + "K" * 24 + d(8),
]
# Nozzle: column 1, rows 6-7 (the two "G" pixels at the far left).


def grid_from_rows(rows, pal):
    w = max(len(r) for r in rows)
    return [[(pal[c] if c != "." else None) for c in r.ljust(w, ".")] for r in rows]


def grid_from_fn(w, h, fn):
    return [[fn(x, y) for x in range(w)] for y in range(h)]


def planet(x, y, spot=(2, 1)):
    cx = cy = 15.5
    R = 10.5
    dx, dy = x - cx, y - cy
    dist = math.hypot(dx, dy)
    a = -0.35
    rx = dx * math.cos(a) - dy * math.sin(a)
    ry = dx * math.sin(a) + dy * math.cos(a)
    e = math.hypot(rx / 15, ry / 4.2)
    if 0.84 < e < 1.1 and (ry > 0 or dist > R + 1):
        return YELLOW if e < 0.97 else ORANGE
    if dist <= R:
        if (x - 5) % 6 == 0 or (y - 5) % 6 == 0:
            return TEAL_LINE
        tx, ty = (x - 5) // 6, (y - 5) // 6
        light = (-dx - dy) / (2 * R)
        if x == 10 and y == 9:
            return WHITE
        if (tx, ty) == spot:
            return ORANGE_HI if light > 0.15 else ORANGE if light > -0.25 else ORANGE_DK
        return TEAL_HI if light > 0.2 else TEAL if light > -0.25 else TEAL_DK
    if dist <= R + 1.1:
        return INK
    return None


def clock(x, y, turn=0.62):
    cx, cy, R = 13, 17, 12
    if y <= 1 and 10 <= x <= 16:
        return NEBULA_HI if y == 0 else NEBULA
    if 2 <= y <= 5 and 12 <= x <= 14:
        return PURPLE
    if 21 <= x <= 23 and 5 <= y <= 7:
        return PURPLE
    dx, dy = x - cx, y - cy
    dist = math.hypot(dx, dy)
    if dist > R + 0.5:
        return None
    if dist > R - 0.6:
        return INK
    if dist > R - 2.6:
        return NEBULA_HI if dx + dy < -2 else NEBULA
    ux, uy = x - cx + 9, y - cy + 9
    if ux % 6 == 0 or uy % 6 == 0:
        return VOID
    tx, ty = ux // 6, uy // 6
    sweep = turn * math.pi * 2
    hx, hy = math.sin(sweep), -math.cos(sweep)
    t = 0.0
    while t <= 8:
        if round(cx + hx * t) == x and round(cy + hy * t) == y:
            return WHITE
        t += 0.5
    if x == cx and cy - 7 <= y < cy:
        return WHITE
    if tx == 1 and ty == 1:
        return ORANGE
    ang = math.atan2(6 * tx - 6, -(6 * ty - 6))
    if ang < 0:
        ang += math.pi * 2
    if ang <= sweep:
        return TEAL_HI if (ux % 6 == 1 and uy % 6 == 1) else TEAL
    return NAVY_TILE


def favicon(x, y):
    if x in (0, 15) or y in (0, 15):
        return INK
    if x in (5, 10) or y in (5, 10):
        return VOID
    tx = 0 if x < 5 else 1 if x < 10 else 2
    ty = 0 if y < 5 else 1 if y < 10 else 2
    lx, ly = x - tx * 5, y - ty * 5
    if tx == 2 and ty == 0:
        return YELLOW if (lx == 1 and ly == 1) else ORANGE
    if lx == 1 and ly == 1:
        return TEAL_HI
    return TEAL if (tx + ty) % 2 == 0 else TEAL_MID


def scale(x, y, tilt=0):
    dy = round(tilt * (16.5 - x) / 10)  # beam offset: left pan down when tilt > 0

    def block(x0, y0, fill, line, hi):
        if not (x0 <= x <= x0 + 7 and y0 <= y <= y0 + 7):
            return None
        if x in (x0, x0 + 7) or y in (y0, y0 + 7):
            return INK
        if x == x0 + 4 or y == y0 + 4:
            return line
        if x == x0 + 1 and y == y0 + 1:
            return hi
        return fill

    a = block(3, 4 + tilt, TEAL, TEAL_LINE, TEAL_HI)
    if a:
        return a
    b = block(23, 4 - tilt, NEBULA, DEEP, NEBULA_HI)
    if b:
        return b
    if 12 <= y - dy <= 13 and 1 <= x <= 32:
        if x in (1, 32):
            return INK
        if x in (16, 17):
            return ORANGE
        return HULL if y - dy == 12 else HULL_MID
    if 14 <= y <= 23:
        hw = (y - 14) // 2 + 1
        if 17 - hw <= x <= 16 + hw:
            return PURPLE if x in (17 - hw, 16 + hw) else NEBULA
    if 24 <= y <= 25 and 9 <= x <= 24:
        return HULL_DK
    return None


def console(x, y, spot=(4, 1, 4)):
    sx, sy, pitch = spot  # pitch: tile size in px, 8 = zoomed in, 2 = zoomed out
    if y <= 3:
        if (x == 0 and y <= 1) or (x == 1 and y == 0):
            return None
        if x == 33 and y in (0, 3):
            return None
        if x in (18, 26):
            return None
        return NEBULA if x < 18 else ORANGE if x < 26 else YELLOW
    if x <= 6:
        if y in (12, 19):
            return None
        if x == 0 and y == 25:
            return None
        return NEBULA if y < 12 else ORANGE if y < 19 else TEAL
    if x in (7, 8) and y == 4:
        return NEBULA
    if x == 7 and y == 5:
        return NEBULA
    if y >= 6 and 9 <= x <= 32:
        ux, uy = x - 9, y - 6
        if uy >= 16:
            if 23 <= y <= 24:
                if 9 <= x <= 17:
                    return TEAL
                if 19 <= x <= 23:
                    return YELLOW
                if 25 <= x <= 32:
                    return NEBULA
            return None
        if ux % pitch == pitch - 1 or uy % pitch == pitch - 1:
            return None
        tx, ty = ux // pitch, uy // pitch
        if (tx, ty) == (sx, sy):
            return ORANGE
        if ux // 4 > uy // 4 + 3:
            return NAVY_TILE
        return TEAL_MID if (tx + ty * 2) % 3 == 0 else TEAL
    return None


def hole(x, y):
    """Page-transition black hole: black disc with stair-stepped accretion rings."""
    r = math.hypot(x - 31.5, y - 31.5)
    for limit, colour in (
        (23, "#000000"),
        (24.5, YELLOW),
        (27, ORANGE),
        (28.5, ORANGE_DK),
    ):
        if r < limit:
            return colour
    return None


# ---------------------------------------------------------------- star tiles
def star_tile(seed, n, w, h, size, colors, alpha_range):
    """Seamless horizontally-repeating star tile (stars never straddle the seam)."""
    rnd = random.Random(seed)
    stars = []
    for _ in range(n):
        x = rnd.randrange(0, (w - size) // 2) * 2
        y = rnd.randrange(0, (h - size) // 2) * 2
        c = rnd.choice(colors)
        a = round(rnd.uniform(*alpha_range), 2)
        stars.append((x, y, c, a))
    return stars


GLYPHS = {
    "Z": ["###", "..#", ".#.", "#..", "###"],
    "A": [".#.", "#.#", "###", "#.#", "#.#"],
    "R": ["##.", "#.#", "##.", "#.#", "#.#"],
    "C": ["###", "#..", "#..", "#..", "###"],
    "D": ["##.", "#.#", "#.#", "#.#", "##."],
    "E": ["###", "#..", "###", "#..", "###"],
    "G": ["###", "#..", "#.#", "#.#", "###"],
    "H": ["#.#", "#.#", "###", "#.#", "#.#"],
    "I": ["###", ".#.", ".#.", ".#.", "###"],
    "K": ["#.#", "#.#", "##.", "#.#", "#.#"],
    "M": ["#.#", "###", "###", "#.#", "#.#"],
    "N": ["###", "#.#", "#.#", "#.#", "#.#"],
    "O": ["###", "#.#", "#.#", "#.#", "###"],
    "P": ["###", "#.#", "###", "#..", "#.."],
    "S": ["###", "#..", "###", "..#", "###"],
    "U": ["#.#", "#.#", "#.#", "#.#", "###"],
    " ": [],
}
# Star-words that drift through the splash; ZARR first so its files are stable.
WORDS = ["ZARR", "ICECHUNK", "CODEC", "CHUNK GRID", "COMPRESSION", "SHARD"]


def word_points(word, x0, y0, unit, seed=77):
    rnd = random.Random(seed)
    pts = []
    for i, letter in enumerate(word):
        for r, row in enumerate(GLYPHS[letter]):
            for c, ch in enumerate(row):
                if ch == "#":
                    jx, jy = rnd.randint(-1, 1), rnd.randint(-1, 1)
                    pts.append((x0 + i * 4 * unit + c * unit + jx, y0 + r * unit + jy))
    return pts


# ---------------------------------------------------------------- writers
def svg_from_grid(grid, title):
    h, w = len(grid), len(grid[0])
    rects = []
    for y, row in enumerate(grid):
        x = 0
        while x < w:  # merge horizontal runs of one colour into a single rect
            c = row[x]
            if c is None:
                x += 1
                continue
            run = 1
            while x + run < w and row[x + run] == c:
                run += 1
            rects.append(f'<rect x="{x}" y="{y}" width="{run}" height="1" fill="{c}"/>')
            x += run
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'shape-rendering="crispEdges" role="img" aria-label="{title}">\n<title>{title}</title>\n'
        + "\n".join(rects)
        + "\n</svg>\n"
    )


def svg_stars(stars, w, h, size):
    rects = [
        f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{c}" fill-opacity="{a}"/>'
        for x, y, c, a in stars
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'shape-rendering="crispEdges" aria-hidden="true">\n'
        + "\n".join(rects)
        + "\n</svg>\n"
    )


def png_from_grid(grid, path, scale_by, bg=None):
    from PIL import Image

    h, w = len(grid), len(grid[0])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0) if bg is None else bg)
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            if c:
                img.putpixel(
                    (x, y), tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) + (255,)
                )
    img.resize((w * scale_by, h * scale_by), Image.NEAREST).save(path)


def strip(w, h, fn, key, values):
    """Frames side by side, one per value of fn's `key` argument (hover animations)."""
    frames = [
        grid_from_fn(w, h, lambda x, y, v=v: fn(x, y, **{key: v})) for v in values
    ]
    return [[c for f in frames for c in f[y]] for y in range(h)]


FRAMES = 8  # must match steps() in spacetimer.css
PLANET_SPOTS = [(2, 1), (3, 1), (3, 2), (2, 2), (1, 2), (0, 2), (0, 1), (1, 1)]
CLOCK_TURNS = [(0.62 + i / FRAMES) % 1 for i in range(FRAMES)]
CONSOLE_SPOTS = [
    (1, 0, 8),
    (1, 0, 8),
    (2, 1, 4),
    (3, 1, 4),
    (6, 2, 2),
    (7, 2, 2),
    (3, 1, 4),
    (2, 1, 4),
]
SCALE_TILTS = [0, 1, 2, 1, 0, -1, -2, -1]

SPRITES = {
    "ship": (grid_from_rows(SHIP_ROWS, SHIP_PAL), "spacetimer chunk cruiser"),
    "logo-planet": (grid_from_fn(32, 32, planet), "spacetimer logo: sharded planet"),
    "favicon": (grid_from_fn(16, 16, favicon), "spacetimer favicon"),
    "icon-space": (grid_from_fn(32, 32, planet), "Space-optimized"),
    "icon-time": (grid_from_fn(27, 30, clock), "Time-optimized"),
    "icon-web": (grid_from_fn(34, 26, console), "Web-optimized"),
    "icon-space-strip": (
        strip(32, 32, planet, "spot", PLANET_SPOTS),
        "Space-optimized",
    ),
    "icon-time-strip": (strip(27, 30, clock, "turn", CLOCK_TURNS), "Time-optimized"),
    "icon-web-strip": (strip(34, 26, console, "spot", CONSOLE_SPOTS), "Web-optimized"),
    "icon-balanced": (grid_from_fn(34, 26, scale), "Balanced"),
    "hole": (grid_from_fn(64, 64, hole), "black hole"),
    "icon-balanced-strip": (strip(34, 26, scale, "tilt", SCALE_TILTS), "Balanced"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true", help="also export PNGs")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, (grid, title) in SPRITES.items():
        (out / f"{name}.svg").write_text(svg_from_grid(grid, title))

    # Seamless star tiles for the parallax layers (1024 px wide, 360 px tall).
    W, H = 1024, 360
    (out / "stars-far.svg").write_text(
        svg_stars(
            star_tile(3, 90, W, H, 2, [WHITE, WHITE, WHITE, NEBULA], (0.35, 0.8)),
            W,
            H,
            2,
        )
    )
    (out / "stars-mid.svg").write_text(
        svg_stars(
            star_tile(9, 34, W, H, 4, [WHITE, WHITE, NEBULA], (0.6, 0.95)), W, H, 4
        )
    )
    (out / "stars-dust.svg").write_text(
        svg_stars(star_tile(15, 14, W, H, 6, [TEAL], (0.3, 0.4)), W, H, 6)
    )

    # Constellation easter eggs: a "dim" version that looks like any other far
    # star, and a "lit" version the splash animation steps up to.
    for i, word in enumerate(WORDS):
        slug = word.lower().replace(" ", "-")
        pts = word_points(word, 3, 3, 11, seed=77 + i)
        zw, zh = len(word) * 4 * 11, 5 * 11 + 6
        rnd = random.Random(5 + i)
        dim = [(x, y, WHITE, round(rnd.uniform(0.3, 0.55), 2)) for x, y in pts]
        (out / f"{slug}-dim.svg").write_text(svg_stars(dim, zw, zh, 2))
        lit = [(x - 1, y - 1, WHITE, 1) for x, y in pts]
        (out / f"{slug}-lit.svg").write_text(svg_stars(lit, zw, zh, 4))

    if args.png:
        png = out / "png"
        png.mkdir(exist_ok=True)
        for n in (16, 32, 48):
            png_from_grid(SPRITES["favicon"][0], png / f"favicon-{n}.png", n // 16)
        png_from_grid(SPRITES["logo-planet"][0], png / "logo-planet-512.png", 16)
        png_from_grid(SPRITES["logo-planet"][0], png / "logo-planet-256.png", 8)
        png_from_grid(
            SPRITES["logo-planet"][0],
            png / "apple-touch-icon.png",
            5,
            bg=(11, 10, 31, 255),
        )
        for s in (4, 8):
            png_from_grid(SPRITES["ship"][0], png / f"ship-{s}x.png", s)
        for k in ("icon-space", "icon-time", "icon-web", "icon-balanced"):
            png_from_grid(SPRITES[k][0], png / f"{k}-4x.png", 4)
    print(f"wrote sprites to {out}")


if __name__ == "__main__":
    main()
