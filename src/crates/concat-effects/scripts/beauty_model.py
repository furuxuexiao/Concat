#!/usr/bin/env python3
"""concat.beauty's shader in Python, and the check of its probes.

    src/crates/concat-effects/scripts/beauty_model.py             # every probe
    src/crates/concat-effects/scripts/beauty_model.py --print     # with values
    src/crates/concat-effects/scripts/beauty_model.py --clusters  # needs numpy

A probe is one flat colour, where smoothing has nothing to even, so what
it pins is Whiten and the skin mask: the `expect` colours in
packages/concat.beauty/fixtures.toml are what `pixel` below gives for each
probe's input, rounded to the half floats the renderer stores, and
concat-render's `every_package_probe_holds` checks them on the GPU. After
changing effect.wgsl, change this to match, rerun it, and take any value it
shows moved into fixtures.toml. It exits non-zero when a fixture is further
from the model than the probe's tolerance.

`--clusters` runs `picture`, the whole shader over a picture - the blur,
the line test, the blemish fill and the shape test that tells a cluster of
spots from lips - on the close clusters of concat-render's
`beauty_bounds_close_clusters`, and prints how far each still stands out,
as that test measures it; it matches the GPU to about 0.001. Use it to try
a change to the shape test before building.

The probe check needs only the standard library (Python 3.11 or later, for
tomllib); `--clusters` needs numpy.
"""
import math
import struct
import sys
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "packages" / "concat.beauty"


def luma(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def hue_of(c):
    mx, mn = max(c), min(c)
    d = mx - mn
    if d < 1e-4:
        return 0.0
    r, g, b = c
    if mx == r:
        h = (g - b) / d
    elif mx == g:
        h = 2.0 + (b - r) / d
    else:
        h = 4.0 + (r - g) / d
    return (h / 6.0 - math.floor(h / 6.0)) * 360.0


def signed_hue(c):
    h = hue_of(c)
    return h - 360.0 if h > 180.0 else h


def skins(c):
    """The shader's `skins`: (to smooth, to whiten)."""
    h = signed_hue(c)
    mx = max(c)
    chroma = mx - min(c)
    sat = chroma / max(mx, 1e-6)
    rest = ((1 - smoothstep(40, 53, h)) * smoothstep(0, 0.08, chroma)
            * (1 - smoothstep(0.65, 0.85, sat)) * smoothstep(0.12, 0.22, mx))
    return smoothstep(0, 10, h) * rest, smoothstep(7, 17, h) * rest


def whiten(c, w, pale):
    """The shader's Whiten on `c` (display encoding) at `w` (0..1), by `pale`."""
    if w <= 0 or pale <= 0:
        return list(c)
    below = [min(max(x, 0.0), 1.0) for x in c]
    lifted = [x ** (1.0 / (1.0 + w * 0.25)) for x in below]
    y = luma(lifted)
    lifted = [y + (x - y) * (1.0 - w * 0.15) + (o - b) for x, o, b in zip(lifted, c, below)]
    return [o + (l - o) * pale for o, l in zip(c, lifted)]


def pixel(c, smoothing, w):
    """The shader on a flat colour `c` (display encoding): smoothing evens
    a flat colour into itself and fills nothing, so only Whiten is left."""
    if not all(math.isfinite(x) for x in c):
        return list(c)
    return whiten(c, w / 100.0, skins(c)[1])


def to_display(x):
    return math.copysign(abs(x) ** (1 / 2.4), x)


def from_display(x):
    return math.copysign(abs(x) ** 2.4, x)


def half(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]


def run(probe, params):
    values = {}
    for param in params:
        at = probe.get("at", "default")
        values[param["key"]] = param[at] if at in ("min", "max") else param["default"]
    values.update(probe.get("params", {}))
    rgba = [half(x) for x in probe["input"]]
    out = pixel([to_display(x) for x in rgba[:3]], values["smoothing"], values["whiten"])
    return [half(from_display(x)) for x in out] + [rgba[3]]


def check(show):
    params = tomllib.loads((PACKAGE / "effect.toml").read_text())["param"]
    probes = tomllib.loads((PACKAGE / "fixtures.toml").read_text())["probe"]
    bad = 0
    for probe in probes:
        got = run(probe, params)
        tolerance = probe.get("tolerance", 0.004)
        ok = all(
            abs(g - e) <= tolerance * max(abs(e), 0.01) for g, e in zip(got, probe["expect"])
        )
        bad += not ok
        if show or not ok:
            print(f"{'ok  ' if ok else 'DIFF'} {probe['name']}: model {[round(x, 6) for x in got]}"
                  f", fixture {probe['expect']}")
    return bad


# The whole shader over a picture, with numpy.

TURN = (0.894, 0.447)
DIRS = [(1.0, 0.0), (0.0, 1.0), (0.7071, 0.7071), (0.7071, -0.7071)]


def picture(img, smoothing, w, frame=None):
    """The shader over `img` (H x W x 4, linear light, straight alpha, as
    the GPU tests upload it), laid at the top left of a transparent layer
    `frame` = (width, height) in size. Returns the result in the display
    encoding, straight."""
    import numpy as np

    def ss(a, b, x):
        t = np.clip((x - a) / (b - a), 0.0, 1.0)
        return t * t * (3 - 2 * t)

    def disp(x):
        return np.sign(x) * np.abs(x) ** (1 / 2.4)

    def lu(c):
        return c[..., 0] * 0.2126 + c[..., 1] * 0.7152 + c[..., 2] * 0.0722

    def chroma(c):
        return c[..., :3].max(-1) - c[..., :3].min(-1)

    def shue(c):
        r, g, b = c[..., 0], c[..., 1], c[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        d = mx - np.minimum(np.minimum(r, g), b)
        dd = np.where(d < 1e-4, 1.0, d)
        h = np.where(mx == r, (g - b) / dd, np.where(mx == g, 2 + (b - r) / dd, 4 + (r - g) / dd))
        h = np.where(d < 1e-4, 0.0, (h / 6.0) % 1.0 * 360.0)
        return np.where(h > 180, h - 360, h)

    def skins_np(c, h=None):
        h = shue(c) if h is None else h
        mx = c[..., :3].max(-1)
        ch = chroma(c)
        rest = ((1 - ss(40, 53, h)) * ss(0, 0.08, ch)
                * (1 - ss(0.65, 0.85, ch / np.maximum(mx, 1e-6))) * ss(0.12, 0.22, mx))
        return ss(0, 10, h) * rest, ss(7, 17, h) * rest

    h_, w_, _ = img.shape
    fw, fh = frame or (w_, h_)
    full = np.zeros((fh, fw, 4))
    full[:h_, :w_] = img
    full = full.astype(np.float16).astype(np.float64)
    shown = np.concatenate([disp(full[..., :3]), full[..., 3:]], -1)

    def load(ix, iy):
        return shown[np.clip(iy, 0, fh - 1), np.clip(ix, 0, fw - 1)]

    def level(px, py):
        # The hardware filters the texture in light, then it is encoded.
        x, y = px - 0.5, py - 0.5
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]

        def at(a, b):
            return full[np.clip(b, 0, fh - 1), np.clip(a, 0, fw - 1)]

        v = (at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy)
             + at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy)
        a = np.clip(v[..., 3], 0, 1)
        return disp(v[..., :3]) * a[..., None], a

    ys, xs = np.mgrid[0:h_, 0:w_]
    cx, cy = xs + 0.5, ys + 0.5
    c = load(xs, ys)
    rgb, alpha = c[..., :3], c[..., 3]
    s = smoothing / 100.0
    wt = w / 100.0
    hue = shue(rgb)
    mine_x, mine_y = skins_np(rgb, hue)
    warm = ss(-25, -15, hue) * (1 - ss(55, 65, hue))
    lines = min(fw, fh) / 1080.0
    y0 = lu(rgb)
    tint0 = rgb - y0[..., None]

    t = 1.5 * max(lines, 1.0)
    opacity = np.maximum(alpha, 1e-3)
    line = np.zeros_like(y0)
    for ax, ay in DIRS:
        bx, by = -ay, ax

        def apart(px, py):
            p, a = level(px, py)
            return (lu(p) - y0 * a) / opacity
        A, B = apart(cx + ax * t, cy + ay * t), apart(cx - ax * t, cy - ay * t)
        C, D = apart(cx + bx * t, cy + by * t), apart(cx - bx * t, cy - by * t)
        line = np.maximum(line, np.exp(-(A * A + B * B) / 0.002) * ss(0.05, 0.1, np.minimum(C, D)))

    step = 3.5 * max(lines, 0.4)

    def turned(gx, gy):
        return gx * TURN[0] - gy * TURN[1], gx * TURN[1] + gy * TURN[0]

    def tap_at(gx, gy, f=1.0):
        tx, ty = turned(gx, gy)
        p = load(np.floor(cx + tx * step * f).astype(int), np.floor(cy + ty * step * f).astype(int))
        return p[..., :3], np.clip(p[..., 3], 0, 1)

    def own_kind(trgb, ta):
        ty = lu(trgb)
        dy = ty - y0
        dt = (trgb - ty[..., None]) - tint0
        return np.exp(-(dy * dy) / 0.015 - (dt * dt).sum(-1) / 0.0006) * ta

    total, total_close = np.full_like(y0, 0.25), np.full_like(y0, 0.25)
    acc, acc_close = rgb * 0.25, rgb * 0.25
    fill, ring, ring_y, ring_yy = np.zeros_like(rgb), np.zeros_like(y0), np.zeros_like(y0), np.zeros_like(y0)
    kinds = {(0, 0): np.ones_like(y0)}
    for gy in range(-2, 3):
        for gx in range(-2, 3):
            if gx == 0 and gy == 0:
                continue
            trgb, ta = tap_at(gx, gy)
            theirs_x, theirs_y = skins_np(trgb)
            theirs_x, theirs_y = theirs_x * ta, theirs_y * ta
            ty = lu(trgb)
            dy = ty - y0
            dt = (trgb - ty[..., None]) - tint0
            dtt = (dt * dt).sum(-1)
            off = dtt * 30.0
            gap = 1 - ss(0.18 - off, 0.26 - off, np.abs(dy))
            near = np.exp(-(dy * dy) / (0.2 + (0.002 - 0.2) * line) - dtt / 0.0006) * gap
            acc += trgb * (near * theirs_x)[..., None]
            total += near * theirs_x
            close = near * np.exp(-(dy * dy) / 0.003)
            acc_close += trgb * (close * theirs_x)[..., None]
            total_close += close * theirs_x
            kind = own_kind(trgb, ta)
            kinds[(gx, gy)] = kind
            high = trgb.max(-1)
            plain = theirs_y * ss(0.12, 0.2, chroma(trgb) / np.maximum(high, 1e-6)) * (1 - kind)
            fill += trgb * plain[..., None]
            ring += plain
            ring_y += ty * plain
            ring_yy += ty * ty * plain
    around = fill / np.maximum(ring, 1e-6)[..., None]
    ring_mean = ring_y / np.maximum(ring, 1e-6)
    ring_spread = np.maximum(ring_yy / np.maximum(ring, 1e-6) - ring_mean ** 2, 0)
    darker = ss(0.0, 0.03, ring_mean - y0)
    redder = ss(2.0, 8.0, shue(around) - hue)
    candidate = (ss(0.55, 0.85, ring / 24.0) * (1 - ss(0.004, 0.01, ring_spread)) * (1 - line)
                 * ss(0.05, 0.1, chroma(rgb)) * warm * np.maximum(darker, redder)
                 * (1 - ss(0.2, 0.3, np.abs(lu(around) - y0))))

    # spot_shape, worked out everywhere (the shader only where candidate > 0).
    for gy in range(-1, 2):
        for gx in range(-1, 2):
            if (gx, gy) != (0, 0):
                kinds[(gx, gy)] = np.minimum(kinds[(gx, gy)], own_kind(*tap_at(gx, gy, 0.5)))
    far, joined = np.zeros_like(y0), np.zeros_like(y0)
    for gy in range(-2, 3):
        for gx in range(-2, 3):
            if gx * gx + gy * gy < 4:
                continue
            via = np.zeros_like(y0)
            for vy in range(math.floor(gy / 2), math.ceil(gy / 2) + 1):
                for vx in range(math.floor(gx / 2), math.ceil(gx / 2) + 1):
                    via = np.maximum(via, kinds[(vx, vy)])
            link = kinds[(gx, gy)] * via
            link = np.where(link < 0.01, 0.0, link)
            for eighth in range(2, 8):
                link = np.minimum(link, own_kind(*tap_at(gx, gy, eighth / 8.0)))
            far = np.maximum(far, link)
            joined += link
    k = kinds
    alike = joined + k[(-1, -1)] + k[(0, -1)] + k[(1, -1)] + k[(-1, 0)] + k[(1, 0)] + k[(-1, 1)] + k[(0, 1)] + k[(1, 1)]
    along = np.maximum(np.minimum(k[(1, 0)], k[(-1, 0)]), np.minimum(k[(0, 1)], k[(0, -1)]))
    across = np.minimum(np.maximum(k[(1, 0)], k[(-1, 0)]), np.maximum(k[(0, 1)], k[(0, -1)]))
    shape = (1 - ss(0.3, 0.6, np.maximum(far, along * (1 - across)))) * (1 - ss(0.15, 0.3, alike / 24.0))
    blemish = candidate * shape

    lighter = ss(0.04, 0.1, y0 - ring_mean)[..., None]
    even = acc / total[..., None] * (1 - lighter) + acc_close / total_close[..., None] * lighter
    goal = rgb + (even - rgb) * mine_x[..., None]
    smoothed = goal + (rgb - goal) * 0.1
    worked = (s > 0) & ((mine_x > 0) | ((chroma(rgb) > 0.05) & (warm > 0)))
    out = rgb + (smoothed - rgb) * (1 - (1 - s) ** 3)
    filled = around + (rgb - around) * 0.1
    spot = np.where(worked, blemish * ss(0.0, 0.4, s), 0.0)
    out = np.where(worked[..., None], out + (filled - out) * spot[..., None], rgb)
    pale = mine_y + (skins_np(out)[1] - mine_y) * spot
    if wt > 0:
        below = np.clip(out, 0, 1)
        lifted = below ** (1 / (1 + wt * 0.25))
        yl = lu(lifted)[..., None]
        lifted = yl + (lifted - yl) * (1 - wt * 0.15) + (out - below)
        out = out + (lifted - out) * pale[..., None]
    return out


def clusters():
    """concat-render's `beauty_bounds_close_clusters`, at 1080 lines."""
    import numpy as np

    def skin_field(feature):
        img = np.zeros((80, 64, 4))
        for y in range(80):
            for x in range(64):
                pore = 0.03 if (x * 7 + y * 13) % 5 < 2 else -0.02
                f = feature(x, y)
                img[y, x] = f if f is not None else [0.45 + pore, 0.25 + pore, 0.15 + pore, 1.0]
        return img

    def shown(img):
        return np.maximum(img[..., :3], 0) ** (1 / 2.4)

    for name, colour in [("acne", [0.33, 0.13, 0.08, 1.0]), ("red spots", [0.40, 0.10, 0.08, 1.0])]:
        for apart in [7.0, 9.0]:
            centres = [(32 - apart / 2, 30), (32 + apart / 2, 30), (32, 30 + apart * 0.866)]

            def off(x, y):
                return min(math.hypot(x + 0.5 - a, y + 0.5 - b) for a, b in centres)
            img = skin_field(lambda x, y: colour if off(x, y) <= 2.5 else None)
            before = shown(img)
            for smoothing, w in [(25, 30), (50, 30), (100, 30), (25, 100), (50, 100), (100, 100)]:
                got = picture(img, smoothing, w, frame=(1080, 1080))
                after = np.maximum(np.sign(got) * np.abs(got) ** 2.4, 0) ** (1 / 2.4)
                ring = [(x, y) for x in range(4, 60) for y in (4, 58)]
                sb = np.mean([before[y, x] for x, y in ring], 0)
                sa = np.mean([after[y, x] for x, y in ring], 0)
                worst = max(
                    np.abs(after[y, x] - sa).max() / np.abs(before[y, x] - sb).max()
                    for y in range(10, 54) for x in range(10, 54) if off(x, y) <= 2.5
                )
                print(f"{name} {apart:.0f} px apart, smoothing {smoothing}, whiten {w}: "
                      f"stands out {worst:.3f} as far")


if __name__ == "__main__":
    if "--clusters" in sys.argv:
        clusters()
        sys.exit(0)
    failed = check("--print" in sys.argv)
    print(f"{failed} probe(s) differ from the model" if failed else "every probe matches the model")
    sys.exit(1 if failed else 0)
