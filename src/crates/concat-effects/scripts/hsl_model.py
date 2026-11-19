#!/usr/bin/env python3
"""concat.hsl's shader, line for line in Python, and the check of its probes.

    src/crates/concat-effects/scripts/hsl_model.py           # every probe
    src/crates/concat-effects/scripts/hsl_model.py --print   # with values

The `expect` colours in packages/concat.hsl/fixtures.toml are what this
model gives for each probe's input, rounded to the half floats the renderer
stores; concat-render's `every_package_probe_holds` then checks them on the
GPU. After changing effect.wgsl, change this to match, rerun it, and copy
any value it prints that moved into fixtures.toml. It exits non-zero when a
fixture is further from the model than the probe's tolerance.

Standard library only (Python 3.11 or later, for tomllib).
"""
import math
import struct
import sys
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "packages" / "concat.hsl"

# The band centres, in degrees round the HSV wheel (CENTRES in the shader).
CENTRES = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0]
BANDS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]


def luma(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def fract(x):
    return x - math.floor(x)


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
    return fract(h / 6.0) * 360.0


def from_hsv(h, s, v):
    h = fract(h / 360.0)
    out = []
    for k in (1.0, 2.0 / 3.0, 1.0 / 3.0):
        p = abs(fract(h + k) * 6.0 - 3.0)
        ramp = min(max(p - 1.0, 0.0), 1.0)
        out.append(v * (1.0 + (ramp - 1.0) * s))
    return out


def to_display(x):
    return math.copysign(abs(x) ** (1 / 2.4), x)


def from_display(x):
    return math.copysign(abs(x) ** 2.4, x)


def effect(c, p):
    """The shader's `effect` on one colour `c` in the display encoding,
    with the sliders `p` (every key, -100..100)."""
    if not all(math.isfinite(x) for x in c):
        return list(c)
    rgb = [max(x, 0.0) for x in c]
    value = max(rgb)
    chroma = max(rgb) - min(rgb)
    hsv = (hue_of(rgb), chroma / value if value > 1e-6 else 0.0, value)
    hues = [p[b + "_hue"] for b in BANDS]
    sats = [p[b + "_sat"] for b in BANDS]
    lums = [p[b + "_lum"] for b in BANDS]
    i = 7
    for n in range(7):
        if CENTRES[n] <= hsv[0] < CENTRES[n + 1]:
            i = n
    j = (i + 1) % 8
    span = CENTRES[j] - CENTRES[i]
    along = hsv[0] - CENTRES[i]
    if j == 0:
        span += 360.0
        along = fract(along / 360.0) * 360.0
    linear = along / span
    t = smoothstep(0.0, 1.0, linear)
    gate = smoothstep(0.0, 0.15, chroma)
    turns = [h * 0.3 for h in hues]
    shares = [1.0] * 8
    for n in range(8):
        m = (n + 1) % 8
        gap = (CENTRES[m] + 360.0 if m == 0 else CENTRES[m]) - CENTRES[n]
        squeeze = turns[n] - turns[m]
        if squeeze > gap * 0.8:
            share = gap * 0.8 / squeeze
            shares[n] = min(shares[n], share)
            shares[m] = min(shares[m], share)
    head = turns[i] * shares[i]
    tail = turns[j] * shares[j]
    hue = (head + (tail - head) * linear) * gate
    sat = (sats[i] + (sats[j] - sats[i]) * t) * gate
    lum = (lums[i] + (lums[j] - lums[i]) * t) * gate
    if hue == 0.0 and sat == 0.0 and lum == 0.0:
        return list(c)
    turned = from_hsv(hsv[0] + hue, hsv[1], hsv[2])
    y = luma(turned)
    low, high = min(turned), max(turned)
    wanted = 1.0 + sat / 100.0
    push = wanted
    if wanted > 1.0:
        most = 1e6
        if y > low:
            most = min(most, y / (y - low))
        if high > y:
            most = min(most, (max(high, 1.0) - y) / (high - y))
            most = min(most, 1.0 + max(1.0 - high, 0.0) / 0.25)
        room = max(most - 1.0, 0.0)
        d = wanted - 1.0
        r = d / max(room, 1e-4)
        push = 1.0 + d / (1.0 + r * r * r) ** (1.0 / 3.0)
    out = [y + (x - y) * push for x in turned]
    if lum < 0.0:
        out = [x * (1.0 + lum * 0.004) for x in out]
    elif lum > 0.0:
        before = luma(out)
        high_now = max(out)
        if 1e-6 < before < 1.0:
            aim = before + (1.0 - before) * lum * 0.004
            room = 1.0 / high_now if high_now * aim / before > 1.0 else aim / before
            lifted = [x * max(room, 1.0) for x in out]
            reached = luma(lifted)
            ceiling = [max(x, 1.0) for x in lifted]
            span = luma(ceiling) - reached
            rest = (aim - reached) / span if (reached < aim and span > 1e-6) else 0.0
            rest = min(max(rest, 0.0), 1.0)
            out = [a + (b - a) * rest for a, b in zip(lifted, ceiling)]
    return [max(x, 0.0) + min(o, 0.0) for x, o in zip(out, c)]


def half(x):
    """`x` as the renderer's half floats store it."""
    return struct.unpack("<e", struct.pack("<e", x))[0]


def run(probe, params):
    values = {}
    for param in params:
        at = probe.get("at", "default")
        values[param["key"]] = param[at] if at in ("min", "max") else param["default"]
    values.update(probe.get("params", {}))
    rgba = [half(x) for x in probe["input"]]
    shown = [to_display(x) for x in rgba[:3]]
    out = effect(shown, values)
    return [half(from_display(x)) for x in out] + [rgba[3]]


def check(manifest, fixtures, show):
    params = tomllib.loads(manifest.read_text())["param"]
    probes = tomllib.loads(fixtures.read_text())["probe"]
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


if __name__ == "__main__":
    failed = check(PACKAGE / "effect.toml", PACKAGE / "fixtures.toml", "--print" in sys.argv)
    print(f"{failed} probe(s) differ from the model" if failed else "every probe matches the model")
    sys.exit(1 if failed else 0)
