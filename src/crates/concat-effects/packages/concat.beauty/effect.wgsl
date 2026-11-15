struct Params { smoothing: f32, whiten: f32 }

// How much a colour may be skin, read off the colour alone, as two masks
// from one reading: x to smooth, from a ruddy cheek (0 degrees, in full
// by 10); y to whiten, from 7 degrees, in full by 17, so lips are not
// paled. Both fade out by 53 degrees, drop paint-strong saturation, and
// drop what is darker than near-black hair or deep shadow. Brown hair,
// wood or a beige wall the very colour of skin still counts: telling
// those apart needs a face, which a filter does not have.
fn skins(rgb: vec3<f32>) -> vec2<f32> {
    return skins_at(rgb, signed_hue(rgb));
}

// Hue in degrees, -180..180, so the reds either side of 0 sit together.
fn signed_hue(rgb: vec3<f32>) -> f32 {
    let raw = hue_of(rgb);
    return select(raw, raw - 360.0, raw > 180.0);
}

// `skins` for a colour whose signed hue `h` is already known.
fn skins_at(rgb: vec3<f32>, h: f32) -> vec2<f32> {
    let mx = max(max(rgb.r, rgb.g), rgb.b);
    let chroma = chroma_of(rgb);
    let sat = chroma / max(mx, 1e-6);
    let rest = (1.0 - smoothstep(40.0, 53.0, h))
        * smoothstep(0.0, 0.08, chroma)
        * (1.0 - smoothstep(0.65, 0.85, sat))
        * smoothstep(0.12, 0.22, mx);
    return vec2<f32>(smoothstep(0.0, 10.0, h), smoothstep(7.0, 17.0, h)) * rest;
}

// Whether every channel of `c` is a number and finite: a pixel that is not
// is read as transparent, so one bad pixel never spreads to its neighbours.
fn finite(c: vec4<f32>) -> bool {
    let exponent = bitcast<vec4<u32>>(c) & vec4<u32>(0x7f800000u);
    return all(exponent != vec4<u32>(0x7f800000u));
}

// The layer's colour at `uv`, between pixels, read at the full-size level
// without derivatives, so a branch on what the pixel holds may read it -
// premultiplied by its alpha (held to 0..1), which is handed back beside
// it, so what a transparent pixel holds counts for nothing. A reading with
// a bad pixel in it reads as transparent.
fn level(uv: vec2<f32>) -> vec4<f32> {
    let c = into_space(textureSampleLevel(source, source_sampler, uv, 0.0));
    let a = clamp(c.a, 0.0, 1.0);
    return select(vec4<f32>(0.0), vec4<f32>(c.rgb * a, a), finite(c));
}

// The layer's colour at whole pixel `at`, clamped to the layer, its alpha
// held to 0..1; a pixel that is not a finite number reads as transparent.
fn pixel(at: vec2<f32>) -> vec4<f32> {
    let last = vec2<i32>(frame.size) - vec2<i32>(1);
    let c = into_space(textureLoad(source, clamp(vec2<i32>(floor(at)), vec2<i32>(0), last), 0));
    return select(vec4<f32>(0.0), vec4<f32>(c.rgb, clamp(c.a, 0.0, 1.0)), finite(c));
}

// The frame's shorter side over 1080: what the reaches below scale by, so a
// picture is smoothed alike whether it stands upright or lies on its side.
fn lines() -> f32 {
    return min(frame.size.x, frame.size.y) / 1080.0;
}

// How much the pixel lies on a thin line - a hair, a brow hair, a crease, a
// lash: along some direction its neighbours a pixel and a half away - at 1080
// pixels on the frame's shorter side and below, and as far again in
// proportion above - match it, while across that direction both are clearly
// lighter. A spot matches nowhere or everywhere, so it is never a line.
// Each neighbour is read premultiplied, so it is compared with the pixel's
// own brightness `y0` at the neighbour's opacity, and the difference read
// back at the pixel's own: a hair on a half-transparent layer, or across a
// fading edge, is told as on an opaque one, while a transparent neighbour
// differs by nothing and never draws a line.
fn on_line(uv: vec2<f32>, y0: f32, alpha: f32) -> f32 {
    let t = texel() * 1.5 * max(lines(), 1.0);
    let opacity = max(alpha, 1e-3);
    // Read without derivatives, so the reading may turn on the pixel.
    var dirs = array<vec2<f32>, 4>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.7071, 0.7071),
        vec2<f32>(0.7071, -0.7071),
    );
    var line = 0.0;
    for (var n = 0; n < 4; n++) {
        let along = dirs[n];
        let across = vec2<f32>(-along.y, along.x);
        let a = apart(level(uv + along * t), y0) / opacity;
        let b = apart(level(uv - along * t), y0) / opacity;
        let c = apart(level(uv + across * t), y0) / opacity;
        let d = apart(level(uv - across * t), y0) / opacity;
        let same = exp(-(a * a + b * b) / 0.002);
        let lighter = smoothstep(0.05, 0.1, min(c, d));
        line = max(line, same * lighter);
    }
    return line;
}

// How much brighter the premultiplied reading `n` is than brightness `y0`
// would be at its opacity.
fn apart(n: vec4<f32>, y0: f32) -> f32 {
    return luma(n.rgb) - y0 * n.a;
}

// The cosine and sine the tap grid is turned by: a slope of one in two,
// about 27 degrees, so its taps never line up with the rows and columns of
// a regular texture.
const TURN = vec2<f32>(0.894, 0.447);

// Grid tap `grid` turned, in units of the tap step.
fn turned(grid: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(dot(grid, vec2<f32>(TURN.x, -TURN.y)), dot(grid, TURN.yx));
}

// The index of grid tap (x, y), each -2..2, in the 5x5 `kinds`.
fn cell(x: i32, y: i32) -> i32 {
    return (y + 2) * 5 + (x + 2);
}

// How much `tap` is of a kind with a pixel of brightness `y0` and tint
// `tint0`: close in tint and brightness - loosely enough that textured lips
// still read as one patch - and as opaque as it is.
fn own_kind(tap: vec4<f32>, y0: f32, tint0: vec3<f32>) -> f32 {
    let ty = luma(tap.rgb);
    let dy = ty - y0;
    let dt = (tap.rgb - vec3<f32>(ty)) - tint0;
    return exp(-(dy * dy) / 0.015 - dot(dt, dt) / 0.0006) * tap.a;
}

// Smoothing. Skin is evened by a blur that gathers only skin: a 5x5 grid of
// taps, each weighted by how much it is skin and by how close it is to the
// pixel - tightly in tint, so lips or a beige wall beside a face do not bleed
// in, and loosely in brightness, so a pore or a small spot is pulled to the
// skin round it, but not across a gap of more than about a fifth, so a brow
// keeps its shape. The pixel's own colour counts for a quarter of a tap. A
// tenth of the fine detail is laid back so skin keeps its texture; rosy lips
// share skin's hues, so their texture is softened like skin's while their
// colour and outline stay.
//
// What keeps its line. A pixel on a thin line - a hair, a brow hair, a
// crease, a lash - gathers only what matches its own brightness. A pixel
// clearly lighter than the skin round it - a shine - gathers only skin of its
// own brightness. Near-black lines, writing and the white or pupil of an eye
// are too dark, too grey or too light to be taken for a blemish.
//
// Blemishes. A pixel is filled from the skin round it - in full from a
// smoothing of 40 - when it is darker or redder than that skin and within a
// quarter of its brightness, warm, not on a line, sits in an even ring of
// plain skin (lips' hues, pale eye whites and other spots of its kind not
// counted), almost nothing near it resembles it, and nothing of its kind
// seven pixels off is joined to it. Joined means of its kind all the way
// there, read at every eighth of the way from a quarter on, and through a
// tap of its kind half way: so lips, their tips, and a patch running on
// along one axis like lips seen from afar are told from a spot, while the
// skin between two spots of a cluster keeps them apart. A filled spot is
// whitened as the skin it now is.
//
// Limits, at 1080 pixels on the frame's shorter side. A spot up to six
// pixels across is found at any setting, alone or in a cluster with skin
// between its spots. Lips four pixels tall may lose their outer rows, and
// lips under four pixels tall may be evened into the skin. A spot more than
// six pixels across, or spots touching, may read as a patch and be left
// largely as it is, or evened only in part with its rim kept round an
// evened middle; a red one kept stands out more once Whiten lifts the skin
// round it. Skin darker than about 0.12 to 0.22 in the display encoding is
// left as it is: dark complexions in ordinary light sit well above that, so
// only deep shadow drops out.
//
// Reach. The grid is turned (see `TURN`). Its reach - ten pixels at 1080,
// whatever the slider - follows the frame's shorter side, so a preview at
// 540 and an export at 2160 are smoothed alike, and an upright 1080 by 1920
// frame as a 1920 by 1080 one; below 432 it stops shrinking. Colours are
// compared straight, not premultiplied, and a tap's alpha weights it.
//
// Whiten. A gamma lift below white, so white stays white and what lies past
// it is kept, with skin calmed a little. A colour with a channel below black
// is too saturated to be skin: whiten passes it by, and on its own it is left
// as shot, though a lone one inside skin is filled like a spot.
//
// Not a number. A pixel that is not a finite number is passed through
// untouched, and read by its neighbours as transparent, so it never
// spreads.
//
// Cost. 41 texture reads for a pixel of skin or of a warm colour, one for any
// other; a pixel that may be a blemish - unlike the skin round it - reads up
// to 104 more to follow its shape.
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = sample(uv);
    if (!finite(c)) {
        return c;
    }
    let s = params.smoothing / 100.0;
    let w = params.whiten / 100.0;
    let hue = signed_hue(c.rgb);
    let mine = skins_at(c.rgb, hue);
    // Only skin or a warm colour that could be a blemish is worked on;
    // the sky, foliage and greys cost one read.
    let warm = smoothstep(-25.0, -15.0, hue) * (1.0 - smoothstep(55.0, 65.0, hue));
    var out = c.rgb;
    var spot = 0.0;
    if (s > 0.0 && (mine.x > 0.0 || (chroma_of(c.rgb) > 0.05 && warm > 0.0))) {
        let scale = max(lines(), 0.4);
        let step = texel() * 3.5 * scale;
        let y0 = luma(c.rgb);
        let tint0 = c.rgb - vec3<f32>(y0);
        let line = on_line(uv, y0, c.a);
        var sum = c.rgb * 0.25;
        var total = 0.25;
        var sum_close = c.rgb * 0.25;
        var total_close = 0.25;
        var fill = vec3<f32>(0.0);
        var ring = 0.0;
        var kinds = array<f32, 25>();
        kinds[cell(0, 0)] = 1.0;
        var ring_y = 0.0;
        var ring_yy = 0.0;
        for (var y: i32 = -2; y <= 2; y++) {
            for (var x: i32 = -2; x <= 2; x++) {
                if (x == 0 && y == 0) {
                    continue;
                }
                // Each tap reads one whole pixel, never a blend of two across an
                // edge, whose in-between colour could pass for either side.
                let tap = pixel((uv + turned(vec2<f32>(f32(x), f32(y))) * step) * frame.size);
                let theirs = skins(tap.rgb) * tap.a;
                let ty = luma(tap.rgb);
                let dy = ty - y0;
                let dt = (tap.rgb - vec3<f32>(ty)) - tint0;
                // The gap allowed in brightness narrows as the tint differs,
                // so a hazel iris, darker and a shade off, is not skin's.
                let off = dot(dt, dt) * 30.0;
                let gap = 1.0 - smoothstep(0.18 - off, 0.26 - off, abs(dy));
                let near = exp(-(dy * dy) / mix(0.2, 0.002, line) - dot(dt, dt) / 0.0006) * gap;
                sum += tap.rgb * near * theirs.x;
                total += near * theirs.x;
                let close = near * exp(-(dy * dy) / 0.003);
                sum_close += tap.rgb * close * theirs.x;
                total_close += close * theirs.x;
                let kind = own_kind(tap, y0, tint0);
                kinds[cell(x, y)] = kind;
                // Plain skin for the ring has some colour to it: the white
                // of an eye, warm but pale, is not skin round an iris.
                let high = max(max(tap.r, tap.g), tap.b);
                // Nor is another spot like this one, beside it in a cluster.
                let plain = theirs.y * smoothstep(0.12, 0.2, chroma_of(tap.rgb) / max(high, 1e-6))
                    * (1.0 - kind);
                fill += tap.rgb * plain;
                ring += plain;
                ring_y += ty * plain;
                ring_yy += ty * ty * plain;
            }
        }
        let around = fill / max(ring, 1e-6);
        // The ring must be even: skin round a spot, not the patchwork of
        // skin, brow and white round an iris.
        let ring_mean = ring_y / max(ring, 1e-6);
        let ring_spread = max(ring_yy / max(ring, 1e-6) - ring_mean * ring_mean, 0.0);
        // A blemish is darker or redder than the skin round it; a shine on
        // a nose or a gold stud, lighter or yellower, is kept.
        let darker = smoothstep(0.0, 0.03, ring_mean - y0);
        let redder = smoothstep(2.0, 8.0, signed_hue(around) - hue);
        // All a blemish needs but its shape.
        let candidate = smoothstep(0.55, 0.85, ring / 24.0)
            * (1.0 - smoothstep(0.004, 0.01, ring_spread))
            * (1.0 - line)
            * smoothstep(0.05, 0.1, chroma_of(c.rgb))
            * warm
            * max(darker, redder)
            * (1.0 - smoothstep(0.2, 0.3, abs(luma(around) - y0)));
        var blemish = 0.0;
        if (candidate > 0.0) {
            blemish = candidate * spot_shape(uv, step, y0, tint0, &kinds);
        }
        // A pixel clearly lighter than the skin round it - a shine on a nose
        // or a cheekbone - gathers only skin of its own brightness, so the
        // shine stays.
        let lighter = smoothstep(0.04, 0.1, y0 - ring_mean);
        let even = mix(sum / total, sum_close / total_close, lighter);
        let goal = mix(c.rgb, even, mine.x);
        let smoothed = goal + (c.rgb - goal) * 0.1;
        // The slider eases in: half way already does seven eighths of
        // what the most does.
        let eased = 1.0 - (1.0 - s) * (1.0 - s) * (1.0 - s);
        out = mix(c.rgb, smoothed, eased);
        // The fill comes in faster than the blur, in full by 40, so the
        // default already takes a spot out.
        let filled = around + (c.rgb - around) * 0.1;
        spot = blemish * smoothstep(0.0, 0.4, s);
        out = mix(out, filled, spot);
    }
    // A spot filled with skin is whitened as the skin it now is, so it
    // does not come back paler than the skin round it.
    let pale = mix(mine.y, skins(out).y, spot);
    if (w > 0.0 && pale > 0.0) {
        let below = clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
        let gamma = vec3<f32>(1.0 + w * 0.25);
        var lifted = lift_gamma_gain(below, vec3<f32>(0.0), gamma, vec3<f32>(1.0));
        lifted = saturation(lifted, 1.0 - w * 0.15) + (out - below);
        out = mix(out, lifted, pale);
    }
    return vec4<f32>(out, c.a);
}

// How much of the pixel's own kind lies `f` of the way to grid tap `grid`.
fn kind_toward(uv: vec2<f32>, step: vec2<f32>, grid: vec2<f32>, f: f32, y0: f32, tint0: vec3<f32>) -> f32 {
    return own_kind(pixel((uv + turned(grid) * step * f) * frame.size), y0, tint0);
}

// How much a candidate blemish is shaped like a spot, from 1 for a spot
// standing alone - or in a cluster with skin between its spots - to 0 for
// a patch: `kinds` holds how much each grid tap is of the pixel's kind.
// A tap counts only if it is joined to the pixel: one beside it (a grid
// step off) through the point half way there; one further off through a tap
// beside the pixel on its way, and along the straight way there, read at
// every eighth from a quarter on. Up to 104 reads.
fn spot_shape(
    uv: vec2<f32>,
    step: vec2<f32>,
    y0: f32,
    tint0: vec3<f32>,
    kinds: ptr<function, array<f32, 25>>,
) -> f32 {
    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            if (x == 0 && y == 0) {
                continue;
            }
            let half_way = kind_toward(uv, step, vec2<f32>(f32(x), f32(y)), 0.5, y0, tint0);
            (*kinds)[cell(x, y)] = min((*kinds)[cell(x, y)], half_way);
        }
    }
    // The pixel's own kind seven or more pixels off (at 1080), joined to
    // it, means a patch too large to be a spot: lips, or the tip of them.
    var far = 0.0;
    var joined = 0.0;
    for (var y: i32 = -2; y <= 2; y++) {
        for (var x: i32 = -2; x <= 2; x++) {
            if (x * x + y * y < 4) {
                continue;
            }
            var via = 0.0;
            for (var vy = floor(f32(y) / 2.0); vy <= ceil(f32(y) / 2.0); vy += 1.0) {
                for (var vx = floor(f32(x) / 2.0); vx <= ceil(f32(x) / 2.0); vx += 1.0) {
                    via = max(via, (*kinds)[cell(i32(vx), i32(vy))]);
                }
            }
            var link = (*kinds)[cell(x, y)] * via;
            // A trace is no join, and costs no reads.
            if (link < 0.01) {
                continue;
            }
            for (var eighth = 2; eighth < 8 && link > 0.0; eighth++) {
                let f = f32(eighth) / 8.0;
                link = min(link, kind_toward(uv, step, vec2<f32>(f32(x), f32(y)), f, y0, tint0));
            }
            far = max(far, link);
            joined += link;
        }
    }
    // How many taps are the pixel's own kind and of a piece with it: a
    // spot has almost none.
    let k = *kinds;
    let alike = joined + k[cell(-1, -1)] + k[cell(0, -1)] + k[cell(1, -1)] + k[cell(-1, 0)]
        + k[cell(1, 0)] + k[cell(-1, 1)] + k[cell(0, 1)] + k[cell(1, 1)];
    // A patch that runs on both ways along one axis but not the other
    // is long, like a pair of lips seen from afar, not round like a spot.
    let kin = vec4<f32>(k[cell(1, 0)], k[cell(-1, 0)], k[cell(0, 1)], k[cell(0, -1)]);
    let along = max(min(kin.x, kin.y), min(kin.z, kin.w));
    let across = min(max(kin.x, kin.y), max(kin.z, kin.w));
    let long = along * (1.0 - across);
    return (1.0 - smoothstep(0.3, 0.6, max(far, long)))
        * (1.0 - smoothstep(0.15, 0.3, alike / 24.0));
}
