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

// The layer's colour at `uv`, between pixels, read at the full-size level
// without derivatives, so a branch on what the pixel holds may read it -
// weighted by its alpha, so what a transparent pixel holds reads as dark
// and never draws a line.
fn level(uv: vec2<f32>) -> vec3<f32> {
    let c = into_space(textureSampleLevel(source, source_sampler, uv, 0.0));
    return c.rgb * c.a;
}

// The layer's colour at whole pixel `at`, clamped to the layer.
fn pixel(at: vec2<f32>) -> vec4<f32> {
    let last = vec2<i32>(frame.size) - vec2<i32>(1);
    return into_space(textureLoad(source, clamp(vec2<i32>(floor(at)), vec2<i32>(0), last), 0));
}

// How much the pixel lies on a thin line - a hair, a brow hair, a crease, a
// lash: along some direction its neighbours a pixel and a half away - at 1080
// lines and below, and as far again in proportion above - match it, while
// across that direction both are clearly lighter. A spot matches nowhere or
// everywhere, so it is never a line.
fn on_line(uv: vec2<f32>, y0: f32) -> f32 {
    let t = texel() * 1.5 * max(frame.size.y / 1080.0, 1.0);
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
        let a = luma(level(uv + along * t)) - y0;
        let b = luma(level(uv - along * t)) - y0;
        let c = luma(level(uv + across * t)) - y0;
        let d = luma(level(uv - across * t)) - y0;
        let same = exp(-(a * a + b * b) / 0.002);
        let lighter = smoothstep(0.05, 0.1, min(c, d));
        line = max(line, same * lighter);
    }
    return line;
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
// plain skin (lips' hues and pale eye whites not counted), almost nothing
// near it resembles it, and nothing of its kind seven pixels off is joined to
// it - which is how lips, their tips, and a patch running on along one axis
// like lips seen from afar are told from a spot. A filled spot is whitened as
// the skin it now is.
//
// Limits, at 1080 lines of the frame's height (so in a vertical 1080 by 1920
// frame nearly twice as many pixels). A spot up to six pixels across is found
// at any setting. Lips four pixels tall may lose their outer rows, and lips
// under four pixels tall may be evened into the skin. A spot more than six
// pixels across, or a close cluster of five-pixel spots, may read as a patch
// and be left largely as it is, or evened only in part with its rim kept
// round an evened middle; a red one kept stands out more once Whiten lifts
// the skin round it. Skin darker than about 0.12 to 0.22 in the display
// encoding is left as it is: dark complexions in ordinary light sit well
// above that, so only deep shadow drops out.
//
// Reach. The grid is turned 27 degrees, so its taps never line up with the
// rows and columns of a regular texture. Its reach - ten pixels at 1080
// lines, whatever the slider - follows the picture's height, so a preview at
// 540 lines and an export at 2160 are smoothed alike; below 432 lines it
// stops shrinking. Colours are compared straight, not premultiplied, and a
// tap's alpha weights it.
//
// Whiten. A gamma lift below white, so white stays white and what lies past
// it is kept, with skin calmed a little. A colour with a channel below black
// is too saturated to be skin: whiten passes it by, and on its own it is left
// as shot, though a lone one inside skin is filled like a spot.
//
// Cost. 41 texture reads for a pixel of skin or of a warm colour, one for any
// other.
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = sample(uv);
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
        let scale = max(frame.size.y / 1080.0, 0.4);
        let step = texel() * 3.5 * scale;
        let y0 = luma(c.rgb);
        let tint0 = c.rgb - vec3<f32>(y0);
        let line = on_line(uv, y0);
        var sum = c.rgb * 0.25;
        var total = 0.25;
        var sum_close = c.rgb * 0.25;
        var total_close = 0.25;
        var fill = vec3<f32>(0.0);
        var ring = 0.0;
        var kinds = array<f32, 25>();
        kinds[12] = 1.0;
        var ring_y = 0.0;
        var ring_yy = 0.0;
        for (var y: i32 = -2; y <= 2; y++) {
            for (var x: i32 = -2; x <= 2; x++) {
                if (x == 0 && y == 0) {
                    continue;
                }
                let grid = vec2<f32>(f32(x), f32(y));
                let turned = vec2<f32>(
                    dot(grid, vec2<f32>(0.894, -0.447)),
                    dot(grid, vec2<f32>(0.447, 0.894)),
                );
                // Each tap reads one whole pixel, never a blend of two across an
                // edge, whose in-between colour could pass for either side.
                let tap = pixel((uv + turned * step) * frame.size);
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
                // How much the tap is the pixel's own kind, close in tint
                // and brightness - loosely enough that textured lips still
                // read as one patch.
                let own_kind = exp(-(dy * dy) / 0.015 - dot(dt, dt) / 0.0006) * tap.a;
                kinds[(y + 2) * 5 + (x + 2)] = own_kind;
                // Plain skin for the ring has some colour to it: the white
                // of an eye, warm but pale, is not skin round an iris.
                let high = max(max(tap.r, tap.g), tap.b);
                // Nor is another spot like this one, beside it in a cluster.
                let plain = theirs.y * smoothstep(0.12, 0.2, chroma_of(tap.rgb) / max(high, 1e-6))
                    * (1.0 - own_kind);
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
        // The pixel's own kind seven or more pixels off (at 1080 lines),
        // joined to it through a tap of its kind on the way, means a patch
        // too large to be a spot: lips, or the tip of them. Small spots side
        // by side in a cluster have skin between them and are not joined;
        // spots five pixels across with centres under about ten pixels
        // apart can chain across that skin and read as one patch.
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
                        via = max(via, kinds[(i32(vy) + 2) * 5 + (i32(vx) + 2)]);
                    }
                }
                far = max(far, kinds[(y + 2) * 5 + (x + 2)] * via);
                joined += kinds[(y + 2) * 5 + (x + 2)] * via;
            }
        }
        // How many taps are the pixel's own kind and of a piece with it: a
        // spot has almost none, while another spot beside it in a cluster
        // is not of a piece with it and does not count.
        let alike = joined + kinds[6] + kinds[7] + kinds[8] + kinds[11] + kinds[13]
            + kinds[16] + kinds[17] + kinds[18];
        // A patch that runs on both ways along one axis but not the other
        // is long, like a pair of lips seen from afar, not round like a spot.
        let kin = vec4<f32>(kinds[13], kinds[11], kinds[17], kinds[7]);
        let along = max(min(kin.x, kin.y), min(kin.z, kin.w));
        let across = min(max(kin.x, kin.y), max(kin.z, kin.w));
        let long = along * (1.0 - across);
        // A blemish is darker or redder than the skin round it; a shine on
        // a nose or a gold stud, lighter or yellower, is kept.
        let darker = smoothstep(0.0, 0.03, ring_mean - y0);
        let redder = smoothstep(2.0, 8.0, signed_hue(around) - hue);
        let blemish = smoothstep(0.55, 0.85, ring / 24.0)
            * (1.0 - smoothstep(0.004, 0.01, ring_spread))
            * (1.0 - line)
            * (1.0 - smoothstep(0.3, 0.6, max(far, long)))
            * (1.0 - smoothstep(0.15, 0.3, alike / 24.0))
            * smoothstep(0.05, 0.1, chroma_of(c.rgb))
            * warm
            * max(darker, redder)
            * (1.0 - smoothstep(0.2, 0.3, abs(luma(around) - y0)));
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
