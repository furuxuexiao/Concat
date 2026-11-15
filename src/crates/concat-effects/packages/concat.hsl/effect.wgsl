struct Params {
    red_hue: f32, red_sat: f32, red_lum: f32,
    orange_hue: f32, orange_sat: f32, orange_lum: f32,
    yellow_hue: f32, yellow_sat: f32, yellow_lum: f32,
    green_hue: f32, green_sat: f32, green_lum: f32,
    cyan_hue: f32, cyan_sat: f32, cyan_lum: f32,
    blue_hue: f32, blue_sat: f32, blue_lum: f32,
    purple_hue: f32, purple_sat: f32, purple_lum: f32,
    magenta_hue: f32, magenta_sat: f32, magenta_lum: f32,
}

// The band centres, in degrees round the HSV wheel: red, orange, yellow,
// green, cyan, blue, purple, magenta - the eight a grading panel shows.
const CENTRES = array<f32, 8>(0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0);

// A colour from HSV: hue in degrees, saturation 0..1, and value.
fn from_hsv(hsv: vec3<f32>) -> vec3<f32> {
    let h = fract(hsv.x / 360.0);
    let p = abs(fract(vec3<f32>(h) + vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3<f32>(3.0));
    let ramp = clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
    return hsv.z * mix(vec3<f32>(1.0), ramp, hsv.y);
}

// Whether every channel of `c` is a number and finite.
fn finite(c: vec4<f32>) -> bool {
    let exponent = bitcast<vec4<u32>>(c) & vec4<u32>(0x7f800000u);
    return all(exponent != vec4<u32>(0x7f800000u));
}

// Each band's sliders, weighted between the two centres a hue sits
// between: all of one band at its centre, none at its neighbours', so the
// weights always sum to one and no hue is moved by more than two bands.
// Saturation and luminance hand over along a smoothstep. Hue hands over
// in a straight line, and two neighbours - the one before turned past the
// one after far enough to fold a colour past the one beside it - are scaled
// back together until they no longer can: the wheel is only ever stretched or squeezed,
// never reversed. The move fades out below a chroma of 0.15 - absolute, not
// relative to brightness, so the near-black noise of a shadow is left alone
// too, and a muted colour such as dim foliage or a shadowed sky takes only
// part of the move - and a grey stays grey.
//
// Everything here is read in the display encoding (the package's `space`),
// the picture as it is seen, not linear light: hue, saturation, chroma and
// luminance are those of the encoded colour, which is what a grading panel
// shows.
//
// A pixel that is not a finite number is passed through untouched.
fn effect(uv: vec2<f32>) -> vec4<f32> {
    let c = sample(uv);
    if (!finite(c)) {
        return c;
    }
    let rgb = max(c.rgb, vec3<f32>(0.0));
    let value = max(max(rgb.r, rgb.g), rgb.b);
    let chroma = chroma_of(rgb);
    let hsv = vec3<f32>(hue_of(rgb), select(0.0, chroma / value, value > 1e-6), value);
    var hues = array<f32, 8>(
        params.red_hue, params.orange_hue, params.yellow_hue, params.green_hue,
        params.cyan_hue, params.blue_hue, params.purple_hue, params.magenta_hue,
    );
    var sats = array<f32, 8>(
        params.red_sat, params.orange_sat, params.yellow_sat, params.green_sat,
        params.cyan_sat, params.blue_sat, params.purple_sat, params.magenta_sat,
    );
    var lums = array<f32, 8>(
        params.red_lum, params.orange_lum, params.yellow_lum, params.green_lum,
        params.cyan_lum, params.blue_lum, params.purple_lum, params.magenta_lum,
    );
    var centres = CENTRES;
    var i = 7;
    for (var n = 0; n < 7; n++) {
        if (hsv.x >= centres[n] && hsv.x < centres[n + 1]) {
            i = n;
        }
    }
    let j = (i + 1) % 8;
    var span = centres[j] - centres[i];
    var along = hsv.x - centres[i];
    if (j == 0) {
        span += 360.0;
        along = fract(along / 360.0) * 360.0;
    }
    let linear = along / span;
    let t = smoothstep(0.0, 1.0, linear);
    let gate = smoothstep(0.0, 0.15, chroma);
    // Up to 30 degrees each way; plus turns towards the next band round
    // the wheel (red to orange, blue to purple), minus towards the one
    // before. Between two neighbours the turn hands over in a straight
    // line, so the colours between them fold back only if the band before
    // is turned further forward than the band after by more than the gap
    // between their centres. Where it is by more than 0.8 of the gap, both
    // are eased back by the same share, so that pair comes back within the
    // gap - at most about 0.87 of it when three bands are turned in a chain,
    // 0.8 for a pair alone. So a band 30 degrees from its neighbour turns at
    // most 24 towards it while that neighbour rests, and the full 30 away
    // from it; two neighbours turned apart, stretching the colours between
    // them, are never eased. A hue between them then still moves forward as
    // its input does, and each share is the band's own, so the turn is
    // continuous across every centre.
    var turns = array<f32, 8>();
    for (var n = 0; n < 8; n++) {
        turns[n] = hues[n] * 0.3;
    }
    var shares = array<f32, 8>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    for (var n = 0; n < 8; n++) {
        let m = (n + 1) % 8;
        let gap = select(centres[m], centres[m] + 360.0, m == 0) - centres[n];
        // How far the band before is turned past the band after: what
        // squeezes the colours between them, and past the gap folds them.
        let squeeze = turns[n] - turns[m];
        if (squeeze > gap * 0.8) {
            let share = gap * 0.8 / squeeze;
            shares[n] = min(shares[n], share);
            shares[m] = min(shares[m], share);
        }
    }
    let head = turns[i] * shares[i];
    let tail = turns[j] * shares[j];
    let hue = mix(head, tail, linear) * gate;
    let sat = mix(sats[i], sats[j], t) * gate;
    let lum = mix(lums[i], lums[j], t) * gate;
    if (hue == 0.0 && sat == 0.0 && lum == 0.0) {
        return c;
    }
    // Hue turned in HSV, so the colour keeps its saturation and value.
    let turned = from_hsv(vec3<f32>(hsv.x + hue, hsv.y, hsv.z));
    // Saturation pushes the colour from or towards its own luminance, so a
    // band taken to grey keeps its brightness and a push never changes it.
    // A push up eases into the most it may take - the push that would bring
    // a channel to black, or the brightest to white (or to where it already
    // is, past white) - so richness tapers off smoothly near black and white,
    // and a colour at white or past it is left as it is. A near-pure colour,
    // a channel already near black, gains little.
    let y = luma(turned);
    let low = min(min(turned.r, turned.g), turned.b);
    let high = max(max(turned.r, turned.g), turned.b);
    let wanted = 1.0 + sat / 100.0;
    var push = wanted;
    if (wanted > 1.0) {
        var most = 1e6;
        if (y > low) {
            most = min(most, y / (y - low));
        }
        if (high > y) {
            most = min(most, (max(high, 1.0) - y) / (high - y));
            // Near white the room is also held to the headroom left, so a
            // pale colour - its brightest channel close to its luminance -
            // tapers off mostly over the last quarter of brightness rather
            // than in the last few hundredths.
            most = min(most, 1.0 + max(1.0 - high, 0.0) / 0.25);
        }
        let room = max(most - 1.0, 0.0);
        // Close to what the slider asks while there is room, easing into
        // the most only as the two meet.
        let d = wanted - 1.0;
        let r = d / max(room, 1e-4);
        push = 1.0 + d / pow(1.0 + r * r * r, 1.0 / 3.0);
    }
    var out = vec3<f32>(y) + (turned - vec3<f32>(y)) * push;
    // Luminance moves brightness - the luminance of the colour: down
    // scales the colour, to 0.6 at -100; up aims 0.4 of the way from its
    // luminance to white at 100. The colour is scaled towards that aim,
    // keeping its hue and saturation, until its brightest channel reaches
    // white; only what is left of the aim is then made up by blending
    // towards white, so a colour pales only when it has no room left.
    if (lum < 0.0) {
        out *= 1.0 + lum * 0.004;
    } else if (lum > 0.0) {
        let before = luma(out);
        let high_now = max(max(out.r, out.g), out.b);
        if (before > 1e-6 && before < 1.0) {
            let aim = before + (1.0 - before) * lum * 0.004;
            let room = select(aim / before, 1.0 / high_now, high_now * aim / before > 1.0);
            let lifted = out * max(room, 1.0);
            let reached = luma(lifted);
            // The rest of the aim, by blending towards white - or towards
            // the colour itself where a channel is already past white, so
            // no channel is ever lowered - by just the share that makes it
            // up, which falls to nothing as the colour nears white.
            let ceiling = max(lifted, vec3<f32>(1.0));
            let span = luma(ceiling) - reached;
            let rest = select(0.0, (aim - reached) / span, reached < aim && span > 1e-6);
            out = mix(lifted, ceiling, clamp(rest, 0.0, 1.0));
        }
    }
    // The work above is on the colour within gamut; whatever lay below
    // black is laid back on, so a slider barely moved barely moves it.
    return vec4<f32>(max(out, vec3<f32>(0.0)) + min(c.rgb, vec3<f32>(0.0)), c.a);
}
