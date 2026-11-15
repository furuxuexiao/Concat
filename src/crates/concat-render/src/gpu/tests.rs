// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! The GPU compositor against the CPU reference: the same plans, and the
//! pictures must agree. Every test skips on a machine without an adapter,
//! and says so.

use std::collections::BTreeMap;
use std::sync::Arc;

use concat_core::frame::Frame;
use concat_core::shader::{Lut, RevealMap, ShaderPass};
use concat_core::timeline::{Blend, Transform};

use super::*;
use crate::CpuCompositor;
use crate::metrics::ssim;
use crate::plan::{Backdrop, Crop, Transition, Warp, detached_clip};

fn gpu() -> Option<WgpuCompositor> {
    let compositor = WgpuCompositor::new();
    if compositor.is_none() {
        // CI installs a software Vulkan driver so this suite runs there; a
        // machine that says it must run and has no adapter is a broken
        // setup, not a skip.
        assert!(
            std::env::var_os("CONCAT_REQUIRE_GPU").is_none(),
            "CONCAT_REQUIRE_GPU is set and no GPU adapter is usable"
        );
        eprintln!("no usable GPU adapter; skipping");
    }
    compositor
}

/// [`gpu`], for the suites that draw every package or a wide blur: skipped
/// on a software adapter unless `CONCAT_REQUIRE_GPU` is set, as the effect
/// cards are (concat-host's cards.rs). On Windows' WARP each of them ran
/// past five minutes, every compile going through FXC and every pixel
/// through the CPU, and never finished. Linux's lavapipe runs them under
/// `CONCAT_REQUIRE_GPU`, and a real GPU always does.
fn heavy_gpu() -> Option<WgpuCompositor> {
    let gpu = gpu()?;
    if gpu.on_software_adapter() && std::env::var_os("CONCAT_REQUIRE_GPU").is_none() {
        eprintln!("a software adapter only, and CONCAT_REQUIRE_GPU is not set; skipping");
        return None;
    }
    Some(gpu)
}

fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Frame {
    let mut frame = Frame::transparent(width, height);
    frame.fill(rgba);
    frame
}

/// A picture with something in it everywhere: a diagonal gradient and a
/// dark bar, so a misplaced or mirrored picture is caught.
fn gradient(width: u32, height: u32) -> Frame {
    let mut frame = Frame::transparent(width, height);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width.max(1)) as u8;
            let g = (y * 255 / height.max(1)) as u8;
            let bar = x > width / 3 && x < width / 2 && y > height / 4;
            let b = if bar { 20 } else { 200 };
            frame.set_pixel(x, y, [r, g, b, 255]);
        }
    }
    frame
}

fn layer(frame: Frame) -> PlannedLayer {
    PlannedLayer::picture(detached_clip(), Arc::new(frame))
}

fn plan(width: u32, height: u32, layers: Vec<PlannedLayer>) -> FramePlan {
    FramePlan {
        layers,
        ..FramePlan::empty(width, height)
    }
}

/// A package's pass: `id` with `body` as its shader and `params` as the
/// manifest's parameter table, resolved to `values`, drawn on the display
/// level - the picture a look written by eye reads.
fn package(
    id: &str,
    body: &str,
    params: &str,
    values: &[(&str, f64)],
    intensity: f32,
) -> ShaderPass {
    let manifest = concat_effects::Manifest::parse(&format!(
        "format = 2\n[effect]\nid = \"{id}\"\nname = \"Test\"\nkind = \"effect\"\n{params}\n[wgsl]\nentry = \"effect.wgsl\"\nspace = \"display\"\n"
    ))
    .expect("a manifest");
    let shader = concat_effects::Shader::compile(&manifest, body).expect("compiles");
    let values: BTreeMap<String, f64> = values
        .iter()
        .map(|(key, value)| ((*key).to_owned(), *value))
        .collect();
    shader.pass(&values, &manifest.params, intensity, None, None)
}

const INVERT: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(vec3<f32>(1.0) - c.rgb, c.a); }";

/// Both compositors draw `plan`; the pictures must agree to `least` by
/// SSIM, and every pixel's alpha must be opaque.
fn assert_parity(name: &str, plan: &FramePlan, least: f64) {
    let Some(mut gpu) = gpu() else { return };
    let expected = CpuCompositor.render(plan);
    let actual = gpu.render(plan);
    let score = ssim(&expected, &actual);
    eprintln!("parity {name}: ssim {score:.4}");
    assert!(
        score >= least,
        "{name}: cpu and gpu differ, ssim {score:.4} < {least}"
    );
    assert!(
        actual.pixels().chunks_exact(4).all(|px| px[3] == 255),
        "{name}: opaque"
    );
}

#[test]
fn empty_output_is_opaque_black() {
    let Some(mut gpu) = gpu() else { return };
    let frame = gpu.render(&FramePlan::empty(4, 4));
    assert_eq!(frame.pixel(0, 0), Some([0, 0, 0, 255]));
    assert_eq!(frame.pixel(3, 3), Some([0, 0, 0, 255]));
}

#[test]
fn plain_layers_match_the_cpu_reference_to_the_pixel() {
    let Some(mut gpu) = gpu() else { return };
    let mut top = layer(solid(3, 3, [0, 0, 255, 160]));
    top.transform = Transform {
        offset_x: 2.0 / 8.0,
        offset_y: 1.0 / 8.0,
        ..Transform::default()
    };
    let plan = plan(8, 8, vec![layer(solid(8, 8, [255, 0, 0, 255])), top]);
    let expected = CpuCompositor.render(&plan);
    let actual = gpu.render(&plan);
    for (index, (want, got)) in expected
        .pixels()
        .chunks_exact(4)
        .zip(actual.pixels().chunks_exact(4))
        .enumerate()
    {
        for channel in 0..4 {
            let difference = (i32::from(want[channel]) - i32::from(got[channel])).abs();
            assert!(
                difference <= 1,
                "pixel {index} channel {channel}: cpu {want:?} vs gpu {got:?}"
            );
        }
    }
}

/// A stabilized frame is its source resampled through the correction: a
/// warp of a whole number of pixels draws each pixel from that far over,
/// to the level, and the edge it pulls into view holds the source's last
/// column, as the sampler reads past an edge.
#[test]
fn a_stabilized_frame_is_the_source_through_its_warp() {
    let Some(mut gpu) = gpu() else { return };
    let (w, h) = (64u32, 48u32);
    let source = gradient(w, h);
    let mut shifted = layer(source.clone());
    // Each point drawn is read 6 px right of and 3 px below itself.
    shifted.warp = Warp {
        a: 1.0,
        b: 0.0,
        tx: 6.0 / h as f32,
        ty: 3.0 / h as f32,
    };
    let out = gpu.render(&plan(w, h, vec![shifted]));
    for y in 0..h {
        for x in 0..w {
            let want = source
                .pixel((x + 6).min(w - 1), (y + 3).min(h - 1))
                .expect("inside");
            let got = out.pixel(x, y).expect("inside");
            for channel in 0..3 {
                let difference = (i32::from(want[channel]) - i32::from(got[channel])).abs();
                assert!(
                    difference <= 1,
                    "({x}, {y}) channel {channel}: want {want:?}, got {got:?}"
                );
            }
        }
    }

    // A turn and a zoom about the centre: a point of the picture reads
    // from where the turn and the zoom send it. Probed on the smooth
    // gradient, away from the bar, whose edge the two blend differently
    // (the GPU in light, the reference in its stored levels).
    let (angle, zoom) = (8f32.to_radians(), 1.0 / 1.15);
    let mut turned = layer(source.clone());
    turned.warp = Warp {
        a: zoom * angle.cos(),
        b: zoom * angle.sin(),
        tx: 0.0,
        ty: 0.0,
    };
    let out = gpu.render(&plan(w, h, vec![turned.clone()]));
    let probe = |x: f32, y: f32| {
        let (nx, ny) = (
            (x - w as f32 / 2.0) / h as f32,
            (y - h as f32 / 2.0) / h as f32,
        );
        let (mx, my) = (
            turned.warp.a * nx - turned.warp.b * ny,
            turned.warp.b * nx + turned.warp.a * ny,
        );
        (
            mx * h as f32 + w as f32 / 2.0,
            my * h as f32 + h as f32 / 2.0,
        )
    };
    for (x, y) in [(52u32, 10u32), (10, 36), (48, 40), (40, 6)] {
        let (sx, sy) = probe(x as f32 + 0.5, y as f32 + 0.5);
        let want = crate::reference::sample_bilinear(&source, sx - 0.5, sy - 0.5);
        let got = out.pixel(x, y).expect("inside");
        for channel in 0..3 {
            assert!(
                (want[channel] - f32::from(got[channel])).abs() <= 3.0,
                "({x}, {y}) channel {channel}: want {want:?}, got {got:?}"
            );
        }
    }

    // And the CPU reference agrees, alone, cropped, and made first for an
    // effect to run over.
    assert_parity("stabilized", &plan(w, h, vec![turned.clone()]), 0.99);
    let mut cropped = turned.clone();
    cropped.crop = Crop::of([0.1, 0.0, 0.2, 0.1]);
    cropped.flip_h = true;
    assert_parity(
        "stabilized and cropped",
        &plan(w, h, vec![cropped.clone()]),
        0.99,
    );
    let mut prepared = cropped;
    prepared.effects = vec![package("test.invert", INVERT, "", &[], 1.0)];
    let geometry = prepared.geometry(&source, w, h);
    assert!(prepared.needs_preparing(&geometry));
    let inverted = gpu.render(&plan(w, h, vec![prepared]));
    let mut plain = turned;
    plain.crop = Crop::of([0.1, 0.0, 0.2, 0.1]);
    plain.flip_h = true;
    let straight = gpu.render(&plan(w, h, vec![plain]));
    // The invert ran over the warped picture: inside the fitted picture,
    // each pixel is the other's inverse.
    let middle = (w / 2, h / 2);
    let (a, b) = (
        inverted.pixel(middle.0, middle.1).expect("inside"),
        straight.pixel(middle.0, middle.1).expect("inside"),
    );
    for channel in 0..3 {
        assert!(
            (i32::from(a[channel]) + i32::from(b[channel]) - 255).abs() <= 2,
            "{a:?} vs {b:?}"
        );
    }
}

/// A pass runs over the layer before it is placed: an invert shader
/// over a red frame composites cyan, and at half intensity the mix, in
/// light - half of each channel, which BT.1886's 2.4 gamma stores at 191.
#[test]
fn a_pass_treats_the_layer_before_it_is_placed() {
    let Some(mut gpu) = gpu() else { return };
    let red = solid(4, 4, [255, 0, 0, 255]);
    let mut full = layer(red.clone());
    full.effects = vec![package("test.invert", INVERT, "", &[], 1.0)];
    let out = gpu.render(&plan(4, 4, vec![full]));
    assert_eq!(&out.pixels()[..3], &[0, 255, 255]);
    let mut half = layer(red);
    half.effects = vec![package("test.invert", INVERT, "", &[], 0.5)];
    let out = gpu.render(&plan(4, 4, vec![half]));
    let p = &out.pixels()[..3];
    assert!(
        p.iter().all(|channel| (189..=193).contains(channel)),
        "{p:?}"
    );
}

/// A pass reads its table through `lut()`: a table that answers green
/// to every colour turns a red frame green, and a pass without one is
/// handed the identity and changes nothing.
#[test]
fn a_pass_samples_its_table_and_the_identity_without_one() {
    let Some(mut gpu) = gpu() else { return };
    let body = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(lut(c.rgb), c.a); }";
    let green = Lut::from_rgb(2, &[0.0, 1.0, 0.0].repeat(8)).expect("a table");
    let mut tabled = layer(solid(4, 4, [255, 0, 0, 255]));
    let mut pass = package("test.table", body, "", &[], 1.0);
    pass.lut = Some(Arc::new(green));
    tabled.effects = vec![pass];
    let out = gpu.render(&plan(4, 4, vec![tabled]));
    assert_eq!(&out.pixels()[..3], &[0, 255, 0]);
    let mut plain = layer(solid(4, 4, [255, 0, 0, 255]));
    plain.effects = vec![package("test.table", body, "", &[], 1.0)];
    let out = gpu.render(&plan(4, 4, vec![plain]));
    assert_eq!(&out.pixels()[..3], &[255, 0, 0]);
}

/// A treatment runs its passes over the stack beneath its track and
/// nothing above it, blended back by its strength, without the frame
/// leaving the GPU: an invert over a red ground under a blue quarter
/// turns the ground cyan and leaves the blue alone.
#[test]
fn a_treatment_treats_the_stack_beneath_its_track_only() {
    let Some(mut gpu) = gpu() else { return };
    let ground = layer(solid(8, 8, [255, 0, 0, 255]));
    let mut blue = layer(solid(4, 4, [0, 0, 255, 255]));
    blue.track = 2;
    blue.transform = Transform {
        offset_x: -2.0 / 8.0,
        offset_y: -2.0 / 8.0,
        ..Transform::default()
    };
    let treated = |strength: f32| FramePlan {
        treatments: vec![PlannedTreatment {
            track: 1,
            effects: vec![package("concat.invert", INVERT, "", &[], 1.0)],
            strength,
        }],
        ..plan(8, 8, vec![ground.clone(), blue.clone()])
    };
    let out = gpu.render(&treated(1.0));
    // Top-left is under the blue quarter; bottom-right is treated ground.
    assert_eq!(&out.pixels()[..3], &[0, 0, 255]);
    let last = out.pixels().len() - 4;
    assert_eq!(&out.pixels()[last..last + 3], &[0, 255, 255]);
    // At half strength the ground is halfway between red and cyan in light:
    // a half of each channel, which BT.1886's 2.4 gamma stores at 191.
    let out = gpu.render(&treated(0.5));
    let p = &out.pixels()[last..last + 3];
    assert!(
        p.iter().all(|channel| (189..=193).contains(channel)),
        "{p:?}"
    );
}

/// The parity suite: one plan per thing a frame is made of besides its
/// effects, drawn by both compositors, alike by SSIM.
#[test]
fn every_kind_of_layer_matches_the_cpu_reference() {
    assert_parity("plain", &plan(64, 48, vec![layer(gradient(64, 48))]), 0.999);

    // Fitted: a wide picture inside a square, letterboxed.
    assert_parity("fitted", &plan(64, 64, vec![layer(gradient(48, 16))]), 0.99);

    let mut placed = layer(gradient(32, 24));
    placed.transform = Transform {
        scale: 1.4,
        rotation: 30.0,
        offset_x: 0.1,
        offset_y: -0.05,
        stretch_x: 1.2,
        ..Transform::default()
    };
    assert_parity(
        "placed",
        &plan(
            64,
            64,
            vec![layer(solid(64, 64, [40, 40, 40, 255])), placed],
        ),
        0.98,
    );

    let mut cropped = layer(gradient(64, 48));
    cropped.crop = Crop::of([0.25, 0.1, 0.2, 0.3]);
    assert_parity("cropped", &plan(64, 48, vec![cropped]), 0.99);

    let mut flipped = layer(gradient(64, 48));
    flipped.flip_h = true;
    flipped.flip_v = true;
    assert_parity("flipped", &plan(64, 48, vec![flipped]), 0.999);

    let mut blended = layer(gradient(64, 48));
    blended.opacity = 0.5;
    blended.blend = Blend::Multiply;
    assert_parity(
        "blended",
        &plan(64, 48, vec![layer(gradient(64, 48)), blended]),
        0.99,
    );

    // Lighten and Darken at partial opacity: the two blends the GPU draws
    // over a copy of the ground, held to the CPU's own line.
    for (name, blend) in [("lightened", Blend::Lighten), ("darkened", Blend::Darken)] {
        let mut over = layer(solid(64, 48, [200, 60, 140, 255]));
        over.opacity = 0.3;
        over.blend = blend;
        assert_parity(
            name,
            &plan(64, 48, vec![layer(gradient(64, 48)), over]),
            0.99,
        );
    }

    let mut masked = layer(gradient(64, 64));
    let mut mask = Frame::transparent(32, 32);
    for y in 0..32u32 {
        for x in 0..32u32 {
            let d = ((x as f32 - 16.0).powi(2) + (y as f32 - 16.0).powi(2)).sqrt();
            let a = ((16.0 - d) / 8.0).clamp(0.0, 1.0);
            mask.set_pixel(x, y, [255, 255, 255, (a * 255.0) as u8]);
        }
    }
    masked.mask = Some(Arc::new(mask));
    assert_parity(
        "masked",
        &plan(64, 64, vec![layer(solid(64, 64, [0, 60, 0, 255])), masked]),
        0.99,
    );

    let mut cut = layer(gradient(64, 48));
    cut.transitions = vec![
        Transition::FadeTo {
            colour: [0.0, 0.0, 0.0],
            amount: 0.5,
        },
        Transition::Wipe {
            uncovered: 0.6,
            from_right: false,
        },
    ];
    assert_parity("faded and wiped", &plan(64, 48, vec![cut]), 0.99);
}

/// A feathered circle on a white picture over black, read in the working
/// space where white is one: the light is the coverage. Whole in the
/// middle, half on the edge, nothing past the band - and the same number
/// the CPU's definition gives at each pixel's centre.
#[test]
fn a_feathered_circle_keeps_its_middle_and_fades_across_its_edge() {
    use concat_core::mask::{MaskShape, ShapeMask};
    let Some(mut gpu) = gpu() else { return };
    let mut circle = layer(solid(100, 100, [255, 255, 255, 255]));
    let shape = ShapeMask {
        shape: MaskShape::Circle,
        width: 0.6,
        height: 0.6,
        feather: 0.4,
        ..ShapeMask::default()
    };
    circle.shape = Some(shape);
    let drawn = working(&mut gpu, &plan(100, 100, vec![circle]));
    let at = |x: usize, y: usize| drawn[y * 100 + x][0];
    let expected = |x: usize, y: usize| {
        shape.coverage(
            (x as f32 + 0.5) / 100.0,
            (y as f32 + 0.5) / 100.0,
            100.0,
            100.0,
            0.01,
        )
    };
    // The middle, a band inside the edge, the edge, and past the band.
    assert!((at(50, 50) - 1.0).abs() < 0.01, "middle {}", at(50, 50));
    assert!(
        (at(50 + 19, 50) - 1.0).abs() < 0.01,
        "inside {}",
        at(69, 50)
    );
    assert!((at(80, 50) - 0.5).abs() < 0.06, "edge {}", at(80, 50));
    assert!((at(50, 80) - 0.5).abs() < 0.06, "edge {}", at(50, 80));
    assert!(at(50 + 41, 50) < 0.01, "outside {}", at(91, 50));
    assert!(at(2, 2) < 0.001, "corner {}", at(2, 2));
    for (x, y) in [(75, 50), (80, 50), (85, 50), (50, 72), (64, 64), (71, 71)] {
        let (got, want) = (at(x, y), expected(x, y));
        assert!(
            (got - want).abs() < 0.01,
            "({x}, {y}): gpu {got}, cpu {want}"
        );
    }
}

/// Every shape against the CPU reference: turned, moved, feathered and
/// inverted, on a picture that is itself placed, cropped and carrying a
/// cutout's matte - the masks multiply.
#[test]
fn shape_masks_match_the_cpu_reference() {
    use concat_core::mask::{MaskShape, ShapeMask};
    for shape in MaskShape::ALL {
        for invert in [false, true] {
            let mut masked = layer(gradient(64, 48));
            masked.transform = Transform {
                scale: 0.9,
                rotation: 12.0,
                offset_x: 0.05,
                ..Transform::default()
            };
            masked.crop = Crop::of([0.1, 0.0, 0.05, 0.1]);
            // A stabilized clip's correction: the mask is on the picture
            // as corrected, so it holds still with it.
            masked.warp = Warp {
                a: 1.06,
                b: 0.03,
                tx: 0.02,
                ty: -0.01,
            };
            masked.shape = Some(ShapeMask {
                shape,
                x: 0.05,
                y: -0.04,
                width: 0.7,
                height: 0.5,
                rotation: 20.0,
                roundness: 0.5,
                feather: 0.15,
                invert,
            });
            let mut matte = Frame::transparent(16, 16);
            for y in 0..16u32 {
                for x in 0..16u32 {
                    let a = if x < 3 { 0 } else { 255 };
                    matte.set_pixel(x, y, [255, 255, 255, a]);
                }
            }
            masked.mask = Some(Arc::new(matte));
            assert_parity(
                &format!("{shape:?} mask, inverted {invert}"),
                &plan(64, 48, vec![layer(solid(64, 48, [0, 60, 0, 255])), masked]),
                0.98,
            );
        }
    }
}

/// A masked picture never counts as covering the frame, and its canvas
/// background is drawn whole behind it - the mask cuts the picture, not
/// what stands behind it.
#[test]
fn a_masked_pictures_background_is_whole() {
    use concat_core::mask::ShapeMask;
    let mut masked = layer(solid(64, 48, [200, 0, 0, 255]));
    assert!(masked.covers(64, 48, true));
    masked.shape = Some(ShapeMask::default());
    assert!(!masked.covers(64, 48, true));
    let behind = masked
        .backdrop(&Backdrop::Colour([0, 0, 255]), 64, 48, |_| None)
        .expect("a colour always draws");
    assert!(behind.shape.is_none());
    let Some(mut gpu) = gpu() else { return };
    let frame = gpu.render(&plan(64, 48, vec![behind, masked]));
    let corner = frame.pixel(1, 1).expect("a pixel");
    let middle = frame.pixel(32, 24).expect("a pixel");
    assert!(corner[2] > 200 && corner[0] < 30, "blue behind: {corner:?}");
    assert!(middle[0] > 180 && middle[2] < 30, "red kept: {middle:?}");
}

/// What cannot be drawn is not drawn, the same way on both sides: a
/// layer with a NaN opacity, one far outside the frame, one with no
/// picture, and an opacity past one is one.
#[test]
fn what_cannot_be_drawn_is_skipped_alike() {
    let Some(mut gpu) = gpu() else { return };
    let mut nan = layer(solid(8, 8, [255, 0, 0, 255]));
    nan.opacity = f32::NAN;
    let mut away = layer(solid(8, 8, [255, 0, 0, 255]));
    away.transform = Transform {
        offset_x: 40.0,
        ..Transform::default()
    };
    let mut none = layer(solid(8, 8, [255, 0, 0, 255]));
    none.source = None;
    let plan = plan(8, 8, vec![nan, away, none]);
    assert_eq!(gpu.render(&plan).pixel(4, 4), Some([0, 0, 0, 255]));
    assert_eq!(
        CpuCompositor.render(&plan).pixel(4, 4),
        Some([0, 0, 0, 255])
    );
    let mut over = layer(solid(8, 8, [0, 200, 0, 255]));
    over.opacity = 7.0;
    let plan = FramePlan {
        layers: vec![over],
        ..FramePlan::empty(8, 8)
    };
    assert_eq!(gpu.render(&plan).pixel(1, 1), Some([0, 200, 0, 255]));
}

/// A package that runs within the budget passes its trial; a device that
/// answers in time is kept.
#[test]
fn a_benign_package_survives_its_trial() {
    let Some(mut gpu) = gpu() else { return };
    let pass = package("test.trial", INVERT, "", &[], 1.0);
    gpu.trial(&pass, std::time::Duration::from_secs(5))
        .expect("an invert is quick");
    assert!(!gpu.is_dead());
    // The compositor is still good for a frame afterwards.
    let frame = gpu.render(&plan(4, 4, vec![layer(solid(4, 4, [0, 0, 255, 255]))]));
    assert_eq!(&frame.pixels()[..3], &[0, 0, 255]);
}

/// A module the driver refuses - here, one that is not WGSL at all, which
/// only reaches the device because the pass was built by hand rather than
/// by the catalogue - is caught in its error scope: the layer draws
/// untreated, the device is not dead, and the trial says no.
#[test]
fn a_pass_the_driver_refuses_is_skipped_and_fails_its_trial() {
    let Some(mut gpu) = gpu() else { return };
    let mut broken = package("test.broken", INVERT, "", &[], 1.0);
    broken.key = "test.broken@1#garbage".to_owned();
    broken.source = Arc::from("this is not a shader");
    let mut over = layer(solid(4, 4, [0, 200, 0, 255]));
    over.effects = vec![broken.clone()];
    let frame = gpu.render(&plan(4, 4, vec![over]));
    assert_eq!(frame.pixel(1, 1), Some([0, 200, 0, 255]), "drawn untreated");
    assert!(!gpu.is_dead());
    assert!(
        gpu.trial_at(&broken, 64, std::time::Duration::from_secs(5))
            .is_err()
    );
    // And a good pass still runs on the same compositor.
    let mut over = layer(solid(4, 4, [0, 200, 0, 255]));
    over.effects = vec![package("test.trial", INVERT, "", &[], 1.0)];
    let frame = gpu.render(&plan(4, 4, vec![over]));
    assert_eq!(frame.pixel(1, 1), Some([255, 55, 255, 255]));
}

#[test]
fn output_size_changes_are_handled() {
    let Some(mut gpu) = gpu() else { return };
    let small = gpu.render(&plan(4, 4, vec![layer(solid(4, 4, [255, 255, 255, 255]))]));
    assert_eq!((small.width(), small.height()), (4, 4));
    let large = gpu.render(&plan(
        16,
        8,
        vec![layer(solid(16, 8, [255, 255, 255, 255]))],
    ));
    assert_eq!((large.width(), large.height()), (16, 8));
    assert_eq!(large.pixel(15, 7), Some([255, 255, 255, 255]));
}

/// A pass reads its title's reveal map through `reveal_order()`: the
/// first word's half of the canvas reads 0, the second word's half
/// reads 1, and a pass without one - the common case, any pass over
/// anything that is not a title - reads the identity, 0 everywhere.
#[test]
fn a_pass_reads_its_reveal_map_and_the_identity_everywhere() {
    let Some(mut gpu) = gpu() else { return };
    let body = "fn effect(uv: vec2<f32>) -> vec4<f32> { let r = reveal_order(uv); return vec4<f32>(r, r, r, 1.0); }";
    let map = RevealMap::from_rects(4, 4, &[(0, 0, 2, 4), (2, 0, 2, 4)]);
    let mut revealed = layer(solid(4, 4, [0, 0, 0, 255]));
    let mut pass = package("test.reveal", body, "", &[], 1.0);
    pass.reveal_map = Some(Arc::new(map));
    revealed.effects = vec![pass];
    let out = gpu.render(&plan(4, 4, vec![revealed]));
    let pixels = out.pixels();
    assert_eq!(pixels[0], 0, "{:?}", &pixels[..4]);
    assert_eq!(pixels[2 * 4], 255, "{:?}", &pixels[8..12]);

    let mut plain = layer(solid(4, 4, [0, 0, 0, 255]));
    plain.effects = vec![package("test.reveal", body, "", &[], 1.0)];
    let out = gpu.render(&plan(4, 4, vec![plain]));
    assert_eq!(&out.pixels()[..4], &[0, 0, 0, 255]);
}

/// A transition combines its two inputs through its shader: a trivial
/// dissolve over two solid colours must equal `from` at progress 0, `to`
/// at progress 1, and the exact half-and-half mix at progress 0.5. The
/// golden every packaged transition's own shader is measured against.
#[test]
fn a_transition_combines_its_two_inputs_by_progress() {
    let Some(mut gpu) = gpu() else { return };
    let manifest = concat_effects::Manifest::parse(
        "format = 2\n[effect]\nid = \"test.dissolve\"\nname = \"Dissolve\"\nkind = \"transition\"\n[transition]\nentry = \"effect.wgsl\"\n",
    )
    .expect("a manifest");
    let shader = concat_effects::TransitionShader::compile(
        &manifest,
        "fn transition(uv: vec2<f32>, progress: f32) -> vec4<f32> { return mix(from_at(uv), to_at(uv), progress); }",
    )
    .expect("compiles");
    let red = solid(4, 4, [255, 0, 0, 255]);
    let blue = solid(4, 4, [0, 0, 255, 255]);

    let mut at = |progress: f32| {
        let pass = shader.pass(&Default::default(), &[], progress, None);
        gpu.combine(4, 4, 0.0, &red, &blue, &pass)
            .expect("a GPU combine")
    };
    assert_eq!(&at(0.0).pixels()[..3], &[255, 0, 0], "all outgoing at 0");
    assert_eq!(&at(1.0).pixels()[..3], &[0, 0, 255], "all incoming at 1");
    // The package mixes in light, so its half is half of each light,
    // which BT.1886's 2.4 gamma stores at 191.
    let half = at(0.5);
    assert!(
        half.pixels()[..3]
            .iter()
            .zip([191u8, 0, 191])
            .all(|(got, want)| got.abs_diff(want) <= 2),
        "the half mix: {:?}",
        &half.pixels()[..3]
    );
}

/// Every packaged effect and filter with a shader actually renders on
/// the GPU at its default settings - naga's validation at load catches
/// a broken shader's syntax and types, but only a real pipeline creation
/// and draw catches a binding or layout mistake.
#[test]
fn every_shader_package_renders_at_its_defaults() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let source = solid(4, 4, [200, 120, 60, 255]);
    let catalogue = concat_effects::Catalogue::builtin();
    for package in catalogue.packages() {
        let Some(shader) = package.shader() else {
            continue;
        };
        let values = package.resolve(&Default::default());
        let pass = shader.pass(
            &values,
            &package.manifest.params,
            1.0,
            package.lut().cloned(),
            None,
        );
        let mut treated = layer(source.clone());
        treated.effects = vec![pass];
        let out = gpu.render(&plan(4, 4, vec![treated]));
        assert_eq!(out.pixels().len(), 4 * 4 * 4, "{}", package.id());
    }
}

/// A pass reads how long its own clip has been on screen through
/// `frame.clip_time` - the gap between the frame's own time and where the
/// clip begins on the timeline, not the timeline's absolute clock. A
/// clip starting at 2s, five seconds into the timeline, has been on
/// screen for exactly three.
#[test]
fn a_pass_reads_its_layers_clip_relative_time() {
    let Some(mut gpu) = gpu() else { return };
    let body = "fn effect(uv: vec2<f32>) -> vec4<f32> { if (abs(frame.clip_time - 3.0) < 0.001) { return vec4<f32>(0.0, 1.0, 0.0, 1.0); } return vec4<f32>(1.0, 0.0, 0.0, 1.0); }";
    let mut timed = layer(solid(4, 4, [0, 0, 0, 255]));
    timed.clip_start = concat_core::time::Rational::approximate(2.0).expect("a rational");
    timed.effects = vec![package("test.cliptime", body, "", &[], 1.0)];
    let mut p = plan(4, 4, vec![timed]);
    p.time = concat_core::time::Rational::approximate(5.0).expect("a rational");
    let out = gpu.render(&p);
    assert_eq!(
        &out.pixels()[..3],
        &[0, 255, 0],
        "clip_time should read 3.0"
    );
}

/// Every packaged transition's pipeline actually creates and runs on the
/// GPU, across its whole progress range - naga's validation at load
/// catches a broken shader's syntax and types, but only a real pipeline
/// creation catches a binding or layout mistake.
#[test]
fn every_packaged_transition_combines_across_its_progress_range() {
    let Some(mut gpu) = gpu() else { return };
    let red = solid(4, 4, [255, 0, 0, 255]);
    let blue = solid(4, 4, [0, 0, 255, 255]);
    let catalogue = concat_effects::Catalogue::builtin();
    for package in catalogue.packages() {
        if package.kind() != concat_effects::Kind::Transition {
            continue;
        }
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let pass = catalogue
                .transition_pass(package.id(), &Default::default(), progress)
                .unwrap_or_else(|| panic!("{} has no transition pass", package.id()));
            gpu.combine(4, 4, 0.0, &red, &blue, &pass)
                .unwrap_or_else(|| {
                    panic!("{} failed to combine at progress {progress}", package.id())
                });
        }
    }
}

/// Halfway through, every packaged transition but the plain dissolves draws
/// something a dissolve does not: "all the transitions look the same, just
/// a fade" (#448) is a shader that compiled and ran and drew the mix
/// anyway. Over two busy pictures, so a slide or a zoom moves something.
#[test]
fn halfway_through_a_packaged_transition_is_not_a_dissolve() {
    let Some(mut gpu) = gpu() else { return };
    let (width, height) = (64, 36);
    let from = gradient(width, height);
    let mut to = Frame::transparent(width, height);
    for y in 0..height {
        for x in 0..width {
            let on = (x / 8 + y / 8) % 2 == 0;
            to.set_pixel(
                x,
                y,
                if on {
                    [240, 240, 40, 255]
                } else {
                    [20, 90, 200, 255]
                },
            );
        }
    }
    let mix = crate::reference::mix_frames(&from, &to, 0.5);
    let catalogue = concat_effects::Catalogue::builtin();
    let mut report = Vec::new();
    for package in catalogue.packages() {
        if package.kind() != concat_effects::Kind::Transition {
            continue;
        }
        let pass = catalogue
            .transition_pass(package.id(), &Default::default(), 0.5)
            .expect("a pass");
        let out = gpu
            .combine(width, height, 0.0, &from, &to, &pass)
            .expect("combines");
        let differs = out
            .pixels()
            .iter()
            .zip(mix.pixels())
            .map(|(a, b)| u32::from(a.abs_diff(*b)))
            .sum::<u32>() as f64
            / out.pixels().len() as f64;
        report.push(format!("{} {differs:.2}", package.id()));
        if !package.id().contains("dissolve") {
            assert!(
                differs > 2.0,
                "{} is a dissolve halfway: {differs:.2}",
                package.id()
            );
        }
    }
    eprintln!("{}", report.join("\n"));
}

/// A transition's knob, set on the cut, changes what the shader draws:
/// Push to the right is not Push to the left.
#[test]
fn a_transitions_knob_changes_what_it_draws() {
    let Some(mut gpu) = gpu() else { return };
    let (width, height) = (64, 36);
    let (from, to) = (
        gradient(width, height),
        solid(width, height, [30, 200, 60, 255]),
    );
    let catalogue = concat_effects::Catalogue::builtin();
    let draw = |gpu: &mut WgpuCompositor, direction: f64| {
        let params = BTreeMap::from([("direction".to_owned(), direction)]);
        let pass = catalogue
            .transition_pass("concat.push", &params, 0.5)
            .expect("push has a pass");
        gpu.combine(width, height, 0.0, &from, &to, &pass)
            .expect("combines")
    };
    let left = draw(&mut gpu, 0.0);
    let right = draw(&mut gpu, 1.0);
    assert_ne!(
        left.pixels(),
        right.pixels(),
        "the direction knob reached the shader"
    );
}

/// A texture read back into a frame, for comparing a picture that stayed
/// on the device with one that was read back on the way.
fn read_texture(gpu: &WgpuCompositor, texture: &wgpu::Texture) -> Frame {
    let (width, height) = (texture.width(), texture.height());
    let padded = (width as usize * 4).div_ceil(256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: (padded * height as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device answers");
    let data = slice.get_mapped_range().expect("mapped");
    let mut frame = Frame::transparent(width, height);
    let row = width as usize * 4;
    for y in 0..height as usize {
        frame.pixels_mut()[y * row..(y + 1) * row]
            .copy_from_slice(&data[y * padded..y * padded + row]);
    }
    frame
}

/// A packaged transition drawn whole on the device is the picture the
/// read-back path makes: the same stacks, the same shader, only without
/// the round trips through memory.
#[test]
fn a_transition_kept_on_the_device_matches_the_read_back_one() {
    let Some(mut gpu) = gpu() else { return };
    let (width, height) = (32, 18);
    let plan = |colour: [u8; 4]| FramePlan {
        time: concat_core::time::Rational::new(1, 2),
        width,
        height,
        layers: vec![PlannedLayer::picture(
            detached_clip(),
            Arc::new(solid(width, height, colour)),
        )],
        treatments: Vec::new(),
        output: concat_core::frame::Signal::Sdr,
    };
    let (from, to) = (plan([220, 30, 30, 255]), plan([30, 30, 220, 255]));
    let catalogue = concat_effects::Catalogue::builtin();
    let package = catalogue
        .packages()
        .find(|package| package.kind() == concat_effects::Kind::Transition)
        .expect("a transition package");
    let pass = catalogue
        .transition_pass(package.id(), &Default::default(), 0.5)
        .expect("a pass");

    let from_frame = gpu.render(&from);
    let to_frame = gpu.render(&to);
    let read_back = gpu
        .combine(width, height, from.seconds(), &from_frame, &to_frame, &pass)
        .expect("combines");
    let texture = gpu
        .render_transition_texture(&from, &to, &pass)
        .expect("combines on the device");
    let kept = read_texture(&gpu, &texture);
    let worst = kept
        .pixels()
        .iter()
        .zip(read_back.pixels())
        .enumerate()
        .filter(|(index, _)| index % 4 != 3)
        .map(|(_, (a, b))| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(
        worst <= 1,
        "{}: the two paths differ by {worst} a channel",
        package.id()
    );
}

/// A frame the device has drawn is found on it again when a later
/// composite asks for it - a scrub back over ground the monitor showed -
/// rather than uploaded again: the layer pool keeps what it uploaded until
/// its budget is spent.
#[test]
fn a_frame_drawn_before_is_not_uploaded_again() {
    let Some(mut gpu) = gpu() else { return };
    let frames: Vec<Arc<Frame>> = (0..10u8)
        .map(|shade| Arc::new(solid(64, 36, [shade * 20, 40, 90, 255])))
        .collect();
    let draw = |gpu: &mut WgpuCompositor, frame: &Arc<Frame>| {
        let plan = FramePlan {
            time: concat_core::time::Rational::new(0, 1),
            width: 64,
            height: 36,
            layers: vec![PlannedLayer::picture(detached_clip(), Arc::clone(frame))],
            treatments: Vec::new(),
            output: concat_core::frame::Signal::Sdr,
        };
        gpu.render_texture(&plan).expect("draws");
    };
    for frame in &frames {
        draw(&mut gpu, frame);
    }
    let first_pass = gpu.uploads;
    assert!(first_pass >= 10, "each new frame is uploaded once");
    for frame in frames.iter().rev() {
        draw(&mut gpu, frame);
    }
    assert_eq!(gpu.uploads, first_pass, "the way back is all cache hits");
}

/// The pool keeps what it uploaded only up to its limits: a long run of
/// distinct frames - here more small ones than the texture cap - leaves it
/// holding no more than the cap and the byte budget allow.
/// The pool costs system memory unless the card has its own, and its
/// budget says so.
#[test]
fn the_pool_budget_follows_what_the_textures_cost() {
    let discrete = pool_budget_for(wgpu::DeviceType::DiscreteGpu);
    let integrated = pool_budget_for(wgpu::DeviceType::IntegratedGpu);
    let software = pool_budget_for(wgpu::DeviceType::Cpu);
    assert_eq!(discrete, 512 * 1024 * 1024);
    assert!(discrete > integrated && integrated > software);
}

#[test]
fn the_pool_stays_within_its_limits() {
    let Some(mut gpu) = gpu() else { return };
    for index in 0..(POOL_TEXTURES as u32 + 76) {
        let frame = Arc::new(solid(4, 4, [(index % 251) as u8, 7, 9, 255]));
        let plan = FramePlan {
            time: concat_core::time::Rational::new(0, 1),
            width: 4,
            height: 4,
            layers: vec![PlannedLayer::picture(detached_clip(), frame)],
            treatments: Vec::new(),
            output: concat_core::frame::Signal::Sdr,
        };
        gpu.render_texture(&plan).expect("draws");
    }
    assert!(
        gpu.pool_textures <= POOL_TEXTURES,
        "{} textures",
        gpu.pool_textures
    );
    assert!(gpu.pool_bytes <= gpu.pool_budget);
    // Shrunk when playback stops: down to the share asked for, oldest
    // first, and back to nothing on zero.
    gpu.shrink_pool(0.25);
    assert!(gpu.pool_bytes <= gpu.pool_budget / 4);
    gpu.shrink_pool(0.0);
    assert_eq!((gpu.pool_bytes, gpu.pool_textures), (0, 0));
    let counted: usize = gpu.pool.values().map(Vec::len).sum();
    assert_eq!(counted, gpu.pool_textures, "the count is the pool's");
}

/// A stack of passes takes turns between two textures and still applies
/// every pass in order: invert, turn the channels, invert again is the
/// turn alone, and five inverts are one. The stack claims two textures of
/// the layer's size whatever its length, beside the frame it reads and
/// the canvas.
#[test]
fn a_stack_of_passes_takes_turns_between_two_textures() {
    let Some(mut gpu) = gpu() else { return };
    const TURN: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(c.g, c.b, c.r, c.a); }";
    let invert = || package("concat.invert", INVERT, "", &[], 1.0);
    let turn = || package("concat.turn", TURN, "", &[], 1.0);
    let colour = [200, 90, 30, 255];
    let stacked = |effects: Vec<ShaderPass>| {
        let mut picture = layer(solid(8, 8, colour));
        picture.effects = effects;
        plan(8, 8, vec![picture])
    };

    let out = gpu.render(&stacked(vec![invert(), turn(), invert()]));
    let got = &out.pixels()[..3];
    assert!(
        got.iter()
            .zip([90, 30, 200])
            .all(|(got, want)| got.abs_diff(want) <= 1),
        "invert, turn, invert gave {got:?}"
    );
    // The frame, the two turns, and the canvas the frame is drawn on in
    // the working format before it is resolved.
    let after_three = gpu.pool[&(8, 8)].len();
    assert!(after_three <= 4, "{after_three} textures for three passes");

    let out = gpu.render(&stacked((0..5).map(|_| invert()).collect()));
    let got = &out.pixels()[..3];
    assert!(
        got.iter()
            .zip([55, 165, 225])
            .all(|(got, want)| got.abs_diff(want) <= 1),
        "five inverts gave {got:?}"
    );
    // One more: the new frame, the first still cached; the turns are reused.
    let after_five = gpu.pool[&(8, 8)].len();
    assert!(
        after_five <= after_three + 1,
        "{after_five} textures after five passes, {after_three} after three"
    );
}

/// The LUTs kept on the device are bounded: a run of more distinct looks
/// than the cache holds leaves it at its cap, the one just drawn among
/// them.
#[test]
fn the_lut_cache_keeps_the_recent_looks_only() {
    let Some(mut gpu) = gpu() else { return };
    let body = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(lut(c.rgb), c.a); }";
    let mut last = 0;
    for index in 0..(LUT_CACHE as u32 + 8) {
        let shade = index as f32 / 64.0;
        let table = Arc::new(Lut::from_rgb(2, &[shade, 0.5, 0.25].repeat(8)).expect("a table"));
        last = table.id;
        let mut looked = layer(solid(4, 4, [255, 0, 0, 255]));
        let mut pass = package("test.table", body, "", &[], 1.0);
        pass.lut = Some(table);
        looked.effects = vec![pass];
        gpu.render(&plan(4, 4, vec![looked]));
    }
    assert!(
        gpu.luts.len() <= LUT_CACHE,
        "{} looks cached",
        gpu.luts.len()
    );
    assert!(gpu.luts.contains_key(&last), "the look just drawn is kept");
    assert_eq!(gpu.luts.len(), gpu.luts_drawn.len());
}

/// The stack is drawn in half floats, so a pass that darkens a picture to
/// a tenth and one that brings it back up leave it as it was: in eight
/// bits the tenth rounds to a whole level and the way back multiplies the
/// rounding by ten.
#[test]
fn a_stack_keeps_its_precision_between_passes() {
    let Some(mut gpu) = gpu() else { return };
    const DOWN: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(c.rgb * 0.1, c.a); }";
    const UP: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(c.rgb * 10.0, c.a); }";
    let mut picture = layer(solid(8, 8, [203, 157, 97, 255]));
    picture.effects = vec![
        package("test.down", DOWN, "", &[], 1.0),
        package("test.up", UP, "", &[], 1.0),
    ];
    let out = gpu.render(&plan(8, 8, vec![picture]));
    let got = &out.pixels()[..3];
    assert!(
        got.iter()
            .zip([203, 157, 97])
            .all(|(got, want)| got.abs_diff(want) <= 1),
        "down and up again gave {got:?}"
    );
}

/// A deep frame of `level` a channel, sixteen bits, in `signal`.
fn deep(width: u32, height: u32, level: f32, signal: concat_core::frame::Signal) -> Frame {
    let value = (level.clamp(0.0, 1.0) * 65535.0).round() as u16;
    let pixel: Vec<u8> = [value, value, value, u16::MAX]
        .iter()
        .flat_map(|channel| channel.to_le_bytes())
        .collect();
    Frame::from_rgba64(
        width,
        height,
        pixel.repeat((width * height) as usize),
        signal,
    )
    .expect("a deep frame")
}

/// An HDR clip is converted into the working space on the GPU as it
/// uploads, and conformed to the SDR timeline there: HLG's reference white
/// (75 % of the signal, 203 nits) lands a little under SDR white, where
/// BT.2390's roll-off leaves room for the highlights above it, grey still;
/// the master's peak - all of HLG's signal, or 1000 nits of PQ - is SDR
/// white; black is black.
#[test]
fn an_hdr_frame_is_conformed_to_sdr_on_the_way_in() {
    use concat_core::frame::Signal;
    let Some(mut gpu) = gpu() else { return };
    let mut drawn = |frame: Frame| {
        let out = gpu.render(&plan(4, 4, vec![layer(frame)]));
        let p = out.pixel(1, 1).expect("in the frame");
        [p[0], p[1], p[2]]
    };
    let white = drawn(deep(4, 4, 0.75, Signal::Hlg));
    assert!(
        white.iter().all(|channel| (205..=245).contains(channel))
            && white[0].abs_diff(white[1]) <= 2
            && white[1].abs_diff(white[2]) <= 2,
        "HLG reference white came out {white:?}"
    );
    let peak = drawn(deep(4, 4, 1.0, Signal::Hlg));
    assert!(
        peak.iter().all(|channel| *channel >= 252),
        "HLG peak came out {peak:?}"
    );
    let black = drawn(deep(4, 4, 0.0, Signal::Hlg));
    assert!(
        black.iter().all(|channel| *channel <= 2),
        "HLG black came out {black:?}"
    );
    // 1000 nits in PQ is 0.7518 of the signal.
    let pq_peak = drawn(deep(4, 4, 0.7518, Signal::Pq));
    assert!(
        pq_peak.iter().all(|channel| *channel >= 250),
        "PQ 1000 nits came out {pq_peak:?}"
    );
}

/// `plan` drawn into the working space and read back as it stands there,
/// before the resolve: light, 1.0 SDR white, nothing clipped.
fn working(gpu: &mut WgpuCompositor, plan: &FramePlan) -> Vec<[f32; 4]> {
    let (draws, vertices) = gpu.prepare(plan);
    gpu.write_vertices(&vertices);
    let (canvas, texture, view) = gpu.canvas(plan.width, plan.height);
    let encoder = gpu.encode(&view, &texture, &draws, wgpu::Color::BLACK);
    gpu.queue.submit([encoder.finish()]);
    let out = read_floats(gpu, (plan.width, plan.height), canvas);
    gpu.retire();
    out
}

/// On an HDR timeline an HDR clip keeps its light above white: HLG's peak
/// is 1000 nits, 4.93 times SDR white, and its reference white SDR white
/// itself - where on an SDR timeline the peak is conformed down to white.
/// The same frame drawn for both is two pictures, kept apart. For the SDR
/// screen the resolve rolls the HDR timeline off with the maths the SDR
/// timeline's upload uses, so a lone clip looks the same on either; an SDR
/// clip's white goes a little under white there, leaving room above it.
#[test]
fn an_hdr_timeline_keeps_the_light_above_white() {
    use concat_core::frame::Signal;
    let Some(mut gpu) = gpu() else { return };
    let peak = deep(4, 4, 1.0, Signal::Hlg);
    let hdr = |layers| FramePlan {
        output: Signal::Hlg,
        ..plan(4, 4, layers)
    };
    let level = |pixels: Vec<[f32; 4]>| pixels[5];
    let sdr_light = level(working(&mut gpu, &plan(4, 4, vec![layer(peak.clone())])));
    assert!(
        (sdr_light[1] - 1.0).abs() < 0.01,
        "conformed: {sdr_light:?}"
    );
    let hdr_light = level(working(&mut gpu, &hdr(vec![layer(peak.clone())])));
    assert!(
        hdr_light[..3]
            .iter()
            .all(|channel| (channel - 1000.0 / 203.0).abs() < 0.05),
        "kept: {hdr_light:?}"
    );
    let white = level(working(
        &mut gpu,
        &hdr(vec![layer(deep(4, 4, 0.75, Signal::Hlg))]),
    ));
    assert!(
        (white[1] - 1.0).abs() < 0.01,
        "HLG reference white: {white:?}"
    );

    let shown = |gpu: &mut WgpuCompositor, plan: &FramePlan| {
        let out = gpu.render(plan);
        let p = out.pixel(1, 1).expect("in the frame");
        [p[0], p[1], p[2]]
    };
    for signal_level in [0.75, 1.0, 0.5] {
        let frame = deep(4, 4, signal_level, Signal::Hlg);
        let on_sdr = shown(&mut gpu, &plan(4, 4, vec![layer(frame.clone())]));
        let on_hdr = shown(&mut gpu, &hdr(vec![layer(frame)]));
        assert!(
            on_sdr.iter().zip(on_hdr).all(|(a, b)| a.abs_diff(b) <= 1),
            "HLG {signal_level}: {on_sdr:?} on SDR, {on_hdr:?} on HDR"
        );
    }
    let sdr_white = shown(
        &mut gpu,
        &hdr(vec![layer(solid(4, 4, [255, 255, 255, 255]))]),
    );
    assert!(
        sdr_white
            .iter()
            .all(|channel| (225..=240).contains(channel)),
        "SDR white on an HDR timeline: {sdr_white:?}"
    );
}

/// Delivered for an HDR file, an HDR timeline's frame comes back as the
/// signal it was made from: an HLG clip's levels through the working space
/// and out again, SDR white at HLG's reference white (75 %, 203 nits), and
/// the same for PQ, whose 203 nits are 58 % of its signal. Sixteen bits a
/// channel, opaque; an SDR timeline's frames stay eight-bit SDR.
#[test]
fn an_hdr_timeline_is_delivered_as_its_signal() {
    use concat_core::frame::{Depth, Signal};
    let Some(mut gpu) = gpu() else { return };
    gpu.deliver_hdr(true);
    let level = |frame: &Frame| {
        assert_eq!(frame.depth(), Depth::Sixteen);
        let at = (4 + 1) * 8;
        let channel = |offset: usize| {
            f64::from(u16::from_le_bytes([
                frame.pixels()[at + offset],
                frame.pixels()[at + offset + 1],
            ])) / 65_535.0
        };
        assert_eq!(channel(6), 1.0, "opaque");
        [channel(0), channel(2), channel(4)]
    };
    for signal in [Signal::Hlg, Signal::Pq] {
        let delivered = |gpu: &mut WgpuCompositor, frame: Frame| {
            gpu.render(&FramePlan {
                output: signal,
                ..plan(4, 4, vec![layer(frame)])
            })
        };
        for going in [0.25, 0.5, 0.75, 0.9] {
            let out = delivered(&mut gpu, deep(4, 4, going as f32, signal));
            assert_eq!(out.signal(), signal);
            let got = level(&out);
            assert!(
                got.iter().all(|channel| (channel - going).abs() < 0.003),
                "{signal:?} {going} came back {got:?}"
            );
        }
        let white = level(&delivered(&mut gpu, solid(4, 4, [255, 255, 255, 255])));
        let want = if signal == Signal::Hlg { 0.75 } else { 0.5807 };
        assert!(
            white.iter().all(|channel| (channel - want).abs() < 0.003),
            "{signal:?}: SDR white came out {white:?}"
        );
    }
    let sdr = gpu.render(&plan(4, 4, vec![layer(solid(4, 4, [255, 255, 255, 255]))]));
    assert_eq!(sdr.depth(), Depth::Eight, "an SDR timeline stays SDR");
}

/// A pass of a format 2 package - the scene-linear contract - with `body`
/// as its shader and no knobs, at `intensity`.
fn scene_linear(id: &str, body: &str, intensity: f32) -> ShaderPass {
    let manifest = concat_effects::Manifest::parse(&format!(
        "format = 2\n[effect]\nid = \"{id}\"\nname = \"Test\"\nkind = \"effect\"\n[wgsl]\nentry = \"effect.wgsl\"\n"
    ))
    .expect("a manifest");
    let shader = concat_effects::Shader::compile(&manifest, body).expect("compiles");
    shader.pass(&BTreeMap::new(), &manifest.params, intensity, None, None)
}

/// Within `tolerance` of `want` a channel, relative, for the colour and
/// alpha alike.
fn near(got: [f32; 4], want: [f32; 4], tolerance: f32) -> bool {
    got.iter()
        .zip(want)
        .all(|(got, want)| (got - want).abs() <= tolerance * want.abs().max(0.01))
}

/// Every built-in package's probes: a picture of one working-space colour
/// through its shader comes out as the probe pins, in floats.
#[test]
fn every_package_probe_holds() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let mut failures = Vec::new();
    let mut probed = 0;
    for package in concat_effects::Catalogue::builtin().packages() {
        for (n, probe) in package.probes.iter().enumerate() {
            let got = match (package.probe_pass(probe), package.probe_transition(probe)) {
                (Some(pass), _) => gpu.probe(&[pass], probe.input, 8, 0.0),
                (None, Some(cut)) => {
                    gpu.probe_transition(&cut, probe.input, probe.to.unwrap_or_default(), 8, 0.0)
                }
                (None, None) => panic!("{}: a probe with no shader", package.id()),
            }
            .expect("reads back");
            probed += 1;
            if let Err(error) = probe.check(got) {
                failures.push(format!("{}: {}\n  {error}", package.id(), probe.label(n)));
            }
        }
    }
    eprintln!("{probed} probes");
    assert!(probed > 0, "no package has a probe");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The prelude's hue turn leaves a colour as it was when it turns it by
/// nothing - the identity every HSL band rests on.
#[test]
fn a_hue_turn_of_nothing_is_the_colour_it_was() {
    let Some(mut gpu) = gpu() else { return };
    const TURN: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(hue_rotate(c.rgb, 0.0), c.a); }";
    let pass = package("test.turn", TURN, "", &[], 1.0);
    for colour in [
        [0.8, 0.05, 0.05, 1.0],
        [0.05, 0.6, 0.1, 1.0],
        [0.1, 0.2, 0.9, 1.0],
    ] {
        let got = gpu
            .probe(std::slice::from_ref(&pass), colour, 8, 0.0)
            .expect("reads back");
        // Half floats through the display encoding and back: a hundredth
        // of a channel, where the turn that was wrong moved one by 0.09.
        assert!(near(got, colour, 0.01), "{colour:?} came back {got:?}");
    }
}

/// Halation's glow is not held at white: a brighter highlight lights the
/// picture around it more.
#[test]
fn halation_glows_brighter_from_light_past_white() {
    let Some(mut gpu) = gpu() else { return };
    const GLOW: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(halation(uv, vec3<f32>(0.0), 0.5, 4.0, vec3<f32>(1.0), 1.0), c.a); }";
    let pass = package("test.glow", GLOW, "", &[], 1.0);
    let white = gpu
        .probe(std::slice::from_ref(&pass), [1.0, 1.0, 1.0, 1.0], 8, 0.0)
        .expect("reads back");
    let past = gpu
        .probe(&[pass], [4.0, 4.0, 4.0, 1.0], 8, 0.0)
        .expect("reads back");
    assert!(
        past[0] > white[0] * 1.5,
        "white glows {white:?}, four times white {past:?}"
    );
}

/// A format 2 stack hands its light on unclipped: four times the light and
/// a quarter of it again is the picture it was, a highlight far above
/// white in between.
#[test]
fn a_scene_linear_stack_keeps_the_light_above_white() {
    let Some(mut gpu) = gpu() else { return };
    const UP: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(c.rgb * 4.0, c.a); }";
    const DOWN: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(c.rgb * 0.25, c.a); }";
    let colour = [0.7, 0.35, 0.1, 1.0];
    let linear = [
        scene_linear("test.up", UP, 1.0),
        scene_linear("test.down", DOWN, 1.0),
    ];
    let between = gpu.probe(&linear[..1], colour, 8, 0.0).expect("reads back");
    assert!(near(between, [2.8, 1.4, 0.4, 1.0], 0.004), "{between:?}");
    let got = gpu.probe(&linear, colour, 8, 0.0).expect("reads back");
    assert!(near(got, colour, 0.004), "there and back gave {got:?}");
}

/// `to_log` puts middle grey, SDR white and black where ACEScct does, and
/// `from_log` brings any level back - a highlight, the toe, a negative
/// channel - as it went in.
#[test]
fn the_log_space_goes_there_and_back() {
    let Some(mut gpu) = gpu() else { return };
    let to_log = scene_linear(
        "test.log",
        "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(to_log(c.rgb), c.a); }",
        1.0,
    );
    let got = gpu
        .probe(std::slice::from_ref(&to_log), [0.18, 1.0, 0.0, 1.0], 4, 0.0)
        .expect("reads back");
    assert!(near(got, [0.4136, 0.5548, 0.0729, 1.0], 0.002), "{got:?}");

    let round = scene_linear(
        "test.round",
        "fn effect(uv: vec2<f32>) -> vec4<f32> { let c = sample(uv); return vec4<f32>(from_log(to_log(c.rgb)), c.a); }",
        1.0,
    );
    for colour in [[6.0, 0.18, -0.02, 1.0], [0.001, 0.0, 1.0, 1.0]] {
        let got = gpu
            .probe(std::slice::from_ref(&round), colour, 4, 0.0)
            .expect("reads back");
        assert!(near(got, colour, 0.004), "{colour:?} came back {got:?}");
    }
}

/// A format 2 pass is mixed over the untouched layer in light: a quarter
/// of the way from 0.2 to white is 0.4.
#[test]
fn a_scene_linear_pass_mixes_by_intensity_in_light() {
    let Some(mut gpu) = gpu() else { return };
    let white = scene_linear(
        "test.white",
        "fn effect(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(1.0, 1.0, 1.0, sample(uv).a); }",
        0.25,
    );
    let got = gpu
        .probe(&[white], [0.2, 0.2, 0.2, 1.0], 4, 0.0)
        .expect("reads back");
    assert!(near(got, [0.4, 0.4, 0.4, 1.0], 0.004), "{got:?}");
}

/// No look clips: a highlight four times SDR white comes out of every
/// built-in look drawn in the display space brighter than white does, as
/// light that went in brighter should.
#[test]
fn every_look_carries_a_highlight_past_white() {
    let Some(mut gpu) = gpu() else { return };
    let luma = |c: [f32; 4]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let mut clipped = Vec::new();
    for package in concat_effects::Catalogue::builtin().packages() {
        let display = package
            .manifest
            .wgsl
            .as_ref()
            .is_some_and(|wgsl| wgsl.space == concat_effects::Space::Display);
        if !display {
            continue;
        }
        let pass = package.trial_pass().expect("a shader");
        let white = gpu
            .probe(std::slice::from_ref(&pass), [1.0, 1.0, 1.0, 1.0], 8, 0.0)
            .expect("reads back");
        let bright = gpu
            .probe(&[pass], [4.0, 4.0, 4.0, 1.0], 8, 0.0)
            .expect("reads back");
        if luma(bright) <= luma(white) * 1.5 {
            clipped.push(format!(
                "{}: white {white:?}, 4x white {bright:?}",
                package.id()
            ));
        }
    }
    assert!(clipped.is_empty(), "\n{}", clipped.join("\n"));
}

/// `pixels`, straight RGBA in the working space and row by row, written
/// into a pooled texture `width` by `height` as half floats: a picture
/// with the light past white and the alpha an eight-bit frame cannot hold.
fn written(gpu: &mut WgpuCompositor, (width, height): (u32, u32), pixels: &[[f32; 4]]) -> usize {
    let index = gpu.claim(width, height);
    let texels: Vec<u8> = pixels
        .iter()
        .flatten()
        .flat_map(|channel| half::f16::from_f32(*channel).to_le_bytes())
        .collect();
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu.pool[&(width, height)][index].texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * WORK_BYTES as u32),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    index
}

/// The pooled texture at `index`, `width` by `height`, read back whole in
/// the half floats it holds.
fn read_floats(gpu: &WgpuCompositor, size: (u32, u32), index: usize) -> Vec<[f32; 4]> {
    read_region(gpu, size, index, size)
}

/// The top-left `width` by `height` pixels of the `frame`-sized texture
/// `index`, read back in floats.
fn read_region(
    gpu: &WgpuCompositor,
    frame: (u32, u32),
    index: usize,
    (width, height): (u32, u32),
) -> Vec<[f32; 4]> {
    let row = width as usize * WORK_BYTES as usize;
    let padded = row.div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test float readback"),
        size: (padded * height as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        gpu.pool[&frame][index].texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device answers");
    let data = slice.get_mapped_range().expect("mapped");
    (0..height as usize)
        .flat_map(|y| data[y * padded..y * padded + row].chunks_exact(8))
        .map(|texel| {
            let channel = |at: usize| half::f16::from_le_bytes([texel[at], texel[at + 1]]).to_f32();
            [channel(0), channel(2), channel(4), channel(6)]
        })
        .collect()
}

/// `passes` over `picture`, `size` pixels, laid at the top left of an
/// otherwise transparent layer `frame` pixels in size - what an effect that
/// scales with the layer's size sees of a picture that size - with only
/// `picture`'s own pixels read back. A transparent black pixel costs an
/// effect like Beauty one read, so a large frame stays cheap to test.
fn treated_within(
    gpu: &mut WgpuCompositor,
    frame: (u32, u32),
    size: (u32, u32),
    picture: &[[f32; 4]],
    passes: &[ShaderPass],
) -> Vec<[f32; 4]> {
    assert!(
        size.0 <= frame.0 && size.1 <= frame.1,
        "the picture fits the frame"
    );
    gpu.used.values_mut().for_each(|used| *used = 0);
    gpu.composites += 1;
    let index = gpu.claim(frame.0, frame.1);
    let write = |gpu: &WgpuCompositor, (width, height): (u32, u32), texels: &[u8]| {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu.pool[&frame][index].texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * WORK_BYTES as u32),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    };
    // The pool may hand back a texture something else was drawn into: it
    // is cleared on the device, before the picture is written over it.
    let view = gpu.pool[&frame][index]
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("test clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    gpu.queue.submit([encoder.finish()]);
    let texels: Vec<u8> = picture
        .iter()
        .flatten()
        .flat_map(|channel| half::f16::from_f32(*channel).to_le_bytes())
        .collect();
    write(gpu, size, &texels);
    let drawn = gpu.run_passes(frame.0, frame.1, index, passes, 0.0, 0.0);
    let out = read_region(gpu, frame, drawn, size);
    gpu.retire();
    out
}

/// `passes` over `picture`, `size` pixels, as a layer's effects run: the
/// last picture drawn, read back in floats.
fn treated(
    gpu: &mut WgpuCompositor,
    size: (u32, u32),
    picture: &[[f32; 4]],
    passes: &[ShaderPass],
) -> Vec<[f32; 4]> {
    gpu.used.values_mut().for_each(|used| *used = 0);
    gpu.composites += 1;
    let source = written(gpu, size, picture);
    let drawn = gpu.run_passes(size.0, size.1, source, passes, 0.0, 0.0);
    let out = read_floats(gpu, size, drawn);
    gpu.retire();
    out
}

/// A pass of a format 2 package drawn in several passes: `body` as its
/// shader, `passes` its `[[wgsl.pass]]` tables, `params` its knobs, at
/// `intensity`.
fn in_passes(body: &str, passes: &str, params: &str, intensity: f32) -> ShaderPass {
    let manifest = concat_effects::Manifest::parse(&format!(
        "format = 2\n[effect]\nid = \"test.passes\"\nname = \"Test\"\nkind = \"effect\"\n{params}\n[wgsl]\nentry = \"effect.wgsl\"\n{passes}"
    ))
    .expect("a manifest");
    let shader = concat_effects::Shader::compile(&manifest, body).expect("compiles");
    shader.pass(&BTreeMap::new(), &manifest.params, intensity, None, None)
}

/// The picture `picture`, `size` pixels, blurred by a Gaussian of `sigma`
/// pixels summed directly over both axes at `(x, y)`, out to five sigmas,
/// the edge clamped as the GPU's sampler clamps it: premultiplied colour,
/// then alpha, in doubles.
fn gaussian_at(
    picture: &[[f32; 4]],
    (width, height): (u32, u32),
    (x, y): (i64, i64),
    sigma: f64,
) -> [f64; 4] {
    let reach = (5.0 * sigma).ceil() as i64;
    let (mut sum, mut total) = ([0.0f64; 4], 0.0f64);
    for dy in -reach..=reach {
        let row = (y + dy).clamp(0, i64::from(height) - 1);
        for dx in -reach..=reach {
            let column = (x + dx).clamp(0, i64::from(width) - 1);
            let weight = (-((dx * dx + dy * dy) as f64) / (2.0 * sigma * sigma)).exp();
            let [r, g, b, a] = picture[(row * i64::from(width) + column) as usize].map(f64::from);
            for (into, value) in sum.iter_mut().zip([r * a, g * a, b * a, a]) {
                *into += weight * value;
            }
            total += weight;
        }
    }
    sum.map(|value| value / total)
}

/// A picture `size` pixels with everything a blur can get wrong in it: a
/// gradient, a checkerboard, dots of light `peak` times white, a hard edge
/// past white, a half-transparent strip down the left, and a transparent
/// band whose colour must not bleed into anything.
fn busy(size: (u32, u32), peak: f32) -> Vec<[f32; 4]> {
    (0..size.1)
        .flat_map(|y| (0..size.0).map(move |x| (x, y)))
        .map(|(x, y)| {
            let checker = if (x / 6 + y / 6) % 2 == 0 { 0.1 } else { 0.6 };
            let mut rgb = [
                0.02 + 0.9 * x as f32 / size.0 as f32,
                checker,
                0.05 + 0.5 * y as f32 / size.1 as f32,
            ];
            if x >= 150 {
                rgb = [1.5, 0.05, 0.05];
            }
            if x % 40 / 2 == 10 && y % 30 / 2 == 7 {
                rgb = [peak; 3];
            }
            match y {
                80..90 => [5.0, 5.0, 5.0, 0.0],
                _ if x < 20 => [rgb[0], rgb[1], rgb[2], 0.5],
                _ => [rgb[0], rgb[1], rgb[2], 1.0],
            }
        })
        .collect()
}

/// `picture`, `size` pixels, at `(x, y)` in pixels - a pixel's centre at
/// its whole number - read between its pixels as `sample_premultiplied`
/// reads it: the four about it premultiplied and mixed bilinearly, the edge
/// clamped as the GPU's sampler clamps it; in doubles.
fn premultiplied_at(
    picture: &[[f32; 4]],
    (width, height): (u32, u32),
    (x, y): (f64, f64),
) -> [f64; 4] {
    let x = x.clamp(0.0, f64::from(width - 1));
    let y = y.clamp(0.0, f64::from(height - 1));
    let (left, top) = (x.floor(), y.floor());
    let (fx, fy) = (x - left, y - top);
    let mut mixed = [0.0f64; 4];
    for (down, wy) in [(0, 1.0 - fy), (1, fy)] {
        for (across, wx) in [(0, 1.0 - fx), (1, fx)] {
            let column = (left as i64 + across).min(i64::from(width) - 1);
            let row = (top as i64 + down).min(i64::from(height) - 1);
            let [r, g, b, a] = picture[(row * i64::from(width) + column) as usize].map(f64::from);
            for (into, value) in mixed.iter_mut().zip([r * a, g * a, b * a, a]) {
                *into += wx * wy * value;
            }
        }
    }
    mixed
}

/// How far `got` is from `want` at a grid of pixels - every `stride`th of
/// each row and column, and the last, where `want` has an answer -
/// premultiplied: the worst, where, and the root mean square.
fn measured(
    got: &[[f32; 4]],
    (width, height): (u32, u32),
    stride: usize,
    want: impl Fn(i64, i64) -> Option<[f64; 4]>,
) -> (f64, (i64, i64), f64) {
    let along = |length: u32| {
        (0..i64::from(length))
            .step_by(stride)
            .chain([i64::from(length) - 1])
            .collect::<Vec<_>>()
    };
    let (mut worst, mut at, mut squares, mut count) = (0.0f64, (0, 0), 0.0f64, 0);
    for &y in &along(height) {
        for &x in &along(width) {
            let Some(want) = want(x, y) else { continue };
            let [r, g, b, a] = got[(y * i64::from(width) + x) as usize].map(f64::from);
            for (got, want) in [r * a, g * a, b * a, a].into_iter().zip(want) {
                let error = (got - want).abs();
                squares += error * error;
                count += 1;
                if error > worst {
                    (worst, at) = (error, (x, y));
                }
            }
        }
    }
    (worst, at, (squares / f64::from(count)).sqrt())
}

/// The Gaussian blur - across and then down, at a power of two fewer
/// pixels where its radius is wide, and read back over the layer - is the
/// 2-D Gaussian summed directly, at every radius from its least to its
/// most: over a picture with a gradient, a checkerboard, dots of light six
/// times white, a hard edge past white, a half-transparent strip, and a
/// transparent band whose colour must not bleed into anything.
#[test]
fn the_gaussian_blur_is_a_direct_gaussian() {
    const PEAK: f64 = 6.0;
    let Some(mut gpu) = heavy_gpu() else { return };
    let blur = concat_effects::Catalogue::builtin()
        .get("concat.gaussian-blur")
        .expect("a built-in");
    let size = (192, 108);
    let picture = busy(size, PEAK as f32);
    // Below a radius of six the last pass blurs down `across` itself, and
    // `down` is a sliver nothing reads.
    for (radius, across, down) in [
        (1.0, 1, 64),
        (4.0, 1, 64),
        (10.0, 2, 2),
        (24.0, 8, 8),
        (50.0, 16, 16),
    ] {
        let pass = blur
            .pass(&BTreeMap::from([("radius".to_owned(), radius)]), None)
            .expect("a shader");
        assert_eq!(
            pass.stages
                .iter()
                .map(|stage| stage.shrink)
                .collect::<Vec<_>>(),
            [[across, 1], [down, down]],
            "radius {radius}"
        );
        let got = treated(&mut gpu, size, &picture, &[pass]);
        // Every pixel where the sum is quick, a grid of them - the edges
        // among them - where it is not.
        let stride = (radius / 4.0).max(1.0) as usize;
        let (worst, at, rms) = measured(&got, size, stride, |x, y| {
            Some(gaussian_at(&picture, size, (x, y), radius))
        });
        eprintln!("radius {radius}: worst {worst:.5} at {at:?}, rms {rms:.6}");
        // Half floats, the fetch between two pixels, and the sum cut at
        // three sigmas each leave a trace; together they come to a quarter
        // of a percent of the brightest light at worst.
        assert!(
            worst < 0.0025 * PEAK,
            "radius {radius}: {worst} off at {at:?}"
        );
        assert!(rms < 0.0005 * PEAK, "radius {radius}: rms {rms}");
    }
}

/// A pass's picture is the layer's size divided by its shrink, rounded
/// up, and a later pass measures it through `<target>_texel`.
#[test]
fn a_pass_draws_its_picture_at_its_shrunk_size() {
    let Some(mut gpu) = gpu() else { return };
    let pass = in_passes(
        "fn small(uv: vec2<f32>) -> vec4<f32> { return sample(uv); }\n\
         fn effect(uv: vec2<f32>) -> vec4<f32> { let seen = small_at(uv); return vec4<f32>(1.0 / small_texel(), seen.b, 1.0); }",
        "[[wgsl.pass]]\ntarget = \"small\"\nshrink = [\"4\", \"2\"]\n",
        "",
        1.0,
    );
    for (side, want) in [(8, [2.0, 4.0]), (10, [3.0, 5.0])] {
        let got = gpu
            .probe(std::slice::from_ref(&pass), [0.1, 0.2, 0.3, 1.0], side, 0.0)
            .expect("reads back");
        assert!(
            near(got, [want[0], want[1], 0.3, 1.0], 0.004),
            "{side}: {got:?}"
        );
    }
}

/// A pass's picture holds what its function returned - past white and
/// below zero, nothing clipped or mixed - and only the last pass mixes
/// by intensity: a quarter of the way from 0.2 to the picture, once.
#[test]
fn only_the_last_pass_mixes_by_intensity() {
    let Some(mut gpu) = gpu() else { return };
    let at = |intensity: f32| {
        in_passes(
            "fn light(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(6.0, -0.5, 1.0, 1.0); }\n\
             fn effect(uv: vec2<f32>) -> vec4<f32> { return light_at(uv); }",
            "[[wgsl.pass]]\ntarget = \"light\"\nshrink = [\"2\", \"2\"]\n",
            "",
            intensity,
        )
    };
    let whole = gpu
        .probe(&[at(1.0)], [0.2, 0.2, 0.2, 1.0], 8, 0.0)
        .expect("reads back");
    assert!(near(whole, [6.0, -0.5, 1.0, 1.0], 0.004), "{whole:?}");
    let quarter = gpu
        .probe(&[at(0.25)], [0.2, 0.2, 0.2, 1.0], 8, 0.0)
        .expect("reads back");
    assert!(near(quarter, [1.65, 0.025, 0.4, 1.0], 0.004), "{quarter:?}");
}

/// A stack of packages drawn in passes hands its pictures from one package
/// to the next: five blurs claim no more textures than two, and draw a
/// flat picture as it was.
#[test]
fn a_stack_of_blurs_shares_its_pictures() {
    let Some(mut gpu) = gpu() else { return };
    let blur = concat_effects::Catalogue::builtin()
        .get("concat.gaussian-blur")
        .expect("a built-in")
        .trial_pass()
        .expect("a shader");
    let colour = [0.3, 2.0, 0.05, 1.0];
    let two = gpu
        .probe(&vec![blur.clone(); 2], colour, 64, 0.0)
        .expect("reads back");
    let after_two = gpu.pool_textures;
    let five = gpu
        .probe(&vec![blur; 5], colour, 64, 0.0)
        .expect("reads back");
    assert_eq!(gpu.pool_textures, after_two, "five blurs claimed more");
    for got in [two, five] {
        assert!(near(got, colour, 0.004), "{got:?}");
    }
    // At the default radius, ten: across at half the width, then down at
    // half of both.
    assert!(gpu.pool.contains_key(&(32, 64)) && gpu.pool.contains_key(&(32, 32)));
}

/// A look made from a table reads it in the space it was made for and
/// carries a level past the table's ends on past them: through the
/// identity, a highlight four times white comes out as it went in, read in
/// the display encoding or in log. And a table's values come through finer
/// than eight bits.
#[test]
fn a_table_is_read_in_its_space_and_carried_past_its_ends() {
    let Some(mut gpu) = gpu() else { return };
    let look = |space: &str, lut: Lut| {
        let manifest = concat_effects::Manifest::parse(&format!(
            "format = 2\n[effect]\nid = \"test.look\"\nname = \"Look\"\nkind = \"filter\"\n\
             [wgsl]\nentry = \"effect.wgsl\"\nspace = \"{space}\"\n"
        ))
        .expect("a manifest");
        concat_effects::Shader::compile(&manifest, concat_effects::looks::SHADER)
            .expect("compiles")
            .pass(&BTreeMap::new(), &[], 1.0, Some(Arc::new(lut)), None)
    };
    for space in ["display", "log"] {
        let identity = look(space, Lut::identity(33));
        for colour in [[4.0, 0.5, 0.02, 1.0], [0.18, 0.18, 0.18, 1.0]] {
            let got = gpu
                .probe(std::slice::from_ref(&identity), colour, 8, 0.0)
                .expect("reads back");
            assert!(
                near(got, colour, 0.01),
                "{space}: {colour:?} came out {got:?}"
            );
        }
    }
    // One colour everywhere, a value eight bits would round to 31/255.
    let flat = Lut::from_rgb(2, &[0.123_456, 0.5, 0.25].repeat(8)).expect("a table");
    let got = gpu
        .probe(&[look("display", flat)], [0.3, 0.3, 0.3, 1.0], 8, 0.0)
        .expect("reads back");
    let want = [0.123_456f32, 0.5, 0.25].map(|level| level.powf(2.4));
    assert!(
        near(got, [want[0], want[1], want[2], 1.0], 0.004),
        "{got:?} against {want:?}"
    );
}

/// The motion blur - a coarse Gaussian along its line, then straight lines
/// drawn between its samples - is the Gaussian of sigma `length` along the
/// line summed directly, over the busy picture: along the rows, down the
/// columns and at a slant, from its shortest streak to its longest.
#[test]
fn a_motion_blur_is_a_gaussian_along_its_line() {
    const PEAK: f64 = 6.0;
    let Some(mut gpu) = gpu() else { return };
    let blur = concat_effects::Catalogue::builtin()
        .get("concat.motion-blur")
        .expect("a built-in");
    let size = (192, 108);
    let picture = busy(size, PEAK as f32);
    for (length, angle) in [
        (2.0, 0.0),
        (5.0, 0.0),
        (18.0, 0.0),
        (60.0, 0.0),
        (18.0, 90.0),
        (12.0, 30.0),
    ] {
        let pass = blur
            .pass(
                &BTreeMap::from([("length".to_owned(), length), ("angle".to_owned(), angle)]),
                None,
            )
            .expect("a shader");
        let got = treated(&mut gpu, size, &picture, &[pass]);
        let (sin, cos) = f64::to_radians(angle).sin_cos();
        let step = (length / 16.0).min(0.25);
        let reach = (5.0 * length / step).ceil() as i64;
        let streak = |x: i64, y: i64| {
            let (mut sum, mut total) = ([0.0f64; 4], 0.0f64);
            for k in -reach..=reach {
                let t = k as f64 * step;
                let weight = (-t * t / (2.0 * length * length)).exp();
                let seen =
                    premultiplied_at(&picture, size, (x as f64 + t * cos, y as f64 - t * sin));
                for (into, value) in sum.iter_mut().zip(seen) {
                    *into += weight * value;
                }
                total += weight;
            }
            sum.map(|value| value / total)
        };
        let (worst, at, rms) = measured(&got, size, 3, |x, y| Some(streak(x, y)));
        eprintln!("length {length} at {angle}°: worst {worst:.5} at {at:?}, rms {rms:.6}");
        assert!(
            worst < 0.01 * PEAK,
            "length {length} at {angle}°: {worst} off at {at:?}"
        );
        assert!(
            rms < 0.0015 * PEAK,
            "length {length} at {angle}°: rms {rms}"
        );
    }
}

/// The zoom blur - three passes of sixteen scales - is the four thousand
/// and ninety-six scales of the picture about its centre summed directly,
/// about the middle and about a corner: to the half float over a ramp,
/// where the passes compose exactly, and over the busy picture as near as
/// passes that read each other between pixels come - they soften a streak
/// across its line by about half a pixel, which shows only where a streak
/// grazes a dot six times white.
#[test]
fn a_zoom_blur_is_its_scales_summed() {
    const PEAK: f64 = 6.0;
    let Some(mut gpu) = gpu() else { return };
    let zoom = concat_effects::Catalogue::builtin()
        .get("concat.zoom-blur")
        .expect("a built-in");
    let size = (192, 108);
    let ramp: Vec<[f32; 4]> = (0..size.1)
        .flat_map(|y| {
            (0..size.0).map(move |x| [x as f32 / size.0 as f32, 0.5, y as f32 / size.1 as f32, 1.0])
        })
        .collect();
    for (picture, worst_of, rms_of) in [
        (ramp, 0.002, 0.0005),
        (busy(size, PEAK as f32), 0.025 * PEAK, 0.002 * PEAK),
    ] {
        for (amount, centre) in [
            (25.0, (50.0, 50.0)),
            (100.0, (50.0, 50.0)),
            (60.0, (10.0, 90.0)),
        ] {
            let pass = zoom
                .pass(
                    &BTreeMap::from([
                        ("amount".to_owned(), amount),
                        ("x".to_owned(), centre.0),
                        ("y".to_owned(), centre.1),
                    ]),
                    None,
                )
                .expect("a shader");
            let got = treated(&mut gpu, size, &picture, &[pass]);
            let span = -(1.0 - f64::min(amount / 100.0, 0.99)).ln();
            let (cx, cy) = (centre.0 / 100.0, centre.1 / 100.0);
            let (width, height) = (f64::from(size.0), f64::from(size.1));
            let (worst, at, rms) = measured(&got, size, 4, |x, y| {
                let (u, v) = ((x as f64 + 0.5) / width, (y as f64 + 0.5) / height);
                let mut sum = [0.0f64; 4];
                for k in 0..4096 {
                    let scale = (-span * f64::from(k) / 4096.0).exp();
                    let (su, sv) = (cx + (u - cx) * scale, cy + (v - cy) * scale);
                    let seen =
                        premultiplied_at(&picture, size, (su * width - 0.5, sv * height - 0.5));
                    for (into, value) in sum.iter_mut().zip(seen) {
                        *into += value / 4096.0;
                    }
                }
                Some(sum)
            });
            eprintln!("zoom {amount} about {centre:?}: worst {worst:.5} at {at:?}, rms {rms:.6}");
            assert!(worst < worst_of, "zoom {amount}: {worst} off at {at:?}");
            assert!(rms < rms_of, "zoom {amount}: rms {rms}");
        }
    }
}

/// The fisheye draws every pixel of the frame from somewhere inside it, at
/// full strength and at any shape of frame - no black holes, as earlier
/// builds drew - and pulls the picture in towards the middle without ever
/// folding it back: a ramp across the frame still climbs from left to
/// right along the middle row, the middle and the corners where they were.
#[test]
fn the_fisheye_fills_the_frame() {
    let Some(mut gpu) = gpu() else { return };
    let fisheye = concat_effects::Catalogue::builtin()
        .get("concat.fisheye")
        .expect("a built-in");
    let pass = fisheye
        .pass(&BTreeMap::from([("strength".to_owned(), 100.0)]), None)
        .expect("a shader");
    for size in [(96, 54), (54, 96), (64, 64)] {
        let flat = vec![[0.3f32, 0.5, 0.7, 1.0]; (size.0 * size.1) as usize];
        let got = treated(&mut gpu, size, &flat, std::slice::from_ref(&pass));
        for (n, pixel) in got.iter().enumerate() {
            assert!(
                near(*pixel, [0.3, 0.5, 0.7, 1.0], 0.004),
                "{size:?}: pixel {n} is {pixel:?}"
            );
        }
        let ramp: Vec<[f32; 4]> = (0..size.1)
            .flat_map(|_| (0..size.0).map(|x| [(x as f32 + 0.5) / size.0 as f32, 0.0, 0.0, 1.0]))
            .collect();
        let got = treated(&mut gpu, size, &ramp, std::slice::from_ref(&pass));
        let row = &got[(size.1 / 2 * size.0) as usize..][..size.0 as usize];
        assert!(
            row.windows(2).all(|pair| pair[1][0] >= pair[0][0]),
            "{size:?}: the middle row folds back"
        );
        let quarter = row[(size.0 / 4) as usize][0];
        assert!(
            quarter > 0.3 && quarter < 0.5,
            "{size:?}: a quarter of the way across reads {quarter}, not pulled towards the middle"
        );
        let corner = got[0][0];
        assert!(
            (corner - ramp[0][0]).abs() < 0.01,
            "{size:?}: the corner moved to {corner}"
        );
    }
}

/// The software-only constructor opens the software adapter and nothing
/// else; a machine without one - a Mac - gets None rather than a GPU.
#[test]
fn the_software_compositor_draws_on_a_software_adapter_or_not_at_all() {
    if let Some(gpu) = WgpuCompositor::software() {
        assert_eq!(gpu.adapter_info().device_type, wgpu::DeviceType::Cpu);
    }
}

/// No wait on the device is without an end: an export whose card stopped
/// answering sat in one for good (#297). Each goes through `wait`, which
/// gives up after `DEVICE_WAIT`.
#[test]
fn no_wait_on_the_device_is_unbounded() {
    let source = include_str!("../gpu.rs");
    assert!(
        !source.contains(concat!("wait_", "indefinitely")),
        "every poll is bounded by DEVICE_WAIT"
    );
    assert!(DEVICE_WAIT >= std::time::Duration::from_secs(10));
    assert!(DEVICE_WAIT <= std::time::Duration::from_secs(60));
}

/// A frame the device refuses part of - here a layer wider than any
/// texture it can make - is a failed frame, not a wrong one handed on:
/// the compositor says it is lost, and the export draws the frame again
/// on another device or stops with the reason.
#[test]
fn a_frame_the_device_refuses_is_a_failed_frame() {
    let Some(mut gpu) = gpu() else { return };
    let past = gpu.device.limits().max_texture_dimension_2d + 16;
    let plan = plan(16, 16, vec![layer(solid(past, 2, [255, 0, 0, 255]))]);
    let frame = gpu.render(&plan);
    assert!(gpu.lost(), "the refused frame is reported");
    assert_eq!(frame.pixel(8, 8), Some([0, 0, 0, 255]));
}

/// The device is asked for no more than its adapter has: an OpenGL ES
/// phone's limits are under WebGPU's defaults, and asking for those opened
/// nothing there (#312). A desktop adapter still gets WebGPU's defaults
/// and its own texture size.
#[test]
fn the_device_asks_for_what_the_adapter_has() {
    let phone = wgpu::Limits {
        max_texture_dimension_2d: 4096,
        ..wgpu::Limits::downlevel_webgl2_defaults()
    };
    let asked = limits_for(&phone);
    assert!(asked.check_limits(&phone), "the phone opens a device");
    assert_eq!(asked.max_texture_dimension_2d, 4096);

    let desktop = wgpu::Limits {
        max_texture_dimension_2d: 16384,
        ..wgpu::Limits::default()
    };
    let asked = limits_for(&desktop);
    assert!(asked.check_limits(&desktop));
    assert!(wgpu::Limits::default().check_limits(&asked));
    assert_eq!(asked.max_texture_dimension_2d, 16384);
}

/// On a device that cannot draw in half floats the eight-bit fallback
/// draws the same picture, to within its rounding: an opaque gradient
/// with a translucent layer blended over it.
#[test]
fn the_eight_bit_working_format_draws_the_same_picture() {
    let Some(mut gpu) = gpu() else { return };
    let mut fallback =
        WgpuCompositor::drawing_in(gpu.device.clone(), gpu.queue.clone(), WORK_FALLBACK);
    let mut top = layer(solid(16, 16, [20, 200, 90, 140]));
    top.transform = Transform {
        offset_x: 0.25,
        ..Transform::default()
    };
    let plan = plan(32, 32, vec![layer(gradient(32, 32)), top]);
    let expected = gpu.render(&plan);
    let actual = fallback.render(&plan);
    assert!(!fallback.lost());
    for (want, got) in expected.pixels().iter().zip(actual.pixels()) {
        assert!(
            (i32::from(*want) - i32::from(*got)).abs() <= 2,
            "half floats {want} vs eight bits {got}"
        );
    }
    assert_eq!(fallback.probe(&[], [0.5; 4], 4, 0.0), None);
}

/// A frame bigger than the device's textures is refused up front, at the
/// document's ceiling or the device's, whichever is lower.
#[test]
fn the_largest_frame_is_what_the_device_holds() {
    let Some(gpu) = gpu() else { return };
    assert_eq!(
        gpu.max_side(),
        concat_core::frame::MAX_SIDE.min(gpu.device.limits().max_texture_dimension_2d)
    );
}

// ── backgrounds ──

/// A wide picture, red on its left half and blue on its right, the way a
/// landscape shot sits in a vertical frame.
fn halves(width: u32, height: u32) -> Frame {
    let mut frame = Frame::transparent(width, height);
    for y in 0..height {
        for x in 0..width {
            let colour = if x < width / 2 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            frame.set_pixel(x, y, colour);
        }
    }
    frame
}

/// The standard normal's distribution function, by Abramowitz and
/// Stegun's 7.1.26: to a few parts in ten million, far finer than a
/// frame's eight bits.
fn normal_cdf(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

/// An output level back to light, the inverse of the resolve's 1/2.4.
fn light(level: u8) -> f64 {
    (f64::from(level) / 255.0).powf(2.4)
}

/// A blurred background is the picture enlarged to cover the frame and
/// blurred by the spread asked for, beneath the picture and nowhere over
/// it: in the bars of a wide red-and-blue picture in a tall frame, the
/// light along a row is the step from red to blue at the frame's middle,
/// blurred by a Gaussian of `sigma` output pixels, while the picture
/// itself comes out exactly as it does over black.
#[test]
fn a_blurred_background_is_the_picture_covering_the_frame_blurred() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let blur = concat_effects::Catalogue::builtin()
        .get("concat.gaussian-blur")
        .expect("a built-in");
    let (width, height) = (90, 160);
    let sigma = 12.0;
    let picture = layer(halves(64, 36));
    let backdrop = picture
        .backdrop(&Backdrop::Blur { sigma }, width, height, |radius| {
            blur.pass(
                &BTreeMap::from([("radius".to_owned(), f64::from(radius))]),
                None,
            )
        })
        .expect("a picture to blur");
    assert_eq!(backdrop.effects.len(), 1, "the blur, and only the blur");

    let alone = gpu.render(&plan(width, height, vec![picture.clone()]));
    let over = gpu.render(&plan(width, height, vec![backdrop, picture]));

    // 64x36 fitted into 90 across is 90x51, from row 54: the picture.
    for y in 58..100 {
        for x in 0..width {
            assert_eq!(
                over.pixel(x, y),
                alone.pixel(x, y),
                "the picture at {x},{y}"
            );
        }
    }
    // Above and below it, the bars: black alone, the blurred step over it.
    let mut worst = 0.0f64;
    for y in [10, 30, 130, 150] {
        assert_eq!(alone.pixel(45, y), Some([0, 0, 0, 255]), "a bar alone");
        for x in 0..width {
            let blue = normal_cdf((f64::from(x) + 0.5 - f64::from(width) / 2.0) / f64::from(sigma));
            let [r, g, b, a] = over.pixel(x, y).expect("in the frame");
            assert_eq!((g, a), (0, 255), "at {x},{y}");
            for (got, want) in [(light(r), 1.0 - blue), (light(b), blue)] {
                worst = worst.max((got - want).abs());
                assert!(
                    (got - want).abs() < 0.02,
                    "at {x},{y}: light {got:.4} where the blurred step is {want:.4}"
                );
            }
        }
    }
    eprintln!("blurred background: worst {worst:.4} of light off the Gaussian step");
}

/// A colour background fills the bars with the colour, exactly, and a
/// still covers the frame in its own shape: a tall still behind a wide
/// picture in a wide frame fills the bars from its middle band.
#[test]
fn a_colour_or_a_still_fills_the_bars() {
    let Some(mut gpu) = gpu() else { return };
    let (width, height) = (90, 160);
    let picture = layer(halves(64, 36));
    let none = |_| None;
    let colour = picture
        .backdrop(&Backdrop::Colour([10, 200, 90]), width, height, none)
        .expect("a colour");
    let frame = gpu.render(&plan(width, height, vec![colour, picture.clone()]));
    for (x, y) in [(0, 0), (89, 0), (45, 20), (0, 159), (89, 159), (45, 140)] {
        assert_eq!(frame.pixel(x, y), Some([10, 200, 90, 255]), "at {x},{y}");
    }
    assert_eq!(frame.pixel(20, 80), Some([255, 0, 0, 255]));
    assert_eq!(frame.pixel(70, 80), Some([0, 0, 255, 255]));

    // A still of three bands, top to bottom, twice as tall as it is wide:
    // covering the 90x160 frame it is 90 across and 180 down, so the frame
    // shows the middle band whole and the outer two cut short.
    let mut still = Frame::transparent(20, 40);
    for y in 0..40 {
        for x in 0..20 {
            let band = match y {
                0..10 => [255, 255, 0, 255],
                10..30 => [0, 255, 255, 255],
                _ => [255, 0, 255, 255],
            };
            still.set_pixel(x, y, band);
        }
    }
    let image = picture
        .backdrop(&Backdrop::Image(Arc::new(still)), width, height, none)
        .expect("a still");
    let frame = gpu.render(&plan(width, height, vec![image, picture]));
    assert_eq!(frame.pixel(45, 40), Some([0, 255, 255, 255]));
    assert_eq!(frame.pixel(45, 120), Some([0, 255, 255, 255]));
    assert_eq!(frame.pixel(45, 2), Some([255, 255, 0, 255]));
    assert_eq!(frame.pixel(45, 157), Some([255, 0, 255, 255]));
}

/// The background goes with its picture through a wipe: halfway, the half
/// of the frame the picture has gone from is black, bars and all, and the
/// edge crosses the bars where it crosses the picture.
#[test]
fn a_background_is_wiped_with_its_picture() {
    let Some(mut gpu) = gpu() else { return };
    let (width, height) = (90, 160);
    let mut picture = layer(halves(64, 36));
    picture.transitions = vec![Transition::Wipe {
        uncovered: 0.5,
        from_right: false,
    }];
    let colour = picture
        .backdrop(&Backdrop::Colour([0, 255, 0]), width, height, |_| None)
        .expect("a colour");
    let frame = gpu.render(&plan(width, height, vec![colour, picture]));
    assert_eq!(
        frame.pixel(20, 80),
        Some([255, 0, 0, 255]),
        "the picture's kept half"
    );
    assert_eq!(
        frame.pixel(70, 80),
        Some([0, 0, 0, 255]),
        "the picture's wiped half"
    );
    assert_eq!(
        frame.pixel(20, 20),
        Some([0, 255, 0, 255]),
        "the bar's kept half"
    );
    assert_eq!(
        frame.pixel(70, 20),
        Some([0, 0, 0, 255]),
        "the bar's wiped half"
    );
    assert_eq!(
        frame.pixel(43, 20),
        Some([0, 255, 0, 255]),
        "the edge is the picture's"
    );
    assert_eq!(
        frame.pixel(47, 20),
        Some([0, 0, 0, 255]),
        "the edge is the picture's"
    );
}

/// Light from the display encoding a look is written in (BT.1886's 2.4).
fn lit(display: [f32; 3]) -> [f32; 4] {
    [
        display[0].powf(2.4),
        display[1].powf(2.4),
        display[2].powf(2.4),
        1.0,
    ]
}

/// The display encoding of `rgb`, the space the colour packages work in.
fn shown(rgb: [f32; 4]) -> [f32; 3] {
    [
        rgb[0].max(0.0).powf(1.0 / 2.4),
        rgb[1].max(0.0).powf(1.0 / 2.4),
        rgb[2].max(0.0).powf(1.0 / 2.4),
    ]
}

/// The fully saturated colour at `hue` degrees, at HSV saturation `s` and
/// value `v`, in the display encoding.
fn hsv(hue: f32, s: f32, v: f32) -> [f32; 3] {
    let h = (hue.rem_euclid(360.0)) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [r, g, b].map(|c| v * (1.0 - s + s * c))
}

/// Hue in degrees and chroma of a colour in the display encoding.
fn hue_and_chroma(rgb: [f32; 3]) -> (f32, f32) {
    let mx = rgb[0].max(rgb[1]).max(rgb[2]);
    let mn = rgb[0].min(rgb[1]).min(rgb[2]);
    let d = mx - mn;
    if d < 1e-6 {
        return (0.0, 0.0);
    }
    let h = if mx == rgb[0] {
        (rgb[1] - rgb[2]) / d
    } else if mx == rgb[1] {
        2.0 + (rgb[2] - rgb[0]) / d
    } else {
        4.0 + (rgb[0] - rgb[1]) / d
    };
    ((h / 6.0).rem_euclid(1.0) * 360.0, d)
}

fn moved(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)
}

const BANDS: [(&str, f32); 8] = [
    ("red", 0.0),
    ("orange", 30.0),
    ("yellow", 60.0),
    ("green", 120.0),
    ("cyan", 180.0),
    ("blue", 240.0),
    ("purple", 270.0),
    ("magenta", 300.0),
];

/// Every band's saturation slider at -100, across the whole wheel: the
/// band's own centre turns grey, its neighbours' centres are untouched,
/// and the move fades from one to the other without a step.
#[test]
fn every_hsl_band_owns_its_centre_and_hands_over_smoothly() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let size = (720, 1);
    let at = |hue: f32| ((hue.rem_euclid(360.0)) * 2.0).round() as usize;
    let picture: Vec<[f32; 4]> = (0..size.0)
        .map(|x| lit(hsv(x as f32 / 2.0, 0.8, 0.8)))
        .collect();
    for (n, (band, centre)) in BANDS.iter().enumerate() {
        let pass = hsl
            .pass(&BTreeMap::from([(format!("{band}_sat"), -100.0)]), None)
            .expect("a shader");
        let got = treated(&mut gpu, size, &picture, &[pass]);
        let (_, left) = hue_and_chroma(shown(got[at(*centre)]));
        assert!(left < 0.01, "{band} at its centre kept chroma {left}");
        for neighbour in [BANDS[(n + 7) % 8].1, BANDS[(n + 1) % 8].1] {
            let i = at(neighbour);
            assert!(
                moved(got[i], picture[i]) < 1e-3,
                "{band} moved its neighbour at {neighbour}"
            );
        }
        // No step between two hues half a degree apart.
        let amounts: Vec<f32> = got
            .iter()
            .zip(&picture)
            .map(|(a, b)| moved(*a, *b))
            .collect();
        for x in 0..amounts.len() {
            let step = (amounts[x] - amounts[(x + 1) % amounts.len()]).abs();
            assert!(
                step < 0.02,
                "{band}: a seam of {step} at {}",
                x as f32 / 2.0
            );
        }
    }
}

/// Every band's hue slider turns it towards the next band round the wheel
/// at plus, and back at minus: the full 30 degrees away from a neighbour,
/// and towards one 60 degrees off, but at most 24 towards a neighbour only
/// 30 degrees off, so the wheel is never folded - turning away from a
/// neighbour stretches the colours between and is never held back.
#[test]
fn hsl_hue_turns_towards_the_next_band() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let mut failures = Vec::new();
    for (n, (band, centre)) in BANDS.iter().enumerate() {
        let picture = vec![lit(hsv(*centre, 0.8, 0.8))];
        let gap = |m: usize| {
            (BANDS[m].1 - centre)
                .rem_euclid(360.0)
                .min(360.0 - (BANDS[m].1 - centre).rem_euclid(360.0))
        };
        for slider in [100.0f64, -100.0] {
            // The neighbour the turn heads for: 24 if it is 30 off, else 30.
            let towards = if slider > 0.0 {
                (n + 1) % 8
            } else {
                (n + 7) % 8
            };
            let reach = if gap(towards) < 45.0 { 24.0 } else { 30.0 };
            let pass = hsl
                .pass(&BTreeMap::from([(format!("{band}_hue"), slider)]), None)
                .expect("a shader");
            let got = treated(&mut gpu, (1, 1), &picture, &[pass]);
            let (hue, _) = hue_and_chroma(shown(got[0]));
            let turned = (hue - centre + 540.0).rem_euclid(360.0) - 180.0;
            let want = reach * (slider as f32 / 100.0);
            eprintln!("{band} at {slider}: turned {turned:.2} degrees, want {want}");
            if (turned - want).abs() >= 1.0 {
                failures.push(format!(
                    "{band} at {slider}: turned {turned} degrees, not {want}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Every band's hue slider at either end, across the whole wheel: the
/// output hue only ever moves forward as the input hue does, so a gradient
/// through a turned band stretches or squeezes but never folds back - two
/// neighbours at opposite ends included.
#[test]
fn hsl_hue_never_folds_the_wheel() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let size = (720, 1);
    let picture: Vec<[f32; 4]> = (0..size.0)
        .map(|x| lit(hsv(x as f32 / 2.0, 0.8, 0.8)))
        .collect();
    let mut settings: Vec<BTreeMap<String, f64>> = Vec::new();
    for (band, _) in BANDS {
        for end in [-100.0, 100.0] {
            settings.push(BTreeMap::from([(format!("{band}_hue"), end)]));
        }
    }
    for n in 0..8 {
        // A chain of three at its worst - hard one way, a little, hard the
        // other way - where a pair comes back to about 0.87 of its gap, and
        // the mirror of it.
        for sign in [1.0, -1.0] {
            settings.push(BTreeMap::from([
                (format!("{}_hue", BANDS[n].0), 100.0 * sign),
                (format!("{}_hue", BANDS[(n + 1) % 8].0), 20.0 * sign),
                (format!("{}_hue", BANDS[(n + 2) % 8].0), -100.0 * sign),
            ]));
        }
        settings.push(BTreeMap::from([
            (format!("{}_hue", BANDS[n].0), 100.0),
            (format!("{}_hue", BANDS[(n + 1) % 8].0), -100.0),
        ]));
        settings.push(BTreeMap::from([
            (format!("{}_hue", BANDS[n].0), -100.0),
            (format!("{}_hue", BANDS[(n + 1) % 8].0), 100.0),
        ]));
    }
    for values in settings {
        let pass = hsl.pass(&values, None).expect("a shader");
        let got = treated(&mut gpu, size, &picture, &[pass]);
        let hues: Vec<f32> = got.iter().map(|px| hue_and_chroma(shown(*px)).0).collect();
        for x in 0..hues.len() {
            let next = hues[(x + 1) % hues.len()];
            let step = (next - hues[x] + 540.0).rem_euclid(360.0) - 180.0;
            assert!(
                step > 0.0,
                "{values:?}: hue folds back by {step} at {}",
                x as f32 / 2.0
            );
        }
    }
}

/// Every band at saturation -100 together turns the whole wheel grey,
/// whatever the hue: the bands' weights sum to one everywhere.
#[test]
fn hsl_bands_together_cover_the_whole_wheel() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let size = (720, 1);
    let picture: Vec<[f32; 4]> = (0..size.0)
        .map(|x| lit(hsv(x as f32 / 2.0, 0.8, 0.8)))
        .collect();
    let values: BTreeMap<String, f64> = BANDS
        .iter()
        .map(|(band, _)| (format!("{band}_sat"), -100.0))
        .collect();
    let pass = hsl.pass(&values, None).expect("a shader");
    let got = treated(&mut gpu, size, &picture, &[pass]);
    for (x, px) in got.iter().enumerate() {
        let (_, chroma) = hue_and_chroma(shown(*px));
        assert!(chroma < 0.01, "hue {} kept chroma {chroma}", x as f32 / 2.0);
    }
}

/// Every band's luminance at either end lifts or lowers every colour by
/// a real share of the room it has - a sky and a yellow already near
/// white included - and moves smoothly round the wheel.
#[test]
fn hsl_luminance_moves_every_colour_both_ways() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let colours = [
        ("sky", [0.45, 0.7, 0.95]),
        ("deep blue", [0.1, 0.2, 0.8]),
        ("yellow", [0.95, 0.9, 0.2]),
        ("pale skin", [0.9, 0.7, 0.6]),
        ("leaf", [0.3, 0.6, 0.2]),
        ("pure red at white", [1.0, 0.1, 0.1]),
    ];
    let picture: Vec<[f32; 4]> = colours.iter().map(|(_, c)| lit(*c)).collect();
    let all = |end: f64| -> BTreeMap<String, f64> {
        BANDS
            .iter()
            .map(|(band, _)| (format!("{band}_lum"), end))
            .collect()
    };
    let bright = |px: [f32; 4]| {
        let d = shown(px);
        0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]
    };
    for end in [100.0, -100.0] {
        let pass = hsl.pass(&all(end), None).expect("a shader");
        let got = treated(&mut gpu, (colours.len() as u32, 1), &picture, &[pass]);
        for (i, (name, _)) in colours.iter().enumerate() {
            let (before, after) = (bright(picture[i]), bright(got[i]));
            eprintln!("{name} at {end}: {before:.3} -> {after:.3}");
            if end > 0.0 {
                assert!(
                    after - before > 0.3 * (1.0 - before),
                    "{name} barely lifted: {before} -> {after}"
                );
            } else {
                assert!(
                    before - after > 0.3 * before,
                    "{name} barely lowered: {before} -> {after}"
                );
            }
        }
    }
    let size = (720, 1);
    let wheel: Vec<[f32; 4]> = (0..size.0)
        .map(|x| lit(hsv(x as f32 / 2.0, 0.8, 0.8)))
        .collect();
    let pass = hsl.pass(&all(100.0), None).expect("a shader");
    let got = treated(&mut gpu, size, &wheel, &[pass]);
    for x in 0..got.len() {
        let step = moved(got[x], got[(x + 1) % got.len()]);
        assert!(step < 0.03, "a step of {step} at {}", x as f32 / 2.0);
    }
}

/// Luminance at its most across a run of oranges whose brightest channel
/// is already past white and whose brightness crosses white: no channel
/// is lowered and neighbours never jump apart.
#[test]
fn hsl_luminance_is_smooth_past_white() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let size = (200, 1);
    let ramp: Vec<[f32; 4]> = (0..size.0)
        .map(|x| {
            let r = 1.5 + 0.6 * x as f32 / size.0 as f32;
            lit([r, r * 0.455, r * 0.214])
        })
        .collect();
    let values: BTreeMap<String, f64> = BANDS
        .iter()
        .map(|(band, _)| (format!("{band}_lum"), 100.0))
        .collect();
    let pass = hsl.pass(&values, None).expect("a shader");
    let got = treated(&mut gpu, size, &ramp, &[pass]);
    for x in 0..got.len() {
        let (before, after) = (shown(ramp[x]), shown(got[x]));
        for c in 0..3 {
            assert!(
                after[c] >= before[c] - 1e-3,
                "channel {c} lowered at {x}: {before:?} -> {after:?}"
            );
        }
        if x + 1 < got.len() {
            let step = moved(got[x], got[x + 1]);
            assert!(step < 0.05, "a jump of {step} at {x}");
        }
    }
}

/// Greys stay grey, and nothing comes out negative or not a number, with
/// every slider at either end: grey, near-grey, black, white past 1.0, a
/// bright colour past 1.0, and a half-transparent pixel.
#[test]
fn hsl_at_its_ends_keeps_greys_and_stays_finite() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let picture = vec![
        [0.2, 0.2, 0.2, 1.0],
        [0.2, 0.2, 0.205, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [3.0, 3.0, 3.0, 1.0],
        [4.0, 0.6, 0.1, 1.0],
        [0.3, 0.2, 0.1, 0.5],
    ];
    for end in [-100.0, 100.0] {
        let values: BTreeMap<String, f64> = BANDS
            .iter()
            .flat_map(|(band, _)| ["hue", "sat", "lum"].map(|p| (format!("{band}_{p}"), end)))
            .collect();
        let pass = hsl.pass(&values, None).expect("a shader");
        let got = treated(&mut gpu, (6, 1), &picture, &[pass]);
        for (i, px) in got.iter().enumerate() {
            assert!(
                px.iter().all(|c| c.is_finite() && *c >= 0.0),
                "pixel {i} at {end}: {px:?}"
            );
        }
        for i in [0, 1, 2, 3] {
            assert!(
                moved(got[i], picture[i]) < 2e-3 * picture[i][0].max(1.0),
                "grey {i} moved at {end}"
            );
        }
        assert!((got[5][3] - 0.5).abs() < 1e-3, "alpha changed at {end}");
    }
}

/// A face's worth of test cases side by side, in light, drawn `k` times over: a
/// beige wall within a few percent of the skin's brightness with writing on it;
/// skin with pores, a small dark spot, a red spot of acne, a larger brown spot,
/// lips, a ruddy cheek with lips inside it, a one and a two pixel dark line,
/// light and dark brown hair strands one and two pixels wide, a brow, a grey-
/// white eye and a warm-white hazel one; the jaw against a dark background,
/// rosy lips on their own, black hair with strands, and a grainy sky that fills
/// the rest of the frame.
fn portrait(size: (u32, u32), k: u32) -> Vec<[f32; 4]> {
    (0..size.1)
        .flat_map(|y| (0..size.0).map(move |x| (x, y)))
        .map(|(x, y)| {
            let (x, y) = (x / k, y / k);
            let pore = if (x * 7 + y * 13) % 5 < 2 {
                0.03
            } else {
                -0.02
            };
            let strand = if x % 2 == 0 { 0.004 } else { 0.0 };
            let within = |xs: std::ops::Range<u32>, ys: std::ops::Range<u32>| {
                xs.contains(&x) && ys.contains(&y)
            };
            let sky = [0.05 + pore * 0.5, 0.1 + pore, 0.4 + pore, 1.0];
            if y >= 112 {
                return sky;
            }
            match x {
                _ if within(3..5, 40..60) => lit([0.05, 0.05, 0.05]),
                _ if within(18..21, 40..43) => [0.33, 0.13, 0.05, 1.0],
                _ if within(30..33, 40..43) => lit([0.62, 0.32, 0.27]),
                _ if within(18..27, 48..57) => lit([0.55, 0.40, 0.31]),
                _ if within(33..45, 46..60) => lit([0.75, 0.43, 0.38]),
                _ if within(11..21, 86..96) => lit([0.75, 0.43, 0.38]),
                _ if within(28..48, 84..85) => lit([0.55, 0.38, 0.28]),
                _ if within(28..48, 88..89) || within(28..48, 92..94) => lit([0.45, 0.28, 0.20]),
                _ if within(30..44, 98..104) => lit([0.38, 0.26, 0.19]),
                _ if within(36..38, 107..109) => lit([0.08, 0.07, 0.06]),
                _ if within(34..40, 106..110) => lit([0.45, 0.35, 0.2]),
                _ if within(30..44, 105..111) => lit([0.90, 0.85, 0.80]),
                _ if within(8..24, 62..112) => {
                    let [r, g, b, a] = lit([0.80, 0.52, 0.47]);
                    [r + pore, g + pore, b + pore, a]
                }
                _ if within(28..48, 62..63) || within(28..48, 66..68) => lit([0.08, 0.07, 0.07]),
                _ if within(35..39, 73..77) => lit([0.2, 0.15, 0.12]),
                _ if within(30..44, 72..78) => lit([0.92, 0.92, 0.9]),
                0..8 => lit([0.70, 0.60, 0.50]),
                8..48 => [0.45 + pore, 0.25 + pore, 0.15 + pore, 1.0],
                48..56 => [0.02, 0.02, 0.02, 1.0],
                56..64 => lit([0.75, 0.42, 0.38]),
                64..72 => [0.004 + strand, 0.003 + strand, 0.002 + strand, 1.0],
                _ => sky,
            }
        })
        .collect()
}

/// The spread of luminance over a block of the portrait, in its own
/// pixels, drawn `k` times over.
fn spread(
    picture: &[[f32; 4]],
    width: u32,
    k: u32,
    xs: std::ops::Range<u32>,
    ys: std::ops::Range<u32>,
) -> f32 {
    let lumas: Vec<f32> = (ys.start * k..ys.end * k)
        .flat_map(|y| (xs.start * k..xs.end * k).map(move |x| (x, y)))
        .map(|(x, y)| {
            let [r, g, b, _] = picture[(y * width + x) as usize];
            0.2126 * r + 0.7152 * g + 0.0722 * b
        })
        .collect();
    let mean = lumas.iter().sum::<f32>() / lumas.len() as f32;
    (lumas.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / lumas.len() as f32).sqrt()
}

/// Beauty at `smoothing` and `whiten` over the portrait drawn `k` times
/// over, in a frame 1080 lines tall at `k` 1: the source and the result.
/// Beauty's reach follows the frame's shorter side, so here and in the
/// tests below a frame "1080 lines tall" is at least as wide as it is tall:
/// the picture is laid at its top left, the rest transparent.
fn beauty(
    gpu: &mut WgpuCompositor,
    k: u32,
    smoothing: f64,
    whiten: f64,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    beauty_in(gpu, k, 1080 * k, smoothing, whiten)
}

/// Beauty as `beauty`, in a frame `height` lines tall and as wide.
fn beauty_in(
    gpu: &mut WgpuCompositor,
    k: u32,
    height: u32,
    smoothing: f64,
    whiten: f64,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    beauty_framed(gpu, k, ((96 * k).max(height), height), smoothing, whiten)
}

/// Beauty as `beauty`, over the portrait as tall as `frame`, laid at the
/// top left of a frame that size.
fn beauty_framed(
    gpu: &mut WgpuCompositor,
    k: u32,
    frame: (u32, u32),
    smoothing: f64,
    whiten: f64,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let package = concat_effects::Catalogue::builtin()
        .get("concat.beauty")
        .expect("a built-in");
    let size = (96 * k, frame.1);
    let picture = portrait(size, k);
    let pass = package
        .pass(
            &BTreeMap::from([
                ("smoothing".to_owned(), smoothing),
                ("whiten".to_owned(), whiten),
            ]),
            None,
        )
        .expect("a shader");
    let got = treated_within(gpu, frame, size, &picture, &[pass]);
    (picture, got)
}

/// The portrait's pixel `(x, y)`, drawn `k` times over: the middle of its
/// block.
fn at_k(frame: &[[f32; 4]], k: u32, (x, y): (u32, u32)) -> [f32; 4] {
    frame[((y * k + k / 2) * 96 * k + x * k + k / 2) as usize]
}

/// How far `spot` stands apart from the two points `around` it, as the
/// largest channel difference in the display encoding.
fn standing(frame: &[[f32; 4]], k: u32, spot: (u32, u32), around: [(u32, u32); 2]) -> f32 {
    let (a, b) = (
        shown(at_k(frame, k, around[0])),
        shown(at_k(frame, k, around[1])),
    );
    let s = shown(at_k(frame, k, spot));
    (0..3)
        .map(|c| ((a[c] + b[c]) / 2.0 - s[c]).abs())
        .fold(0.0, f32::max)
}

/// Beauty's smoothing at its most, on a 1080-line frame and on the same
/// portrait drawn twice as large in a 2160-line one, alike: pores even out
/// across open skin, by the jaw, by the wall and on a ruddy cheek; a dark spot,
/// a red spot of acne and a nine pixel brown spot go; and the jaw keeps its
/// edge, neither dragged dark nor bleeding, while lips inside the skin, rosy
/// lips on their own, black hair, the sky and the wall stay as they were, and
/// lips inside the ruddy cheek take no more than a trace.
#[test]
fn beauty_smooths_skin_and_nothing_else() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let lu = |p: [f32; 4]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    for k in [1, 2] {
        let (picture, got) = beauty(&mut gpu, k, 100.0, 0.0);
        let width = 96 * k;
        for (name, xs, ys) in [
            ("open skin", 26..40, 8..32),
            ("skin by the jaw", 42..48, 8..32),
            ("skin by the wall", 8..14, 8..32),
            ("a ruddy cheek", 10..22, 64..78),
        ] {
            let before = spread(&picture, width, k, xs.clone(), ys.clone());
            let after = spread(&got, width, k, xs, ys);
            eprintln!("x{k} {name}: grain {before:.5} -> {after:.5}");
            assert!(
                after < before * 0.6,
                "x{k}: {name} kept its pores: {before} -> {after}"
            );
        }
        for (name, spot, around) in [
            ("dark spot", (19, 41), [(13, 41), (25, 41)]),
            ("acne", (31, 41), [(26, 41), (37, 41)]),
            ("brown spot", (22, 52), [(14, 52), (30, 52)]),
        ] {
            let before = standing(&picture, k, spot, around);
            let after = standing(&got, k, spot, around);
            eprintln!("x{k} {name}: stands out {before:.4} -> {after:.4}");
            assert!(
                after < before * 0.5,
                "x{k}: {name} stayed: {before} -> {after}"
            );
        }
        // The jaw: its last column of skin ends up as bright as the skin
        // beside it once that is evened - not dragged towards the dark -
        // and the dark keeps clear of the skin.
        let evened = (26..40)
            .flat_map(|x| (8..32).map(move |y| (x, y)))
            .map(|p| lu(at_k(&got, k, p)))
            .sum::<f32>()
            / (14.0 * 24.0);
        for y in 8..44 {
            let edge = lu(at_k(&got, k, (47, y)));
            assert!(
                (edge - evened).abs() < 0.02,
                "x{k}: the jaw at row {y} is {edge}, the skin {evened}"
            );
            let dark = moved(at_k(&got, k, (48, y)), at_k(&picture, k, (48, y)));
            assert!(
                dark < 0.005,
                "x{k}: the dark beside the jaw at row {y} took {dark}"
            );
        }
        for y in 8..80 {
            for x in (64..72).chain(74..90) {
                let drift = moved(at_k(&got, k, (x, y)), at_k(&picture, k, (x, y)));
                assert!(drift < 0.01, "x{k}: {x},{y} is not skin and moved {drift}");
            }
            // Lips share hues with a ruddy cheek, so the blur may gather
            // them - but only from lips like themselves.
            for x in 58..62 {
                let drift = moved(at_k(&got, k, (x, y)), at_k(&picture, k, (x, y)));
                assert!(drift < 0.02, "x{k}: the lips at {x},{y} moved {drift}");
            }
            // A wall the very tint of skin counts as skin to a colour
            // mask; the blur's tint weight holds what it takes from the
            // face beside it to a trace.
            for x in (0..3).chain(5..8) {
                let drift = moved(at_k(&got, k, (x, y)), at_k(&picture, k, (x, y)));
                assert!(
                    drift < 0.02,
                    "x{k}: the wall at {x},{y} took {drift} from the face"
                );
            }
        }
        for (x, y) in [(36, 50), (39, 52), (43, 54), (33, 52), (39, 46), (39, 59)] {
            let drift = moved(at_k(&got, k, (x, y)), at_k(&picture, k, (x, y)));
            assert!(drift < 0.02, "x{k}: the lips at {x},{y} moved {drift}");
        }
        // Lips inside a ruddy cheek are a degree of hue from it, told apart
        // only by tint and brightness: at the most they may take a trace.
        for (x, y) in [(16, 91), (11, 91), (20, 91), (16, 86), (16, 95)] {
            let drift = moved(at_k(&got, k, (x, y)), at_k(&picture, k, (x, y)));
            assert!(
                drift < 0.025,
                "x{k}: the lips in the cheek at {x},{y} moved {drift}"
            );
        }
    }
}

/// What smoothing must leave sharp at 30, 50 (the default), 75 and 100, in
/// a 1080-line frame, a 540-line one, a 2160-line one with the portrait
/// drawn twice as large, and an upright 1080 by 1920 one: one and two pixel dark lines, writing on a beige
/// wall, light and dark brown hair strands, a brow, and two eyes - their
/// pupils and irises against their whites, every pixel of the hazel iris
/// and its white barely moved; and up to the default, lips inside a ruddy
/// cheek - a degree of hue from it - moved by no more than a trace. Near-
/// black and grey features keep 85% of their contrast throughout; warm
/// brown ones 85% up to the default and 70% past it.
#[test]
fn beauty_keeps_lines_hair_brows_and_eyes() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let lu = |p: [f32; 4]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    // An upright 1080 by 1920 frame is held to the same as a 1080-line one.
    for (k, frame) in [
        (1, (1080, 1080)),
        (1, (540, 540)),
        (2, (2160, 2160)),
        (1, (1080, 1920)),
    ] {
        for smoothing in [30.0, 50.0, 75.0, 100.0] {
            let (picture, got) = beauty_framed(&mut gpu, k, frame, smoothing, 0.0);
            let at = |frame: &[[f32; 4]], p: (u32, u32)| at_k(frame, k, p);
            let contrast = |frame: &[[f32; 4]], line: (u32, u32), around: [(u32, u32); 2]| {
                (lu(at(frame, around[0])) + lu(at(frame, around[1]))) / 2.0 - lu(at(frame, line))
            };
            let warm = if smoothing > 50.0 { 0.7 } else { 0.85 };
            let tag = format!("{} by {}, {smoothing}", frame.0, frame.1);
            for (name, line, around, least) in [
                ("a one pixel line", (38, 62), [(38, 60), (38, 64)], 0.85),
                ("a two pixel line", (38, 66), [(38, 64), (38, 70)], 0.85),
                ("writing on the wall", (3, 50), [(1, 50), (6, 50)], 0.85),
                ("the grey eye's pupil", (37, 75), [(32, 75), (42, 75)], 0.85),
                ("a light brown strand", (38, 84), [(38, 82), (38, 86)], warm),
                ("a brown strand", (38, 88), [(38, 86), (38, 90)], warm),
                (
                    "a two pixel brown strand",
                    (38, 92),
                    [(38, 90), (38, 95)],
                    warm,
                ),
                ("a brow", (37, 101), [(37, 96), (46, 101)], warm),
                ("the hazel iris", (35, 108), [(31, 108), (42, 108)], 0.85),
                ("the pupil", (37, 108), [(31, 108), (42, 108)], 0.85),
            ] {
                let before = contrast(&picture, line, around);
                let after = contrast(&got, line, around);
                eprintln!("{tag}: {name} {before:.4} -> {after:.4}");
                assert!(
                    after > before * least,
                    "{tag}: {name} faded: {before} -> {after}"
                );
            }
            // Every pixel of the hazel eye but its pupil, white included.
            let limit = if smoothing > 50.0 { 0.04 } else { 0.02 };
            let mut worst = (0.0f32, (0, 0));
            for y in 105..111 {
                for x in 30..44 {
                    if (36..38).contains(&x) && (107..109).contains(&y) {
                        continue;
                    }
                    let drift = moved(at(&got, (x, y)), at(&picture, (x, y)));
                    if drift > worst.0 {
                        worst = (drift, (x, y));
                    }
                }
            }
            eprintln!(
                "{tag}: the hazel eye moved {:.4} at {:?} {:?} -> {:?}",
                worst.0,
                worst.1,
                shown(at(&picture, worst.1)),
                shown(at(&got, worst.1))
            );
            assert!(
                worst.0 < limit,
                "{tag}: the hazel eye moved {} at {:?}",
                worst.0,
                worst.1
            );
            if smoothing <= 50.0 {
                for lips in [(16, 91), (11, 91), (20, 91), (16, 86), (16, 95)] {
                    let drift = moved(at(&got, lips), at(&picture, lips));
                    assert!(
                        drift < 0.025,
                        "{tag}: the lips in the cheek at {lips:?} moved {drift}"
                    );
                }
            }
            let drift = moved(at(&got, (32, 75)), at(&picture, (32, 75)));
            assert!(drift < 0.03, "{tag}: the grey eye's white moved {drift}");
        }
    }
}

/// Whiten keeps a highlight past white: skin brighter than white stays
/// past it, its excess kept rather than pulled back.
#[test]
fn beauty_whiten_keeps_highlights_past_white() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let beauty = concat_effects::Catalogue::builtin()
        .get("concat.beauty")
        .expect("a built-in");
    let picture = vec![lit([1.4, 1.1, 0.95]), lit([1.2, 0.95, 0.8])];
    let pass = beauty
        .pass(
            &BTreeMap::from([("smoothing".to_owned(), 0.0), ("whiten".to_owned(), 100.0)]),
            None,
        )
        .expect("a shader");
    let got = treated(&mut gpu, (2, 1), &picture, &[pass]);
    for (i, px) in got.iter().enumerate() {
        let (before, after) = (shown(picture[i]), shown(*px));
        eprintln!("{before:?} -> {after:?}");
        assert!(
            after[0] > 1.0,
            "highlight {i} pulled under white: {after:?}"
        );
        assert!(
            before[0] - after[0] < 0.02,
            "highlight {i} lowered: {before:?} -> {after:?}"
        );
    }
}

/// The blur's reach follows the frame's shorter side: the portrait drawn once
/// in a 540-line frame is smoothed as the same portrait drawn twice as
/// large in a 1080-line one.
#[test]
fn beauty_reach_follows_the_frame() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let (small_before, small) = beauty_in(&mut gpu, 1, 540, 100.0, 0.0);
    let (large_before, large) = beauty_in(&mut gpu, 2, 1080, 100.0, 0.0);
    for (xs, ys) in [(26..40, 8..32), (10..22, 64..78)] {
        let a = spread(&small, 96, 1, xs.clone(), ys.clone())
            / spread(&small_before, 96, 1, xs.clone(), ys.clone());
        let b =
            spread(&large, 192, 2, xs.clone(), ys.clone()) / spread(&large_before, 192, 2, xs, ys);
        eprintln!("grain left: {a:.3} at 540 lines, {b:.3} at 1080");
        assert!(
            (a - b).abs() < 0.08,
            "540 lines left {a} of the grain, 1080 lines {b}"
        );
    }
    let a = standing(&small, 1, (22, 52), [(14, 52), (30, 52)])
        / standing(&small_before, 1, (22, 52), [(14, 52), (30, 52)]);
    let b = standing(&large, 2, (22, 52), [(14, 52), (30, 52)])
        / standing(&large_before, 2, (22, 52), [(14, 52), (30, 52)]);
    eprintln!("brown spot left: {a:.3} at 540 lines, {b:.3} at 1080");
    assert!(
        (a - b).abs() < 0.1,
        "540 lines left {a} of the spot, 1080 lines {b}"
    );
}

/// The reach follows the frame's shorter side, not its height: the
/// portrait in an upright 1080 by 1920 frame comes out as in a 1920 by
/// 1080 one, pixel for pixel, at the default and at the most - so lips and
/// eyes in a vertical video keep what they keep at 1080 lines (see
/// `beauty_keeps_lines_hair_brows_and_eyes`) - and lips, rosy lips and
/// the hazel eye barely move there at the most.
#[test]
fn beauty_upright_frame_is_smoothed_as_one_on_its_side() {
    let Some(mut gpu) = heavy_gpu() else { return };
    for (smoothing, whiten) in [(50.0, 30.0), (100.0, 0.0)] {
        let (_, upright) = beauty_framed(&mut gpu, 1, (1080, 1920), smoothing, whiten);
        let (picture, side) = beauty_framed(&mut gpu, 1, (1920, 1080), smoothing, whiten);
        let mut worst = (0.0f32, 0usize);
        for (i, px) in side.iter().enumerate() {
            let drift = moved(upright[i], *px);
            if drift > worst.0 {
                worst = (drift, i);
            }
        }
        eprintln!(
            "{smoothing}, whiten {whiten}: upright and on its side differ by {:.5}",
            worst.0
        );
        assert!(
            worst.0 < 1e-3,
            "{smoothing}: upright and on its side differ by {} at {}",
            worst.0,
            worst.1
        );
        if whiten == 0.0 {
            for (x, y) in [
                (36, 50),
                (39, 52),
                (43, 54),
                (33, 52),
                (60, 20),
                (60, 70),
                (35, 108),
            ] {
                let drift = moved(at_k(&upright, 1, (x, y)), at_k(&picture, 1, (x, y)));
                eprintln!("upright at {smoothing}: ({x}, {y}) moved {drift:.4}");
                assert!(
                    drift < 0.03,
                    "upright at {smoothing}: ({x}, {y}) moved {drift}"
                );
            }
        }
    }
}

/// A pixel that is not a number, or is infinite, in skin or in its alpha,
/// at the default and the most, spreads no further than the renderer's own
/// reading of the picture carries it: every pixel outside the three by
/// three block round it comes out finite, and within 0.01 of where it does
/// with that pixel transparent. (The block itself is the renderer's: a
/// filtered read at a pixel's centre, which every package's `sample` and
/// the mix by intensity make, can carry a pixel that is not a number into
/// its eight neighbours whatever the package does; Beauty's own reads,
/// seven pixels and more each way, no longer widen it.)
#[test]
fn beauty_keeps_a_bad_pixel_to_itself() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let size = (64, 1080);
    for bad in [
        [f32::NAN, f32::NAN, f32::NAN, 1.0],
        [0.45, f32::NAN, 0.15, 1.0],
        [0.45, 0.25, 0.15, f32::NAN],
        [f32::INFINITY, 0.25, 0.15, 1.0],
        [0.45, f32::NEG_INFINITY, 0.15, 1.0],
    ] {
        let field = |odd: [f32; 4]| skin_field(size, 1, |x, y| (x == 32 && y == 32).then_some(odd));
        for (smoothing, whiten) in [(50.0, 30.0), (100.0, 100.0)] {
            let got = beautified_with(&mut gpu, size, &field(bad), smoothing, whiten);
            let clear = beautified_with(&mut gpu, size, &field([0.0; 4]), smoothing, whiten);
            let tag = format!("{bad:?} at {smoothing}, whiten {whiten}");
            let mut worst = 0.0f32;
            for (i, px) in got.iter().enumerate() {
                let (x, y) = (i % 64, i / 64);
                if x.abs_diff(32) <= 1 && y.abs_diff(32) <= 1 {
                    continue;
                }
                assert!(px.iter().all(|c| c.is_finite()), "{tag}: {px:?} at {i}");
                worst = worst.max(moved(*px, clear[i]));
            }
            eprintln!("{tag}: the rest within {worst:.5} of a transparent pixel there");
            assert!(worst < 0.01, "{tag}: the rest moved {worst}");
        }
    }
}

/// HSL with every slider at either end passes a pixel that is not a number
/// or is infinite by: every finite pixel round a run of them comes out
/// finite and as it would alone - all but the one just before the run,
/// which the renderer's own filtered read at a pixel's centre reaches into
/// whatever the package does (see `beauty_keeps_a_bad_pixel_to_itself`).
#[test]
fn hsl_passes_a_bad_pixel_by() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let bad = [
        [f32::NAN, f32::NAN, f32::NAN, 1.0],
        [0.5, f32::NAN, 0.1, 1.0],
        [0.5, 0.2, 0.1, f32::NAN],
        [f32::INFINITY, 0.2, 0.1, 1.0],
        [0.5, f32::NEG_INFINITY, 0.1, 1.0],
    ];
    let good = [
        [0.5f32, 0.2, 0.1, 1.0],
        [0.1, 0.2, 0.5, 1.0],
        [0.3, 0.3, 0.3, 0.5],
    ];
    let picture: Vec<[f32; 4]> = good.iter().chain(&bad).chain(&good).copied().collect();
    for end in [-100.0, 100.0] {
        let values: BTreeMap<String, f64> = BANDS
            .iter()
            .flat_map(|(band, _)| ["hue", "sat", "lum"].map(|p| (format!("{band}_{p}"), end)))
            .collect();
        let pass = hsl.pass(&values, None).expect("a shader");
        let got = treated(
            &mut gpu,
            (picture.len() as u32, 1),
            &picture,
            std::slice::from_ref(&pass),
        );
        let alone = treated(&mut gpu, (good.len() as u32, 1), &good, &[pass]);
        let after = good.len() + bad.len();
        let checked = (0..good.len() - 1)
            .map(|n| (n, got[n]))
            .chain((0..good.len()).map(|n| (n, got[after + n])));
        for (n, px) in checked {
            assert!(
                px.iter().all(|c| c.is_finite()) && moved(px, alone[n]) < 1e-6,
                "{end}: {:?} came out {px:?}, alone {:?}",
                good[n],
                alone[n]
            );
        }
    }
}

/// Hair on a layer that is not opaque is told as on an opaque one: a light
/// and a dark brown strand a pixel wide and a two pixel one across skin
/// that is opaque, half transparent throughout, or fading from opaque to a
/// quarter across the strands, keep at least 85% of how far they stand
/// from the skin in straight colour, at the default and at the most.
#[test]
fn beauty_keeps_hair_on_a_layer_not_opaque() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let size = (64, 1080);
    let light = lit([0.55, 0.38, 0.28]);
    let brown = lit([0.45, 0.28, 0.20]);
    // The alpha of column `x` of each layer.
    let alpha = |name: &str, x: u32| match name {
        "opaque" => 1.0,
        "half transparent" => 0.5,
        _ => (1.0 - (x as f32 - 16.0) / 32.0 * 0.75).clamp(0.25, 1.0),
    };
    let mut failures = Vec::new();
    for name in ["opaque", "half transparent", "fading"] {
        let mut picture = skin_field(size, 1, |_, y| match y {
            20 => Some(light),
            30 | 40 | 41 => Some(brown),
            _ => None,
        });
        for (i, px) in picture.iter_mut().enumerate() {
            px[3] = alpha(name, i as u32 % 64);
        }
        let lu = |p: [f32; 4]| {
            let d = shown(p);
            0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]
        };
        for smoothing in [50.0, 100.0] {
            let got = beautified(&mut gpu, size, &picture, smoothing);
            for (strand, row, above, below) in [
                ("a light brown strand", 20u32, 17u32, 23u32),
                ("a brown strand", 30, 27, 33),
                ("a two pixel brown strand", 40, 37, 44),
            ] {
                let mut least = f32::MAX;
                for x in 10..54u32 {
                    let at = |frame: &[[f32; 4]], y: u32| lu(frame[(y * 64 + x) as usize]);
                    let stands = |frame: &[[f32; 4]]| {
                        (at(frame, above) + at(frame, below)) / 2.0 - at(frame, row)
                    };
                    least = least.min(stands(&got) / stands(&picture));
                }
                eprintln!("{name}, {smoothing}: {strand} keeps {least:.3}");
                if least < 0.85 {
                    failures.push(format!("{name}, {smoothing}: {strand} kept {least}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Whiten lifts skin and leaves lips, hair and the sky; with both sliders
/// at nothing the picture is as shot.
#[test]
fn beauty_whitens_skin_alone_and_rests_at_nothing() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let (picture, got) = beauty(&mut gpu, 1, 0.0, 100.0);
    assert!(
        at_k(&got, 1, (30, 20))[1] > at_k(&picture, 1, (30, 20))[1] * 1.05,
        "skin was not lifted"
    );
    for spot in [(40, 52), (60, 20), (66, 20), (80, 20)] {
        let drift = moved(at_k(&got, 1, spot), at_k(&picture, 1, spot));
        assert!(drift < 0.01, "{spot:?} is not skin and moved {drift}");
    }
    let (picture, got) = beauty(&mut gpu, 1, 0.0, 0.0);
    assert!(
        got.iter().zip(&picture).all(|(a, b)| moved(*a, *b) < 1e-3),
        "at nothing the picture moved"
    );
}

/// A field of skin with pores, `size` pixels, drawn `k` times over, with
/// `feature` painted where it says.
fn skin_field(
    size: (u32, u32),
    k: u32,
    feature: impl Fn(u32, u32) -> Option<[f32; 4]>,
) -> Vec<[f32; 4]> {
    (0..size.1)
        .flat_map(|y| (0..size.0).map(move |x| (x, y)))
        .map(|(x, y)| {
            let (x, y) = (x / k, y / k);
            let pore = if (x * 7 + y * 13) % 5 < 2 {
                0.03
            } else {
                -0.02
            };
            feature(x, y).unwrap_or([0.45 + pore, 0.25 + pore, 0.15 + pore, 1.0])
        })
        .collect()
}

/// Beauty over `picture`, `size` pixels, smoothing only.
fn beautified(
    gpu: &mut WgpuCompositor,
    size: (u32, u32),
    picture: &[[f32; 4]],
    smoothing: f64,
) -> Vec<[f32; 4]> {
    beautified_with(gpu, size, picture, smoothing, 0.0)
}

/// Beauty over `picture`, `size` pixels, at `smoothing` and `whiten`, in a
/// frame as tall as the picture and as wide (so its shorter side is the
/// picture's height, which Beauty's reach follows).
fn beautified_with(
    gpu: &mut WgpuCompositor,
    size: (u32, u32),
    picture: &[[f32; 4]],
    smoothing: f64,
    whiten: f64,
) -> Vec<[f32; 4]> {
    let package = concat_effects::Catalogue::builtin()
        .get("concat.beauty")
        .expect("a built-in");
    let pass = package
        .pass(
            &BTreeMap::from([
                ("smoothing".to_owned(), smoothing),
                ("whiten".to_owned(), whiten),
            ]),
            None,
        )
        .expect("a shader");
    treated_within(gpu, (size.0.max(size.1), size.1), size, picture, &[pass])
}

/// Acne three, five and six pixels across, placed four ways against the
/// pixel grid, loses at least 60% of how far it stands out at the default -
/// every pixel of it, not just the middle - and at least 40% at 25, at 1080
/// lines and drawn twice as large at 2160, with no halo round it; a lighter
/// shine and a gold stud the same size keep at least 85% of theirs at the
/// default and at the most. Red and pink spots, a degree or two from lips
/// in hue, lose at least 45% at the default; and at 25 may be left as they
/// were, never standing out a tenth further; and every spot stays gone with
/// whiten at nothing, at its default and at its most - a filled spot is
/// whitened like the skin.
#[test]
fn beauty_takes_spots_and_keeps_shine() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let acne = [0.33, 0.13, 0.08, 1.0];
    let shine = lit([0.86, 0.72, 0.62]);
    let gold = lit([0.85, 0.68, 0.25]);
    let red = [0.40, 0.10, 0.08, 1.0];
    let pink = [0.45, 0.16, 0.14, 1.0];
    let mut cases = Vec::new();
    for radius in [1.5f32, 2.5, 3.0] {
        cases.push(("acne", acne, radius, 50.0, 0.4));
        cases.push(("acne", acne, radius, 25.0, 0.6));
    }
    // Red and pink spots are a degree or two from lips in hue, so the
    // shape tests that keep lips weigh on them too: they lose at least 45%.
    for colour in [red, pink] {
        cases.push(("red spot", colour, 2.5, 50.0, 0.55));
        cases.push(("red spot", colour, 2.5, 25.0, 1.1));
    }
    for smoothing in [50.0, 100.0] {
        cases.push(("shine", shine, 1.5, smoothing, -0.85));
        cases.push(("gold stud", gold, 1.5, smoothing, -0.85));
    }
    // Several placements of each spot against the pixel grid, since which
    // pixels a disc covers changes its shape.
    let placements = [(32.0f32, 32.0f32), (32.5, 32.0), (32.0, 32.6), (32.3, 31.7)];
    for k in [1, 2] {
        let size = (64 * k, 1080 * k);
        for ((&(name, colour, radius, smoothing, bound), whiten), (cx, cy)) in cases
            .iter()
            // Whiten lifts skin more than a shine already near white, which
            // is tone, not smoothing: what must be kept is checked without it.
            .flat_map(|case| {
                let levels: &[f64] = if case.4 > 0.0 {
                    &[0.0, 30.0, 100.0]
                } else {
                    &[0.0]
                };
                levels.iter().map(move |&w| (case, w))
            })
            .flat_map(|pair| placements.map(|at| (pair, at)))
        {
            let inside = |x: u32, y: u32| {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                dx * dx + dy * dy <= radius * radius
            };
            let picture = skin_field(size, k, |x, y| inside(x, y).then_some(colour));
            let got = beautified_with(&mut gpu, size, &picture, smoothing, whiten);
            let at = |frame: &[[f32; 4]], (x, y): (u32, u32)| {
                shown(frame[((y * k + k / 2) * 64 * k + x * k + k / 2) as usize])
            };
            // The skin round the spot, as each frame has it: a ring well
            // clear of the spot, averaged.
            let skin = |frame: &[[f32; 4]]| {
                let ring: Vec<[f32; 3]> = (20..44)
                    .flat_map(|x| (20..44).map(move |y| (x, y)))
                    .filter(|&(x, y)| {
                        let (dx, dy) = (x as f32 + 0.5 - 32.0, y as f32 + 0.5 - 32.0);
                        let d = (dx * dx + dy * dy).sqrt();
                        d > 8.0 && d < 12.0
                    })
                    .map(|p| at(frame, p))
                    .collect();
                let n = ring.len() as f32;
                [0, 1, 2].map(|c| ring.iter().map(|p| p[c]).sum::<f32>() / n)
            };
            let apart =
                |a: [f32; 3], b: [f32; 3]| (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max);
            let (skin_before, skin_after) = (skin(&picture), skin(&got));
            let mut before = 0.0f32;
            let mut after = 0.0f32;
            let mut halo = 0.0f32;
            for y in 24..40 {
                for x in 24..40 {
                    let (dx, dy) = (x as f32 + 0.5 - 32.0, y as f32 + 0.5 - 32.0);
                    let d = (dx * dx + dy * dy).sqrt();
                    if inside(x, y) {
                        before = before.max(apart(at(&picture, (x, y)), skin_before));
                        after = after.max(apart(at(&got, (x, y)), skin_after));
                    } else if d > radius + 1.0 && d < radius + 3.0 {
                        halo = halo.max(apart(at(&got, (x, y)), skin_after));
                    }
                }
            }
            let tag =
                format!("x{k} {radius:.1} px {name} at ({cx}, {cy}), {smoothing}, whiten {whiten}");
            eprintln!("{tag}: worst {before:.4} -> {after:.4}, halo {halo:.4}");
            if bound > 0.0 {
                assert!(after < before * bound, "{tag} stayed: {before} -> {after}");
            } else {
                assert!(
                    after > before * -bound,
                    "{tag} was flattened: {before} -> {after}"
                );
            }
            assert!(halo < 0.04, "{tag} left a halo of {halo}");
        }
    }
}

/// Skin fading out at the edge of a layer: the frame comes out the same
/// whether the transparent pixels beyond it hold junk or skin - nothing
/// of what cannot be seen bleeds into what can - and every alpha comes out
/// as it went in.
#[test]
fn beauty_keeps_a_fading_edge_clean() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let size = (64, 1080);
    let edge = |beyond: [f32; 4]| {
        skin_field(size, 1, move |x, _| match x {
            40..48 => Some([0.45, 0.25, 0.15, 1.0 - (x - 39) as f32 / 9.0]),
            48.. => Some(beyond),
            _ => None,
        })
    };
    let junk = edge([0.0, 5.0, 0.0, 0.0]);
    let clean = edge([0.45, 0.25, 0.15, 0.0]);
    for smoothing in [50.0, 100.0] {
        let a = beautified(&mut gpu, size, &junk, smoothing);
        let b = beautified(&mut gpu, size, &clean, smoothing);
        for y in 8..56 {
            for x in 0..64 {
                let i = (y * 64 + x) as usize;
                assert!(
                    (a[i][3] - junk[i][3]).abs() < 1e-3,
                    "alpha changed at {x},{y}"
                );
                if junk[i][3] > 0.0 {
                    let drift = moved(a[i], b[i]);
                    assert!(
                        drift < 1e-3,
                        "{smoothing}: what lies beyond reached {x},{y}: {drift}"
                    );
                }
            }
        }
    }
}

/// Smoothing at its default already takes at least half the texture out
/// of open skin at 1080 lines. Drawn twice as large at 2160, every pore is
/// a two-pixel block whose grain is coarser for the same reach, so there
/// it takes at least 45% out. A dark complexion in ordinary light is
/// smoothed as much as a fair one.
#[test]
fn beauty_default_smooths_clearly() {
    let Some(mut gpu) = heavy_gpu() else { return };
    for (k, most) in [(1, 0.5), (2, 0.55)] {
        let (picture, got) = beauty(&mut gpu, k, 50.0, 0.0);
        let before = spread(&picture, 96 * k, k, 26..40, 8..32);
        let after = spread(&got, 96 * k, k, 26..40, 8..32);
        eprintln!("x{k} default: grain {before:.5} -> {after:.5}");
        assert!(
            after < before * most,
            "x{k}: the default kept {after} of {before}"
        );
    }
    let size = (64, 1080);
    let dark = skin_field(size, 1, |x, y| {
        let pore = if (x * 7 + y * 13) % 5 < 2 {
            0.006
        } else {
            -0.004
        };
        Some([0.06 + pore, 0.03 + pore, 0.018 + pore, 1.0])
    });
    let got = beautified(&mut gpu, size, &dark, 50.0);
    let grain = |frame: &[[f32; 4]]| {
        let lumas: Vec<f32> = (8..56)
            .flat_map(|y| (8..56).map(move |x| (x, y)))
            .map(|(x, y)| {
                let d = shown(frame[(y * 64 + x) as usize]);
                0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]
            })
            .collect();
        let mean = lumas.iter().sum::<f32>() / lumas.len() as f32;
        (lumas.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / lumas.len() as f32).sqrt()
    };
    let (before, after) = (grain(&dark), grain(&got));
    eprintln!("dark complexion default: grain {before:.5} -> {after:.5}");
    assert!(
        after < before * 0.5,
        "a dark complexion kept {after} of {before}"
    );
}

/// HSL on a colour with a channel a little below black matches the same
/// colour clipped to black with that channel laid back on. Beauty leaves a
/// colour with a channel below black - too saturated to be skin - as shot
/// on its own, that channel included, and evens a lone one inside skin
/// into the skin like a spot, to within 0.08 in the display encoding of
/// the skin round it (from some 0.7 off).
#[test]
fn colour_below_black_is_carried_through() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let catalogue = concat_effects::Catalogue::builtin();
    let hsl = catalogue.get("concat.hsl").expect("a built-in");
    let pass = hsl
        .pass(&BTreeMap::from([("orange_lum".to_owned(), 100.0)]), None)
        .expect("a shader");
    let picture = vec![[0.5, 0.12, -0.0005, 1.0], [0.5, 0.12, 0.0, 1.0]];
    let got = treated(&mut gpu, (2, 1), &picture, &[pass]);
    let laid_back = 0.0005f32.powf(1.0 / 2.4);
    let (a, b) = (shown(got[0]), shown(got[1]));
    let (a_blue, b_blue) = (to_display_signed(got[0][2]), to_display_signed(got[1][2]));
    eprintln!("{a:?} vs {b:?}");
    assert!(
        (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3,
        "{a:?} vs {b:?}"
    );
    assert!(
        (a_blue - (b_blue - laid_back)).abs() < 1e-3,
        "the channel below black was not laid back: {a_blue} vs {b_blue}"
    );
    let beauty = catalogue.get("concat.beauty").expect("a built-in");
    let pass = beauty
        .pass(
            &BTreeMap::from([("smoothing".to_owned(), 50.0), ("whiten".to_owned(), 100.0)]),
            None,
        )
        .expect("a shader");
    let alone = vec![[0.45, 0.25, -0.01, 1.0]];
    let got = treated(&mut gpu, (1, 1), &alone, std::slice::from_ref(&pass));
    eprintln!("alone: {:?}", got[0]);
    assert!(
        moved(got[0], alone[0]) < 1e-3 && got[0][2] < 0.0,
        "a colour below black was not left as shot: {:?}",
        got[0]
    );
    let size = (64, 1080);
    let odd = [0.45, 0.25, -0.01, 1.0];
    let field = skin_field(size, 1, |x, y| (x == 32 && y == 32).then_some(odd));
    let got = treated_within(&mut gpu, (1080, 1080), size, &field, &[pass]);
    let i = (32 * 64 + 32) as usize;
    let around: Vec<[f32; 3]> = (28..37)
        .flat_map(|x| (28..37).map(move |y| (x, y)))
        .filter(|&(x, y)| (x, y) != (32, 32))
        .map(|(x, y)| shown(got[(y * 64 + x) as usize]))
        .collect();
    let n = around.len() as f32;
    let skin = [0, 1, 2].map(|c| around.iter().map(|p| p[c]).sum::<f32>() / n);
    let here = shown(got[i]);
    let off = (0..3)
        .map(|c| (here[c] - skin[c]).abs())
        .fold(0.0, f32::max);
    eprintln!("in skin: {here:?}, the skin round it {skin:?}");
    assert!(
        got[i][2] > 0.0 && off < 0.08,
        "a lone colour below black inside skin was not evened into it: {here:?} vs {skin:?}"
    );
}

/// The display encoding of one channel, negative values mirrored.
fn to_display_signed(v: f32) -> f32 {
    v.signum() * v.abs().powf(1.0 / 2.4)
}

/// Lips are not spots: ellipses six to twelve pixels tall, red and rosy,
/// flat and textured, on pored skin, at the default and at the most, at
/// 1080 lines and drawn twice as large at 2160 - every pixel of them, tips
/// included, moves by less than 0.03 in linear light (0.05 for textured
/// lips, whose texture rosy lips may have softened), their mean colour
/// shifts by under 0.005 in the display encoding, and textured lips keep at
/// least a quarter of their texture at the default. Lips only four pixels
/// tall, as on a face far off, keep their middle rows to those bounds;
/// their outer rows may be evened like a spot, as far as the skin round
/// them but no further.
#[test]
fn beauty_keeps_small_lips_whole() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let red = [0.75, 0.30, 0.30];
    let rosy = [0.75, 0.43, 0.38];
    let mut failures = Vec::new();
    for k in [1, 2] {
        let size = (64 * k, 1080 * k);
        for (tall, wide) in [(4.0f32, 16.0f32), (6.0, 20.0), (8.0, 30.0), (12.0, 40.0)] {
            for (name, colour, grain) in [
                ("red", red, 0.0f32),
                ("rosy", rosy, 0.0),
                ("textured red", red, 0.02),
                ("textured rosy", rosy, 0.02),
            ] {
                let inside = |x: u32, y: u32| {
                    let (dx, dy) = (
                        (x as f32 + 0.5 - 32.0) / (wide / 2.0),
                        (y as f32 + 0.5 - 32.0) / (tall / 2.0),
                    );
                    dx * dx + dy * dy <= 1.0
                };
                let picture = skin_field(size, k, |x, y| {
                    inside(x, y).then(|| {
                        // Texture in the display encoding: a shade lighter or
                        // darker, pixel by pixel.
                        let t = if (x * 5 + y * 3) % 4 < 2 {
                            grain
                        } else {
                            -grain
                        };
                        lit([colour[0] + t, colour[1] + t, colour[2] + t])
                    })
                });
                for smoothing in [50.0, 100.0] {
                    let got = beautified_with(&mut gpu, size, &picture, smoothing, 30.0);
                    let mut worst = (0.0f32, (0, 0));
                    for y in 10..54 {
                        for x in 10..54 {
                            if !inside(x, y) {
                                continue;
                            }
                            // Lips only four pixels tall, as on a face far
                            // off, may lose their outer rows: those are held
                            // to a looser bound, their middle to the full one.
                            if tall < 6.0 && (y as f32 + 0.5 - 32.0).abs() > 1.0 {
                                let i = ((y * k + k / 2) * 64 * k + x * k + k / 2) as usize;
                                let drift = moved(got[i], picture[i]);
                                // They may become skin, but no more than skin:
                                // never further from what they were than the
                                // skin round them is, pores and all.
                                let skin = [0.45f32 + 0.03, 0.25 + 0.03, 0.15 + 0.03, 1.0];
                                if drift >= moved(picture[i], skin) + 0.01 {
                                    failures.push(format!(
                                        "x{k} {name} lips {tall}x{wide} at {smoothing}: \
                                         outer row moved {drift} at ({x}, {y})"
                                    ));
                                }
                                continue;
                            }
                            let i = ((y * k + k / 2) * 64 * k + x * k + k / 2) as usize;
                            let drift = moved(got[i], picture[i]);
                            if drift > worst.0 {
                                worst = (drift, (x, y));
                            }
                        }
                    }
                    let tag = format!("x{k} {name} lips {tall}x{wide} at {smoothing}");
                    // The lips' mean colour holds, and textured lips keep at
                    // least a quarter of their texture at the default.
                    let (mut mean_a, mut mean_b, mut n) = ([0.0f32; 3], [0.0f32; 3], 0.0f32);
                    let (mut var_a, mut var_b) = (0.0f32, 0.0f32);
                    let mut lumas = Vec::new();
                    for y in 10..54 {
                        for x in 10..54 {
                            if inside(x, y) && (tall >= 6.0 || (y as f32 + 0.5 - 32.0).abs() <= 1.0)
                            {
                                let i = ((y * k + k / 2) * 64 * k + x * k + k / 2) as usize;
                                let (a, b) = (shown(picture[i]), shown(got[i]));
                                for c in 0..3 {
                                    mean_a[c] += a[c];
                                    mean_b[c] += b[c];
                                }
                                lumas.push((a[1], b[1]));
                                n += 1.0;
                            }
                        }
                    }
                    let shift = (0..3)
                        .map(|c| ((mean_a[c] - mean_b[c]) / n).abs())
                        .fold(0.0, f32::max);
                    let (ma, mb) = (mean_a[1] / n, mean_b[1] / n);
                    for (a, b) in &lumas {
                        var_a += (a - ma).powi(2);
                        var_b += (b - mb).powi(2);
                    }
                    if shift >= 0.005 {
                        failures.push(format!("{tag}: mean colour shifted {shift}"));
                    }
                    if grain > 0.0 && smoothing <= 50.0 && var_b < var_a * 0.0625 {
                        failures.push(format!("{tag}: texture kept {}", (var_b / var_a).sqrt()));
                    }
                    eprintln!("{tag}: worst {:.4} at {:?}", worst.0, worst.1);
                    // Textured lips share skin's hues when rosy, so their
                    // texture may be softened; their colour and shape stay.
                    let most = if grain > 0.0 { 0.05 } else { 0.03 };
                    if worst.0 >= most {
                        failures.push(format!("{tag} moved {} at {:?}", worst.0, worst.1));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Spots in a cluster each go, every pixel of them, at the default, at
/// 1080 lines and drawn twice as large at 2160: three acne or red spots
/// three pixels across with centres seven to fourteen pixels apart, or five
/// pixels across with centres eleven or fourteen apart, each lose at least
/// 60% of how far they stand out; five-pixel spots seven or nine pixels
/// apart, with only two to four pixels of skin between them, at least half.
/// No cluster leaves a halo.
#[test]
fn beauty_takes_spots_in_a_cluster() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let acne = [0.33, 0.13, 0.08, 1.0];
    let red = [0.40, 0.10, 0.08, 1.0];
    let mut failures = Vec::new();
    for k in [1, 2] {
        let size = (64 * k, 1080 * k);
        for (name, colour) in [("acne", acne), ("red spots", red)] {
            for radius in [1.5f32, 2.5] {
                for apart in [7.0f32, 9.0, 11.0, 14.0] {
                    // Five-pixel spots closer than about ten pixels, with
                    // little skin between them, are held to half.
                    let close = radius > 2.0 && apart < 10.0;
                    let centres = [
                        (32.0 - apart / 2.0, 30.0),
                        (32.0 + apart / 2.0, 30.0),
                        (32.0, 30.0 + apart * 0.866),
                    ];
                    let which = |x: u32, y: u32| {
                        centres.iter().position(|&(cx, cy)| {
                            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                            dx * dx + dy * dy <= radius * radius
                        })
                    };
                    let picture = skin_field(size, k, |x, y| which(x, y).map(|_| colour));
                    let got = beautified_with(&mut gpu, size, &picture, 50.0, 30.0);
                    let at = |frame: &[[f32; 4]], (x, y): (u32, u32)| {
                        shown(frame[((y * k + k / 2) * 64 * k + x * k + k / 2) as usize])
                    };
                    let skin = |frame: &[[f32; 4]]| {
                        let around: Vec<[f32; 3]> = (4..60)
                            .flat_map(|x| [4u32, 58].map(move |y| (x, y)))
                            .map(|p| at(frame, p))
                            .collect();
                        let n = around.len() as f32;
                        [0, 1, 2].map(|c| around.iter().map(|p| p[c]).sum::<f32>() / n)
                    };
                    let apart_from = |a: [f32; 3], b: [f32; 3]| {
                        (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)
                    };
                    let (skin_before, skin_after) = (skin(&picture), skin(&got));
                    let mut before = [0.0f32; 3];
                    let mut after = [0.0f32; 3];
                    for y in 10..54 {
                        for x in 10..54 {
                            if let Some(n) = which(x, y) {
                                before[n] =
                                    before[n].max(apart_from(at(&picture, (x, y)), skin_before));
                                after[n] = after[n].max(apart_from(at(&got, (x, y)), skin_after));
                            }
                        }
                    }
                    let mut halo = 0.0f32;
                    for y in 10..54 {
                        for x in 10..54 {
                            let near = centres.iter().any(|&(cx, cy)| {
                                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                                let d = (dx * dx + dy * dy).sqrt();
                                d > radius + 1.0 && d < radius + 3.0
                            });
                            if near && which(x, y).is_none() {
                                halo = halo.max(apart_from(at(&got, (x, y)), skin_after));
                            }
                        }
                    }
                    for n in 0..3 {
                        let tag = format!("x{k} {name} {radius:.1} px, {apart} px apart, spot {n}");
                        eprintln!("{tag}: {:.4} -> {:.4}, halo {halo:.4}", before[n], after[n]);
                        let most = if close { 0.5 } else { 0.4 };
                        if after[n] >= before[n] * most {
                            failures.push(format!("{tag} stayed: {} -> {}", before[n], after[n]));
                        }
                    }
                    if halo >= 0.04 {
                        failures.push(format!(
                            "x{k} {name} {radius:.1} px, {apart} px apart: halo {halo}"
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A spot past six pixels across may be left as it is or evened only in
/// part, as the package says - its rim may be kept round an evened middle -
/// but it is bounded: spots seven to sixteen pixels across, placed four
/// ways against the pixel grid, at 25, the default and the most with whiten
/// at 30, and at the default and the most with whiten at its most, at 1080
/// lines and drawn twice as large at 2160. No pixel of one stands out more
/// than a quarter further than it did (whiten lifting the skin round it),
/// or 0.45 further at full whiten, and it leaves no halo (0.04 in the
/// display encoding).
#[test]
fn beauty_leaves_large_spots_whole() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let acne = [0.33, 0.13, 0.08, 1.0];
    let red = [0.40, 0.10, 0.08, 1.0];
    let placements = [(32.0f32, 32.0f32), (32.5, 32.0), (32.0, 32.6), (32.3, 31.7)];
    let mut failures = Vec::new();
    for k in [1, 2] {
        let size = (64 * k, 1080 * k);
        for (name, colour) in [("acne", acne), ("red spot", red)] {
            for across in [7.0f32, 8.0, 10.0, 12.0, 16.0] {
                let radius = across / 2.0;
                for (cx, cy) in placements {
                    let off = |x: u32, y: u32| {
                        let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                        (dx * dx + dy * dy).sqrt()
                    };
                    let picture =
                        skin_field(size, k, |x, y| (off(x, y) <= radius).then_some(colour));
                    for (smoothing, whiten) in [
                        (25.0, 30.0),
                        (50.0, 30.0),
                        (100.0, 30.0),
                        (50.0, 100.0),
                        (100.0, 100.0),
                    ] {
                        let got = beautified_with(&mut gpu, size, &picture, smoothing, whiten);
                        let at = |frame: &[[f32; 4]], (x, y): (u32, u32)| {
                            shown(frame[((y * k + k / 2) * 64 * k + x * k + k / 2) as usize])
                        };
                        let skin = |frame: &[[f32; 4]]| {
                            let around: Vec<[f32; 3]> = (4..60)
                                .flat_map(|x| [4u32, 59].map(move |y| (x, y)))
                                .map(|p| at(frame, p))
                                .collect();
                            let n = around.len() as f32;
                            [0, 1, 2].map(|c| around.iter().map(|p| p[c]).sum::<f32>() / n)
                        };
                        let apart = |a: [f32; 3], b: [f32; 3]| {
                            (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)
                        };
                        let (skin_before, skin_after) = (skin(&picture), skin(&got));
                        let (mut worst, mut halo) = (0.0f32, 0.0f32);
                        for y in 8..56 {
                            for x in 8..56 {
                                let d = off(x, y);
                                if d <= radius {
                                    let before = apart(at(&picture, (x, y)), skin_before);
                                    let left = apart(at(&got, (x, y)), skin_after) / before;
                                    worst = worst.max(left);
                                } else if d > radius + 1.0 && d < radius + 3.0 {
                                    halo = halo.max(apart(at(&got, (x, y)), skin_after));
                                }
                            }
                        }
                        let tag = format!(
                            "x{k} {across} px {name} at ({cx}, {cy}), {smoothing}, whiten {whiten}"
                        );
                        eprintln!("{tag}: worst {worst:.3}, halo {halo:.4}");
                        let most = if whiten > 50.0 { 1.45 } else { 1.25 };
                        if worst > most {
                            failures.push(format!("{tag}: stands out {worst} as far"));
                        }
                        if halo >= 0.04 {
                            failures.push(format!("{tag}: halo {halo}"));
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Lips too small to tell from a spot, bounded: lips three pixels tall at
/// 1080 lines, and lips four to eight pixels tall at 1920 lines on the
/// frame's shorter side (where the reach is nearly twice as many pixels),
/// may be evened into the skin, as the package says - but every channel of
/// every pixel of them lands between the lips' own and the skin's - from its
/// darkest pore to the brightest the skin came out - at the default and at
/// the most, and with whiten at its most. Lips twelve pixels tall at 1920
/// lines keep the full bounds of lips: every pixel within 0.03 in
/// linear light, the mean within 0.005 in the display encoding.
#[test]
fn beauty_bounds_lips_too_small_to_tell() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let red = lit([0.75, 0.30, 0.30]);
    let rosy = lit([0.75, 0.43, 0.38]);
    let skin = [0.45f32 + 0.03, 0.25 + 0.03, 0.15 + 0.03, 1.0];
    let mut failures = Vec::new();
    for (height, tall, wide, full) in [
        (1080, 3.0f32, 14.0f32, false),
        (1080, 3.0, 20.0, false),
        (1920, 4.0, 16.0, false),
        (1920, 6.0, 20.0, false),
        (1920, 8.0, 30.0, false),
        (1920, 12.0, 40.0, true),
    ] {
        for (name, colour) in [("red", red), ("rosy", rosy)] {
            let size = (64, height);
            let inside = |x: u32, y: u32| {
                let dx = (x as f32 + 0.5 - 32.0) / (wide / 2.0);
                let dy = (y as f32 + 0.5 - 32.0) / (tall / 2.0);
                dx * dx + dy * dy <= 1.0
            };
            let picture = skin_field(size, 1, |x, y| inside(x, y).then_some(colour));
            for (smoothing, whiten) in [(50.0, 30.0), (100.0, 30.0), (50.0, 100.0)] {
                let got = beautified_with(&mut gpu, size, &picture, smoothing, whiten);
                // The brightest the skin itself came out, well clear of the lips.
                let mut lifted = [0.0f32; 3];
                for y in (2..12).chain(52..62) {
                    for x in 2..62 {
                        let px = got[(y * 64 + x) as usize];
                        for c in 0..3 {
                            lifted[c] = lifted[c].max(px[c]);
                        }
                    }
                }
                let tag = format!(
                    "{name} lips {tall}x{wide} at {height} lines, {smoothing}, whiten {whiten}"
                );
                let (mut worst, mut a, mut b, mut n) = (0.0f32, [0.0f32; 3], [0.0f32; 3], 0.0f32);
                for y in 10..54 {
                    for x in 10..54 {
                        if !inside(x, y) {
                            continue;
                        }
                        let i = (y * 64 + x) as usize;
                        let drift = moved(got[i], picture[i]);
                        worst = worst.max(drift);
                        // Each channel lands between the lips' own and the
                        // skin's, from its darkest pore to whitened, give or take 0.01.
                        for c in 0..3 {
                            let low = picture[i][c].min(skin[c] - 0.05) - 0.01;
                            let high = picture[i][c].max(lifted[c]) + 0.01;
                            if got[i][c] < low || got[i][c] > high {
                                failures.push(format!(
                                "{tag}: ({x}, {y}) channel {c} at {} is not between lips and skin",
                                got[i][c]
                            ));
                            }
                        }
                        let (pa, pb) = (shown(picture[i]), shown(got[i]));
                        for c in 0..3 {
                            a[c] += pa[c];
                            b[c] += pb[c];
                        }
                        n += 1.0;
                    }
                }
                let shift = (0..3)
                    .map(|c| ((a[c] - b[c]) / n).abs())
                    .fold(0.0, f32::max);
                eprintln!("{tag}: worst {worst:.4}, mean shift {shift:.4}");
                if full && (worst >= 0.03 || shift >= 0.005) {
                    failures.push(format!("{tag}: moved {worst}, mean {shift}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Every band's saturation at its most, across a wheel of mid to strong
/// saturations and brightnesses up to white: no channel goes below black,
/// none goes past white that was not already, chroma rises below white, and
/// there is no seam between neighbours.
#[test]
fn hsl_saturation_at_its_most_stops_at_black_and_white() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let values: BTreeMap<String, f64> = BANDS
        .iter()
        .map(|(band, _)| (format!("{band}_sat"), 100.0))
        .collect();
    let pass = hsl.pass(&values, None).expect("a shader");
    for (s, v) in [
        (0.5f32, 0.6f32),
        (0.5, 1.0),
        (0.8, 0.8),
        (0.95, 1.0),
        (0.3, 0.9),
        (0.95, 0.7),
    ] {
        let size = (720, 1);
        let wheel: Vec<[f32; 4]> = (0..size.0)
            .map(|x| lit(hsv(x as f32 / 2.0, s, v)))
            .collect();
        let got = treated(&mut gpu, size, &wheel, std::slice::from_ref(&pass));
        for x in 0..got.len() {
            let (before, after) = (shown(wheel[x]), shown(got[x]));
            for c in 0..3 {
                assert!(
                    got[x][c] >= -1e-4,
                    "s {s} v {v}: channel {c} below black at {x}"
                );
                assert!(
                    after[c] <= before[c].max(1.0) + 2e-3,
                    "s {s} v {v}: channel {c} past white at {x}: {before:?} -> {after:?}"
                );
            }
            let (_, c0) = hue_and_chroma(before);
            let (_, c1) = hue_and_chroma(after);
            if v < 0.95 {
                assert!(
                    c1 > c0 + 0.01,
                    "s {s} v {v}: chroma did not rise at {x}: {c0} -> {c1}"
                );
            }
            // No seam: a step round the wheel out is at most three times
            // the step in, in the display encoding.
            let apart = |a: [f32; 4], b: [f32; 4]| {
                let (a, b) = (shown(a), shown(b));
                (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)
            };
            let step = apart(got[x], got[(x + 1) % got.len()]);
            let input = apart(wheel[x], wheel[(x + 1) % wheel.len()]);
            assert!(
                step <= 3.0 * input + 1e-3,
                "s {s} v {v}: a seam of {step} for {input} at {x}"
            );
        }
    }
}

/// Saturation up along a ramp of brightness, from half way to white, past
/// white and a little beyond, at several hues - pale yellow, cyan and cream
/// among them - and at +40 (the card's own blue) and +100: brightness only
/// rises along the ramp, each step out is at most three times the step in,
/// and no channel jumps where the ramp crosses white.
#[test]
fn hsl_saturation_keeps_a_ramp_to_white_in_order() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let steps = 220u32;
    for amount in [40.0, 100.0] {
        let values: BTreeMap<String, f64> = BANDS
            .iter()
            .map(|(band, _)| (format!("{band}_sat"), amount))
            .collect();
        let pass = hsl.pass(&values, None).expect("a shader");
        for top in [
            [0.45f32, 0.6, 1.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.9, 0.3],
            [0.3, 1.0, 0.4],
            [1.0, 0.97, 0.7],
            [0.7, 1.0, 0.97],
            [1.0, 0.9, 0.8],
        ] {
            let ramp: Vec<[f32; 4]> = (0..steps)
                .map(|i| {
                    let f = 0.5 + 0.55 * i as f32 / (steps - 1) as f32;
                    lit([top[0] * f, top[1] * f, top[2] * f])
                })
                .collect();
            let got = treated(&mut gpu, (steps, 1), &ramp, std::slice::from_ref(&pass));
            let bright = |px: [f32; 4]| {
                let d = shown(px);
                0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]
            };
            for i in 0..steps as usize - 1 {
                let (a, b) = (bright(got[i]), bright(got[i + 1]));
                assert!(
                    b > a - 1e-3,
                    "{top:?} at {amount}: brightness fell at {i}: {a} -> {b}"
                );
                let step = (0..3)
                    .map(|c| (shown(got[i + 1])[c] - shown(got[i])[c]).abs())
                    .fold(0.0, f32::max);
                let input = (0..3)
                    .map(|c| (shown(ramp[i + 1])[c] - shown(ramp[i])[c]).abs())
                    .fold(0.0, f32::max);
                assert!(
                    step <= 3.0 * input + 1e-3,
                    "{top:?} at {amount}: a step of {step} for {input} in at {i}"
                );
            }
        }
    }
}

/// Saturation at +100 over a grid of mid colours, chroma 0.15 or more,
/// brightest channel at most 0.75, in the display encoding: chroma never
/// more than doubles; it at least half again where the brightest channel is
/// at most 0.6, and at least 1.35 times everywhere in the grid; brightness
/// is kept throughout. Close to double where there is room: the median gain
/// is at least 1.85 where the brightest channel is at most 0.6, and at least
/// 1.75 over the whole grid. The least gain falls on a bright yellow or
/// yellow-green, as the package says, not on a green.
#[test]
fn hsl_saturation_doubles_mid_colours() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let hsl = concat_effects::Catalogue::builtin()
        .get("concat.hsl")
        .expect("a built-in");
    let values: BTreeMap<String, f64> = BANDS
        .iter()
        .map(|(band, _)| (format!("{band}_sat"), 100.0))
        .collect();
    let pass = hsl.pass(&values, None).expect("a shader");
    let levels = [0.2f32, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75];
    let mut colours = Vec::new();
    for r in levels {
        for g in levels {
            for b in levels {
                let c = [r, g, b];
                let (_, chroma) = hue_and_chroma(c);
                if chroma >= 0.15 {
                    colours.push(c);
                }
            }
        }
    }
    let picture: Vec<[f32; 4]> = colours.iter().map(|c| lit(*c)).collect();
    let got = treated(&mut gpu, (colours.len() as u32, 1), &picture, &[pass]);
    let bright = |d: [f32; 3]| 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2];
    let mut failures = Vec::new();
    let mut least = (f32::MAX, 0.0f32);
    let (mut gains, mut roomy) = (Vec::new(), Vec::new());
    for (i, colour) in colours.iter().enumerate() {
        let after = shown(got[i]);
        let (h0, c0) = hue_and_chroma(*colour);
        let (_, c1) = hue_and_chroma(after);
        let gain = c1 / c0;
        if gain < least.0 {
            least = (gain, h0);
        }
        let y = bright(*colour);
        let high = colour[0].max(colour[1]).max(colour[2]);
        gains.push(gain);
        if high <= 0.6 {
            roomy.push(gain);
        }
        if high <= 0.6 && gain < 1.5 {
            failures.push(format!("{colour:?} has room and gained only {gain}"));
        }
        if !(1.35..=2.0 + 1e-3).contains(&gain) {
            failures.push(format!("{colour:?} gained {gain}"));
        }
        if (bright(after) - y).abs() >= 2e-3 {
            failures.push(format!("{colour:?}: brightness moved"));
        }
    }
    let median = |mut v: Vec<f32>| {
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    let (all, with_room) = (median(gains), median(roomy));
    eprintln!(
        "{} colours, least gain {:.3} at {:.1} degrees, median {all:.3}, median with room {with_room:.3}",
        colours.len(),
        least.0,
        least.1,
    );
    if with_room < 1.85 {
        failures.push(format!("median gain with room only {with_room}"));
    }
    if all < 1.75 {
        failures.push(format!("median gain only {all}"));
    }
    if !(45.0..=100.0).contains(&least.1) {
        failures.push(format!("least gain at {} degrees, not a yellow", least.1));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Close clusters of five-pixel spots, acne-coloured and red, with
/// centres seven and nine pixels apart - two to four pixels of skin between
/// them - at 1080 lines and drawn twice as large at 2160, with whiten at its
/// default 30 and at its most. From a smoothing of 50, where the fill is in
/// full, every pixel of every spot loses at least half of how far it stands
/// out. At 25, where the fill is only part way in, none ends up standing out
/// further than it did at whiten 30, or more than a tenth further at full
/// whiten, which lifts the skin round it. No cluster leaves a halo (0.04 in
/// the display encoding).
#[test]
fn beauty_bounds_close_clusters() {
    let Some(mut gpu) = heavy_gpu() else { return };
    let acne = [0.33, 0.13, 0.08, 1.0];
    let red = [0.40, 0.10, 0.08, 1.0];
    let mut failures = Vec::new();
    for k in [1u32, 2] {
        let size = (64 * k, 1080 * k);
        for (name, colour) in [("acne", acne), ("red spots", red)] {
            for apart in [7.0f32, 9.0] {
                let centres = [
                    (32.0 - apart / 2.0, 30.0),
                    (32.0 + apart / 2.0, 30.0),
                    (32.0, 30.0 + apart * 0.866),
                ];
                let off = |x: u32, y: u32| {
                    centres
                        .iter()
                        .map(|&(cx, cy)| {
                            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                            (dx * dx + dy * dy).sqrt()
                        })
                        .fold(f32::MAX, f32::min)
                };
                let picture = skin_field(size, k, |x, y| (off(x, y) <= 2.5).then_some(colour));
                for (smoothing, whiten, most) in [
                    (25.0, 30.0, 1.0f32),
                    (50.0, 30.0, 0.5),
                    (100.0, 30.0, 0.5),
                    (25.0, 100.0, 1.1),
                    (50.0, 100.0, 0.5),
                    (100.0, 100.0, 0.5),
                ] {
                    let got = beautified_with(&mut gpu, size, &picture, smoothing, whiten);
                    let at = |frame: &[[f32; 4]], x: u32, y: u32| {
                        shown(frame[((y * k + k / 2) * 64 * k + x * k + k / 2) as usize])
                    };
                    let skin = |frame: &[[f32; 4]]| {
                        let around: Vec<[f32; 3]> = (4..60)
                            .flat_map(|x| [4u32, 58].map(move |y| at(frame, x, y)))
                            .collect();
                        let n = around.len() as f32;
                        [0, 1, 2].map(|c| around.iter().map(|p| p[c]).sum::<f32>() / n)
                    };
                    let apart_from = |a: [f32; 3], b: [f32; 3]| {
                        (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)
                    };
                    let (sb, sa) = (skin(&picture), skin(&got));
                    let (mut worst, mut halo) = (0.0f32, 0.0f32);
                    for y in 10..54 {
                        for x in 10..54 {
                            let d = off(x, y);
                            if d <= 2.5 {
                                let ratio = apart_from(at(&got, x, y), sa)
                                    / apart_from(at(&picture, x, y), sb);
                                worst = worst.max(ratio);
                            } else if d > 3.5 && d < 5.5 {
                                halo = halo.max(apart_from(at(&got, x, y), sa));
                            }
                        }
                    }
                    let tag = format!(
                        "x{k} {name} {apart} px apart at smoothing {smoothing}, whiten {whiten}"
                    );
                    eprintln!("{tag}: worst {worst:.3}, halo {halo:.4}");
                    if worst > most {
                        failures.push(format!("{tag}: stands out {worst} as far"));
                    }
                    if halo >= 0.04 {
                        failures.push(format!("{tag}: halo {halo}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
