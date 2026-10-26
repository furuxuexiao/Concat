// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! 补帧（帧插值）纯逻辑：尺寸补边、RGBA 与平面 RGB 张量互转、输出帧调度。
//!
//! 本模块不依赖任何推理后端，在所有特性组合下均可编译与测试。

/// 插值模型要求的尺寸对齐粒度（像素）。
pub const PAD_ALIGN: u32 = 32;

/// 将 `(w, h)` 各自向上补齐到 [`PAD_ALIGN`] 的倍数。
///
/// 已对齐的尺寸保持不变，`0` 仍为 `0`。
pub fn padded_size(w: u32, h: u32) -> (u32, u32) {
    (
        w.div_ceil(PAD_ALIGN) * PAD_ALIGN,
        h.div_ceil(PAD_ALIGN) * PAD_ALIGN,
    )
}

/// 把 `w×h` 的 RGBA8 缓冲转换为补边后的平面 RGB（CHW 布局，`3×ph×pw`）f32 张量，取值 0..1。
///
/// 补边区域复制最近的边缘像素，避免模型在边界处看到硬边。返回 `(张量, pw, ph)`，
/// 其中 `(pw, ph) == padded_size(w, h)`。
///
/// # Panics
/// 当 `rgba.len() < w * h * 4` 时 panic。
pub fn rgba_to_planar_rgb(rgba: &[u8], w: u32, h: u32) -> (Vec<f32>, u32, u32) {
    let (pw, ph) = padded_size(w, h);
    let (wu, hu, pwu, phu) = (w as usize, h as usize, pw as usize, ph as usize);
    assert!(rgba.len() >= wu * hu * 4, "RGBA 缓冲长度不足");
    let plane = pwu * phu;
    let mut out = vec![0.0f32; plane * 3];
    if wu == 0 || hu == 0 {
        return (out, pw, ph);
    }
    for y in 0..phu {
        let sy = y.min(hu - 1);
        for x in 0..pwu {
            let sx = x.min(wu - 1);
            let s = (sy * wu + sx) * 4;
            let d = y * pwu + x;
            for c in 0..3 {
                out[c * plane + d] = f32::from(rgba[s + c]) / 255.0;
            }
        }
    }
    (out, pw, ph)
}

/// [`rgba_to_planar_rgb`] 的逆操作：把 `3×ph×pw` 平面 RGB 张量裁回 `w×h` 的 RGBA8 缓冲。
///
/// 数值先夹紧到 0..1 再量化（NaN 视为 0），alpha 恒为 255。
///
/// # Panics
/// 当 `w > pw`、`h > ph` 或张量长度不足 `3 * pw * ph` 时 panic。
pub fn planar_rgb_to_rgba(planar: &[f32], pw: u32, ph: u32, w: u32, h: u32) -> Vec<u8> {
    assert!(w <= pw && h <= ph, "裁剪尺寸超出补边尺寸");
    let (wu, hu, pwu) = (w as usize, h as usize, pw as usize);
    let plane = pwu * ph as usize;
    assert!(planar.len() >= plane * 3, "平面张量长度不足");
    let mut out = Vec::with_capacity(wu * hu * 4);
    for y in 0..hu {
        for x in 0..wu {
            let d = y * pwu + x;
            for c in 0..3 {
                out.push(unit_to_u8(planar[c * plane + d]));
            }
            out.push(255);
        }
    }
    out
}

fn unit_to_u8(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// 有理数帧率：每秒 `num / den` 帧（例如 NTSC 为 `30000/1001`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fps {
    /// 分子。
    pub num: u32,
    /// 分母。
    pub den: u32,
}

impl Fps {
    /// 以分子、分母构造帧率。
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// 构造整数帧率，如 `Fps::integer(30)`。
    pub const fn integer(fps: u32) -> Self {
        Self { num: fps, den: 1 }
    }
}

/// 重定时方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retime {
    /// 转换到目标帧率，时长不变（如 30 → 60 fps）。
    ToFps(Fps),
    /// 按倍数慢放：每个源帧间隔细分为 `n` 份，按源帧率播放（2× / 4× 慢动作）。
    SlowMotion(u32),
}

/// 一个输出帧对应的插值位置：在源帧 `src_index` 与 `src_index + 1` 之间，比例 `t`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScheduledFrame {
    /// 起始源帧下标。
    pub src_index: usize,
    /// 插值比例，`t ∈ [0, 1)`；`0` 表示直接使用 `src_index` 帧。
    pub t: f64,
}

/// 用精确有理数计算每个输出帧对应的 `(源帧下标, t)`。
///
/// * `src`：源帧率；`retime`：目标帧率或慢放倍数；`frame_count`：源帧数。
/// * `cuts`：升序排列的镜头起始帧下标（新镜头第一帧）。跨越镜头切换的插值没有意义，
///   此时 `t` 取 `0` 并保持时间上最近的那一帧。
/// * 末帧之后无可插值的帧，落在最后一个间隔内的输出帧保持末帧。
///
/// 输出帧数为 `ceil(frame_count × 目标帧率 / 源帧率)`；慢放为 `frame_count × n`。
/// 帧率或倍数为 0 时返回空序列。
pub fn frame_schedule(
    src: Fps,
    retime: Retime,
    frame_count: usize,
    cuts: &[usize],
) -> Vec<ScheduledFrame> {
    // 输出帧 k 对应的源位置 p_k = k * a / b。
    let (a, b): (u128, u128) = match retime {
        Retime::ToFps(dst) => (
            u128::from(src.num) * u128::from(dst.den),
            u128::from(src.den) * u128::from(dst.num),
        ),
        Retime::SlowMotion(n) => (1, u128::from(n)),
    };
    if a == 0 || b == 0 || frame_count == 0 {
        return Vec::new();
    }
    let n = frame_count as u128;
    let count = (n * b).div_ceil(a);
    let mut out = Vec::with_capacity(count as usize);
    for k in 0..count {
        let pos = k * a;
        let i = (pos / b) as usize;
        let rem = pos % b;
        let frame = if rem == 0 || i + 1 >= frame_count {
            ScheduledFrame {
                src_index: i,
                t: 0.0,
            }
        } else if cuts.binary_search(&(i + 1)).is_ok() {
            let nearest = if rem * 2 < b { i } else { i + 1 };
            ScheduledFrame {
                src_index: nearest,
                t: 0.0,
            }
        } else {
            ScheduledFrame {
                src_index: i,
                t: rem as f64 / b as f64,
            }
        };
        out.push(frame);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_sizes() {
        assert_eq!(padded_size(1920, 1080), (1920, 1088));
        assert_eq!(padded_size(1, 1), (32, 32));
        assert_eq!(padded_size(64, 33), (64, 64));
        assert_eq!(padded_size(0, 0), (0, 0));
    }

    #[test]
    fn planar_roundtrip_with_padding() {
        let (w, h) = (3u32, 2u32);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| [i as u8 * 10, 100, 255 - i as u8, 255])
            .collect();
        let (t, pw, ph) = rgba_to_planar_rgb(&rgba, w, h);
        assert_eq!((pw, ph), (32, 32));
        assert_eq!(t.len(), 3 * 32 * 32);
        // 补边复制边缘像素：右下角等于原图右下像素。
        let plane = 32 * 32;
        assert!((t[plane - 1] - 50.0 / 255.0).abs() < 1e-6);
        assert_eq!(planar_rgb_to_rgba(&t, pw, ph, w, h), rgba);
    }

    #[test]
    fn schedule_30_to_60_doubles() {
        let s = frame_schedule(Fps::integer(30), Retime::ToFps(Fps::integer(60)), 10, &[]);
        assert_eq!(s.len(), 20);
        assert_eq!(
            s[4],
            ScheduledFrame {
                src_index: 2,
                t: 0.0
            }
        );
        assert_eq!(
            s[5],
            ScheduledFrame {
                src_index: 2,
                t: 0.5
            }
        );
        // 末帧之后保持。
        assert_eq!(
            s[19],
            ScheduledFrame {
                src_index: 9,
                t: 0.0
            }
        );
    }

    #[test]
    fn schedule_ntsc_rational() {
        let s = frame_schedule(
            Fps::new(30000, 1001),
            Retime::ToFps(Fps::new(60000, 1001)),
            100,
            &[],
        );
        assert_eq!(s.len(), 200);
    }

    #[test]
    fn schedule_slow_motion_4x() {
        let s = frame_schedule(Fps::integer(30), Retime::SlowMotion(4), 5, &[]);
        assert_eq!(s.len(), 20);
        assert_eq!(
            s[4],
            ScheduledFrame {
                src_index: 1,
                t: 0.0
            }
        );
        assert_eq!(
            s[6],
            ScheduledFrame {
                src_index: 1,
                t: 0.5
            }
        );
        assert!(s.iter().all(|f| (0.0..1.0).contains(&f.t)));
    }

    #[test]
    fn schedule_downsample() {
        let s = frame_schedule(Fps::integer(60), Retime::ToFps(Fps::integer(30)), 10, &[]);
        let idx: Vec<usize> = s.iter().map(|f| f.src_index).collect();
        assert_eq!(idx, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn schedule_holds_nearest_at_cut() {
        let s = frame_schedule(Fps::integer(30), Retime::SlowMotion(4), 3, &[1]);
        assert_eq!(
            s[1],
            ScheduledFrame {
                src_index: 0,
                t: 0.0
            }
        );
        assert_eq!(
            s[3],
            ScheduledFrame {
                src_index: 1,
                t: 0.0
            }
        );
        // 第二个间隔不跨镜头，正常插值。
        assert_eq!(
            s[5],
            ScheduledFrame {
                src_index: 1,
                t: 0.25
            }
        );
    }

    #[test]
    fn schedule_zero_rate_is_empty() {
        assert!(frame_schedule(Fps::integer(30), Retime::SlowMotion(0), 5, &[]).is_empty());
    }
}
