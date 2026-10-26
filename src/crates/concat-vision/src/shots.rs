// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! 镜头切分纯逻辑：帧签名、帧差与自适应阈值的硬切检测（含闪光抑制）。
//!
//! 本模块不依赖任何推理后端，在所有特性组合下均可编译与测试。

/// 亮度直方图分箱数。
pub const LUMA_BINS: usize = 32;
/// 每个色度通道（U / V）的直方图分箱数。
pub const CHROMA_BINS: usize = 16;
/// 灰度缩略图边长（`THUMB_SIZE × THUMB_SIZE`）。
pub const THUMB_SIZE: usize = 8;

/// 一帧画面的紧凑签名，用于计算帧差。
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSignature {
    /// 归一化（总和为 1）的亮度直方图。
    pub luma_hist: [f32; LUMA_BINS],
    /// 归一化的 U（蓝色差）直方图。
    pub u_hist: [f32; CHROMA_BINS],
    /// 归一化的 V（红色差）直方图。
    pub v_hist: [f32; CHROMA_BINS],
    /// 8×8 灰度缩略图，按行存储，取值 0..1。
    pub thumb: [f32; THUMB_SIZE * THUMB_SIZE],
}

impl FrameSignature {
    /// 由 `w×h` 的 RGBA8 缓冲计算签名（BT.601 YUV）。空画面得到全零签名。
    ///
    /// # Panics
    /// 当 `rgba.len() < w * h * 4` 时 panic。
    pub fn from_rgba(rgba: &[u8], w: u32, h: u32) -> Self {
        let (wu, hu) = (w as usize, h as usize);
        let total = wu * hu;
        assert!(rgba.len() >= total * 4, "RGBA 缓冲长度不足");
        let mut sig = FrameSignature {
            luma_hist: [0.0; LUMA_BINS],
            u_hist: [0.0; CHROMA_BINS],
            v_hist: [0.0; CHROMA_BINS],
            thumb: [0.0; THUMB_SIZE * THUMB_SIZE],
        };
        if total == 0 {
            return sig;
        }
        let mut cell_sum = [0.0f64; THUMB_SIZE * THUMB_SIZE];
        let mut cell_cnt = [0u32; THUMB_SIZE * THUMB_SIZE];
        for (idx, px) in rgba.chunks_exact(4).take(total).enumerate() {
            let (r, g, b) = (f32::from(px[0]), f32::from(px[1]), f32::from(px[2]));
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let u = ((b - y) * 0.564 + 128.0).clamp(0.0, 255.0);
            let v = ((r - y) * 0.713 + 128.0).clamp(0.0, 255.0);
            sig.luma_hist[bin(y, LUMA_BINS)] += 1.0;
            sig.u_hist[bin(u, CHROMA_BINS)] += 1.0;
            sig.v_hist[bin(v, CHROMA_BINS)] += 1.0;
            let (x, yy) = (idx % wu, idx / wu);
            let cell = (yy * THUMB_SIZE / hu) * THUMB_SIZE + x * THUMB_SIZE / wu;
            cell_sum[cell] += f64::from(y);
            cell_cnt[cell] += 1;
        }
        let inv = 1.0 / total as f32;
        sig.luma_hist.iter_mut().for_each(|v| *v *= inv);
        sig.u_hist.iter_mut().for_each(|v| *v *= inv);
        sig.v_hist.iter_mut().for_each(|v| *v *= inv);
        for ((t, s), c) in sig.thumb.iter_mut().zip(cell_sum).zip(cell_cnt) {
            if c > 0 {
                *t = (s / f64::from(c) / 255.0) as f32;
            }
        }
        sig
    }
}

fn bin(v: f32, bins: usize) -> usize {
    ((v.max(0.0) as usize) * bins / 256).min(bins - 1)
}

fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// 两帧签名之间的差异，取值 0..1（0 表示相同）。
///
/// 加权组合：亮度直方图距离 35%、色度直方图距离 15%、缩略图平均绝对差 50%。
/// 直方图对运动不敏感，缩略图对构图变化敏感，两者互补。
pub fn frame_diff(a: &FrameSignature, b: &FrameSignature) -> f32 {
    let luma = l1(&a.luma_hist, &b.luma_hist) / 2.0;
    let chroma = (l1(&a.u_hist, &b.u_hist) + l1(&a.v_hist, &b.v_hist)) / 4.0;
    let thumb = l1(&a.thumb, &b.thumb) / (THUMB_SIZE * THUMB_SIZE) as f32;
    (0.35 * luma + 0.15 * chroma + 0.5 * thumb).clamp(0.0, 1.0)
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    }
}

/// 从帧差序列检测硬切，返回每个新镜头第一帧的下标（升序）。
///
/// * `diffs[i]` 为第 `i` 帧与第 `i + 1` 帧之间的差（见 [`frame_diff`]），
///   检出时返回 `i + 1`。
/// * 自适应阈值 = max(局部中位数 × 系数, 绝对下限)；局部窗口为前后约 0.5 秒（不含自身）。
///   `sensitivity ∈ [0, 1]`：越大越敏感（系数 6.0 → 1.5，下限 0.35 → 0.08）。
/// * 闪光抑制：突变后约 0.1 秒内又出现突变（画面回落）视为闪光，整组忽略。
/// * `min_shot_secs`：镜头最短时长，短于它的切点被丢弃（从第 0 帧起算）。
///
/// `fps` 非正或非有限时按 25 处理。
pub fn detect_cuts(diffs: &[f32], fps: f64, sensitivity: f32, min_shot_secs: f64) -> Vec<usize> {
    let n = diffs.len();
    if n == 0 {
        return Vec::new();
    }
    let s = if sensitivity.is_nan() {
        0.5
    } else {
        sensitivity.clamp(0.0, 1.0)
    };
    let fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        25.0
    };
    let factor = 6.0 - 4.5 * s;
    let floor = 0.35 - 0.27 * s;
    let radius = ((fps * 0.5).round() as usize).max(2);
    let flash_win = ((fps * 0.1).round() as usize).max(1);
    let min_gap = if min_shot_secs.is_finite() {
        (min_shot_secs.max(0.0) * fps).round() as usize
    } else {
        0
    };

    let mut candidates = Vec::new();
    let mut window = Vec::with_capacity(radius * 2);
    for (i, &d) in diffs.iter().enumerate() {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(n);
        window.clear();
        window.extend(
            diffs[lo..hi]
                .iter()
                .enumerate()
                .filter(|(j, _)| lo + j != i)
                .map(|(_, &x)| x),
        );
        let thr = (median(&mut window) * factor).max(floor);
        if d > thr {
            candidates.push(i);
        }
    }

    let mut cuts = Vec::new();
    let mut last = 0usize;
    let mut k = 0;
    while k < candidates.len() {
        let mut j = k + 1;
        while j < candidates.len() && candidates[j] - candidates[j - 1] <= flash_win {
            j += 1;
        }
        if j > k + 1 {
            // 突变后马上回落：闪光，整组忽略。
            k = j;
            continue;
        }
        let cut = candidates[k] + 1;
        if cut - last >= min_gap {
            cuts.push(cut);
            last = cut;
        }
        k += 1;
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 32;
    const H: u32 = 24;

    fn frame(color: [u8; 3], seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut out = Vec::with_capacity((W * H * 4) as usize);
        for _y in 0..H {
            for x in 0..W {
                for &c in &color {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    let noise = ((state >> 16) % 9) as i32 - 4;
                    let v = i32::from(c) + x as i32 / 2 + noise;
                    out.push(v.clamp(0, 255) as u8);
                }
                out.push(255);
            }
        }
        out
    }

    fn diffs(frames: &[Vec<u8>]) -> Vec<f32> {
        let sigs: Vec<FrameSignature> = frames
            .iter()
            .map(|f| FrameSignature::from_rgba(f, W, H))
            .collect();
        sigs.windows(2).map(|p| frame_diff(&p[0], &p[1])).collect()
    }

    const A: [u8; 3] = [40, 60, 120];
    const B: [u8; 3] = [220, 140, 40];
    const C: [u8; 3] = [20, 20, 20];

    #[test]
    fn identical_frames_have_zero_diff() {
        let f = frame(A, 1);
        let s = FrameSignature::from_rgba(&f, W, H);
        assert_eq!(frame_diff(&s, &s), 0.0);
        assert!((s.luma_hist.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn detects_hard_cut() {
        let frames: Vec<Vec<u8>> = (0..60)
            .map(|i| frame(if i < 30 { A } else { B }, i))
            .collect();
        assert_eq!(detect_cuts(&diffs(&frames), 30.0, 0.5, 0.5), vec![30]);
    }

    #[test]
    fn flash_is_not_a_cut() {
        let frames: Vec<Vec<u8>> = (0..40)
            .map(|i| frame(if i == 20 { [255, 255, 255] } else { A }, i))
            .collect();
        assert!(detect_cuts(&diffs(&frames), 30.0, 0.5, 0.5).is_empty());
    }

    #[test]
    fn short_shot_is_dropped() {
        let frames: Vec<Vec<u8>> = (0..65)
            .map(|i| {
                let c = if i < 30 {
                    A
                } else if i < 35 {
                    B
                } else {
                    C
                };
                frame(c, i)
            })
            .collect();
        let d = diffs(&frames);
        assert_eq!(detect_cuts(&d, 30.0, 0.5, 0.5), vec![30]);
        assert_eq!(detect_cuts(&d, 30.0, 0.5, 0.0), vec![30, 35]);
    }

    #[test]
    fn empty_input() {
        assert!(detect_cuts(&[], 30.0, 0.5, 0.5).is_empty());
    }
}
