// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! 智能重构图纯逻辑：主体中心估计、轨迹平滑、裁剪偏移与关键帧精简。
//!
//! 本模块不依赖任何推理后端，在所有特性组合下均可编译与测试。

/// 掩码加权质心（像素坐标，像素中心为 `x + 0.5`）。
///
/// `mask` 为 `w×h` 的按行权重（如分割概率），非正值与非有限值忽略。
/// 掩码为空或全零时返回 `None`。
pub fn mask_centroid(mask: &[f32], w: u32, h: u32) -> Option<(f32, f32)> {
    let wu = w as usize;
    let total = wu * h as usize;
    let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    for (idx, &m) in mask.iter().take(total).enumerate() {
        if m > 0.0 && m.is_finite() {
            let m = f64::from(m);
            sx += ((idx % wu) as f64 + 0.5) * m;
            sy += ((idx / wu) as f64 + 0.5) * m;
            sw += m;
        }
    }
    (sw > 0.0).then(|| ((sx / sw) as f32, (sy / sw) as f32))
}

/// 灰度梯度幅值加权质心，作为没有分割掩码时的显著性兜底（像素坐标）。
///
/// 梯度为中心差分 `|gx| + |gy|`。画面过小或完全平坦时返回画面中心。
///
/// # Panics
/// 当 `gray.len() < w * h` 时 panic。
pub fn saliency_centroid(gray: &[u8], w: u32, h: u32) -> (f32, f32) {
    let (wu, hu) = (w as usize, h as usize);
    assert!(gray.len() >= wu * hu, "灰度缓冲长度不足");
    let center = (w as f32 / 2.0, h as f32 / 2.0);
    if wu < 3 || hu < 3 {
        return center;
    }
    let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..hu - 1 {
        for x in 1..wu - 1 {
            let i = y * wu + x;
            let gx = i32::from(gray[i + 1]) - i32::from(gray[i - 1]);
            let gy = i32::from(gray[i + wu]) - i32::from(gray[i - wu]);
            let m = f64::from(gx.abs() + gy.abs());
            sx += (x as f64 + 0.5) * m;
            sy += (y as f64 + 0.5) * m;
            sw += m;
        }
    }
    if sw > 0.0 {
        ((sx / sw) as f32, (sy / sw) as f32)
    } else {
        center
    }
}

/// 零相位高斯平滑：对称核卷积（非因果，无滞后），边界按端点延拓。
///
/// `sigma` 以帧为单位；`sigma <= 0` 或 NaN 时原样返回。
pub fn gaussian_smooth(values: &[f32], sigma: f32) -> Vec<f32> {
    if values.is_empty() || sigma.is_nan() || sigma <= 0.0 {
        return values.to_vec();
    }
    let r = ((sigma * 3.0).ceil() as usize).min(values.len()).max(1);
    let kernel: Vec<f32> = (0..=2 * r)
        .map(|i| {
            let d = i as f32 - r as f32;
            (-d * d / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let last = values.len() as isize - 1;
    (0..values.len())
        .map(|i| {
            let (mut acc, mut wsum) = (0.0f32, 0.0f32);
            for (k, &kw) in kernel.iter().enumerate() {
                let j = (i as isize + k as isize - r as isize).clamp(0, last) as usize;
                acc += values[j] * kw;
                wsum += kw;
            }
            acc / wsum
        })
        .collect()
}

/// 死区（回差）滤波：偏离保持值不超过 `dead_zone` 时保持不动，超过时只追随超出的部分。
///
/// 用于消除主体轻微抖动引起的镜头漂移；输出连续无跳变。
pub fn apply_dead_zone(values: &[f32], dead_zone: f32) -> Vec<f32> {
    let dz = if dead_zone.is_finite() {
        dead_zone.max(0.0)
    } else {
        0.0
    };
    let mut out = Vec::with_capacity(values.len());
    let Some(&first) = values.first() else {
        return out;
    };
    let mut held = first;
    for &v in values {
        let delta = v - held;
        if delta.abs() > dz {
            held = v - dz.copysign(delta);
        }
        out.push(held);
    }
    out
}

/// 单轴中心轨迹平滑：高斯 → 死区 → 高斯，得到无滞后、无抖动的平滑镜头路径。
pub fn smooth_path(values: &[f32], sigma: f32, dead_zone: f32) -> Vec<f32> {
    gaussian_smooth(
        &apply_dead_zone(&gaussian_smooth(values, sigma), dead_zone),
        sigma,
    )
}

/// 在 `src_w×src_h` 画面中，能铺满目标宽高比 `target_aspect`（宽 / 高）的最大裁剪尺寸。
///
/// 宽高比非正或非有限、或源尺寸为 0 时返回源尺寸。
pub fn crop_size(src_w: u32, src_h: u32, target_aspect: f32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 || !target_aspect.is_finite() || target_aspect <= 0.0 {
        return (src_w, src_h);
    }
    let a = f64::from(target_aspect);
    let src_aspect = f64::from(src_w) / f64::from(src_h);
    if src_aspect > a {
        let cw = ((f64::from(src_h) * a).round() as u32).clamp(1, src_w);
        (cw, src_h)
    } else {
        let ch = ((f64::from(src_w) / a).round() as u32).clamp(1, src_h);
        (src_w, ch)
    }
}

/// 逐帧裁剪方案：裁剪框尺寸与每帧左上角偏移。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CropPlan {
    /// 裁剪框宽度（像素）。
    pub width: u32,
    /// 裁剪框高度（像素）。
    pub height: u32,
    /// 每帧裁剪框左上角 `(x, y)`，保证 `x + width <= src_w` 且 `y + height <= src_h`。
    pub offsets: Vec<(u32, u32)>,
}

/// 由每帧主体中心（像素坐标）计算铺满目标宽高比的裁剪偏移。
///
/// 偏移以中心对齐后按覆盖约束夹紧，裁剪框永不越出画面；非有限中心按画面居中处理。
pub fn fill_offsets(
    src_w: u32,
    src_h: u32,
    target_aspect: f32,
    centers: &[(f32, f32)],
) -> CropPlan {
    let (cw, ch) = crop_size(src_w, src_h, target_aspect);
    let (max_x, max_y) = (src_w - cw, src_h - ch);
    CropPlan {
        width: cw,
        height: ch,
        offsets: centers
            .iter()
            .map(|&(cx, cy)| (clamp_offset(cx, cw, max_x), clamp_offset(cy, ch, max_y)))
            .collect(),
    }
}

fn clamp_offset(center: f32, size: u32, max: u32) -> u32 {
    let c = if center.is_finite() {
        f64::from(center)
    } else {
        (f64::from(size) + f64::from(max)) / 2.0
    };
    (c - f64::from(size) / 2.0)
        .round()
        .clamp(0.0, f64::from(max)) as u32
}

fn perpendicular_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt();
    if len == 0.0 {
        return ((p.0 - a.0).powi(2) + (p.1 - a.1).powi(2)).sqrt();
    }
    ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / len
}

/// Ramer–Douglas–Peucker 折线精简，返回保留点的下标（升序，必含首尾）。
///
/// 典型用法：`points` 为 `(帧号, 偏移)`，`epsilon` 为允许的像素误差，结果作为关键帧。
/// 迭代实现，不会因长序列而栈溢出。
pub fn rdp(points: &[(f32, f32)], epsilon: f32) -> Vec<usize> {
    let n = points.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((s, e)) = stack.pop() {
        if e <= s + 1 {
            continue;
        }
        let (mut idx, mut dmax) = (s, -1.0f32);
        for (i, &p) in points.iter().enumerate().take(e).skip(s + 1) {
            let d = perpendicular_distance(p, points[s], points[e]);
            if d > dmax {
                dmax = d;
                idx = i;
            }
        }
        if dmax > epsilon {
            keep[idx] = true;
            stack.push((s, idx));
            stack.push((idx, e));
        }
    }
    keep.iter()
        .enumerate()
        .filter_map(|(i, &k)| k.then_some(i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_centroid_basic() {
        let mut m = vec![0.0f32; 16];
        m[4 + 3] = 1.0;
        assert_eq!(mask_centroid(&m, 4, 4), Some((3.5, 1.5)));
        assert_eq!(mask_centroid(&[0.0; 16], 4, 4), None);
    }

    #[test]
    fn saliency_finds_edge() {
        let (w, h) = (20u32, 10u32);
        let gray: Vec<u8> = (0..w * h)
            .map(|i| if i % w < 10 { 0 } else { 255 })
            .collect();
        let (cx, _) = saliency_centroid(&gray, w, h);
        assert!((cx - 10.0).abs() < 1.0, "cx = {cx}");
        assert_eq!(saliency_centroid(&[7; 200], w, h), (10.0, 5.0));
    }

    #[test]
    fn crop_sizes() {
        assert_eq!(crop_size(1920, 1080, 9.0 / 16.0), (608, 1080));
        let (w, h) = crop_size(1080, 1920, 16.0 / 9.0);
        assert!(w == 1080 && (607..=608).contains(&h));
        assert_eq!(crop_size(1920, 1080, 0.0), (1920, 1080));
    }

    #[test]
    fn offsets_stay_in_bounds() {
        let centers = [
            (-100.0, -100.0),
            (5000.0, 5000.0),
            (960.0, 540.0),
            (f32::NAN, f32::INFINITY),
        ];
        let plan = fill_offsets(1920, 1080, 9.0 / 16.0, &centers);
        assert_eq!((plan.width, plan.height), (608, 1080));
        for &(x, y) in &plan.offsets {
            assert!(x + plan.width <= 1920 && y + plan.height <= 1080);
        }
        assert_eq!(plan.offsets[0], (0, 0));
        assert_eq!(plan.offsets[1], (1312, 0));
        assert_eq!(plan.offsets[2], (656, 0));
        assert_eq!(plan.offsets[3], (656, 0));
    }

    #[test]
    fn smoothing_behaviour() {
        let constant = vec![5.0f32; 20];
        assert!(
            gaussian_smooth(&constant, 3.0)
                .iter()
                .all(|v| (v - 5.0).abs() < 1e-5)
        );
        let jitter: Vec<f32> = (0..40)
            .map(|i| 100.0 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let dz = apply_dead_zone(&jitter, 3.0);
        assert!(dz.iter().all(|&v| v == dz[0]));
        let step: Vec<f32> = (0..60).map(|i| if i < 30 { 0.0 } else { 100.0 }).collect();
        let s = smooth_path(&step, 2.0, 5.0);
        assert!(s[0].abs() < 1.0 && (s[59] - 95.0).abs() < 1.0);
        assert!(s.windows(2).all(|p| p[1] >= p[0] - 1e-4));
    }

    #[test]
    fn rdp_simplifies() {
        let line: Vec<(f32, f32)> = (0..100).map(|i| (i as f32, 2.0 * i as f32)).collect();
        assert_eq!(rdp(&line, 0.5), vec![0, 99]);
        let v: Vec<(f32, f32)> = (0..21)
            .map(|i| (i as f32, (i as f32 - 10.0).abs() * 10.0))
            .collect();
        assert_eq!(rdp(&v, 0.5), vec![0, 10, 20]);
        assert_eq!(rdp(&v[..2], 0.5), vec![0, 1]);
    }
}
