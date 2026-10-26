//! Spleeter 2stems（人声 / 伴奏）模型的前处理与后处理。
//!
//! 前处理：每声道 STFT（n_fft 4096、hop 1024），取前 1024 个频点的幅度，
//! 帧数补零到 512 的倍数，组成形状 `[2, frames/512, 512, 1024]` 的行优先张量
//! （单声道输入复制声道 0）。推理由调用方闭包完成，返回人声与伴奏幅度估计。
//! 后处理：按维纳式软掩码乘回复数谱（1024 以上频点掩码为 0），再 iSTFT。

use crate::stft::{Spectrogram, StftConfig, istft, stft};

/// STFT 帧长。
pub const N_FFT: usize = 4096;
/// STFT 帧移。
pub const HOP: usize = 1024;
/// 送入模型的频点数。
pub const F_BINS: usize = 1024;
/// 模型时间段长度（帧）。
pub const T_SEGMENT: usize = 512;

const EPS: f32 = 1e-10;

/// 由人声幅度 `v` 与伴奏幅度 `a` 计算人声软掩码：
/// `(v² + 5e-11) / (v² + a² + 1e-10)`。伴奏掩码为 `vocal_mask(a, v)`，两者之和为 1。
pub fn vocal_mask(v: f32, a: f32) -> f32 {
    (v * v + EPS / 2.0) / (v * v + a * a + EPS)
}

/// 前处理结果：模型输入张量及重建所需的复数谱。
#[derive(Clone, Debug)]
pub struct SpleeterInput {
    /// 幅度张量，形状为 `shape`，行优先。
    pub data: Vec<f32>,
    /// `[2, 段数, 512, 1024]`。
    pub shape: [usize; 4],
    /// 两个声道的复数谱（单声道时为同一谱的两份拷贝）。
    pub spectra: Vec<Spectrogram>,
    /// 原始输入声道数（1 或 2），输出按此声道数返回。
    pub input_channels: usize,
}

/// 分离结果，每个分量的声道数与输入相同，长度与输入相同。
#[derive(Clone, Debug, Default)]
pub struct Separated {
    /// 人声，按声道排列。
    pub vocals: Vec<Vec<f32>>,
    /// 伴奏，按声道排列。
    pub accompaniment: Vec<Vec<f32>>,
}

/// 前处理：计算 STFT 并组装模型输入。
///
/// # Panics
///
/// 声道数不是 1 或 2 时 panic。
pub fn prepare(channels: &[Vec<f32>]) -> SpleeterInput {
    assert!(
        matches!(channels.len(), 1 | 2),
        "Spleeter 只接受 1 或 2 个声道"
    );
    let cfg = StftConfig {
        n_fft: N_FFT,
        hop: HOP,
    };
    let first = stft(&channels[0], cfg);
    let second = match channels.get(1) {
        Some(ch) => stft(ch, cfg),
        None => first.clone(),
    };
    let spectra = vec![first, second];
    let frames = spectra.iter().map(|s| s.frames.len()).max().unwrap_or(0);
    let padded = frames.div_ceil(T_SEGMENT).max(1) * T_SEGMENT;
    let segments = padded / T_SEGMENT;
    let mut data = vec![0f32; 2 * padded * F_BINS];
    for (c, spec) in spectra.iter().enumerate() {
        for (t, frame) in spec.frames.iter().enumerate() {
            let base = (c * padded + t) * F_BINS;
            for (d, z) in data[base..base + F_BINS].iter_mut().zip(&frame[..F_BINS]) {
                *d = z.norm();
            }
        }
    }
    SpleeterInput {
        data,
        shape: [2, segments, T_SEGMENT, F_BINS],
        spectra,
        input_channels: channels.len(),
    }
}

/// 后处理：由模型输出的人声 / 伴奏幅度计算掩码，乘回复数谱并 iSTFT。
///
/// # Panics
///
/// 模型输出长度与输入张量不一致时 panic。
pub fn postprocess(input: &SpleeterInput, vocals_mag: &[f32], accomp_mag: &[f32]) -> Separated {
    assert_eq!(vocals_mag.len(), input.data.len(), "人声输出长度不匹配");
    assert_eq!(accomp_mag.len(), input.data.len(), "伴奏输出长度不匹配");
    let padded = input.shape[1] * input.shape[2];
    let mut out = Separated::default();
    for c in 0..input.input_channels {
        let spec = &input.spectra[c];
        let mut voc = spec.clone();
        let mut acc = spec.clone();
        for (t, (vf, af)) in voc.frames.iter_mut().zip(acc.frames.iter_mut()).enumerate() {
            let base = (c * padded + t) * F_BINS;
            for (f, (vz, az)) in vf.iter_mut().zip(af.iter_mut()).enumerate() {
                if f < F_BINS {
                    let v = vocals_mag[base + f];
                    let a = accomp_mag[base + f];
                    *vz = vz.scale(vocal_mask(v, a));
                    *az = az.scale(vocal_mask(a, v));
                } else {
                    *vz = vz.scale(0.0);
                    *az = az.scale(0.0);
                }
            }
        }
        out.vocals.push(istft(&voc));
        out.accompaniment.push(istft(&acc));
    }
    out
}

/// 一步完成人声分离：前处理 → `infer(张量, 形状)` → 后处理。
///
/// `infer` 返回 `(人声幅度, 伴奏幅度)`，形状与输入张量相同。
pub fn separate<F>(channels: &[Vec<f32>], mut infer: F) -> Separated
where
    F: FnMut(&[f32], [usize; 4]) -> (Vec<f32>, Vec<f32>),
{
    let input = prepare(channels);
    let (v, a) = infer(&input.data, input.shape);
    postprocess(&input, &v, &a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_sum_to_one() {
        for &(v, a) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (0.3, 0.7), (5.0, 1e-3)] {
            let s = vocal_mask(v, a) + vocal_mask(a, v);
            assert!((s - 1.0).abs() < 1e-6, "{v} {a} -> {s}");
        }
    }

    #[test]
    fn shape_and_identity_separation() {
        let sr = 44_100.0f32;
        let x: Vec<f32> = (0..44_100)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin())
            .collect();
        let mut seen = [0usize; 4];
        let out = separate(std::slice::from_ref(&x), |data, shape| {
            seen = shape;
            assert_eq!(data.len(), shape.iter().product::<usize>());
            (data.to_vec(), vec![0.0; data.len()])
        });
        assert_eq!(seen, [2, 1, 512, 1024]);
        assert_eq!(out.vocals.len(), 1);
        assert_eq!(out.vocals[0].len(), x.len());
        let r = N_FFT..x.len() - N_FFT;
        let err = x[r.clone()]
            .iter()
            .zip(&out.vocals[0][r.clone()])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "人声重建误差 {err}");
        let acc = out.accompaniment[0][r]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(acc < 1e-3, "伴奏应接近 0: {acc}");
    }
}
