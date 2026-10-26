//! 短时傅里叶变换（STFT）与逆变换（iSTFT）。
//!
//! 使用周期 Hann 窗、不居中（第一帧从样本 0 开始），末尾补零使所有样本被覆盖；
//! 逆变换采用加窗重叠相加，并除以窗平方和归一化。

use std::f64::consts::PI;

use crate::fft::{Complex, irfft, rfft};

/// STFT 参数。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StftConfig {
    /// 帧长（FFT 点数），必须是 2 的幂。
    pub n_fft: usize,
    /// 帧移（样本数），必须大于 0。
    pub hop: usize,
}

/// 复数谱图。
#[derive(Clone, Debug)]
pub struct Spectrogram {
    /// 生成该谱图的参数。
    pub config: StftConfig,
    /// 原始信号长度（样本数），iSTFT 按此截断。
    pub len: usize,
    /// 逐帧频谱，每帧 `n_fft/2+1` 个频点。
    pub frames: Vec<Vec<Complex>>,
}

/// 长度为 `n` 的周期 Hann 窗。
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()) as f32)
        .collect()
}

/// 长度为 `len` 的信号在给定参数下的帧数（不居中、末尾补零）。
pub fn frame_count(len: usize, cfg: StftConfig) -> usize {
    if len <= cfg.n_fft {
        1
    } else {
        (len - cfg.n_fft).div_ceil(cfg.hop) + 1
    }
}

/// 计算实信号的 STFT。
///
/// # Panics
///
/// `n_fft` 不是 2 的幂或 `hop == 0` 时 panic。
pub fn stft(signal: &[f32], cfg: StftConfig) -> Spectrogram {
    assert!(cfg.n_fft.is_power_of_two(), "n_fft 必须是 2 的幂");
    assert!(cfg.hop > 0, "hop 必须大于 0");
    let win = hann(cfg.n_fft);
    let n = frame_count(signal.len(), cfg);
    let mut frames = Vec::with_capacity(n);
    let mut buf = vec![0f32; cfg.n_fft];
    for f in 0..n {
        let start = f * cfg.hop;
        for (i, (b, w)) in buf.iter_mut().zip(&win).enumerate() {
            *b = signal.get(start + i).copied().unwrap_or(0.0) * w;
        }
        frames.push(rfft(&buf));
    }
    Spectrogram {
        config: cfg,
        len: signal.len(),
        frames,
    }
}

/// 由谱图重建时域信号（加窗重叠相加，按窗平方和归一化），长度为 `spec.len`。
pub fn istft(spec: &Spectrogram) -> Vec<f32> {
    let cfg = spec.config;
    let win = hann(cfg.n_fft);
    let total = spec.frames.len().saturating_sub(1) * cfg.hop + cfg.n_fft;
    let mut out = vec![0f32; total];
    let mut norm = vec![0f32; total];
    for (f, frame) in spec.frames.iter().enumerate() {
        let x = irfft(frame, cfg.n_fft);
        let start = f * cfg.hop;
        let o = &mut out[start..start + cfg.n_fft];
        let nm = &mut norm[start..start + cfg.n_fft];
        for (((o, nm), &s), &w) in o.iter_mut().zip(nm.iter_mut()).zip(&x).zip(&win) {
            *o += s * w;
            *nm += w * w;
        }
    }
    for (o, &nv) in out.iter_mut().zip(&norm) {
        if nv > 1e-8 {
            *o /= nv;
        }
    }
    out.truncate(spec.len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_istft_reconstructs() {
        let mut s = 12345u32;
        let x: Vec<f32> = (0..10_000)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        for cfg in [
            StftConfig {
                n_fft: 512,
                hop: 128,
            },
            StftConfig {
                n_fft: 1024,
                hop: 256,
            },
        ] {
            let y = istft(&stft(&x, cfg));
            assert_eq!(y.len(), x.len());
            let max_err = x[cfg.n_fft..x.len() - cfg.n_fft]
                .iter()
                .zip(&y[cfg.n_fft..x.len() - cfg.n_fft])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 1e-4, "重建误差过大: {max_err}");
        }
    }

    #[test]
    fn frame_count_covers_signal() {
        let cfg = StftConfig { n_fft: 8, hop: 4 };
        assert_eq!(frame_count(3, cfg), 1);
        assert_eq!(frame_count(8, cfg), 1);
        assert_eq!(frame_count(9, cfg), 2);
        assert_eq!(frame_count(12, cfg), 2);
        assert_eq!(frame_count(13, cfg), 3);
    }
}
