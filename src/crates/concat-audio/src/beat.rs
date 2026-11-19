//! 节拍检测：谱通量 onset 包络 → 自相关估计 BPM → 动态规划节拍跟踪，
//! 并按每 N 拍给出剪辑切点（音乐卡点）。

use crate::stft::{StftConfig, stft};

/// 节拍检测参数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeatConfig {
    /// onset 分析的 STFT 帧长（2 的幂）。
    pub n_fft: usize,
    /// onset 分析的帧移（样本数）。
    pub hop: usize,
    /// 允许的最小 BPM。
    pub min_bpm: f64,
    /// 允许的最大 BPM。
    pub max_bpm: f64,
    /// 速度先验中心（BPM），用于抑制倍频 / 半频误判。
    pub prior_bpm: f64,
    /// 动态规划的节奏紧度，越大越贴近估计速度。
    pub tightness: f64,
}

impl Default for BeatConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop: 512,
            min_bpm: 60.0,
            max_bpm: 200.0,
            prior_bpm: 120.0,
            tightness: 100.0,
        }
    }
}

/// onset 强度包络。
#[derive(Clone, Debug)]
pub struct OnsetEnvelope {
    /// 每帧 onset 强度（已按标准差归一化）。
    pub values: Vec<f32>,
    /// 帧率（帧/秒）。
    pub frame_rate: f64,
    /// 第 0 帧对应的时间（秒，帧中心）。
    pub time_offset: f64,
}

/// 节拍检测结果。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeatResult {
    /// 估计速度（BPM），信号太短时为 0。
    pub bpm: f64,
    /// 节拍时间（秒），升序。
    pub beats: Vec<f64>,
}

/// 计算谱通量 onset 包络（对数幅度的正向差分之和）。
pub fn onset_envelope(samples: &[f32], sample_rate: u32, cfg: &BeatConfig) -> OnsetEnvelope {
    let sr = f64::from(sample_rate.max(1));
    let spec = stft(
        samples,
        StftConfig {
            n_fft: cfg.n_fft,
            hop: cfg.hop,
        },
    );
    let logmag: Vec<Vec<f32>> = spec
        .frames
        .iter()
        .map(|f| f.iter().map(|z| (1.0 + 100.0 * z.norm()).ln()).collect())
        .collect();
    let mut values = vec![0f32; logmag.len()];
    for (v, w) in values.iter_mut().skip(1).zip(logmag.windows(2)) {
        *v = w[1].iter().zip(&w[0]).map(|(c, p)| (c - p).max(0.0)).sum();
    }
    let n = values.len().max(1) as f32;
    let mean = values.iter().sum::<f32>() / n;
    let std = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n).sqrt();
    if std > 0.0 {
        values.iter_mut().for_each(|v| *v /= std);
    }
    OnsetEnvelope {
        values,
        frame_rate: sr / cfg.hop as f64,
        time_offset: cfg.n_fft as f64 / 2.0 / sr,
    }
}

/// 用 onset 包络的自相关（乘以对数高斯速度先验）估计 BPM；包络太短时返回 0。
pub fn estimate_bpm(env: &OnsetEnvelope, cfg: &BeatConfig) -> f64 {
    let n = env.values.len();
    if n < 4 {
        return 0.0;
    }
    let mean = env.values.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let x: Vec<f64> = env.values.iter().map(|&v| f64::from(v) - mean).collect();
    let fr = env.frame_rate;
    let min_lag = ((60.0 * fr / cfg.max_bpm).floor() as usize).max(1);
    let max_lag = ((60.0 * fr / cfg.min_bpm).ceil() as usize).min(n - 2);
    if max_lag <= min_lag {
        return 0.0;
    }
    let score = |lag: usize| -> f64 {
        let ac = x.iter().zip(&x[lag..]).map(|(a, b)| a * b).sum::<f64>() / (n - lag) as f64;
        let bpm = 60.0 * fr / lag as f64;
        let w = (-0.5 * (bpm / cfg.prior_bpm).log2().powi(2)).exp();
        ac * w
    };
    let scores: Vec<f64> = (min_lag - 1..=max_lag + 1).map(score).collect();
    let mut best = 1;
    for i in 1..scores.len() - 1 {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    let (l, c, r) = (scores[best - 1], scores[best], scores[best + 1]);
    let denom = l - 2.0 * c + r;
    let delta = if denom < 0.0 {
        (0.5 * (l - r) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let lag = (min_lag - 1 + best) as f64 + delta;
    60.0 * fr / lag
}

/// 给定速度，用动态规划（Ellis 2007）在 onset 包络上跟踪节拍，返回节拍时间（秒）。
pub fn track_beats(env: &OnsetEnvelope, bpm: f64, cfg: &BeatConfig) -> Vec<f64> {
    let n = env.values.len();
    if n == 0 || bpm <= 0.0 {
        return Vec::new();
    }
    let period = 60.0 * env.frame_rate / bpm;
    let lo = ((period / 2.0).round() as usize).max(1);
    let hi = ((period * 2.0).round() as usize).max(lo);
    let mut score = vec![0f64; n];
    let mut back = vec![usize::MAX; n];
    for t in 0..n {
        let mut best = f64::NEG_INFINITY;
        let mut arg = usize::MAX;
        if t >= lo {
            let start = t.saturating_sub(hi);
            for (prev, &sp) in score.iter().enumerate().take(t - lo + 1).skip(start) {
                let d = ((t - prev) as f64 / period).ln();
                let s = sp - cfg.tightness * d * d;
                if s > best {
                    best = s;
                    arg = prev;
                }
            }
        }
        let local = f64::from(env.values[t]);
        if arg != usize::MAX && best > 0.0 {
            score[t] = local + best;
            back[t] = arg;
        } else {
            score[t] = local;
        }
    }
    let tail = n.saturating_sub(period.ceil() as usize);
    let mut t = tail;
    for i in tail..n {
        if score[i] > score[t] {
            t = i;
        }
    }
    let mut frames = vec![t];
    while back[t] != usize::MAX {
        t = back[t];
        frames.push(t);
    }
    frames.reverse();
    frames
        .into_iter()
        .map(|f| env.time_offset + f as f64 / env.frame_rate)
        .collect()
}

/// 完整节拍检测：onset 包络 → BPM → 节拍时间。
pub fn detect_beats(samples: &[f32], sample_rate: u32, cfg: &BeatConfig) -> BeatResult {
    let env = onset_envelope(samples, sample_rate, cfg);
    let bpm = estimate_bpm(&env, cfg);
    let beats = track_beats(&env, bpm, cfg);
    BeatResult { bpm, beats }
}

/// 每 `every_n` 拍取一个切点（从第一拍开始）。`every_n` 为 0 时按 1 处理。
pub fn cut_points(beats: &[f64], every_n: usize) -> Vec<f64> {
    beats.iter().step_by(every_n.max(1)).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn click_track(sr: u32, secs: f64, bpm: f64) -> Vec<f32> {
        let n = (f64::from(sr) * secs) as usize;
        let mut x = vec![0f32; n];
        let interval = 60.0 / bpm;
        let mut t = 0.25;
        while t < secs {
            let start = (t * f64::from(sr)) as usize;
            for (k, v) in x.iter_mut().skip(start).take(300).enumerate() {
                let tt = k as f32 / sr as f32;
                *v += (2.0 * std::f32::consts::PI * 1000.0 * tt).sin() * (-tt * 200.0).exp();
            }
            t += interval;
        }
        x
    }

    #[test]
    fn detects_120_bpm_clicks() {
        let sr = 22_050;
        let x = click_track(sr, 10.0, 120.0);
        let cfg = BeatConfig {
            n_fft: 1024,
            hop: 256,
            ..BeatConfig::default()
        };
        let r = detect_beats(&x, sr, &cfg);
        assert!((r.bpm - 120.0).abs() < 2.0, "BPM = {}", r.bpm);
        assert!(r.beats.len() >= 15, "节拍数 {}", r.beats.len());
        let mut gaps: Vec<f64> = r.beats.windows(2).map(|w| w[1] - w[0]).collect();
        gaps.sort_by(f64::total_cmp);
        let median = gaps[gaps.len() / 2];
        assert!((median - 0.5).abs() < 0.03, "中位拍间隔 {median}");
    }

    #[test]
    fn cut_points_every_n() {
        let beats = [0.0, 0.5, 1.0, 1.5, 2.0];
        assert_eq!(cut_points(&beats, 2), vec![0.0, 1.0, 2.0]);
        assert_eq!(cut_points(&beats, 0), beats.to_vec());
    }
}
