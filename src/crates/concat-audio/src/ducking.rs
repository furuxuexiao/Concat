//! 自动闪避（ducking）：由语音区间生成背景音乐的音量包络关键帧。

use crate::vad::Segment;

/// 闪避参数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DuckConfig {
    /// 语音期间的增益（dB，通常为负，如 -12）。
    pub duck_db: f32,
    /// 进入闪避的渐变时长（秒），在语音开始前完成。
    pub attack_s: f64,
    /// 退出闪避的渐变时长（秒），在语音结束后开始。
    pub release_s: f64,
}

impl Default for DuckConfig {
    fn default() -> Self {
        Self {
            duck_db: -12.0,
            attack_s: 0.3,
            release_s: 0.5,
        }
    }
}

/// 音量包络关键帧，相邻关键帧之间按 dB 线性插值。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keyframe {
    /// 时间（秒）。
    pub time: f64,
    /// 增益（dB）。
    pub gain_db: f32,
}

fn push(kf: &mut Vec<Keyframe>, time: f64, gain_db: f32) {
    if let Some(last) = kf.last()
        && (last.time - time).abs() < 1e-9
        && last.gain_db == gain_db
    {
        return;
    }
    kf.push(Keyframe { time, gain_db });
}

/// 由语音区间生成闪避包络关键帧，时间非递减，覆盖 `[0, duration]`。
///
/// 渐变相互重叠的区间会合并为一次连续闪避。
pub fn duck_envelope(segments: &[Segment], duration: f64, cfg: &DuckConfig) -> Vec<Keyframe> {
    let mut sorted = segments.to_vec();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut blocks: Vec<(f64, f64)> = Vec::new();
    for s in sorted {
        let (a, b) = (s.start.max(0.0), s.end.min(duration));
        if b <= a {
            continue;
        }
        match blocks.last_mut() {
            Some(last) if a - cfg.attack_s <= last.1 + cfg.release_s => {
                last.1 = last.1.max(b);
            }
            _ => blocks.push((a, b)),
        }
    }
    let mut kf = vec![Keyframe {
        time: 0.0,
        gain_db: 0.0,
    }];
    for (a, b) in blocks {
        push(&mut kf, (a - cfg.attack_s).max(0.0), 0.0);
        push(&mut kf, a, cfg.duck_db);
        push(&mut kf, b, cfg.duck_db);
        push(&mut kf, (b + cfg.release_s).min(duration), 0.0);
    }
    if let Some(last) = kf.last().copied()
        && last.time < duration
    {
        kf.push(Keyframe {
            time: duration,
            gain_db: last.gain_db,
        });
    }
    kf
}

/// 在关键帧之间线性插值，返回时间 `t` 处的增益（dB）；无关键帧时为 0。
pub fn gain_at(keyframes: &[Keyframe], t: f64) -> f32 {
    match keyframes.iter().position(|k| k.time > t) {
        None => keyframes.last().map_or(0.0, |k| k.gain_db),
        Some(0) => keyframes[0].gain_db,
        Some(i) => {
            let (p, n) = (keyframes[i - 1], keyframes[i]);
            let r = ((t - p.time) / (n.time - p.time)) as f32;
            p.gain_db + (n.gain_db - p.gain_db) * r
        }
    }
}

/// dB 转线性增益。
pub fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// 把包络作用到单声道样本上（原地）。
pub fn apply_envelope(samples: &mut [f32], sample_rate: u32, keyframes: &[Keyframe]) {
    let sr = f64::from(sample_rate.max(1));
    for (i, s) in samples.iter_mut().enumerate() {
        *s *= db_to_gain(gain_at(keyframes, i as f64 / sr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_ramps_and_holds() {
        let cfg = DuckConfig {
            duck_db: -12.0,
            attack_s: 0.5,
            release_s: 1.0,
        };
        let kf = duck_envelope(
            &[Segment {
                start: 2.0,
                end: 4.0,
            }],
            10.0,
            &cfg,
        );
        assert!(kf.windows(2).all(|w| w[0].time <= w[1].time));
        assert_eq!(gain_at(&kf, 0.5), 0.0);
        assert!((gain_at(&kf, 1.75) + 6.0).abs() < 1e-4);
        assert_eq!(gain_at(&kf, 3.0), -12.0);
        assert!((gain_at(&kf, 4.5) + 6.0).abs() < 1e-4);
        assert_eq!(gain_at(&kf, 6.0), 0.0);
        assert_eq!(kf.last().map(|k| k.time), Some(10.0));
    }

    #[test]
    fn close_segments_merge() {
        let cfg = DuckConfig::default();
        let segs = [
            Segment {
                start: 1.0,
                end: 2.0,
            },
            Segment {
                start: 2.5,
                end: 3.0,
            },
        ];
        let kf = duck_envelope(&segs, 5.0, &cfg);
        assert_eq!(gain_at(&kf, 2.25), cfg.duck_db);
        assert!((db_to_gain(-6.0) - 0.501).abs() < 1e-3);
    }
}
