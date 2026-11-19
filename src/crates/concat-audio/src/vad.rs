//! 基于短时能量的语音活动检测（VAD）。

/// 一段时间区间（秒）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// 起始时间（秒）。
    pub start: f64,
    /// 结束时间（秒）。
    pub end: f64,
}

impl Segment {
    /// 区间时长（秒）。
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// VAD 参数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadConfig {
    /// 分析帧长（秒）。
    pub frame_s: f64,
    /// 能量阈值（dBFS，按帧 RMS 计算），高于此值视为语音。
    pub threshold_db: f32,
    /// 短于此时长的语音段被丢弃（秒）。
    pub min_speech_s: f64,
    /// 短于此时长的静音间隙被合并（秒）。
    pub min_silence_s: f64,
    /// 每段前后各扩展的时长（秒）。
    pub padding_s: f64,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_s: 0.02,
            threshold_db: -40.0,
            min_speech_s: 0.2,
            min_silence_s: 0.3,
            padding_s: 0.1,
        }
    }
}

/// 检测单声道信号中的语音区间，按时间升序返回且互不重叠。
pub fn detect_speech(samples: &[f32], sample_rate: u32, cfg: &VadConfig) -> Vec<Segment> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let sr = f64::from(sample_rate);
    let frame = ((cfg.frame_s * sr).round() as usize).max(1);
    let dur = samples.len() as f64 / sr;

    let mut raw = Vec::new();
    let mut cur: Option<f64> = None;
    for (i, chunk) in samples.chunks(frame).enumerate() {
        let energy = chunk.iter().map(|&x| f64::from(x * x)).sum::<f64>() / chunk.len() as f64;
        let db = 10.0 * energy.max(1e-20).log10();
        let t = (i * frame) as f64 / sr;
        let active = db > f64::from(cfg.threshold_db);
        match (active, cur) {
            (true, None) => cur = Some(t),
            (false, Some(s)) => {
                raw.push(Segment { start: s, end: t });
                cur = None;
            }
            _ => {}
        }
    }
    if let Some(s) = cur {
        raw.push(Segment { start: s, end: dur });
    }

    let mut merged: Vec<Segment> = Vec::new();
    for s in raw {
        match merged.last_mut() {
            Some(last) if s.start - last.end < cfg.min_silence_s => last.end = s.end,
            _ => merged.push(s),
        }
    }

    let mut out: Vec<Segment> = Vec::new();
    for s in merged
        .into_iter()
        .filter(|s| s.duration() >= cfg.min_speech_s)
    {
        let s = Segment {
            start: (s.start - cfg.padding_s).max(0.0),
            end: (s.end + cfg.padding_s).min(dur),
        };
        match out.last_mut() {
            Some(last) if s.start <= last.end => last.end = last.end.max(s.end),
            _ => out.push(s),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_burst_in_silence() {
        let sr = 16_000u32;
        let mut x = vec![0.0f32; 3 * sr as usize];
        let mut s = 1u32;
        for (i, v) in x.iter_mut().enumerate() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1e-4;
            let t = i as f32 / sr as f32;
            *v = noise
                + if (1.0..2.0).contains(&t) {
                    0.5 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                } else {
                    0.0
                };
        }
        let segs = detect_speech(&x, sr, &VadConfig::default());
        assert_eq!(segs.len(), 1, "{segs:?}");
        assert!((segs[0].start - 0.9).abs() < 0.03, "{segs:?}");
        assert!((segs[0].end - 2.1).abs() < 0.03, "{segs:?}");
    }

    #[test]
    fn merges_short_gaps_and_drops_clicks() {
        let sr = 1000u32;
        let mut x = vec![0.0f32; 3000];
        x[500..1000].iter_mut().for_each(|v| *v = 0.5);
        x[1100..1500].iter_mut().for_each(|v| *v = 0.5);
        x[2500..2550].iter_mut().for_each(|v| *v = 0.5);
        let cfg = VadConfig {
            padding_s: 0.0,
            ..VadConfig::default()
        };
        let segs = detect_speech(&x, sr, &cfg);
        assert_eq!(
            segs,
            vec![Segment {
                start: 0.5,
                end: 1.5
            }]
        );
    }
}
