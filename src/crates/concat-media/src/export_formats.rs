//! 导出格式参数：纯音频导出（WAV / MP3 / M4A / FLAC）与 GIF 导出。
//!
//! 本模块只包含纯逻辑：参数结构、校验以及 FFmpeg 编码器名 / 滤镜字符串的生成，
//! 不直接调用 FFmpeg。

use serde::{Deserialize, Serialize};
use std::fmt;

/// 导出参数校验错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormatError {
    /// 码率超出允许范围（kbps）。
    InvalidBitrate(u32),
    /// 采样率不在支持列表中（Hz）。
    InvalidSampleRate(u32),
    /// GIF 帧率超出允许范围。
    InvalidFps(u32),
    /// GIF 宽度超出允许范围（像素）。
    InvalidWidth(u32),
    /// 源视频尺寸无效（宽或高为 0）。
    InvalidSourceSize {
        /// 源宽度。
        width: u32,
        /// 源高度。
        height: u32,
    },
}

impl fmt::Display for ExportFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBitrate(b) => write!(
                f,
                "码率 {b} kbps 无效，应在 {MIN_BITRATE_KBPS}..={MAX_BITRATE_KBPS} 之间"
            ),
            Self::InvalidSampleRate(r) => write!(f, "采样率 {r} Hz 不受支持"),
            Self::InvalidFps(v) => {
                write!(
                    f,
                    "GIF 帧率 {v} 无效，应在 {MIN_GIF_FPS}..={MAX_GIF_FPS} 之间"
                )
            }
            Self::InvalidWidth(w) => write!(
                f,
                "GIF 宽度 {w} 无效，应在 {MIN_GIF_WIDTH}..={MAX_GIF_WIDTH} 之间"
            ),
            Self::InvalidSourceSize { width, height } => {
                write!(f, "源尺寸 {width}x{height} 无效")
            }
        }
    }
}

impl std::error::Error for ExportFormatError {}

/// 允许的最小音频码率（kbps）。
pub const MIN_BITRATE_KBPS: u32 = 32;
/// 允许的最大音频码率（kbps）。
pub const MAX_BITRATE_KBPS: u32 = 320;
/// 支持的音频采样率（Hz）。
pub const SUPPORTED_SAMPLE_RATES: [u32; 4] = [22050, 32000, 44100, 48000];
/// GIF 最小帧率。
pub const MIN_GIF_FPS: u32 = 1;
/// GIF 最大帧率。
pub const MAX_GIF_FPS: u32 = 30;
/// GIF 最小宽度（像素）。
pub const MIN_GIF_WIDTH: u32 = 64;
/// GIF 最大宽度（像素）。
pub const MAX_GIF_WIDTH: u32 = 1920;

/// 纯音频导出格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AudioFormat {
    /// 无损 PCM 16 位 WAV。
    Wav,
    /// MP3（libmp3lame）。
    Mp3,
    /// AAC 封装为 M4A。
    M4a,
    /// 无损 FLAC。
    Flac,
}

impl AudioFormat {
    /// 所有支持的音频格式。
    pub const ALL: [AudioFormat; 4] = [Self::Wav, Self::Mp3, Self::M4a, Self::Flac];

    /// 对应的 FFmpeg 编码器名。
    pub fn encoder(self) -> &'static str {
        match self {
            Self::Wav => "pcm_s16le",
            Self::Mp3 => "libmp3lame",
            Self::M4a => "aac",
            Self::Flac => "flac",
        }
    }

    /// 输出文件扩展名（不含点）。
    pub fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::M4a => "m4a",
            Self::Flac => "flac",
        }
    }

    /// 该格式是否使用码率参数（有损格式为 true，WAV/FLAC 为 false）。
    pub fn has_bitrate(self) -> bool {
        matches!(self, Self::Mp3 | Self::M4a)
    }
}

/// 纯音频导出选项。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioExportOptions {
    /// 输出格式。
    pub format: AudioFormat,
    /// 码率（kbps），仅对有损格式生效；WAV/FLAC 忽略。
    pub bitrate_kbps: u32,
    /// 采样率（Hz）。
    pub sample_rate: u32,
}

impl Default for AudioExportOptions {
    fn default() -> Self {
        Self {
            format: AudioFormat::Mp3,
            bitrate_kbps: 192,
            sample_rate: 44100,
        }
    }
}

impl AudioExportOptions {
    /// 校验参数：码率 32..=320（WAV/FLAC 忽略码率），采样率必须在支持列表中。
    pub fn validate(&self) -> Result<(), ExportFormatError> {
        if self.format.has_bitrate()
            && !(MIN_BITRATE_KBPS..=MAX_BITRATE_KBPS).contains(&self.bitrate_kbps)
        {
            return Err(ExportFormatError::InvalidBitrate(self.bitrate_kbps));
        }
        if !SUPPORTED_SAMPLE_RATES.contains(&self.sample_rate) {
            return Err(ExportFormatError::InvalidSampleRate(self.sample_rate));
        }
        Ok(())
    }

    /// 实际生效的码率：有损格式返回 `Some(kbps)`，无损格式返回 `None`。
    pub fn effective_bitrate_kbps(&self) -> Option<u32> {
        self.format.has_bitrate().then_some(self.bitrate_kbps)
    }
}

/// GIF 循环模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GifLoop {
    /// 无限循环。
    Infinite,
    /// 只播放一次，不循环。
    Once,
    /// 循环指定次数。
    Count(u16),
}

impl GifLoop {
    /// 映射为 FFmpeg GIF muxer 的 `-loop` 值：Infinite=0，Once=-1，Count(n)=n。
    pub fn ffmpeg_value(self) -> i32 {
        match self {
            Self::Infinite => 0,
            Self::Once => -1,
            Self::Count(n) => i32::from(n),
        }
    }
}

/// GIF 导出选项。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GifOptions {
    /// 帧率（1..=30）。
    pub fps: u32,
    /// 输出宽度（64..=1920 像素），高度按源比例计算。
    pub width: u32,
    /// 循环模式。
    pub loop_mode: GifLoop,
}

impl Default for GifOptions {
    fn default() -> Self {
        Self {
            fps: 15,
            width: 480,
            loop_mode: GifLoop::Infinite,
        }
    }
}

impl GifOptions {
    /// 校验参数：fps 1..=30，width 64..=1920。
    pub fn validate(&self) -> Result<(), ExportFormatError> {
        if !(MIN_GIF_FPS..=MAX_GIF_FPS).contains(&self.fps) {
            return Err(ExportFormatError::InvalidFps(self.fps));
        }
        if !(MIN_GIF_WIDTH..=MAX_GIF_WIDTH).contains(&self.width) {
            return Err(ExportFormatError::InvalidWidth(self.width));
        }
        Ok(())
    }

    /// 按源尺寸等比计算输出高度，四舍五入后取偶数（向上取偶，最小为 2）。
    pub fn height_for(&self, src_w: u32, src_h: u32) -> Result<u32, ExportFormatError> {
        if src_w == 0 || src_h == 0 {
            return Err(ExportFormatError::InvalidSourceSize {
                width: src_w,
                height: src_h,
            });
        }
        let w = u64::from(self.width);
        let sw = u64::from(src_w);
        let sh = u64::from(src_h);
        let h = (w * sh + sw / 2) / sw;
        let even = h.div_ceil(2) * 2;
        Ok(u32::try_from(even.max(2)).unwrap_or(u32::MAX - 1))
    }

    /// FFmpeg `-loop` 参数值。
    pub fn loop_value(&self) -> i32 {
        self.loop_mode.ffmpeg_value()
    }

    /// 公共前缀滤镜：抽帧与 lanczos 缩放。
    fn base_filter(&self) -> String {
        format!("fps={},scale={}:-2:flags=lanczos", self.fps, self.width)
    }

    /// 第一遍生成调色板的滤镜字符串（palettegen，stats_mode=diff）。
    pub fn palettegen_filter(&self) -> String {
        format!("{},palettegen=stats_mode=diff", self.base_filter())
    }

    /// 第二遍应用调色板的 filter_complex 字符串（输入 1 为调色板，dither=sierra2_4a）。
    pub fn paletteuse_filter(&self) -> String {
        format!(
            "{}[x];[x][1:v]paletteuse=dither=sierra2_4a",
            self.base_filter()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_encoders_and_extensions() {
        assert_eq!(AudioFormat::Wav.encoder(), "pcm_s16le");
        assert_eq!(AudioFormat::Mp3.encoder(), "libmp3lame");
        assert_eq!(AudioFormat::M4a.encoder(), "aac");
        assert_eq!(AudioFormat::Flac.encoder(), "flac");
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
        assert_eq!(AudioFormat::M4a.extension(), "m4a");
        assert_eq!(AudioFormat::Flac.extension(), "flac");
    }

    #[test]
    fn audio_has_bitrate() {
        assert!(!AudioFormat::Wav.has_bitrate());
        assert!(AudioFormat::Mp3.has_bitrate());
        assert!(AudioFormat::M4a.has_bitrate());
        assert!(!AudioFormat::Flac.has_bitrate());
    }

    #[test]
    fn audio_validate_bitrate_bounds() {
        let mut o = AudioExportOptions::default();
        assert!(o.validate().is_ok());
        o.bitrate_kbps = 32;
        assert!(o.validate().is_ok());
        o.bitrate_kbps = 320;
        assert!(o.validate().is_ok());
        o.bitrate_kbps = 31;
        assert_eq!(o.validate(), Err(ExportFormatError::InvalidBitrate(31)));
        o.bitrate_kbps = 321;
        assert_eq!(o.validate(), Err(ExportFormatError::InvalidBitrate(321)));
        o.format = AudioFormat::M4a;
        assert!(o.validate().is_err());
    }

    #[test]
    fn audio_lossless_ignores_bitrate() {
        for format in [AudioFormat::Wav, AudioFormat::Flac] {
            let o = AudioExportOptions {
                format,
                bitrate_kbps: 0,
                sample_rate: 48000,
            };
            assert!(o.validate().is_ok());
            assert_eq!(o.effective_bitrate_kbps(), None);
        }
        let o = AudioExportOptions::default();
        assert_eq!(o.effective_bitrate_kbps(), Some(192));
    }

    #[test]
    fn audio_validate_sample_rate() {
        for rate in SUPPORTED_SAMPLE_RATES {
            let o = AudioExportOptions {
                sample_rate: rate,
                ..Default::default()
            };
            assert!(o.validate().is_ok());
        }
        let o = AudioExportOptions {
            format: AudioFormat::Wav,
            bitrate_kbps: 0,
            sample_rate: 16000,
        };
        assert_eq!(
            o.validate(),
            Err(ExportFormatError::InvalidSampleRate(16000))
        );
    }

    #[test]
    fn audio_serde_camel_case() {
        let o = AudioExportOptions {
            format: AudioFormat::M4a,
            bitrate_kbps: 256,
            sample_rate: 48000,
        };
        let json = serde_json::to_string(&o).unwrap();
        assert_eq!(
            json,
            r#"{"format":"m4a","bitrateKbps":256,"sampleRate":48000}"#
        );
        let back: AudioExportOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back, o);
    }

    #[test]
    fn gif_validate_bounds() {
        let mut g = GifOptions::default();
        assert!(g.validate().is_ok());
        g.fps = 0;
        assert_eq!(g.validate(), Err(ExportFormatError::InvalidFps(0)));
        g.fps = 31;
        assert_eq!(g.validate(), Err(ExportFormatError::InvalidFps(31)));
        g.fps = 30;
        assert!(g.validate().is_ok());
        g.fps = 1;
        g.width = 63;
        assert_eq!(g.validate(), Err(ExportFormatError::InvalidWidth(63)));
        g.width = 1921;
        assert_eq!(g.validate(), Err(ExportFormatError::InvalidWidth(1921)));
        g.width = 64;
        assert!(g.validate().is_ok());
        g.width = 1920;
        assert!(g.validate().is_ok());
    }

    #[test]
    fn gif_height_for_even_aspect() {
        let g = GifOptions {
            width: 480,
            ..Default::default()
        };
        assert_eq!(g.height_for(1920, 1080).unwrap(), 270);
        // 480 * 1080 / 1440 = 360
        assert_eq!(g.height_for(1440, 1080).unwrap(), 360);
        assert_eq!(g.height_for(1280, 720).unwrap(), 270);
        let g = GifOptions {
            width: 100,
            ..Default::default()
        };
        // 100 * 3 / 4 = 75 -> 取偶 76
        assert_eq!(g.height_for(4, 3).unwrap(), 76);
        // 极扁源最小为 2
        assert_eq!(g.height_for(10000, 1).unwrap(), 2);
        for (w, h) in [(1920, 1080), (1000, 777), (333, 1000)] {
            assert_eq!(g.height_for(w, h).unwrap() % 2, 0);
        }
        assert!(matches!(
            g.height_for(0, 100),
            Err(ExportFormatError::InvalidSourceSize { .. })
        ));
        assert!(g.height_for(100, 0).is_err());
    }

    #[test]
    fn gif_loop_mapping() {
        assert_eq!(GifLoop::Infinite.ffmpeg_value(), 0);
        assert_eq!(GifLoop::Once.ffmpeg_value(), -1);
        assert_eq!(GifLoop::Count(3).ffmpeg_value(), 3);
        let g = GifOptions {
            loop_mode: GifLoop::Count(5),
            ..Default::default()
        };
        assert_eq!(g.loop_value(), 5);
    }

    #[test]
    fn gif_filters() {
        let g = GifOptions {
            fps: 12,
            width: 640,
            loop_mode: GifLoop::Infinite,
        };
        assert_eq!(
            g.palettegen_filter(),
            "fps=12,scale=640:-2:flags=lanczos,palettegen=stats_mode=diff"
        );
        assert_eq!(
            g.paletteuse_filter(),
            "fps=12,scale=640:-2:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a"
        );
    }

    #[test]
    fn gif_serde_camel_case() {
        let g = GifOptions {
            fps: 10,
            width: 320,
            loop_mode: GifLoop::Count(2),
        };
        let json = serde_json::to_string(&g).unwrap();
        assert_eq!(json, r#"{"fps":10,"width":320,"loopMode":{"count":2}}"#);
        let back: GifOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back, g);
        let once: GifLoop = serde_json::from_str(r#""once""#).unwrap();
        assert_eq!(once, GifLoop::Once);
    }

    #[test]
    fn error_display() {
        assert!(
            ExportFormatError::InvalidBitrate(1)
                .to_string()
                .contains("1")
        );
        assert!(
            ExportFormatError::InvalidSourceSize {
                width: 0,
                height: 1
            }
            .to_string()
            .contains("0x1")
        );
    }
}
