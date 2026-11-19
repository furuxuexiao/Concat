//! 纯 Rust 音频分析工具：FFT/STFT、Spleeter 2stems 前后处理（人声分离）、
//! 能量 VAD、自动闪避音量包络与节拍检测（音乐卡点）。
//!
//! 本 crate 不依赖任何外部库；模型推理由调用方以闭包注入。

pub mod beat;
pub mod ducking;
pub mod fft;
pub mod spleeter;
pub mod stft;
pub mod vad;

pub use beat::{BeatConfig, BeatResult, cut_points, detect_beats};
pub use ducking::{DuckConfig, Keyframe, duck_envelope, gain_at};
pub use fft::Complex;
pub use spleeter::{Separated, separate};
pub use stft::{Spectrogram, StftConfig, istft, stft};
pub use vad::{Segment, VadConfig, detect_speech};
