// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! 端到端：GIF 两遍调色板编码与多格式音频混音导出，真正跑一遍链接的 FFmpeg。

use std::path::{Path, PathBuf};
use std::process::Command;

use concat_core::animate::Track;
use concat_media::Error;
use concat_media::audio::{AudioClip, mix_to_file_as};
use concat_media::export_formats::{AudioExportOptions, AudioFormat, GifLoop, GifOptions};
use concat_media::gif::encode_gif;

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("concat-export-{}-{name}", std::process::id()))
}

/// 一秒 440 Hz 立体声 16 位 WAV，直接按字节写出，不依赖任何程序。
fn tone_wav(path: &Path, seconds: f64) {
    let rate: u32 = 48_000;
    let count = (seconds * f64::from(rate)) as usize;
    let mut data = Vec::with_capacity(count * 4);
    for index in 0..count {
        let t = index as f64 / f64::from(rate);
        let value = (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 32767.0) as i16;
        data.extend_from_slice(&value.to_le_bytes());
        data.extend_from_slice(&value.to_le_bytes());
    }
    let mut bytes = Vec::with_capacity(44 + data.len());
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 4).to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(path, bytes).expect("writes the wav");
}

fn clip(path: &Path, duration: f64) -> AudioClip {
    AudioClip {
        path: path.to_path_buf(),
        stream: None,
        start: 0.0,
        duration,
        source_start: 0.0,
        speed: 1.0,
        preserve_pitch: true,
        volume: 1.0,
        volume_curve: Track::default(),
        fade_in: 0.0,
        fade_out: 0.0,
        filter_chain: String::new(),
    }
}

/// 用系统的 `ffmpeg` 生成 1 秒 testsrc 视频；没有该程序时返回 `None`。
fn testsrc_video(path: &Path) -> Option<()> {
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=160x120:rate=10",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(path)
        .status()
        .ok()?;
    status.success().then_some(())
}

#[test]
fn gif_is_written_with_a_gif89a_header() {
    let video = scratch("src.mp4");
    if testsrc_video(&video).is_none() {
        eprintln!("skipped: no `ffmpeg` program to make the test video");
        return;
    }
    let gif = scratch("out.gif");
    let opts = GifOptions {
        fps: 10,
        width: 96,
        loop_mode: GifLoop::Infinite,
    };
    encode_gif(&video, &gif, &opts).expect("encodes the gif");
    let bytes = std::fs::read(&gif).expect("the gif exists");
    assert!(bytes.starts_with(b"GIF89a"), "header {:?}", &bytes[..6]);
    // 逻辑屏幕宽度（小端），即缩放后的宽度。
    assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 96);
    // 无限循环写出 NETSCAPE2.0 扩展块。
    assert!(bytes.windows(11).any(|w| w == b"NETSCAPE2.0"));
    let _ = std::fs::remove_file(&video);
    let _ = std::fs::remove_file(&gif);
}

#[test]
fn gif_options_out_of_range_are_refused_before_opening_anything() {
    let opts = GifOptions {
        fps: 0,
        ..Default::default()
    };
    let result = encode_gif(Path::new("/nonexistent.mp4"), &scratch("x.gif"), &opts);
    assert!(matches!(result, Err(Error::InvalidOptions { .. })));
}

#[test]
fn audio_exports_in_every_format() {
    let source = scratch("tone.wav");
    tone_wav(&source, 1.0);
    for format in AudioFormat::ALL {
        let output = scratch(&format!("mix.{}", format.extension()));
        let options = AudioExportOptions {
            format,
            bitrate_kbps: 128,
            sample_rate: 44_100,
        };
        match mix_to_file_as(&[clip(&source, 1.0)], &[], 1.0, &output, &options) {
            Ok(()) => {}
            // 链接的 FFmpeg 没编进 libmp3lame 时是清晰的 Missing，不算失败。
            Err(Error::Missing { what, name }) => {
                assert_eq!(what, "encoder");
                assert_eq!(name, format.encoder());
                eprintln!("skipped {format:?}: no {name} encoder");
                continue;
            }
            Err(error) => panic!("{format:?}: {error}"),
        }
        let bytes = std::fs::read(&output).expect("the export exists");
        match format {
            AudioFormat::Wav => {
                assert!(bytes.starts_with(b"RIFF"));
                assert_eq!(&bytes[8..12], b"WAVE");
                // fmt 块里的采样率与声道数。
                let at = bytes
                    .windows(4)
                    .position(|w| w == b"fmt ")
                    .expect("fmt chunk");
                let channels = u16::from_le_bytes([bytes[at + 10], bytes[at + 11]]);
                let rate = u32::from_le_bytes(bytes[at + 12..at + 16].try_into().unwrap());
                assert_eq!((channels, rate), (2, 44_100));
            }
            AudioFormat::Flac => assert!(bytes.starts_with(b"fLaC")),
            AudioFormat::M4a => assert_eq!(&bytes[4..8], b"ftyp"),
            AudioFormat::Mp3 => {
                assert!(bytes.starts_with(b"ID3") || (bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0))
            }
        }
        let probed = concat_media::probe(&output).expect("probes the export");
        let audio = probed.audio.expect("has a sound stream");
        assert_eq!(audio.sample_rate, 44_100, "{format:?}");
        let _ = std::fs::remove_file(&output);
    }
    let _ = std::fs::remove_file(&source);
}

#[test]
fn audio_options_out_of_range_are_refused() {
    let options = AudioExportOptions {
        format: AudioFormat::Mp3,
        bitrate_kbps: 1000,
        sample_rate: 44_100,
    };
    let result = mix_to_file_as(&[], &[], 1.0, &scratch("bad.mp3"), &options);
    assert!(matches!(result, Err(Error::InvalidOptions { .. })));
}
