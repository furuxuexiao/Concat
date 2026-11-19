// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Concat contributors

//! Where a file is going: the export sheet's platform presets.
//!
//! A person cutting for Douyin, WeChat Channels or Bilibili wants the
//! file the platform takes without a second encode, and does not want to
//! look up what that is. A preset is that lookup: the codec, the frame
//! rate and the bitrate a platform's upload takes, and what about the
//! frame it asks for - upright or wide, how many lines - so the sheet can
//! say when the edit is not cut for where it is going.
//!
//! The numbers are the platforms' own where they publish them, and the
//! commonly published upload values where they do not:
//!
//! - Bilibili 1080P: average bitrate at most 6000 kbps in H.264, no more
//!   than 1920 x 1080, 8-bit 4:2:0, at most 60 fps; above that the upload
//!   is encoded again (Bilibili creator centre, "视频上传清晰度规则补充说明",
//!   https://www.bilibili.com/opus/123109117214361042).
//! - Bilibili 4K: 20000 kbps in H.264, the short side at least 1600, up to
//!   4096 x 2160 and 120 fps (Bilibili's open platform,
//!   https://open.bilibili.com/doc/4/0c532c6a-e6fb-0aff-8021-905ae2409095,
//!   and its support assistant's 4K guidance).
//! - Douyin: upright video, 720p or more (Douyin open platform,
//!   https://open.douyin.com/platform/resource/docs/ability/content-management/douyin-publish-solution);
//!   1080 x 1920 in H.264 at 8 Mbps or more is the commonly published
//!   recommendation, and 10 Mbps is taken here.
//! - WeChat Channels: 16:9 or 9:16, 1080p or more, no HDR (Channels help
//!   centre, "视频号发表视频/图文有什么格式要求？"); 6 Mbps H.264 at 30 fps
//!   is the commonly published value, as anything faster is shown at 30.
//! - YouTube 1080p: 8 Mbps, 12 at 48 fps and above (YouTube's recommended
//!   upload encoding settings), the sheet's own default.
//!
//! Pure: no FFmpeg, so the window and tests read it alike.

/// Which way round a platform wants the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Taller than wide: 9:16.
    Upright,
    /// Wider than tall: 16:9.
    Wide,
    /// Either, as the platform shows both.
    Either,
}

/// A destination the export sheet offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    /// 抖音.
    Douyin,
    /// 微信视频号.
    Channels,
    /// 哔哩哔哩, 1080P.
    Bilibili,
    /// 哔哩哔哩, 4K.
    Bilibili4k,
    /// YouTube, 1080p.
    YouTube,
}

/// Something about the edit a platform would not take as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// The frame is the other way round from the platform's.
    Orientation(Orientation),
    /// Fewer lines than the platform shows well, as the short side.
    TooSmall(u32),
    /// More lines than this preset is for, as the short side: the upload
    /// is scaled down, and another preset is the one for it.
    TooLarge(u32),
    /// A ten-bit or HDR file, which the platform encodes again to 8-bit
    /// SDR or refuses.
    EightBitOnly,
}

impl Platform {
    /// Every preset, in the order the sheet lists them.
    pub const ALL: [Platform; 5] = [
        Platform::Douyin,
        Platform::Channels,
        Platform::Bilibili,
        Platform::Bilibili4k,
        Platform::YouTube,
    ];

    /// A stable name, for logs and saved choices; never shown.
    pub fn id(self) -> &'static str {
        match self {
            Platform::Douyin => "douyin",
            Platform::Channels => "wechat-channels",
            Platform::Bilibili => "bilibili-1080p",
            Platform::Bilibili4k => "bilibili-4k",
            Platform::YouTube => "youtube-1080p",
        }
    }

    /// Which way round the platform shows a frame best.
    pub fn orientation(self) -> Orientation {
        match self {
            Platform::Douyin => Orientation::Upright,
            Platform::Channels => Orientation::Either,
            Platform::Bilibili | Platform::Bilibili4k | Platform::YouTube => Orientation::Wide,
        }
    }

    /// The fewest lines, as the short side, the platform shows well.
    pub fn min_short_side(self) -> u32 {
        match self {
            Platform::Bilibili4k => 1600,
            _ => 1080,
        }
    }

    /// The most lines this preset is for, as the short side.
    pub fn max_short_side(self) -> u32 {
        match self {
            Platform::Bilibili4k => 2160,
            _ => 1080,
        }
    }

    /// The fastest rate the platform plays, in frames a second.
    pub fn max_fps(self) -> f64 {
        match self {
            Platform::Channels => 30.0,
            // Bilibili's 4K takes 120; the sheet's fastest rate is 60.
            _ => 60.0,
        }
    }

    /// Whether the platform takes only 8-bit SDR.
    pub fn eight_bit_only(self) -> bool {
        // Bilibili's and Channels' rules say so; Douyin's 8-bit H.264 is
        // what every upright upload is played as. YouTube takes HDR.
        !matches!(self, Platform::YouTube)
    }

    /// The rate a timeline at `own` frames a second is exported at: its
    /// own, unless that is faster than the platform plays, when it is the
    /// fastest exact divisor of its own the platform plays - every second
    /// frame of a 50 fps cut is 25, of 59.94 is 29.97 - so no frame is
    /// shown twice and none is dropped unevenly, which a 50 fps cut
    /// brought down to 30 judders with. Never made faster than it was
    /// cut; an unknown rate is the platform's fastest.
    pub fn fps_for(self, own: f64) -> f64 {
        let max = self.max_fps();
        if !(own.is_finite() && own > 0.0) {
            return max;
        }
        // The smallest whole step that brings it under the platform's
        // fastest, so the fastest of the exact divisors.
        let step = (own / (max + 0.01)).ceil().max(1.0);
        own / step
    }

    /// The H.264 bitrate the file is written at, in kbps, at `fps`.
    pub fn kbps(self, fps: f64) -> u32 {
        let fast = fps > 30.5;
        match self {
            Platform::Douyin => 10_000,
            Platform::Channels => 6_000,
            // An average, which Bilibili's 1080P holds at 6000 whatever
            // the rate.
            Platform::Bilibili => 6_000,
            Platform::Bilibili4k => 20_000,
            Platform::YouTube => {
                if fast {
                    12_000
                } else {
                    8_000
                }
            }
        }
    }

    /// What about a `width` by `height` edit, `deep` when it is written at
    /// ten bits or HDR, the platform would not take as it is. Empty when
    /// it is cut for the platform.
    pub fn advise(self, width: u32, height: u32, deep: bool) -> Vec<Advice> {
        let mut advice = Vec::new();
        let short = width.min(height);
        let shape = if height > width {
            Orientation::Upright
        } else if width > height {
            Orientation::Wide
        } else {
            Orientation::Either
        };
        let wanted = self.orientation();
        if wanted != Orientation::Either && shape != wanted {
            advice.push(Advice::Orientation(wanted));
        }
        if short < self.min_short_side() {
            advice.push(Advice::TooSmall(self.min_short_side()));
        } else if short > self.max_short_side() {
            advice.push(Advice::TooLarge(self.max_short_side()));
        }
        if deep && self.eight_bit_only() {
            advice.push(Advice::EightBitOnly);
        }
        advice
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids are distinct, so a saved choice reads back as the one made.
    #[test]
    fn every_preset_has_its_own_id() {
        let mut ids: Vec<&str> = Platform::ALL.iter().map(|p| p.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), Platform::ALL.len());
    }

    /// Bilibili's 1080P is held under its 6000 kbps average, its 4K at
    /// 20000, Douyin's over the 8 Mbps an upright 1080p wants.
    #[test]
    fn bitrates_keep_to_what_each_platform_takes() {
        assert!(Platform::Bilibili.kbps(30.0) <= 6_000);
        assert!(Platform::Bilibili.kbps(60.0) <= 6_000);
        assert_eq!(Platform::Bilibili4k.kbps(30.0), 20_000);
        assert!(Platform::Douyin.kbps(30.0) >= 8_000);
        assert_eq!(Platform::YouTube.kbps(30.0), 8_000);
        assert_eq!(Platform::YouTube.kbps(60.0), 12_000);
    }

    /// A timeline keeps its own rate where the platform plays it, and
    /// comes down to the platform's fastest where it does not.
    #[test]
    fn a_rate_is_kept_unless_the_platform_is_slower() {
        assert_eq!(Platform::Douyin.fps_for(25.0), 25.0);
        assert_eq!(Platform::Douyin.fps_for(60.0), 60.0);
        assert_eq!(Platform::Channels.fps_for(60.0), 30.0);
        assert_eq!(
            Platform::Channels.fps_for(30000.0 / 1001.0),
            30000.0 / 1001.0
        );
        assert_eq!(Platform::Bilibili.fps_for(f64::NAN), 60.0);
        assert_eq!(Platform::Bilibili.fps_for(0.0), 60.0);
    }

    /// A rate faster than the platform plays comes down by a whole step:
    /// every second (or third) frame, never a judder of uneven drops.
    #[test]
    fn a_faster_rate_comes_down_to_an_exact_divisor() {
        let ntsc = |fps: f64| fps * 1000.0 / 1001.0;
        assert_eq!(Platform::Channels.fps_for(50.0), 25.0);
        assert_eq!(Platform::Channels.fps_for(ntsc(60.0)), ntsc(30.0));
        assert_eq!(Platform::Channels.fps_for(60.0), 30.0);
        assert_eq!(Platform::Channels.fps_for(30.0), 30.0);
        assert_eq!(Platform::Channels.fps_for(25.0), 25.0);
        assert_eq!(Platform::Channels.fps_for(ntsc(30.0)), ntsc(30.0));
        assert_eq!(Platform::Channels.fps_for(48.0), 24.0);
        assert_eq!(Platform::Channels.fps_for(120.0), 30.0);
        assert_eq!(Platform::Channels.fps_for(100.0), 25.0);
        assert_eq!(Platform::Douyin.fps_for(120.0), 60.0);
        assert_eq!(Platform::Douyin.fps_for(ntsc(120.0)), ntsc(60.0));
        assert_eq!(Platform::Douyin.fps_for(ntsc(60.0)), ntsc(60.0));
    }

    /// A wide 1080p edit is what Bilibili wants; an upright one is told
    /// it is the wrong way round, a 720p one that it is small, and a 4K
    /// one that Bilibili's 1080P scales it down.
    #[test]
    fn bilibili_advice() {
        assert!(Platform::Bilibili.advise(1920, 1080, false).is_empty());
        assert_eq!(
            Platform::Bilibili.advise(1080, 1920, false),
            vec![Advice::Orientation(Orientation::Wide)]
        );
        assert_eq!(
            Platform::Bilibili.advise(1280, 720, false),
            vec![Advice::TooSmall(1080)]
        );
        assert_eq!(
            Platform::Bilibili.advise(3840, 2160, true),
            vec![Advice::TooLarge(1080), Advice::EightBitOnly]
        );
        assert!(Platform::Bilibili4k.advise(3840, 2160, false).is_empty());
        assert_eq!(
            Platform::Bilibili4k.advise(1920, 1080, false),
            vec![Advice::TooSmall(1600)]
        );
    }

    /// Douyin wants the frame upright; Channels takes either way round.
    #[test]
    fn upright_platforms_say_so() {
        assert!(Platform::Douyin.advise(1080, 1920, false).is_empty());
        assert_eq!(
            Platform::Douyin.advise(1920, 1080, false),
            vec![Advice::Orientation(Orientation::Upright)]
        );
        assert!(Platform::Channels.advise(1920, 1080, false).is_empty());
        assert!(Platform::Channels.advise(1080, 1920, false).is_empty());
        assert_eq!(
            Platform::Douyin.advise(1080, 1080, false),
            vec![Advice::Orientation(Orientation::Upright)]
        );
        assert!(Platform::YouTube.advise(1920, 1080, true).is_empty());
    }
}
