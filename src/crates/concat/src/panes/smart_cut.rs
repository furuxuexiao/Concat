// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! The smart cut sheet (智能剪口播): a clip's speech as editable text, and
//! the cuts a talking-head edit usually wants.
//!
//! Opened from a clip's menu, it runs the clip's sound through the same
//! transcriber the captions sheet uses and keeps what was heard as
//! [`Unit`]s, one per timed piece, in seconds from the clip's start on the
//! timeline (`t / clip.speed`). Two ways to cut from there, each one
//! [`Command::RemoveSpans`], so one undo step:
//!
//! - the transcript as text: delete words, and "Apply text edit" takes out
//!   the stretches of the clip the deleted words were said in (see
//!   [`smart_cut::diff`]);
//! - the suggested cuts - filler words, repeats, long pauses - found by
//!   [`smart_cut::find_cuts`], each with a check box, and "Delete selected".
//!
//! After a cut the transcript is moved along with the timeline
//! ([`smart_cut::after_cut`]), so both lists stay true to the clip.

use std::sync::Arc;

use concat_project::Command;
use concat_project::smart_cut::{self, Cut, CutKind, CutOptions, Transcript, Unit};
use concat_speech::JobStage;
use concat_speech::transcribe::Segment;

use crate::host::{on_ui_in_project, spawn_in_project};
use crate::i18n::{t, tf};
use crate::panes::Msg;
use crate::panes::pace::{Pace, Stage};
use crate::panes::settings::installed;
use crate::studio::Studio;
use crate::ui::{SmartCutRowData, SmartCutSheetData};

/// Seconds whisper takes over one thirty-second window, per megabyte of
/// model: the captions sheet's first guess, before anything is timed.
const WINDOW_SECONDS_PER_MB: f32 = 1.0 / 160.0;

/// Everything that can happen to the smart cut sheet.
#[derive(Clone, Debug)]
pub enum SmartCutMsg {
    /// A clip's menu: the sheet on that clip's speech.
    OpenOnClip(String),
    Close,
    /// The transcript as the person has edited it.
    TextEdited(String),
    /// Cut what was deleted from the transcript.
    ApplyText,
    /// A suggested cut's check box, by its row.
    CutToggled(i32, bool),
    SelectAll,
    SelectNone,
    /// The shortest pause that is shortened, in seconds.
    PauseMinChanged(f32),
    /// How much of a shortened pause is kept, in seconds.
    PauseKeepChanged(f32),
    /// Cut the checked suggestions.
    DeleteSelected,
    /// Transcribe the clip again.
    Retry,
    Cancel,
    /// The transcriber's worker reporting the stage it is in.
    Progress(JobStage),
    /// The transcriber's worker is done: what was said, or why not.
    Finished(Result<Vec<Segment>, String>),
}

/// The smart cut sheet's state.
#[derive(Default)]
pub struct SmartCutPane {
    pub open: bool,
    pub running: bool,
    /// The clip being cut.
    clip: Option<String>,
    /// Its name, for the sheet's heading.
    clip_label: String,
    /// What was heard, in seconds from the clip's start on the timeline.
    units: Vec<Unit>,
    /// The transcript as it was heard, before any edit.
    original: String,
    /// The transcript as edited.
    text: String,
    /// Bumped whenever [`Self::text`] is replaced from here rather than
    /// typed, so the sheet re-seeds its box.
    revision: i32,
    /// The suggestions, and which are checked.
    cuts: Vec<Cut>,
    checked: Vec<bool>,
    pause_min: f64,
    pause_keep: f64,
    pub pace: Pace,
    message: String,
}

impl SmartCutPane {
    /// Applies one message. The studio is the rest of the window; while
    /// this runs the studio's copy of the pane is a blank it must not read.
    pub fn update(&mut self, msg: SmartCutMsg, studio: &mut Studio) {
        match msg {
            SmartCutMsg::OpenOnClip(_) if self.running => self.open = true,
            SmartCutMsg::OpenOnClip(id) => {
                let defaults = CutOptions::default();
                *self = SmartCutPane {
                    open: true,
                    clip: Some(id),
                    pause_min: defaults.pause_min,
                    pause_keep: defaults.pause_keep,
                    ..SmartCutPane::default()
                };
                self.transcribe(studio);
            }
            SmartCutMsg::Close => self.open = false,
            SmartCutMsg::TextEdited(text) => self.text = text,
            SmartCutMsg::ApplyText => self.apply_text(studio),
            SmartCutMsg::CutToggled(row, on) => {
                if let Some(checked) = usize::try_from(row)
                    .ok()
                    .and_then(|row| self.checked.get_mut(row))
                {
                    *checked = on;
                }
            }
            SmartCutMsg::SelectAll => self.checked.iter_mut().for_each(|c| *c = true),
            SmartCutMsg::SelectNone => self.checked.iter_mut().for_each(|c| *c = false),
            SmartCutMsg::PauseMinChanged(seconds) => {
                self.pause_min = f64::from(seconds).clamp(0.1, 10.0);
                self.pause_keep = self.pause_keep.min(self.pause_min);
                self.find(studio);
            }
            SmartCutMsg::PauseKeepChanged(seconds) => {
                self.pause_keep = f64::from(seconds).clamp(0.0, self.pause_min);
                self.find(studio);
            }
            SmartCutMsg::DeleteSelected => {
                let chosen: Vec<Cut> = self
                    .cuts
                    .iter()
                    .zip(&self.checked)
                    .filter(|(_, on)| **on)
                    .map(|(cut, _)| *cut)
                    .collect();
                let spans = smart_cut::cut_spans(&chosen);
                self.remove(studio, spans);
            }
            SmartCutMsg::Retry => self.transcribe(studio),
            SmartCutMsg::Cancel => {
                studio.host.transcriber.cancel();
                self.running = false;
            }
            SmartCutMsg::Progress(stage) => self.pace.report(stage),
            SmartCutMsg::Finished(result) => {
                self.running = false;
                match result {
                    Ok(segments) => {
                        let Some(clip) = self.clip.as_ref().and_then(|id| studio.clip(id)) else {
                            self.message = t("captions.clipNoLongerTimeline");
                            return;
                        };
                        self.units = units_of(&segments, clip.speed);
                        if self.units.is_empty() {
                            self.message = t("captions.nothingSaidClip");
                        }
                        self.refresh_text();
                        self.find(studio);
                    }
                    Err(error) if error.contains("cancel") => {}
                    Err(error) => {
                        if !self.open {
                            studio.notify(&error, true);
                        }
                        self.message = error;
                    }
                }
            }
        }
    }

    /// The clip's sound through the transcriber on a worker.
    fn transcribe(&mut self, studio: &mut Studio) {
        self.message.clear();
        let Some(clip) = self.clip.as_ref().and_then(|id| studio.clip(id)).cloned() else {
            self.message = t("captions.clipNoLongerTimeline");
            return;
        };
        let project = studio.project();
        self.clip_label = project
            .media_by_id(&clip.media_id)
            .map_or(clip.name.clone(), |media| media.name.clone());
        let Some(media) = project.media_by_id(&clip.media_id).cloned() else {
            self.message = t("captions.clipHasNoFile");
            return;
        };
        let transcribers = installed(&studio.settings.transcribers);
        let Some(model) = transcribers
            .iter()
            .find(|model| model.active)
            .or_else(|| transcribers.first())
            .map(|model| (model.id.clone(), model.megabytes))
        else {
            self.message = t("captions.downloadTranscriberModelFirst");
            return;
        };
        let (model, megabytes) = model;
        let request = concat_speech::transcribe::TranscribeRequest {
            path: media.path.clone(),
            audio_stream: clip.audio_stream,
            source_start: clip.source_start,
            window: clip.duration * clip.speed,
            model_id: model.clone(),
            language: crate::panes::captions::language(studio).map(str::to_owned),
        };
        let windows = (request.window / 30.0).ceil().max(1.0) as f32;
        let loaded = studio.host.transcriber.loaded_model().as_deref() == Some(model.as_str());
        self.pace = Pace::new(vec![
            (Stage::Reading, 0.2 + request.window as f32 / 400.0),
            (
                Stage::Loading,
                if loaded { 0.0 } else { 0.5 + megabytes / 400.0 },
            ),
            (
                Stage::Working,
                windows * (0.3 + megabytes * WINDOW_SECONDS_PER_MB),
            ),
        ]);
        self.units.clear();
        self.cuts.clear();
        self.checked.clear();
        self.refresh_text();
        self.running = true;
        let dirs = studio.host.dirs.clone();
        let transcriber = Arc::clone(&studio.host.transcriber);
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                transcriber.transcribe(&dirs, &request, move |stage| {
                    on_ui_in_project(epoch, move |studio, _, _| {
                        studio.handle(Msg::SmartCut(SmartCutMsg::Progress(stage)));
                    });
                })
            },
            |studio, _, _, result| studio.handle(Msg::SmartCut(SmartCutMsg::Finished(result))),
        );
    }

    /// The transcript box shows the units as heard, afresh.
    fn refresh_text(&mut self) {
        self.original = Transcript::from_units(&self.units).text;
        self.text = self.original.clone();
        self.revision += 1;
    }

    /// The suggestions for the units as they stand, all checked.
    fn find(&mut self, studio: &Studio) {
        let duration = self
            .clip
            .as_ref()
            .and_then(|id| studio.clip(id))
            .map(|clip| clip.duration);
        let opts = CutOptions {
            pause_min: self.pause_min,
            pause_keep: self.pause_keep,
            duration,
            ..CutOptions::default()
        };
        self.cuts = if self.units.is_empty() {
            Vec::new()
        } else {
            smart_cut::find_cuts(&self.units, &opts)
        };
        self.checked = vec![true; self.cuts.len()];
    }

    /// Cuts the words deleted from the transcript.
    fn apply_text(&mut self, studio: &mut Studio) {
        if self.text == self.original {
            self.message = t("smartCut.nothingDeleted");
            return;
        }
        let deleted = smart_cut::diff(&self.units, &self.text);
        let spans: Vec<(f64, f64)> = deleted
            .iter()
            .filter_map(|&index| self.units.get(index))
            .map(|unit| (unit.start, unit.end))
            .collect();
        self.remove(studio, smart_cut::merge_spans(&spans));
    }

    /// Takes `spans` - seconds from the clip's start - out of the
    /// timeline, every track, as one command, and moves the transcript
    /// along with it.
    fn remove(&mut self, studio: &mut Studio, spans: Vec<(f64, f64)>) {
        let Some(clip) = self.clip.as_ref().and_then(|id| studio.clip(id)) else {
            self.message = t("captions.clipNoLongerTimeline");
            return;
        };
        let (start, duration) = (clip.start, clip.duration);
        let spans: Vec<(f64, f64)> = spans
            .into_iter()
            .map(|(a, b)| (a.clamp(0.0, duration), b.clamp(0.0, duration)))
            .filter(|(a, b)| b - a > 1e-3)
            .collect();
        if spans.is_empty() {
            self.message = t("smartCut.nothingDeleted");
            return;
        }
        let seconds: f64 = spans.iter().map(|(a, b)| b - a).sum();
        let count = spans.len();
        let timeline = spans.iter().map(|(a, b)| (start + a, start + b)).collect();
        self.message.clear();
        studio.flush_commit();
        studio.apply(Command::RemoveSpans {
            spans: timeline,
            track_ids: Vec::new(),
        });
        self.units = smart_cut::after_cut(&self.units, &spans);
        self.refresh_text();
        self.find(studio);
        studio.notify(
            &tf("smartCut.removed", &[&count, &format!("{seconds:.1}")]),
            false,
        );
    }

    /// The sheet as Slint shows it.
    pub fn data(&self) -> SmartCutSheetData {
        let rows: Vec<SmartCutRowData> = self
            .cuts
            .iter()
            .zip(&self.checked)
            .map(|(cut, on)| SmartCutRowData {
                kind: kind_label(cut.kind).into(),
                time: format!("{} - {}", clock(cut.start), clock(cut.end)).into(),
                words: words_in(&self.units, cut).into(),
                checked: *on,
            })
            .collect();
        SmartCutSheetData {
            open: self.open,
            running: self.running,
            clip: self.clip_label.as_str().into(),
            has_transcript: !self.units.is_empty(),
            text: self.text.as_str().into(),
            revision: self.revision,
            cuts: std::rc::Rc::new(slint::VecModel::from(rows)).into(),
            selected: self.checked.iter().filter(|on| **on).count() as i32,
            pause_min: self.pause_min as f32,
            pause_keep: self.pause_keep as f32,
            progress: self.pace.floor,
            ceiling: self.pace.ceiling,
            step: self.pace.step,
            stage: match self.pace.stage() {
                Some(Stage::Loading) => t("captions.loadingTheModel"),
                Some(Stage::Working) => t("captions.transcribing"),
                _ => t("captions.readingTheSound"),
            }
            .into(),
            message: self.message.as_str().into(),
        }
    }
}

/// What was heard as units, a timed piece each - or a segment each, where
/// whisper timed no pieces - in seconds from the clip's start on the
/// timeline: the sound's seconds over the clip's speed.
fn units_of(segments: &[Segment], speed: f64) -> Vec<Unit> {
    let speed = if speed > 0.0 { speed } else { 1.0 };
    segments
        .iter()
        .flat_map(|segment| {
            if segment.pieces.is_empty() {
                vec![Unit::new(
                    segment.text.trim(),
                    segment.start / speed,
                    segment.end / speed,
                )]
            } else {
                segment
                    .pieces
                    .iter()
                    .map(|piece| {
                        Unit::new(piece.text.as_str(), piece.start / speed, piece.end / speed)
                    })
                    .collect()
            }
        })
        .filter(|unit| !unit.text.trim().is_empty())
        .collect()
}

/// A suggestion's kind, in words.
fn kind_label(kind: CutKind) -> String {
    match kind {
        CutKind::Filler => t("smartCut.filler"),
        CutKind::Confirm => t("smartCut.verbalTic"),
        CutKind::Repeat => t("smartCut.repeat"),
        CutKind::Pause => t("smartCut.pause"),
    }
}

/// The words a cut takes out, for its row; empty for a pause.
fn words_in(units: &[Unit], cut: &Cut) -> String {
    if cut.kind == CutKind::Pause {
        return String::new();
    }
    units
        .iter()
        .filter(|unit| unit.start >= cut.start - 1e-6 && unit.end <= cut.end + 1e-6)
        .map(|unit| unit.text.trim())
        .collect::<Vec<_>>()
        .join("")
}

/// Seconds as minutes, seconds and hundredths: `1:05.20`.
fn clock(seconds: f64) -> String {
    let hundredths = (seconds.max(0.0) * 100.0).round() as u64;
    format!(
        "{}:{:02}.{:02}",
        hundredths / 6000,
        hundredths / 100 % 60,
        hundredths % 100
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_reads_minutes_and_hundredths() {
        assert_eq!(clock(65.2), "1:05.20");
        assert_eq!(clock(0.0), "0:00.00");
    }

    #[test]
    fn units_follow_speed() {
        let segments = vec![Segment {
            start: 0.0,
            end: 2.0,
            text: "你好".into(),
            pieces: Vec::new(),
        }];
        let units = units_of(&segments, 2.0);
        assert_eq!(units.len(), 1);
        assert!((units[0].end - 1.0).abs() < 1e-9);
    }
}
