// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! The captions sheet: the tray's Captions tool, as a form and then as a
//! progress report.
//!
//! One switch says what it captions: "Use Sample Voice", on, takes
//! the words from a clip with sound, chosen in the sheet from the clips on
//! the timeline that have any; off, from a script typed in it. It opens on
//! when a clip with sound is selected, or when a clip's menu opened it on
//! that clip, and off otherwise. Either lands as one batch of title clips,
//! one undo step.
//! The transcriber runs on a worker and reports as
//! [`CaptionsMsg::Progress`], ending as [`CaptionsMsg::Finished`].
//!
//! The sheet also reads a SubRip file in as captions, with the same look,
//! and writes the timeline's captions out as one.
//!
//! A transcription is cut into lines the way CapCut cuts its captions:
//! at the punctuation, no wider than the frame suits, the closing comma
//! dropped (see [`concat_project::caption_lines`]). With a script typed
//! or read in while the switch is on, the captions are the script's
//! words, timed by what the transcriber heard - CapCut's 文稿匹配.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use concat_project::Command;
use concat_project::caption_lines::{self, LineRules, Timed};
use concat_project::model::{Clip, ClipKind, MediaItem, MediaOrigin, TextStyle, VoiceKind};
use concat_project::srt;
use concat_speech::JobStage;
use concat_speech::transcribe::Segment;

use crate::host::{on_ui_in_project, spawn_in_project};
use crate::i18n::{self, t, tf};
use crate::panes::Msg;
use crate::panes::pace::{Pace, Stage, Timings};
use crate::panes::settings::installed;
use crate::studio::Studio;
use crate::ui::CaptionsSheetData;

/// Where a caption sits, by the sheet's row: a frame-height fraction from
/// the centre, positive down. Bottom, centre, top.
const CAPTION_OFFSETS: [f64; 3] = [0.35, 0.0, -0.35];
/// A caption's cap height by the sheet's row, as a fraction of the frame.
const CAPTION_SIZES: [f64; 3] = [0.04, 0.05, 0.065];
/// A rough speaking rate, for a script's timing and the speech sheet's
/// estimate.
pub const CHARS_PER_SECOND: f32 = 14.0;

/// Everything that can happen to the captions sheet.
#[derive(Clone, Debug)]
pub enum CaptionsMsg {
    /// The tray's Captions tool: the selected clip's voice when it has
    /// sound, the script otherwise.
    Open,
    /// The tray's "Auto captions": no sheet to fill in. The selected clip's
    /// voice, else every clip on the timeline that may be speaking - music
    /// left out - one after another, transcribed at once with the
    /// installed model, in Simplified Chinese under a Simplified
    /// Chinese interface and Traditional under 繁體中文, unless another
    /// language was picked, lines cut by the
    /// frame; see [`CaptionsPane::auto_caption`].
    AutoCaption,
    /// A clip's menu: the sheet on that clip's voice.
    OpenOnClip(String),
    Close,
    /// The library's Captions page came into view, or left it: a desk's
    /// home for what the phone's sheet shows.
    Shown(bool),
    /// The "Use Sample Voice" switch.
    VoiceToggled(bool),
    /// Row in the list of clips with sound.
    SourceChanged(i32),
    TextEdited(String),
    ModelChanged(i32),
    /// Row in the language list: 0 is "Auto-detect", then
    /// [`language_list`] in its order.
    LanguageChanged(i32),
    PlacementChanged(i32),
    SizeChanged(i32),
    /// Run the pass the sheet describes.
    Begin,
    Cancel,
    /// The transcriber's worker reporting the stage it is in.
    Progress(JobStage),
    /// The transcriber's worker is done: what was said, or why not.
    Finished(Result<Vec<Segment>, String>),
    /// Asks for a SubRip file to read in.
    ImportSrt,
    /// The SubRip file chosen: its cues as captions.
    ImportSrtFrom(PathBuf),
    /// How long a line may run: 0 by the frame, 1 short, 2 long.
    LineLengthChanged(i32),
    /// Drop the comma or full stop a line ends on.
    DropPunctuationToggled(bool),
    /// Asks for a plain-text script to read in.
    ImportScript,
    /// The script file chosen: its words as the sheet's script.
    ImportScriptFrom(PathBuf),
    /// Asks where to write the captions as SubRip.
    ExportSrt,
    /// Where they go.
    ExportSrtTo(PathBuf),
}

/// The clip a transcription is running over, as the finished segments
/// need it: where it starts on the timeline and how fast it plays.
#[derive(Clone, Debug)]
struct Subject {
    start: f64,
    speed: f64,
    /// The look chosen when the run began.
    look: (f64, f64),
    /// The transcriber model, and the thirty-second windows whisper reads
    /// the sound in: what the run's timing is kept by.
    model: String,
    windows: f32,
    /// How the words are cut into lines.
    rules: LineRules,
    /// The script the words are matched to, when one was given.
    script: Option<String>,
    /// Where the clip stands among the run's clips, 0 first: when two of
    /// them caption the same moment, the lower rank's words are kept.
    rank: usize,
    /// The clip's span on the timeline, start and end.
    span: (f64, f64),
}

/// One clip's captions, heard and waiting for the run's other clips.
#[derive(Clone, Debug)]
struct Gathered {
    rank: usize,
    span: (f64, f64),
    look: (f64, f64),
    lines: Vec<Placed>,
}

/// A caption in timeline seconds.
#[derive(Clone, Debug, PartialEq)]
struct Placed {
    text: String,
    start: f64,
    duration: f64,
}

/// Seconds whisper takes over one thirty-second window, per megabyte of
/// model, before a run on the model has been timed: about a second for
/// base and ten for medium on eight cores of a recent CPU.
const WINDOW_SECONDS_PER_MB: f32 = 1.0 / 160.0;

/// The captions sheet's state.
#[derive(Default)]
pub struct CaptionsPane {
    /// The sheet is up: a phone's, or a window too narrow for the library.
    pub open: bool,
    /// The library's Captions page is up, on a desk.
    pub shown: bool,
    /// "Use Sample Voice": the words come from [`Self::clip`]'s sound
    /// when on, from [`Self::text`] when off.
    pub from_voice: bool,
    /// The clip whose voice is transcribed: one with sound, chosen from the
    /// timeline's. Kept while the switch is off, so turning it back on
    /// returns to the same clip. The panel floats over a live editor, so
    /// a clip with sound selected since is the one shown and run instead;
    /// see [`Self::subject_clip`].
    pub clip: Option<String>,
    /// The sole selection when [`Self::clip`] was chosen - opened on, or
    /// picked from the list - so a later selection can be told from it.
    chosen_with: Option<String>,
    /// The script's words.
    pub text: String,
    /// Row in the installed transcriber list.
    pub model: usize,
    /// The language the voice speaks, by whisper's code; `None` is
    /// whisper's to hear. Remembered across runs as
    /// `Preferences::caption_language`, so a person who captions in
    /// Indonesian says so once. An English-only model ignores it.
    pub language: Option<String>,
    /// 0 bottom, 1 centre, 2 top.
    pub placement: usize,
    /// 0 small, 1 medium, 2 large.
    pub size: usize,
    pub running: bool,
    /// How far the running transcription looks.
    pub pace: Pace,
    /// Why the last run failed, when it did.
    pub message: String,
    /// The clip under the running transcription.
    subject: Option<Subject>,
    /// How long each model took this session, for the next run's bar.
    timings: Timings,
    /// What the style picker's search holds.
    pub style_query: String,
    /// The clips still to transcribe in this run, in timeline order, with
    /// their rank; see [`auto_subjects`].
    queue: Vec<(String, usize)>,
    /// What the run's clips heard so far, placed together when the last
    /// one is done - one undo step.
    gathered: Vec<Gathered>,
    /// The captions the last run placed, by timeline: a run over the same
    /// clips replaces them instead of stacking a second set.
    placed: Option<(String, Vec<String>)>,
}

impl CaptionsPane {
    /// Applies one message. The studio is the rest of the window; while
    /// this runs the studio's copy of the pane is a blank it must not read.
    pub fn update(&mut self, msg: CaptionsMsg, studio: &mut Studio) {
        match msg {
            // A run in progress is shown as it is, however the panel is
            // asked for again: closing it never stopped the run.
            CaptionsMsg::Open | CaptionsMsg::OpenOnClip(_) if self.running => {
                self.open = true;
                self.show_in_library(studio);
            }
            CaptionsMsg::AutoCaption if self.running => {
                studio.notify(&t("captions.autoAlreadyRunning"), false);
                self.show_in_library(studio);
            }
            CaptionsMsg::AutoCaption => self.auto_caption(studio),
            CaptionsMsg::Open => {
                let selected = studio.sole_selection();
                self.open_on(studio, selected, false);
                self.show_in_library(studio);
            }
            CaptionsMsg::OpenOnClip(id) => {
                self.open_on(studio, Some(id), true);
                self.show_in_library(studio);
            }
            CaptionsMsg::Close => self.open = false,
            CaptionsMsg::Shown(shown) => {
                // First shown with nothing asked for yet: the panel as the
                // tray's button would set it up, without the sheet.
                if shown
                    && !self.shown
                    && !self.open
                    && !self.running
                    && self.clip.is_none()
                    && self.text.is_empty()
                {
                    let selected = studio.sole_selection();
                    self.open_on(studio, selected, false);
                    self.open = false;
                }
                self.shown = shown;
            }
            CaptionsMsg::VoiceToggled(on) => {
                self.from_voice = on;
                self.message.clear();
                let sources = sources(studio);
                if on
                    && !self
                        .clip
                        .as_ref()
                        .is_some_and(|id| sources.iter().any(|s| &s.id == id))
                {
                    self.clip = sources.first().map(|source| source.id.clone());
                    self.chosen_with = studio.sole_selection();
                }
            }
            CaptionsMsg::SourceChanged(index) => {
                if let Some(source) = usize::try_from(index)
                    .ok()
                    .and_then(|index| sources(studio).into_iter().nth(index))
                {
                    self.clip = Some(source.id);
                    self.chosen_with = studio.sole_selection();
                    self.message.clear();
                }
            }
            CaptionsMsg::TextEdited(text) => self.text = text,
            CaptionsMsg::ModelChanged(index) => self.model = index.max(0) as usize,
            CaptionsMsg::LanguageChanged(index) => {
                self.language = language_at(index);
                studio.prefs.caption_language = self.language.clone();
                studio.prefs.save(&studio.host.dirs);
            }
            CaptionsMsg::PlacementChanged(index) => {
                self.placement = (index.max(0) as usize).min(2);
            }
            CaptionsMsg::SizeChanged(index) => self.size = (index.max(0) as usize).min(2),
            CaptionsMsg::LineLengthChanged(index) => {
                studio.prefs.captions.line_length = index.clamp(0, 2) as u8;
                studio.prefs.save(&studio.host.dirs);
            }
            CaptionsMsg::DropPunctuationToggled(on) => {
                studio.prefs.captions.drop_end_punctuation = on;
                studio.prefs.save(&studio.host.dirs);
            }
            CaptionsMsg::ImportScript => crate::host::pick_files_then(
                t("captions.importScript"),
                Some((t("captions.plainText"), &["txt"])),
                |studio, _, _, paths| {
                    if let Some(path) = paths.into_iter().next() {
                        studio.handle(Msg::Captions(CaptionsMsg::ImportScriptFrom(path)));
                    }
                },
            ),
            CaptionsMsg::ImportScriptFrom(path) => self.import_script(&path),
            CaptionsMsg::Begin => {
                if self.from_voice {
                    self.run_sound(studio);
                } else {
                    self.run_script(studio);
                }
            }
            CaptionsMsg::Cancel => {
                studio.host.transcriber.cancel();
                self.running = false;
                self.open = false;
                self.queue.clear();
                self.gathered.clear();
            }
            CaptionsMsg::Progress(stage) => self.pace.report(stage),
            CaptionsMsg::Finished(result) => {
                self.running = false;
                let subject = self.subject.take();
                if let (Ok(_), Some(subject)) = (&result, &subject) {
                    self.timings
                        .keep(&subject.model, self.pace.learned(subject.windows));
                }
                match result {
                    Ok(segments) => {
                        let Some(subject) = subject else {
                            return;
                        };
                        let lines =
                            caption_lines_of(&segments, subject.script.as_deref(), &subject.rules)
                                .into_iter()
                                .map(|line| Placed {
                                    text: line.text,
                                    start: subject.start + line.start / subject.speed,
                                    duration: ((line.end - line.start) / subject.speed).max(0.2),
                                })
                                .collect();
                        self.gathered.push(Gathered {
                            rank: subject.rank,
                            span: subject.span,
                            look: subject.look,
                            lines,
                        });
                        if !self.next_in_queue(studio) {
                            self.place_gathered(studio);
                        }
                    }
                    // Asked for: the sheet is already on its way down.
                    Err(error) if error.contains("cancel") => {
                        self.queue.clear();
                        self.gathered.clear();
                        self.open = false;
                    }
                    // Closed while it ran: the panel is not there to say
                    // it, so the notice does, and the panel too next time.
                    // What the run's earlier clips heard is kept.
                    Err(error) => {
                        self.queue.clear();
                        if !self.gathered.is_empty() {
                            self.place_gathered(studio);
                        }
                        if !self.open {
                            studio.notify(&error, true);
                        }
                        self.message = error;
                    }
                }
            }
            // Asked with the studio unborrowed, on a turn of its own; see
            // `host::after_dialog`.
            CaptionsMsg::ImportSrt => crate::host::pick_files_then(
                t("captions.importSrt"),
                Some((t("captions.subRipSubtitles"), &["srt"])),
                |studio, _, _, paths| {
                    if let Some(path) = paths.into_iter().next() {
                        studio.handle(Msg::Captions(CaptionsMsg::ImportSrtFrom(path)));
                    }
                },
            ),
            CaptionsMsg::ImportSrtFrom(path) => self.import_srt(studio, &path),
            CaptionsMsg::ExportSrt => self.ask_export_srt(studio),
            CaptionsMsg::ExportSrtTo(path) => self.export_srt(studio, &path),
        }
    }

    /// A plain-text file's words as the sheet's script: typed captions
    /// with the switch off, the words a transcription is matched to with
    /// it on.
    fn import_script(&mut self, path: &Path) {
        let text = std::fs::read(path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| {
                caption_lines::decode_script(&bytes).ok_or_else(|| t("captions.scriptNotUtf8"))
            });
        match text {
            Ok(text) if text.trim().is_empty() => self.message = t("captions.nothingToCaptionYet"),
            Ok(text) => {
                self.text = text.trim().to_owned();
                self.message.clear();
            }
            Err(error) => {
                log::warn!("captions: could not read {}: {error}", path.display());
                self.message = tf("captions.couldNotReadScript", &[&error]);
            }
        }
    }

    /// A SubRip file's cues as captions where the file times them, with
    /// the sheet's look: one batch, one undo step. A file that is not
    /// UTF-8 is read as best it can be.
    fn import_srt(&mut self, studio: &mut Studio, path: &Path) {
        let cues = std::fs::read(path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| {
                srt::parse(&String::from_utf8_lossy(&bytes)).map_err(|error| error.to_string())
            });
        let cues = match cues {
            Ok(cues) => cues,
            Err(error) => {
                log::warn!(
                    "captions: reading the subtitles in {} failed: {error}; no captions were placed",
                    crate::studio::file_name_of(&path.to_string_lossy())
                );
                self.message = tf("captions.couldNotReadSubtitles", &[&error]);
                return;
            }
        };
        let look = self.look();
        let (base, font) = caption_base(studio);
        let commands = srt_commands(cues, look, base.as_ref());
        if commands.is_empty() {
            self.message = t("captions.noSubtitlesInFile");
            return;
        }
        let count = commands.len();
        self.open = false;
        log::info!(
            "captions: placed {count} captions from the subtitles in {}",
            crate::studio::file_name_of(&path.to_string_lossy())
        );
        let _told = crate::studio::Asked::told();
        studio.apply(Command::Batch {
            commands: font.into_iter().chain(commands).collect(),
        });
        studio.notify(&tf("captions.addedCaptions", &[&count]), false);
    }

    /// Asks where to write the captions, offering the project's folder
    /// and its name. A phone has no such dialog: the file is written
    /// beside the project under a name nothing there has, and handed to
    /// the phone to put where its file manager shows it.
    fn ask_export_srt(&mut self, studio: &mut Studio) {
        if export_lane(studio).is_none() {
            self.message = t("captions.noCaptionsToExport");
            return;
        }
        let name = format!("{}.srt", file_stem(studio));
        // No project, no project folder: Documents, as anywhere else a
        // file is written with nothing better to offer.
        let folder = studio
            .session
            .as_ref()
            .map(|session| PathBuf::from(session.path()))
            .unwrap_or_else(|| PathBuf::from(crate::studio::home_folder("Documents")));
        if cfg!(any(target_os = "android", target_os = "ios")) {
            let path = unique_path(&folder, &name);
            self.export_srt(studio, &path);
            return;
        }
        let title = t("captions.exportSrt");
        let family = t("captions.subRipSubtitles");
        crate::host::after_dialog(
            move || {
                let chosen = crate::platform::save_file(
                    &title,
                    &folder,
                    &name,
                    (family.as_str(), &["srt"]),
                )?;
                Some(
                    if chosen
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("srt"))
                    {
                        chosen
                    } else {
                        chosen.with_extension("srt")
                    },
                )
            },
            |studio, _, _, path| studio.handle(Msg::Captions(CaptionsMsg::ExportSrtTo(path))),
        );
    }

    /// Writes the captions to `path` as SubRip, and says how many and
    /// where - on a phone, once the phone has put the file where it shows
    /// it.
    fn export_srt(&mut self, studio: &mut Studio, path: &Path) {
        let Some(lane) = export_lane(studio) else {
            self.message = t("captions.noCaptionsToExport");
            return;
        };
        let cues = srt::title_cues(studio.timeline(), Some(&lane.track_id));
        if let Err(error) = std::fs::write(path, srt::write(&cues)) {
            log::warn!(
                "captions: writing the subtitles to {} failed: {error}; nothing was exported",
                crate::studio::file_name_of(&path.to_string_lossy())
            );
            self.message = tf("captions.couldNotWriteSubtitles", &[&error]);
            return;
        }
        log::info!(
            "captions: exported {} captions to {}",
            cues.len(),
            crate::studio::file_name_of(&path.to_string_lossy())
        );
        self.open = false;
        let count = cues.len();
        // Which lane was written goes without saying when it was the
        // selected title's, or the only lane with titles; otherwise the
        // notice says it was the busiest, and how to pick another.
        let busiest = lane.busiest && srt::title_lanes(studio.timeline()) > 1;
        let say = move |studio: &mut Studio, place: &str| {
            // Each key spelled out where it is used: scripts/locales.py
            // finds the strings the source asks for by reading it.
            let said = if busiest {
                tf("captions.exportedBusiestLane", &[&count, &place])
            } else {
                tf("captions.exportedCaptions", &[&count, &place])
            };
            studio.notify(&said, false);
        };
        let written = path.to_path_buf();
        let handed = crate::platform::publish_document(
            path.to_path_buf(),
            "application/x-subrip",
            move |result| {
                crate::host::on_ui(move |studio, _, _| match result {
                    Ok(place) => say(studio, &place),
                    Err(error) => {
                        log::warn!(
                            "captions: wanted {} handed to the phone's files and it was not: {error}; it stays in the project's folder",
                            crate::studio::file_name_of(&written.to_string_lossy())
                        );
                        studio.notify(
                            &tf(
                                "captions.couldNotPublishSubtitles",
                                &[&written.display(), &error],
                            ),
                            true,
                        );
                    }
                });
            },
        );
        if !handed {
            say(studio, &path.display().to_string());
        }
    }

    /// The sheet opened afresh. `clip` is the clip it was opened on, if
    /// any: the switch starts on when that clip has sound, or when a clip's
    /// menu asked for its voice (`voice`), in which case a clip without
    /// sound falls to the first clip that has some.
    /// On a desk the panel is the library's Captions page, not a sheet:
    /// the library is put in the dock if it was taken out, and turned to
    /// that page. A phone, or a window too narrow for the library beside
    /// the work, keeps the sheet.
    fn show_in_library(&mut self, studio: &mut Studio) {
        if studio.meters.phone || studio.compact {
            return;
        }
        self.open = false;
        if !studio.dock.contains(crate::ui::PaneKind::Media) {
            crate::add_pane(studio, crate::ui::PaneKind::Media);
            studio.remember_layout_soon();
        }
        studio.captions_jump += 1;
    }

    /// One press, captions: what the tray's "Auto captions" does. The
    /// selected clip when it has sound - the person's own pick, even a song;
    /// else every clip on the timeline that may be speaking, one after
    /// another in timeline order, each clip's captions put at its own place
    /// on the timeline. What the timeline already says is music - beats
    /// found in it, its voice taken out, a name like "bgm", a long quiet
    /// bed under the voice - is left out, and so is what cannot be heard
    /// (muted, at zero, on a muted lane); see [`hearing`] and
    /// [`auto_subjects`]. The transcriber listens for speech before it
    /// transcribes, so a clip of music alone that slips through still adds
    /// nothing; where two clips caption the same moment, the likelier
    /// voice's words are kept. A second press over the same clips replaces
    /// the captions the first one placed instead of adding a second set.
    /// The language is the one picked in the sheet before, else Chinese
    /// under a Chinese interface - Simplified under 简体中文, Traditional
    /// under 繁體中文 (only asked of whisper, by its language and prompt:
    /// Simplified is also converted afterwards, but there is no
    /// Simplified-to-Traditional pass, so Traditional output may mix the
    /// two scripts; see concat_speech::hans), else whisper's to hear. An
    /// English-only model is passed over for an installed one that hears
    /// the language, and said when there is none. Lines are cut as the
    /// sheet's saved choice says, by the frame unless changed, and end
    /// without a comma or full stop unless that was turned off. With no
    /// transcriber installed, or nothing to hear on the timeline, the panel
    /// opens on why.
    fn auto_caption(&mut self, studio: &mut Studio) {
        let heard: Vec<Heard> = {
            let timeline = studio.timeline();
            let project = studio.project();
            timeline
                .clips
                .iter()
                .filter(|clip| studio.clip_has_sound(clip))
                .map(|clip| Heard {
                    id: clip.id.clone(),
                    start: clip.start,
                    duration: clip.duration,
                    volume: clip.volume,
                    bed: clip.kind == ClipKind::Audio && clip.detached_from.is_none(),
                    hearing: hearing(
                        clip,
                        project.media_by_id(&clip.media_id),
                        timeline
                            .track(&clip.track_id)
                            .is_some_and(|track| track.muted),
                    ),
                })
                .collect()
        };
        let selected = studio.sole_selection();
        let subjects = auto_subjects(selected.as_deref(), &heard);
        let Some((first, _)) = subjects.first().cloned() else {
            self.open_on(studio, None, false);
            self.message = if nothing_heard(&heard) {
                t("captions.autoNothingWithSound")
            } else {
                t("captions.autoOnlyMusic")
            };
            studio.notify(&self.message, true);
            self.show_in_library(studio);
            return;
        };
        self.open_on(studio, Some(first), true);
        // `open_on` kept the sheet's remembered pick when it is still one
        // the list offers.
        self.language = auto_language(&i18n::current(), self.language.as_deref());
        let wants_other = self.language.as_deref().is_some_and(|code| code != "en");
        let ids: Vec<String> = installed(&studio.settings.transcribers)
            .iter()
            .map(|model| model.id.clone())
            .collect();
        let (model, deaf) = hearing_model(&ids, self.model, wants_other);
        self.model = model;
        log::info!(
            "captions: auto · language {} · {} of {} clips with sound · model {}",
            self.language.as_deref().unwrap_or("heard"),
            subjects.len(),
            heard.len(),
            ids.get(model).map_or("none", String::as_str),
        );
        if deaf {
            studio.notify(&t("captions.englishOnlyModel"), true);
        }
        self.queue = subjects;
        if self.next_in_queue(studio) {
            studio.notify(&t("captions.autoStarted"), false);
        } else if !self.message.is_empty() {
            studio.notify(&self.message, true);
        }
        self.show_in_library(studio);
    }

    /// Starts the run's next clip, passing over one gone from the timeline
    /// or without a file; false when none is left to start.
    fn next_in_queue(&mut self, studio: &mut Studio) -> bool {
        while !self.queue.is_empty() {
            let (id, rank) = self.queue.remove(0);
            if let Some(clip) = studio.clip(&id).cloned() {
                self.transcribe(studio, clip, rank);
                if self.running {
                    return true;
                }
            }
        }
        false
    }

    /// The run's captions onto the timeline as one batch: one undo step,
    /// taking with it the captions the last run placed over the same
    /// clips, so pressing again replaces rather than stacks.
    fn place_gathered(&mut self, studio: &mut Studio) {
        let gathered = std::mem::take(&mut self.gathered);
        self.open = false;
        let Some(look) = gathered.first().map(|clip| clip.look) else {
            return;
        };
        let spans: Vec<(f64, f64)> = gathered.iter().map(|clip| clip.span).collect();
        let from = spans
            .iter()
            .map(|span| span.0)
            .fold(f64::INFINITY, f64::min);
        let lines = merge_lines(gathered);
        if lines.is_empty() {
            // A run over several clips heard nothing on the timeline, not
            // in "that clip".
            log::info!(
                "captions: placed none · nothing was said in {} clips",
                spans.len()
            );
            let said = if spans.len() > 1 {
                t("captions.nothingSaidTimeline")
            } else {
                t("captions.nothingSaidClip")
            };
            studio.notify(&said, true);
            return;
        }
        let timeline = studio.timeline();
        let timeline_id = timeline.id.clone();
        let earlier: Vec<(String, f64)> = match &self.placed {
            Some((id, placed)) if *id == timeline_id => placed
                .iter()
                .filter_map(|id| timeline.clip(id))
                .filter(|clip| clip.kind == ClipKind::Text)
                .map(|clip| (clip.id.clone(), clip.start))
                .collect(),
            _ => Vec::new(),
        };
        let stale = replaced(&earlier, &spans);
        let before: HashSet<String> = timeline.clips.iter().map(|clip| clip.id.clone()).collect();
        let (base, font) = caption_base(studio);
        let count = lines.len();
        let remove = (!stale.is_empty()).then(|| Command::RemoveClips {
            clip_ids: stale.clone(),
            ripple: false,
        });
        let commands: Vec<Command> = remove
            .into_iter()
            .chain(font)
            .chain(lines.into_iter().map(|line| {
                caption_clip(line.text, line.start, line.duration, look, base.as_ref())
            }))
            .collect();
        log::info!(
            "captions: placed {count} captions from {from:.2} s over {} clips, replacing {}",
            spans.len(),
            stale.len()
        );
        let _told = crate::studio::Asked::told();
        studio.apply(Command::Batch { commands });
        let mut placed: Vec<String> = earlier
            .into_iter()
            .map(|(id, _)| id)
            .filter(|id| !stale.contains(id))
            .collect();
        placed.extend(
            studio
                .timeline()
                .clips
                .iter()
                .filter(|clip| clip.kind == ClipKind::Text && !before.contains(&clip.id))
                .map(|clip| clip.id.clone()),
        );
        self.placed = Some((timeline_id, placed));
        if stale.is_empty() {
            studio.notify(&tf("captions.addedCaptions", &[&count]), false);
        } else {
            studio.notify(&tf("captions.replacedCaptions", &[&count]), false);
        }
    }

    fn open_on(&mut self, studio: &Studio, clip: Option<String>, voice: bool) {
        let installed = installed(&studio.settings.transcribers);
        let model = installed.iter().position(|model| model.active).unwrap_or(0);
        let sources = sources(studio);
        let sounding = clip.filter(|id| sources.iter().any(|source| &source.id == id));
        let from_voice = voice || sounding.is_some();
        let clip = sounding.or_else(|| {
            from_voice
                .then(|| sources.first().map(|source| source.id.clone()))
                .flatten()
        });
        *self = CaptionsPane {
            open: true,
            from_voice,
            clip,
            chosen_with: studio.sole_selection(),
            model,
            // A code the list no longer has is heard instead.
            language: studio
                .prefs
                .caption_language
                .clone()
                .filter(|code| language_row(Some(code)) > 0),
            placement: 0,
            size: 1,
            timings: std::mem::take(&mut self.timings),
            placed: self.placed.take(),
            shown: self.shown,
            ..CaptionsPane::default()
        };
    }

    /// The clip the panel transcribes, and the one its list shows - the
    /// same answer for both, so it never runs a clip it is not showing.
    /// The editor stays live under the panel: a clip with sound selected
    /// since the panel's clip was chosen is the one meant now, and is
    /// taken. Otherwise the chosen clip while it is still on the timeline
    /// with sound, and none - nothing to run - when it is not.
    fn subject_clip(&self, studio: &Studio, sources: &[Source]) -> Option<String> {
        follow_selection(
            studio.sole_selection(),
            self.chosen_with.as_deref(),
            self.clip.as_deref(),
            |id| sources.iter().any(|source| source.id == id),
        )
    }

    /// A caption's look, by the sheet's rows: where it sits and its size.
    fn look(&self) -> (f64, f64) {
        (
            CAPTION_OFFSETS[self.placement.min(2)],
            CAPTION_SIZES[self.size.min(2)],
        )
    }

    /// The script as titles, one after another from the playhead.
    fn run_script(&mut self, studio: &mut Studio) {
        let lines = script_captions(&self.text);
        if lines.is_empty() {
            self.message = t("captions.nothingToCaptionYet");
            return;
        }
        let look = self.look();
        let mut at = f64::from(studio.playhead);
        let (base, font) = caption_base(studio);
        let commands: Vec<Command> = lines
            .into_iter()
            .map(|(text, seconds)| {
                let command = caption_clip(text, at, seconds, look, base.as_ref());
                at += seconds;
                command
            })
            .collect();
        let count = commands.len();
        self.open = false;
        log::info!(
            "captions: placed {count} captions from a script, from {:.2} s",
            studio.playhead
        );
        let _told = crate::studio::Asked::told();
        studio.apply(Command::Batch {
            commands: font.into_iter().chain(commands).collect(),
        });
        studio.notify(&tf("captions.addedCaptions", &[&count]), false);
    }

    /// The sheet's clip through the transcriber on a worker, reporting into
    /// the sheet as it goes.
    fn run_sound(&mut self, studio: &mut Studio) {
        // The clip the panel shows, and no other: see `subject_clip`.
        let Some(clip) = self
            .subject_clip(studio, &sources(studio))
            .and_then(|id| studio.clip(&id))
            .cloned()
        else {
            self.message = t("captions.clipNoLongerTimeline");
            return;
        };
        self.queue.clear();
        self.gathered.clear();
        self.transcribe(studio, clip, 0);
    }

    /// One clip through the transcriber on a worker; `rank` its place
    /// among the run's clips. The sheet's list shows it while it runs.
    fn transcribe(&mut self, studio: &mut Studio, clip: Clip, rank: usize) {
        self.clip = Some(clip.id.clone());
        self.chosen_with = studio.sole_selection();
        let Some(media) = studio.project().media_by_id(&clip.media_id).cloned() else {
            self.message = t("captions.clipHasNoFile");
            return;
        };
        let Some((model, megabytes)) = installed(&studio.settings.transcribers)
            .get(self.model)
            .map(|model| (model.id.clone(), model.megabytes))
        else {
            self.message = t("captions.downloadTranscriberModelFirst");
            return;
        };
        let request = concat_speech::transcribe::TranscribeRequest {
            path: media.path.clone(),
            audio_stream: clip.audio_stream,
            source_start: clip.source_start,
            window: clip.duration * clip.speed,
            model_id: model.clone(),
            language: (!concat_speech::transcribe::english_only(&model))
                .then(|| request_language(self.language.as_deref()))
                .flatten()
                .map(str::to_owned),
        };
        let dirs = studio.host.dirs.clone();
        let transcriber = Arc::clone(&studio.host.transcriber);
        // The run as the bar lays it out: the sound read in at a few
        // hundred times its length, the model loaded unless it still is,
        // and whisper a window at a time - at the pace the last run on
        // this model measured, or the estimate by its size.
        let windows = (request.window / 30.0).ceil().max(1.0) as f32;
        let learned = self.timings.get(&model);
        let loaded = studio.host.transcriber.loaded_model().as_deref() == Some(model.as_str());
        let load = if loaded {
            0.0
        } else {
            learned.load.unwrap_or(0.5 + megabytes / 400.0)
        };
        let per_window = learned
            .per_unit
            .unwrap_or(0.3 + megabytes * WINDOW_SECONDS_PER_MB);
        self.pace = Pace::new(vec![
            (Stage::Reading, 0.2 + request.window as f32 / 400.0),
            (Stage::Loading, load),
            (Stage::Working, windows * per_window),
        ]);
        let script = self.text.trim();
        self.subject = Some(Subject {
            start: clip.start,
            speed: clip.speed,
            look: self.look(),
            model,
            windows,
            rules: line_rules(studio),
            script: (!script.is_empty()).then(|| script.to_owned()),
            rank,
            span: (clip.start, clip.start + clip.duration),
        });
        self.running = true;
        self.message.clear();
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                transcriber.transcribe(&dirs, &request, move |stage| {
                    on_ui_in_project(epoch, move |studio, _, _| {
                        studio.handle(Msg::Captions(CaptionsMsg::Progress(stage)));
                    });
                })
            },
            |studio, _, _, result| studio.handle(Msg::Captions(CaptionsMsg::Finished(result))),
        );
    }

    /// Whether the model the sheet has chosen hears only English.
    fn english_only(&self, studio: &Studio) -> bool {
        installed(&studio.settings.transcribers)
            .get(self.model)
            .is_some_and(|model| concat_speech::transcribe::english_only(&model.id))
    }

    /// The sheet's language list: "Auto-detect", then every language
    /// whisper knows by its own name. An English-only model hears English
    /// and nothing else, so its list is English alone, and is not a choice.
    /// Empty while the sheet is closed, which is most of the time.
    pub fn language_rows(&self, studio: &Studio) -> Vec<slint::SharedString> {
        if !self.open && !self.shown {
            return Vec::new();
        }
        let language = crate::i18n::current();
        if self.english_only(studio) {
            return vec![shown_name("en", ENGLISH, &language).into()];
        }
        std::iter::once(t("captions.autoDetect").into())
            .chain(language_list().map(|(code, name)| shown_name(code, name, &language).into()))
            .collect()
    }

    /// The sheet as Slint shows it.
    pub fn data(&self, studio: &Studio) -> CaptionsSheetData {
        let sources = if self.open || self.shown {
            sources(studio)
        } else {
            Vec::new()
        };
        let source = self
            .subject_clip(studio, &sources)
            .and_then(|id| sources.iter().position(|source| source.id == id))
            .map_or(-1, |index| index as i32);
        let labels: Vec<slint::SharedString> = sources
            .into_iter()
            .map(|source| source.label.into())
            .collect();
        let english_only = self.english_only(studio);
        CaptionsSheetData {
            open: self.open,
            from_voice: self.from_voice,
            sources: std::rc::Rc::new(slint::VecModel::from(labels)).into(),
            source,
            text: self.text.as_str().into(),
            model: self.model as i32,
            language: if english_only {
                0
            } else {
                language_row(self.language.as_deref()) as i32
            },
            language_fixed: english_only,
            placement: self.placement as i32,
            size: self.size as i32,
            running: self.running,
            progress: self.pace.floor,
            ceiling: self.pace.ceiling,
            step: self.pace.step,
            left: self.pace.left,
            stage: match self.pace.stage() {
                Some(Stage::Loading) => t("captions.loadingTheModel"),
                Some(Stage::Working) => t("captions.transcribing"),
                _ => t("captions.readingTheSound"),
            }
            .into(),
            ready: !installed(&studio.settings.transcribers).is_empty(),
            message: self.message.as_str().into(),
            line_length: i32::from(studio.prefs.captions.line_length.min(2)),
            drop_punctuation: studio.prefs.captions.drop_end_punctuation,
        }
    }
}

/// The language a transcription is asked for: the one picked on the
/// sheet, or with none picked - "Auto-detect" - none, so the transcriber
/// hears which language it is, whatever language the interface is in.
/// Chinese in Traditional characters is its own row, picked like any other.
fn request_language(picked: Option<&str>) -> Option<&str> {
    picked
}

/// How the sheet's choices cut lines: the line as long as the choice says,
/// or by the frame - short in a tall one, a phone's, long in a wide one.
fn line_rules(studio: &Studio) -> LineRules {
    let prefs = &studio.prefs.captions;
    let max_width = match prefs.line_length {
        1 => LineRules::VERTICAL_WIDTH,
        2 => LineRules::HORIZONTAL_WIDTH,
        _ => {
            let (width, height) = studio.output_size();
            LineRules::width_for_frame(width, height)
        }
    };
    LineRules {
        max_width,
        strip_trailing: prefs.drop_end_punctuation,
        ..LineRules::default()
    }
}

/// A transcription as caption lines, in the transcription's seconds: each
/// segment a phrase of its timed tokens - or of itself, when whisper timed
/// none - and with a script, the script's words timed by them. A script
/// nothing in the speech lines up with leaves the words as heard.
fn caption_lines_of(
    segments: &[Segment],
    script: Option<&str>,
    rules: &LineRules,
) -> Vec<caption_lines::Line> {
    let heard: Vec<Vec<Timed>> = segments
        .iter()
        .map(|segment| {
            if segment.pieces.is_empty() {
                vec![Timed::new(
                    segment.text.as_str(),
                    segment.start,
                    segment.end,
                )]
            } else {
                segment
                    .pieces
                    .iter()
                    .map(|piece| Timed::new(piece.text.as_str(), piece.start, piece.end))
                    .collect()
            }
        })
        .collect();
    let matched = script
        .map(|script| caption_lines::align_script(script, &heard))
        .filter(|matched| !matched.is_empty());
    caption_lines::lines(matched.as_ref().unwrap_or(&heard), rules)
}

/// English, as the language list names it.
const ENGLISH: &str = "English";

/// A row of the language list as the sheet shows it: under a Simplified
/// Chinese interface, the language's Chinese name, so a list of ninety-odd
/// languages reads in the interface's words; under any other, its own name.
/// The two Chinese rows keep theirs, which say which characters they write.
fn shown_name(code: &str, own: &str, interface: &str) -> String {
    if interface == "zh-Hans"
        && !code.starts_with("zh")
        && let Some((_, chinese)) = ZH_HANS_NAMES.iter().find(|(known, _)| *known == code)
    {
        return (*chinese).to_owned();
    }
    own.to_owned()
}

/// Whisper's languages by their names in Simplified Chinese.
pub(crate) const ZH_HANS_NAMES: &[(&str, &str)] = &[
    ("af", "南非荷兰语"),
    ("az", "阿塞拜疆语"),
    ("id", "印度尼西亚语"),
    ("ms", "马来语"),
    ("jw", "爪哇语"),
    ("su", "巽他语"),
    ("bs", "波斯尼亚语"),
    ("br", "布列塔尼语"),
    ("ca", "加泰罗尼亚语"),
    ("cs", "捷克语"),
    ("sn", "绍纳语"),
    ("cy", "威尔士语"),
    ("da", "丹麦语"),
    ("de", "德语"),
    ("et", "爱沙尼亚语"),
    ("en", "英语"),
    ("es", "西班牙语"),
    ("eu", "巴斯克语"),
    ("fo", "法罗语"),
    ("fr", "法语"),
    ("gl", "加利西亚语"),
    ("ha", "豪萨语"),
    ("hr", "克罗地亚语"),
    ("is", "冰岛语"),
    ("it", "意大利语"),
    ("sw", "斯瓦希里语"),
    ("ht", "海地克里奥尔语"),
    ("la", "拉丁语"),
    ("lv", "拉脱维亚语"),
    ("lb", "卢森堡语"),
    ("lt", "立陶宛语"),
    ("ln", "林加拉语"),
    ("hu", "匈牙利语"),
    ("mg", "马达加斯加语"),
    ("mt", "马耳他语"),
    ("mi", "毛利语"),
    ("nl", "荷兰语"),
    ("no", "挪威语（书面）"),
    ("nn", "挪威语（新挪威语）"),
    ("oc", "奥克语"),
    ("haw", "夏威夷语"),
    ("uz", "乌兹别克语"),
    ("pl", "波兰语"),
    ("pt", "葡萄牙语"),
    ("ro", "罗马尼亚语"),
    ("sq", "阿尔巴尼亚语"),
    ("sk", "斯洛伐克语"),
    ("sl", "斯洛文尼亚语"),
    ("so", "索马里语"),
    ("fi", "芬兰语"),
    ("sv", "瑞典语"),
    ("tl", "他加禄语"),
    ("vi", "越南语"),
    ("tr", "土耳其语"),
    ("tk", "土库曼语"),
    ("yo", "约鲁巴语"),
    ("el", "希腊语"),
    ("ba", "巴什基尔语"),
    ("be", "白俄罗斯语"),
    ("bg", "保加利亚语"),
    ("kk", "哈萨克语"),
    ("mk", "马其顿语"),
    ("mn", "蒙古语"),
    ("ru", "俄语"),
    ("sr", "塞尔维亚语"),
    ("tt", "鞑靼语"),
    ("tg", "塔吉克语"),
    ("uk", "乌克兰语"),
    ("hy", "亚美尼亚语"),
    ("yi", "意第绪语"),
    ("he", "希伯来语"),
    ("ur", "乌尔都语"),
    ("ar", "阿拉伯语"),
    ("sd", "信德语"),
    ("fa", "波斯语"),
    ("ps", "普什图语"),
    ("ne", "尼泊尔语"),
    ("mr", "马拉地语"),
    ("sa", "梵语"),
    ("hi", "印地语"),
    ("as", "阿萨姆语"),
    ("bn", "孟加拉语"),
    ("pa", "旁遮普语"),
    ("gu", "古吉拉特语"),
    ("ta", "泰米尔语"),
    ("te", "泰卢固语"),
    ("kn", "卡纳达语"),
    ("ml", "马拉雅拉姆语"),
    ("si", "僧伽罗语"),
    ("th", "泰语"),
    ("lo", "老挝语"),
    ("bo", "藏语"),
    ("my", "缅甸语"),
    ("ka", "格鲁吉亚语"),
    ("ko", "韩语"),
    ("am", "阿姆哈拉语"),
    ("km", "高棉语"),
    ("zh", "中文"),
    ("ja", "日语"),
    ("yue", "粤语"),
];

/// Chinese written in Traditional characters: to whisper it is Chinese,
/// and the transcriber prompts it in the characters asked for.
const TRADITIONAL_CHINESE: (&str, &str) = ("zh-Hant", "繁體中文");

/// The languages the sheet offers after "Auto-detect", by the code the
/// transcriber takes and each by its own name: every language whisper
/// knows, Chinese twice over - in Simplified characters, then in
/// Traditional - since the characters are the captions' and whisper
/// writes either.
fn language_list() -> impl Iterator<Item = (&'static str, &'static str)> {
    concat_speech::transcribe::LANGUAGES
        .iter()
        .flat_map(|&(code, name)| {
            if code == "zh" {
                vec![("zh", "简体中文"), TRADITIONAL_CHINESE]
            } else {
                vec![(code, name)]
            }
        })
}

/// The language list's row for `code`: 0, "Auto-detect", for none and
/// for a code the list does not have.
fn language_row(code: Option<&str>) -> usize {
    code.and_then(|code| language_list().position(|(known, _)| known == code))
        .map_or(0, |index| index + 1)
}

/// The language at row `index` of the list; `None` for "Auto-detect" and
/// for a row past its end.
fn language_at(index: i32) -> Option<String> {
    usize::try_from(index)
        .ok()
        .and_then(|row| row.checked_sub(1))
        .and_then(|row| language_list().nth(row))
        .map(|(code, _)| code.to_owned())
}

/// Imported cues keep the file's timing, including captions shorter than
/// generated speech segments. The title command applies the engine's minimum
/// clip duration where needed.
fn srt_commands(cues: Vec<srt::Cue>, look: (f64, f64), base: Option<&TextStyle>) -> Vec<Command> {
    cues.into_iter()
        .filter(|cue| !cue.text.trim().is_empty())
        .map(|cue| {
            caption_clip(
                cue.text.trim().to_owned(),
                cue.start,
                cue.end - cue.start,
                look,
                base,
            )
        })
        .collect()
}

/// The lane an SRT export writes out.
struct ExportLane {
    track_id: String,
    /// Chosen as the lane with the most titles, not by a selected title.
    busiest: bool,
}

/// The lane whose titles are written out as captions: the selected
/// title's; or with no title selected, the lane with the most titles -
/// captions come many to a lane, a title card one or two, and nothing
/// else in the document tells them apart (see `srt::busiest_title_lane`).
/// None when the timeline has no title.
fn export_lane(studio: &Studio) -> Option<ExportLane> {
    let selected = studio
        .sole_selection()
        .and_then(|id| studio.clip(&id))
        .filter(|clip| clip.kind == ClipKind::Text)
        .map(|clip| clip.track_id.clone());
    match selected {
        Some(track_id) => Some(ExportLane {
            track_id,
            busiest: false,
        }),
        None => srt::busiest_title_lane(studio.timeline()).map(|track_id| ExportLane {
            track_id,
            busiest: true,
        }),
    }
}

/// The name an export is offered under: the project's, as one file name,
/// or "captions" with no project or no usable name.
fn file_stem(studio: &Studio) -> String {
    let name = studio
        .session
        .as_ref()
        .map(|session| session.settings().name)
        .unwrap_or_default();
    let stem: String = name
        .trim()
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    let stem = stem.trim_matches(|c: char| c == '.' || c.is_whitespace());
    if stem.is_empty() {
        "captions".to_owned()
    } else {
        stem.to_owned()
    }
}

/// `name` in `dir`, or `name (2)`, `name (3)`... until nothing there is
/// written over.
fn unique_path(dir: &Path, name: &str) -> PathBuf {
    let first = dir.join(name);
    if !first.exists() {
        return first;
    }
    let (stem, ext) = match name.rfind('.') {
        Some(dot) if dot > 0 => (&name[..dot], &name[dot..]),
        _ => (name, ""),
    };
    (2..)
        .map(|n| dir.join(format!("{stem} ({n}){ext}")))
        .find(|path| !path.exists())
        .unwrap_or(first)
}

/// [`CaptionsPane::subject_clip`] with the studio taken out: `selected`
/// the sole selection, `chosen_with` the selection when `clip` was chosen,
/// and `sounds` whether a clip is on the timeline with sound.
fn follow_selection(
    selected: Option<String>,
    chosen_with: Option<&str>,
    clip: Option<&str>,
    sounds: impl Fn(&str) -> bool,
) -> Option<String> {
    if let Some(selected) = selected.filter(|id| sounds(id))
        && Some(selected.as_str()) != chosen_with
    {
        return Some(selected);
    }
    clip.filter(|id| sounds(id)).map(str::to_owned)
}

/// A clip on the timeline whose voice can be captioned, as the sheet's
/// list shows it.
struct Source {
    id: String,
    label: String,
}

/// The language "Auto captions" asks for when the sheet has none picked:
/// Simplified Chinese under a Simplified Chinese interface, Traditional
/// under a Traditional one (繁體中文, `zh-TW`), else none - whisper's to
/// hear. `saved` is the sheet's remembered pick, which wins.
fn auto_language(interface: &str, saved: Option<&str>) -> Option<String> {
    match saved {
        Some(code) => Some(code.to_owned()),
        None if interface == "zh-Hans" => Some("zh".to_owned()),
        None if matches!(interface, "zh-TW" | "zh-Hant" | "zh-HK") => Some("zh-Hant".to_owned()),
        None => None,
    }
}

/// What a clip's sound is, as far as the timeline can tell without
/// listening. In rank order: a voice before a clip that may be one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Hearing {
    /// Speech, by what was done to it or where it came from.
    Voice,
    /// Nothing says either way: a camera's sound, a plain audio file.
    Plain,
    /// Music, by what was done to it or what it is called.
    Music,
    /// Not heard in the mix at all.
    Silent,
}

/// A clip with sound as [`auto_subjects`] weighs it.
#[derive(Clone, Debug)]
struct Heard {
    id: String,
    start: f64,
    duration: f64,
    volume: f64,
    /// An audio file of its own - not a video, nor sound detached from
    /// one - so possibly a music bed laid under the cut.
    bed: bool,
    hearing: Hearing,
}

impl Heard {
    fn overlap(&self, other: &Heard) -> f64 {
        overlap(
            (self.start, self.start + self.duration),
            (other.start, other.start + other.duration),
        )
    }
}

/// Seconds two spans share.
fn overlap(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.1.min(b.1) - a.0.max(b.0)).max(0.0)
}

/// What a clip's sound is, from what the project already knows about it:
/// silent when muted, at zero or on a muted lane; music when beats were
/// found in it, when its voice was taken out or set apart as singing, or
/// when it or its file is called music ("bgm", "配乐"); a voice when its
/// speech was set apart, when the speech sheet read it aloud, or when it is
/// called one ("voiceover", "旁白"); plain otherwise.
fn hearing(clip: &Clip, media: Option<&MediaItem>, track_muted: bool) -> Hearing {
    if track_muted || clip.muted == Some(true) || clip.volume <= 0.0 {
        return Hearing::Silent;
    }
    if let Some(mix) = clip.voice {
        return match mix {
            mix if mix.voice <= 0.0 && mix.background <= 0.0 => Hearing::Silent,
            mix if mix.voice <= 0.0 || mix.kind == VoiceKind::Vocals => Hearing::Music,
            _ => Hearing::Voice,
        };
    }
    if !clip.beats.is_empty() {
        return Hearing::Music;
    }
    if media.is_some_and(|media| media.origin == Some(MediaOrigin::Speech)) {
        return Hearing::Voice;
    }
    let names = [
        clip.name.as_str(),
        media.map_or("", |media| media.name.as_str()),
    ];
    names
        .iter()
        .find_map(|name| name_hint(name))
        .unwrap_or(Hearing::Plain)
}

/// What a name says a sound is, when it says: a voice's words before
/// music's, so "music_interview" is someone talking.
fn name_hint(name: &str) -> Option<Hearing> {
    const VOICE_WORDS: [&str; 11] = [
        "voice",
        "voiceover",
        "vo",
        "vox",
        "narration",
        "narrator",
        "dialog",
        "dialogue",
        "interview",
        "podcast",
        "speech",
    ];
    const VOICE_CJK: [&str; 11] = [
        "旁白", "配音", "解说", "解說", "人声", "人聲", "口播", "录音", "錄音", "对白", "對白",
    ];
    const MUSIC_WORDS: [&str; 8] = [
        "bgm",
        "music",
        "song",
        "instrumental",
        "soundtrack",
        "ost",
        "beat",
        "beats",
    ];
    const MUSIC_CJK: [&str; 7] = ["配乐", "配樂", "音乐", "音樂", "伴奏", "背景乐", "背景樂"];
    let lower = name.to_lowercase();
    // A word whole, or with a number after it: "bgm2", "vo01".
    let said = |words: &[&str]| {
        lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|token| {
                words.iter().any(|word| {
                    token
                        .strip_prefix(word)
                        .is_some_and(|rest| rest.chars().all(|c| c.is_ascii_digit()))
                })
            })
    };
    if said(&VOICE_WORDS) || VOICE_CJK.iter().any(|word| lower.contains(word)) {
        Some(Hearing::Voice)
    } else if said(&MUSIC_WORDS) || MUSIC_CJK.iter().any(|word| lower.contains(word)) {
        Some(Hearing::Music)
    } else {
        None
    }
}

/// A plain audio file laid under the cut as music: longer than a clip it
/// runs under, covering at least half of it, and either under a voice or
/// quieter than what it is under - background music turned down beneath
/// the talking. A voiceover over quiet pictures is not quieter than them,
/// and stays.
fn is_bed(clip: &Heard, others: &[&Heard]) -> bool {
    clip.bed
        && clip.hearing == Hearing::Plain
        && others.iter().any(|other| {
            other.id != clip.id
                && clip.duration > other.duration
                && clip.overlap(other) >= 0.5 * other.duration
                && (other.hearing == Hearing::Voice || clip.volume < other.volume)
        })
}

/// The clips "Auto captions" transcribes, in timeline order, each with its
/// rank - 0 the likeliest voice - for where two caption the same moment.
/// The selected clip alone when it has sound to hear. Else every clip that
/// may be speaking: voices and plain sound, but not music, a music bed
/// (see [`is_bed`]) or what is silent. Ranked voices first, then a
/// video's own sound before a plain audio file that may be a song laid
/// under it at full level, then the louder, then the longer. Empty when
/// nothing is left.
fn auto_subjects(selected: Option<&str>, heard: &[Heard]) -> Vec<(String, usize)> {
    if let Some(clip) = selected.and_then(|id| {
        heard
            .iter()
            .find(|clip| clip.id == id && clip.hearing != Hearing::Silent)
    }) {
        return vec![(clip.id.clone(), 0)];
    }
    let live: Vec<&Heard> = heard
        .iter()
        .filter(|clip| matches!(clip.hearing, Hearing::Voice | Hearing::Plain))
        .collect();
    let mut chosen: Vec<&Heard> = live
        .iter()
        .copied()
        .filter(|clip| !is_bed(clip, &live))
        .collect();
    let mut ranked = chosen.clone();
    ranked.sort_by(|a, b| {
        a.hearing
            .cmp(&b.hearing)
            .then(a.bed.cmp(&b.bed))
            .then(b.volume.total_cmp(&a.volume))
            .then(b.duration.total_cmp(&a.duration))
            .then(a.start.total_cmp(&b.start))
    });
    chosen.sort_by(|a, b| a.start.total_cmp(&b.start).then_with(|| a.id.cmp(&b.id)));
    chosen
        .iter()
        .map(|clip| {
            let rank = ranked.iter().position(|other| other.id == clip.id);
            (clip.id.clone(), rank.unwrap_or(0))
        })
        .collect()
}

/// Whether "Auto captions", finding nothing to run, found nothing to
/// hear at all - no clip with sound, or every one muted, at zero or on a
/// muted lane - which is nothing to caption, not "only music".
fn nothing_heard(heard: &[Heard]) -> bool {
    heard.iter().all(|clip| clip.hearing == Hearing::Silent)
}

/// A run's captions as one list in timeline order. Where two clips'
/// captions take the same moment - the same words from two microphones,
/// or a voice and the song under it - more than half of the shorter, the
/// better-ranked clip's are kept.
fn merge_lines(mut gathered: Vec<Gathered>) -> Vec<Placed> {
    gathered.sort_by_key(|clip| clip.rank);
    let mut kept: Vec<(usize, Placed)> = Vec::new();
    for (index, clip) in gathered.into_iter().enumerate() {
        for line in clip.lines {
            let span = (line.start, line.start + line.duration);
            let clash = kept.iter().any(|(from, other)| {
                *from != index
                    && overlap(span, (other.start, other.start + other.duration))
                        > 0.5 * line.duration.min(other.duration)
            });
            if !clash {
                kept.push((index, line));
            }
        }
    }
    let mut lines: Vec<Placed> = kept.into_iter().map(|(_, line)| line).collect();
    lines.sort_by(|a, b| a.start.total_cmp(&b.start));
    lines
}

/// Of the captions an earlier run placed (each by id and where it starts),
/// those a new run over `spans` replaces: the ones that start inside them.
fn replaced(earlier: &[(String, f64)], spans: &[(f64, f64)]) -> Vec<String> {
    earlier
        .iter()
        .filter(|(_, start)| {
            spans
                .iter()
                .any(|(from, to)| *start >= *from - 1e-6 && *start < *to)
        })
        .map(|(id, _)| id.clone())
        .collect()
}

/// The model a run asks: the chosen one, unless it hears only English and
/// another language is wanted - then the first installed model that hears
/// more. True besides when every installed model hears only English, which
/// the person is told: whisper's English model writes English for Chinese.
fn hearing_model(installed: &[String], chosen: usize, wants_other: bool) -> (usize, bool) {
    let english_only = |index: usize| {
        installed
            .get(index)
            .is_some_and(|id| concat_speech::transcribe::english_only(id))
    };
    if !wants_other || !english_only(chosen) {
        return (chosen, false);
    }
    match (0..installed.len()).find(|index| !english_only(*index)) {
        Some(index) => (index, false),
        None => (chosen, true),
    }
}

/// The clips on the current timeline that have sound, in the order they
/// play: an audio clip, or a video whose file has an audio stream.
fn sources(studio: &Studio) -> Vec<Source> {
    let project = studio.project();
    let mut clips: Vec<_> = studio
        .timeline()
        .clips
        .iter()
        .filter(|clip| studio.clip_has_sound(clip))
        .map(|clip| {
            let name = project
                .media_by_id(&clip.media_id)
                .map_or(clip.name.as_str(), |media| media.name.as_str());
            (clip.start, clip.id.clone(), name.to_owned())
        })
        .collect();
    clips.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    clips
        .into_iter()
        .map(|(start, id, name)| Source {
            id,
            label: source_label(&name, start),
        })
        .collect()
}

/// A clip as the list names it: its file and where it starts on the
/// timeline, so two clips of one file are told apart.
fn source_label(name: &str, start: f64) -> String {
    let seconds = start.max(0.0).floor() as u64;
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if hours > 0 {
        format!("{name} · {hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{name} · {minutes}:{seconds:02}")
    }
}

/// One caption as a title clip: its words, when and for how long, and
/// its look.
fn caption_clip(
    text: String,
    start: f64,
    duration: f64,
    look: (f64, f64),
    base: Option<&TextStyle>,
) -> Command {
    let (offset_y, font_size) = look;
    // The kept look where there is one - family, weight, colours, stroke,
    // shadow, background - at the caption's own size and place; the
    // bundled face otherwise.
    let style = match base {
        Some(base) => TextStyle {
            content: text,
            font_size,
            max_width: 0.0,
            max_height: 0.0,
            // Runs are some words' looks, and these are other words.
            runs: Vec::new(),
            ..base.clone()
        },
        None => TextStyle {
            content: text,
            font_family: "Hanken Grotesk".to_owned(),
            font_size,
            font_weight: 600.0,
            ..TextStyle::default()
        },
    };
    Command::AddTextClip {
        track_id: None,
        above: true,
        start,
        style: Some(style),
        duration: Some(duration),
        offset_y: Some(offset_y),
    }
}

/// The look captions are set in, before their own size and place: the
/// text preset picked on the sheet (`Preferences::caption_style`), with
/// the font it brings registered on the project in the same undo step;
/// or, for the plain preset or none, the look a title was kept as.
fn caption_base(studio: &Studio) -> (Option<TextStyle>, Option<Command>) {
    let picked = studio
        .prefs
        .caption_style
        .as_deref()
        .filter(|id| *id != "default")
        .and_then(|id| studio.text_presets.iter().find(|preset| preset.id == id));
    match picked {
        Some(preset) => (
            Some(preset.style.clone()),
            crate::presets::install_font(&studio.host.dirs, preset)
                .map(|(family, path)| Command::AddFont { family, path }),
        ),
        None => (studio.prefs.title_style.clone(), None),
    }
}

/// The preset the sheet shows as picked: the one remembered, while it is
/// still on the list, else the plain one.
pub fn caption_style_id(studio: &Studio) -> &str {
    studio
        .prefs
        .caption_style
        .as_deref()
        .filter(|id| studio.text_presets.iter().any(|preset| preset.id == *id))
        .unwrap_or("default")
}

/// Longest a caption line gets before it is wrapped: about what two lines
/// of broadcast subtitle hold, and what a reader takes in at a glance.
const CAPTION_CHARS: usize = 42;

/// A script as caption lines, each with how long it stays up: a line's
/// reading time at [`CHARS_PER_SECOND`], held to one second at least so
/// a short word is not a flicker, and seven at most so a long line does
/// not hang. A line break in the script is a break the author asked for;
/// within a paragraph a sentence is a caption, and a long sentence wraps
/// at its words.
fn script_captions(text: &str) -> Vec<(String, f64)> {
    text.lines()
        .flat_map(sentences)
        .flat_map(|sentence| wrap_caption(&sentence))
        .map(|line| {
            let seconds =
                (line.chars().count() as f64 / f64::from(CHARS_PER_SECOND)).clamp(1.0, 7.0);
            (line, seconds)
        })
        .collect()
}

/// A paragraph's sentences. A full stop, question or exclamation mark ends
/// one when it is followed by space or by the end - so "3.5" and "e.g." hold
/// together - and the CJK marks end one on their own.
fn sentences(paragraph: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = paragraph.chars().peekable();
    while let Some(ch) = chars.next() {
        current.push(ch);
        let ends = match ch {
            '。' | '！' | '？' => true,
            '.' | '!' | '?' => chars.peek().is_none_or(|next| next.is_whitespace()),
            _ => false,
        };
        if ends {
            let sentence = current.trim();
            if !sentence.is_empty() {
                out.push(sentence.to_owned());
            }
            current.clear();
        }
    }
    let rest = current.trim();
    if !rest.is_empty() {
        out.push(rest.to_owned());
    }
    out
}

/// A sentence in lines of at most [`CAPTION_CHARS`], broken between words;
/// a word longer than a line, or a run of CJK with no spaces, is broken
/// where it must be.
fn wrap_caption(sentence: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut line_chars = 0;
    for word in sentence.split_whitespace() {
        let word_chars = word.chars().count();
        if line_chars > 0 && line_chars + 1 + word_chars > CAPTION_CHARS {
            lines.push(std::mem::take(&mut line));
            line_chars = 0;
        }
        if word_chars > CAPTION_CHARS {
            let mut piece = String::new();
            for ch in word.chars() {
                piece.push(ch);
                if piece.chars().count() == CAPTION_CHARS {
                    lines.push(std::mem::take(&mut piece));
                }
            }
            line = piece;
            line_chars = line.chars().count();
            continue;
        }
        if line_chars > 0 {
            line.push(' ');
            line_chars += 1;
        }
        line.push_str(word);
        line_chars += word_chars;
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::{script_captions, source_label};
    use concat_project::caption_lines::LineRules;
    use concat_speech::transcribe::{Piece, Segment};

    /// A transcription is cut at its punctuation and timed by its tokens;
    /// with a script, the script's words are the captions.
    #[test]
    fn a_transcription_becomes_lines_and_a_script_takes_its_timing() {
        let piece = |text: &str, start: f64| Piece {
            text: text.to_owned(),
            start,
            end: start + 1.0,
        };
        let segments = [Segment {
            start: 0.0,
            end: 6.0,
            text: "大家好，欢迎来到我的平道。".to_owned(),
            pieces: vec![
                piece("大家好", 0.0),
                piece("，", 1.0),
                piece("欢迎来到", 2.0),
                piece("我的", 3.0),
                piece("平道", 4.0),
                piece("。", 5.0),
            ],
        }];
        let rules = LineRules {
            max_width: 10.0,
            ..LineRules::default()
        };
        let heard = super::caption_lines_of(&segments, None, &rules);
        let texts: Vec<&str> = heard.iter().map(|line| line.text.as_str()).collect();
        assert_eq!(texts, ["大家好", "欢迎来到我的平道"]);
        assert_eq!((heard[0].start, heard[0].end), (0.0, 1.0));
        assert_eq!((heard[1].start, heard[1].end), (2.0, 5.0));
        let matched =
            super::caption_lines_of(&segments, Some("大家好！欢迎来到我的频道。"), &rules);
        let texts: Vec<&str> = matched.iter().map(|line| line.text.as_str()).collect();
        assert_eq!(texts, ["大家好！", "欢迎来到我的频道"]);
        // A script that has nothing to do with the speech: the words heard.
        let unrelated = super::caption_lines_of(&segments, Some("。"), &rules);
        assert_eq!(unrelated.len(), 2);
    }

    #[test]
    fn imported_srt_preserves_short_cues_and_multiline_text() {
        let cues = concat_project::srt::parse(
            "1\n00:00:01,000 --> 00:00:01,100\nOlá\nAção\n\n2\n00:00:02,000 --> 00:00:03,000\nFim\n",
        ).unwrap();
        let commands = super::srt_commands(cues.clone(), (0.35, 0.04), None);
        let mut editor = concat_project::Editor::new();
        editor
            .apply(concat_project::Command::Batch { commands })
            .unwrap();
        let exported = concat_project::srt::write(&concat_project::srt::title_cues(
            editor.project().active(),
            None,
        ));
        assert_eq!(concat_project::srt::parse(&exported).unwrap(), cues);
        assert!(editor.undo());
        assert!(editor.project().active().clips.is_empty());
        assert!(editor.redo());
        assert_eq!(
            concat_project::srt::title_cues(editor.project().active(), None).len(),
            2
        );
    }

    /// The panel floats over a live editor: a clip with sound selected
    /// since the panel's clip was chosen is the one it shows and runs; a
    /// pick from its list holds until the selection moves again; and a
    /// chosen clip gone from the timeline is not run unseen.
    #[test]
    fn the_captioned_clip_follows_a_new_selection_of_a_clip_with_sound() {
        let sounds = |id: &str| ["a", "b", "c"].contains(&id);
        let follow = |selected: Option<&str>, chosen_with: Option<&str>, clip: Option<&str>| {
            super::follow_selection(selected.map(str::to_owned), chosen_with, clip, sounds)
        };
        // Opened on A, A still selected: A.
        assert_eq!(
            follow(Some("a"), Some("a"), Some("a")).as_deref(),
            Some("a")
        );
        // B selected on the timeline since: B.
        assert_eq!(
            follow(Some("b"), Some("a"), Some("a")).as_deref(),
            Some("b")
        );
        // C picked from the list with B selected: C, until the selection moves.
        assert_eq!(
            follow(Some("b"), Some("b"), Some("c")).as_deref(),
            Some("c")
        );
        // A title selected - no sound - leaves the choice alone.
        assert_eq!(
            follow(Some("t"), Some("b"), Some("c")).as_deref(),
            Some("c")
        );
        // The chosen clip deleted: nothing to run, rather than a clip unseen.
        assert_eq!(follow(None, None, Some("gone")), None);
    }

    /// A clip with sound as auto captions weighs it.
    fn heard(id: &str, start: f64, duration: f64, hearing: super::Hearing) -> super::Heard {
        super::Heard {
            id: id.to_owned(),
            start,
            duration,
            volume: 1.0,
            bed: false,
            hearing,
        }
    }

    fn ids(subjects: &[(String, usize)]) -> Vec<&str> {
        subjects.iter().map(|(id, _)| id.as_str()).collect()
    }

    /// "Auto captions" takes the selected clip when it has sound - even
    /// music, the person's own pick - else every clip that may be speaking,
    /// in timeline order; nothing when nothing can be heard.
    #[test]
    fn auto_captions_takes_the_selected_voice_else_every_voice_in_order() {
        use super::Hearing::{Music, Plain, Silent, Voice};
        let clips = [
            heard("c", 40.0, 12.0, Plain),
            heard("a", 0.0, 4.0, Voice),
            heard("b", 10.0, 30.0, Plain),
            heard("m", 0.0, 90.0, Music),
            heard("s", 60.0, 9.0, Silent),
        ];
        let all = super::auto_subjects(None, &clips);
        assert_eq!(ids(&all), ["a", "b", "c"]);
        // Voices rank first, then the longer.
        let rank = |id: &str| all.iter().find(|(of, _)| of == id).unwrap().1;
        assert_eq!((rank("a"), rank("b"), rank("c")), (0, 1, 2));
        assert_eq!(ids(&super::auto_subjects(Some("c"), &clips)), ["c"]);
        assert_eq!(ids(&super::auto_subjects(Some("m"), &clips)), ["m"]);
        // A title selected, or a silent clip, is passed over.
        assert_eq!(
            ids(&super::auto_subjects(Some("t"), &clips)),
            ["a", "b", "c"]
        );
        assert_eq!(
            ids(&super::auto_subjects(Some("s"), &clips)),
            ["a", "b", "c"]
        );
        assert!(super::auto_subjects(None, &[]).is_empty());
        let music_only = [heard("m", 0.0, 90.0, Music), heard("s", 0.0, 9.0, Silent)];
        assert!(super::auto_subjects(None, &music_only).is_empty());
        assert!(!super::nothing_heard(&music_only));
        // Every clip muted: nothing to caption, not "only music".
        let muted = [heard("a", 0.0, 9.0, Silent), heard("b", 9.0, 9.0, Silent)];
        assert!(super::auto_subjects(None, &muted).is_empty());
        assert!(super::nothing_heard(&muted));
        assert!(super::nothing_heard(&[]));
    }

    /// Background music longer than the voice it runs under is not
    /// captioned - turned down under a talking head, or under anything
    /// called a voice - while a voiceover over quiet pictures is.
    #[test]
    fn auto_captions_leaves_out_a_music_bed_longer_than_the_voice() {
        use super::Hearing::{Plain, Voice};
        let bed = |volume: f64| super::Heard {
            volume,
            bed: true,
            ..heard("bgm", 0.0, 120.0, Plain)
        };
        // A talking head, 20 s, with an unnamed song at a third under it.
        let talking = [heard("v", 5.0, 20.0, Plain), bed(0.3)];
        assert_eq!(ids(&super::auto_subjects(None, &talking)), ["v"]);
        // At full level, under a clip known to be speech.
        let named = [heard("v", 5.0, 20.0, Voice), bed(1.0)];
        assert_eq!(ids(&super::auto_subjects(None, &named)), ["v"]);
        // Two plain audio files, the longer one quieter: the music.
        let two = [
            super::Heard {
                bed: true,
                ..heard("narration", 0.0, 50.0, Plain)
            },
            bed(0.5),
        ];
        assert_eq!(ids(&super::auto_subjects(None, &two)), ["narration"]);
        // A voiceover at full level over quieter b-roll is the voice: kept.
        let voiceover = super::Heard {
            bed: true,
            ..heard("vo", 0.0, 60.0, Plain)
        };
        let broll = super::Heard {
            volume: 0.4,
            ..heard("broll", 10.0, 5.0, Plain)
        };
        assert_eq!(
            ids(&super::auto_subjects(None, &[voiceover, broll])),
            ["vo", "broll"]
        );
    }

    /// An unnamed song at full level, longer than the camera clip with
    /// sound it runs under, is not a bed - but the clip's own sound still
    /// ranks first, and its words are kept where the two clash.
    #[test]
    fn a_full_level_song_never_outranks_the_camera_clip_it_runs_under() {
        use super::Hearing::Plain;
        let song = super::Heard {
            bed: true,
            ..heard("bgm", 0.0, 120.0, Plain)
        };
        let clips = [heard("v", 5.0, 20.0, Plain), song];
        let subjects = super::auto_subjects(None, &clips);
        let rank = |id: &str| subjects.iter().find(|(of, _)| of == id).map(|s| s.1);
        assert_eq!(rank("v"), Some(0));
        assert_eq!(rank("bgm"), Some(1));
        let line = |text: &str, start: f64| super::Placed {
            text: text.to_owned(),
            start,
            duration: 2.0,
        };
        let gathered = |id: &str, lines: Vec<super::Placed>| super::Gathered {
            rank: rank(id).unwrap(),
            span: (0.0, 120.0),
            look: (0.35, 0.05),
            lines,
        };
        let merged = super::merge_lines(vec![
            gathered("bgm", vec![line("la la", 6.0), line("na na", 60.0)]),
            gathered("v", vec![line("hello", 6.5), line("bye", 12.0)]),
        ]);
        let texts: Vec<&str> = merged.iter().map(|line| line.text.as_str()).collect();
        assert_eq!(texts, ["hello", "bye", "na na"]);
    }

    /// What the timeline knows of a clip's sound says whether it is a
    /// voice, music or nothing heard.
    #[test]
    fn a_clip_is_heard_as_voice_music_or_silence_by_what_the_project_knows() {
        use super::Hearing::{Music, Plain, Silent, Voice};
        use concat_project::model::{Clip, ClipKind, MediaItem, MediaOrigin, VoiceKind, VoiceMix};
        let audio = |name: &str| Clip::blank("c1", "t1", ClipKind::Audio, name, 0.0, 10.0);
        let file = |name: &str| MediaItem {
            name: name.to_owned(),
            ..MediaItem::default()
        };
        let heard = |clip: &Clip, media: &MediaItem| super::hearing(clip, Some(media), false);
        let plain = audio("take 3");
        assert_eq!(heard(&plain, &file("take 3.m4a")), Plain);
        assert_eq!(super::hearing(&plain, None, true), Silent);
        assert_eq!(
            heard(
                &Clip {
                    volume: 0.0,
                    ..plain.clone()
                },
                &file("a")
            ),
            Silent
        );
        assert_eq!(
            heard(
                &Clip {
                    muted: Some(true),
                    ..plain.clone()
                },
                &file("a")
            ),
            Silent
        );
        assert_eq!(heard(&plain, &file("BGM_02.mp3")), Music);
        assert_eq!(heard(&plain, &file("summer-song.wav")), Music);
        assert_eq!(heard(&plain, &file("背景音乐.mp3")), Music);
        assert_eq!(heard(&plain, &file("vo01.wav")), Voice);
        assert_eq!(heard(&plain, &file("旁白 第二段.m4a")), Voice);
        assert_eq!(heard(&plain, &file("music_interview.mp3")), Voice);
        // Not a word of its own: "Beethoven", "ghost", "volume test".
        assert_eq!(heard(&plain, &file("Beethoven ghost volume.wav")), Plain);
        let beats = Clip {
            beats: vec![0.5, 1.0],
            ..plain.clone()
        };
        assert_eq!(heard(&beats, &file("take 3")), Music);
        let read_aloud = MediaItem {
            origin: Some(MediaOrigin::Speech),
            ..file("tts.wav")
        };
        assert_eq!(heard(&plain, &read_aloud), Voice);
        let mix = |voice: f64, background: f64, kind: VoiceKind| Clip {
            voice: Some(VoiceMix {
                voice,
                background,
                kind,
            }),
            ..plain.clone()
        };
        assert_eq!(
            heard(&mix(1.0, 0.2, VoiceKind::Speech), &file("bgm")),
            Voice
        );
        assert_eq!(heard(&mix(0.0, 1.0, VoiceKind::Speech), &file("a")), Music);
        assert_eq!(heard(&mix(1.0, 0.0, VoiceKind::Vocals), &file("a")), Music);
        assert_eq!(heard(&mix(0.0, 0.0, VoiceKind::Speech), &file("a")), Silent);
    }

    /// Where two clips caption the same moment the better-ranked clip's
    /// words are kept; the rest of both are, in timeline order.
    #[test]
    fn overlapping_captions_from_two_clips_keep_the_likelier_voice() {
        let line = |text: &str, start: f64, duration: f64| super::Placed {
            text: text.to_owned(),
            start,
            duration,
        };
        let clip = |rank: usize, lines: Vec<super::Placed>| super::Gathered {
            rank,
            span: (0.0, 100.0),
            look: (0.35, 0.05),
            lines,
        };
        let song = clip(1, vec![line("la la", 1.0, 2.0), line("na na", 40.0, 2.0)]);
        let voice = clip(0, vec![line("hello", 1.5, 2.0), line("bye", 10.0, 1.0)]);
        let merged = super::merge_lines(vec![song, voice]);
        let texts: Vec<&str> = merged.iter().map(|line| line.text.as_str()).collect();
        assert_eq!(texts, ["hello", "bye", "na na"]);
        // One clip's own lines never push each other out.
        let tight = clip(0, vec![line("a", 0.0, 2.0), line("b", 0.5, 2.0)]);
        assert_eq!(super::merge_lines(vec![tight]).len(), 2);
    }

    /// Pressing again replaces the captions the last press placed over the
    /// same clips, and leaves those it placed over other clips.
    #[test]
    fn a_second_press_replaces_the_first_set_instead_of_stacking() {
        let earlier = [
            ("c1".to_owned(), 0.0),
            ("c2".to_owned(), 4.5),
            ("c3".to_owned(), 30.0),
        ];
        assert_eq!(super::replaced(&earlier, &[(0.0, 10.0)]), ["c1", "c2"]);
        assert_eq!(
            super::replaced(&earlier, &[(0.0, 10.0), (25.0, 40.0)]),
            ["c1", "c2", "c3"]
        );
        assert!(super::replaced(&earlier, &[(50.0, 60.0)]).is_empty());
        assert!(super::replaced(&[], &[(0.0, 10.0)]).is_empty());
    }

    /// An English-only model is passed over for one that hears Chinese,
    /// and said when it is all there is.
    #[test]
    fn an_english_only_model_is_not_used_for_chinese() {
        let models =
            |ids: &[&str]| -> Vec<String> { ids.iter().map(|id| (*id).to_owned()).collect() };
        assert_eq!(
            super::hearing_model(&models(&["base.en", "small"]), 0, true),
            (1, false)
        );
        assert_eq!(
            super::hearing_model(&models(&["base.en"]), 0, true),
            (0, true)
        );
        assert_eq!(
            super::hearing_model(&models(&["base.en", "small"]), 0, false),
            (0, false)
        );
        assert_eq!(
            super::hearing_model(&models(&["small", "base.en"]), 0, true),
            (0, false)
        );
        assert_eq!(super::hearing_model(&[], 0, true), (0, false));
    }

    /// The language asked for: the sheet's pick, else Simplified Chinese
    /// under the Simplified interface, else whisper's to hear.
    #[test]
    fn auto_captions_asks_for_simplified_under_a_chinese_interface() {
        assert_eq!(super::auto_language("zh-Hans", None).as_deref(), Some("zh"));
        assert_eq!(super::auto_language("en", None), None);
        assert_eq!(super::auto_language("ja", None), None);
        assert_eq!(
            super::auto_language("zh-Hans", Some("en")).as_deref(),
            Some("en")
        );
        assert_eq!(
            super::auto_language("en", Some("zh-Hant")).as_deref(),
            Some("zh-Hant")
        );
        // What it asks for is a row the sheet's list has.
        assert!(super::language_row(super::auto_language("zh-Hans", None).as_deref()) > 0);
    }

    /// Under 繁體中文 the captions come out in Traditional characters,
    /// unless the sheet's pick says otherwise.
    #[test]
    fn auto_captions_asks_for_traditional_under_a_traditional_interface() {
        assert_eq!(
            super::auto_language("zh-TW", None).as_deref(),
            Some("zh-Hant")
        );
        assert_eq!(
            super::auto_language("zh-TW", Some("zh")).as_deref(),
            Some("zh")
        );
        let traditional = super::auto_language("zh-TW", None);
        assert_eq!(
            super::language_row(traditional.as_deref()),
            super::language_row(Some("zh-Hant"))
        );
        assert!(super::language_row(traditional.as_deref()) > 0);
    }

    /// Row 0 is "Auto-detect", and a language's row reads back as the
    /// same language; a code the list lacks, or a row past it, is heard.
    #[test]
    fn a_language_row_reads_back_as_its_code() {
        assert_eq!(super::language_row(None), 0);
        assert_eq!(super::language_row(Some("klingon")), 0);
        assert_eq!(super::language_at(0), None);
        assert_eq!(super::language_at(-1), None);
        assert_eq!(super::language_at(100_000), None);
        let row = super::language_row(Some("id"));
        assert!(row > 0);
        assert_eq!(super::language_at(row as i32).as_deref(), Some("id"));
        assert_eq!(
            super::language_list().nth(row - 1).map(|(_, name)| name),
            Some("Bahasa Indonesia")
        );
        // Chinese is offered in both its characters, one row after the
        // other, and each reads back as itself.
        let simplified = super::language_row(Some("zh"));
        let traditional = super::language_row(Some("zh-Hant"));
        assert!(simplified > 0);
        assert_eq!(traditional, simplified + 1);
        assert_eq!(
            super::language_at(traditional as i32).as_deref(),
            Some("zh-Hant")
        );
        assert_eq!(super::request_language(Some("zh-Hant")), Some("zh-Hant"));
        assert_eq!(super::request_language(Some("zh")), Some("zh"));
    }

    /// "Auto-detect" asks for no language whatever the interface is in:
    /// a Traditional Chinese interface no longer turns it into Traditional
    /// characters, and the Simplified one does not turn it into anything.
    #[test]
    fn auto_detect_hears_the_language_under_any_interface() {
        let _language = crate::i18n::language_lock();
        let root =
            std::env::temp_dir().join(format!("concat-captions-auto-{}", std::process::id()));
        let dirs = concat_host::AppDirs::under(&root);
        for code in ["zh-TW", "zh-Hans", "en"] {
            crate::i18n::select(code, &dirs);
            assert_eq!(super::request_language(None), None, "under {code}");
        }
        crate::i18n::select(crate::i18n::ENGLISH, &dirs);
    }

    /// Under a Simplified Chinese interface every language of the list
    /// reads by its Chinese name, and the two Chinese rows by their own.
    #[test]
    fn the_language_list_reads_in_chinese_under_a_chinese_interface() {
        for (code, own) in super::language_list() {
            let shown = super::shown_name(code, own, "zh-Hans");
            assert!(
                shown
                    .chars()
                    .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)),
                "{code} reads {shown:?}"
            );
            assert_eq!(super::shown_name(code, own, "en"), own);
        }
        assert_eq!(super::shown_name("en", "English", "zh-Hans"), "英语");
        assert_eq!(
            super::shown_name("zh-Hant", "繁體中文", "zh-Hans"),
            "繁體中文"
        );
        assert_eq!(super::shown_name("zh", "简体中文", "zh-Hans"), "简体中文");
    }

    #[test]
    fn imported_srt_ignores_empty_captions() {
        let cues = vec![concat_project::srt::Cue {
            start: 0.0,
            end: 1.0,
            text: "  ".into(),
        }];
        assert!(super::srt_commands(cues, (0.35, 0.04), None).is_empty());
    }

    /// A clip in the voice list is its file and where it starts, in
    /// minutes and seconds, with hours once there are any.
    #[test]
    fn a_voice_is_named_by_its_file_and_where_it_starts() {
        assert_eq!(source_label("Interview.mp4", 0.0), "Interview.mp4 · 0:00");
        assert_eq!(source_label("Interview.mp4", 72.9), "Interview.mp4 · 1:12");
        assert_eq!(source_label("Talk.wav", 3_725.0), "Talk.wav · 1:02:05");
        assert_eq!(source_label("Talk.wav", -1.0), "Talk.wav · 0:00");
    }

    /// A script becomes one caption per sentence, a hand line break is
    /// kept, a long sentence wraps at its words, and each line is held for
    /// its reading time within one to seven seconds.
    #[test]
    fn a_script_is_cut_into_readable_lines() {
        let lines = script_captions(
            "Hello there. This is version 3.5, mind!\n\nA sentence that runs on for far \
             longer than a caption line has any business running on for. Ok?",
        );
        let text: Vec<&str> = lines.iter().map(|(line, _)| line.as_str()).collect();
        assert_eq!(
            text,
            [
                "Hello there.",
                "This is version 3.5, mind!",
                "A sentence that runs on for far longer",
                "than a caption line has any business",
                "running on for.",
                "Ok?",
            ]
        );
        assert!(
            lines
                .iter()
                .all(|(_, seconds)| (1.0..=7.0).contains(seconds))
        );
        assert_eq!(lines[0].1, 1.0);
        assert!(lines[2].1 > lines[0].1);
        assert!(script_captions("  \n ").is_empty());
        assert_eq!(script_captions("你好。再见！").len(), 2);
    }
}
