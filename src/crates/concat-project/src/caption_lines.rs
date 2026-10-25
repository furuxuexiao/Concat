// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! Caption lines: what was said, cut into lines a reader takes in at a
//! glance, each timed to the words it holds.
//!
//! Whisper hands back phrases as it heard them, and in Chinese a phrase
//! can run on for forty characters with its commas inside it - far more
//! than a phone's width of screen. CapCut's captions (剪映's 智能字幕) are
//! cut at the punctuation instead: a clause is a line, short clauses of
//! one sentence share one, a clause longer than a line is split in even
//! halves rather than a full line and a scrap, and the line's closing
//! comma or full stop is dropped, since the line break already says it.
//!
//! [`lines`] does that cut. Its input is phrases of timed spans - a
//! whisper segment and its tokens, say - and every character is timed
//! within the span it came in, so a line split out of a phrase starts and
//! ends where its own words do rather than where the phrase did. A phrase
//! ends a line whatever its punctuation; so does a pause in the speech.
//!
//! [`align_script`] is the other half of CapCut's 文稿匹配: a script's
//! words timed by a transcript of the same speech, so the captions read
//! as written - names spelled right, punctuation as the author put it -
//! and come up when they are said.
//!
//! Widths are counted in full-width columns: a Chinese, Japanese or Korean
//! character is one, a Latin letter, digit or space half of one. That way
//! one limit fits both, and 24 columns is the 48 characters an English
//! subtitle line has always been allowed.

/// Words and when they were said, in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct Timed {
    /// The words, as heard or written.
    pub text: String,
    /// When the first of them begins.
    pub start: f64,
    /// When the last of them ends; never before `start`.
    pub end: f64,
}

impl Timed {
    /// `text` said between `start` and `end`.
    pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            text: text.into(),
            start,
            end: end.max(start),
        }
    }
}

/// How lines are cut.
#[derive(Clone, Debug, PartialEq)]
pub struct LineRules {
    /// The widest a line may be, in full-width columns (see the module
    /// notes). A clause wider than this is split into even parts.
    pub max_width: f64,
    /// Drop the comma, full stop or colon a line ends on, as CapCut's
    /// captions do. A question or exclamation mark stays: it says
    /// something the line break does not.
    pub strip_trailing: bool,
    /// A silence this long, in seconds, between two characters ends the
    /// line, whatever the punctuation says.
    pub pause: f64,
}

impl LineRules {
    /// The line for a tall frame, a phone's: sixteen characters.
    pub const VERTICAL_WIDTH: f64 = 16.0;
    /// The line for a wide frame: twenty-four characters.
    pub const HORIZONTAL_WIDTH: f64 = 24.0;

    /// The width that suits a frame of `width` × `height`: a tall frame
    /// gets [`Self::VERTICAL_WIDTH`], any other [`Self::HORIZONTAL_WIDTH`].
    pub fn width_for_frame(width: u32, height: u32) -> f64 {
        if height > width {
            Self::VERTICAL_WIDTH
        } else {
            Self::HORIZONTAL_WIDTH
        }
    }
}

impl Default for LineRules {
    fn default() -> Self {
        Self {
            max_width: Self::HORIZONTAL_WIDTH,
            strip_trailing: true,
            pause: 0.8,
        }
    }
}

/// One caption line, and when it is on screen.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// The words.
    pub text: String,
    /// When the first word begins, in the phrases' seconds.
    pub start: f64,
    /// When the last word ends.
    pub end: f64,
}

/// One character and its share of the span it came in.
#[derive(Clone, Copy, Debug)]
struct Glyph {
    ch: char,
    start: f64,
    end: f64,
    /// The last character of a phrase: a line ends here.
    last_of_phrase: bool,
}

/// How firmly a clause is closed: a sentence's end, or a pause inside one.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Close {
    /// A comma, enumeration comma, semicolon or colon: the next clause of
    /// the same sentence may share the line.
    Soft,
    /// A full stop, question or exclamation mark, the end of a phrase or
    /// a pause in the speech: the next clause starts a line.
    Hard,
}

/// The phrases' words cut into caption lines by `rules`. Each phrase is
/// its spans in order; a phrase of one span is timed evenly across it.
/// A line of nothing but punctuation and space is no line.
pub fn lines(phrases: &[Vec<Timed>], rules: &LineRules) -> Vec<Line> {
    let glyphs = glyphs(phrases);
    let max_width = rules.max_width.max(1.0);
    let mut out = Vec::new();
    // The clauses waiting to share a line, and how the last one closed.
    let mut pending: Vec<Glyph> = Vec::new();
    let mut pending_close = Close::Hard;
    for (clause, close) in clauses(&glyphs, rules.pause) {
        let joined = width(&pending) + width(clause);
        if !pending.is_empty() && (pending_close == Close::Hard || joined > max_width) {
            emit(&mut out, &pending, max_width, rules);
            pending.clear();
        }
        pending.extend_from_slice(clause);
        pending_close = close;
    }
    emit(&mut out, &pending, max_width, rules);
    out
}

/// A script's words, timed by a transcript of the same speech: each
/// written character takes the time of the heard one it lines up with, by
/// the fewest edits between the two, letters and digits compared without
/// case and everything else left out of the comparison. A written word
/// the speech has no match for is spread between its neighbours that do.
///
/// One phrase per line of the script, as [`lines`] takes them: a line
/// break in the script always breaks the captions. Empty when nothing was
/// heard, or when the script has no words.
///
/// A script and a transcript too long to compare whole - beyond some six
/// thousand characters each - are lined up by their proportions instead,
/// which holds as long as the speech keeps an even pace.
pub fn align_script(script: &str, heard: &[Vec<Timed>]) -> Vec<Vec<Timed>> {
    let heard: Vec<Glyph> = glyphs(heard)
        .into_iter()
        .filter(|glyph| compared(glyph.ch))
        .collect();
    let written: Vec<char> = script.chars().collect();
    let keys: Vec<usize> = (0..written.len())
        .filter(|&index| compared(written[index]))
        .collect();
    if heard.is_empty() || keys.is_empty() {
        return Vec::new();
    }

    // Each compared written character's time, where it lines up with one
    // heard; None where it lines up with nothing.
    let mut times: Vec<Option<(f64, f64)>> = vec![None; keys.len()];
    let fits = keys.len().saturating_mul(heard.len()) <= ALIGN_CELLS;
    if fits {
        let a: Vec<char> = keys.iter().map(|&index| fold(written[index])).collect();
        let b: Vec<char> = heard.iter().map(|glyph| fold(glyph.ch)).collect();
        for (i, j) in edit_pairs(&a, &b) {
            times[i] = Some((heard[j].start, heard[j].end));
        }
    } else {
        let (first, last) = (heard[0].start, heard[heard.len() - 1].end);
        let step = (last - first) / keys.len() as f64;
        for (i, time) in times.iter_mut().enumerate() {
            let start = first + step * i as f64;
            *time = Some((start, start + step));
        }
    }
    fill_gaps(&mut times, heard[0].start, heard[heard.len() - 1].end);

    // Every written character with a time: a compared one its own, the
    // punctuation and spaces around it the edge of the word they follow
    // (or, before the first word, the one they lead into).
    let mut timed = vec![(0.0, 0.0); written.len()];
    let mut key = 0;
    let mut previous: Option<f64> = None;
    for (index, slot) in timed.iter_mut().enumerate() {
        if key < keys.len() && keys[key] == index {
            let time = times[key].expect("every gap filled");
            *slot = time;
            previous = Some(time.1);
            key += 1;
        } else {
            let at = previous.unwrap_or_else(|| times[0].expect("every gap filled").0);
            *slot = (at, at);
        }
    }

    let mut phrases = Vec::new();
    let mut phrase = Vec::new();
    for (ch, (start, end)) in written.into_iter().zip(timed) {
        if ch == '\n' {
            if !phrase.is_empty() {
                phrases.push(std::mem::take(&mut phrase));
            }
            continue;
        }
        phrase.push(Timed::new(ch.to_string(), start, end));
    }
    if !phrase.is_empty() {
        phrases.push(phrase);
    }
    phrases.retain(|phrase| phrase.iter().any(|span| span.text.chars().any(compared)));
    phrases
}

/// A script file's words: UTF-8 with or without its byte-order mark, or
/// UTF-16 with one, as Windows' Notepad writes it. None for anything
/// else (a GBK file from an older Chinese Windows, say) rather than a
/// script of replacement characters; the person is asked to save it as
/// UTF-8.
pub fn decode_script(bytes: &[u8]) -> Option<String> {
    let utf16 = |bytes: &[u8], word: fn([u8; 2]) -> u16| {
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|pair| word([pair[0], pair[1]]))
            .collect();
        String::from_utf16(&units).ok()
    };
    let text = match bytes {
        [0xEF, 0xBB, 0xBF, rest @ ..] => String::from_utf8(rest.to_vec()).ok(),
        [0xFF, 0xFE, rest @ ..] => utf16(rest, u16::from_le_bytes),
        [0xFE, 0xFF, rest @ ..] => utf16(rest, u16::from_be_bytes),
        _ => String::from_utf8(bytes.to_vec()).ok(),
    }?;
    Some(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// The most cells the alignment's table may have: a byte each, so a few
/// tens of megabytes at the most, for a script of six thousand characters
/// against a transcript as long. Beyond that, proportions; see
/// [`align_script`].
const ALIGN_CELLS: usize = 36_000_000;

/// How wide `text` reads, in full-width columns.
pub fn text_width(text: &str) -> f64 {
    text.chars().map(char_width).sum()
}

fn width(glyphs: &[Glyph]) -> f64 {
    glyphs.iter().map(|glyph| char_width(glyph.ch)).sum()
}

fn char_width(ch: char) -> f64 {
    if wide(ch) { 1.0 } else { 0.5 }
}

/// A character set full-width: the CJK ideographs and their punctuation,
/// kana, hangul, and the full-width forms. Close enough to East Asian
/// Width's Wide and Fullwidth for laying out a caption.
fn wide(ch: char) -> bool {
    matches!(ch as u32,
        0x1100..=0x115F
        | 0x2E80..=0xA4CF
        | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE4F
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x20000..=0x3FFFD)
}

/// Every character of the phrases, each timed by its even share of the
/// span it came in.
fn glyphs(phrases: &[Vec<Timed>]) -> Vec<Glyph> {
    let mut out = Vec::new();
    for phrase in phrases {
        let first = out.len();
        for span in phrase {
            let count = span.text.chars().count();
            let share = (span.end - span.start).max(0.0) / count.max(1) as f64;
            for (index, ch) in span.text.chars().enumerate() {
                let ch = if ch.is_whitespace() { ' ' } else { ch };
                out.push(Glyph {
                    ch,
                    start: span.start + share * index as f64,
                    end: span.start + share * (index + 1) as f64,
                    last_of_phrase: false,
                });
            }
        }
        if let Some(last) = out[first..].last_mut() {
            last.last_of_phrase = true;
        }
    }
    out
}

/// The glyphs cut into clauses: each ends on its punctuation - with any
/// closing quote or bracket after it - at the end of a phrase, or before
/// a pause of `pause` seconds.
fn clauses(glyphs: &[Glyph], pause: f64) -> Vec<(&[Glyph], Close)> {
    let mut out = Vec::new();
    let mut from = 0;
    let mut index = 0;
    while index < glyphs.len() {
        let glyph = glyphs[index];
        let next = glyphs.get(index + 1);
        let mut close = punctuation_close(glyph.ch, next.map(|next| next.ch));
        let mut to = index + 1;
        if close.is_some() {
            // The quote or bracket the sentence closes goes with it.
            while to < glyphs.len() && closing(glyphs[to].ch) && !glyphs[to - 1].last_of_phrase {
                to += 1;
            }
        }
        let last = glyphs[to - 1];
        let paused = glyphs
            .get(to)
            .is_some_and(|after| after.start - last.end >= pause);
        if last.last_of_phrase || paused {
            close = Some(Close::Hard);
        }
        if let Some(close) = close {
            out.push((&glyphs[from..to], close));
            from = to;
        }
        index = to;
    }
    if from < glyphs.len() {
        out.push((&glyphs[from..], Close::Hard));
    }
    out
}

/// How `ch` closes a clause, if it is punctuation that does. A Latin mark
/// does only before a space, the end, or a CJK character, so "3.5" and
/// "e.g" hold together; a CJK mark always does.
fn punctuation_close(ch: char, next: Option<char>) -> Option<Close> {
    match ch {
        '。' | '！' | '？' | '…' => Some(Close::Hard),
        '，' | '、' | '；' | '：' => Some(Close::Soft),
        '.' | '!' | '?' | ',' | ';' | ':' => {
            let spaced = next.is_none_or(|next| next.is_whitespace() || wide(next));
            spaced.then_some(if matches!(ch, '.' | '!' | '?') {
                Close::Hard
            } else {
                Close::Soft
            })
        }
        _ => None,
    }
}

/// Closing quotes and brackets: they stay with what they close.
fn closing(ch: char) -> bool {
    matches!(
        ch,
        '”' | '’' | '」' | '』' | '）' | '》' | '】' | '〉' | ')' | ']' | '"' | '\''
    )
}

/// Opening quotes and brackets: they stay with what they open.
fn opening(ch: char) -> bool {
    matches!(
        ch,
        '“' | '‘' | '「' | '『' | '（' | '《' | '【' | '〈' | '(' | '['
    )
}

/// Punctuation a line does not begin with: it belongs to the line before.
fn no_start(ch: char) -> bool {
    closing(ch)
        || matches!(
            ch,
            '，' | '。'
                | '、'
                | '；'
                | '：'
                | '！'
                | '？'
                | '…'
                | ','
                | '.'
                | ';'
                | ':'
                | '!'
                | '?'
        )
}

/// What a line drops from its end when [`LineRules::strip_trailing`] is
/// on: the marks that only say "pause here".
fn stripped(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。' | '、' | '；' | '：' | '…' | ',' | '.' | ';' | ':'
    ) || ch.is_whitespace()
}

/// One clause or run of clauses as lines: as it is when it fits, in even
/// parts when it does not.
fn emit(out: &mut Vec<Line>, glyphs: &[Glyph], max_width: f64, rules: &LineRules) {
    let mut rest = glyphs;
    while !rest.is_empty() {
        let total = width(rest);
        if total <= max_width {
            push_line(out, rest, rules);
            break;
        }
        let parts = (total / max_width).ceil();
        let target = total / parts;
        let at = break_at(rest, target, max_width);
        push_line(out, &rest[..at], rules);
        rest = &rest[at..];
    }
}

/// Where to end the first line of `glyphs`: the allowed break nearest
/// `target` columns that keeps the line within `max_width`, or failing any
/// (a word longer than a line), as far as the line reaches.
fn break_at(glyphs: &[Glyph], target: f64, max_width: f64) -> usize {
    let mut best: Option<(usize, f64)> = None;
    let mut reach = 1;
    let mut so_far = 0.0;
    for at in 1..glyphs.len() {
        so_far += char_width(glyphs[at - 1].ch);
        if so_far > max_width {
            break;
        }
        reach = at;
        if may_break(glyphs[at - 1].ch, glyphs[at].ch) {
            let off = (so_far - target).abs();
            if best.is_none_or(|(_, best_off)| off < best_off) {
                best = Some((at, off));
            }
        }
    }
    best.map_or(reach, |(at, _)| at)
}

/// Whether a line may end between `before` and `after`: at a space, or on
/// either side of a CJK character, but never inside a Latin word, before
/// punctuation that closes, or after a bracket that opens.
fn may_break(before: char, after: char) -> bool {
    if no_start(after) || opening(before) {
        return false;
    }
    before.is_whitespace() || after.is_whitespace() || wide(before) || wide(after)
}

/// The glyphs as a line: spaces run together and trimmed, the line's
/// leading punctuation and - by the rules - its closing pause dropped,
/// timed from its first word to its last. Nothing is pushed for a line
/// with no words.
fn push_line(out: &mut Vec<Line>, glyphs: &[Glyph], rules: &LineRules) {
    let mut kept: Vec<Glyph> = Vec::with_capacity(glyphs.len());
    for glyph in glyphs {
        let space = glyph.ch == ' ';
        if space && kept.last().is_none_or(|last| last.ch == ' ') {
            continue;
        }
        if kept.is_empty() && no_start(glyph.ch) && !closing(glyph.ch) {
            continue;
        }
        kept.push(*glyph);
    }
    while kept
        .last()
        .is_some_and(|last| last.ch == ' ' || (rules.strip_trailing && stripped(last.ch)))
    {
        kept.pop();
    }
    if !kept.iter().any(|glyph| glyph.ch.is_alphanumeric()) {
        return;
    }
    let words = || kept.iter().filter(|glyph| glyph.ch.is_alphanumeric());
    let start = words()
        .map(|glyph| glyph.start)
        .fold(f64::INFINITY, f64::min);
    let end = words().map(|glyph| glyph.end).fold(start, f64::max);
    out.push(Line {
        text: kept.iter().map(|glyph| glyph.ch).collect(),
        start,
        end,
    });
}

/// Whether a character takes part in lining a script up with what was
/// heard: letters and digits, of any script.
fn compared(ch: char) -> bool {
    ch.is_alphanumeric()
}

/// A character as it is compared: without case, and with a full-width
/// letter or digit as its ASCII self.
fn fold(ch: char) -> char {
    let ch = match ch as u32 {
        0xFF10..=0xFF19 | 0xFF21..=0xFF3A | 0xFF41..=0xFF5A => {
            char::from_u32(ch as u32 - 0xFEE0).unwrap_or(ch)
        }
        _ => ch,
    };
    ch.to_lowercase().next().unwrap_or(ch)
}

/// The pairs `(i, j)` that line `a[i]` up with `b[j]` in a cheapest edit
/// from `a` to `b`, where a match costs nothing and a substitution,
/// insertion or deletion one each. A substitution is a pair too: a word
/// misheard is still where it was said.
fn edit_pairs(a: &[char], b: &[char]) -> Vec<(usize, usize)> {
    // Which way each cell's cheapest edit came from.
    const DIAGONAL: u8 = 0;
    const UP: u8 = 1;
    const LEFT: u8 = 2;
    let (n, m) = (a.len(), b.len());
    let mut from = vec![DIAGONAL; (n + 1) * (m + 1)];
    let mut row: Vec<u32> = (0..=m as u32).collect();
    for cell in from.iter_mut().take(m + 1).skip(1) {
        *cell = LEFT;
    }
    for i in 1..=n {
        let mut diagonal = row[0];
        row[0] = i as u32;
        from[i * (m + 1)] = UP;
        for j in 1..=m {
            let substitute = diagonal + u32::from(a[i - 1] != b[j - 1]);
            let delete = row[j] + 1;
            let insert = row[j - 1] + 1;
            diagonal = row[j];
            let (cost, way) = if substitute <= delete && substitute <= insert {
                (substitute, DIAGONAL)
            } else if delete <= insert {
                (delete, UP)
            } else {
                (insert, LEFT)
            };
            row[j] = cost;
            from[i * (m + 1) + j] = way;
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        match from[i * (m + 1) + j] {
            DIAGONAL => {
                pairs.push((i - 1, j - 1));
                i -= 1;
                j -= 1;
            }
            UP => i -= 1,
            _ => j -= 1,
        }
    }
    pairs.reverse();
    pairs
}

/// Times for the written characters nothing heard lined up with: spread
/// evenly between the neighbours that have one, or from `first` before
/// the first and to `last` after the last.
fn fill_gaps(times: &mut [Option<(f64, f64)>], first: f64, last: f64) {
    let mut index = 0;
    while index < times.len() {
        if times[index].is_some() {
            index += 1;
            continue;
        }
        let gap_start = index;
        while index < times.len() && times[index].is_none() {
            index += 1;
        }
        let from = if gap_start == 0 {
            first
        } else {
            times[gap_start - 1].expect("before the gap").1
        };
        let to = times
            .get(index)
            .copied()
            .flatten()
            .map_or(last, |(start, _)| start)
            .max(from);
        let step = (to - from) / (index - gap_start) as f64;
        for (k, time) in times[gap_start..index].iter_mut().enumerate() {
            let start = from + step * k as f64;
            *time = Some((start, start + step));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phrase(text: &str, start: f64, end: f64) -> Vec<Timed> {
        vec![Timed::new(text, start, end)]
    }

    fn texts(lines: &[Line]) -> Vec<&str> {
        lines.iter().map(|line| line.text.as_str()).collect()
    }

    fn rules(max_width: f64, strip_trailing: bool) -> LineRules {
        LineRules {
            max_width,
            strip_trailing,
            ..LineRules::default()
        }
    }

    /// A long Chinese phrase is cut at its commas, and the clauses of one
    /// sentence share a line while they fit it.
    #[test]
    fn a_chinese_phrase_is_cut_at_its_punctuation() {
        let heard = [phrase(
            "大家好，欢迎来到我的频道。今天我们来聊一聊，怎么用手机剪出好看的视频！",
            0.0,
            10.0,
        )];
        let cut = lines(&heard, &rules(16.0, false));
        assert_eq!(
            texts(&cut),
            [
                "大家好，欢迎来到我的频道。",
                "今天我们来聊一聊，",
                "怎么用手机剪出好看的视频！"
            ]
        );
    }

    /// CapCut's captions drop the comma and full stop a line ends on, and
    /// keep its question and exclamation marks.
    #[test]
    fn trailing_pauses_are_dropped_and_tone_is_kept() {
        let heard = [phrase("你吃饭了吗？我吃了，你呢。真的吗！", 0.0, 6.0)];
        let cut = lines(&heard, &rules(6.0, true));
        assert_eq!(texts(&cut), ["你吃饭了吗？", "我吃了", "你呢", "真的吗！"]);
    }

    /// A clause wider than a line is split in even parts, not a full line
    /// and a scrap.
    #[test]
    fn a_long_clause_is_split_evenly() {
        // Twenty characters into lines of at most sixteen: ten and ten.
        let heard = [phrase(
            "一二三四五六七八九十一二三四五六七八九十",
            0.0,
            20.0,
        )];
        let cut = lines(&heard, &rules(16.0, true));
        assert_eq!(
            texts(&cut),
            ["一二三四五六七八九十", "一二三四五六七八九十"]
        );
        // Each half is timed by its own characters.
        assert_eq!((cut[0].start, cut[0].end), (0.0, 10.0));
        assert_eq!((cut[1].start, cut[1].end), (10.0, 20.0));
    }

    /// The width limit counts a Latin character as half a Chinese one, and
    /// a Latin word is never broken in two.
    #[test]
    fn latin_words_break_at_spaces() {
        let heard = [phrase(
            "This sentence is a fair bit longer than one caption line can hold",
            0.0,
            6.0,
        )];
        let cut = lines(&heard, &rules(16.0, true));
        assert_eq!(
            texts(&cut),
            [
                "This sentence is a fair",
                "bit longer than one",
                "caption line can hold"
            ]
        );
        assert!(cut.iter().all(|line| text_width(&line.text) <= 16.0));
        // Mixed text: the English word stays whole beside the Chinese.
        let mixed = lines(
            &[phrase(
                "我们今天用iPhone拍摄了一段很长很长的视频素材",
                0.0,
                5.0,
            )],
            &rules(10.0, true),
        );
        assert!(mixed.iter().any(|line| line.text.contains("iPhone")));
        assert!(mixed.iter().all(|line| text_width(&line.text) <= 10.0));
    }

    /// "3.5" and "e.g" are not sentence ends; a full stop before a space is.
    #[test]
    fn latin_marks_close_a_clause_only_before_a_space() {
        let cut = lines(
            &[phrase("Version 3.5 is out. Get it now", 0.0, 3.0)],
            &rules(24.0, false),
        );
        assert_eq!(texts(&cut), ["Version 3.5 is out.", "Get it now"]);
    }

    /// Every phrase ends a line, a pause ends one too, and a closing quote
    /// stays with the sentence it closes.
    #[test]
    fn phrases_pauses_and_quotes() {
        let heard = [
            phrase("他说", 0.0, 1.0),
            vec![
                Timed::new("“我们走吧。”", 1.0, 2.0),
                Timed::new("好的", 4.0, 5.0),
                Timed::new("马上", 5.0, 6.0),
            ],
        ];
        let cut = lines(&heard, &rules(16.0, false));
        assert_eq!(texts(&cut), ["他说", "“我们走吧。”", "好的马上"]);
        assert_eq!((cut[2].start, cut[2].end), (4.0, 6.0));
    }

    /// A line is timed from its first word to its last, whatever
    /// punctuation sits around them; a line of punctuation alone is none.
    #[test]
    fn lines_are_timed_by_their_words() {
        let heard = [
            vec![
                Timed::new("你好", 1.0, 2.0),
                Timed::new("，", 2.0, 3.0),
                Timed::new("世界", 3.0, 4.0),
            ],
            phrase("。", 5.0, 6.0),
        ];
        let cut = lines(&heard, &rules(16.0, true));
        assert_eq!(texts(&cut), ["你好，世界"]);
        assert_eq!((cut[0].start, cut[0].end), (1.0, 4.0));
        assert!(lines(&[], &LineRules::default()).is_empty());
    }

    #[test]
    fn a_tall_frame_gets_the_short_line() {
        assert_eq!(LineRules::width_for_frame(1080, 1920), 16.0);
        assert_eq!(LineRules::width_for_frame(1920, 1080), 24.0);
        assert_eq!(LineRules::width_for_frame(1080, 1080), 24.0);
    }

    /// The script's words, with a misheard character and a missing one,
    /// timed by the transcript: each written character where it was said,
    /// and the one never heard between its neighbours.
    #[test]
    fn a_script_takes_the_timing_of_what_was_heard() {
        // Heard: 大家好 欢迎来到我的平道 (频 misheard as 平), one char a second.
        let heard = [
            phrase("大家好", 0.0, 3.0),
            phrase("欢迎来到我的平道", 4.0, 12.0),
        ];
        let script = "大家好！\n欢迎来到我们的频道。";
        let timed = align_script(script, &heard);
        assert_eq!(timed.len(), 2, "one phrase per line of the script");
        let cut = lines(&timed, &rules(16.0, true));
        assert_eq!(texts(&cut), ["大家好！", "欢迎来到我们的频道"]);
        assert_eq!((cut[0].start, cut[0].end), (0.0, 3.0));
        assert_eq!((cut[1].start, cut[1].end), (4.0, 12.0));
        // 们 was never heard: it sits between 我 (8-9) and 的 (9-10).
        let line = &timed[1];
        let men = line.iter().find(|span| span.text == "们").unwrap();
        assert_eq!((men.start, men.end), (9.0, 9.0));
        let pin = line.iter().find(|span| span.text == "频").unwrap();
        assert_eq!((pin.start, pin.end), (10.0, 11.0));
    }

    #[test]
    fn english_scripts_line_up_without_case() {
        let heard = [phrase("hello world how are you", 0.0, 2.3)];
        let timed = align_script("Hello, World! How are you?", &heard);
        let cut = lines(&timed, &rules(24.0, true));
        assert_eq!(texts(&cut), ["Hello, World!", "How are you?"]);
        assert!(cut[0].end <= cut[1].start + 1e-9);
        assert!(align_script("", &heard).is_empty());
        assert!(align_script("words", &[]).is_empty());
    }

    #[test]
    fn script_files_are_read_as_utf8_or_utf16() {
        assert_eq!(
            decode_script("你好\r\n世界".as_bytes()).as_deref(),
            Some("你好\n世界")
        );
        let bom = [&[0xEF, 0xBB, 0xBF][..], "字幕".as_bytes()].concat();
        assert_eq!(decode_script(&bom).as_deref(), Some("字幕"));
        let le: Vec<u8> = [0xFF, 0xFE]
            .into_iter()
            .chain("文稿".encode_utf16().flat_map(u16::to_le_bytes))
            .collect();
        assert_eq!(decode_script(&le).as_deref(), Some("文稿"));
        let be: Vec<u8> = [0xFE, 0xFF]
            .into_iter()
            .chain("文稿".encode_utf16().flat_map(u16::to_be_bytes))
            .collect();
        assert_eq!(decode_script(&be).as_deref(), Some("文稿"));
        // 你好 in GBK.
        assert_eq!(decode_script(&[0xC4, 0xE3, 0xBA, 0xC3]), None);
    }

    #[test]
    fn edits_pair_what_lines_up() {
        let a: Vec<char> = "abcd".chars().collect();
        let b: Vec<char> = "axcd".chars().collect();
        assert_eq!(edit_pairs(&a, &b), [(0, 0), (1, 1), (2, 2), (3, 3)]);
        let b: Vec<char> = "acd".chars().collect();
        assert_eq!(edit_pairs(&a, &b), [(0, 0), (2, 1), (3, 2)]);
    }
}
