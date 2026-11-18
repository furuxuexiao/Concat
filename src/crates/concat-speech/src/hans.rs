// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Concat contributors

//! Traditional Chinese characters written as Simplified ones.
//!
//! Whisper is told to write Simplified Chinese by the prompt it starts
//! from (see `transcribe::chinese_prompt`), and a prompt only leans: a
//! run still slips into Traditional for a word or a whole window, and a
//! caption track that reads 這是 in one line and 这是 in the next is not
//! one a Chinese editor would ship. Captions asked for in Simplified are
//! put through [`to_simplified`] after the run, so they are Simplified
//! whatever whisper wrote.
//!
//! The tables are OpenCC's (see `hans_table.rs`): character by character,
//! with the phrases where that would be wrong (乾隆 stays 乾隆, 乾淨 is
//! 干净) looked for first, longest first. Text already in Simplified is
//! left as it is: the keys are Traditional forms, and the few OpenCC also
//! uses in Simplified (乾 of 乾隆) are kept by the phrases that need them. Every
//! conversion keeps the number of characters, which is what lets
//! [`to_simplified_pieces`] convert a caption's timed pieces as one text
//! and hand each piece its own characters back.

use crate::hans_table::{CHARS, PHRASES};

/// The longest phrase in [`PHRASES`], in characters.
fn longest_phrase() -> usize {
    PHRASES
        .iter()
        .map(|(phrase, _)| phrase.chars().count())
        .max()
        .unwrap_or(0)
}

/// One character's Simplified form: itself when it has none.
fn simplified_char(c: char) -> char {
    CHARS
        .binary_search_by(|(key, _)| key.cmp(&c))
        .map(|index| CHARS[index].1)
        .unwrap_or(c)
}

/// `text` in Simplified characters, with the same number of characters.
pub fn to_simplified(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let longest = longest_phrase();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    'next: while at < chars.len() {
        // Longest phrase first, so 乾坤大挪移 is found before 乾坤.
        for length in (2..=longest.min(chars.len() - at)).rev() {
            let candidate: String = chars[at..at + length].iter().collect();
            if let Ok(index) = PHRASES.binary_search_by(|(key, _)| (*key).cmp(&candidate)) {
                out.push_str(PHRASES[index].1);
                at += length;
                continue 'next;
            }
        }
        out.push(simplified_char(chars[at]));
        at += 1;
    }
    out
}

/// `pieces` in Simplified characters, converted as the one text they make
/// together - so a phrase split across two pieces is still found - and
/// handed back each with as many characters as it had.
pub fn to_simplified_pieces(pieces: &[&str]) -> Vec<String> {
    let whole: String = pieces.concat();
    let converted: Vec<char> = to_simplified(&whole).chars().collect();
    let mut at = 0;
    pieces
        .iter()
        .map(|piece| {
            let count = piece.chars().count();
            let end = (at + count).min(converted.len());
            let part: String = converted[at..end].iter().collect();
            at = end;
            part
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tables are sorted, which the binary searches rely on, and every
    /// phrase keeps its length, which the pieces rely on.
    #[test]
    fn the_tables_are_sorted_and_phrases_keep_their_length() {
        assert!(CHARS.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(PHRASES.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for (traditional, simplified) in PHRASES {
            assert_eq!(
                traditional.chars().count(),
                simplified.chars().count(),
                "{traditional}"
            );
        }
        assert!(CHARS.len() > 2000, "only {} characters", CHARS.len());
    }

    /// What whisper writes when it slips: Traditional, turned Simplified.
    #[test]
    fn traditional_is_written_simplified() {
        assert_eq!(to_simplified("這是一個測試"), "这是一个测试");
        assert_eq!(to_simplified("歡迎來到我們的頻道"), "欢迎来到我们的频道");
        assert_eq!(
            to_simplified("今天我們來講講視頻剪輯。"),
            "今天我们来讲讲视频剪辑。"
        );
    }

    /// Simplified text, punctuation, Latin and digits come out as they went in.
    #[test]
    fn simplified_text_is_left_alone() {
        for text in [
            "这是一个测试，欢迎关注！",
            "著名的显著著作",
            "乾隆年间",
            "4K 视频 60fps",
            "",
        ] {
            assert_eq!(to_simplified(text), text);
        }
    }

    /// A phrase wins over its characters one by one.
    #[test]
    fn phrases_win_over_characters() {
        assert_eq!(to_simplified("乾淨"), "干净");
        assert_eq!(to_simplified("乾隆"), "乾隆");
        assert_eq!(to_simplified("一目瞭然"), "一目了然");
    }

    /// Pieces are converted together and keep their own characters.
    #[test]
    fn pieces_keep_their_boundaries() {
        let pieces = ["歡迎", "來到", "乾", "隆年間"];
        assert_eq!(
            to_simplified_pieces(&pieces),
            vec!["欢迎", "来到", "乾", "隆年间"]
        );
        assert_eq!(to_simplified_pieces(&[]), Vec::<String>::new());
    }
}
