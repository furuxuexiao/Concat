# Chinese captions (中文字幕)

How the Captions sheet makes captions that read like CapCut's (剪映's
智能字幕), and what each choice in it does.

## Language (语言)

The choice sits in the sheet under *Transcribe this clip's voice* and is
remembered. Since 0.2.11 the list is upstream's: *Auto-detect*, then every
language whisper knows, each by its own name. The fork adds Chinese twice
over - **简体中文** and **繁體中文**, one after the other - since whisper
writes either:

| Choice | What the transcriber is told |
|---|---|
| Auto-detect (自动检测) | Whisper hears the language over the speech alone (upstream's detection, over up to three stretches of it). Chinese heard is written in Simplified characters, as if 简体中文 had been picked. Under a 繁體中文 interface the request is 繁體中文 instead. |
| 简体中文 (`zh`) | Chinese, prompted in Simplified characters with punctuation, so the captions come out in 简体 and punctuated rather than 繁體 and run together. |
| 繁體中文 (`zh-Hant`) | Chinese, prompted in Traditional characters with punctuation. |
| Any other language | That language, with no prompt. An English-only model (`*.en`) is always told English, and the list then shows English alone. |

A caption that only repeats the prompt (whisper falling back on it over
music or noise) is dropped.

Pick a **multilingual** model (Tiny, Base or Small "(Multilingual)") in
Settings › Transcriber for Chinese; the `.en` models only hear English.

## Lines (每行字数, 去掉句末标点)

A transcription is cut into lines at its punctuation (，。！？；：、 and
their Latin forms): a clause is a line, short clauses of one sentence share
one while they fit, and a clause longer than a line is split into even
parts. Each line comes up and goes when its own words are said, timed by
whisper's per-token timestamps.

- **Line length**: *By frame* is 16 characters in a tall (vertical) frame and
  24 in a wide one; or pick 16 or 24. A Latin letter, digit or space counts
  as half a character, so 24 is the 48 characters English subtitles have
  always had, and an English word is never broken.
- **Drop the comma or full stop a line ends on** (on by default): the line
  break already says it. Question and exclamation marks stay.

## Script matching (文稿匹配)

With *Transcribe this clip's voice* on, paste a script into the Script box,
or **Import script…** from a `.txt` file (UTF-8, or UTF-16 with a byte-order
mark; a GBK file is refused with a request to save it as UTF-8). The
captions then use the script's words and punctuation, each character timed
by the transcript character it lines up with (fewest edits, so a misheard
同音字 still takes its time); a line break in the script always breaks the
captions. A script over about six thousand characters against as long a
transcript is lined up by proportion instead.

## Apply a look to all captions (应用到全部字幕)

Select one caption, style it in the Text inspector, and press **Apply to
all on this lane**: every title on that lane - where one run's captions
land - takes its font, size, weight, colours, stroke, shadow and
background, keeping its own words. One undo step.

## Credits

No third-party code, data, fonts or other assets were added for these
features, and no new dependency. The behaviour follows CapCut's captions as
a user sees them; the line cutting, the script alignment (a plain
edit-distance alignment) and the Chinese prompts are Concat's own.
Transcription is whisper.cpp, as before; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
