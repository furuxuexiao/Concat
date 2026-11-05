//! 智能剪口播的纯逻辑。
//!
//! 本模块不依赖项目里的其它类型，只用简单的 [`Unit`]（一个字或一个词及其时间，单位秒）：
//!
//! - 文字剪辑：[`Transcript`] 把单元拼成文稿并保留“字符 → 单元”映射，[`diff`] 比较原文与
//!   编辑后的文稿，返回被删掉的单元下标。
//! - 自动精简：[`find_cuts`] 找出口水词、需确认词、重复词和过长停顿，返回带类型的 [`Cut`]。
//! - 时间重映射：[`merge_spans`]、[`ripple_time`]、[`after_cut`]。
//! - 智能划重点：[`highlights`] 在一行字幕里挑出重点字符区间，[`relocate`] 在文字改动后
//!   把区间搬到新位置。

use std::collections::HashMap;

/// LCS 动态规划允许的最大格数，超过后改用贪心对齐。
pub const LCS_CELL_LIMIT: usize = 4_000_000;

/// [`highlights`] 默认每行最多划的重点数。
pub const DEFAULT_MAX_HIGHLIGHTS: usize = 2;

const EPS: f64 = 1e-6;
const GREEDY_WINDOW: usize = 64;
const MAX_REPEAT_LEN: usize = 6;

/// 口水词：出现在任何位置都可以删。
const FILLER_CHARS: &[char] = &['嗯', '呃', '啊', '额', '唔'];
/// 需确认的口头禅：默认关闭，且一侧必须是边界、标点或停顿。
const CONFIRM_WORDS: &[&str] = &["那个", "就是", "然后", "对吧"];
/// 单字重复只在这些口吃常见字上算重复。
const STUTTER_CHARS: &[char] = &['我', '你', '他', '她', '它', '这', '那', '就', '是', '在'];
/// 可以 ABAB 重叠的常见双字动词，“研究研究”这类不算重复。
const REDUP_VERBS: &[&str] = &[
    "研究", "讨论", "休息", "考虑", "商量", "学习", "了解", "打扫", "整理", "收拾", "锻炼", "活动",
    "介绍", "准备", "检查", "练习", "交流", "体验", "尝试", "思考", "琢磨", "熟悉", "比划", "溜达",
    "打听", "参观", "调查", "分析", "总结", "放松", "感受", "享受", "批评", "教育", "表扬", "庆祝",
];
const CN_NUM: &str = "零一二三四五六七八九十百千万亿两";
const NUM_UNITS: &[&str] = &[
    "小时", "分钟", "公里", "公斤", "万", "亿", "个", "年", "月", "日", "天", "倍", "元", "块",
    "人", "次", "岁", "秒", "米", "斤", "克", "名", "位", "%", "％", "k", "K", "w", "W",
];
const CN_UNITS: &[&str] = &[
    "小时", "分钟", "公里", "公斤", "年", "月", "天", "倍", "成", "元", "块", "人", "次", "岁",
    "秒", "米", "斤", "克", "名", "位",
];
const STOP_CHARS: &str = "的了是我你他她它们这那个就在也都和有不吗呢吧啊么着把被给说要会能很还又一";

/// 一个转写单元（一个字或一个词）及其在素材里的时间区间，单位秒。
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    /// 单元文字，可以带标点。
    pub text: String,
    /// 开始时间（秒）。
    pub start: f64,
    /// 结束时间（秒）。
    pub end: f64,
}

impl Unit {
    /// 便捷构造一个单元。
    pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }
}

/// 由单元依次拼接成的文稿，附带字符与单元之间的映射（字符按 `char` 计数）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    /// 拼接后的全文。
    pub text: String,
    /// 第 k 个字符属于哪个单元。
    pub char_unit: Vec<usize>,
    /// 每个单元在全文中的字符区间 `[start, end)`。
    pub unit_ranges: Vec<(usize, usize)>,
}

impl Transcript {
    /// 把单元按顺序拼成文稿（不插入分隔符）。
    pub fn from_units(units: &[Unit]) -> Self {
        let mut text = String::new();
        let mut char_unit = Vec::new();
        let mut unit_ranges = Vec::with_capacity(units.len());
        for (i, u) in units.iter().enumerate() {
            let s = char_unit.len();
            for c in u.text.chars() {
                text.push(c);
                char_unit.push(i);
            }
            unit_ranges.push((s, char_unit.len()));
        }
        Self {
            text,
            char_unit,
            unit_ranges,
        }
    }
}

/// 比较原文和编辑后的文字，返回原文中被删除的字符下标（按 `char` 计，升序）。
///
/// 先去掉公共前后缀，中间部分做 LCS；格数超过 [`LCS_CELL_LIMIT`] 时改用贪心对齐。
pub fn diff_chars(original: &str, edited: &str) -> Vec<usize> {
    let a: Vec<char> = original.chars().collect();
    let b: Vec<char> = edited.chars().collect();
    align(&a, &b)
        .iter()
        .enumerate()
        .filter(|(_, m)| m.is_none())
        .map(|(i, _)| i)
        .collect()
}

/// 文字剪辑：用户把文稿改成 `edited` 后，返回被删除的单元下标（升序）。
///
/// 一个单元的非空白字符全部被删掉才算删除；只有空白的单元不会被报告。
pub fn diff(units: &[Unit], edited: &str) -> Vec<usize> {
    let t = Transcript::from_units(units);
    let a: Vec<char> = t.text.chars().collect();
    let b: Vec<char> = edited.chars().collect();
    let map = align(&a, &b);
    t.unit_ranges
        .iter()
        .enumerate()
        .filter(|&(_, &(s, e))| {
            let mut visible = (s..e).filter(|&k| !a[k].is_whitespace()).peekable();
            visible.peek().is_some() && visible.all(|k| map[k].is_none())
        })
        .map(|(i, _)| i)
        .collect()
}

/// 对齐两个字符序列：返回 `a` 中每个字符在 `b` 中对应的位置，被删除的为 `None`。
fn align(a: &[char], b: &[char]) -> Vec<Option<usize>> {
    let mut map = vec![None; a.len()];
    let mut pre = 0;
    while pre < a.len() && pre < b.len() && a[pre] == b[pre] {
        map[pre] = Some(pre);
        pre += 1;
    }
    let mut suf = 0;
    while suf < a.len() - pre && suf < b.len() - pre && a[a.len() - 1 - suf] == b[b.len() - 1 - suf]
    {
        map[a.len() - 1 - suf] = Some(b.len() - 1 - suf);
        suf += 1;
    }
    let am = &a[pre..a.len() - suf];
    let bm = &b[pre..b.len() - suf];
    if am.is_empty() || bm.is_empty() {
        return map;
    }
    let cells = (am.len() + 1).saturating_mul(bm.len() + 1);
    let mid = if cells <= LCS_CELL_LIMIT {
        lcs_map(am, bm)
    } else {
        greedy_map(am, bm)
    };
    for (i, m) in mid.into_iter().enumerate() {
        map[pre + i] = m.map(|j| j + pre);
    }
    map
}

fn lcs_map(a: &[char], b: &[char]) -> Vec<Option<usize>> {
    let (n, m) = (a.len(), b.len());
    let w = m + 1;
    let mut dp = vec![0u32; (n + 1) * w];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i * w + j] = if a[i] == b[j] {
                dp[(i + 1) * w + j + 1] + 1
            } else {
                dp[(i + 1) * w + j].max(dp[i * w + j + 1])
            };
        }
    }
    let mut map = vec![None; n];
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            map[i] = Some(j);
            i += 1;
            j += 1;
        } else if dp[(i + 1) * w + j] >= dp[i * w + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    map
}

fn greedy_map(a: &[char], b: &[char]) -> Vec<Option<usize>> {
    let mut map = vec![None; a.len()];
    let mut j = 0;
    for (i, &c) in a.iter().enumerate() {
        if j >= b.len() {
            break;
        }
        if c == b[j] {
            map[i] = Some(j);
            j += 1;
            continue;
        }
        let ahead_a = &a[i + 1..(i + 1 + GREEDY_WINDOW).min(a.len())];
        if ahead_a.contains(&b[j]) {
            // a[i] 被删除，b[j] 稍后会匹配上。
            continue;
        }
        // b[j] 是插入的字符：在 b 的近处找 a[i]。
        let window = &b[j..(j + GREEDY_WINDOW).min(b.len())];
        if let Some(off) = window.iter().position(|&x| x == c) {
            map[i] = Some(j + off);
            j += off + 1;
        }
    }
    map
}

/// 剪切的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CutKind {
    /// 口水词：嗯、呃、啊、额、唔。
    Filler,
    /// 需确认的口头禅：那个、就是、然后、对吧。
    Confirm,
    /// 连续重复（删第一次出现）。
    Repeat,
    /// 过长停顿（含首尾空白）。
    Pause,
}

/// 一段建议剪掉的时间区间，单位秒。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cut {
    /// 开始时间（秒）。
    pub start: f64,
    /// 结束时间（秒）。
    pub end: f64,
    /// 剪切类型。
    pub kind: CutKind,
}

/// [`find_cuts`] 的选项。
#[derive(Debug, Clone, PartialEq)]
pub struct CutOptions {
    /// 删除口水词（默认开）。
    pub fillers: bool,
    /// 删除需确认的口头禅（默认关）。
    pub confirm_words: bool,
    /// 删除连续重复（默认开）。
    pub repeats: bool,
    /// 缩短停顿（默认开）。
    pub pauses: bool,
    /// 达到该长度（秒）的停顿才缩短，默认 0.5。
    pub pause_min: f64,
    /// 缩短后保留的停顿长度（秒），默认 0.15。
    pub pause_keep: f64,
    /// 需确认词两侧多长的空隙（秒）算“停顿边界”，默认 0.3。
    pub confirm_gap: f64,
    /// 素材总时长（秒）；给出时会处理结尾空白。
    pub duration: Option<f64>,
}

impl Default for CutOptions {
    fn default() -> Self {
        Self {
            fillers: true,
            confirm_words: false,
            repeats: true,
            pauses: true,
            pause_min: 0.5,
            pause_keep: 0.15,
            confirm_gap: 0.3,
            duration: None,
        }
    }
}

fn is_punct(c: char) -> bool {
    c.is_whitespace()
        || c.is_ascii_punctuation()
        || "，。、！？；：“”‘’（）《》【】…—·「」『』～".contains(c)
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// 字符区间 `[cs, ce)` 是否正好落在单元边界上（单元里多出的只能是标点）；
/// 是则返回首尾单元下标。
fn aligned_units(t: &Transcript, chars: &[char], cs: usize, ce: usize) -> Option<(usize, usize)> {
    if cs >= ce || ce > chars.len() {
        return None;
    }
    let u0 = t.char_unit[cs];
    let u1 = t.char_unit[ce - 1];
    let left_ok = chars[t.unit_ranges[u0].0..cs].iter().all(|&c| is_punct(c));
    let right_ok = chars[ce..t.unit_ranges[u1].1].iter().all(|&c| is_punct(c));
    (left_ok && right_ok).then_some((u0, u1))
}

/// 找出建议剪掉的片段：口水词、需确认词、重复词和过长停顿。
///
/// 结果按开始时间排序，不同类型之间可能重叠；交给 [`merge_spans`] 合并后再剪。
pub fn find_cuts(units: &[Unit], opts: &CutOptions) -> Vec<Cut> {
    let mut cuts = Vec::new();
    let t = Transcript::from_units(units);
    let chars: Vec<char> = t.text.chars().collect();

    if opts.fillers {
        for u in units {
            let mut core = u.text.chars().filter(|&c| !is_punct(c)).peekable();
            if core.peek().is_some() && core.all(|c| FILLER_CHARS.contains(&c)) {
                cuts.push(Cut {
                    start: u.start,
                    end: u.end,
                    kind: CutKind::Filler,
                });
            }
        }
    }
    if opts.confirm_words {
        find_confirm(units, &t, &chars, opts, &mut cuts);
    }
    if opts.repeats {
        find_repeats(units, &t, &chars, &mut cuts);
    }
    if opts.pauses {
        find_pauses(units, opts, &mut cuts);
    }
    cuts.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)));
    cuts
}

fn find_confirm(
    units: &[Unit],
    t: &Transcript,
    chars: &[char],
    opts: &CutOptions,
    cuts: &mut Vec<Cut>,
) {
    for w in CONFIRM_WORDS {
        let wc: Vec<char> = w.chars().collect();
        for s in find_all(chars, &wc) {
            let e = s + wc.len();
            let Some((u0, u1)) = aligned_units(t, chars, s, e) else {
                continue;
            };
            let left = s == 0
                || is_punct(chars[s - 1])
                || u0 == 0
                || units[u0].start - units[u0 - 1].end >= opts.confirm_gap - EPS;
            let right = e == chars.len()
                || is_punct(chars[e])
                || u1 + 1 == units.len()
                || units[u1 + 1].start - units[u1].end >= opts.confirm_gap - EPS;
            if left || right {
                cuts.push(Cut {
                    start: units[u0].start,
                    end: units[u1].end,
                    kind: CutKind::Confirm,
                });
            }
        }
    }
}

fn repeat_word_ok(w: &[char]) -> bool {
    if w.iter().all(|c| c.is_ascii_digit()) {
        return false;
    }
    match w.len() {
        1 => STUTTER_CHARS.contains(&w[0]),
        2 => {
            // AA 叠词（看看、谢谢）和 ABAB 动词重叠（研究研究）不算重复。
            let s: String = w.iter().collect();
            w[0] != w[1] && !REDUP_VERBS.contains(&s.as_str())
        }
        _ => true,
    }
}

fn find_repeats(units: &[Unit], t: &Transcript, chars: &[char], cuts: &mut Vec<Cut>) {
    let idx: Vec<usize> = (0..chars.len()).filter(|&k| !is_punct(chars[k])).collect();
    let cc: Vec<char> = idx.iter().map(|&k| chars[k]).collect();
    let mut i = 0;
    while i < cc.len() {
        let mut found = None;
        for l in (1..=MAX_REPEAT_LEN).rev() {
            if i + 2 * l > cc.len() || cc[i..i + l] != cc[i + l..i + 2 * l] {
                continue;
            }
            if !repeat_word_ok(&cc[i..i + l]) {
                continue;
            }
            let cs = idx[i];
            let ce = idx[i + l - 1] + 1;
            if let Some((u0, u1)) = aligned_units(t, chars, cs, ce) {
                let u2 = t.char_unit[idx[i + l]];
                if u2 > u1 {
                    found = Some((l, units[u0].start, units[u2].start));
                    break;
                }
            }
        }
        if let Some((l, start, end)) = found {
            cuts.push(Cut {
                start,
                end,
                kind: CutKind::Repeat,
            });
            i += l;
        } else {
            i += 1;
        }
    }
}

fn find_pauses(units: &[Unit], opts: &CutOptions, cuts: &mut Vec<Cut>) {
    let keep = opts.pause_keep.max(0.0);
    let min = opts.pause_min.max(keep);
    let mut push = |start: f64, end: f64| {
        if end - start > EPS {
            cuts.push(Cut {
                start,
                end,
                kind: CutKind::Pause,
            });
        }
    };
    let Some(first) = units.first() else {
        return;
    };
    if first.start >= min - EPS {
        push(0.0, first.start - keep);
    }
    for w in units.windows(2) {
        let gap = w[1].start - w[0].end;
        if gap >= min - EPS {
            let half = keep / 2.0;
            push(w[0].end + half, w[1].start - half);
        }
    }
    if let (Some(last), Some(d)) = (units.last(), opts.duration)
        && d - last.end >= min - EPS
    {
        push(last.end + keep, d);
    }
}

/// 合并时间区间：排序后把重叠或相接的区间并起来，丢弃空区间。
pub fn merge_spans(spans: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = spans.iter().copied().filter(|(s, e)| e - s > EPS).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(v.len());
    for (s, e) in v {
        match out.last_mut() {
            Some(last) if s <= last.1 + EPS => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
}

/// 把一组 [`Cut`] 转成合并后的时间区间。
pub fn cut_spans(cuts: &[Cut]) -> Vec<(f64, f64)> {
    let spans: Vec<(f64, f64)> = cuts.iter().map(|c| (c.start, c.end)).collect();
    merge_spans(&spans)
}

/// 剪掉 `spans` 后，原时间 `t` 对应的新时间；落在被剪区间内的时间映射到该区间的起点。
pub fn ripple_time(t: f64, spans: &[(f64, f64)]) -> f64 {
    let mut removed = 0.0;
    for (s, e) in merge_spans(spans) {
        if t >= e {
            removed += e - s;
        } else if t > s {
            return s - removed;
        } else {
            break;
        }
    }
    t - removed
}

/// 剪掉 `spans` 后重映射单元：完全落在被剪区间里的单元被丢弃，其余单元平移时间。
pub fn after_cut(units: &[Unit], spans: &[(f64, f64)]) -> Vec<Unit> {
    let merged = merge_spans(spans);
    units
        .iter()
        .filter(|u| {
            !merged
                .iter()
                .any(|&(s, e)| u.start >= s - EPS && u.end <= e + EPS)
        })
        .map(|u| Unit {
            text: u.text.clone(),
            start: ripple_time(u.start, &merged),
            end: ripple_time(u.end, &merged),
        })
        .collect()
}

fn find_all(hay: &[char], pat: &[char]) -> Vec<usize> {
    if pat.is_empty() || pat.len() > hay.len() {
        return Vec::new();
    }
    hay.windows(pat.len())
        .enumerate()
        .filter(|(_, w)| *w == pat)
        .map(|(i, _)| i)
        .collect()
}

fn match_unit(chars: &[char], pos: usize, list: &[&str]) -> usize {
    list.iter()
        .map(|u| u.chars().collect::<Vec<char>>())
        .filter(|u| chars.len() >= pos + u.len() && chars[pos..pos + u.len()] == u[..])
        .map(|u| u.len())
        .max()
        .unwrap_or(0)
}

fn number_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c.is_ascii_digit() && (i == 0 || !chars[i - 1].is_ascii_alphanumeric()) {
            let mut j = i;
            while j < n
                && (chars[j].is_ascii_digit()
                    || ((chars[j] == '.' || chars[j] == ',')
                        && j + 1 < n
                        && chars[j + 1].is_ascii_digit()))
            {
                j += 1;
            }
            let u = match_unit(chars, j, NUM_UNITS);
            out.push((i, j + u));
            i = j + u;
            continue;
        }
        if CN_NUM.contains(c) && (i == 0 || !CN_NUM.contains(chars[i - 1])) {
            let mut j = i;
            while j < n && CN_NUM.contains(chars[j]) {
                j += 1;
            }
            let run = j - i;
            let u = match_unit(chars, j, CN_UNITS);
            let big_tail = run >= 2 && "百千万亿".contains(chars[j - 1]);
            let lone_one = run == 1 && c == '一';
            if !lone_one && (u > 0 || big_tail) {
                out.push((i, j + u));
                i = j + u;
            } else {
                i = j;
            }
            continue;
        }
        i += 1;
    }
    out
}

fn english_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if chars[i].is_ascii_alphabetic() && (i == 0 || !chars[i - 1].is_ascii_alphanumeric()) {
            let mut j = i;
            while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '+' || chars[j] == '#')
            {
                j += 1;
            }
            if j - i >= 2 {
                out.push((i, j));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn bigram_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let ok = |c: char| is_cjk(c) && !STOP_CHARS.contains(c);
    let mut counts: HashMap<(char, char), (usize, usize)> = HashMap::new();
    for (i, w) in chars.windows(2).enumerate() {
        if ok(w[0]) && ok(w[1]) {
            counts.entry((w[0], w[1])).or_insert((0, i)).0 += 1;
        }
    }
    let mut v: Vec<(usize, usize)> = counts.into_values().filter(|(n, _)| *n >= 2).collect();
    v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    v.into_iter().map(|(_, p)| (p, p + 2)).collect()
}

/// 智能划重点：在一行字幕里挑出最多 `max` 个互不重叠的重点，返回字符区间 `[start, end)`
/// （按 `char` 计，升序）。
///
/// 优先级：用户关键词 > 数字或中文数字加单位 > 英文词 > 行内高频双字词。
pub fn highlights(line: &str, keywords: &[&str], max: usize) -> Vec<(usize, usize)> {
    let chars: Vec<char> = line.chars().collect();
    let mut kw_spans = Vec::new();
    for kw in keywords {
        let k: Vec<char> = kw.trim().chars().collect();
        if let Some(&p) = find_all(&chars, &k).first() {
            kw_spans.push((p, p + k.len()));
        }
    }
    let tiers = [
        kw_spans,
        number_spans(&chars),
        english_spans(&chars),
        bigram_spans(&chars),
    ];
    let mut chosen: Vec<(usize, usize)> = Vec::new();
    for sp in tiers.into_iter().flatten() {
        if chosen.len() >= max {
            break;
        }
        if !chosen.iter().any(|c| c.0 < sp.1 && sp.0 < c.1) {
            chosen.push(sp);
        }
    }
    chosen.sort();
    chosen
}

/// 文字从 `old_text` 改成 `new_text` 后，把旧的字符区间搬到新文字里的对应位置。
///
/// 区间文字在新文字中找不到时丢弃；出现多次时取离对齐位置最近的一处。
pub fn relocate(old_text: &str, new_text: &str, spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let a: Vec<char> = old_text.chars().collect();
    let b: Vec<char> = new_text.chars().collect();
    let map = align(&a, &b);
    spans
        .iter()
        .filter_map(|&(s, e)| {
            if s >= e || e > a.len() {
                return None;
            }
            let expected = (s..e)
                .find_map(|k| map[k].map(|j| j.saturating_sub(k - s)))
                .or_else(|| (0..s).rev().find_map(|k| map[k].map(|j| j + (s - k))))
                .unwrap_or(0);
            find_all(&b, &a[s..e])
                .into_iter()
                .min_by_key(|p| p.abs_diff(expected))
                .map(|p| (p, p + e - s))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 逐字单元，每字 0.2 秒，首尾相接。
    fn cu(s: &str) -> Vec<Unit> {
        s.chars()
            .enumerate()
            .map(|(i, c)| Unit::new(c.to_string(), i as f64 * 0.2, (i + 1) as f64 * 0.2))
            .collect()
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn kinds(cuts: &[Cut], k: CutKind) -> Vec<(f64, f64)> {
        cuts.iter()
            .filter(|c| c.kind == k)
            .map(|c| (c.start, c.end))
            .collect()
    }

    fn no_pause() -> CutOptions {
        CutOptions {
            pauses: false,
            ..CutOptions::default()
        }
    }

    #[test]
    fn transcript_maps_chars_to_units() {
        let units = vec![
            Unit::new("今天", 0.0, 0.4),
            Unit::new("天气", 0.4, 0.8),
            Unit::new("很好。", 0.8, 1.2),
        ];
        let t = Transcript::from_units(&units);
        assert_eq!(t.text, "今天天气很好。");
        assert_eq!(t.char_unit, vec![0, 0, 1, 1, 2, 2, 2]);
        assert_eq!(t.unit_ranges, vec![(0, 2), (2, 4), (4, 7)]);
    }

    #[test]
    fn diff_single_deleted_char_unit() {
        let units = cu("大家好嗯我是小明");
        assert_eq!(diff(&units, "大家好我是小明"), vec![3]);
    }

    #[test]
    fn diff_word_units_repeated_word() {
        let units = vec![
            Unit::new("今天", 0.0, 0.4),
            Unit::new("我们", 0.4, 0.8),
            Unit::new("我们", 0.8, 1.2),
            Unit::new("聊聊", 1.2, 1.6),
            Unit::new("AI", 1.6, 2.0),
        ];
        assert_eq!(diff(&units, "今天我们聊聊AI"), vec![2]);
    }

    #[test]
    fn diff_replacement_counts_as_deletion() {
        let units = cu("今天天气很好");
        assert_eq!(diff(&units, "今天天气不错"), vec![4, 5]);
        assert_eq!(diff_chars("今天天气很好", "今天天气不错"), vec![4, 5]);
    }

    #[test]
    fn diff_partial_unit_edit_keeps_unit() {
        let units = vec![Unit::new("你好", 0.0, 0.4), Unit::new("世界", 0.4, 0.8)];
        assert_eq!(diff(&units, "你世界"), Vec::<usize>::new());
        assert_eq!(diff(&units, "世界"), vec![0]);
    }

    #[test]
    fn diff_unchanged_and_all_deleted() {
        let units = cu("一切照旧");
        assert!(diff(&units, "一切照旧").is_empty());
        assert_eq!(diff(&units, ""), vec![0, 1, 2, 3]);
    }

    #[test]
    fn diff_middle_deletions_with_lcs() {
        let units = cu("我们今天呃来聊一聊那个剪辑");
        assert_eq!(diff(&units, "我们今天来聊一聊剪辑"), vec![4, 9, 10]);
    }

    #[test]
    fn diff_large_input_uses_greedy() {
        let pattern: Vec<char> = "春夏秋冬东南西北".chars().collect();
        let a: String = (0..4200).map(|i| pattern[i % 8]).collect();
        let b: String = a
            .chars()
            .enumerate()
            .filter(|(i, _)| *i != 10 && *i != 4190)
            .map(|(_, c)| c)
            .collect();
        assert_eq!(diff_chars(&a, &b), vec![10, 4190]);
    }

    #[test]
    fn fillers_anywhere() {
        let units = cu("嗯今天呃我们聊啊");
        let cuts = find_cuts(&units, &no_pause());
        let f = kinds(&cuts, CutKind::Filler);
        assert_eq!(f.len(), 3);
        assert!(approx(f[0].0, 0.0) && approx(f[0].1, 0.2));
        assert!(approx(f[1].0, 0.6) && approx(f[1].1, 0.8));
        assert!(approx(f[2].0, 1.4) && approx(f[2].1, 1.6));
    }

    #[test]
    fn filler_unit_with_punct_and_mixed_unit() {
        let units = vec![
            Unit::new("呃，", 0.0, 0.3),
            Unit::new("嗯好", 0.3, 0.6),
            Unit::new("额额", 0.6, 0.9),
        ];
        let f = kinds(&find_cuts(&units, &no_pause()), CutKind::Filler);
        assert_eq!(f.len(), 2);
        assert!(approx(f[0].0, 0.0));
        assert!(approx(f[1].0, 0.6));
    }

    #[test]
    fn confirm_words_off_by_default() {
        let units = cu("那个，我们开始");
        assert!(kinds(&find_cuts(&units, &no_pause()), CutKind::Confirm).is_empty());
    }

    #[test]
    fn confirm_words_need_boundary() {
        let opts = CutOptions {
            confirm_words: true,
            ..no_pause()
        };
        let units = cu("那个，我们开始");
        let c = kinds(&find_cuts(&units, &opts), CutKind::Confirm);
        assert_eq!(c.len(), 1);
        assert!(approx(c[0].0, 0.0) && approx(c[0].1, 0.4));

        let units = cu("我就是喜欢");
        assert!(kinds(&find_cuts(&units, &opts), CutKind::Confirm).is_empty());

        let units = cu("好的然后，我们走");
        let c = kinds(&find_cuts(&units, &opts), CutKind::Confirm);
        assert_eq!(c.len(), 1);
        assert!(approx(c[0].0, 0.4) && approx(c[0].1, 0.8));
    }

    #[test]
    fn confirm_word_after_pause() {
        let opts = CutOptions {
            confirm_words: true,
            ..no_pause()
        };
        let units = vec![
            Unit::new("我们", 0.0, 0.4),
            Unit::new("就是", 1.0, 1.4),
            Unit::new("喜欢", 1.4, 1.8),
        ];
        let c = kinds(&find_cuts(&units, &opts), CutKind::Confirm);
        assert_eq!(c.len(), 1);
        assert!(approx(c[0].0, 1.0) && approx(c[0].1, 1.4));
    }

    #[test]
    fn repeat_single_stutter_char() {
        let units = cu("我我觉得");
        let r = kinds(&find_cuts(&units, &no_pause()), CutKind::Repeat);
        assert_eq!(r.len(), 1);
        assert!(approx(r[0].0, 0.0) && approx(r[0].1, 0.2));
    }

    #[test]
    fn repeat_triple_keeps_last() {
        let units = cu("我我我想说");
        let r = kinds(&find_cuts(&units, &no_pause()), CutKind::Repeat);
        assert_eq!(r.len(), 2);
        assert!(approx(r[0].0, 0.0) && approx(r[1].0, 0.2) && approx(r[1].1, 0.4));
    }

    #[test]
    fn repeat_word_across_punct() {
        let units = cu("我们，我们去吃饭");
        let r = kinds(&find_cuts(&units, &no_pause()), CutKind::Repeat);
        assert_eq!(r.len(), 1);
        assert!(approx(r[0].0, 0.0) && approx(r[0].1, 0.6));
    }

    #[test]
    fn repeat_word_units() {
        let units = vec![
            Unit::new("这个", 0.0, 0.4),
            Unit::new("这个", 0.4, 0.8),
            Unit::new("方案", 0.8, 1.2),
        ];
        let r = kinds(&find_cuts(&units, &no_pause()), CutKind::Repeat);
        assert_eq!(r.len(), 1);
        assert!(approx(r[0].0, 0.0) && approx(r[0].1, 0.4));
    }

    #[test]
    fn reduplication_is_not_repeat() {
        for s in [
            "我们看看吧",
            "回去研究研究",
            "大家休息休息",
            "谢谢谢谢",
            "好好学习",
        ] {
            let units = cu(s);
            assert!(
                kinds(&find_cuts(&units, &no_pause()), CutKind::Repeat).is_empty(),
                "{s}"
            );
        }
    }

    #[test]
    fn longer_phrase_repeat() {
        let units = cu("我觉得我觉得可以");
        let r = kinds(&find_cuts(&units, &no_pause()), CutKind::Repeat);
        assert_eq!(r.len(), 1);
        assert!(approx(r[0].0, 0.0) && approx(r[0].1, 0.6));
    }

    #[test]
    fn pauses_including_head_and_tail() {
        let units = vec![
            Unit::new("大家", 2.0, 2.4),
            Unit::new("好", 2.4, 2.6),
            Unit::new("今天", 3.6, 4.0),
            Unit::new("开始", 4.3, 4.7),
        ];
        let opts = CutOptions {
            duration: Some(6.0),
            ..CutOptions::default()
        };
        let p = kinds(&find_cuts(&units, &opts), CutKind::Pause);
        assert_eq!(p.len(), 3);
        assert!(approx(p[0].0, 0.0) && approx(p[0].1, 1.85));
        assert!(approx(p[1].0, 2.675) && approx(p[1].1, 3.525));
        assert!(approx(p[2].0, 4.85) && approx(p[2].1, 6.0));
    }

    #[test]
    fn short_pauses_untouched() {
        let units = vec![Unit::new("你好", 0.1, 0.5), Unit::new("世界", 0.8, 1.2)];
        let opts = CutOptions {
            duration: Some(1.4),
            ..CutOptions::default()
        };
        assert!(find_cuts(&units, &opts).is_empty());
        assert!(find_cuts(&[], &opts).is_empty());
    }

    #[test]
    fn cuts_sorted() {
        let mut units = cu("嗯我我觉得");
        units.push(Unit::new("好", 3.0, 3.2));
        let cuts = find_cuts(&units, &CutOptions::default());
        assert!(cuts.windows(2).all(|w| w[0].start <= w[1].start));
        assert!(cuts.iter().any(|c| c.kind == CutKind::Pause));
        assert!(cuts.iter().any(|c| c.kind == CutKind::Repeat));
        assert!(cuts.iter().any(|c| c.kind == CutKind::Filler));
    }

    #[test]
    fn merge_spans_works() {
        let m = merge_spans(&[(3.0, 4.0), (0.0, 1.0), (0.5, 2.0), (2.0, 2.5), (5.0, 5.0)]);
        assert_eq!(m, vec![(0.0, 2.5), (3.0, 4.0)]);
        assert!(merge_spans(&[]).is_empty());
    }

    #[test]
    fn ripple_time_maps() {
        let spans = [(1.0, 2.0), (3.0, 3.5)];
        assert!(approx(ripple_time(0.5, &spans), 0.5));
        assert!(approx(ripple_time(1.5, &spans), 1.0));
        assert!(approx(ripple_time(2.5, &spans), 1.5));
        assert!(approx(ripple_time(4.0, &spans), 2.5));
        assert!(approx(ripple_time(3.2, &spans), 2.0));
    }

    #[test]
    fn after_cut_remaps_units() {
        let units = cu("嗯大家好");
        let cuts = find_cuts(&units, &no_pause());
        let out = after_cut(&units, &cut_spans(&cuts));
        let text: String = out.iter().map(|u| u.text.as_str()).collect();
        assert_eq!(text, "大家好");
        assert!(approx(out[0].start, 0.0) && approx(out[2].end, 0.6));
    }

    #[test]
    fn after_cut_text_edit_roundtrip() {
        let units = cu("今天呃天气不错");
        let del = diff(&units, "今天天气不错");
        let spans: Vec<(f64, f64)> = del
            .iter()
            .map(|&i| (units[i].start, units[i].end))
            .collect();
        let out = after_cut(&units, &spans);
        assert_eq!(out.len(), 6);
        assert!(approx(out[2].start, 0.4));
    }

    #[test]
    fn highlights_keyword_first() {
        let h = highlights("今年营收增长了30%，AI业务翻倍", &["AI业务"], 2);
        let line: Vec<char> = "今年营收增长了30%，AI业务翻倍".chars().collect();
        let got: Vec<String> = h
            .iter()
            .map(|&(s, e)| line[s..e].iter().collect())
            .collect();
        assert_eq!(got, vec!["30%", "AI业务"]);
    }

    #[test]
    fn highlights_numbers_and_chinese_numbers() {
        let line = "我们用了三年时间，做到一万人";
        let chars: Vec<char> = line.chars().collect();
        let h = highlights(line, &[], 2);
        let got: Vec<String> = h
            .iter()
            .map(|&(s, e)| chars[s..e].iter().collect())
            .collect();
        assert_eq!(got, vec!["三年", "一万人"]);
    }

    #[test]
    fn highlights_skip_plain_yi() {
        let line = "这是一个好方法";
        assert!(highlights(line, &[], 2).is_empty());
    }

    #[test]
    fn highlights_english_then_bigram() {
        let line = "剪辑软件里剪辑最快的是Concat";
        let chars: Vec<char> = line.chars().collect();
        let h = highlights(line, &[], 2);
        let got: Vec<String> = h
            .iter()
            .map(|&(s, e)| chars[s..e].iter().collect())
            .collect();
        assert_eq!(got, vec!["剪辑", "Concat"]);
    }

    #[test]
    fn highlights_respects_max_and_overlap() {
        let line = "GPT 和 GPU 都涨了 50% 和 80%";
        assert_eq!(highlights(line, &[], 1).len(), 1);
        assert_eq!(highlights(line, &[], 3).len(), 3);
        let h = highlights("增长50%", &["增长50"], 2);
        assert_eq!(h, vec![(0, 4)]);
        assert_eq!(highlights("abc", &[], 0), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn relocate_after_insert_and_delete() {
        let old = "今年营收增长了30%";
        let spans = vec![(7, 10)];
        assert_eq!(
            relocate(old, "嗯，今年营收增长了30%", &spans),
            vec![(9, 12)]
        );
        assert_eq!(relocate(old, "营收增长30%", &spans), vec![(4, 7)]);
        assert!(relocate(old, "营收增长不少", &spans).is_empty());
    }

    #[test]
    fn relocate_picks_nearest_occurrence() {
        let old = "AI 改变 AI";
        let spans = vec![(6, 8)];
        assert_eq!(relocate(old, "AI 真的改变 AI", &spans), vec![(8, 10)]);
        assert_eq!(relocate(old, "AI 改变 AI", &[(0, 2)]), vec![(0, 2)]);
    }
}
