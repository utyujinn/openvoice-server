// MeloTTS' sentence splitting for English (melo/split_utils.py: split_sentences_latin and txtsplit), step by step, quirks
// included — for example a "." between digits gets a space after it ("$5.50" becomes "$5. 50"), as it does in the Python,
// and a look at the character after the next one returns "nothing" one position early at the end of the text, which
// Python's `"" in "abc"` counts as a match.

use regex::Regex;
use std::sync::OnceLock;

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("a fixed pattern"))
}

pub fn split_sentences_latin(text: &str) -> Vec<String> {
    static STOPS: OnceLock<Regex> = OnceLock::new();
    static COMMA: OnceLock<Regex> = OnceLock::new();
    static DOUBLE: OnceLock<Regex> = OnceLock::new();
    static SINGLE: OnceLock<Regex> = OnceLock::new();
    static BRACKETS: OnceLock<Regex> = OnceLock::new();
    let text = re(&STOPS, "[。！？；]").replace_all(text, ".");
    let text = re(&COMMA, "[，]").replace_all(&text, ",");
    let text = re(&DOUBLE, "[“”]").replace_all(&text, "\"");
    let text = re(&SINGLE, "[‘’]").replace_all(&text, "'");
    let text = re(&BRACKETS, r#"[<>()\[\]"«»]+"#).replace_all(&text, "");
    txtsplit(&text, 256, 512).into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

// "c in '!?\n'" for a single character; peek() gives None where Python gives "" (and `"" in anything` is true).
fn in_set(c: Option<char>, set: &str) -> bool {
    match c {
        None => true,
        Some(c) => set.contains(c),
    }
}

pub fn txtsplit(text: &str, desired_length: usize, max_length: usize) -> Vec<String> {
    static BLANK_LINES: OnceLock<Regex> = OnceLock::new();
    static SPACES: OnceLock<Regex> = OnceLock::new();
    static AFTER_STOP: OnceLock<Regex> = OnceLock::new();
    static ONLY_MARKS: OnceLock<Regex> = OnceLock::new();
    let text = re(&BLANK_LINES, r"\n\n+").replace_all(text, "\n");
    let text = re(&SPACES, r"\s+").replace_all(&text, " ");
    let text = re(&AFTER_STOP, r"([,.?!])").replace_all(&text, "$1 ");
    let text = re(&SPACES, r"\s+").replace_all(&text, " ").into_owned();
    let chars: Vec<char> = text.chars().collect();

    let end_pos = chars.len() as i64 - 1;
    let mut pos: i64 = -1;
    let mut in_quote = false;
    let mut current: Vec<char> = Vec::new();
    let mut split_pos: Vec<i64> = Vec::new();
    let mut result: Vec<String> = Vec::new();

    // seek(delta): moves one character at a time; the quote flag flips on every move that lands on a quote mark
    let seek = |delta: i64, pos: &mut i64, in_quote: &mut bool, current: &mut Vec<char>| -> char {
        for _ in 0..delta.abs() {
            if delta < 0 {
                *pos -= 1;
                current.pop();
            } else {
                *pos += 1;
                current.push(chars[*pos as usize]);
            }
            if chars[*pos as usize] == '"' {
                *in_quote = !*in_quote;
            }
        }
        chars[*pos as usize]
    };
    let peek = |delta: i64, pos: i64| -> Option<char> {
        let p = pos + delta;
        if p < end_pos && p >= 0 {
            Some(chars[p as usize])
        } else {
            None
        }
    };

    while pos < end_pos {
        let mut c = seek(1, &mut pos, &mut in_quote, &mut current);
        if current.len() >= max_length {
            if !split_pos.is_empty() && current.len() as f64 > desired_length as f64 / 2.0 {
                let d = pos - split_pos[split_pos.len() - 1];
                seek(-d, &mut pos, &mut in_quote, &mut current);
            } else {
                while !"!?.\n ".contains(c) && pos > 0 && current.len() > desired_length {
                    c = seek(-1, &mut pos, &mut in_quote, &mut current);
                }
            }
            result.push(current.iter().collect());
            current.clear();
            split_pos.clear();
        } else if !in_quote && ("!?\n".contains(c) || (".,".contains(c) && in_set(peek(1, pos), "\n "))) {
            while pos < chars.len() as i64 - 1 && current.len() < max_length && in_set(peek(1, pos), "!?.") {
                c = seek(1, &mut pos, &mut in_quote, &mut current);
            }
            split_pos.push(pos);
            if current.len() >= desired_length {
                result.push(current.iter().collect());
                current.clear();
                split_pos.clear();
            }
        } else if in_quote && peek(1, pos) == Some('"') && in_set(peek(2, pos), "\n ") {
            seek(2, &mut pos, &mut in_quote, &mut current);
            split_pos.push(pos);
        }
    }
    result.push(current.iter().collect());
    let only_marks = re(&ONLY_MARKS, r"^[\s\.,;:!?]*$");
    result.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty() && !only_marks.is_match(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn splits_match_the_python() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/text_en.json");
        let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).expect("golden file")).expect("valid JSON");
        let mut bad = Vec::new();
        for case in &cases {
            let text = case["text"].as_str().unwrap();
            let want: Vec<&str> = case["split"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            let got = split_sentences_latin(text);
            if got != want {
                bad.push(format!("{text:?}: want {want:?}, got {got:?}"));
            }
        }
        assert!(bad.is_empty(), "{} of {} differ: {:#?}", bad.len(), cases.len(), &bad[..bad.len().min(4)]);
    }
}
