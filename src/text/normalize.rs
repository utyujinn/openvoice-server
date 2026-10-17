// MeloTTS' English text normalizing (melo/text/english.py text_normalize and english_utils/*), written to give the same
// text as the Python for the same input: lower case, times ("3:30pm"), numbers (currency, decimals, ordinals, years),
// abbreviations ("mr." and the like). The checks are against tests/golden/*.json, made by running the Python.

use super::numwords::{cardinal, ordinal_words, year_words};
use regex::{Captures, Regex};
use std::sync::OnceLock;

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("a fixed pattern"))
}

// The words are written out for at most this many digits (inflect goes up to "decillion", 36 digits); a longer run of
// digits is left as it is rather than failing the whole sentence, as the Python library would.
const MAX_DIGITS: usize = 36;

fn words_or_digits(digits: &str, convert: impl Fn(&str) -> String) -> String {
    let significant = digits.trim_start_matches(['-', '+']).trim_start_matches('0');
    if significant.len() > MAX_DIGITS {
        digits.to_string()
    } else {
        convert(digits)
    }
}

pub fn expand_time_english(text: &str) -> String {
    static TIME: OnceLock<Regex> = OnceLock::new();
    // The "a\\.m\\." alternatives are the Python source's, character for character (a backslash and any character, not a
    // dot): they never match ordinary text, so only "am" and "pm" count, and "a.m." is left to the abbreviation steps.
    let time = re(
        &TIME,
        r"(?ix)\b((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3])):([0-5][0-9])\s*(a\\.m\\.|am|pm|p\\.m\\.|a\\.m|p\\.m)?\b",
    );
    time.replace_all(text, |c: &Captures| {
        let mut hour: u32 = c[1].parse().expect("hours are digits");
        let mut past_noon = hour >= 12;
        let mut words: Vec<String> = Vec::new();
        if hour > 12 {
            hour -= 12;
        } else if hour == 0 {
            hour = 12;
            past_noon = true;
        }
        words.push(cardinal(&hour.to_string()));
        let minute: u32 = c[6].parse().expect("minutes are digits");
        if minute > 0 {
            if minute < 10 {
                words.push("oh".to_string());
            }
            words.push(cardinal(&minute.to_string()));
        }
        match c.get(7) {
            None => words.push(if past_noon { "p m" } else { "a m" }.to_string()),
            Some(marker) => {
                // list(am_pm.replace(".", "")): one entry per letter
                for letter in marker.as_str().replace('.', "").chars() {
                    words.push(letter.to_string());
                }
            }
        }
        words.join(" ")
    })
    .into_owned()
}

struct Currency {
    // false for "¥", whose table has no entry for exactly one unit
    has_one_key: bool,
    one: &'static str,
    many: &'static str,
    cent_one: Option<&'static str>,
    cent_many: &'static str,
}

fn currency(symbol: &str) -> Currency {
    match symbol {
        "$" => Currency { has_one_key: true, one: "dollar", many: "dollars", cent_one: Some("cent"), cent_many: "cents" },
        "£" => Currency { has_one_key: true, one: "pound sterling", many: "pounds sterling", cent_one: Some("penny"), cent_many: "pence" },
        // "¥" has no word for exactly one, nor for one hundredth: "yen" and "sen" are used for any count.
        _ => Currency { has_one_key: false, one: "yen", many: "yen", cent_one: None, cent_many: "sen" },
    }
}

fn expand_currency(c: &Captures) -> String {
    let unit = currency(&c[1]);
    let value = c[2].replace(',', "");
    let parts: Vec<&str> = value.split('.').collect();
    if parts.len() > 2 {
        return format!("{} {}", &c[2], unit.many);
    }
    let mut text: Vec<String> = Vec::new();
    let integer: u128 = if parts[0].is_empty() { 0 } else { parts[0].parse().unwrap_or(u128::MAX) };
    if integer > 0 {
        let word = if integer == 1 { unit.one } else { unit.many };
        text.push(format!("{} {}", parts[0].trim_start_matches('0'), word));
    }
    let fraction: u128 = if parts.len() > 1 && !parts[1].is_empty() { parts[1].parse().unwrap_or(u128::MAX) } else { 0 };
    if fraction > 0 {
        // The Python looks the unit up by fraction / 100 in a table keyed 0.01, 0.02, 1 and 2, so a fraction of exactly
        // 100 or 200 lands on the keys for one and two whole units.
        let word = match fraction {
            1 => unit.cent_one.unwrap_or(unit.cent_many),
            100 if unit.has_one_key => unit.one,
            200 => unit.many,
            _ => unit.cent_many,
        };
        text.push(format!("{} {}", parts[1].trim_start_matches('0'), word));
    }
    if text.is_empty() {
        return format!("zero {}", unit.many);
    }
    text.join(" ")
}

fn expand_number(digits: &str) -> String {
    // "-5", "007", "2005": the sign and leading zeros are read as a number, so `int("007")` is 7.
    let negative = digits.starts_with('-');
    let body = digits.trim_start_matches('-');
    let significant = body.trim_start_matches('0');
    let numeric = if significant.is_empty() { "0" } else { significant };
    let value: Option<u128> = numeric.parse().ok();
    let signed = if negative && numeric != "0" { format!("-{numeric}") } else { numeric.to_string() };
    if let (Some(n), false) = (value, negative) {
        if 1000 < n && n < 3000 {
            if n == 2000 {
                return "two thousand".to_string();
            }
            if 2000 < n && n < 2010 {
                return format!("two thousand {}", cardinal(&(n % 100).to_string()));
            }
            if n % 100 == 0 {
                return format!("{} hundred", cardinal(&(n / 100).to_string()));
            }
            return year_words(&n.to_string()).replace(", ", " ");
        }
    }
    words_or_digits(&signed, cardinal)
}

pub fn normalize_numbers(text: &str) -> String {
    static COMMA_NUMBER: OnceLock<Regex> = OnceLock::new();
    static CURRENCY: OnceLock<Regex> = OnceLock::new();
    static DECIMAL: OnceLock<Regex> = OnceLock::new();
    static ORDINAL: OnceLock<Regex> = OnceLock::new();
    static NUMBER: OnceLock<Regex> = OnceLock::new();
    let text = re(&COMMA_NUMBER, r"([0-9][0-9,]+[0-9])").replace_all(text, |c: &Captures| c[1].replace(',', ""));
    let text = re(&CURRENCY, r"(£|\$|¥)([0-9,\.]*[0-9]+)").replace_all(&text, |c: &Captures| expand_currency(c));
    let text = re(&DECIMAL, r"([0-9]+\.[0-9]+)").replace_all(&text, |c: &Captures| c[1].replace('.', " point "));
    let text = re(&ORDINAL, r"[0-9]+(?:st|nd|rd|th)").replace_all(&text, |c: &Captures| {
        let digits: String = c[0].chars().take_while(|ch| ch.is_ascii_digit()).collect();
        // too many digits to write out: the whole thing, ending included, is left as it was
        if digits.trim_start_matches('0').len() > MAX_DIGITS {
            c[0].to_string()
        } else {
            ordinal_words(&c[0])
        }
    });
    re(&NUMBER, r"-?[0-9]+").replace_all(&text, |c: &Captures| expand_number(&c[0])).into_owned()
}

pub fn expand_abbreviations(text: &str) -> String {
    static ABBREVIATIONS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    let list = ABBREVIATIONS.get_or_init(|| {
        [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
        .iter()
        .map(|(short, long)| (Regex::new(&format!(r"(?i)\b{short}\.")).expect("a fixed pattern"), *long))
        .collect()
    });
    let mut text = text.to_string();
    for (regex, replacement) in list {
        text = regex.replace_all(&text, *replacement).into_owned();
    }
    text
}

pub fn text_normalize(text: &str) -> String {
    let text = text.to_lowercase();
    let text = expand_time_english(&text);
    let text = normalize_numbers(&text);
    expand_abbreviations(&text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn steps_match_the_python() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/numbers.json");
        let g: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("golden file")).expect("valid JSON");
        let mut bad = Vec::new();
        for case in g["normalize"].as_array().expect("a list") {
            let input = case["in"].as_str().unwrap();
            for (step, got) in [("numbers", normalize_numbers(input)), ("time", expand_time_english(input)), ("abbr", expand_abbreviations(input))] {
                let want = case[step].as_str().unwrap();
                if got != want {
                    bad.push(format!("{step}({input:?}): want {want:?}, got {got:?}"));
                }
            }
        }
        assert!(bad.is_empty(), "{} differ: {:#?}", bad.len(), &bad[..bad.len().min(6)]);
    }

    #[test]
    fn a_very_long_run_of_digits_is_left_alone() {
        let digits = "9".repeat(50);
        assert_eq!(normalize_numbers(&digits), digits);
    }
}
