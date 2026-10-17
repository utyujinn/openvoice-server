// English number words, written to give exactly what the Python `inflect` library gives for the two calls MeloTTS makes
// (see tests/golden/numbers.json, produced from the real library):
//   number_to_words(n, andword="")                         -> `cardinal`
//   number_to_words(n, andword="", zero="oh", group=2)     -> `year_words`   (the 1001..2999 reading, "nineteen oh five")
//   number_to_words("21st")                                -> `ordinal_words`
// The algorithm is inflect's own, step by step (including the regex based digit walking), because its output has
// small quirks — a comma after "thousand" but not after "million one" — that a cleaner rewrite would not reproduce.

use regex::Regex;
use std::sync::OnceLock;

const UNIT: [&str; 10] = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const TEEN: [&str; 10] = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TEN: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const MILL: [&str; 12] = [
    " ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion", " sextillion", " septillion", " octillion", " nonillion",
    " decillion",
];

struct Args<'a> {
    andword: &'a str,
    zero: &'a str,
    one: &'a str,
}

fn tenfn(tens: usize, units: usize, mindex: usize) -> String {
    if tens != 1 {
        let hyphen = if tens != 0 && units != 0 { "-" } else { "" };
        format!("{}{}{}{}", TEN[tens], hyphen, UNIT[units], MILL[mindex])
    } else {
        format!("{}{}", TEEN[units], MILL[mindex])
    }
}

fn hundfn(args: &Args, hundreds: usize, tens: usize, units: usize, mindex: usize) -> String {
    if hundreds != 0 {
        let and = if tens != 0 || units != 0 { format!(" {} ", args.andword) } else { String::new() };
        return format!("{} hundred{}{}{}, ", UNIT[hundreds], and, tenfn(tens, units, 0), MILL[mindex]);
    }
    if tens != 0 || units != 0 {
        return format!("{}{}, ", tenfn(tens, units, 0), MILL[mindex]);
    }
    String::new()
}

// The last `k` digits of `text` when they are next to each other and nothing but non-digits follows them: what the
// Python pattern (\d){k}(?=\D*\Z) finds first. Returns their byte range and values.
fn last_digits(text: &str, k: usize) -> Option<(std::ops::Range<usize>, Vec<usize>)> {
    let bytes = text.as_bytes();
    let end = bytes.iter().rposition(|b| b.is_ascii_digit())? + 1;
    if end < k || !bytes[end - k..end].iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((end - k..end, bytes[end - k..end].iter().map(|b| (b - b'0') as usize).collect()))
}

fn digit(c: char) -> usize {
    c.to_digit(10).expect("only digits are passed here") as usize
}

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("a fixed pattern"))
}

// inflect's enword(): group 0 handles the number as a whole in words; 1, 2 and 3 read it in groups of that many digits.
fn enword(args: &Args, num: &str, group: u8) -> String {
    static DIGIT_GROUP: OnceLock<Regex> = OnceLock::new();
    static TWO_DIGITS: OnceLock<Regex> = OnceLock::new();
    static THREE_DIGITS: OnceLock<Regex> = OnceLock::new();
    let digit_group = re(&DIGIT_GROUP, r"(\d)");
    let two_digits = re(&TWO_DIGITS, r"(\d)(\d)");
    let three_digits = re(&THREE_DIGITS, r"(\d)(\d)(\d)");

    let group1 = |c: &regex::Captures| {
        let units = digit(c[1].chars().next().expect("one digit"));
        if units == 1 {
            format!(" {}, ", args.one)
        } else if units != 0 {
            format!("{}, ", UNIT[units])
        } else {
            format!(" {}, ", args.zero)
        }
    };
    let group1b = |c: &regex::Captures| {
        let units = digit(c[1].chars().next().expect("one digit"));
        if units != 0 {
            format!("{}, ", UNIT[units])
        } else {
            format!(" {}, ", args.zero)
        }
    };
    let group2 = |c: &regex::Captures| {
        let tens = digit(c[1].chars().next().expect("a digit"));
        let units = digit(c[2].chars().next().expect("a digit"));
        if tens != 0 {
            format!("{}, ", tenfn(tens, units, 0))
        } else if units != 0 {
            format!(" {} {}, ", args.zero, UNIT[units])
        } else {
            format!(" {} {}, ", args.zero, args.zero)
        }
    };
    let group3 = |c: &regex::Captures| {
        let hundreds = digit(c[1].chars().next().expect("a digit"));
        let tens = digit(c[2].chars().next().expect("a digit"));
        let units = digit(c[3].chars().next().expect("a digit"));
        let hunword = if hundreds == 1 {
            format!(" {}", args.one)
        } else if hundreds != 0 {
            UNIT[hundreds].to_string()
        } else {
            format!(" {}", args.zero)
        };
        let tenword = if tens != 0 {
            tenfn(tens, units, 0)
        } else if units != 0 {
            format!(" {} {}", args.zero, UNIT[units])
        } else {
            format!(" {} {}", args.zero, args.zero)
        };
        format!("{} {}, ", hunword, tenword)
    };

    match group {
        1 => digit_group.replace_all(num, &group1).into_owned(),
        2 => {
            let num = two_digits.replace_all(num, &group2).into_owned();
            digit_group.replacen(&num, 1, &group1b).into_owned()
        }
        3 => {
            let num = three_digits.replace_all(num, &group3).into_owned();
            let num = two_digits.replacen(&num, 1, &group2).into_owned();
            digit_group.replacen(&num, 1, &group1).into_owned()
        }
        _ => {
            let significant = num.trim().trim_start_matches('0');
            if significant.is_empty() {
                return args.zero.to_string();
            }
            if significant == "1" {
                return args.one.to_string();
            }
            let mut num = num.trim_start().trim_start_matches('0').to_string();
            let mut mill_count = 0;
            // inflect walks the number from its right end: the pattern (\d)(\d)(\d)(?=\D*\Z) matches the last three digits
            // that have only non-digits after them. The replaced part turns into words, which are not digits any more,
            // so the next match is the three digits before it.
            while let Some((range, d)) = last_digits(&num, 3) {
                let piece = hundfn(args, d[0], d[1], d[2], mill_count);
                mill_count += 1;
                num.replace_range(range, &piece);
            }
            if let Some((range, d)) = last_digits(&num, 2) {
                let piece = format!("{}, ", tenfn(d[0], d[1], mill_count));
                num.replace_range(range, &piece);
            }
            if let Some((range, d)) = last_digits(&num, 1) {
                let piece = format!("{}{}, ", UNIT[d[0]], MILL[mill_count]);
                num.replace_range(range, &piece);
            }
            num
        }
    }
}

// inflect's ordinal for a spelled number: the last word gets an ordinal ending.
fn sub_ord(value: &str) -> String {
    static ORDINAL_SUFF: OnceLock<Regex> = OnceLock::new();
    let suff = re(&ORDINAL_SUFF, r"(ty|one|two|three|five|eight|nine|twelve)\z");
    let new = suff
        .replace(value, |c: &regex::Captures| match &c[1] {
            "ty" => "tieth",
            "one" => "first",
            "two" => "second",
            "three" => "third",
            "five" => "fifth",
            "eight" => "eighth",
            "nine" => "ninth",
            "twelve" => "twelfth",
            _ => unreachable!("the pattern lists these only"),
        })
        .into_owned();
    if new == value {
        format!("{new}th")
    } else {
        new
    }
}

// inflect.number_to_words for a whole number given as a string: an optional sign, digits, and optionally an ordinal
// ending ("21st"). `group` is 0 (the usual reading), or 2 for the year-like reading.
pub fn number_to_words(num: &str, group: u8, andword: &str, zero: &str) -> String {
    static NON_DIGIT: OnceLock<Regex> = OnceLock::new();
    static WS_COMMA: OnceLock<Regex> = OnceLock::new();
    static COMMA_WORD: OnceLock<Regex> = OnceLock::new();
    static WS: OnceLock<Regex> = OnceLock::new();
    let args = Args { andword, zero, one: "one" };
    let comma = ",";

    let sign = match num.trim_start().chars().next() {
        Some('+') => "plus",
        Some('-') => "minus",
        _ => "",
    };
    // "0th" style numbers: when the whole string is just an ending, inflect treats it as zero.
    let mut num = num.to_string();
    let endings = ["th", "st", "nd", "rd"];
    if endings.contains(&num.as_str()) {
        num = zero.to_string();
    }
    let myord = num.len() >= 2 && num.is_char_boundary(num.len() - 2) && endings.contains(&&num[num.len() - 2..]);
    if myord {
        num.truncate(num.len() - 2);
    }

    // Only the first (and here the only) chunk exists: Melo never passes a decimal point.
    let chunk = re(&NON_DIGIT, r"[^0-9]").replace_all(&num, "").into_owned();
    let chunk = if chunk.is_empty() { "0".to_string() } else { chunk };
    let mut chunk = enword(&args, &chunk, group);
    if chunk.ends_with(", ") {
        chunk.truncate(chunk.len() - 2);
    }
    let mut chunk = re(&WS_COMMA, r"\s+,").replace_all(&chunk, ",").into_owned();
    if group == 0 {
        // The first chunk: " , <last word>" becomes " and <last word>" (empty andword leaves two spaces, squeezed below).
        chunk = re(&COMMA_WORD, r", (\S+)\s+\z").replace(&chunk, format!(" {andword} $1").as_str()).into_owned();
    }
    let chunk = re(&WS, r"\s+").replace_all(&chunk, " ").trim().to_string();

    let mut numchunks: Vec<String> = chunk.split(&format!("{comma} ")).map(str::to_string).collect();
    if myord {
        let last = numchunks.last_mut().expect("split gives at least one piece");
        *last = sub_ord(last);
    }
    let value = if group != 0 {
        numchunks.join(", ")
    } else {
        // inflect's _render: the first piece, then each next piece after a comma.
        let mut out = numchunks[0].clone();
        for piece in &numchunks[1..] {
            out.push_str(comma);
            out.push(' ');
            out.push_str(piece);
        }
        out
    };
    if sign.is_empty() {
        value
    } else {
        format!("{sign} {value}")
    }
}

// The three readings MeloTTS uses.
pub fn cardinal(num: &str) -> String {
    number_to_words(num, 0, "", "zero")
}

pub fn year_words(num: &str) -> String {
    number_to_words(num, 2, "", "oh")
}

pub fn ordinal_words(num: &str) -> String {
    number_to_words(num, 0, "and", "zero")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::collections::BTreeMap;

    fn golden() -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/numbers.json");
        serde_json::from_str(&std::fs::read_to_string(path).expect("the golden file is in the repository")).expect("valid JSON")
    }

    fn check(section: &str, convert: impl Fn(&str) -> String) {
        let g = golden();
        let map: BTreeMap<String, String> = serde_json::from_value(g[section].clone()).expect("a string map");
        let mut bad = Vec::new();
        for (input, want) in &map {
            let got = convert(input);
            if &got != want {
                bad.push(format!("{input}: want {want:?}, got {got:?}"));
            }
        }
        assert!(bad.is_empty(), "{} of {} differ in {section}; first: {:?}", bad.len(), map.len(), &bad[..bad.len().min(5)]);
    }

    #[test]
    fn cardinals_match_inflect() {
        check("cardinal", cardinal);
    }

    #[test]
    fn year_readings_match_inflect() {
        check("year", year_words);
    }

    #[test]
    fn readings_with_and_match_inflect() {
        check("plain_and", |n| number_to_words(n, 0, "and", "zero"));
    }

    #[test]
    fn ordinals_match_inflect() {
        check("ordinals", ordinal_words);
    }
}
