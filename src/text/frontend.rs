// The whole English text front end of MeloTTS: text -> pieces (sentences) -> for each piece the model inputs (symbol ids,
// tones, language ids) and the BERT inputs (token ids and how many phonemes each token covers). Everything here is checked
// against the Python on the golden sentences.

use super::g2p::{g2p, Dictionaries};
use super::normalize::text_normalize;
use super::split::split_sentences_latin;
use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct MeloConfig {
    symbols: Vec<String>,
    tone_start: i64,
    lang_id: i64,
    add_blank: bool,
    sampling_rate: u32,
    spk2id: HashMap<String, i64>,
}

pub struct Frontend {
    dicts: Dictionaries,
    tokenizer: Tokenizer,
    symbol_to_id: HashMap<String, i64>,
    symbol_set: HashSet<String>,
    tone_start: i64,
    lang_id: i64,
    add_blank: bool,
    pub sampling_rate: u32,
    pub speakers: HashMap<String, i64>,
}

// What one piece of text turns into: the inputs of the base voice model (with a blank between every two symbols), and of BERT.
#[derive(Debug, PartialEq)]
pub struct Encoded {
    pub norm: String,
    pub x: Vec<i64>,
    pub tone: Vec<i64>,
    pub lang: Vec<i64>,
    // phonemes each BERT token (with [CLS] and [SEP]) is spread over, counted with the blanks
    pub word2ph: Vec<usize>,
    pub bert_ids: Vec<u32>,
}

// BERT's position table has 512 entries, [CLS] and [SEP] among them: a longer text is an error inside the model. The sentence splitter
// counts characters, but numbers are written out in words ("1234567" is a dozen tokens), so a piece can still be too long; such
// a piece is cut again at word boundaries until each part fits with room to spare.
const MAX_BERT_POSITIONS: usize = 512;
const MAX_PIECE_TOKENS: usize = 480;

pub fn camel_split(text: &str) -> String {
    static CAMEL: OnceLock<Regex> = OnceLock::new();
    CAMEL.get_or_init(|| Regex::new(r"([a-z])([A-Z])").expect("a fixed pattern")).replace_all(text, "$1 $2").into_owned()
}

fn intersperse<T: Clone>(items: &[T], blank: T) -> Vec<T> {
    let mut out = Vec::with_capacity(items.len() * 2 + 1);
    out.push(blank.clone());
    for item in items {
        out.push(item.clone());
        out.push(blank.clone());
    }
    out
}

impl Frontend {
    pub fn load(dir: &Path) -> anyhow::Result<Frontend> {
        let config: MeloConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("melo_en.json"))?)?;
        let tokenizer = Tokenizer::from_file(dir.join("bert_tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Frontend {
            dicts: Dictionaries::load(dir)?,
            tokenizer,
            symbol_to_id: config.symbols.iter().enumerate().map(|(i, s)| (s.clone(), i as i64)).collect(),
            symbol_set: config.symbols.iter().cloned().collect(),
            tone_start: config.tone_start,
            lang_id: config.lang_id,
            add_blank: config.add_blank,
            sampling_rate: config.sampling_rate,
            speakers: config.spk2id,
        })
    }

    // The text cut into the pieces the base voice model is run on one after another.
    pub fn pieces(&self, text: &str) -> Vec<String> {
        split_sentences_latin(text).iter().map(|piece| camel_split(piece)).flat_map(|piece| self.fit_to_bert(piece)).collect()
    }

    // How many BERT tokens (with [CLS] and [SEP]) the piece comes to once normalized; "too many" if it cannot be tokenized.
    fn token_count(&self, piece: &str) -> usize {
        self.tokenizer.encode(text_normalize(piece).as_str(), true).map(|e| e.get_ids().len()).unwrap_or(usize::MAX)
    }

    fn fit_to_bert(&self, piece: String) -> Vec<String> {
        if self.token_count(&piece) <= MAX_PIECE_TOKENS {
            return vec![piece];
        }
        let mut parts: Vec<String> = Vec::new();
        let mut current = String::new();
        for word in piece.split_whitespace() {
            let candidate = if current.is_empty() { word.to_string() } else { format!("{current} {word}") };
            if self.token_count(&candidate) <= MAX_PIECE_TOKENS {
                current = candidate;
                continue;
            }
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            if self.token_count(word) <= MAX_PIECE_TOKENS {
                current = word.to_string();
            } else {
                // one "word" that is too long by itself (a pasted string without spaces): cut it by characters
                let letters: Vec<char> = word.chars().collect();
                parts.extend(letters.chunks(60).map(|chunk| chunk.iter().collect::<String>()));
            }
        }
        if !current.is_empty() {
            parts.push(current);
        }
        parts
    }

    pub fn encode(&self, piece: &str) -> anyhow::Result<Encoded> {
        let norm = text_normalize(piece);
        let encoding = self.tokenizer.encode(norm.as_str(), true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let all_tokens = encoding.get_tokens();
        anyhow::ensure!(all_tokens.len() >= 2 && all_tokens.len() <= MAX_BERT_POSITIONS, "the text is {} tokens long, more than the model takes ({MAX_BERT_POSITIONS})", all_tokens.len());
        // without [CLS] and [SEP]: the phonemes are worked out for the words themselves
        let tokens: Vec<String> = all_tokens[1..all_tokens.len() - 1].to_vec();
        let phonemes = g2p(&tokens, &self.dicts, &self.symbol_set);
        let ids: Vec<i64> = phonemes.phones.iter().map(|p| self.symbol_to_id.get(p).copied().unwrap_or(0)).collect();
        let tones: Vec<i64> = phonemes.tones.iter().map(|t| t + self.tone_start).collect();
        let langs: Vec<i64> = vec![self.lang_id; ids.len()];
        let mut word2ph = phonemes.word2ph;
        let (x, tone, lang) = if self.add_blank {
            for w in word2ph.iter_mut() {
                *w *= 2;
            }
            word2ph[0] += 1;
            (intersperse(&ids, 0), intersperse(&tones, 0), intersperse(&langs, 0))
        } else {
            (ids, tones, langs)
        };
        let bert_ids = encoding.get_ids().to_vec();
        anyhow::ensure!(bert_ids.len() == word2ph.len(), "BERT has {} tokens but the phoneme groups are {}", bert_ids.len(), word2ph.len());
        Ok(Encoded { norm, x, tone, lang, word2ph, bert_ids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn data_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))
    }

    fn ints(v: &Value) -> Vec<i64> {
        v.as_array().unwrap().iter().map(|n| n.as_i64().unwrap()).collect()
    }

    // Compares every piece of a golden file with the Python; returns how many pieces were checked, or None when the file is
    // not there (the large random one is not kept in the repository, see tests/golden/README.md).
    fn compare_with(name: &str) -> Option<usize> {
        let frontend = Frontend::load(&data_dir()).expect("the data files are in the repository");
        let path = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
        let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).ok()?).unwrap();
        let mut bad = Vec::new();
        let mut checked = 0;
        for case in &cases {
            let text = case["text"].as_str().unwrap();
            let pieces = frontend.pieces(text);
            let want_pieces = case["pieces"].as_array().unwrap();
            if pieces.len() != want_pieces.len() {
                bad.push(format!("{text:?}: {} pieces, want {}", pieces.len(), want_pieces.len()));
                continue;
            }
            for (piece, want) in pieces.iter().zip(want_pieces) {
                checked += 1;
                if piece != want["piece"].as_str().unwrap() {
                    bad.push(format!("piece text: {piece:?} vs {:?}", want["piece"]));
                    continue;
                }
                let got = match frontend.encode(piece) {
                    Ok(e) => e,
                    Err(e) => {
                        bad.push(format!("{piece:?}: {e}"));
                        continue;
                    }
                };
                let mut problems = Vec::new();
                if got.norm != want["norm"].as_str().unwrap() {
                    problems.push(format!("norm {:?} vs {:?}", got.norm, want["norm"]));
                }
                if got.x != ints(&want["x"]) {
                    problems.push(format!("x {:?} vs {:?}", got.x, ints(&want["x"])));
                }
                if got.tone != ints(&want["tone"]) {
                    problems.push("tone".to_string());
                }
                if got.lang != ints(&want["lang"]) {
                    problems.push("lang".to_string());
                }
                if got.word2ph.iter().map(|&w| w as i64).collect::<Vec<_>>() != ints(&want["word2ph_blank"]) {
                    problems.push(format!("word2ph {:?} vs {:?}", got.word2ph, want["word2ph_blank"]));
                }
                if got.bert_ids.iter().map(|&w| w as i64).collect::<Vec<_>>() != ints(&want["bert_ids"]) {
                    problems.push("bert_ids".to_string());
                }
                if !problems.is_empty() {
                    bad.push(format!("{piece:?}: {}", problems.join("; ")));
                }
            }
        }
        assert!(bad.is_empty(), "{} of {} pieces differ in {name}: {:#?}", bad.len(), checked, &bad[..bad.len().min(5)]);
        Some(checked)
    }

    #[test]
    fn every_golden_piece_matches_the_python() {
        assert!(compare_with("text_en.json").expect("the small golden file is in the repository") >= 50);
    }

    // Anything a user can type reaches the front end, in a process that must not die: no input may panic. A tiny deterministic
    // generator (no dependency) writes strings from many scripts, combining marks, control characters, digits of other
    // scripts, and long runs.
    #[test]
    fn arbitrary_text_never_panics() {
        let frontend = Frontend::load(&data_dir()).expect("the data files are in the repository");
        let mut alphabet: Vec<char> = "abcXYZ019 .,!?;:'\"-#@&%$()[]<>\n\t\r/\\_*+=~^|`".chars().collect();
        alphabet.extend(['£', '¥', '€', 'é', 'ü', 'ß', 'ı', 'İ', 'α', 'б', 'я', 'א', 'ا', 'अ', 'ก', 'あ', 'ア', '你', '好', '한', '글']);
        alphabet.extend(['\u{1f600}', '\u{1f468}', '\u{200d}', '\u{301}', '\u{300}', '\u{200b}', '\u{feff}', '\u{202e}', '\u{0}', '\u{7}', '\u{1b}', '\u{2028}', '\u{fffd}', '\u{10ffff}']);
        alphabet.extend(['１', '١', '①', '²', '½', 'Ⅷ']);
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut texts: Vec<String> = ["", " ", ".", "...", "#", "##", "#hash", "\"", "\"\"\"", "-", "$", "$.", "$.5", "12:99", "99:99pm", "1st2nd3rd", "\n\n\n"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        texts.push("a".repeat(5000));
        texts.push("1".repeat(100));
        texts.push(format!("{}th", "9".repeat(40)));
        texts.push("x\"".repeat(400));
        for _ in 0..400 {
            let length = (next() % 400) as usize;
            texts.push((0..length).map(|_| alphabet[(next() % alphabet.len() as u64) as usize]).collect());
        }
        for _ in 0..8 {
            let length = (next() % 6000) as usize;
            texts.push((0..length).map(|_| alphabet[(next() % alphabet.len() as u64) as usize]).collect());
        }
        let mut errors = 0;
        for text in &texts {
            for piece in frontend.pieces(text) {
                if frontend.encode(&piece).is_err() {
                    errors += 1;
                }
            }
        }
        // An error is an acceptable answer (the app shows it); a panic is not. Most text must still work.
        assert!(errors < texts.len() / 5, "{errors} of {} texts gave an error", texts.len());
    }

    // Long runs of numbers become many tokens: no piece may come out longer than BERT takes, however the text is made.
    #[test]
    fn pieces_always_fit_the_bert_positions() {
        let frontend = Frontend::load(&data_dir()).expect("the data files are in the repository");
        let numbers: String = (0..400).map(|i| format!("{} ", 1_000_003 * (i + 1))).collect();
        let texts = [numbers, "9".repeat(3000), "12345678901234567890 ".repeat(60), "a b c d e f g h i j ".repeat(120), "x".repeat(2000)];
        for text in &texts {
            let pieces = frontend.pieces(text);
            assert!(!pieces.is_empty());
            for piece in pieces {
                let encoded = frontend.encode(&piece).unwrap_or_else(|e| panic!("{e} for a piece of {} characters", piece.chars().count()));
                assert!(encoded.bert_ids.len() <= 512);
            }
        }
    }

    #[test]
    fn random_texts_match_the_python() {
        match compare_with("text_en_fuzz.json") {
            Some(n) => assert!(n > 500),
            None => eprintln!("text_en_fuzz.json is not here; make it with the script in tests/golden/README.md"),
        }
    }
}
