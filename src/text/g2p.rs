// English text to phonemes and tones the way MeloTTS does it (melo/text/english.py g2p): the text is cut into words by the
// BERT word-piece tokenizer, each word is looked up in the CMU dictionary (with its syllables), and a word that is not
// there is spelled out by g2p_en's small neural network. The output is checked against the Python on every golden
// sentence (tests/golden/text_en.json).

use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

// The phonemes MeloTTS' `arpa` set holds (stress digit forms and a few bare ones): only these are split into a lower case
// sound and a tone. Anything else coming from the neural network is kept as it is, with tone 0 (and usually ends up as "UNK").
const ARPA: [&str; 71] = [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0",
    "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
];

// g2p_en's decoder output alphabet, in the order of its checkpoint.
const G2P_PHONEMES: [&str; 74] = [
    "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B",
    "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N",
    "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
];

struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

// g2p_en's grapheme-to-phoneme network: a GRU encoder over the letters and a GRU decoder that writes phonemes greedily
// (at most 20 of them). It is a direct port of its numpy code.
pub struct G2pNet {
    enc_emb: Matrix,
    enc_w_ih: Matrix,
    enc_w_hh: Matrix,
    enc_b_ih: Vec<f32>,
    enc_b_hh: Vec<f32>,
    dec_emb: Matrix,
    dec_w_ih: Matrix,
    dec_w_hh: Matrix,
    dec_b_ih: Vec<f32>,
    dec_b_hh: Vec<f32>,
    fc_w: Matrix,
    fc_b: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn matvec(w: &Matrix, x: &[f32], bias: &[f32]) -> Vec<f32> {
    (0..w.rows).map(|r| w.row(r).iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + bias[r]).collect()
}

fn gru_cell(x: &[f32], h: &[f32], w_ih: &Matrix, w_hh: &Matrix, b_ih: &[f32], b_hh: &[f32]) -> Vec<f32> {
    let ih = matvec(w_ih, x, b_ih);
    let hh = matvec(w_hh, h, b_hh);
    let size = h.len();
    let mut out = vec![0.0f32; size];
    for i in 0..size {
        let r = sigmoid(ih[i] + hh[i]);
        let z = sigmoid(ih[size + i] + hh[size + i]);
        let n = (ih[2 * size + i] + r * hh[2 * size + i]).tanh();
        out[i] = (1.0 - z) * n + z * h[i];
    }
    out
}

impl G2pNet {
    pub fn load(path: &Path) -> anyhow::Result<G2pNet> {
        let bytes = std::fs::read(path)?;
        let too_short = || anyhow::anyhow!("{} is cut short or damaged", path.display());
        let header_len = u32::from_le_bytes(bytes.get(0..4).ok_or_else(too_short)?.try_into()?) as usize;
        let header: serde_json::Value = serde_json::from_slice(bytes.get(4..4 + header_len).ok_or_else(too_short)?)?;
        let mut offset = 4 + header_len;
        let mut take = |name: &str| -> anyhow::Result<Matrix> {
            let shape: Vec<usize> = serde_json::from_value(header[name].clone())?;
            let (rows, cols) = match shape.as_slice() {
                [rows, cols] => (*rows, *cols),
                [len] => (1, *len),
                _ => anyhow::bail!("{name} has a shape that is not 1 or 2 dimensional"),
            };
            let count = rows.checked_mul(cols).ok_or_else(too_short)?;
            let end = offset.checked_add(count.checked_mul(4).ok_or_else(too_short)?).ok_or_else(too_short)?;
            let data: Vec<f32> = bytes.get(offset..end).ok_or_else(too_short)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            offset = end;
            Ok(Matrix { rows, cols, data })
        };
        let enc_emb = take("enc_emb")?;
        let enc_w_ih = take("enc_w_ih")?;
        let enc_w_hh = take("enc_w_hh")?;
        let enc_b_ih = take("enc_b_ih")?.data;
        let enc_b_hh = take("enc_b_hh")?.data;
        let dec_emb = take("dec_emb")?;
        let dec_w_ih = take("dec_w_ih")?;
        let dec_w_hh = take("dec_w_hh")?;
        let dec_b_ih = take("dec_b_ih")?.data;
        let dec_b_hh = take("dec_b_hh")?.data;
        let fc_w = take("fc_w")?;
        let fc_b = take("fc_b")?.data;
        // The sizes must fit each other, or the network indexes outside its own arrays while running.
        let hidden = enc_w_hh.cols;
        anyhow::ensure!(
            hidden > 0
                && enc_w_ih.rows == 3 * hidden
                && enc_w_hh.rows == 3 * hidden
                && enc_w_ih.cols == enc_emb.cols
                && dec_w_ih.rows == 3 * hidden
                && dec_w_hh.rows == 3 * hidden
                && dec_w_hh.cols == hidden
                && dec_w_ih.cols == dec_emb.cols
                && fc_w.cols == hidden
                && fc_w.rows == G2P_PHONEMES.len()
                && fc_b.len() == fc_w.rows
                && enc_b_ih.len() == 3 * hidden
                && enc_b_hh.len() == 3 * hidden
                && dec_b_ih.len() == 3 * hidden
                && dec_b_hh.len() == 3 * hidden
                && enc_emb.rows >= 29
                && dec_emb.rows >= G2P_PHONEMES.len(),
            "g2p_en.bin does not hold the network this code expects"
        );
        Ok(G2pNet { enc_emb, enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh, dec_emb, dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh, fc_w, fc_b })
    }

    // graphemes: <pad>, <unk>, </s>, a..z
    fn grapheme_index(c: char) -> usize {
        match c {
            'a'..='z' => 3 + (c as usize - 'a' as usize),
            _ => 1,
        }
    }

    pub fn predict(&self, word: &str) -> Vec<&'static str> {
        let mut indices: Vec<usize> = word.chars().map(Self::grapheme_index).collect();
        indices.push(2); // </s>
        let hidden = self.enc_w_hh.cols;
        let mut h = vec![0.0f32; hidden];
        for &i in &indices {
            h = gru_cell(self.enc_emb.row(i), &h, &self.enc_w_ih, &self.enc_w_hh, &self.enc_b_ih, &self.enc_b_hh);
        }
        let mut dec = self.dec_emb.row(2).to_vec(); // <s>
        let mut phonemes = Vec::new();
        for _ in 0..20 {
            h = gru_cell(&dec, &h, &self.dec_w_ih, &self.dec_w_hh, &self.dec_b_ih, &self.dec_b_hh);
            let logits = matvec(&self.fc_w, &h, &self.fc_b);
            let (best, _) = logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
            if best == 3 {
                break;
            }
            phonemes.push(*G2P_PHONEMES.get(best).unwrap_or(&"<unk>"));
            dec = self.dec_emb.row(best).to_vec();
        }
        phonemes
    }
}

pub struct Dictionaries {
    // The CMU dictionary as MeloTTS ships it: WORD -> syllables -> phonemes with stress digits
    pub melo: HashMap<String, Vec<Vec<String>>>,
    // Words only the NLTK copy of the CMU dictionary has (g2p_en uses it before guessing): lower case word -> phonemes
    pub extra: HashMap<String, Vec<String>>,
    pub net: G2pNet,
}

impl Dictionaries {
    pub fn load(dir: &Path) -> anyhow::Result<Dictionaries> {
        let mut melo = HashMap::with_capacity(130_000);
        for line in std::fs::read_to_string(dir.join("cmudict_melo.tsv"))?.lines() {
            if let Some((word, rest)) = line.split_once('\t') {
                melo.insert(word.to_string(), rest.split(" - ").map(|syllable| syllable.split(' ').map(str::to_string).collect()).collect());
            }
        }
        let mut extra = HashMap::new();
        for line in std::fs::read_to_string(dir.join("cmudict_extra.tsv"))?.lines() {
            if let Some((word, rest)) = line.split_once('\t') {
                extra.insert(word.to_string(), rest.split(' ').map(str::to_string).collect());
            }
        }
        Ok(Dictionaries { melo, extra, net: G2pNet::load(&dir.join("g2p_en.bin"))? })
    }

    // g2p_en's G2p()(word) for one token of the word-piece tokenizer, with its homograph step left out (the homographs are
    // all in the MeloTTS dictionary, which is asked first, except 18 rare plurals): clean the spelling, then the NLTK
    // dictionary, then the network. A token without a letter (punctuation) is returned as it is.
    fn spell_out(&self, token: &str) -> Vec<String> {
        static NOT_ALLOWED: OnceLock<Regex> = OnceLock::new();
        let cleaned: String = token.nfd().filter(|c| !is_combining_mark(*c)).collect::<String>().to_lowercase();
        let cleaned = NOT_ALLOWED.get_or_init(|| Regex::new(r"[^ a-z'.,?!\-]").expect("a fixed pattern")).replace_all(&cleaned, "").into_owned();
        let cleaned = cleaned.replace("i.e.", "that is").replace("e.g.", "for example");
        let mut phones: Vec<String> = Vec::new();
        for word in cleaned.split_whitespace() {
            if !word.chars().any(|c| c.is_ascii_lowercase()) {
                phones.push(word.to_string());
            } else if let Some(known) = self.extra.get(word) {
                phones.extend(known.iter().cloned());
            } else {
                phones.extend(self.net.predict(word).into_iter().map(str::to_string));
            }
        }
        phones
    }
}

// "AH0" -> ("ah", 1); "S" -> ("s", 0): the tone is the stress digit plus one.
fn refine_ph(phoneme: &str) -> (String, i64) {
    match phoneme.chars().last() {
        Some(last) if last.is_ascii_digit() => (phoneme[..phoneme.len() - 1].to_lowercase(), last.to_digit(10).expect("a digit") as i64 + 1),
        _ => (phoneme.to_lowercase(), 0),
    }
}

fn post_replace_ph(phoneme: &str, symbols: &std::collections::HashSet<String>) -> String {
    let mapped = match phoneme {
        "：" | "；" | "，" | "·" | "、" => ",",
        "。" => ".",
        "！" => "!",
        "？" => "?",
        "\n" => ".",
        "..." => "…",
        "v" => "V",
        other => other,
    };
    if symbols.contains(mapped) {
        mapped.to_string()
    } else {
        "UNK".to_string()
    }
}

// melo.text.japanese.distribute_phone: n_phone sounds spread over n_word pieces, the first piece with the fewest taking the next.
fn distribute_phone(n_phone: usize, n_word: usize) -> Vec<usize> {
    let mut per_word = vec![0usize; n_word];
    for _ in 0..n_phone {
        let min = *per_word.iter().min().expect("at least one word");
        let index = per_word.iter().position(|&n| n == min).expect("the minimum is in the list");
        per_word[index] += 1;
    }
    per_word
}

pub struct Phonemes {
    pub phones: Vec<String>,
    pub tones: Vec<i64>,
    pub word2ph: Vec<usize>,
}

// `tokens` are the word-piece tokens of the normalized text (without the special tokens).
pub fn g2p(tokens: &[String], dicts: &Dictionaries, symbols: &std::collections::HashSet<String>) -> Phonemes {
    // A token starting with "#" continues the previous word ("##ing"); a lone "#" is such a token too, and joins the
    // previous group as an empty piece, exactly as in the Python.
    let mut groups: Vec<Vec<String>> = Vec::new();
    for token in tokens {
        if !token.starts_with('#') || groups.is_empty() {
            groups.push(vec![token.clone()]);
        } else {
            groups.last_mut().expect("checked").push(token.replace('#', ""));
        }
    }
    let mut phones: Vec<String> = Vec::new();
    let mut tones: Vec<i64> = Vec::new();
    let mut word2ph: Vec<usize> = Vec::new();
    for group in &groups {
        let word: String = group.concat();
        let mut phone_len = 0;
        if let Some(syllables) = dicts.melo.get(&word.to_uppercase()) {
            for phoneme in syllables.iter().flatten() {
                let (p, t) = refine_ph(phoneme);
                phones.push(p);
                tones.push(t);
                phone_len += 1;
            }
        } else {
            for phoneme in dicts.spell_out(&word).into_iter().filter(|p| p != " ") {
                if ARPA.contains(&phoneme.as_str()) {
                    let (p, t) = refine_ph(&phoneme);
                    phones.push(p);
                    tones.push(t);
                } else {
                    phones.push(phoneme);
                    tones.push(0);
                }
                phone_len += 1;
            }
        }
        word2ph.extend(distribute_phone(phone_len, group.len()));
    }
    let mut phones: Vec<String> = phones.iter().map(|p| post_replace_ph(p, symbols)).collect();
    phones.insert(0, "_".to_string());
    phones.push("_".to_string());
    tones.insert(0, 0);
    tones.push(0);
    word2ph.insert(0, 1);
    word2ph.push(1);
    Phonemes { phones, tones, word2ph }
}
