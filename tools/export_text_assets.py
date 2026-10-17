"""Writes what a Rust port of MeloTTS' English text front end needs, and golden outputs of the Python one to test it
against. Assets go to <repo>/openvoice-server/data, goldens to <repo>/openvoice-server/tests/golden."""
import json, os, struct, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = Path(r"C:\Users\Utyujin\Dev\Mutelink")
DATA = REPO / "openvoice-server" / "data"
GOLD = REPO / "openvoice-server" / "tests" / "golden"
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
sys.path.insert(0, str(ROOT / "MeloTTS"))

import numpy as np

TEXTS = [
    # plain
    "Hello, nice to meet you.", "Could you pass me that, please? Thank you so much.", "I can't hear you very well, could you say that again?",
    "Thank you for today. See you again soon, good night!", "Yes!", "Hi.", "Okay, okay, okay...", "What?! No way!", "Well... I guess so.",
    # numbers, ordinals, currency, decimals, years, times
    "I have 3 cats and 21 dogs.", "It costs $5.50 today.", "She was born in 1999 and he in 2005.", "The year 2000 was special.", "Call me at 1,234 times.",
    "That is the 3rd time, and the 22nd attempt, and the 101st try.", "Pi is about 3.14159.", "The meeting is at 3:30pm, not 12:00.", "Wake me up at 7:05 am.",
    "It weighs 100 pounds, or £20, or ¥3000.", "There are 1000000 stars.", "Room 404 is on the 4th floor.", "I scored 0 points.", "Temperatures hit -5 degrees.",
    # abbreviations
    "Mr. Smith met Dr. Jones and Mrs. Brown.", "St. John and Co. are in Ft. Worth.", "Capt. Lee and Sgt. Kim said hi.",
    # names, OOV words, brands, hyphens, apostrophes
    "Mutelink works with VRChat.", "Nyasuke said xyzzy and blorptastic.", "I'm sure you've seen it; it's amazing!", "A well-known state-of-the-art system.",
    "Don't you think we should've gone?", "Ahaha! Oh my gosh, that's so funny, I can't stop laughing!", "Hmm... let me think... okay, maybe this way? ...No, wait, that's wrong.",
    "Mmh... haah... phew... phew... haah... ugh, I'm sooo tired!", "Zzz. huh? Oh, I wasn't sleeping! I was just resting my eyes.",
    # caps, camel case, mixed
    "THIS IS LOUD", "ThisIsCamelCase and anotherOne", "NASA and the FBI use GPS.", "iPhone and YouTube are brands.",
    # punctuation and unusual characters
    "Wait; what: really? Yes, really!", "He said \u201chello\u201d and she said \u2018bye\u2019.", "Caf\u00e9 na\u00efve fa\u00e7ade.", "Price: 5\u20ac!", "100% sure, 50/50 chance.", "a, b, c, and d.", "...",
    "Tom & Jerry @ home #1", "emoji \U0001F600 test", "\u4f60\u597d hello", "line one\nline two", "   spaces   everywhere   ",
    # long
    "Well, I was thinking that maybe we could go to the park tomorrow afternoon if the weather is nice, and then get some ice cream afterwards, what do you think about that idea?",
    "I didn\u2019t know what to do. I said please kill her because it would be better than being kidnapped,\u201d Ben, whose surname CNN is not using for security concerns, said on Wednesday. \u201cIt\u2019s a nightmare. I said \u2018please kill her, don\u2019t take her there.\u2019\u201d",
    " ".join(["This is a longer passage, written so that the splitter has to cut it into several pieces."] * 8),
]


def main():
    from melo.api import TTS
    from melo import commons, utils
    from melo.text import cleaner, english, language_tone_start_map, language_id_map
    from melo.split_utils import split_sentence
    import re
    from g2p_en import G2p
    from transformers import AutoTokenizer
    import g2p_en

    DATA.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    tts = TTS(language="EN", device="cpu")
    hps = tts.hps

    # --- assets
    with open(DATA / "cmudict_melo.tsv", "w", encoding="utf8", newline="\n") as f:
        for word, syllables in english.eng_dict.items():
            f.write(word + "\t" + " - ".join(" ".join(s) for s in syllables) + "\n")
    g = G2p()
    melo_words = set(english.eng_dict.keys())
    # g2p_en looks a word up in the NLTK copy of the CMU dictionary (after cleaning its spelling, so also words MeloTTS' own dictionary
    # has under another spelling): all of it, first pronunciation.
    with open(DATA / "cmudict_extra.tsv", "w", encoding="utf8", newline="\n") as f:
        for word in sorted(g.cmu):
            f.write(word + "\t" + " ".join(g.cmu[word][0]) + "\n")
    # g2p_en network, as float32 arrays in a fixed order (names, shapes in the header line)
    npz = np.load(Path(g2p_en.__file__).parent / "checkpoint20.npz")
    order = ["enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh", "dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh", "fc_w", "fc_b"]
    header = json.dumps({k: list(npz[k].shape) for k in order})
    with open(DATA / "g2p_en.bin", "wb") as f:
        f.write(struct.pack("<I", len(header)))
        f.write(header.encode())
        for k in order:
            f.write(npz[k].astype("<f4").tobytes())
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok.backend_tokenizer.save(str(DATA / "bert_tokenizer.json"))
    cfg = {"symbols": hps.symbols, "module_symbols": list(english.symbols), "spk2id": dict(hps.data.spk2id), "add_blank": hps.data.add_blank, "sampling_rate": hps.data.sampling_rate,
           "tone_start": language_tone_start_map["EN"], "lang_id": language_id_map["EN"]}
    (DATA / "melo_en.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf8")
    print("assets written", {p.name: p.stat().st_size for p in DATA.iterdir()})

    # --- golden outputs: exactly what TTS.tts_to_file does before the neural network
    cases = []
    for text in TEXTS:
        pieces = split_sentence(text, language_str="EN")
        out_pieces = []
        for t in pieces:
            t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            norm, phones, tones, word2ph = cleaner.clean_text(t, "EN")
            tokens = tok.tokenize(norm)
            ids, tns, lang = utils.cleaned_text_to_sequence(phones, tones, "EN", tts.symbol_to_id)
            ids, tns, lang = commons.intersperse(ids, 0), commons.intersperse(tns, 0), commons.intersperse(lang, 0)
            w2p = [w * 2 for w in word2ph]
            w2p[0] += 1
            bert_ids = tok(norm)["input_ids"]
            out_pieces.append({"piece": t, "norm": norm, "tokens": tokens, "phones": phones, "tones": tones, "word2ph": word2ph, "x": ids, "tone": tns, "lang": lang,
                               "word2ph_blank": w2p, "bert_ids": bert_ids})
        cases.append({"text": text, "split": split_sentence(text, language_str="EN"), "pieces": out_pieces})
    (GOLD / "text_en.json").write_text(json.dumps(cases, ensure_ascii=False, indent=0), encoding="utf8")
    print("golden cases:", len(cases), "pieces:", sum(len(c["pieces"]) for c in cases))


if __name__ == "__main__":
    main()
