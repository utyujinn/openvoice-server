"""A second, larger golden file of random English-ish texts (words from the dictionary, numbers, punctuation, quotes,
odd characters, long passages), run through the real MeloTTS front end, to compare the Rust port against."""
import json, os, random, re, sys
from pathlib import Path

ROOT = Path(r"C:\Users\Utyujin\Dev\tts-test\lang_spike\openvoice")
OUT = Path(r"C:\Users\Utyujin\Dev\Mutelink\openvoice-server\tests\golden\text_en_fuzz.json")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
sys.path.insert(0, str(ROOT / "MeloTTS"))


def main():
    from melo.api import TTS
    from melo import commons, utils
    from melo.text import cleaner, english
    from melo.split_utils import split_sentence
    from transformers import AutoTokenizer

    tts = TTS(language="EN", device="cpu")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    rnd = random.Random(20261018)
    words = [w.lower() for w in english.eng_dict if w.isalpha()]
    oov = ["mutelink", "vrchat", "nyasuke", "xyzzy", "blorptastic", "qwertyuiop", "zxcvbn", "kawaii", "senpai", "ramen", "uwu", "frobnicate", "quux", "gigachad"]
    marks = [",", ".", "!", "?", ";", ":", "...", " -", "\u2014", "(", ")", "\"", "'", "\u201c", "\u201d", "\u2018", "\u2019", "#", "@", "&", "%", "/", "\n", "\t"]
    numbers = ["1", "7", "12", "99", "100", "101", "1,000", "2,500", "1999", "2005", "2000", "3.14", "0.5", "$5", "$5.50", "\u00a33", "\u00a52000", "3rd", "21st", "1:30pm", "12:00", "7:05 am", "-4", "007", "10000000"]
    odd = ["caf\u00e9", "na\u00efve", "\u4f60\u597d", "\U0001F600", "\u00fc", "\u00df", "\u0130stanbul", "e\u0301", "\u2026", "\u00bd"]
    texts = []
    for _ in range(700):
        parts = []
        for _ in range(rnd.randrange(1, 40)):
            r = rnd.random()
            if r < 0.62:
                w = rnd.choice(words)
                parts.append(w.capitalize() if rnd.random() < 0.15 else (w.upper() if rnd.random() < 0.05 else w))
            elif r < 0.72:
                parts.append(rnd.choice(oov))
            elif r < 0.82:
                parts.append(rnd.choice(numbers))
            elif r < 0.9:
                parts.append(rnd.choice(marks))
            elif r < 0.94:
                parts.append(rnd.choice(odd))
            else:
                parts.append(rnd.choice(words).capitalize() + rnd.choice(words).capitalize())
        sep = rnd.choice([" ", " ", " ", "  "])
        texts.append(sep.join(parts))
    # long passages that force the splitter's length rules
    for n in (600, 900, 1500, 2600):
        sentence_list = []
        while sum(len(s) + 1 for s in sentence_list) < n:
            sentence_list.append(" ".join(rnd.choice(words) for _ in range(rnd.randrange(3, 30))) + rnd.choice([".", "!", "?", ",", ".", "."]))
        texts.append(" ".join(sentence_list))
    texts.append("".join(rnd.choice(words) for _ in range(400)))  # one very long word
    texts.append("quote \"" + " ".join(rnd.choice(words) for _ in range(150)) + ".\" Then more. " + " ".join(rnd.choice(words) for _ in range(120)))

    cases, skipped = [], 0
    for text in texts:
        try:
            pieces = split_sentence(text, language_str="EN")
            out = []
            for t in pieces:
                t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
                norm, phones, tones, word2ph = cleaner.clean_text(t, "EN")
                ids, tns, lang = utils.cleaned_text_to_sequence(phones, tones, "EN", tts.symbol_to_id)
                ids, tns, lang = commons.intersperse(ids, 0), commons.intersperse(tns, 0), commons.intersperse(lang, 0)
                w2p = [w * 2 for w in word2ph]
                w2p[0] += 1
                out.append({"piece": t, "norm": norm, "x": ids, "tone": tns, "lang": lang, "word2ph_blank": w2p, "bert_ids": tok(norm)["input_ids"]})
            cases.append({"text": text, "split": pieces, "pieces": out})
        except Exception as e:  # the Python itself fails on some inputs (e.g. a text starting with "#"); the Rust must not crash, but there is nothing to compare
            skipped += 1
    OUT.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf8")
    print("fuzz cases:", len(cases), "pieces:", sum(len(c["pieces"]) for c in cases), "python failures skipped:", skipped)


if __name__ == "__main__":
    main()
