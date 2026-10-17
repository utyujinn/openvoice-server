"""Golden data for the Rust port of MeloTTS' English number / time / abbreviation normalizing (which uses `inflect`)."""
import json, os, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLD = Path(r"C:\Users\Utyujin\Dev\Mutelink") / "openvoice-server" / "tests" / "golden"
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
sys.path.insert(0, str(ROOT / "MeloTTS"))


def main():
    import inflect
    from melo.text.english_utils.number_norm import normalize_numbers
    from melo.text.english_utils.time_norm import expand_time_english
    from melo.text.english_utils.abbreviations import expand_abbreviations

    e = inflect.engine()
    rnd = random.Random(7)
    ints = list(range(-30, 3100)) + [rnd.randrange(3100, 10**6) for _ in range(400)] + [rnd.randrange(10**6, 10**15) for _ in range(300)] + [10**k for k in range(3, 22)] + [10**k + 1 for k in range(3, 22)]
    cardinal = {str(n): e.number_to_words(n, andword="") for n in ints}
    year = {str(n): e.number_to_words(n, andword="", zero="oh", group=2) for n in range(1001, 3000)}
    plain_and = {str(n): e.number_to_words(n) for n in list(range(0, 400)) + [rnd.randrange(400, 10**7) for _ in range(150)]}
    ordinals = {}
    for n in list(range(0, 400)) + [rnd.randrange(400, 10**6) for _ in range(150)]:
        for suf in ("st", "nd", "rd", "th"):
            s = f"{n}{suf}"
            ordinals[s] = e.number_to_words(s)
    texts = [
        "i have 3 cats and 21 dogs.", "it costs $5.50 today.", "$0.01 and $0.02 and $1 and $2 and $1.01", "£3 and £1.50 and £0.99", "\u00a52000 and \u00a50.5", "\u20ac5 is not matched",
        "in 1999 and 2005, 2000, 2010, 1900, 1001, 2999, 3000, 999, 1000, 100, 101", "1,234 and 12,345,678 and 1,2", "3.14159 and 0.5 and 10.01 and 1.2.3", "the 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 101st, 1000th",
        "-5 and -0 and 5-3 and a-1", "12345678901234567890", "007 and 0010", "3:30pm", "12:00", "7:05 am", "0:00", "23:59 p.m.", "1:00 a.m.", "at 12:30 and 9:15pm and 10:45 AM",
        "mr. smith and dr. jones and mrs. brown", "st. john co. ft. worth capt. lee sgt. kim", "mr smith (no dot)", "Mr. Smith", "MR. SMITH", "gen. lt. col. rev. hon. esq. ltd. jr. maj. drs.",
        "5 6 7", "a1b2", "1e5", "50%", "1/2", "#1", "no numbers here",
    ]
    norm = [{"in": t, "numbers": normalize_numbers(t), "time": expand_time_english(t), "abbr": expand_abbreviations(t)} for t in texts]
    GOLD.mkdir(parents=True, exist_ok=True)
    (GOLD / "numbers.json").write_text(json.dumps({"cardinal": cardinal, "year": year, "plain_and": plain_and, "ordinals": ordinals, "normalize": norm}, ensure_ascii=False), encoding="utf8")
    print("numbers golden:", len(cardinal), len(year), len(plain_and), len(ordinals), len(norm))
    for n in (0, 5, 21, 100, 101, 1234, 100000, 1000001, -7):
        print(n, "->", repr(e.number_to_words(n, andword="")), "|", repr(e.number_to_words(n)))
    print(repr(e.number_to_words(1905, andword="", zero="oh", group=2)), repr(e.number_to_words(2010, andword="", zero="oh", group=2)), repr(e.number_to_words("21st")), repr(e.number_to_words("101st")))


if __name__ == "__main__":
    main()
