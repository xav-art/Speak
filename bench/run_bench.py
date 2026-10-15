"""
Speak benchmark: runs every clip in bench/clips through the real dictation
pipeline (Whisper, dictionary, fast path or the LLM) and reports speed and accuracy.

Nothing is typed or pasted, your speak_config.json is never written, and your
personal dictionary is not used (an empty one is, or a copy of the file given
with --dictionary), so results are comparable between machines and over time.

Usage (from the Speak folder, with Speak's Python):
  python -X utf8 bench/run_bench.py --label baseline
  python -X utf8 bench/run_bench.py --label small --set model=small --repeat 3
  python -X utf8 bench/run_bench.py --scenario other     # last language is a different one
  python -X utf8 bench/run_bench.py --dictionary bench/dictionary_common.json   # a filled dictionary
  python -X utf8 bench/run_bench.py --compare bench/results/A.json bench/results/B.json

Scenarios set the "last language" Speak remembers before each clip:
  same   the clip's own language (a user who dictates in one language; default)
  other  a different language (the user just switched)
  none   no remembered language (first dictation after install)

Metrics:
  WER       word error rate of the final text against the expected text, after
            lowercasing and removing punctuation (lower is better)
  STT WER   word error rate of the raw Whisper text against what was spoken,
            ignoring fillers such as "um" and "øh"
  lang      share of clips whose language was detected correctly
  fast      share of clips that skipped the LLM
  phantom   no-speech clips that produced text (should be 0)
"""

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(HERE, "results")
LANGS = ("en", "da", "de", "fr")
SEED = 1234

# Fillers ignored when scoring raw Whisper output: whether Whisper writes
# them down is not an error, removing them is the cleanup's job.
SCORING_FILLERS = {"um", "umm", "uh", "uhm", "uhh", "hmm", "hm", "erm", "ehm",
                   "øh", "øhm", "øhh", "äh", "ähm", "ähh", "euh"}


# ---- Scoring (pure) -----------------------------------------------------------

def normalize_words(text, drop=()):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'") and w.strip("'") not in drop]


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer_counts(reference, hypothesis, drop=()):
    """(errors, reference words). A no-speech clip with output counts as 1 error per word."""
    ref, hyp = normalize_words(reference, drop), normalize_words(hypothesis, drop)
    return edit_distance(ref, hyp), len(ref)


def ratio(errors, words):
    return errors / words if words else 0.0


# ---- Running ------------------------------------------------------------------

def git_revision():
    try:
        rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        # Untracked files count too: a new module that is not committed yet
        # changes what is being measured.
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def last_language_for(clip_lang, scenario):
    if scenario == "none":
        return None
    if scenario == "other":
        return "da" if clip_lang in (None, "en") else "en"
    return clip_lang or "en"


def parse_overrides(pairs):
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def run(args):
    sys.path.insert(0, ROOT)
    import ctranslate2
    import speak
    import speak_meeting
    from speak_dictionary import Dictionary

    tmp = tempfile.mkdtemp(prefix="speak-bench-")
    # Anything that tries to save the config lands in the temp folder, never
    # in the user's speak_config.json.
    speak.CONFIG_PATH = os.path.join(tmp, "speak_config.json")
    cfg = speak.load_config(speak.CONFIG_PATH)
    cfg.update(parse_overrides(args.set))

    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    if args.only:
        corpus = [c for c in corpus if args.only in c["id"]]

    if args.dictionary:
        # A copy: the run must never change the file it was given.
        shutil.copyfile(args.dictionary, os.path.join(tmp, "speak_dictionary.json"))
    t0 = time.time()
    app = speak.SpeakApp(cfg)
    app.dictionary = Dictionary(tmp)
    load_seconds = time.time() - t0

    rows = []
    for clip in corpus:
        audio = speak_meeting.read_wav(os.path.join(HERE, "clips", clip["id"] + ".wav"))
        timings = []
        for _ in range(args.repeat):
            app.config["last_language"] = last_language_for(clip["lang"], args.scenario)
            # Whisper samples when greedy decoding looks wrong (temperature
            # fallback) and Qwen samples at temperature 0.1. Fixed seeds per
            # clip make most results repeatable. Not all: CTranslate2 samples
            # on its own worker thread, which this seed does not reach, so a
            # clip that needs Whisper's fallback can change between runs.
            ctranslate2.set_random_seed(SEED)
            if app.local_llm:
                app.local_llm.set_seed(SEED)
            with app.engine_lock:
                t0 = time.perf_counter()
                raw, lang, prob = app.transcribe(audio)
                t1 = time.perf_counter()
                stats = {}
                final = app.clean_text(raw, lang, verbose=False, stats=stats) if raw else ""
                t2 = time.perf_counter()
            timings.append((t1 - t0, t2 - t1))
        stt = statistics.median(t[0] for t in timings)
        cleanup = statistics.median(t[1] for t in timings)
        errors, words = wer_counts(clip["expected"], final)
        stt_errors, stt_words = wer_counts(clip["text"], raw, SCORING_FILLERS)
        if clip["kind"] == "silence":
            errors = stt_errors = len(normalize_words(final))
        missing = [s for s in clip.get("must_contain", []) if s not in final]
        rows.append({
            "id": clip["id"], "kind": clip["kind"], "lang": clip["lang"],
            "detected": lang if raw else None, "lang_prob": round(prob, 3) if raw else None,
            "raw": raw, "final": final, "expected": clip["expected"],
            "llm": bool(stats.get("llm")), "stt_s": round(stt, 3), "cleanup_s": round(cleanup, 3),
            "total_s": round(stt + cleanup, 3),
            "errors": errors, "words": words, "stt_errors": stt_errors, "stt_words": stt_words,
            "missing": missing,
        })
        print_row(rows[-1])

    result = {
        "label": args.label, "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "revision": git_revision(), "scenario": args.scenario, "repeat": args.repeat,
        "machine": platform.processor() or platform.machine(), "load_s": round(load_seconds, 2),
        "config": {k: cfg.get(k) for k in ("model", "local_model", "ai_mode", "whisper_threads", "llm_threads",
                                           "llm_fast_path", "language", "language_check_short_phrases")},
        "overrides": parse_overrides(args.set),
        "dictionary": os.path.basename(args.dictionary) if args.dictionary else None,
        "summary": summarize(rows), "clips": rows,
    }
    print_summary(result)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.out or os.path.join(RESULTS_DIR, f"{stamp}-{args.label}-{args.scenario}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {os.path.abspath(path)}")
    shutil.rmtree(tmp, ignore_errors=True)


def summarize(rows):
    speech = [r for r in rows if r["kind"] != "silence"]
    silence = [r for r in rows if r["kind"] == "silence"]

    def block(subset):
        if not subset:
            return None
        return {
            "clips": len(subset),
            "wer": round(ratio(sum(r["errors"] for r in subset), sum(r["words"] for r in subset)), 4),
            "stt_wer": round(ratio(sum(r["stt_errors"] for r in subset), sum(r["stt_words"] for r in subset)), 4),
            "lang_ok": round(sum(r["detected"] == r["lang"] for r in subset) / len(subset), 3),
            "fast_path": round(sum(not r["llm"] for r in subset) / len(subset), 3),
            "median_total_s": round(statistics.median(r["total_s"] for r in subset), 3),
            "mean_total_s": round(statistics.mean(r["total_s"] for r in subset), 3),
            "mean_stt_s": round(statistics.mean(r["stt_s"] for r in subset), 3),
            "mean_cleanup_s": round(statistics.mean(r["cleanup_s"] for r in subset), 3),
            "format_misses": sum(bool(r["missing"]) for r in subset),
        }

    out = {"all": block(speech),
           "phantom_text": sum(bool(r["final"]) for r in silence),
           "silence_clips": len(silence)}
    for lang in LANGS:
        out[lang] = block([r for r in speech if r["lang"] == lang])
    for kind in sorted({r["kind"] for r in speech}):
        out["kind:" + kind] = block([r for r in speech if r["kind"] == kind])
    return out


def print_row(r):
    flag = "LLM " if r["llm"] else "fast"
    lang = f"{r['detected'] or '-'}{'' if r['detected'] == r['lang'] or r['kind'] == 'silence' else '!'}"
    print(f"{r['id']:<26} {lang:<4} {flag} {r['stt_s']:5.2f}+{r['cleanup_s']:5.2f}s "
          f"err {r['errors']}/{r['words']:<3} {json.dumps(r['final'], ensure_ascii=False)[:90]}")


def print_summary(result):
    s = result["summary"]
    print(f"\n== {result['label']} ({result['revision']}, scenario {result['scenario']}, "
          f"model {result['config']['model']}, repeat {result['repeat']}, "
          f"dictionary {result.get('dictionary') or 'empty'}) ==")
    print(f"{'group':<16}{'clips':>6}{'WER':>8}{'STT WER':>9}{'lang':>7}{'fast':>7}"
          f"{'median s':>10}{'mean s':>8}{'stt s':>7}{'llm s':>7}{'fmt':>5}")
    for key, b in s.items():
        if isinstance(b, dict):
            print(f"{key:<16}{b['clips']:>6}{b['wer']:>8.1%}{b['stt_wer']:>9.1%}{b['lang_ok']:>7.0%}"
                  f"{b['fast_path']:>7.0%}{b['median_total_s']:>10.2f}{b['mean_total_s']:>8.2f}"
                  f"{b['mean_stt_s']:>7.2f}{b['mean_cleanup_s']:>7.2f}{b['format_misses']:>5}")
    print(f"phantom text on no-speech clips: {s['phantom_text']}/{s['silence_clips']}")


def compare(path_a, path_b):
    with open(path_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        b = json.load(f)
    print(f"A = {a['label']} ({a['revision']}, {a['scenario']})   B = {b['label']} ({b['revision']}, {b['scenario']})")
    print(f"{'group':<16}{'WER A':>8}{'WER B':>8}{'lang A':>8}{'lang B':>8}{'fast A':>8}{'fast B':>8}"
          f"{'mean s A':>10}{'mean s B':>10}")
    for key, ba in a["summary"].items():
        bb = b["summary"].get(key)
        if isinstance(ba, dict) and isinstance(bb, dict):
            print(f"{key:<16}{ba['wer']:>8.1%}{bb['wer']:>8.1%}{ba['lang_ok']:>8.0%}{bb['lang_ok']:>8.0%}"
                  f"{ba['fast_path']:>8.0%}{bb['fast_path']:>8.0%}{ba['mean_total_s']:>10.2f}{bb['mean_total_s']:>10.2f}")
    print(f"phantom text: A {a['summary']['phantom_text']}, B {b['summary']['phantom_text']}")
    by_id = {r["id"]: r for r in a["clips"]}
    print("\nClips whose final text changed:")
    for rb in b["clips"]:
        ra = by_id.get(rb["id"])
        if ra and ra["final"] != rb["final"]:
            print(f"  {rb['id']}  (errors {ra['errors']} -> {rb['errors']})")
            print(f"    A: {json.dumps(ra['final'], ensure_ascii=False)}")
            print(f"    B: {json.dumps(rb['final'], ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run", help="name stored with the results")
    parser.add_argument("--scenario", default="same", choices=["same", "other", "none"])
    parser.add_argument("--repeat", type=int, default=1, help="runs per clip; timings are the median")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config key (JSON value)")
    parser.add_argument("--only", help="only run clips whose id contains this text")
    parser.add_argument("--dictionary", metavar="JSON",
                        help="run with a copy of this dictionary file instead of an empty one")
    parser.add_argument("--out", help="where to save the JSON results")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="compare two saved runs")
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
