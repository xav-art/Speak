"""
Speak cleanup benchmark: runs fixed transcripts through the real cleanup step
(dictionary, fast path or Qwen, the faithfulness check) and scores the result.
No audio and no Whisper, so a run takes about two minutes: use it to tune the
cleanup prompt (speak.build_cleanup_prompt) and speak_text's rules, then
confirm with the full benchmark (run_bench.py).

The transcripts in cleanup_cases.json are what Whisper wrote for the
benchmark clips, plus written cases for commands, corrections, questions and
meeting turns in all four languages. "expected" is the clean text; words
Whisper got wrong stay wrong in it, so only the cleanup is scored.

Usage (from the Speak folder, with Speak's Python):
  python -X utf8 bench/run_cleanup_bench.py
  python -X utf8 bench/run_cleanup_bench.py --only da- --out bench/results/cleanup.json
  python -X utf8 bench/run_cleanup_bench.py --cases bench/cleanup_cases_hard.json       --llm models/llm-eval/Qwen3-1.7B-Q4_K_M.gguf --format qwen3

Scores: WER ignores case and punctuation (did the cleanup keep the words?).
"form" counts word tokens that differ with case and punctuation kept, commas
ignored (Danish start commas are optional): capitals, sentence ends, German
nouns. A form error is not always wrong; read the outputs.
"""

import argparse
import ctypes
import re
import json
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import speak                                # noqa: E402
import speak_meeting                        # noqa: E402
from speak_dictionary import Dictionary     # noqa: E402
from run_bench import edit_distance, wer_counts   # noqa: E402


def structure(text):
    """(lines, list items): the formatting a spoken command should produce."""
    lines = [line for line in text.strip().split("\n") if line.strip()]
    return len(lines), sum(1 for line in lines if line.lstrip().startswith(("-", "•", "*")))


def form_tokens(text):
    text = re.sub(r"\s+([?!:;])", r"\1", text.replace(",", " "))   # "va ?" = "va?"
    return text.split()


def peak_ram_mb():
    """Peak working set of this process (Windows), in MB."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] +                    [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    try:
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return 0
    except Exception:
        return 0
    return c.PeakWorkingSetSize / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="only run cases whose id contains this text")
    parser.add_argument("--out", help="where to save the JSON results")
    parser.add_argument("--cases", default=os.path.join(HERE, "cleanup_cases.json"), help="the cases file")
    parser.add_argument("--llm", help="a GGUF to test instead of the configured model")
    parser.add_argument("--format", default="chatml", choices=("chatml", "qwen3", "gemma", "gemma4"),
                        help="the chat format of --llm")
    args = parser.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]

    speak.speak_base.QUIET = True   # the per-call logs would bury the table
    cfg = speak.load_config(os.path.join(tempfile.mkdtemp(), "speak_config.json"))   # defaults, never the user's
    app = speak.SpeakApp.__new__(speak.SpeakApp)
    app.config = cfg
    app.engine_lock = threading.RLock()
    app.dictionary = Dictionary(tempfile.mkdtemp())   # empty: results compare across machines
    if args.llm:
        speak.speak_engines.LOCAL_LLM_MODELS["bench"] = {
            "file": os.path.relpath(os.path.abspath(args.llm), speak.speak_engines.LOCAL_LLM_DIR),
            "name": os.path.basename(args.llm), "repo": "", "sha256": "", "size": "", "format": args.format}
        cfg["local_model"] = "bench"
    t_load = time.time()
    app._init_ai()
    t_load = time.time() - t_load
    if not app.local_llm:
        sys.exit("The local LLM could not be loaded: run `speak.py --download-models` first.")
    # Fills the KV cache with the fixed prompt, as the app's warm-up does.
    t_warm = time.time()
    app.clean_text("Um, warm up the model, please, and, uh, check it.", "en", verbose=False)
    t_warm = time.time() - t_warm

    rows, t_all = [], time.time()
    for c in cases:
        stats = {}
        t0 = time.time()
        if c.get("meeting"):
            # As MeetingManager does: the turn without the LLM, then the LLM's
            # version if it is faithful to it.
            base = app.clean_text(c["raw"], c["lang"], use_llm=False, verbose=False)
            cleaned = app.clean_text(c["raw"], c["lang"], verbose=False, meeting=True, stats=stats)
            cleaned = cleaned.lstrip("-*• ").strip()
            prepared = speak.speak_text.prepare_for_llm(base, c["lang"], commands=False)
            out = speak.speak_text.light_cleanup(cleaned if speak_meeting.faithful(prepared, cleaned, c["lang"]) else base)
        else:
            out = app.clean_text(c["raw"], c["lang"], verbose=False, stats=stats)
        seconds = time.time() - t0
        errors, words = wer_counts(c["expected"], out)
        fmt_ok = structure(out) == structure(c["expected"])
        form = edit_distance(form_tokens(c["expected"]), form_tokens(out))
        rows.append({"id": c["id"], "lang": c["lang"], "out": out, "expected": c["expected"], "errors": errors,
                     "words": words, "form": form, "format_ok": fmt_ok, "llm": stats.get("llm"), "seconds": seconds})
        mark = "  " if errors == 0 and fmt_ok else "XX"
        print(f"{mark} {c['id']:<22} {errors:>2}/{words:<3} form {form:>2} {'fmt ' if fmt_ok else 'FMT!'} "
              f"{seconds:5.2f}s  {out!r}",
              flush=True)

    errors = sum(r["errors"] for r in rows)
    words = sum(r["words"] for r in rows)
    perfect = sum(1 for r in rows if r["errors"] == 0 and r["format_ok"])
    llm_times = [r["seconds"] for r in rows if r["llm"]]
    print(f"\nWER {errors}/{words} = {errors / max(1, words):.1%}, perfect {perfect}/{len(rows)}, "
          f"format {sum(r['format_ok'] for r in rows)}/{len(rows)}, "
          f"mean AI case {sum(llm_times) / max(1, len(llm_times)):.2f}s, total {time.time() - t_all:.0f}s")
    for lang in sorted({r["lang"] for r in rows}):
        rs = [r for r in rows if r["lang"] == lang]
        lw = sum(r["words"] for r in rs)
        print(f"  {lang}: WER {sum(r['errors'] for r in rs) / max(1, lw):.1%}, form errors "
              f"{sum(r['form'] for r in rs)}/{lw}, perfect {sum(1 for r in rs if r['errors'] == 0 and r['format_ok'])}"
              f"/{len(rs)}")
    print(f"form errors {sum(r['form'] for r in rows)}/{words}, load {t_load:.1f}s, warm-up {t_warm:.1f}s, "
          f"peak RAM {peak_ram_mb():.0f} MB")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
