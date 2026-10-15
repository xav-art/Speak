"""
Speak meeting benchmark: a synthetic call meeting through the real
MeetingManager.process() (diarization, language detection, Whisper, turns),
AI cleanup and summary off unless set. No real voice and no audio device:
the speakers are Windows TTS voices (bench/meeting/conv, made from conv.json
with tts.ps1). "Mark" is on the microphone and is enrolled as "Me"; the others
come in through the call audio.

Usage (from the Speak folder):
  python -X utf8 bench/run_meeting_bench.py --label seq --set meeting_batch_size=0
  python -X utf8 bench/run_meeting_bench.py --label batched --monologues
  python -X utf8 bench/run_meeting_bench.py --label ai --set ai_mode=local --set meeting_summary=true

--repeat N plays the 11-line conversation N times (default 6: about 10 min).
--monologues puts each speaker's lines back to back (20-40 s turns): batching
and chunking behave differently there than on short turns.

Metrics: Whisper seconds (all transcription calls), total seconds, WER against
the script, "missing" words (in the script but not in the transcript: dropped
speech; a misheard word counts as missing AND extra) and how many turns are
on the right side (Me vs the call). Run variants one after the other, more
than once: timings move by 10 % with background load.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import speak                                # noqa: E402
import speak_meeting as sm                  # noqa: E402
from run_bench import normalize_words, wer_counts   # noqa: E402

DATA = os.path.join(HERE, "meeting")
ME = "Mark"


def build_meeting(repeat, monologues, seed=7):
    """(mic, system, script lines in order) as float32 16 kHz arrays."""
    lines = json.load(open(os.path.join(DATA, "conv.json"), encoding="utf-8"))
    clips = [sm.read_wav(f) for f in sorted(glob.glob(os.path.join(DATA, "conv", "*.wav")))]
    if monologues:
        voices = [line["voice"] for line in lines]
        order = sorted(range(len(lines)), key=lambda i: voices.index(lines[i]["voice"]))
        lines, clips = [lines[i] for i in order], [clips[i] for i in order]
    lines, clips = lines * repeat, clips * repeat
    rng = np.random.default_rng(seed)
    mic, system = [], []
    for i, (line, clip) in enumerate(zip(lines, clips)):
        same_next = i + 1 < len(lines) and lines[i + 1]["voice"] == line["voice"]
        gap = np.zeros(int(sm.SAMPLE_RATE * (0.25 if monologues and same_next else rng.uniform(0.4, 1.0))),
                       np.float32)
        silent = np.zeros_like(clip)
        if line["voice"] == ME:
            mic += [clip * 0.7, gap]
            system += [silent, gap]
        else:
            mic += [silent, gap]
            system += [clip * 0.7, gap]
    mic = np.concatenate(mic)
    mic += rng.normal(0, 0.003, len(mic)).astype(np.float32)
    return mic, np.concatenate(system), lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="meeting")
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--monologues", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="config override, JSON value (e.g. meeting_batch_size=0)")
    args = parser.parse_args()

    overrides = {}
    for kv in args.set:
        key, value = kv.split("=", 1)
        try:
            overrides[key] = json.loads(value)
        except ValueError:
            overrides[key] = value

    mic, system, lines = build_meeting(args.repeat, args.monologues)
    root = tempfile.mkdtemp(prefix="speak-meeting-bench-")
    try:
        folder = os.path.join(root, "recordings", "2026-10-15_10-00-00")
        out_dir = os.path.join(root, "out")
        os.makedirs(folder)
        os.makedirs(out_dir)
        for name, audio in (("mic", mic), ("system", system)):
            writer = sm._WavWriter(os.path.join(folder, f"{name}.wav"))
            writer.write(audio)
            writer.close()
        json.dump({"started": "2026-10-15T10:00:00", "offsets": {"mic": 0.0, "system": 0.0}},
                  open(os.path.join(folder, "meeting.json"), "w"))

        cfg = speak.load_config(os.path.join(root, "speak_config.json"))   # defaults, never the user's
        cfg.update({"warm_up": False, "meeting_folder": out_dir, "meeting_summary": False,
                    "meeting_cleanup": False, "ai_mode": "off", "last_language": "en"})
        cfg.update(overrides)
        app = speak.SpeakApp(cfg, load_speech_model=False)
        m = app.meetings
        m.recordings_dir = os.path.join(root, "recordings")
        m.profile_path = os.path.join(root, "speak_voice.json")
        m.save_profile_from_audio(np.concatenate(
            [sm.read_wav(f) for f in sorted(glob.glob(os.path.join(DATA, "enroll", "*.wav")))]))

        whisper = {"s": 0.0}
        transcribe = m._transcribe_ranges

        def timed(*a, **k):
            t0 = time.time()
            try:
                return transcribe(*a, **k)
            finally:
                whisper["s"] += time.time() - t0
        m._transcribe_ranges = timed

        t0 = time.time()
        md = open(m.process(folder), encoding="utf-8").read()
        total = time.time() - t0
    finally:
        shutil.rmtree(root, ignore_errors=True)

    transcript = md.split("## Transcript", 1)[1]
    turns = re.findall(r"\*\*\[\d\d:\d\d:\d\d\] ([^:]+):\*\* (.*)", transcript)
    reference = " ".join(line["text"] for line in lines)
    hypothesis = " ".join(text for _, text in turns)
    errors, words = wer_counts(reference, hypothesis)
    missing = sum((Counter(normalize_words(reference)) - Counter(normalize_words(hypothesis))).values())
    script = json.load(open(os.path.join(DATA, "conv.json"), encoding="utf-8"))
    side_ok = 0
    for who, text in turns:
        w = set(normalize_words(text))
        best = max(script, key=lambda line: len(w & set(normalize_words(line["text"]))))
        side_ok += (who == "Me") == (best["voice"] == ME)
    result = {"label": args.label, "overrides": overrides, "monologues": args.monologues,
              "audio_min": round(len(system) / sm.SAMPLE_RATE / 60, 2), "whisper_s": round(whisper["s"], 1),
              "total_s": round(total, 1), "wer": round(errors / max(1, words), 4), "missing": missing,
              "words": words, "turns": len(turns), "side_ok": side_ok}
    print("\nRESULT " + json.dumps(result))
    out = os.path.join(HERE, "results", "meeting")
    os.makedirs(out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(out, f"{stamp}-{args.label}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out, "results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
