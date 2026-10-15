"""
Speak long-dictation benchmark: a fake microphone plays a long synthetic
dictation (the benchmark clips of one language, one after the other) into a
real SpeakApp in real time, with the real Whisper and Qwen. It measures the
time from the stop tap to the text, and the word error rate, so transcribing
and cleaning while recording (config live_transcription) can be compared with
a single pass after the stop.

Usage (from the Speak folder, with Speak's Python):
  python -X utf8 bench/run_live_bench.py --model medium --live on en da
  python -X utf8 bench/run_live_bench.py --model medium --live off en da

Nothing is typed or pasted, and your speak_config.json and dictionary are
never written. Each language takes as long as its clips (English about 70 s).
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import speak                                # noqa: E402
import speak_meeting as sm                  # noqa: E402
from speak_dictionary import Dictionary     # noqa: E402
from run_bench import wer_counts            # noqa: E402

SR = 16000


def long_clip(corpus, lang):
    """All the language's clips, 0.7 s apart, and their expected text."""
    parts, texts = [], []
    for c in corpus:
        if c["lang"] == lang:
            parts += [sm.read_wav(os.path.join(HERE, "clips", c["id"] + ".wav")), np.zeros(int(0.7 * SR), np.float32)]
            texts.append(c["expected"])
    return np.concatenate(parts), " ".join(texts)


class FakeStream:
    """A microphone that plays `clip` in real time, then silence."""
    clip = np.zeros(0, np.float32)

    def __init__(self, samplerate, channels, dtype, callback, **kwargs):
        self.callback, self.running = callback, False

    def start(self):
        self.running = True

        def run():
            pos, t0, block = 0, time.monotonic(), 1600
            while self.running:
                chunk = FakeStream.clip[pos:pos + block]
                if len(chunk) < block:
                    chunk = np.concatenate([chunk, np.zeros(block - len(chunk), np.float32)])
                self.callback(chunk.reshape(-1, 1), block, None, None)
                pos += block
                wait = t0 + pos / SR - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.running = False

    def close(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="base", help="Whisper model")
    parser.add_argument("--live", choices=("on", "off"), default="on", help="live_transcription")
    parser.add_argument("--show", action="store_true", help="print the app's log and the text")
    parser.add_argument("langs", nargs="+", help="languages: en, da, de, fr")
    args = parser.parse_args()

    corpus = json.load(open(os.path.join(HERE, "corpus.json"), encoding="utf-8"))
    tmp = tempfile.mkdtemp()
    speak.CONFIG_PATH = os.path.join(tmp, "speak_config.json")   # never the user's settings
    speak.speak_base.QUIET = not args.show
    speak.sd = types.SimpleNamespace(InputStream=FakeStream)
    cfg = speak.load_config(speak.CONFIG_PATH)
    cfg.update({"model": args.model, "live_transcription": args.live == "on"})
    app = speak.SpeakApp(cfg)
    app.dictionary = Dictionary(tempfile.mkdtemp())   # empty, and never the user's
    inserted = []
    app.insert_text = lambda text: inserted.append((time.monotonic(), text))
    app._schedule_clipboard_restore = lambda token: None
    app.play_sound = lambda kind: None
    threading.Thread(target=app._worker_loop, daemon=True).start()

    for lang in args.langs:
        clip, expected = long_clip(corpus, lang)
        FakeStream.clip = clip
        app.config["last_language"] = lang
        inserted.clear()
        app.start_recording()
        time.sleep(len(clip) / SR + 0.8)   # the speaker finishes, then taps stop
        t_stop = time.monotonic()
        app.stop_recording()
        while not inserted:
            time.sleep(0.05)
        t_done, text = inserted[0]
        errors, words = wer_counts(expected, text)
        print(f"{args.model} live {args.live} {lang}: {len(clip) / SR:.0f} s of speech, "
              f"{t_done - t_stop:.2f} s after the stop, WER {errors}/{words} = {errors / words:.1%}", flush=True)
        if args.show:
            print("   text:", text)


if __name__ == "__main__":
    main()
