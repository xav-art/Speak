"""
Speak benchmark: builds the test clips in bench/clips from bench/corpus.json.

The clips are synthetic on purpose: no real person's voice is recorded, and
anyone can rebuild them. Each clip is made less studio-clean the same way
every time (seeded by its id): a random level, a light room reverb,
background noise at 25 dB SNR, and the silence a hotkey tap leaves before
and after speech. The "silence" clips contain no speech at all and test that
Whisper does not invent text.

Voices:
  "Mark", "Zira", "David", "Helle"   Windows voices (Settings > Time & language > Speech)
  "piper:<voice>"                    Piper TTS (https://github.com/rhasspy/piper), offline
  "noise", "hum", "breath"           generated here, no speech

Usage (from a Python that has piper-tts and numpy installed; Speak's own
environment does not need either):
  python bench/make_clips.py --piper-voices <folder with *.onnx voices> [--only <id part>]

The clips are committed to git, so running the benchmark does not need this.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(HERE, "clips")
SAMPLE_RATE = 16000
LEAD_SILENCE = 0.4      # seconds between the start beep and speech
TAIL_SILENCE = 0.6      # seconds between speech and the stop tap
SNR_DB = 25.0

# Same WinRT approach as Speak's meeting test tooling: OneCore voices (Helle,
# Mark...) are not visible to System.Speech, only to Windows.Media.SpeechSynthesis.
WINRT_TTS = r"""
param([string]$JsonPath)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result }
[Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataReader,Windows.Storage.Streams,ContentType=WindowsRuntime] | Out-Null
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
foreach ($it in (Get-Content -Raw -Encoding UTF8 $JsonPath | ConvertFrom-Json)) {
  $voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.DisplayName -eq "Microsoft $($it.voice)" } | Select-Object -First 1
  if (-not $voice) { Write-Error "Windows voice not installed: $($it.voice)"; exit 1 }
  $synth.Voice = $voice
  $stream = Await ($synth.SynthesizeTextToStreamAsync($it.text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
  $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
  $n = Await ($reader.LoadAsync([uint32]$stream.Size)) ([uint32])
  $bytes = New-Object byte[] $n; $reader.ReadBytes($bytes)
  [IO.File]::WriteAllBytes($it.out, $bytes)
}
"""


def read_any_wav(path):
    with wave.open(path, "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM is supported, got {width * 8}-bit")
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return resample(audio, rate, SAMPLE_RATE)


def resample(audio, rate, target):
    """Band-limited FFT resampling (the clip is short, so one FFT is fine)."""
    if rate == target or not len(audio):
        return audio
    n_out = int(round(len(audio) * target / rate))
    spectrum = np.fft.rfft(audio)
    keep = min(len(spectrum), n_out // 2 + 1)
    out = np.zeros(n_out // 2 + 1, dtype=complex)
    out[:keep] = spectrum[:keep]
    return np.fft.irfft(out, n_out) * (n_out / len(audio))


def pink_noise(rng, n):
    white = rng.standard_normal(n + 64)
    kernel = np.exp(-np.arange(64) / 12.0)
    return np.convolve(white, kernel / kernel.sum(), mode="same")[:n]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def degrade(speech, rng):
    """Room-ish, noisy, tap-timed version of a clean TTS clip."""
    speech = speech / (np.max(np.abs(speech)) or 1.0)
    t = np.arange(int(0.15 * SAMPLE_RATE)) / SAMPLE_RATE
    ir = rng.standard_normal(len(t)) * np.exp(-t / 0.04)
    ir[0] = 0.0
    wet = np.convolve(speech, ir / (np.sqrt(np.sum(ir ** 2)) or 1.0))[:len(speech)]
    speech = speech + 0.25 * wet
    audio = np.concatenate([np.zeros(int(LEAD_SILENCE * SAMPLE_RATE)), speech,
                            np.zeros(int(TAIL_SILENCE * SAMPLE_RATE))])
    noise = pink_noise(rng, len(audio))
    noise *= rms(speech) / (rms(noise) or 1.0) / (10 ** (SNR_DB / 20))
    audio = audio + noise
    peak_db = rng.uniform(-12.0, -4.0)
    return audio / (np.max(np.abs(audio)) or 1.0) * (10 ** (peak_db / 20))


def no_speech(kind, rng, seconds=3.0):
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    floor = pink_noise(rng, n)
    floor *= 10 ** (-50 / 20) / (rms(floor) or 1.0)
    if kind == "noise":
        return floor
    if kind == "hum":
        hum = 0.01 * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t))
        clicks = np.zeros(n)
        for start in rng.integers(0, n - 200, size=12):
            clicks[start:start + 200] += rng.standard_normal(200) * np.exp(-np.arange(200) / 30.0) * 0.05
        return floor + hum + clicks
    if kind == "breath":
        out = floor.copy()
        for start in np.arange(0.2, seconds - 0.6, 1.2):
            i = int(start * SAMPLE_RATE)
            m = int(0.5 * SAMPLE_RATE)
            burst = np.convolve(rng.standard_normal(m), np.ones(8) / 8, mode="same")
            out[i:i + m] += burst * np.hanning(m) * 0.03
        return out
    raise ValueError(f"unknown no-speech clip kind: {kind}")


def write_wav(path, audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def synth_windows(items, tmp):
    jobs = [{"text": it["text"], "voice": it["voice"], "out": os.path.join(tmp, it["id"] + ".raw.wav")}
            for it in items]
    if not jobs:
        return {}
    job_file = os.path.join(tmp, "jobs.json")
    script = os.path.join(tmp, "tts.ps1")
    with open(job_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False)
    with open(script, "w", encoding="utf-8-sig") as f:
        f.write(WINRT_TTS)
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, job_file],
                   check=True)
    return {it["id"]: job["out"] for it, job in zip(items, jobs)}


def synth_piper(items, voices_dir, tmp):
    if not items:
        return {}
    from piper import PiperVoice
    loaded, out = {}, {}
    for it in items:
        name = it["voice"].split(":", 1)[1]
        if name not in loaded:
            model = os.path.join(voices_dir, name + ".onnx")
            if not os.path.exists(model):
                sys.exit(f"Piper voice not found: {model}")
            loaded[name] = PiperVoice.load(model)
        path = os.path.join(tmp, it["id"] + ".raw.wav")
        with wave.open(path, "wb") as w:
            loaded[name].synthesize_wav(it["text"], w)
        out[it["id"]] = path
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--piper-voices", help="folder with the Piper *.onnx voices the corpus uses")
    parser.add_argument("--only", help="only rebuild clips whose id contains this text")
    args = parser.parse_args()

    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    if args.only:
        corpus = [it for it in corpus if args.only in it["id"]]
    os.makedirs(CLIPS_DIR, exist_ok=True)

    piper_items = [it for it in corpus if it["voice"].startswith("piper:")]
    if piper_items and not args.piper_voices:
        sys.exit("The corpus uses Piper voices: pass --piper-voices <folder>.")
    windows_items = [it for it in corpus if it["kind"] != "silence" and not it["voice"].startswith("piper:")]

    with tempfile.TemporaryDirectory() as tmp:
        raw = synth_windows(windows_items, tmp)
        raw.update(synth_piper(piper_items, args.piper_voices, tmp))
        for it in corpus:
            rng = np.random.default_rng(zlib.crc32(it["id"].encode("utf-8")))
            if it["kind"] == "silence":
                audio = no_speech(it["voice"], rng)
            else:
                audio = degrade(read_any_wav(raw[it["id"]]), rng)
            write_wav(os.path.join(CLIPS_DIR, it["id"] + ".wav"), audio)
            print(f"{it['id']:<28} {len(audio) / SAMPLE_RATE:5.1f} s")
    print(f"Wrote {len(corpus)} clips to {CLIPS_DIR}")


if __name__ == "__main__":
    main()
