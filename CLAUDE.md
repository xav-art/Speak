# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Speak is an offline voice dictation app for Windows, written in Python 3.12 and run as a console app. Tap a hotkey, speak, and faster-whisper transcribes the audio. Qwen 2.5 (a GGUF model run by llama-cpp-python) cleans up the text, which is then pasted or typed into the focused window. Meeting mode records the mic and WASAPI loopback, diarizes the speakers with sherpa-onnx, and writes a Markdown transcript with minutes. Two constraints shape every change: no network calls at runtime (only one-time model downloads, each checked against a SHA-256), and no admin rights (Windows user space only).

## Commands

```powershell
# Run the app (uses .venv\Scripts\python.exe if setup created one, otherwise global python)
.\start_speak.bat                 # or: python -X utf8 speak.py [--hotkey F8 --model small --language da --ai-mode off]

# Tests (stdlib unittest; no pytest)
python -m unittest test_speak.py
python -m unittest test_speak.TestMeetingHelpers                       # one class
python -m unittest test_speak.TestMeetingHelpers.test_format_timestamp # one test

# Benchmark: real pipeline on synthetic clips, nothing typed, user config untouched (see README "Benchmark")
python -X utf8 bench/run_bench.py --label <name> --repeat 3 [--scenario same|other|none] [--set model=small] [--dictionary bench/dictionary_common.json]
python -X utf8 bench/run_bench.py --compare bench/results/A.json bench/results/B.json

# Headless entry points
python -X utf8 speak.py --download-models                 # fetch models for the current config, then exit
python -X utf8 speak.py --transcribe-meeting <file|recording-folder>  # runs alongside a live Speak window
```

There is no build step and no linter config. `install.ps1` is the installer, updater and uninstaller: it creates `.venv`, installs llama-cpp-python from a prebuilt CPU wheel index (`--no-deps --only-binary`), because PyPI only has the source, and the rest from `requirements.txt`. In this dev checkout there is no `.venv`, so the launchers fall back to global `python`.

## Architecture

`speak.py` is the hub. `SpeakApp` is one class assembled from mixins, so every method is still `app.<name>` (tests, the benchmark and `MeetingManager` rely on that):

- **`speak.py` / `SpeakApp(EnginesMixin, InsertionMixin)`**: owns the config (`load_config`/`save_config`: an atomic write through `.tmp`, and a corrupt file is renamed to `.corrupt-*`), the Win32 hotkey listener thread (`RegisterHotKey` + `PeekMessageW` loop), audio capture, the menu and the learn hotkey. One **worker thread** consumes the `self.jobs` queue in order, with `_process_worker`: `transcribe` (Whisper), then `clean_text` (dictionary, then the fast path or Qwen), then `insert_text`. A Win32 mutex makes it single-instance, except in `--transcribe-meeting` mode. It re-exports the names of the modules below (`speak.build_cleanup_prompt`, `speak.send_keys`, ...), but a test that *patches* a name must patch the module that uses it (`speak_engines.ensure_local_llm`, `speak_insert._user32`).
- **`speak_engines.py` / `EnginesMixin`**: Whisper (`transcribe`, `_decode`, the encoder cache in `_install_whisper_speedups`), Qwen (`clean_text`, the cleanup prompt, `_split_for_llm_lines`, which chunks long dictations at line breaks, then sentences, and returns each chunk's separator so line and paragraph breaks survive) and the model downloads.
- **`speak_insert.py` / `InsertionMixin`**: clipboard paste or `SendInput` typing, chosen by `paste_mode`. After a paste the user's clipboard is restored on a timer (`_schedule_clipboard_restore`); anything that uses the clipboard again first calls `_take_pending_restore`.
- **`speak_base.py`**: paths, `log` (silenced by `speak_base.QUIET`, which the benchmarks set) and the typed Win32 handles `_kernel32`/`_user32`.
- **`speak_text.py`**: pure text rules shared by dictation and meetings, all unit tested: the fast path (`strip_leading_fillers`, `needs_llm`), meeting disfluency check (`has_disfluency`), what code does before Qwen (`prepare_for_llm`: spoken commands, filler sounds), the guard on Qwen's output (`cleanup_is_faithful`), the filter for text Whisper invents on silence (`is_hallucination`). Filler lists are per language on purpose: "er" is a filler in English but "is" in Danish, and "um" is a German word.
- **`speak_dictionary.py` / `Dictionary`**: personal vocabulary in `speak_dictionary.json`. It supplies the Whisper `initial_prompt` (`whisper_prompt()`), mishearing corrections and casing enforcement. `prepare()` returns `(text, hint, applied, needs_llm)`: the hint goes into the Qwen prompt when Qwen runs anyway, but only `needs_llm` (an unconfirmed mishearing, or a name whose capital `apply_casing` does not enforce, see `casing_is_enforced`) takes a short phrase off the fast path. Real dictionaries contain everyday words, so a plain "dictionary word present" check sends most phrases to Qwen. `note_usage` writes the file, so the worker calls it after insertion, outside both locks. `extract_corrections` diffs the original and corrected text for the Ctrl+Alt+L "learn from selection" flow. It rejects rewrites (>50% of words changed), common words and words under 3 letters.
- **`speak_meeting.py`**: pure helper functions at module level (diarization post-processing, turn building, echo removal, `split_clauses`/`render_minutes` for action items). Most tests target these. It also holds `MeetingRecorder` (mic + loopback streamed to WAVs under `recordings/`; both streams reopen after a device loss and fill the gap with silence, and `meeting.json` records each stream's wall-clock length so `correct_drift` can line up the two hardware clocks; it ignores implausible ratios, such as a PC that slept) and `MeetingManager`, which has its own background worker queue. `MeetingManager` receives the `SpeakApp` instance and reuses it: `app._load_whisper`, `app.clean_text`, `app.load_llm`, `app.dictionary`, `app.play_sound`.

**Concurrency rule:** `app.engine_lock` serializes all use of the shared Whisper/LLM engines. The dictation worker holds it per job, and meeting cleanup takes it per turn, so dictation keeps working while a meeting is processed. Model switches from the menu also wait on it. Any new code that touches `app.local_llm` or the Whisper model must take this lock.

**Cleanup prompt design:** one prompt for all languages and for meeting turns (`build_cleanup_prompt` in `speak.py`): a fixed system prompt plus worked examples, then one short note (meeting turn, dictionary hint, "Write in Danish.") and the transcript. llama.cpp reuses the evaluated prefix up to the first differing token, so anything that varies per call must come after the examples (`TestCleanupPrompt` checks this); each token after it costs about 25 ms on a laptop CPU. Code does what it can do reliably before Qwen (`prepare_for_llm`), and every Qwen output passes `cleanup_is_faithful` or the prepared text is used instead. Tune prompts against a Qwen-only harness on fixed transcripts, not only the full benchmark: it runs in two minutes and shows each output.

**Meeting minutes design:** the code finds the clauses (multilingual promise/request/agreement cues), and Qwen only rephrases one clause at a time into JSON (few-shot, no grammar constraint). The **owner comes from the transcript, not the LLM** (who promised, or who agreed to a request). Items are grounded on word stems in their source clause. Cleanup of meeting turns is rejected by `faithful()` (the same check as dictation) if Qwen dropped or changed content words.

## Conventions and gotchas

- Whisper's language detection is unreliable on phrases of a few seconds (on `tiny`/`base`; `LANGUAGE_CHECK_MODELS`, since on `medium` the check fixed nothing and cost 3.4 s). For up to 4 s of speech that Whisper detects as another language than the user's last one, `transcribe` decodes both candidates greedily (`_decode(..., fallback=False)`; the temperature fallback made a wrong-language pass take seconds) and switches to the last language only if its average log-probability is higher by `LANGUAGE_CHECK_MARGIN`: decoded in the wrong language, Whisper translates fluently ("Non, merci" -> "No, thank you.") and scores almost as well. Validate language changes with `bench/run_bench.py --scenario other` as well as `same`.
- Run Python with `-X utf8` (all launchers do). Output includes Danish, German and French text.
- Personal and generated files are gitignored and must never be committed: `speak_config.json`, `speak_dictionary.json`, `speak_voice.json` (voice embedding), `models/`, `recordings/`, `.venv/`. `install.ps1` preserves them on update. Add any new per-user file to both `.gitignore` and the `install.ps1` exclude/keep lists.
- New config keys: add a default in `load_config` and document the key in the README Configuration table.
- Meeting transcripts default to the local `%USERPROFILE%\Documents\Speak Meetings`, never the OneDrive-redirected known folder. This is deliberate, for privacy.
- `speak`, `speak.bat`, `speak.cmd` and `start_speak.bat` are near-duplicate launchers. Keep them in sync. `install_dependencies.bat` is a legacy alias for `setup.cmd`.
- The README's "Performance notes" list optimizations that were measured and rejected (speculative decoding, flash attention, Whisper `hotwords`, more Whisper threads on hybrid CPUs). Do not reintroduce them as defaults.
- Pushing to `xav-art/Speak`: Git Credential Manager may sign in as a work account and get a 403. Push with `git -c credential.helper= -c "credential.helper=!gh auth git-credential" push`. Do not run `gh auth setup-git`.
