"""
Speak - Unit and Regression Test Suite
Run with: python -m unittest test_speak.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

import speak_dictionary as sd
import speak_meeting as sm
import speak_text as st
import speak


class TestDictionaryCore(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(sd.tokenize("Hello, World!"), ["Hello", "World"])
        self.assertEqual(sd.tokenize("  multiple   spaces  "), ["multiple", "spaces"])
        self.assertEqual(sd.tokenize(""), [])

    def test_edit_distance(self):
        self.assertEqual(sd.edit_distance("kitten", "sitting"), 3)
        self.assertEqual(sd.edit_distance("Annika", "Anika"), 1)
        self.assertEqual(sd.edit_distance("test", "test"), 0)
        self.assertEqual(sd.edit_distance("", "abc"), 3)

    def test_extract_corrections_valid_substitution(self):
        orig = "I met with Anika today"
        corr = "I met with Annika today"
        results = sd.extract_corrections(orig, corr, [])
        self.assertEqual(len(results), 1)
        orig_w, new_w, ctx = results[0]
        self.assertEqual(orig_w, "Anika")
        self.assertEqual(new_w, "Annika")

    def test_extract_corrections_ignores_common_words(self):
        # "why" -> "what" should not be learned as a personal jargon word
        orig = "I know why you did that"
        corr = "I know what you did that"
        results = sd.extract_corrections(orig, corr, [])
        self.assertEqual(results, [])

    def test_extract_corrections_ignores_full_rewrites(self):
        # If > 50% words changed, it's a rewrite, not a phonetic correction
        orig = "one two three four five six"
        corr = "apple banana orange cherry grape melon"
        results = sd.extract_corrections(orig, corr, [])
        self.assertEqual(results, [])

    def test_extract_corrections_casing_enforcement(self):
        orig = "we use sap for inventory"
        corr = "we use SAP for inventory"
        results = sd.extract_corrections(orig, corr, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "SAP")


class TestDictionaryPersistenceAndUsage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dict = sd.Dictionary(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_and_reload(self):
        self.dict.add("Øresund Bridge")
        self.dict.add("Business Central")
        self.assertEqual(len(self.dict), 2)

        # Reload from saved file
        reloaded = sd.Dictionary(self.test_dir)
        self.assertEqual(len(reloaded), 2)
        words = reloaded.words()
        self.assertIn("Øresund Bridge", words)
        self.assertIn("Business Central", words)

    def test_note_usage_word_boundary(self):
        self.dict.add("SAP")
        self.assertEqual(self.dict.entries[sd._fold("SAP")]["uses"], 0)

        # "sapphire" must NOT increment usage of "SAP"
        self.dict.note_usage("This is a blue sapphire stone.")
        self.assertEqual(self.dict.entries[sd._fold("SAP")]["uses"], 0)

        # "SAP" as distinct word must increment
        self.dict.note_usage("Check the SAP report.")
        self.assertEqual(self.dict.entries[sd._fold("SAP")]["uses"], 1)

    def test_casing_application(self):
        self.dict.add("Øresund Bridge")
        self.dict.add("PowerBI")
        # Exact casing should be enforced post-transcription
        text = "Welcome to øresund bridge and check powerbi."
        fixed = self.dict.apply_casing(text)
        self.assertEqual(fixed, "Welcome to Øresund Bridge and check PowerBI.")

    def test_lowercase_entries_never_lowercase_a_sentence_start(self):
        self.dict.add("excel")
        self.dict.add("machine learning")
        self.assertEqual(self.dict.apply_casing("Machine learning in Excel."), "Machine learning in Excel.")
        self.assertEqual(self.dict.apply_casing("Done.\nMachine learning next."), "Done.\nMachine learning next.")
        text, hint, applied, needs_llm = self.dict.prepare("Excel is open.", "en")
        # Qwen hears to keep the capital, never "keep 'excel'".
        self.assertEqual((text, needs_llm), ("Excel is open.", False))
        self.assertIn("keep 'Excel'", hint)
        self.assertNotIn("'excel'", hint)
        # Never 'keep Excel' and 'Excel is spelled excel' at once.
        hint = self.dict.prepare("Excel is open, open Excel, open excel", "en")[1]
        self.assertIn("'Excel' is spelled 'excel'", hint)
        self.assertNotIn("keep 'Excel'", hint)

    def test_lowercase_entries_keep_capitals_after_bullets_quotes_and_colons(self):
        self.dict.add("machine learning")
        for text in ("- Machine learning for all.", "\u2022 Machine learning for all.",
                     '"Machine learning is hard."', "Done\u2026 Machine learning next.",
                     "Note: Machine learning rocks.", "(Machine learning) is next."):
            self.assertEqual(self.dict.apply_casing(text), text)
        # At a sentence start only the first letter is the sentence's.
        self.assertEqual(self.dict.apply_casing("Machine Learning is hard."), "Machine learning is hard.")
        self.dict.add("excel")
        self.assertNotIn("'excel'", self.dict.prepare("- Excel is open.", "en")[1])
        # A sentence closed inside brackets or quotes.
        for text in ("(See above.) Machine learning next.", 'He said "done." Machine learning next.'):
            self.assertEqual(self.dict.apply_casing(text), text)
        # Mid-sentence look-alikes are not sentence starts.
        self.assertEqual(self.dict.apply_casing("Ratio 3:1 Machine Learning."), "Ratio 3:1 machine learning.")
        self.assertEqual(self.dict.apply_casing("the e-Machine Learning"), "the e-machine learning")

    def test_lowercase_entries_still_apply_mid_sentence(self):
        self.dict.add("webshop")
        self.dict.add("machine learning")
        self.assertEqual(self.dict.apply_casing("We use Machine Learning."), "We use machine learning.")
        text, hint, applied, needs_llm = self.dict.prepare("Open the Webshop now", "en")
        self.assertIn("'Webshop' is spelled 'webshop'", hint)   # for Qwen, if it runs
        self.assertFalse(needs_llm)                             # but not worth waiting for

    def test_every_occurrence_counts(self):
        self.dict.add("Jira")
        text, hint, applied, needs_llm = self.dict.prepare("Jira and jira", "en")
        self.assertIn("'jira' is spelled 'Jira'", hint)
        self.assertTrue(needs_llm)

    def test_longest_entry_wins_a_span(self):
        self.dict.add("Mette")
        self.dict.add("Mette-Kristine")
        text, hint, applied, needs_llm = self.dict.prepare("ring til mette-kristine", "da")
        self.assertNotIn("'mette' staves", hint)
        self.assertFalse(needs_llm)   # apply_casing fixes Mette-Kristine
        self.assertEqual(self.dict.apply_casing("ring til mette-kristine"), "ring til Mette-Kristine")
        self.assertTrue(self.dict.prepare("ring til mette og Mette-Kristine", "da")[3])

    def test_neighbouring_entries_are_both_seen(self):
        self.dict.add("New York")
        self.dict.add("York Times")
        hint = self.dict.prepare("the new york times", "en")[1]
        self.assertIn("'new york' is spelled 'New York'", hint)
        self.assertIn("'york times' is spelled 'York Times'", hint)

    def _hold_file(self, path, seconds):
        """Opens path with no sharing (as an antivirus scan or backup can) for
        `seconds`, on a thread. Returns once the file is held."""
        import ctypes
        import threading
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        GENERIC_READ, OPEN_EXISTING = 0x80000000, 3
        handle = k32.CreateFileW(path, GENERIC_READ, 0, None, OPEN_EXISTING, 0, None)
        self.assertNotIn(handle, (None, wintypes.HANDLE(-1).value))
        t = threading.Timer(seconds, k32.CloseHandle, args=(handle,))
        t.start()
        self.addCleanup(t.join)
        return t

    def test_locked_file_at_start_is_never_overwritten(self):
        self.dict.add("Annika")
        self.dict.add("Mikkel")
        with open(self.dict.path, encoding="utf-8") as f:
            before = f.read()
        held = self._hold_file(self.dict.path, 1.5)   # longer than load() retries
        locked = sd.Dictionary(self.test_dir)
        self.assertEqual(len(locked), 0)
        self.assertTrue(locked.read_only)   # so the app can say nothing is saved
        held.join()
        locked.add("Learned")   # would have replaced both words with this one
        with open(self.dict.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual([n for n in os.listdir(self.test_dir) if ".corrupt-" in n], [])

    def test_briefly_locked_file_is_read_after_a_retry(self):
        self.dict.add("Annika")
        self._hold_file(self.dict.path, 0.15)
        self.assertEqual(sd.Dictionary(self.test_dir).words(), ["Annika"])

    def test_failed_save_leaves_no_temp_file(self):
        self.dict.add("Annika")
        held = self._hold_file(self.dict.path, 1.0)   # longer than save() retries
        self.dict.add("Mikkel")
        held.join()
        self.assertFalse(os.path.exists(self.dict.path + ".tmp"))
        self.dict.save()   # the next save brings the file up to date
        self.assertEqual(len(sd.Dictionary(self.test_dir)), 2)

    def test_concurrent_saves_never_leave_a_broken_file(self):
        import threading
        for i in range(300):
            self.dict.entries[sd._fold(f"Word{i}")] = {"word": f"Word{i}", "source": "manual",
                                                      "uses": 0, "added": "2026-10-14"}
        broken, stop = [], threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    with open(self.dict.path, encoding="utf-8") as f:
                        json.load(f)
                except OSError:
                    pass   # being replaced right now
                except ValueError as e:
                    broken.append(str(e))
                stop.wait(0.002)   # a busy reader, but not one that never lets go

        def usage():
            for _ in range(40):
                self.dict.note_usage("Word1 and Word2")

        def learn():
            for i in range(40):
                self.dict.add(f"Learned{i}")
        self.dict.save()
        threads = [threading.Thread(target=f) for f in (reader, usage, learn)]
        for t in threads:
            t.start()
        threads[1].join()
        threads[2].join()
        stop.set()
        threads[0].join()
        self.assertEqual(broken, [])
        self.assertEqual(len(sd.Dictionary(self.test_dir)), 340)

    def test_wrong_shaped_file_is_kept_aside(self):
        for content in ('{"words": "abc"}', '{"words": null}', '{"words": [{"word": "x", "uses": "many"}]}', "[]"):
            with open(self.dict.path, "w", encoding="utf-8") as f:
                f.write(content)
            reloaded = sd.Dictionary(self.test_dir)   # must not raise
            self.assertEqual(len(reloaded), 0)
            self.assertFalse(reloaded.read_only)
            self.assertFalse(os.path.exists(self.dict.path), content)
        self.assertEqual(len([n for n in os.listdir(self.test_dir) if ".corrupt-" in n]), 4)

    def test_unreadable_file_is_kept_aside_not_overwritten(self):
        with open(self.dict.path, "w", encoding="utf-8") as f:
            f.write('{"words": [{"word": "Annika"')   # cut off mid-write
        reloaded = sd.Dictionary(self.test_dir)
        self.assertEqual(len(reloaded), 0)
        kept = [n for n in os.listdir(self.test_dir) if ".corrupt-" in n]
        self.assertEqual(len(kept), 1)
        with open(os.path.join(self.test_dir, kept[0]), encoding="utf-8") as f:
            self.assertIn("Annika", f.read())

    def test_correctly_spelled_words_do_not_need_the_llm(self):
        self.dict.add("Jira")
        self.dict.add("Service")
        text, hint, applied, needs_llm = self.dict.prepare("Open Jira.", "en")
        self.assertIn("keep 'Jira'", hint)        # still a hint if Qwen runs anyway
        self.assertFalse(needs_llm)
        text, hint, applied, needs_llm = self.dict.prepare("Service is down.", "en")
        self.assertFalse(needs_llm)

    def test_casing_fixed_by_apply_casing_does_not_need_the_llm(self):
        self.dict.add("SAP")
        self.dict.add("Business Central")
        for raw in ("Check sap.", "Open business central."):
            text, hint, applied, needs_llm = self.dict.prepare(raw, "en")
            self.assertIn("is spelled", hint)
            self.assertFalse(needs_llm, raw)

    def test_name_casing_and_unconfirmed_mishearings_need_the_llm(self):
        self.dict.add("Mette")
        # apply_casing leaves 'mette' alone (it would also capitalize ordinary
        # words), so only Qwen can fix it.
        self.assertTrue(self.dict.prepare("ring til mette", "da")[3])
        self.dict.record_correction("Anika", "Annika")   # seen once, no context yet
        text, hint, applied, needs_llm = self.dict.prepare("Ask Anika.", "en")
        self.assertEqual(applied, [])
        self.assertIn("'Anika' is spelled 'Annika'", hint)
        self.assertTrue(needs_llm)

    def test_casing_is_enforced(self):
        enforced = sd.Dictionary.casing_is_enforced
        for word in ("SAP", "iPhone", "PowerBI", "Business Central", "KPI", "machine learning"):
            self.assertTrue(enforced(word), word)
        for word in ("Mette", "Service", "excel"):
            self.assertFalse(enforced(word), word)


class TestSpeakConfigAndFormatting(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cfg_file = os.path.join(self.test_dir, "test_config.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_config_atomic_save_and_corrupt_recovery(self):
        # 1. Save valid config
        data = {"hotkey": "F9", "whisper_threads": 8}
        speak.save_config(data, self.cfg_file)
        loaded = speak.load_config(self.cfg_file)
        self.assertEqual(loaded["hotkey"], "F9")
        self.assertEqual(loaded["whisper_threads"], 8)

        # 2. Corrupt the file
        with open(self.cfg_file, "w", encoding="utf-8") as f:
            f.write("{ INVALID JSON ...")

        # 3. Load should recover defaults and preserve corrupted backup
        recovered = speak.load_config(self.cfg_file)
        self.assertEqual(recovered["hotkey"], "<")  # Default fallback
        # Ensure a backup file was created
        backups = [f for f in os.listdir(self.test_dir) if "corrupt" in f]
        self.assertTrue(len(backups) >= 1)

    def test_hotkey_normalization_and_combo_parsing(self):
        dummy_app = object.__new__(speak.SpeakApp)
        self.assertEqual(dummy_app._normalize_key("<"), "<")
        self.assertEqual(dummy_app._normalize_key(">"), ">")
        self.assertEqual(dummy_app._normalize_key("f8"), "F8")
        self.assertEqual(dummy_app._normalize_key("<>"), "<>")

        mods, vk = dummy_app._parse_combo("CTRL+ALT+L")
        self.assertEqual(mods, speak.MOD_CONTROL | speak.MOD_ALT)
        self.assertEqual(vk, ord("L"))

        mods_f, vk_f = dummy_app._parse_combo("CTRL+SHIFT+F9")
        self.assertEqual(mods_f, speak.MOD_CONTROL | speak.MOD_SHIFT)
        self.assertEqual(vk_f, speak.FUNCTION_KEYS["F9"])

    def test_split_for_llm(self):
        dummy_app = object.__new__(speak.SpeakApp)
        # Short text produces single chunk
        chunks = dummy_app._split_for_llm("Short sentence here.")
        self.assertEqual(len(chunks), 1)

        # Construct long multi-sentence text
        long_text = ". ".join(["This is sentence number " + str(i) for i in range(100)]) + "."
        chunks_long = dummy_app._split_for_llm(long_text)
        self.assertTrue(len(chunks_long) > 1)
        # Reassembled words should match
        total_orig_words = len(long_text.split())
        total_chunked_words = sum(len(c.split()) for c in chunks_long)
        self.assertEqual(total_orig_words, total_chunked_words)


class TestModelDownload(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.src = os.path.join(self.tmpdir, "source.gguf")
        with open(self.src, "wb") as f:
            f.write(b"model bytes" * 1000)
        import hashlib
        with open(self.src, "rb") as f:
            self.sha = hashlib.sha256(f.read()).hexdigest()
        self.url = "file:///" + self.src.replace("\\", "/")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_download_verifies_checksum_and_moves_into_place(self):
        dest = os.path.join(self.tmpdir, "llm", "model.gguf")
        self.assertTrue(speak.download_file(self.url, dest, self.sha, "test model"))
        self.assertTrue(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_download_with_wrong_checksum_leaves_nothing_behind(self):
        dest = os.path.join(self.tmpdir, "llm", "model.gguf")
        self.assertFalse(speak.download_file(self.url, dest, "0" * 64, "test model"))
        self.assertFalse(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_existing_model_is_not_downloaded_again(self):
        old_dir = speak.LOCAL_LLM_DIR
        speak.LOCAL_LLM_DIR = self.tmpdir
        try:
            existing = os.path.join(self.tmpdir, speak.LOCAL_LLM_MODELS["0.5b"]["file"])
            open(existing, "wb").close()
            self.assertEqual(speak.ensure_local_llm("0.5b"), existing)
        finally:
            speak.LOCAL_LLM_DIR = old_dir

    def test_removed_gemini_mode_falls_back_to_local(self):
        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.config = {"ai_mode": "gemini", "local_model": "0.5b"}
        old = speak.ensure_local_llm
        speak.ensure_local_llm = lambda choice: None   # no model: raw Whisper, no download
        try:
            app._init_ai()
        finally:
            speak.ensure_local_llm = old
        self.assertEqual(app.ai_mode, "local")
        self.assertIsNone(app.local_llm)


class TestMeetingHelpers(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(sm.format_timestamp(0), "00:00:00")
        self.assertEqual(sm.format_timestamp(75.9), "00:01:15")
        self.assertEqual(sm.format_timestamp(3725), "01:02:05")
        self.assertEqual(sm.format_timestamp(-3), "00:00:00")

    def test_match_me_accepts_every_cluster_above_threshold(self):
        me = [1.0, 0.0, 0.0]
        embs = {0: [0.9, 0.1, 0.0], 1: [0.1, 1.0, 0.0], 3: [0.8, 0.0, 0.3]}
        self.assertEqual(sm.match_me(embs, me, 0.5), {0, 3})   # one voice split in two clusters
        self.assertEqual(sm.match_me(embs, me, 0.999), set())   # best cluster scores 0.994
        self.assertEqual(sm.match_me(embs, None, 0.5), set())

    def test_drop_call_echo(self):
        system = [(10.0, 20.0, 0)]
        mic = [(0.0, 5.0, 1),      # in-room person, no call audio: kept
               (11.0, 15.0, 1),    # same cluster during call audio: echo, dropped
               (12.0, 14.0, 0),    # "Me" talking over the call: always kept
               (18.0, 24.0, 2)]    # only 2 of 6 s overlap: kept
        kept, dropped = sm.drop_call_echo(mic, system, me_clusters={0})
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [(0.0, 5.0, 1), (12.0, 14.0, 0), (18.0, 24.0, 2)])

    def test_label_speakers_numbers_in_order_of_appearance(self):
        keys = [("system", 4), ("mic", 0), ("system", 1), ("system", 4), ("mic", 2)]
        labels = sm.label_speakers(keys, me_keys={("mic", 0)})
        self.assertEqual(labels, {("system", 4): "Person 1", ("mic", 0): "Me",
                                  ("system", 1): "Person 2", ("mic", 2): "Person 3"})

    def test_language_ranges_group_each_language_across_switches(self):
        segs = [(5.0, 8.0, "en"), (0.0, 4.9, "en"), (9.0, 12.0, "da"), (13.0, 15.0, "da"), (16.0, 18.0, "en")]
        self.assertEqual(sm.language_ranges(segs, pad=0.2),
                         {"en": [(0.0, 8.2), (15.8, 18.2)], "da": [(8.8, 12.2), (12.8, 15.2)]})
        self.assertEqual(sm.language_ranges([]), {})

    def test_build_track_and_map_back_to_meeting_time(self):
        audio = np.arange(10 * sm.SAMPLE_RATE, dtype=np.float32)
        track, pieces = sm.build_track(audio, [(1.0, 2.0), (5.0, 7.0)], gap=0.5)
        self.assertEqual(len(track), int(3.5 * sm.SAMPLE_RATE))
        self.assertEqual(pieces, [(0.0, 1.0, 1.0), (1.5, 5.0, 2.0)])
        self.assertEqual(float(track[int(1.5 * sm.SAMPLE_RATE)]), float(audio[5 * sm.SAMPLE_RATE]))
        self.assertAlmostEqual(sm.track_to_meeting_time(0.5, pieces), 1.5)
        self.assertAlmostEqual(sm.track_to_meeting_time(2.5, pieces), 6.0)
        # A word starting in the gap belongs to the next piece (Whisper stamps
        # the first word after a pause early); a word ending there, to the previous one.
        self.assertAlmostEqual(sm.track_to_meeting_time(1.1, pieces), 5.0)
        self.assertAlmostEqual(sm.track_to_meeting_time(1.1, pieces, is_start=False), 2.0)
        self.assertAlmostEqual(sm.track_to_meeting_time(9.0, pieces), 7.0)    # past the end: clamped
        self.assertEqual(sm.build_track(audio, [])[1], [])

    def test_assign_words(self):
        segments = [(0.0, 3.0, "A"), (3.2, 6.0, "B"), (5.5, 9.0, "C")]
        words = [(0.5, 0.9, " Hello", "en"),    # inside A
                 (3.0, 3.6, " there", "en"),    # straddles: more overlap with B
                 (5.6, 5.9, " both", "en"),     # equal overlap with B and C: the later segment wins
                 (9.5, 9.8, " late", "en"),     # 0.5 s after C: nearest
                 (20.0, 20.4, " echo", "en")]   # far from any kept segment: dropped
        out = sm.assign_words(words, segments)
        self.assertEqual([(w[2], w[4]) for w in out],
                         [(" Hello", "A"), (" there", "B"), (" both", "C"), (" late", "C")])

    def test_build_turns(self):
        words = [(0.0, 0.4, " Hello", "en", "A"), (0.5, 0.9, " everyone.", "en", "A"),
                 (1.5, 1.9, " Hi!", "en", "B"),
                 (2.0, 2.5, " Next", "en", "A"), (9.0, 9.5, " point.", "en", "A")]   # 6.5 s pause: new turn
        turns = sm.build_turns(words)
        self.assertEqual([(t["speaker"], t["text"]) for t in turns],
                         [("A", "Hello everyone."), ("B", "Hi!"), ("A", "Next"), ("A", "point.")])
        self.assertEqual(turns[0]["start"], 0.0)
        self.assertEqual(turns[0]["end"], 0.9)

    def test_chunk_lines(self):
        lines = ["a b c", "d e", "f g h i", "j"]
        count = lambda s: len(s.split())
        chunks = sm.chunk_lines(lines, count, budget=8)   # each line costs its words + 1 (newline)
        self.assertEqual(chunks, ["a b c\nd e", "f g h i\nj"])
        self.assertEqual(sm.chunk_lines(["one very long line here"], count, 2), ["one very long line here"])

    def test_render_markdown(self):
        info = {"title": "Meeting 2026-10-12 14:30", "date": "Monday", "duration": "5 min",
                "sources": "Microphone (in-room)", "speakers": "Me, Person 1", "languages": "English",
                "notes": ["A note."]}
        turns = [{"start": 3, "label": "Me", "text": "Hello."}, {"start": 65, "label": "Person 1", "text": "Hi."}]
        md = sm.render_markdown(info, turns, "## Summary\n\nShort.")
        self.assertIn("# Meeting 2026-10-12 14:30", md)
        self.assertIn("> A note.", md)
        self.assertIn("**[00:00:03] Me:** Hello.", md)
        self.assertIn("**[00:01:05] Person 1:** Hi.", md)
        self.assertLess(md.index("## Summary"), md.index("## Transcript"))
        self.assertNotIn(chr(0x2014), md)   # house style: no em dashes
        self.assertIn("No speech was detected", sm.render_markdown(info, [], None))

    MEETING = [
        ("Me", 31, "Okay, so the photos and the discount bug are the two blockers. Zira, can you chase the studio today?"),
        ("Person 1", 40, "Yes, I will call them after this meeting and send an update before lunch."),
        ("Person 2", 57, "I checked the stock report yesterday. The ceramic bowls are low everywhere."),
        ("Me", 66, "Good point. Let us order more bowls. David, please send me the numbers and I will place the order on Friday."),
        ("Person 3", 75, "Det lyder godt. Jeg giver butikkerne besked."),
    ]

    def _turns(self):
        return [{"label": l, "start": s, "text": t} for l, s, t in self.MEETING]

    def test_split_clauses_cuts_at_promises_requests_and_agreements(self):
        turns = self._turns()
        self.assertEqual(sm.split_clauses(turns[3]["text"]), [
            ("decision", "Let us order more bowls."),
            ("request", "David, please send me the numbers and"),
            ("promise", "I will place the order on Friday.")])
        self.assertEqual(sm.split_clauses(turns[0]["text"]), [("request", "Zira, can you chase the studio today?")])
        self.assertEqual(sm.split_clauses(turns[4]["text"]), [("promise", "Jeg giver butikkerne besked.")])
        self.assertEqual(sm.split_clauses(turns[2]["text"]), [])           # a status update
        self.assertEqual(sm.split_clauses("We agreed to wait."), [("decision", "We agreed to wait.")])   # one cue, not two
        # Seen in the 10-minute test: a clause ran on into the next sentences.
        self.assertEqual(sm.split_clauses("Let us meet on Thursday. Thanks everyone. Today I want to review the launch."),
                         [("decision", "Let us meet on Thursday.")])
        self.assertEqual([c[0] for c in sm.minute_clauses(turns)],
                         [0, 1, 3, 3, 3, 4])

    def test_context_only_for_answers(self):
        turns = self._turns()
        ctx = sm.context_for(turns, 1, turns[1]["text"])
        self.assertEqual(ctx.splitlines()[0], "[Me] " + turns[0]["text"])   # "Yes, I will call them": needs the question
        self.assertTrue(ctx.splitlines()[1].startswith(">> [Person 1] Yes, I will call them"))
        self.assertEqual(sm.context_for(turns, 3, "I will place the order on Friday."),
                         ">> [Me] I will place the order on Friday.")

    def test_owner_and_grounding(self):
        turns = self._turns()
        self.assertEqual(sm.agreed_by_next(turns, 0), "Person 1")
        self.assertIsNone(sm.agreed_by_next(turns, 3))       # "Det lyder godt" is not a yes to the request
        self.assertTrue(sm.grounded("Call the studio", "Yes, I will call them after this meeting."))
        self.assertTrue(sm.grounded("Sende tilbuddet", "Jeg sender tilbuddet i morgen."))   # stem match
        self.assertFalse(sm.grounded("Send the numbers for the bowls", "Det lyder godt."))

    def test_parse_clause_answer(self):
        self.assertEqual(sm.parse_clause_answer('{"keep": true, "text": "Call the studio", "due": "today"}'),
                         {"keep": True, "text": "Call the studio", "due": "today"})
        self.assertEqual(sm.parse_clause_answer('Answer: {"keep": true, "text": "Fix it", "due": ""} done'),
                         {"keep": True, "text": "Fix it", "due": ""})
        self.assertFalse(sm.parse_clause_answer('{"keep": "yes", "text": "x"}')["keep"])   # only a real true
        for bad in ("", "no json here", '{"keep": true, "text": ', "[1, 2]"):
            self.assertIsNone(sm.parse_clause_answer(bad))

    def _item(self, line, kind, clause, text, due="", keep=True):
        return {"line": line, "kind": kind, "clause": clause, "keep": keep, "text": text, "due": due}

    def test_render_minutes(self):
        items = [
            self._item(0, "request", "Zira, can you chase the studio today?", "Chase the studio", "today"),
            self._item(1, "promise", "Yes, I will call them after this meeting and send an update before lunch.",
                       "call the studio and send an update.", "before lunch"),
            self._item(3, "decision", "Let us order more bowls.", "order more bowls"),
            self._item(3, "request", "David, please send me the numbers and", "Send the numbers", "Friday"),
            self._item(3, "promise", "I will place the order on Friday.", "Place the order", "Friday"),
            self._item(3, "promise", "I will place the order on Friday.", "Place the order quickly", "Friday"),
            self._item(4, "promise", "Jeg giver butikkerne besked.", "Send the numbers to David"),   # copied: dropped
            self._item(4, "decision", "Agreed.", "Agree"),                                          # bare: dropped
            self._item(2, "promise", "I checked", "Nothing", keep=False),
            self._item(99, "promise", "x", "Invented thing"),                                       # no such line
        ]
        md = sm.render_minutes("  The team reviewed\nthe launch blockers. ", items, self._turns())
        self.assertIn("## Summary\n\nThe team reviewed the launch blockers.", md)
        self.assertIn("- Order more bowls *(Me, 00:01:06)*", md)
        self.assertNotIn("Chase the studio", md)            # accepted request: listed under the promise
        self.assertIn("- [ ] **Person 1:** Call the studio and send an update *(due: before lunch, said at 00:00:40)*", md)
        # "Friday" belongs to the promise clause, not to the request.
        self.assertIn("- [ ] Send the numbers *(no owner agreed; raised by Me, said at 00:01:06)*", md)
        self.assertIn("- [ ] **Me:** Place the order *(due: Friday, said at 00:01:06)*", md)
        self.assertEqual(md.count("Place the order"), 1)    # near-duplicate merged
        for gone in ("to David", "Agree*", "Nothing", "Invented"):
            self.assertNotIn(gone, md)
        empty = sm.render_minutes("", [], self._turns())
        self.assertEqual(empty.count("_None recorded._"), 2)
        self.assertNotIn("## Summary", empty)


class TestMeetingEchoAndFidelity(unittest.TestCase):
    def test_faithful_accepts_filler_removal_and_rejects_rewrites(self):
        raw = "um so I will uh call them after this meeting"
        self.assertTrue(sm.faithful(raw, "So I will call them after this meeting."))
        # The real failure seen in testing: a Danish sentence rewritten with a new meaning.
        self.assertFalse(sm.faithful("Jeg giver butikkerne besked om at der kommer flere skåle",
                                     "Butikkerne har beskrevet, at der kommer flere skåle"))
        self.assertFalse(sm.faithful("we need the numbers by friday and the photos today", "Numbers by Friday."))
        self.assertTrue(sm.faithful("", ""))
        # Both seen in the end-to-end test: a swapped subject, and a compressed question.
        self.assertFalse(sm.faithful("Det lyder godt. Jeg giver butikkerne besked om, at der kommer flere skåle i næste uge.",
                                     "Det lyder godt. Du giver besked om, at der kommer flere skåle i næste uge."))
        self.assertFalse(sm.faithful("Okay, so the photos and the discount bug are the two blockers. Zira, can you "
                                     "chase the studio today and give us a date?",
                                     "Okay, photos and discount bug are blockers. Zira, chase studio today and give us a date."))
        self.assertTrue(sm.faithful("øh altså jeg tror at vi vi skal flytte det",
                                    "Jeg tror, at vi skal flytte det."))   # fillers and a repeat

    def _envelope_signal(self, seconds, seed):
        rng = np.random.default_rng(seed)
        n = seconds * sm.SAMPLE_RATE
        syllables = np.repeat(rng.uniform(0, 1, n // 1600 + 1), 1600)[:n]   # 100 ms "syllables"
        return (rng.normal(0, 1, n) * syllables * 0.2).astype(np.float32)

    def test_echo_correlation_detects_speaker_leak(self):
        system = self._envelope_signal(20, seed=1)
        segments = [(0.0, 20.0, 0)]
        delay = int(0.08 * sm.SAMPLE_RATE)
        leaky_mic = np.zeros_like(system)
        leaky_mic[delay:] = 0.3 * system[:-delay]
        leaky_mic += np.random.default_rng(2).normal(0, 0.003, len(system)).astype(np.float32)
        headset_mic = self._envelope_signal(20, seed=3)      # someone else's speech, unrelated
        self.assertGreater(sm.echo_correlation(leaky_mic, system, segments), sm.ECHO_CORRELATION)
        self.assertLess(sm.echo_correlation(headset_mic, system, segments), sm.ECHO_CORRELATION)
        self.assertEqual(sm.echo_correlation(leaky_mic, system, []), 0.0)

    def test_mute_segments_pads_for_speaker_delay(self):
        audio = np.ones(3 * sm.SAMPLE_RATE, dtype=np.float32)
        out = sm.mute_segments(audio, [(1.0, 1.5, 0)], before=0.1, after=0.3)
        self.assertEqual(float(out[int(0.85 * sm.SAMPLE_RATE)]), 1.0)
        self.assertEqual(float(np.abs(out[int(0.9 * sm.SAMPLE_RATE):int(1.8 * sm.SAMPLE_RATE)]).max()), 0.0)
        self.assertEqual(float(out[int(1.85 * sm.SAMPLE_RATE)]), 1.0)
        self.assertIs(out, audio)   # in place: no second copy of a long stream

    def test_drop_echo_words(self):
        system = [(10.0, 10.3, " Fixed", "en"), (10.4, 10.7, " by", "en"), (10.8, 11.2, " Thursday.", "en")]
        mic = [(10.1, 10.4, " fixed", "en"),       # echo: same word 0.1 s later
               (10.9, 11.3, " Thursday", "en"),    # echo
               (11.0, 11.3, " Agreed", "en"),      # someone in the room: kept
               (20.0, 20.3, " fixed", "en")]       # same word, much later: kept
        kept = sm.drop_echo_words(mic, system)
        self.assertEqual([w[2] for w in kept], [" Agreed", " fixed"])

    def test_merge_similar_clusters(self):
        embs = {0: [1.0, 0.0, 0.0], 3: [0.95, 0.1, 0.0], 5: [0.0, 1.0, 0.0], 7: [0.9, 0.0, 0.2]}
        self.assertEqual(sm.merge_similar_clusters(embs, 0.6), {0: 0, 3: 0, 5: 5, 7: 0})
        self.assertEqual(sm.merge_similar_clusters(embs, 0.999), {0: 0, 3: 3, 5: 5, 7: 7})
        self.assertEqual(sm.merge_similar_clusters({}, 0.6), {})


class TestMeetingTurnRepair(unittest.TestCase):
    def _turn(self, speaker, start, end, text):
        return {"speaker": speaker, "start": start, "end": end, "lang": "en", "text": text}

    def test_fix_split_sentences_moves_short_tail_to_next_speaker(self):
        # Seen in the in-room test: Zira's "One more thing" was cut at the speaker change.
        turns = [self._turn("Helle", 75.5, 82.6, "Jeg giver butikkerne besked. One more"),
                 self._turn("Zira", 82.7, 89.0, "thing, the newsletter goes out on Monday.")]
        fixed = sm.fix_split_sentences(turns)
        self.assertEqual(fixed[0]["text"], "Jeg giver butikkerne besked.")
        self.assertEqual(fixed[1]["text"], "One more thing, the newsletter goes out on Monday.")

    def test_fix_split_sentences_moves_dangling_word(self):
        # Seen in the 10-minute test: David's "I" ended Zira's turn.
        turns = [self._turn("Zira", 40.2, 46.0, "I will call them before lunch. I"),
                 self._turn("David", 46.2, 52.0, "Checked the stock report yesterday.")]
        fixed = sm.fix_split_sentences(turns)
        self.assertEqual(fixed[0]["text"], "I will call them before lunch.")
        self.assertEqual(fixed[1]["text"], "I checked the stock report yesterday.")

    def test_fix_split_sentences_never_moves_words_between_streams(self):
        # Seen in the 10-minute test: a call word was moved onto a person at the mic.
        turns = [self._turn(("system", 1), 40.2, 45.4, "Send an update before lunch. I"),
                 self._turn(("mic", 3), 45.6, 52.0, "Hej alle sammen.")]
        fixed = sm.fix_split_sentences(turns)
        self.assertEqual([t["text"] for t in fixed], ["Send an update before lunch. I", "Hej alle sammen."])

    def test_fix_split_sentences_leaves_normal_turns_alone(self):
        turns = [self._turn("A", 0, 5, "We should ship it. Agreed by all"),
                 self._turn("B", 5.2, 8, "Yes, that works."),                # capitalized: a new sentence
                 self._turn("A", 20, 22, "So the plan is set. Then"),
                 self._turn("B", 30, 32, "maybe later.")]                     # 8 s gap: not a cut sentence
        before = [t["text"] for t in turns]
        self.assertEqual([t["text"] for t in sm.fix_split_sentences(turns)], before)
        long_tail = [self._turn("A", 0, 5, "Fine. This tail is far too long to move"),
                     self._turn("B", 5.1, 8, "because it ends here.")]
        self.assertEqual(sm.fix_split_sentences(long_tail)[0]["text"], "Fine. This tail is far too long to move")

    def test_needs_cleanup(self):
        self.assertTrue(sm.needs_cleanup("So um we should ship it"))
        self.assertTrue(sm.needs_cleanup("Jeg tror øh at det er fint"))
        self.assertTrue(sm.needs_cleanup("we we should ship it"))
        self.assertFalse(sm.needs_cleanup("We should ship it on Friday."))
        self.assertFalse(sm.needs_cleanup("I mean the plan, like the old one."))   # normal words, not disfluencies

    def test_needs_cleanup_is_language_aware(self):
        # "er" is "is" in Danish: it once sent nearly every Danish turn to Qwen.
        self.assertFalse(sm.needs_cleanup("Det er fint, vi ses i morgen.", "da"))
        self.assertFalse(sm.needs_cleanup("Er ist heute nicht da.", "de"))
        self.assertTrue(sm.needs_cleanup("So er we should ship it", "en"))


class TestMeetingAudioFiles(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write(self, path, samples):
        w = sm._WavWriter(path)
        w.write(np.asarray(samples, dtype=np.float32))
        w.close()

    def test_wav_round_trip(self):
        p = os.path.join(self.tmpdir, "a.wav")
        self._write(p, [0.0, 0.5, -0.5, 1.5])   # 1.5 is clipped
        audio = sm.read_wav(p)
        np.testing.assert_allclose(audio, [0.0, 0.5, -0.5, 1.0], atol=1e-3)

    def test_unfinalized_wav_is_still_readable(self):
        # A crash mid-meeting leaves the header saying 0 frames.
        p = os.path.join(self.tmpdir, "crash.wav")
        self._write(p, np.linspace(-0.5, 0.5, 1600))
        with open(p, "r+b") as f:
            f.seek(40)
            f.write((0).to_bytes(4, "little"))
        self.assertEqual(len(sm.read_wav(p)), 1600)

    def test_load_streams_lines_up_offsets_and_detects_own_recordings(self):
        mgr = sm.MeetingManager.__new__(sm.MeetingManager)
        mgr.recordings_dir = os.path.join(self.tmpdir, "recordings")
        folder = os.path.join(mgr.recordings_dir, "2026-10-12_14-30-00")
        os.makedirs(folder)
        self._write(os.path.join(folder, "mic.wav"), np.full(16000, 0.1))
        self._write(os.path.join(folder, "system.wav"), np.full(16000, 0.2))
        with open(os.path.join(folder, "meeting.json"), "w", encoding="utf-8") as f:
            json.dump({"started": "2026-10-12T14:30:00", "offsets": {"mic": 0.0, "system": 0.25}}, f)
        streams, meta, own = mgr._load_streams(folder)
        lengths = {name: len(a) for name, a in streams}
        self.assertEqual(lengths, {"mic": 16000, "system": 16000 + 4000})
        self.assertTrue(own)
        self.assertEqual(meta["started"], "2026-10-12T14:30:00")
        other = os.path.join(self.tmpdir, "imported")
        os.makedirs(other)
        self._write(os.path.join(other, "mic.wav"), np.zeros(100))
        self.assertFalse(mgr._load_streams(other)[2])   # never delete folders Speak did not create


class TestSharedCleanup(unittest.TestCase):
    class FakeDict:
        def __init__(self):
            self.noted = []

        def prepare(self, text, lang):
            return text.replace("sales force", "Salesforce"), "", [("sales force", "Salesforce")], False

        def apply_casing(self, text):
            return text

        def note_usage(self, text):
            self.noted.append(text)

    def test_cleanup_never_counts_usage(self):
        # Meeting turns must not rank the user's words, and dictation counts
        # them in the worker, after the text is in.
        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.config = {}
        app.dictionary = self.FakeDict()
        app.ai_mode, app.local_llm = "off", None
        self.assertEqual(app.clean_text("we use sales force", "en", verbose=False), "we use Salesforce")
        app.clean_text("we use sales force", "en")
        self.assertEqual(app.dictionary.noted, [])

    def test_worker_counts_usage_after_insertion_without_locks(self):
        import threading

        class TrackedLock:
            def __init__(self):
                self.held = False

            def __enter__(self):
                self.held = True

            def __exit__(self, *exc):
                self.held = False

        events = []
        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.config = {"last_language": "en"}
        app.engine_lock, app.clipboard_lock = TrackedLock(), TrackedLock()
        app.dictionary = self.FakeDict()
        app.learn_hotkey_req = "<ctrl>+<alt>+l"
        app.play_sound = lambda kind: None
        app.transcribe = lambda audio: ("we use sales force", "en", 0.99)
        app.clean_text = lambda raw, lang: "we use Salesforce"
        app._schedule_clipboard_restore = lambda token: None
        app.insert_text = lambda text: events.append("insert")

        def note_usage(text):
            events.append(("note", text, app.engine_lock.held, app.clipboard_lock.held))
        app.dictionary.note_usage = note_usage
        app._process_worker(np.zeros(16000, dtype=np.float32))
        self.assertEqual(events, ["insert", ("note", "we use Salesforce", False, False)])


class TestFastPath(unittest.TestCase):
    """speak_text.needs_llm decides whether a dictation waits for Qwen."""

    def assertFast(self, text, lang):
        self.assertFalse(st.needs_llm(st.strip_leading_fillers(text), lang), text)

    def assertSlow(self, text, lang):
        self.assertTrue(st.needs_llm(st.strip_leading_fillers(text), lang), text)

    def test_everyday_short_replies_skip_the_llm(self):
        # These were all sent to Qwen because "no", "mean", "like", "wait",
        # "er" (Danish "is") and "also" (German "so") were treated as fillers.
        self.assertFast("No, thank you.", "en")
        self.assertFast("What do you mean?", "en")
        self.assertFast("I like that idea.", "en")
        self.assertFast("Can you wait for me?", "en")
        self.assertFast("Nej tak.", "da")
        self.assertFast("Det er fint med mig.", "da")
        self.assertFast("Hvornår er du tilbage?", "da")
        self.assertFast("Das ist also kein Problem.", "de")
        self.assertFast("Er kommt morgen.", "de")
        self.assertFast("Oui, ça me va.", "fr")

    def test_leading_filler_is_removed_without_the_llm(self):
        self.assertFast("Um, yes that is right.", "en")
        self.assertFast("Øh, det tror jeg nok.", "da")
        self.assertEqual(st.strip_leading_fillers("Um, I think so."), "I think so.")
        self.assertEqual(st.strip_leading_fillers("uh um... yes."), "Yes.")
        self.assertEqual(st.strip_leading_fillers("Fine. Øh, jeg kommer."), "Fine. Jeg kommer.")
        self.assertEqual(st.strip_leading_fillers("Um."), "")
        # Words that merely start like a filler are kept.
        self.assertEqual(st.strip_leading_fillers("Umbrella first."), "Umbrella first.")
        # Hyphenated words and other punctuation after a filler.
        self.assertEqual(st.strip_leading_fillers("Uh-huh, I agree."), "Uh-huh, I agree.")
        self.assertEqual(st.strip_leading_fillers("Øh- ja"), "Øh- ja")
        self.assertEqual(st.strip_leading_fillers("Um - yes, I think so."), "Yes, I think so.")
        self.assertEqual(st.strip_leading_fillers("Um – ja."), "Ja.")
        self.assertEqual(st.strip_leading_fillers("Um; yes"), "Yes")
        # Words with their own casing keep it.
        self.assertEqual(st.strip_leading_fillers("Um, iPhone is good."), "iPhone is good.")
        self.assertEqual(st.strip_leading_fillers("Uhm... eBay is fine."), "eBay is fine.")
        # Fillers in the middle are left for the LLM, and nothing else is touched.
        self.assertEqual(st.strip_leading_fillers("We should, uh, go. e.g. later"), "We should, uh, go. e.g. later")

    def test_real_work_still_goes_to_the_llm(self):
        self.assertSlow("We should, uh, move it.", "en")                 # filler mid-sentence
        self.assertSlow("we should go to the the store", "en")           # stutter
        self.assertSlow("send email bullet point now", "en")             # command
        self.assertSlow("first line new line second line", "en")
        self.assertSlow("Hej Anna, ny linje tak for i går.", "da")
        self.assertSlow("Hi team, neue Zeile danke.", "en")              # commands in any language
        self.assertSlow("order 55 items", "en")                          # numbers
        self.assertSlow("one two three four five six seven eight nine ten", "en")
        self.assertSlow("It was, like, huge.", "en")                     # soft filler set off by commas
        self.assertSlow("Altså, det er fint.", "da")
        self.assertSlow("Also, ich weiß nicht.", "de")
        self.assertSlow("It is fine, you know.", "en")                   # filler phrase
        self.assertSlow("Det er, du ved, fint.", "da")
        self.assertSlow("So er we go.", "en")                            # English-only filler
        # Single-word commands the cleanup prompts tell Qwen to act on.
        self.assertSlow("I agree period", "en")
        self.assertSlow("That is great exclamation point", "en")
        self.assertSlow("next paragraph thanks", "en")
        self.assertSlow("delete that", "en")
        self.assertSlow("Hallo Anna Absatz danke", "de")
        self.assertSlow("Hej Anna, afsnit tak", "da")
        self.assertSlow("Bonjour à la ligne merci", "fr")

    def test_light_cleanup(self):
        self.assertEqual(st.light_cleanup("  yes   that works "), "Yes that works.")
        self.assertEqual(st.light_cleanup("Is it?"), "Is it?")
        self.assertEqual(st.light_cleanup(""), "")


class TestHallucinationFilter(unittest.TestCase):
    def test_complete_signoffs_are_always_dropped(self):
        for text in ("Thanks for watching!", "Thank you so much for watching. Bye!", "Please subscribe.",
                     "Tak fordi du så med.", "Vielen Dank fürs Zuschauen!", "Merci d'avoir regardé!", "",
                     "Thank you for watching this video.", "Thanks for watching and see you next time.",
                     "Thanks for watching, bye bye.", "Like and subscribe.",
                     "Vielen Dank fürs Zuschauen und bis zum nächsten Mal.", "Merci d'avoir regardé, à bientôt."):
            self.assertTrue(st.is_hallucination(text), text)

    def test_credit_lines_are_dropped_only_when_whisper_doubts_speech(self):
        for text in ("Undertekster af Nordisk Film", "Tekster af: Mette", "Untertitel im Auftrag des ZDF, 2021",
                     "Sous-titres réalisés par la communauté d'Amara.org"):
            self.assertTrue(st.is_hallucination(text, no_speech_prob=0.5), text)
            self.assertFalse(st.is_hallucination(text, no_speech_prob=0.05), text)

    def test_real_dictation_is_kept(self):
        # Dropping real speech is silent data loss: these all once vanished.
        for text in ("Vi mødes hos Nordisk Film i morgen.", "Thank you for watching the shop yesterday.",
                     "Thanks for watching the kids tonight.", "Please subscribe me to the newsletter.",
                     "Underteksterne skal være klar fredag.", "Tekst fra Anna er modtaget.",
                     "Untertitel bitte auf Deutsch.", "Sous-titres en français, merci.",
                     "Captions by the agency arrived.", "Transcribed by the intern.",
                     "Tak fordi du så med på mødet.", "Merci d'avoir regardé le dossier.",
                     "Abonnez-vous à la newsletter.",
                     "Please send the subtitles by Friday, we need them for the video.",
                     "Vi skal have undertekster på videoen inden fredag og sende den til Anna i morgen.",
                     "Can you watch the shop?"):
            self.assertFalse(st.is_hallucination(text), text)

    def test_short_signoff_dropped_only_when_whisper_doubts_speech(self):
        self.assertFalse(st.is_hallucination("Thank you.", no_speech_prob=0.05))
        self.assertTrue(st.is_hallucination("Thank you.", no_speech_prob=0.6))
        self.assertTrue(st.is_hallucination("Tak.", no_speech_prob=0.5))


def _seg(text, no_speech=0.01, logprob=-0.3, tokens=5):
    from types import SimpleNamespace
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob, tokens=[0] * tokens)


class TestTranscribe(unittest.TestCase):
    class FakeWhisper:
        """Detects `detected`; returns the segments listed per language."""

        def __init__(self, by_language, detected, speech_seconds=1.5):
            self.by_language, self.detected, self.speech_seconds = by_language, detected, speech_seconds
            self.calls = []

        def transcribe(self, audio, language=None, **kwargs):
            from types import SimpleNamespace
            self.calls.append((language, kwargs))
            lang = language or self.detected
            info = SimpleNamespace(language=lang, language_probability=0.8 if language is None else 1.0,
                                   duration_after_vad=self.speech_seconds,
                                   all_language_probs=[(self.detected, 0.8), ("da", 0.05)])
            return iter(self.by_language[lang]), info

    def _app(self, whisper, config=None):
        from types import SimpleNamespace
        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.config = {"last_language": "da"} if config is None else config
        app.language = None
        app.dictionary = SimpleNamespace(whisper_prompt=lambda: "Annika, SAP")
        app.whisper_model = whisper
        app.model_size = "base"
        return app

    def _run(self, app):
        return app.transcribe(np.zeros(16000, dtype=np.float32))

    def test_invented_segments_are_removed(self):
        whisper = self.FakeWhisper({"da": [_seg(" Hej Anna."), _seg(" Undertekster af Nordisk Film", no_speech=0.6)]}, "da")
        self.assertEqual(self._run(self._app(whisper)), ("Hej Anna.", "da", 0.8))
        self.assertEqual(whisper.calls[0][1]["initial_prompt"], "Annika, SAP")
        self.assertEqual(len(whisper.calls), 1)   # detection agreed with the last language: one pass

    def test_only_invented_text_means_no_speech(self):
        whisper = self.FakeWhisper({"en": [_seg(" Thank you.", no_speech=0.7)]}, "en")
        self.assertEqual(self._run(self._app(whisper, {}))[0], "")

    def test_short_phrase_keeps_last_language_when_whisper_is_more_sure_of_it(self):
        # "Det er fint med mig" detected as German.
        whisper = self.FakeWhisper({"de": [_seg(" Der fiegend mit mal.", logprob=-1.09)],
                                    "da": [_seg(" Det er fint med mig.", logprob=-0.41)]}, "de")
        self.assertEqual(self._run(self._app(whisper)), ("Det er fint med mig.", "da", 0.05))
        # Detection pass, then both candidates greedily (no temperature fallback).
        self.assertEqual([c[0] for c in whisper.calls], [None, "de", "da"])
        self.assertNotIn("temperature", whisper.calls[0][1])
        self.assertEqual([c[1].get("temperature") for c in whisper.calls[1:]], [0.0, 0.0])

    def test_translation_into_the_last_language_does_not_win(self):
        # Decoded in the wrong language Whisper translates fluently; a small
        # log-probability edge is not enough to switch.
        whisper = self.FakeWhisper({"de": [_seg(" Der fiegend mit mal.", logprob=-0.98)],
                                    "en": [_seg(" That's fine with me.", logprob=-0.80)]}, "de")
        self.assertEqual(self._run(self._app(whisper, {"last_language": "en"}))[:2], ("Der fiegend mit mal.", "de"))

    def test_real_language_switch_is_kept(self):
        whisper = self.FakeWhisper({"fr": [_seg(" Non, merci.", logprob=-0.21)],
                                    "da": [_seg(" Nå, mærsk.", logprob=-0.9)]}, "fr")
        self.assertEqual(self._run(self._app(whisper))[:2], ("Non, merci.", "fr"))

    def test_long_dictation_is_not_decoded_twice(self):
        whisper = self.FakeWhisper({"de": [_seg(" Lang.", logprob=-1.0)], "da": [_seg(" Lang.", logprob=-0.1)]},
                                   "de", speech_seconds=8.0)
        self.assertEqual(self._run(self._app(whisper))[1], "de")
        self.assertEqual(len(whisper.calls), 1)

    def test_check_can_be_turned_off_and_skips_locked_language(self):
        segs = {"de": [_seg(" A.", logprob=-1.0)], "da": [_seg(" B.", logprob=-0.1)]}
        whisper = self.FakeWhisper(segs, "de")
        self._run(self._app(whisper, {"last_language": "da", "language_check_short_phrases": False}))
        self.assertEqual(len(whisper.calls), 1)
        whisper = self.FakeWhisper(segs, "de")
        app = self._app(whisper)
        app.language = "de"
        self.assertEqual(self._run(app)[1], "de")
        self.assertEqual(len(whisper.calls), 1)

    def test_check_runs_on_small_models_only_unless_always(self):
        segs = {"de": [_seg(" Der fiegend mit mal.", logprob=-1.09)], "da": [_seg(" Det er fint med mig.", logprob=-0.41)]}
        missing = object()
        for model, setting, passes in (("tiny", True, 3), ("base", True, 3), ("small", True, 1),
                                       ("medium", True, 1), ("large-v3", True, 1), ("base", missing, 3),
                                       ("medium", missing, 1), ("medium", "always", 3), ("large-v3", " Always", 3),
                                       ("base", 1, 3), ("base", "yes", 3), ("medium", 1, 1),
                                       ("base", False, 1), ("base", None, 1), ("base", "off", 1)):
            with self.subTest(model=model, setting=setting):
                config = {"last_language": "da"}
                if setting is not missing:
                    config["language_check_short_phrases"] = setting
                whisper = self.FakeWhisper(segs, "de")
                app = self._app(whisper, config)
                app.model_size = model
                self._run(app)
                self.assertEqual(len(whisper.calls), passes)

    def test_always_still_skips_long_dictations(self):
        whisper = self.FakeWhisper({"de": [_seg(" Lang.", logprob=-1.0)], "da": [_seg(" Lang.", logprob=-0.1)]},
                                   "de", speech_seconds=5.0)
        app = self._app(whisper, {"last_language": "da", "language_check_short_phrases": "always"})
        app.model_size = "medium"
        self._run(app)
        self.assertEqual(len(whisper.calls), 1)


class TestCleanTextFastPath(unittest.TestCase):
    def _app(self):
        from types import SimpleNamespace
        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.config = {}
        app.ai_mode, app.local_llm = "local", object()
        app.dictionary = SimpleNamespace(prepare=lambda text, lang: (text, "", [], False),
                                         apply_casing=lambda text: text, note_usage=lambda text: None)
        app.llm_calls = []

        def fake_format(text, lang, dict_hint="", quiet=False, meeting=False):
            app.llm_calls.append(text)
            return "LLM: " + text
        app.format_text = fake_format
        return app

    def test_fast_path_and_stats(self):
        app = self._app()
        stats = {}
        self.assertEqual(app.clean_text("um, yes that works", "en", verbose=False, stats=stats), "Yes that works.")
        self.assertEqual(stats, {"llm": False})
        self.assertEqual(app.clean_text("Um.", "en", verbose=False), "")
        self.assertEqual(app.llm_calls, [])

    def test_llm_gets_the_text_without_filler_sounds(self):
        app = self._app()
        stats = {}
        self.assertEqual(app.clean_text("Um, it was, like, huge", "en", verbose=False, stats=stats),
                         "LLM: It was, like, huge")
        self.assertEqual(stats, {"llm": True})
        # Spoken commands are applied for dictation, never for a meeting turn.
        app.clean_text("Hi Anna, new line thanks for, uh, the meeting yesterday", "en", verbose=False)
        app.clean_text("Hi Anna, new line thanks for, uh, the meeting yesterday", "en", verbose=False, meeting=True)
        self.assertEqual(app.llm_calls[-2:], ["Hi Anna,\nThanks for the meeting yesterday",
                                              "Hi Anna, new line thanks for the meeting yesterday"])

    def test_only_filler_sounds_skip_the_llm(self):
        app = self._app()
        app.config = {"llm_fast_path": False}
        stats = {}
        self.assertEqual(app.clean_text("Um, uh.", "en", verbose=False, stats=stats), "")
        self.assertEqual(stats, {"llm": False})
        self.assertEqual(app.llm_calls, [])

    def test_fast_path_can_be_turned_off(self):
        app = self._app()
        app.config = {"llm_fast_path": False}
        app.clean_text("Nej tak.", "da", verbose=False)
        self.assertEqual(app.llm_calls, ["Nej tak."])

    def test_dictionary_words_keep_the_fast_path(self):
        # Dictionaries hold everyday words too ('claim', 'Service'): a short
        # clean phrase containing one must not wait for Qwen.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        app = self._app()
        app.dictionary = sd.Dictionary(tmp)
        for word in ("claim", "Jira", "SAP", "Mette"):
            app.dictionary.add(word)
        self.assertEqual(app.clean_text("open the claim in Jira and sap", "en", verbose=False),
                         "Open the claim in Jira and SAP.")
        self.assertEqual(app.llm_calls, [])
        # A name in the wrong case is the LLM's job: apply_casing leaves it.
        app.clean_text("ring til mette", "da", verbose=False)
        self.assertEqual(app.llm_calls, ["Ring til mette"])


class TestClipboardRestore(unittest.TestCase):
    """The restore after a paste runs on a timer and hands over cleanly when
    Speak needs the clipboard again before it fires."""

    class FakeUser32:
        def __init__(self):
            self.seq = 5

        def GetClipboardSequenceNumber(self):
            return self.seq

    def setUp(self):
        import threading
        self.real_user32 = speak._user32
        speak._user32 = self.fake = self.FakeUser32()
        self.app = speak.SpeakApp.__new__(speak.SpeakApp)
        self.app.clipboard_lock = threading.Lock()
        self.app._pending_restore = self.app._pending_restore_timer = None
        self.app.CLIPBOARD_RESTORE_DELAY = 0.05
        self.restored = []
        self.app._restore_clipboard = lambda token, delay=None: self.restored.append(token[0])
        self.snapshot = {"kind": "text", "text": "the user's clipboard"}

    def tearDown(self):
        speak._user32 = self.real_user32

    def _wait_for_timer(self, until=lambda: False, timeout=2.0):
        """Waits for a timer to have had ample time to fire (or for until())."""
        import time
        deadline = time.time() + timeout
        while time.time() < deadline and not until():
            time.sleep(0.02)
        time.sleep(0.1)

    def test_restore_runs_later_without_blocking(self):
        import time
        self.app.CLIPBOARD_RESTORE_DELAY = 0.5
        t0 = time.time()
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, time.time()))
        self.assertLess(time.time() - t0, 0.3)   # far below the 0.5 s restore delay
        self.assertEqual(self.restored, [])
        self._wait_for_timer(until=lambda: self.restored)
        self.assertEqual(self.restored, [self.snapshot])
        self.assertIsNone(self.app._pending_restore)

    def test_next_paste_inherits_the_users_clipboard(self):
        import time
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, time.time()))
            # A second dictation pastes before the restore: it must restore
            # the user's text later, not the first dictation.
            self.assertEqual(self.app._take_pending_restore()[:2], (self.snapshot, 5))
        self._wait_for_timer()
        self.assertEqual(self.restored, [])   # the cancelled timer did nothing

    def test_next_paste_waits_until_the_previous_one_had_its_time(self):
        # The target app reads the clipboard when it handles Ctrl+V. Speak
        # must not replace it sooner than CLIPBOARD_RESTORE_DELAY after a paste.
        import time
        self.app.CLIPBOARD_RESTORE_DELAY = 0.3
        pasted_at = time.time()
        # (The inner 0.2 s bound below is generous: the wait under test is 0.3 s.)
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, pasted_at))
            self.app._take_pending_restore()
            self.assertGreaterEqual(time.time() - pasted_at, 0.29)
        # Once the delay has passed there is no wait at all.
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, time.time() - 1.0))
            t0 = time.time()
            self.app._take_pending_restore()
            self.assertLess(time.time() - t0, 0.2)   # no 0.3 s wait

    def test_user_copy_in_between_wins(self):
        import time
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, time.time()))
        self.fake.seq = 6   # the user copied something new
        with self.app.clipboard_lock:
            self.assertIsNone(self.app._take_pending_restore())

    def test_flush_on_exit(self):
        import time
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, time.time()))
        self.app.flush_clipboard_restore()
        self.assertEqual(self.restored, [self.snapshot])
        self._wait_for_timer()
        self.assertEqual(self.restored, [self.snapshot])   # not restored twice

    def _prepare_win32_fakes(self):
        """Stubs out keystrokes so insert_text / _learn_from_selection run headless."""
        import types
        self.real_send_keys = speak.send_keys
        speak.send_keys = lambda steps: True
        self.addCleanup(setattr, speak, "send_keys", self.real_send_keys)
        self.app._release_modifiers = lambda timeout: True
        self.app.play_sound = lambda kind: None
        self.app.paste_mode = "auto"
        self.app.get_foreground_info = lambda: ("notepad", "untitled", "notepad.exe")
        self.app._clipboard_snapshot = lambda: {"kind": "text", "text": "fresh snapshot"}
        self.app._open_clipboard = lambda attempts=5: False
        self.app.dictionary = types.SimpleNamespace(learn_from_correction=lambda a, b: [])
        self.typed = []
        self.app.type_unicode_text = self.typed.append

    def test_learn_with_nothing_selected_still_gives_the_clipboard_back(self):
        import time
        self._prepare_win32_fakes()
        self.app.last_inserted_text = "dictation one"
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, time.time() - 1.0))
            self.app._learn_from_selection()   # nothing selected: the copy never happens
        self.assertEqual(self.restored, [self.snapshot])

    def test_failed_clipboard_write_keeps_the_pending_restore(self):
        import time
        self._prepare_win32_fakes()
        self.app._set_clipboard_text = lambda text: False
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, time.time() - 1.0))
            token = self.app.insert_text("dictation two")
        # Pasting would insert stale clipboard content: the text is typed instead.
        self.assertEqual(self.typed, ["dictation two"])
        self.assertIsNotNone(token)
        self.assertEqual(token[0], self.snapshot)

    def test_failed_clipboard_write_keeps_the_original_sequence_number(self):
        # The app holding the clipboard may have just put the user's new copy
        # on it. The restore must see that change (seq 5 -> 6) and skip,
        # not adopt seq 6 and overwrite the copy.
        import time
        self._prepare_win32_fakes()

        def busy_clipboard(text):
            self.fake.seq = 6
            return False
        self.app._set_clipboard_text = busy_clipboard
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore((self.snapshot, 5, time.time() - 1.0))
            token = self.app.insert_text("dictation two")
        self.assertEqual(token[:2], (self.snapshot, 5))

    def test_no_token_schedules_nothing(self):
        with self.app.clipboard_lock:
            self.app._schedule_clipboard_restore(None)
        self.assertIsNone(self.app._pending_restore)


class TestLongDictation(unittest.TestCase):
    def _app(self):
        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.LLM_CHUNK_WORDS = 10
        return app

    def test_sentences_are_kept_whole(self):
        text = "One two three four five six. Seven eight nine ten eleven twelve."
        self.assertEqual(self._app()._split_for_llm(text),
                         ["One two three four five six.", "Seven eight nine ten eleven twelve."])

    def test_text_without_full_stops_is_still_split(self):
        app = self._app()
        commas = "one two three four five six, seven eight nine ten eleven twelve, thirteen fourteen"
        bare = " ".join(f"w{i}" for i in range(25))
        for text in (commas, bare):
            chunks = app._split_for_llm(text)
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(len(c.split()) <= 10 for c in chunks), chunks)
            self.assertEqual(" ".join(chunks).split(), text.split())   # nothing lost or reordered

    def test_no_tiny_last_chunk(self):
        # 21 bare words with a limit of 10: [7, 7, 7], not [10, 10, 1].
        chunks = self._app()._split_for_llm(" ".join(f"w{i}" for i in range(21)))
        self.assertEqual([len(c.split()) for c in chunks], [7, 7, 7])


class TestAutoStop(unittest.TestCase):
    def test_one_stop_thread_at_the_time_limit(self):
        import threading
        import time
        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.sample_rate, app.MAX_RECORDING_SECONDS = 16000, 1
        app.is_recording, app._auto_stop_requested = True, False
        app.audio_frames = [np.zeros((1600, 1), dtype=np.float32)] * 10   # exactly 1 s
        stops = []
        started, release = threading.Event(), threading.Event()

        def slow_stop():
            stops.append(threading.current_thread())
            started.set()
            release.wait(5)   # the stream keeps delivering blocks meanwhile
        app.stop_recording = slow_stop
        block = np.zeros((1600, 1), dtype=np.float32)
        app.audio_callback(block, 1600, None, None)
        self.assertTrue(started.wait(5))
        for _ in range(4):
            app.audio_callback(block, 1600, None, None)
        release.set()
        stops[0].join(5)
        self.assertEqual(len(stops), 1)


class TestKeyInput(unittest.TestCase):
    def test_chord_presses_then_releases_in_reverse(self):
        self.assertEqual(speak.chord(speak.VK_CONTROL, speak.VK_V),
                         [(speak.VK_CONTROL, False), (speak.VK_V, False), (speak.VK_V, True), (speak.VK_CONTROL, True)])

    def test_key_inputs_carry_flags_and_scan_codes(self):
        events = speak.build_key_inputs(speak.chord(speak.VK_SHIFT, speak.VK_INSERT))
        self.assertEqual(len(events), 4)
        self.assertTrue(all(e.type == speak.INPUT_KEYBOARD for e in events))
        flags = [e.u.ki.dwFlags for e in events]
        self.assertEqual(flags[0], 0)
        self.assertEqual(flags[1], speak.KEYEVENTF_EXTENDEDKEY)                       # Insert is extended
        self.assertEqual(flags[2], speak.KEYEVENTF_EXTENDEDKEY | speak.KEYEVENTF_KEYUP)
        self.assertEqual(flags[3], speak.KEYEVENTF_KEYUP)
        self.assertTrue(all(e.u.ki.wScan for e in events))                            # scan codes filled in


class TestSpokenCommandsAndFillers(unittest.TestCase):
    def test_line_break_and_list_commands_in_four_languages(self):
        cases = {
            "Hi Anna, new line thanks for the meeting yesterday.": "Hi Anna,\nThanks for the meeting yesterday.",
            "Hej Anna, ny linje, tak for mødet i går.": "Hej Anna,\nTak for mødet i går.",
            "Hallo Thomas, neue Zeile, danke für das Treffen gestern.": "Hallo Thomas,\nDanke für das Treffen gestern.",
            "Bonjour Marc, à la ligne, merci pour la réunion d'hier.": "Bonjour Marc,\nMerci pour la réunion d'hier.",
            "Here are the next steps. Bullet point update the price list. Bullet point send the newsletter.":
                "Here are the next steps.\n- Update the price list.\n- Send the newsletter.",
            "That was all. New paragraph, next topic.": "That was all.\n\nNext topic.",
        }
        for said, expected in cases.items():
            self.assertEqual(st.prepare_for_llm(said, "en"), expected, said)

    def test_command_words_that_are_ordinary_words_stay(self):
        for text in ("Our new line of candles arrives Monday.", "We launch the new line, which sells well.",
                     "Vores nye linje af lys kommer mandag.", "Delete that file before Monday.",
                     "Add a bullet point here.", "Please scratch that from the list.", "Please scratch that."):
            self.assertEqual(st.prepare_for_llm(text, "en"), text, text)

    def test_scratch_that_deletes_the_sentence_before(self):
        self.assertEqual(st.prepare_for_llm("The budget is ten thousand. Scratch that. The budget is twelve thousand.", "en"),
                         "The budget is twelve thousand.")
        self.assertEqual(st.prepare_for_llm("Slet det. Vi ses i morgen.", "da"), "Vi ses i morgen.")
        self.assertEqual(st.prepare_for_llm("Hi Anna,\nwrong line. Scratch that.\nSee you.", "en"), "Hi Anna,\nSee you.")

    def test_meeting_turns_keep_command_words(self):
        said = "We said new line, uh, for the shelves. Scratch that, it was the other one."
        self.assertEqual(st.prepare_for_llm(said, "en", commands=False),
                         "We said new line for the shelves. Scratch that, it was the other one.")

    def test_filler_sounds_are_removed_with_their_commas(self):
        self.assertEqual(st.prepare_for_llm("Can you, uh, check the invoice and, um, call the supplier?", "en"),
                         "Can you check the invoice and call the supplier?")
        self.assertEqual(st.prepare_for_llm("Euh, est-ce que tu peux, euh, venir ?", "fr"), "Est-ce que tu peux venir ?")
        self.assertEqual(st.prepare_for_llm("Uh-huh, I agree. Um.", "en"), "Uh-huh, I agree.")
        self.assertEqual(st.prepare_for_llm("Um, uh.", "en"), "")

    def test_um_is_a_german_word(self):
        self.assertEqual(st.prepare_for_llm("Um zehn Uhr treffen wir uns, um das zu klären.", "de"),
                         "Um zehn Uhr treffen wir uns, um das zu klären.")
        self.assertEqual(st.strip_leading_fillers("Um zehn Uhr passt.", "de"), "Um zehn Uhr passt.")
        self.assertEqual(st.strip_leading_fillers("Um, ten is fine.", "en"), "Ten is fine.")
        self.assertFalse(st.has_disfluency("Wir treffen uns um zehn.", "de"))
        self.assertEqual(st.strip_leading_fillers("E, ich glaube schon.", "de"), "Ich glaube schon.")
        self.assertEqual(st.strip_leading_fillers("Um, ten is fine."), "Ten is fine.")   # no language: as before

    def test_light_cleanup_keeps_line_breaks(self):
        self.assertEqual(st.light_cleanup("hi  Anna,\n\n\n thanks"), "Hi Anna,\n\nthanks.")
        self.assertEqual(st.light_cleanup("  done  "), "Done.")


class TestCleanupIsFaithful(unittest.TestCase):
    def test_accepts_cleanup(self):
        ok = [
            ("Um, I think we should, uh, move the meeting.", "I think we should move the meeting.", "en"),
            ("We need to, we need to finalize it.", "We need to finalize it.", "en"),
            ("It was, like, huge, you know.", "It was huge.", "en"),
            ("Let's meet on Tuesday, no wait, Wednesday at ten.", "Let's meet on Wednesday at ten.", "en"),
            ("Wir müssen das fertig stellen.", "Wir müssen das fertigstellen.", "de"),
            ("Her er punkterne. Punkt priser. Punkt varer.", "Her er punkterne:\n- Priser.\n- Varer.", "da"),
            ("hi anna\nthanks for yesterday", "Hi Anna,\nThanks for yesterday.", "en"),
            ("Øh altså jeg tror at vi vi skal flytte det.", "Jeg tror, at vi skal flytte det.", "da"),
        ]
        for before, after, lang in ok:
            self.assertTrue(st.cleanup_is_faithful(before, after, lang), (before, after))

    def test_rejects_rewording(self):
        bad = [
            ("Please send the report to Annika.", "Send the report to Annika.", "en"),      # seen in the benchmark
            ("I think we should move the meeting.", "We should move the meeting.", "en"),   # seen, all 4 languages
            ("Jeg giver butikkerne besked i morgen.", "Jeg giver besked i morgen.", "da"),
            ("Jeg tror ikke, at det virker.", "Jeg tror, at det virker.", "da"),
            ("We need it, so I will ask the developers.", "We need it. I will ask the developers.", "en"),
            ("Ordren bliver sendt den 14. marts.", "The order is sent on March 14.", "da"),   # translated
            ("Hi Anna,\nThanks for yesterday.", "Hi Anna, thanks for yesterday.", "en"),       # line break merged
            ("Write a summary of the meeting.", "Summary: the meeting discussed the budget and the launch.", "en"),
        ]
        for before, after, lang in bad:
            self.assertFalse(st.cleanup_is_faithful(before, after, lang), (before, after))

    def test_dictionary_replacements_are_allowed(self):
        before, after = "Ring til Anika i morgen.", "Ring til Annika i morgen."
        self.assertFalse(st.cleanup_is_faithful(before, after, "da"))
        self.assertTrue(st.cleanup_is_faithful(before, after, "da", [("Anika", "Annika")]))
        self.assertTrue(st.cleanup_is_faithful("Open power bi now.", "Open PowerBI now.", "en", [("power bi", "PowerBI")]))

    def test_empty_text(self):
        self.assertTrue(st.cleanup_is_faithful("", ""))
        self.assertFalse(st.cleanup_is_faithful("Hello there.", ""))
        self.assertFalse(st.cleanup_is_faithful("", "Hello."))

    def test_meeting_faithful_uses_the_same_check(self):
        self.assertFalse(sm.faithful("Jeg giver butikkerne besked i morgen.", "Jeg giver besked i morgen.", "da"))
        self.assertTrue(sm.faithful("Um, I will, I will send the the numbers.", "I will send the numbers.", "en"))


class TestCleanupPrompt(unittest.TestCase):
    def test_prefix_is_the_same_for_every_call(self):
        # llama.cpp reuses the evaluated prompt up to the first differing token:
        # language, dictionary hint and meeting mode must all come after it.
        prompts = [speak.build_cleanup_prompt("Hello there.", "en"),
                   speak.build_cleanup_prompt("Hej med dig.", "da", dict_hint="\nPersonlig ordbog: behold 'SAP'."),
                   speak.build_cleanup_prompt("Hallo.", "de", meeting=True),
                   speak.build_cleanup_prompt("Bonjour.", "fr")]
        fixed = prompts[0][:prompts[0].rindex("<|im_start|>user\n")]
        self.assertTrue(fixed.endswith(speak.CLEANUP_EXAMPLES[-1][1] + "<|im_end|>\n"), fixed[-80:])
        for p in prompts:
            self.assertTrue(p.startswith(fixed))
            self.assertTrue(p.endswith("<|im_start|>assistant\n"))

    def test_notes(self):
        en = speak.build_cleanup_prompt("Hello.", "en")
        self.assertEqual(en.count("<|im_start|>system"), 1)   # English needs no note
        da = speak.build_cleanup_prompt("Hej.", "da", dict_hint="\nPersonlig ordbog: behold 'SAP'.", meeting=True)
        note = da.rsplit("<|im_start|>system\n", 1)[1].split("<|im_end|>")[0]
        self.assertEqual(note, f"{speak.MEETING_TURN_NOTE} Personlig ordbog: behold 'SAP'. Write in Danish.")
        self.assertTrue(da.endswith("<|im_start|>user\nHej.<|im_end|>\n<|im_start|>assistant\n"))

    def test_examples_teach_only_what_the_guard_accepts(self):
        for said, cleaned in speak.CLEANUP_EXAMPLES:
            self.assertTrue(st.cleanup_is_faithful(said, cleaned), (said, cleaned))
            self.assertNotIn("<|im_", said + cleaned)


class TestLlmGuard(unittest.TestCase):
    class FakeLlm:
        def __init__(self, answer):
            self.answer, self.prompts = answer, []

        def tokenize(self, data, add_bos=False):
            return data.split()

        def __call__(self, prompt, **kwargs):
            self.prompts.append(prompt)
            answer = self.answer(prompt) if callable(self.answer) else self.answer
            return {"choices": [{"text": answer}], "usage": {"completion_tokens": 3}}

    def _app(self, answer):
        from types import SimpleNamespace
        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.ai_mode, app.llm_ctx = "local", 2048
        app.local_llm = self.FakeLlm(answer)
        app.dictionary = SimpleNamespace(correction_pairs=lambda: [("Anika", "Annika")])
        return app

    def test_rewording_falls_back_to_the_transcript(self):
        app = self._app("Send the report to Annika.")
        self.assertEqual(app.format_text("please send the report to Annika", "en", quiet=True),
                         "Please send the report to Annika.")
        app = self._app("We should send it.")   # dropped words AND a merged line break
        self.assertEqual(app.format_text("Hi Anna,\nwe should send it", "en", quiet=True), "Hi Anna,\nwe should send it.")

    def test_faithful_output_and_dictionary_fix_are_kept(self):
        app = self._app("Ring til Annika i morgen.")
        self.assertEqual(app.format_text("ring til Anika i morgen", "da", quiet=True), "Ring til Annika i morgen.")
        self.assertIn("Write in Danish.", app.local_llm.prompts[0])

    def test_long_dictation_keeps_its_line_breaks(self):
        app = self._app(lambda prompt: prompt.rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>")[0])
        app.LLM_CHUNK_WORDS = 6
        text = "Hi Anna,\nThe photos are ready for the shop. The codes still fail.\n- Fix the codes.\n- Ship it."
        self.assertEqual(app.format_text(text, "en", quiet=True), text)
        chunks = app._split_for_llm_lines(text)
        self.assertGreater(len(app.local_llm.prompts), 1)
        self.assertTrue(all(len(c.split()) <= 6 for c, _ in chunks), chunks)
        self.assertEqual("".join(sep + chunk for chunk, sep in chunks), text)
        # A dictated paragraph break survives chunking.
        para = "Hi Anna, the photos are ready.\n\nThe codes still fail in the shop."
        self.assertEqual("".join(sep + chunk for chunk, sep in app._split_for_llm_lines(para)), para)


class TestCleanupReviewCases(unittest.TestCase):
    """Inputs from the adversarial review of the Phase 2 cleanup."""

    def test_scratch_that(self):
        P = st.prepare_for_llm
        self.assertEqual(P("Hello Anna. The budget is ten. Scratch that. The budget is twelve.", "en"),
                         "Hello Anna. The budget is twelve.")   # was glued: "Anna.The"
        # A full stop after a number or an abbreviation does not end a sentence.
        self.assertEqual(P("Vi mødes den 14. marts. Slet det. Vi mødes den 15. marts.", "da"), "Vi mødes den 15. marts.")
        self.assertEqual(P("See e.g. the attachment. Scratch that. See the appendix.", "en"), "See the appendix.")
        self.assertEqual(P("Send Nr. 14. Scratch that. Send Nr. 15.", "en"), "Send Nr. 15.")
        self.assertEqual(P("One. Two. Three. Scratch that. Scratch that. Four.", "en"), "One. Four.")
        self.assertEqual(P("Wrong. Scratch that. Scratch that. Right.", "en"), "Right.")
        for text in ("Delete that, please.", "Slet det, tak.", "Please scratch that."):
            self.assertEqual(P(text, "en"), text)

    def test_german_e_is_a_filler_only_as_a_leading_e_comma(self):
        self.assertEqual(st.prepare_for_llm("Vitamin E ist gut.", "de"), "Vitamin E ist gut.")
        self.assertEqual(st.prepare_for_llm("Der Buchstabe e kommt oft vor.", "de"), "Der Buchstabe e kommt oft vor.")
        self.assertEqual(st.prepare_for_llm("E, ich glaube schon.", "de"), "Ich glaube schon.")
        self.assertFalse(st.has_disfluency("Wir nehmen Vitamin E.", "de"))
        self.assertFalse(st.cleanup_is_faithful("Wir nehmen Vitamin E.", "Wir nehmen Vitamin.", "de"))

    def test_a_polite_cue_word_allows_no_deletions(self):
        F = st.cleanup_is_faithful
        self.assertFalse(F("Sorry for the delay. Please send the report to Annika and Mikkel in Aarhus.",
                           "Sorry for the delay. Send the report to Annika in Aarhus.", "en"))
        self.assertFalse(F("I'm sorry to hear that. Please let me know if I can help with anything.",
                           "I'm sorry to hear that. Let me know if I can help.", "en"))
        self.assertFalse(F("Sorry, call Anna now.", "Sorry.", "en"))
        self.assertFalse(F("Pardon, appelle Marc maintenant.", "Pardon.", "fr"))
        self.assertFalse(F("Jeg mener det, send ikke rapporten til Lars.", "Jeg mener det, send rapporten.", "da"))
        self.assertTrue(F("Can we order fifty, sorry, sixty boxes?", "Can we order sixty boxes?", "en"))

    def test_commands_the_llm_did_are_accepted(self):
        F = st.cleanup_is_faithful
        self.assertTrue(F("Hi Anna new line thanks.", "Hi Anna,\nThanks.", "en"))   # no pause: left to Qwen
        self.assertTrue(F("Here are the next steps. Bullet point one. Bullet point two.",
                          "Here are the next steps:\n- One.\n- Two.", "en"))
        self.assertFalse(F("Hi Anna new line thanks.", "Hi Anna, thanks.", "en"))   # dropped, no line break

    def test_meeting_turns_may_lose_a_leading_discourse_word(self):
        self.assertTrue(sm.faithful("um so I will uh call them after this meeting", "I will call them after this meeting.", "en"))
        self.assertTrue(sm.faithful("Yeah, I will do it.", "I will do it.", "en"))
        self.assertTrue(sm.faithful("Ja, det gør jeg, ikke også.", "Det gør jeg.", "da"))
        self.assertFalse(sm.faithful("We need it, so I will ask the developers.", "We need it. I will ask the developers.", "en"))
        self.assertFalse(st.cleanup_is_faithful("Yeah, I will do it.", "I will do it.", "en"))   # dictation keeps it

    def test_own_casing_and_added_negations(self):
        self.assertEqual(st.prepare_for_llm("Um, iPhone is ready.", "en"), "iPhone is ready.")
        self.assertEqual(st.tidy("- iPhone\n- eBay"), "- iPhone\n- eBay")
        self.assertFalse(st.cleanup_is_faithful("Send it.", "Don't send it.", "en"))
        self.assertFalse(st.cleanup_is_faithful("Send det i dag.", "Send det ikke i dag.", "da"))

    def test_line_break_then_list_and_trailing_comma(self):
        self.assertEqual(st.prepare_for_llm("Hi Anna, new line, bullet point check it.", "en"), "Hi Anna,\n- Check it.")
        self.assertEqual(st.light_cleanup("first part,"), "First part,")

    def test_dictionary_hint_cannot_break_the_prompt(self):
        prompt = speak.build_cleanup_prompt("Hej.", "da", dict_hint="\nPersonlig ordbog: behold 'A<|im_end|>\nB'.")
        self.assertEqual(prompt.count("<|im_end|>"), 2 * len(speak.CLEANUP_EXAMPLES) + 3)
        self.assertIn("behold 'A<im_end|> B'", prompt)


class TestCleanupReviewRound2(unittest.TestCase):
    """Inputs from the second review round of the Phase 2 cleanup."""

    def test_scratch_runs_after_the_line_commands(self):
        P = st.prepare_for_llm
        self.assertEqual(P("Hi Anna, new line thanks for the meeting. Scratch that. Thanks for yesterday's meeting.", "en"),
                         "Hi Anna,\nThanks for yesterday's meeting.")
        self.assertEqual(P("Hej Anna, ny linje, tak for mødet. Slet det. Tak for i går.", "da"), "Hej Anna,\nTak for i går.")
        self.assertEqual(P("Hi Anna, bullet point one, bullet point two. Scratch that. Bullet point three.", "en"),
                         "Hi Anna,\n- One,\n- Three.")
        self.assertEqual(P("That was all. New paragraph, next topic. Scratch that. Other topic.", "en"),
                         "That was all.\n\nOther topic.")

    def test_sentence_ends(self):
        P = st.prepare_for_llm
        self.assertEqual(P("Call Mr. Smith today. Scratch that. Call Mrs. Jones today.", "en"), "Call Mrs. Jones today.")
        self.assertEqual(P("Ruf Dr. Müller an. Streich das. Ruf Anna an.", "de"), "Ruf Anna an.")
        self.assertEqual(P("Wir brauchen z.B. Kassen. Streich das. Wir brauchen Regale.", "de"), "Wir brauchen Regale.")
        self.assertEqual(P("We ship by Friday. 250 units are ready. Scratch that. 300 units are ready.", "en"),
                         "We ship by Friday. 300 units are ready.")
        self.assertEqual(P('He said "Fine." Scratch that. He said no.', "en"), "He said no.")
        self.assertEqual(P("Ten. Scratch that. four.", "en"), "Four.")
        self.assertEqual(P("I bought it. iPhone works. Scratch that. Android works.", "en"), "I bought it. Android works.")
        for text in ("Vi mødes den 14. marts kl. 10.", "Ring til Dr. Hansen. Hun venter."):
            self.assertEqual("".join(sep + s for sep, s in st.sentences(text)), text)

    def test_punkt_and_puce_sentences_are_list_items(self):
        P = st.prepare_for_llm
        self.assertEqual(P("Her er punkterne. Punkt opdater prislisten. Punkt send nyhedsbrevet.", "da"),
                         "Her er punkterne.\n- Opdater prislisten.\n- Send nyhedsbrevet.")
        self.assertEqual(P("Voici la liste. Puce envoyer le devis.", "fr"), "Voici la liste.\n- Envoyer le devis.")
        text = "Punkt 3 er vigtigt. Punkt to handler om priser."
        self.assertEqual(P(text, "da"), text)

    def test_a_correction_takes_back_only_the_words_before_its_cue(self):
        F = st.cleanup_is_faithful
        good = [("Let's meet on Tuesday, no wait, Wednesday at ten.", "Let's meet on Wednesday at ten.", "en"),
                ("Send it to Peter, no wait, to Paul before lunch.", "Send it to Paul before lunch.", "en"),
                ("I think we should ask Sarah, sorry, Peter first.", "I think we should ask Peter first.", "en"),
                ("Wir treffen uns am Dienstag, nein warte, am Mittwoch um zehn.", "Wir treffen uns am Mittwoch um zehn.", "de"),
                ("Vi ses på tirsdag, nej vent, onsdag klokken ti.", "Vi ses på onsdag klokken ti.", "da"),
                ("On se voit mardi, non attends, mercredi à dix heures.", "On se voit mercredi à dix heures.", "fr"),
                ("Check the budget, scratch that, check the forecast.", "Check the forecast.", "en")]
        for before, after, lang in good:
            self.assertTrue(F(before, after, lang), before)
        bad = [("Send it to Peter, no wait, to Paul before lunch.", "Send it to Paul.", "en"),   # after the cue
               ("I think we should ask Sarah, sorry, Peter first.", "Peter first.", "en"),       # the introduction
               ("Please send the report to Annika and Mikkel in Aarhus, sorry, in Odense.", "Send the report in Odense.", "en"),
               ("I'm sorry to hear that. Please let me know if I can help.", "I'm to hear that. Let me know if I can help.", "en")]
        for before, after, lang in bad:
            self.assertFalse(F(before, after, lang), before)

    def test_command_credit_is_per_phrase(self):
        F = st.cleanup_is_faithful
        self.assertFalse(F("The next point is important. We agree.", "The point is important.\nWe agree.", "en"))
        self.assertFalse(F("Hi Anna new line thanks for the new offer. Bullet point first. Bullet point second.",
                           "Hi Anna,\nThanks for the offer.\n- First.\n- Second.", "en"))
        self.assertTrue(F("Hi Anna new line thanks for the new offer.", "Hi Anna,\nThanks for the new offer.", "en"))
        self.assertFalse(F("Ich komme.", "Ich komme kaum.", "de"))

    def test_meeting_turn_is_compared_with_what_qwen_got(self):
        prepared = st.prepare_for_llm("E, ich, äh, glaube schon.", "de", commands=False)
        self.assertTrue(sm.faithful(prepared, "Ich glaube schon.", "de"))


class TestCleanupReviewRound3(unittest.TestCase):
    """Inputs from the third review round of the Phase 2 cleanup."""

    def test_corrections_across_a_full_stop_and_long_take_backs(self):
        F = st.cleanup_is_faithful
        good = [("I will come on Monday. No wait, Tuesday.", "I will come on Tuesday.", "en"),
                ("I will come on Monday. No, wait. Tuesday.", "I will come on Tuesday.", "en"),
                ("Call Anna. Sorry, call Lars.", "Call Lars.", "en"),
                ("Vi ses mandag. Nej vent, tirsdag.", "Vi ses tirsdag.", "da"),
                ("Ruf Anna an. Nein warte, ruf Lars an.", "Ruf Lars an.", "de"),
                ("On se voit lundi. Non attends, mardi.", "On se voit mardi.", "fr"),
                ("Send it to Peter, Paul and Anna, sorry, Lars.", "Send it to Lars.", "en"),
                ("Let's meet at ten o'clock in the morning, sorry, at noon.", "Let's meet at noon.", "en"),
                ("We need fifty, no wait, no, sixty boxes.", "We need sixty boxes.", "en")]
        for before, after, lang in good:
            self.assertTrue(F(before, after, lang), before)
        bad = [("Call Anna. Sorry, call Lars.", "Lars.", "en"),
               ("No, sorry, I cannot come on Friday.", "I cannot come.", "en"),
               ("Send it to Peter, Paul and Anna, sorry, Lars.", "Lars.", "en")]
        for before, after, lang in bad:
            self.assertFalse(F(before, after, lang), before)

    def test_list_words_only_in_danish_and_french(self):
        P = st.prepare_for_llm
        for text, lang in (("Punkt acht treffen wir uns vor dem Büro.", "de"),
                           ("Punkt zwei der Tagesordnung ist das Budget.", "de"),
                           ("Punkt for punkt gennemgik vi listen.", "da"),
                           ("Punkt seks handler om levering.", "da"),
                           ("Puce électronique défectueuse.", "fr"),
                           ("Punkt opdater prislisten.", "en")):
            self.assertEqual(P(text, lang), text, text)

    def test_numbers_dates_and_separators_around_scratch(self):
        P = st.prepare_for_llm
        self.assertEqual(P("I need 14. The rest later. Scratch that. The rest tomorrow.", "en"), "I need 14. The rest tomorrow.")
        self.assertEqual(P("Vi ses kl. 10. Hun venter. Slet det. Han venter.", "da"), "Vi ses kl. 10. Han venter.")
        self.assertEqual(P("Wir sehen uns am 14. März. Streich das. Am 15. März.", "de"), "Am 15. März.")
        self.assertEqual(P("First. New paragraph. Second. Scratch that. New line. Third.", "en"), "First.\nThird.")
        self.assertEqual(P("First. New paragraph. Second. Scratch that. Third.", "en"), "First.\n\nThird.")

    def test_a_noun_punkt_does_not_use_up_the_new_line_credit(self):
        self.assertTrue(st.cleanup_is_faithful("Et vigtigt punkt er prisen new line tak.",
                                               "Et vigtigt punkt er prisen.\nTak.", "da"))


class TestCleanupReviewRound4(unittest.TestCase):
    """Inputs from the fourth review round of the Phase 2 cleanup."""

    def test_a_correction_never_covers_a_dropped_negation_or_please(self):
        F = st.cleanup_is_faithful
        bad = [("I will not come on Monday, sorry, on Tuesday at ten in the morning.",
                "I will come on Tuesday at ten in the morning.", "en"),
               ("Jeg kommer ikke mandag, nej vent, tirsdag klokken ti om morgenen.",
                "Jeg kommer tirsdag klokken ti om morgenen.", "da"),
               ("Ich komme nicht am Montag, nein warte, am Dienstag um zehn Uhr morgens.",
                "Ich komme am Dienstag um zehn Uhr morgens.", "de"),
               ("Je ne viens pas lundi, non attends, mardi à dix heures du matin.",
                "Je viens mardi à dix heures du matin.", "fr"),
               ("Please send the report to Peter, sorry, to Paul and Anna before the meeting on Friday.",
                "Send the report to Paul and Anna before the meeting on Friday.", "en"),
               ("Call Anna, not Lars, sorry, Mette.", "Call Anna, Mette.", "en")]
        for before, after, lang in bad:
            self.assertFalse(F(before, after, lang), before)
        self.assertTrue(F("I will not come on Monday, sorry, on Tuesday at ten in the morning.",
                          "I will not come on Tuesday at ten in the morning.", "en"))
        self.assertTrue(F("Ruf Anna an. Nein warte, ruf Lars an.", "Ruf Lars an.", "de"))

    def test_german_ordinals_do_not_end_a_sentence(self):
        P = st.prepare_for_llm
        self.assertEqual(P("Wir sind im 2. Quartal fertig. Streich das. Wir sind im 3. Quartal fertig.", "de"),
                         "Wir sind im 3. Quartal fertig.")
        self.assertEqual(P("Das ist der 3. Versuch. Streich das. Das ist der 4. Versuch.", "de"), "Das ist der 4. Versuch.")
        self.assertEqual(P("Treffen am 14. März. Streich das. Am 15.", "de"), "Am 15.")
        self.assertEqual(P("Vi ses kl. 10. Hun venter. Slet det. Han venter.", "da"), "Vi ses kl. 10. Han venter.")

    def test_repeated_line_commands(self):
        self.assertEqual(st.prepare_for_llm("Hi, new line, new line, bye.", "en"), "Hi,\n\nBye.")
        self.assertEqual(st.prepare_for_llm("Items. Bullet point one, bullet point two, bullet point three.", "en"),
                         "Items.\n- One,\n- Two,\n- Three.")

    def test_a_short_correction_may_replace_everything_before_it(self):
        F = st.cleanup_is_faithful
        self.assertTrue(F("Tuesday, no wait, Wednesday.", "Wednesday.", "en"))
        self.assertTrue(F("Ten, sorry, twelve.", "Twelve.", "en"))
        self.assertFalse(F("I think we should ask Sarah, sorry, Peter first.", "Peter first.", "en"))


class TestMeetingRecorderRecovers(unittest.TestCase):
    """The recorder with fake audio libraries: a headset that drops and comes
    back must leave both files on one timeline."""

    def _fakes(self, events):
        import threading
        import types
        import time as _time
        state = {"system_opens": 0, "mic_opens": 0}

        class Speaker:
            def __init__(self, name):
                self.name = name

        class Recorder:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def record(self, numframes):
                _time.sleep(numframes / 16000)
                if state["system_opens"] == 1 and state.get("system_blocks", 0) >= 8:
                    raise RuntimeError("device lost")   # the headset drops after 0.8 s
                state["system_blocks"] = state.get("system_blocks", 0) + 1
                return np.full((numframes, 1), 0.1, np.float32)

        class Device:
            def recorder(self, **kwargs):
                state["system_opens"] += 1
                if 2 <= state["system_opens"] <= 4:
                    raise RuntimeError("no such device")   # gone for three retries
                return Recorder()

        def default_speaker():
            return Speaker("Headset" if state["system_opens"] >= 4 else "Laptop speakers")

        soundcard = types.SimpleNamespace(default_speaker=default_speaker, get_speaker=lambda name: Speaker(name),
                                          get_microphone=lambda id, include_loopback: Device())

        class InputStream:
            def __init__(self, callback, **kwargs):
                self.callback, self.running = callback, False
                state["mic_opens"] += 1
                self.number = state["mic_opens"]

            def start(self):
                self.running = True

                def run():
                    sent = 0
                    while self.running:
                        _time.sleep(0.05)
                        if self.number == 1 and sent >= 12:
                            continue   # the mic stops after 0.6 s; a reopened one works
                        self.callback(np.full((800, 1), 0.2, np.float32), 800, None, None)
                        sent += 1
                threading.Thread(target=run, daemon=True).start()

            def stop(self):
                self.running = False

            def close(self):
                self.running = False

        sounddevice = types.SimpleNamespace(InputStream=InputStream)
        return soundcard, sounddevice, state

    def test_both_streams_survive_a_dropped_headset(self):
        import time as _time
        from unittest import mock
        soundcard, sounddevice, state = self._fakes([])
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        logs = []
        rec = sm.MeetingRecorder(folder, logs.append)
        rec.SYSTEM_RETRY_SECONDS, rec.MIC_STALL_SECONDS, rec.GAP_SECONDS = 0.3, 0.5, 0.3
        with mock.patch.dict(sys.modules, {"soundcard": soundcard, "sounddevice": sounddevice}):
            rec.start()
            _time.sleep(4.0)
            meta = rec.stop()
        mic, system = (sm.read_wav(os.path.join(folder, f"{n}.wav")) for n in ("mic", "system"))
        for name, audio in (("mic", mic), ("system", system)):
            clock = meta["clock"][name]
            expected = clock["seconds"] * sm.SAMPLE_RATE + clock["first_frames"]
            self.assertLess(abs(len(audio) - expected), 0.3 * sm.SAMPLE_RATE, (name, len(audio), expected))
            self.assertTrue(meta["gaps"].get(name), (name, meta))   # the gap was filled with silence
            self.assertGreater(float(np.mean(audio == 0)), 0.1, name)
        self.assertLess(abs(len(mic) / sm.SAMPLE_RATE + meta["offsets"]["mic"]
                            - len(system) / sm.SAMPLE_RATE - meta["offsets"]["system"]), 0.4)
        self.assertEqual(meta["system_device"], "Headset")   # picked up the new default speaker
        self.assertIsNone(meta["system_error"])
        self.assertGreaterEqual(state["mic_opens"], 2)
        joined = "\n".join(logs)
        self.assertIn("System audio stopped", joined)
        self.assertIn("System audio is back (Headset)", joined)
        self.assertIn("microphone stopped", joined)
        self.assertIn("microphone is back", joined)

    def test_system_audio_that_never_works_is_reported(self):
        import time as _time
        import types
        from unittest import mock
        _, sounddevice, _ = self._fakes([])

        def broken():
            raise RuntimeError("no output device")
        soundcard = types.SimpleNamespace(default_speaker=broken, get_speaker=lambda name: broken())
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        logs = []
        rec = sm.MeetingRecorder(folder, logs.append)
        rec.SYSTEM_RETRY_SECONDS = 0.2
        with mock.patch.dict(sys.modules, {"soundcard": soundcard, "sounddevice": sounddevice}):
            rec.start()
            _time.sleep(0.8)
            meta = rec.stop()
        self.assertEqual(meta["system_error"], "no output device")
        self.assertNotIn("system", meta["clock"])
        self.assertEqual(sum("cannot be recorded" in line for line in logs), 1)   # said once, not every retry


class TestClockDrift(unittest.TestCase):
    def test_a_drifting_stream_is_stretched_to_wall_clock(self):
        sr = sm.SAMPLE_RATE
        audio = np.zeros(100 * sr, np.float32)
        audio[50 * sr] = 1.0   # a click at 50 s of device time
        clock = {"seconds": 100.3 - 0.1, "first_frames": int(0.1 * sr)}   # the device clock was 3000 ppm slow
        out = sm.correct_drift(audio, clock, chunk=7 * sr)   # chunks that do not divide the length
        self.assertEqual(len(out), int(100.3 * sr))
        self.assertAlmostEqual(int(np.argmax(out)) / sr, 50.15, delta=0.002)

    def test_noise_and_implausible_drift_are_left_alone(self):
        sr = sm.SAMPLE_RATE
        audio = np.zeros(100 * sr, np.float32)
        self.assertIs(sm.correct_drift(audio, {"seconds": 100.1, "first_frames": 0}), audio)   # < 0.2 s
        self.assertIs(sm.correct_drift(audio, {"seconds": 1000.0, "first_frames": 0}), audio)  # the PC slept
        self.assertIs(sm.correct_drift(audio, None), audio)                                     # old recordings

    def test_load_streams_lines_up_the_clocks(self):
        sr = sm.SAMPLE_RATE
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        for name, seconds in (("mic", 120.0), ("system", 119.7)):
            w = sm._WavWriter(os.path.join(folder, f"{name}.wav"))
            w.write(np.zeros(int(seconds * sr), np.float32))
            w.close()
        meta = {"offsets": {"mic": 0.0, "system": 0.5},
                "clock": {"mic": {"seconds": 120.0, "first_frames": 0, "frames": 120 * sr},
                          "system": {"seconds": 120.0, "first_frames": 0, "frames": int(119.7 * sr)}}}
        with open(os.path.join(folder, "meeting.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        manager = sm.MeetingManager.__new__(sm.MeetingManager)
        manager.log, manager.recordings_dir = (lambda *a: None), tempfile.gettempdir()
        streams, _, _ = manager._load_streams(folder)
        lengths = {name: len(audio) / sr for name, audio in streams}
        self.assertAlmostEqual(lengths["mic"], 120.0, places=2)
        self.assertAlmostEqual(lengths["system"], 120.5, places=2)   # stretched to 120 s, after its 0.5 s offset


class TestMeetingMinutesCues(unittest.TestCase):
    def test_present_tense_promises_and_habits(self):
        kinds = lambda text: [k for k, _ in sm.split_clauses(text)]
        for text in ("Jeg tjekker det i morgen.", "Jeg kigger på tilbuddet.", "Det tager jeg.",
                     "Jeg sørger for at butikkerne får besked.", "Ich kümmere mich darum.",
                     "Je m'en charge.", "Leave it with me.", "I'm on it."):
            self.assertEqual(kinds(text), ["promise"], text)
        for text in ("Jeg tjekker altid mails om morgenen.", "I always check the numbers on Monday.",
                     "Ich schaue immer zuerst in den Kalender."):
            self.assertEqual(kinds(text), [], text)

    def test_short_words_count_for_grounding(self):
        self.assertTrue(sm.grounded("Fix the bug", "Okay, I will fix the bug today."))
        self.assertTrue(sm.grounded("Book møde i maj", "Jeg booker et møde i maj."))
        self.assertFalse(sm.grounded("Send the numbers", "Yes, the one for you."))   # only stop words shared

    def test_a_hyphenated_name_is_a_faithful_cleanup(self):
        self.assertTrue(sm.faithful("Mette Kristine sender det i morgen.", "Mette-Kristine sender det i morgen.", "da"))


class TestMeetingRecorderReviewCases(unittest.TestCase):
    """Inputs from the adversarial review of the Phase 3 recorder."""

    def _recorder(self, **attrs):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        logs = []
        rec = sm.MeetingRecorder(folder, logs.append)
        for k, v in attrs.items():
            setattr(rec, k, v)
        return rec, logs

    @staticmethod
    def _sounddevice(open_seconds=0.0, deliver=lambda number: False):
        import threading
        import time as _time
        import types
        streams = []

        class InputStream:
            def __init__(self, callback, **kwargs):
                _time.sleep(open_seconds if streams else 0.0)   # reopening a Bluetooth device is slow
                self.callback, self.running, self.closed = callback, False, False
                streams.append(self)

            def start(self):
                self.running = True
                number = len(streams)

                def run():
                    while self.running:
                        _time.sleep(0.05)
                        if self.running and deliver(number):
                            self.callback(np.zeros((800, 1), np.float32), 800, None, None)
                threading.Thread(target=run, daemon=True).start()

            def stop(self):
                self.running = False

            def close(self):
                self.running, self.closed = False, True

        return types.SimpleNamespace(InputStream=InputStream), streams

    @staticmethod
    def _soundcard(names):
        """names(): the current default speaker name."""
        import time as _time
        import types

        class Recorder:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def record(self, numframes):
                _time.sleep(numframes / 16000)
                return np.zeros((numframes, 1), np.float32)

        speaker = lambda: types.SimpleNamespace(name=names())
        device = types.SimpleNamespace(recorder=lambda **kw: Recorder())
        return types.SimpleNamespace(default_speaker=speaker, get_speaker=lambda n: speaker(),
                                     get_microphone=lambda id, include_loopback: device)

    def test_stop_during_a_slow_mic_reopen_leaves_no_stream_open(self):
        import time as _time
        from unittest import mock
        sd, streams = self._sounddevice(open_seconds=0.4)   # a dead mic, slow to reopen
        rec, _ = self._recorder(MIC_STALL_SECONDS=0.2)
        with mock.patch.dict(sys.modules, {"sounddevice": sd, "soundcard": self._soundcard(lambda: "Speakers")}):
            rec.start()
            deadline = _time.monotonic() + 5
            while len(streams) < 2 and _time.monotonic() < deadline:
                _time.sleep(0.01)   # until the reopen has started
            _time.sleep(0.05)
            rec.stop()
            _time.sleep(0.6)    # let a reopen that was in flight finish
        self.assertGreaterEqual(len(streams), 2)
        self.assertTrue(all(s.closed or not s.running for s in streams), [(s.running, s.closed) for s in streams])
        self.assertIsNone(rec.mic_stream)

    def test_a_dead_mic_does_not_keep_the_meeting_past_its_limit(self):
        import threading
        import time as _time
        from unittest import mock
        sd, _ = self._sounddevice()
        fired = threading.Event()
        rec, _ = self._recorder(max_seconds=0.5, on_limit=fired.set)
        with mock.patch.dict(sys.modules, {"sounddevice": sd, "soundcard": self._soundcard(lambda: "Speakers")}):
            rec.start()
            self.assertTrue(fired.wait(3))
            rec.stop()

    def test_system_audio_follows_a_new_default_speaker(self):
        import time as _time
        from unittest import mock
        t0 = _time.monotonic()
        names = lambda: "Headset" if _time.monotonic() - t0 < 0.8 else "Laptop speakers"
        sd, _ = self._sounddevice(deliver=lambda n: True)
        rec, logs = self._recorder(SPEAKER_CHECK_SECONDS=0.2)
        with mock.patch.dict(sys.modules, {"sounddevice": sd, "soundcard": self._soundcard(names)}):
            rec.start()
            _time.sleep(1.6)
            meta = rec.stop()
        self.assertEqual(meta["system_device"], "Laptop speakers")
        self.assertTrue(any("now comes from Laptop speakers" in line for line in logs), logs)
        clock = meta["clock"]["system"]
        self.assertLess(abs(clock["frames"] - clock["seconds"] * sm.SAMPLE_RATE - clock["first_frames"]),
                        0.3 * sm.SAMPLE_RATE)   # the switch did not cost the timeline

    def test_an_unfilled_gap_does_not_count_as_clock_drift(self):
        import datetime
        rec, _ = self._recorder(MAX_GAP_SECONDS=2)
        rec._t0, rec.started = 0.0, datetime.datetime.now()
        rec.mic_writer = sm._WavWriter(os.path.join(rec.folder, "mic.wav"))
        rec.system_writer = sm._WavWriter(os.path.join(rec.folder, "system.wav"))
        block = np.zeros(1600, np.float32)
        t = 0.1
        for _ in range(50):
            rec._write("system", rec.system_writer, block, t)
            t += 0.1
        t += 10.0   # the PC slept 10 s; only 2 s are filled
        for _ in range(50):
            rec._write("system", rec.system_writer, block, t)
            t += 0.1
        meta = rec.stop()
        clock = meta["clock"]["system"]
        self.assertAlmostEqual(clock["frames"] / sm.SAMPLE_RATE, clock["seconds"] + clock["first_frames"] / sm.SAMPLE_RATE,
                               delta=0.15)
        audio = np.zeros(clock["frames"], np.float32)
        self.assertIs(sm.correct_drift(audio, clock), audio)   # nothing to stretch
        self.assertEqual(meta["gaps"]["system"], [[5.0, 2.0]])

    def test_dropouts_are_noted_in_the_transcript(self):
        notes = sm.gap_notes({"gaps": {"system": [[754.0, 21.0], [900.0, 1.5]], "mic": [[60.0, 4.0]]}})
        self.assertEqual(len(notes), 2)
        self.assertIn("The call audio dropped out during the recording, at 00:12:34 (21 s)", notes[0])
        self.assertNotIn("00:15:00", notes[0])   # under 3 s is no dropout worth a note
        self.assertIn("The microphone dropped out", notes[1])
        self.assertEqual(sm.gap_notes({}), [])


class TestMeetingMinutesReviewCases(unittest.TestCase):
    def test_habit_words_only_in_the_promise_clause(self):
        kinds = lambda text: [k for k, _ in sm.split_clauses(text)]
        for text in ("Jeg tjekker det hver uge, men jeg sender rapporten i morgen.",
                     "Normally I would, but I will send it tomorrow.",
                     "I'll check it, I usually do that on Mondays anyway.",
                     "I will send it every week from now on.", "Jeg sender det hver uge fra nu af.",
                     "Ich kümmere mich immer noch darum.", "Je m'en occupe toujours."):
            self.assertEqual(kinds(text), ["promise"], text)
        self.assertEqual(kinds("Jeg tjekker altid mails om morgenen."), [])

    def test_refusals_and_opinions_are_no_promises(self):
        kinds = lambda text: [k for k, _ in sm.split_clauses(text)]
        for text in ("Det gør jeg ikke.", "Det gør jeg ikke, det er for dyrt.", "Jeg tjekker det aldrig.",
                     "I will not do that.", "Ich mache das nicht.", "Jeg finder det svært at tro.",
                     "Jeg hører hvad du siger.", "Jeg laver bare sjov."):
            self.assertEqual(kinds(text), [], text)
        self.assertEqual(kinds("I'll send it, not the old one, the new one."), ["promise"])

    def test_a_refusal_does_not_make_an_owner(self):
        turns = [{"label": "A", "text": "Kan du ringe til kunden?"},
                 {"label": "B", "text": "Det gør jeg ikke, det er din kunde."}]
        self.assertIsNone(sm.agreed_by_next(turns, 0))
        turns[1]["text"] = "Det gør jeg i morgen."
        self.assertEqual(sm.agreed_by_next(turns, 0), "B")


class TestMeetingReviewRound2(unittest.TestCase):
    """Inputs from the second review round of Phase 3."""

    def test_cant_is_a_refusal(self):
        kinds = lambda text: [k for k, _ in sm.split_clauses(text)]
        for text in ("I can't send it today.", "I can't, sorry.", "I can't make it before Friday."):
            self.assertEqual(kinds(text), [], text)
        for answer in ("I can't, sorry.", "I can't make it today, no.", "Ja, det gør jeg ikke."):
            turns = [{"label": "A", "text": "Can you send the report today?"}, {"label": "B", "text": answer}]
            self.assertIsNone(sm.agreed_by_next(turns, 0), answer)

    def test_negation_inside_what_is_promised(self):
        kinds = lambda text: [k for k, _ in sm.split_clauses(text)]
        self.assertEqual(kinds("Jeg sørger for at det ikke sker igen."), ["promise"])
        self.assertEqual(kinds("I'll make sure that it does not happen again."), ["promise"])
        self.assertEqual(kinds("I will send it every day until it is fixed."), ["promise"])
        self.assertEqual(kinds("Ich schicke es dir jeden Tag bis es läuft."), ["promise"])

    def test_a_short_reopen_is_filled_exactly(self):
        import datetime
        rec = sm.MeetingRecorder(tempfile.mkdtemp(), lambda *a: None)
        self.addCleanup(shutil.rmtree, rec.folder, True)
        rec._t0, rec.started = 0.0, datetime.datetime.now()
        rec.mic_writer = sm._WavWriter(os.path.join(rec.folder, "mic.wav"))
        rec.system_writer = sm._WavWriter(os.path.join(rec.folder, "system.wav"))
        block = np.zeros(1600, np.float32)
        t = 0.1
        for switch in range(4):
            if switch:
                rec._reopened["system"] = t   # a device switch: 0.3 s, under GAP_SECONDS
                t += 0.3
            for _ in range(20):
                rec._write("system", rec.system_writer, block, t)
                t += 0.1
        meta = rec.stop()
        clock = meta["clock"]["system"]
        self.assertAlmostEqual(clock["frames"] / sm.SAMPLE_RATE, clock["seconds"] + clock["first_frames"] / sm.SAMPLE_RATE,
                               delta=0.01)   # 0.9 s of switches filled: nothing left for the drift correction
        self.assertEqual(len(meta["gaps"]["system"]), 3)

    def test_jitter_after_the_first_open_adds_no_silence(self):
        rec = sm.MeetingRecorder(tempfile.mkdtemp(), lambda *a: None)
        self.addCleanup(shutil.rmtree, rec.folder, True)
        rec._t0 = 0.0
        writer = sm._WavWriter(os.path.join(rec.folder, "system.wav"))
        rec._reopened["system"] = 0.0   # set when the device first opens
        block = np.zeros(1600, np.float32)
        rec._write("system", writer, block, 0.1)
        rec._write("system", writer, block, 0.205)   # 5 ms late: normal jitter
        writer.close()
        self.assertEqual(writer.frames, 3200)

    def test_the_configured_speaker_is_used_again_when_it_is_back(self):
        import time as _time
        import types
        from unittest import mock
        t0 = _time.monotonic()
        opened = []

        class Recorder:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def record(self, numframes):
                _time.sleep(numframes / 16000)
                if opened[-1] == "Headset" and len(opened) == 1 and _time.monotonic() - t0 > 0.4:
                    raise RuntimeError("device lost")
                return np.zeros((numframes, 1), np.float32)

        def get_speaker(name):
            if 0.4 < _time.monotonic() - t0 < 1.0:
                raise IndexError("no such speaker")   # the headset is gone for a moment
            return types.SimpleNamespace(name="Headset")

        def get_microphone(id, include_loopback):
            return types.SimpleNamespace(recorder=lambda **kw: opened.append(id) or Recorder())

        soundcard = types.SimpleNamespace(get_speaker=get_speaker, get_microphone=get_microphone,
                                          default_speaker=lambda: types.SimpleNamespace(name="Laptop speakers"))
        sd, _ = TestMeetingRecorderReviewCases._sounddevice(deliver=lambda n: True)
        rec = sm.MeetingRecorder(tempfile.mkdtemp(), lambda *a: None, speaker="Headset")
        self.addCleanup(shutil.rmtree, rec.folder, True)
        rec.SYSTEM_RETRY_SECONDS, rec.SPEAKER_CHECK_SECONDS = 0.1, 0.2
        with mock.patch.dict(sys.modules, {"soundcard": soundcard, "sounddevice": sd}):
            rec.start()
            _time.sleep(2.0)
            meta = rec.stop()
        self.assertEqual(opened[:2], ["Headset", "Laptop speakers"])
        self.assertEqual(meta["system_device"], "Headset")   # back on the configured speaker

    def test_a_mic_opened_outside_the_lock_is_still_closed_on_stop(self):
        import time as _time
        from unittest import mock
        sd, streams = TestMeetingRecorderReviewCases._sounddevice(open_seconds=2.0)
        rec = sm.MeetingRecorder(tempfile.mkdtemp(), lambda *a: None)
        self.addCleanup(shutil.rmtree, rec.folder, True)
        rec.MIC_STALL_SECONDS = 0.2
        soundcard = TestMeetingRecorderReviewCases._soundcard(lambda: "Speakers")
        with mock.patch.dict(sys.modules, {"sounddevice": sd, "soundcard": soundcard}):
            rec.start()
            _time.sleep(0.9)   # the reopen is now inside its 2 s open
            rec.stop()
            _time.sleep(2.2)
        self.assertTrue(all(s.closed or not s.running for s in streams))


class TestMeetingReviewRound3(unittest.TestCase):
    """Inputs from the third review round of Phase 3."""

    def test_a_refusal_then_a_promise_still_has_an_owner(self):
        for answer in ("Yes, but I can't today, I'll send it tomorrow.", "I can't today. I'll send it tomorrow.",
                       "Ja, men jeg kan ikke i dag, jeg sender den i morgen."):
            turns = [{"label": "A", "text": "Can you send the report today?"}, {"label": "B", "text": answer}]
            self.assertEqual(sm.agreed_by_next(turns, 0), "B", answer)

    def test_a_deadline_does_not_turn_a_habit_into_a_promise(self):
        kinds = lambda text: [k for k, _ in sm.split_clauses(text)]
        for text in ("Ich prüfe das normalerweise bis Mittwoch.", "Ich mache das immer bis Freitag.",
                     "Jeg tjekker altid indtil klokken fem.", "Je vérifie d'habitude jusqu'au vendredi."):
            self.assertEqual(kinds(text), [], text)
        self.assertEqual(kinds("I will send it every day until it is fixed."), ["promise"])
        self.assertEqual(kinds("I will always send it by Friday from now on."), ["promise"])

    def test_typographic_apostrophes(self):
        kinds = lambda text: [k for k, _ in sm.split_clauses(text)]
        self.assertEqual(kinds("I can\u2019t send it today."), [])
        self.assertEqual(kinds("I\u2019ll send it tomorrow."), ["promise"])
        self.assertEqual(sm.split_clauses("I\u2019ll send it tomorrow.")[0][1], "I\u2019ll send it tomorrow.")

    def test_a_late_block_of_the_old_stream_does_not_use_up_the_reopen(self):
        rec = sm.MeetingRecorder(tempfile.mkdtemp(), lambda *a: None)
        self.addCleanup(shutil.rmtree, rec.folder, True)
        rec._t0 = 0.0
        writer = sm._WavWriter(os.path.join(rec.folder, "mic.wav"))
        block = np.zeros(800, np.float32)   # 50 ms blocks
        rec._write("mic", writer, block, 10.05)
        rec._write("mic", writer, block, 10.10)
        rec._reopened["mic"] = 12.65              # the stall reopen
        rec._write("mic", writer, block, 12.60)   # the old stream's delayed block (GIL pause)
        rec._write("mic", writer, block, 12.95)   # the new stream's first block, 0.3 s later
        writer.close()
        expected = (12.95 - 10.05) * sm.SAMPLE_RATE + 800
        self.assertLess(abs(writer.frames - expected), 2, (writer.frames, expected))


class TestMeetingReviewRound4(unittest.TestCase):
    """Inputs from the fourth review round of Phase 3."""

    def test_a_bare_no_then_a_promise_has_an_owner(self):
        for answer, owner in (("No. I'll send it tomorrow.", "B"), ("Nej. Jeg sender den i morgen.", "B"),
                              ("No.", None), ("No, I can't. Peter, can you?", None)):
            turns = [{"label": "A", "text": "Can you send the report today?"}, {"label": "B", "text": answer}]
            self.assertEqual(sm.agreed_by_next(turns, 0), owner, answer)

    def test_a_fast_first_callback_after_a_reopen_is_filled(self):
        rec = sm.MeetingRecorder(tempfile.mkdtemp(), lambda *a: None)
        self.addCleanup(shutil.rmtree, rec.folder, True)
        rec._t0 = 0.0
        writer = sm._WavWriter(os.path.join(rec.folder, "mic.wav"))
        block = np.zeros(800, np.float32)
        rec._write("mic", writer, block, 10.05)
        rec._write("mic", writer, block, 10.10)
        rec._reopened["mic"] = 10.12              # taken when the old stream closed
        rec._write("mic", writer, block, 10.45)   # the new stream's first block, before the reopen returned
        rec._write("mic", writer, block, 10.50)
        writer.close()
        self.assertLess(abs(writer.frames - ((10.50 - 10.05) * sm.SAMPLE_RATE + 800)), 2)
        self.assertEqual(len(rec.gaps["mic"]), 1)

    def test_no_zero_length_gap_rows(self):
        rec = sm.MeetingRecorder(tempfile.mkdtemp(), lambda *a: None)
        self.addCleanup(shutil.rmtree, rec.folder, True)
        rec._t0 = 0.0
        writer = sm._WavWriter(os.path.join(rec.folder, "mic.wav"))
        block = np.zeros(800, np.float32)
        rec._write("mic", writer, block, 10.05)
        rec._reopened["mic"] = 10.06
        rec._write("mic", writer, block, 10.1000001)   # a few microseconds late: float noise
        writer.close()
        self.assertEqual(rec.gaps["mic"], [])
        self.assertEqual(writer.frames, 1600)


if __name__ == "__main__":
    unittest.main()
