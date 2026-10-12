"""
Speak - Unit and Regression Test Suite
Run with: python -m unittest test_speak.py
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

import speak_dictionary as sd
import speak_meeting as sm
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

    def test_needs_llm_fast_path(self):
        dummy_app = object.__new__(speak.SpeakApp)
        # Short clean phrases skip LLM
        self.assertFalse(dummy_app._needs_llm("Yes that sounds good."))
        self.assertFalse(dummy_app._needs_llm("Tak for i dag."))

        # Phrases with verbal fillers require LLM
        self.assertTrue(dummy_app._needs_llm("um yes that is right"))
        self.assertTrue(dummy_app._needs_llm("øh det tror jeg nok"))

        # Stutters require LLM
        self.assertTrue(dummy_app._needs_llm("we should go to the the store"))

        # Spoken commands require LLM
        self.assertTrue(dummy_app._needs_llm("send email bullet point now"))
        self.assertTrue(dummy_app._needs_llm("first line new line second line"))

        # Numbers require LLM
        self.assertTrue(dummy_app._needs_llm("order 55 items"))

        # Phrases longer than 8 words require LLM
        self.assertTrue(dummy_app._needs_llm("one two three four five six seven eight nine ten"))

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
        self.assertEqual(float(audio.min()), 1.0)   # input untouched

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
    def test_meeting_cleanup_skips_usage_counts_and_llm_when_off(self):
        class FakeDict:
            def __init__(self):
                self.noted = []

            def prepare(self, text, lang):
                return text.replace("sales force", "Salesforce"), "", [("sales force", "Salesforce")]

            def apply_casing(self, text):
                return text

            def note_usage(self, text):
                self.noted.append(text)

        app = speak.SpeakApp.__new__(speak.SpeakApp)
        app.config = {}
        app.dictionary = FakeDict()
        app.ai_mode, app.local_llm = "off", None
        self.assertEqual(app.clean_text("we use sales force", "en", verbose=False), "we use Salesforce")
        self.assertEqual(app.dictionary.noted, [])
        app.clean_text("we use sales force", "en")
        self.assertEqual(app.dictionary.noted, ["we use Salesforce"])


if __name__ == "__main__":
    unittest.main()
