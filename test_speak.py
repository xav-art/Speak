"""
Speak - Unit and Regression Test Suite
Run with: python -m unittest test_speak.py
"""

import json
import os
import shutil
import tempfile
import unittest

import speak_dictionary as sd
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


if __name__ == "__main__":
    unittest.main()
