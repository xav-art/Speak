r"""
Speak - Personal Dictionary & Auto-Learning
Author: xav-art (MIT License)

Stores the user's own words (names, jargon, product names, acronyms) and
learns new ones automatically from corrections the user makes to inserted text.

The learning algorithm mirrors OpenWhispr's correctionLearner:
- Word-level LCS alignment between the inserted text and the corrected text
- Only substitutions count (not deletions/insertions of whole phrases)
- Corrected word must be >= 3 chars, not a common everyday word, and
  phonetically close to the original (Levenshtein ratio <= 0.65)
- If more than 50% of the words changed, it was a rewrite, not a correction

How the dictionary is used in the pipeline (see speak.py):
1. Whisper   : words go into initial_prompt, which biases acoustic decoding
               toward the right spelling. This is where most of the win is.
2. Pre-LLM   : learned mishearings ('Anika' -> 'Annika') are replaced
               deterministically once confirmed (same context, or seen 3x).
3. LLM       : only when a dictionary word or a known mishearing is actually
               present in the transcript, a one-line hint is appended to the
               cleanup prompt. Ordinary dictations keep the exact prompt the
               user has tuned - Qwen 1.5B is sensitive to prompt changes.
               The hint alone does not make a short phrase wait for Qwen:
               only what step 4 cannot fix does (see prepare()).
4. Post-LLM  : exact casing is enforced for multi-word / oddly-cased entries
               ('øresund bridge' -> 'Øresund Bridge', 'sap' -> 'SAP').
"""

import json
import os
import re
import threading
import time
import unicodedata

DICTIONARY_FILE = "speak_dictionary.json"

# Whisper prompts are capped at 224 tokens. ~60 words keeps us safely under.
MAX_PROMPT_WORDS = 60

# A mishearing is replaced without asking the LLM once it has been corrected
# this many times (in any context).
AUTO_REPLACE_AFTER = 3

# Everyday words in all four supported languages. Swapping one of these for
# another ("why" -> "what", "ikke" -> "aldrig") is a content edit, not vocabulary.
COMMON_WORDS = set("""
the and for not with you this but his from they say her she will one all would
there their what out about who get which when make can like time just him know
take into year your good some could them see other than then now look only come
over think also back after use two how our work first well way even new want
because any these give day most are was were been has had did does said went
made got came took saw knew thought where why here very much many still too again
off down never every own same another both each few more less last next while
before through under between should might must being have that its yes okay
og det den der jeg ikke til med han hun var kan skal vil har havde blev bliver
som men også hvis når hvor hvad hvem hvordan noget nogen alle ingen meget mere
mest lidt dem dig mig sig vores jeres deres denne dette disse være bare kun
und der die das ist nicht ein eine sie wir ihr mit von auf für aber auch wenn
dann noch schon nur sehr mehr kann muss soll wird wurde haben hatte sind war
les des une est pas que qui dans pour avec sur mais aussi tout tous très plus
sont ont été fait être avoir cette ces son ses leur nous vous elle ils elles
""".split())

_WORD_EDGE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def tokenize(text):
    """Split into words, stripping punctuation from the edges of each word."""
    out = []
    for w in re.split(r"\s+", text or ""):
        w = _WORD_EDGE.sub("", w)
        if w:
            out.append(w)
    return out


def edit_distance(a, b):
    """Levenshtein distance (iterative, two-row)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fold(s):
    return unicodedata.normalize("NFC", s).casefold()


def _word_pattern(word):
    return re.compile(r"(?<!\w)" + re.escape(word).replace(r"\ ", r"\s+") + r"(?!\w)", re.IGNORECASE)


# Opening quotes, brackets and bullet marks that can stand before the first
# word of a sentence ('- Machine learning...', '"Machine learning..."').
_OPENERS = "\"'\u201c\u201d\u2018\u2019\u201e\u00ab\u00bb([{-\u2013\u2022*\u00b7"
_CLOSERS = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb)]}"


def _at_sentence_start(text, pos):
    """True when position pos starts the text, a line, a sentence, a bullet
    or a quote, or follows a colon or an ellipsis."""
    before = text[:pos]
    core = before.rstrip().rstrip(_OPENERS).rstrip()
    if not core or "\n" in before[len(core):]:
        return True
    closed = core.rstrip(_CLOSERS)   # '(See above.) Machine learning'
    return bool(closed) and closed[-1] in ".!?:\u2026"


def find_edited_region(original, field_value):
    """
    The user may have selected more than the dictated sentence when copying the
    corrected text. Find the window in field_value that best matches original.
    """
    if len(field_value) <= len(original) * 1.5:
        return field_value
    if original in field_value:
        return original

    orig_words = tokenize(original)
    field_words = tokenize(field_value)
    window = len(orig_words)
    if field_words and len(field_words) <= window:
        return field_value

    best_start, best_score = 0, -1
    for i in range(len(field_words) - window + 1):
        matches = sum(
            1 for j in range(window)
            if _fold(field_words[i + j]) == _fold(orig_words[j])
        )
        if matches > best_score:
            best_score, best_start = matches, i

    if best_score < window * 0.3:
        return field_value
    return " ".join(field_words[best_start:best_start + window])


def find_substitutions(orig_words, edited_words):
    """
    Word-level LCS alignment -> list of (orig_word, corrected_word, context)
    where context is the neighbouring original word (next, else previous).
    """
    m, n = len(orig_words), len(edited_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        oi = _fold(orig_words[i - 1])
        for j in range(1, n + 1):
            if oi == _fold(edited_words[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    aligned = []  # (orig_word or None, edited_word or None, orig_index)
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and _fold(orig_words[i - 1]) == _fold(edited_words[j - 1]):
            aligned.append((orig_words[i - 1], edited_words[j - 1], i - 1))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            aligned.append((None, edited_words[j - 1], None))
            j -= 1
        else:
            aligned.append((orig_words[i - 1], None, i - 1))
            i -= 1
    aligned.reverse()

    def context_for(idx):
        if idx is None:
            return None
        if idx + 1 < len(orig_words):
            return _fold(orig_words[idx + 1])
        if idx > 0:
            return _fold(orig_words[idx - 1])
        return None

    subs = []
    for k in range(len(aligned)):
        o, e, oi = aligned[k]
        # Matched case-insensitively but spelled differently ("teams" -> "Teams")
        if o is not None and e is not None and o != e:
            subs.append((o, e, context_for(oi)))
            continue
        if k + 1 < len(aligned):
            no, ne, _ = aligned[k + 1]
            if o is not None and e is None and no is None and ne is not None:
                subs.append((o, ne, context_for(oi)))
    return subs


def extract_corrections(original_text, corrected_text, existing_words):
    """
    Compare the text Speak inserted with the text after the user fixed it.
    Returns (original_word, corrected_word, context) triples worth learning.
    """
    if not original_text or not corrected_text:
        return []
    if original_text.strip() == corrected_text.strip():
        return []

    region = find_edited_region(original_text, corrected_text)
    if region == original_text:
        return []

    orig_words = tokenize(original_text)
    edited_words = tokenize(region)
    if not orig_words or not edited_words:
        return []

    subs = find_substitutions(orig_words, edited_words)
    if len(subs) > len(orig_words) * 0.5:
        return []

    known = {_fold(w) for w in existing_words}
    seen = set()
    results = []
    for orig, corrected, context in subs:
        key = _fold(corrected)
        if key in seen:
            continue
        if len(corrected) < 3:
            continue
        if key in COMMON_WORDS:
            continue

        if _fold(orig) == key:
            # Case-only change ("salesforce" -> "Salesforce", "sap" -> "SAP") is
            # a spelling preference worth remembering, but only if it adds capitals.
            if corrected == orig or not any(c.isupper() for c in corrected):
                continue
        else:
            # 0.65 allows phonetic fixes like "Shunade" -> "Sinead" (4/7 = 0.57)
            # while rejecting unrelated word swaps.
            dist = edit_distance(_fold(orig), key)
            if dist / max(len(orig), len(corrected)) > 0.65:
                continue

        # A word already in the dictionary can still teach us a new mishearing.
        results.append((orig, corrected, context))
        seen.add(key)
    return results


class Dictionary:
    """Persistent, thread-safe personal word list stored next to speak.py."""

    def __init__(self, base_dir):
        self.path = os.path.join(base_dir, DICTIONARY_FILE)
        self.lock = threading.Lock()
        # Held across the whole save: two threads writing the same .tmp file
        # (the worker's usage counts and the learn hotkey, say) interleave
        # their bytes and can move a broken file into place.
        self._save_lock = threading.Lock()
        self._read_failed = False   # set by load(): then save() never writes
        self.entries = {}      # fold(word)  -> {"word", "source", "uses", "added"}
        self.corrections = {}  # fold(heard) -> {"heard", "word", "count", "contexts": [...]}
        self.load()

    # ---- persistence -------------------------------------------------------

    def load(self):
        if not os.path.exists(self.path):
            return
        data = None
        for attempt in range(5):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    entries, corrections = self._parse(json.load(f))
                break
            except FileNotFoundError:
                return   # deleted meanwhile: nothing to keep, start empty
            except OSError as e:
                # Locked for a moment (antivirus, backup, another Speak saving).
                if attempt < 4:
                    time.sleep(0.1)
                    continue
                # Never overwrite a file that could not be read.
                self._read_failed = True
                print(f"[Dictionary Warning] {DICTIONARY_FILE} could not be opened ({e}). "
                      f"Speak runs without your words and will not save changes until restarted.", flush=True)
                return
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                # Broken content (bad JSON, or JSON of the wrong shape).
                # Starting empty would overwrite it at the next save: keep it
                # aside, like load_config does.
                backup = f"{self.path}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
                n = 1
                while os.path.exists(backup):   # never replace an earlier backup
                    n += 1
                    backup = f"{self.path}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}-{n}"
                try:
                    os.replace(self.path, backup)
                except OSError:
                    self._read_failed = True
                    backup = None
                print(f"[Dictionary Warning] {DICTIONARY_FILE} could not be read ({e}). "
                      + (f"It was kept as {backup}; starting with an empty dictionary." if backup
                         else "Speak will not save changes until restarted."), flush=True)
                return
        self.entries, self.corrections = entries, corrections

    @staticmethod
    def _parse(data):
        """File content -> (entries, corrections). Raises ValueError,
        TypeError, AttributeError or KeyError when the shape is wrong."""
        if not isinstance(data, dict):
            raise ValueError("not a dictionary file")
        if not isinstance(data.get("words", []), list) or not isinstance(data.get("corrections", []), list):
            raise ValueError("'words' and 'corrections' must be lists")
        entries, corrections = {}, {}
        for e in data.get("words", []):
            if isinstance(e, str):
                e = {"word": e}
            w = str(e.get("word", "")).strip()
            if w:
                entries[_fold(w)] = {
                    "word": w,
                    "source": e.get("source", "manual"),
                    "uses": int(e.get("uses", 0)),
                    "added": e.get("added", time.strftime("%Y-%m-%d")),
                }
        for c in data.get("corrections", []):
            heard = str(c.get("heard", "")).strip()
            word = str(c.get("word", "")).strip()
            if heard and word and _fold(heard) != _fold(word):
                corrections[_fold(heard)] = {
                    "heard": heard,
                    "word": word,
                    "count": int(c.get("count", 1)),
                    "contexts": [str(x) for x in c.get("contexts", []) if x],
                }
        return entries, corrections

    @property
    def read_only(self):
        """True when the file could not be read at startup: changes are kept
        for this session only, so the file is never overwritten."""
        return self._read_failed

    def save(self):
        with self._save_lock:
            with self.lock:
                words = sorted((dict(e) for e in self.entries.values()), key=lambda e: _fold(e["word"]))
                corrections = sorted((dict(c, contexts=list(c["contexts"])) for c in self.corrections.values()),
                                     key=lambda c: _fold(c["heard"]))
            if self._read_failed:
                return
            tmp = self.path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"words": words, "corrections": corrections}, f, indent=2, ensure_ascii=False)
                # On Windows the replace fails while anything else has the
                # file open (a reader, antivirus): retry briefly, never drop
                # the save silently.
                for attempt in range(10):
                    try:
                        os.replace(tmp, self.path)
                        return
                    except PermissionError:
                        if attempt == 9:
                            raise
                        time.sleep(0.05)
            except Exception as e:
                print(f"[Dictionary Warning] Could not save {DICTIONARY_FILE} ({e}).", flush=True)
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # ---- editing -----------------------------------------------------------

    def add(self, word, source="manual"):
        word = str(word).strip().strip(",;")
        if not word:
            return False
        key = _fold(word)
        with self.lock:
            existing = self.entries.get(key)
            if existing and existing["word"] == word:
                return False
            self.entries[key] = {
                "word": word,
                "source": source,
                "uses": existing["uses"] if existing else 0,
                "added": existing["added"] if existing else time.strftime("%Y-%m-%d"),
            }
        self.save()
        return True

    def remove(self, word):
        key = _fold(str(word).strip())
        with self.lock:
            removed = self.entries.pop(key, None) is not None
            for k in [k for k, c in self.corrections.items() if _fold(c["word"]) == key]:
                del self.corrections[k]
                removed = True
        if removed:
            self.save()
        return removed

    def clear(self):
        with self.lock:
            self.entries = {}
            self.corrections = {}
        self.save()

    def words(self):
        with self.lock:
            return [e["word"] for e in self.entries.values()]

    def __len__(self):
        return len(self.entries)

    def record_correction(self, heard, word, context=None):
        key = _fold(heard)
        if key == _fold(word):
            return
        with self.lock:
            c = self.corrections.get(key)
            if c is None:
                c = {"heard": heard, "word": word, "count": 0, "contexts": []}
                self.corrections[key] = c
            c["word"] = word
            c["count"] += 1
            if context and context not in c["contexts"]:
                c["contexts"].append(context)

    def learn_from_correction(self, original_text, corrected_text):
        """Auto-learn: returns list of (heard, corrected) pairs that were learned."""
        triples = extract_corrections(original_text, corrected_text, self.words())
        learned = []
        for heard, corrected, context in triples:
            self.add(corrected, source="learned")
            self.record_correction(heard, corrected, context)
            learned.append((heard, corrected))
        if learned:
            self.save()
        return learned

    def note_usage(self, text):
        """Count how often each word actually shows up in dictated text (used for prompt ranking)."""
        if not text or not self.entries:
            return
        changed = False
        with self.lock:
            for e in self.entries.values():
                if _word_pattern(e["word"]).search(text):
                    e["uses"] += 1
                    changed = True
        if changed:
            self.save()

    # ---- pipeline hooks ----------------------------------------------------

    def prompt_words(self, limit=MAX_PROMPT_WORDS):
        """Most useful words first: frequently used, then most recently added."""
        with self.lock:
            ranked = sorted(
                self.entries.values(),
                key=lambda e: (-e["uses"], e["added"], _fold(e["word"])),
            )
        return [e["word"] for e in ranked[:limit]]

    def whisper_prompt(self):
        """
        Whisper's initial_prompt biases decoding toward the listed spellings.
        A natural-looking comma list works better than a bare word dump.
        """
        words = self.prompt_words()
        if not words:
            return None
        return ", ".join(words) + "."

    def correction_pairs(self):
        """(heard, word) for every learned mishearing: the LLM may make these
        replacements (see speak_text.cleanup_is_faithful)."""
        with self.lock:
            return [(c["heard"], c["word"]) for c in self.corrections.values()]

    @staticmethod
    def casing_is_enforced(word):
        """True when apply_casing() fixes this entry's spelling by itself:
        multi-word entries and words with a capital after the first letter
        ('SAP', 'iPhone', 'Business Central', 'machine learning'). Single
        words with at most a first capital ('Mette', 'webshop') are left to
        the LLM hint: forcing them would also change the ordinary word."""
        return " " in word or word[1:] != word[1:].lower()

    @staticmethod
    def _expected_form(word, found, text, pos):
        """The spelling the occurrence `found` at text[pos] should have. An
        all-lowercase entry ('excel', 'machine learning') says which word it
        is, not that a capital at the start of a sentence is wrong: there it
        keeps the found first letter ('Machine Learning' -> 'Machine learning')."""
        if word == word.lower() and _at_sentence_start(text, pos):
            return found[0] + word[1:]
        return word

    def prepare(self, raw_text, lang):
        """
        Run before the LLM. Returns (text, hint, applied, needs_llm):
        - text     : raw transcript with confirmed mishearings already replaced
        - hint     : one line for the cleanup prompt, or "" when nothing is relevant
        - applied  : list of (heard, word) replaced deterministically
        - needs_llm: True when the hint asks for something only the LLM can do
                     (an unconfirmed mishearing, or a name whose capital
                     apply_casing does not enforce). A word that is already
                     spelled right, or that apply_casing fixes, does not stop
                     a short phrase from skipping the LLM.
        """
        if not raw_text:
            return raw_text, "", [], False
        with self.lock:
            corrections = list(self.corrections.values())
            entries = list(self.entries.values())

        text = raw_text
        applied = []
        pending = []  # (heard, word) still present -> LLM hint
        tokens = [_fold(t) for t in tokenize(text)]
        for c in corrections:
            pat = _word_pattern(c["heard"])
            if not pat.search(text):
                continue
            confirmed = c["count"] >= AUTO_REPLACE_AFTER
            if not confirmed and c["contexts"]:
                heard_tokens = [_fold(t) for t in tokenize(c["heard"])]
                for i, t in enumerate(tokens):
                    if t != heard_tokens[0]:
                        continue
                    end = i + len(heard_tokens)
                    neighbours = set()
                    if end < len(tokens):
                        neighbours.add(tokens[end])
                    if i > 0:
                        neighbours.add(tokens[i - 1])
                    if neighbours & set(c["contexts"]):
                        confirmed = True
                        break
            if confirmed:
                text = pat.sub(c["word"], text)
                applied.append((c["heard"], c["word"]))
            else:
                pending.append((c["heard"], c["word"]))

        # Dictionary words present in the text: exact form -> "keep", other
        # casing -> "is spelled". Every occurrence counts ("Jira and jira"),
        # and the longest entry wins a span: 'Mette' is not a separate match
        # inside 'mette-kristine' when 'Mette-Kristine' is an entry too.
        keep, respell = [], []
        needs_llm = bool(pending)
        claimed = set()   # character offsets inside a longer entry's match
        for e in sorted(entries, key=lambda e: -len(e["word"])):
            word = e["word"]
            wrong, right = None, []
            for m in _word_pattern(word).finditer(text):
                if m.start() in claimed and m.end() - 1 in claimed:
                    continue
                claimed.update(range(m.start(), m.end()))
                expected = self._expected_form(word, m.group(0), text, m.start())
                if m.group(0) == expected:
                    # The form found ('Excel' at a sentence start), not the
                    # entry: "keep 'excel'" would invite Qwen to lowercase it.
                    if expected not in right:
                        right.append(expected)
                elif wrong is None:
                    wrong = (m.group(0), expected)
            # With a respell for this entry, a sentence-start capital in the
            # keep list would contradict it: keep only the entry's own form.
            keep.extend(f for f in right if wrong is None or f == word)
            if wrong is not None:
                respell.append(wrong)
                # A lowercase single word ('webshop') gets the hint when Qwen
                # runs anyway, but is not worth waiting for Qwen.
                if not self.casing_is_enforced(word) and word != word.lower():
                    needs_llm = True

        hint = self._build_hint(lang, pending + respell, keep)
        return text, hint, applied, needs_llm

    @staticmethod
    def _build_hint(lang, pairs, keep):
        if not pairs and not keep:
            return ""
        if lang == "da":
            label, spelled, keep_label, tail = "Personlig ordbog", "staves", "behold", "Brug præcis disse stavemåder."
        elif lang == "de":
            label, spelled, keep_label, tail = "Persönliches Wörterbuch", "wird geschrieben", "behalte", "Verwende genau diese Schreibweisen."
        elif lang == "fr":
            label, spelled, keep_label, tail = "Dictionnaire personnel", "s'écrit", "garde", "Utilise exactement ces orthographes."
        else:
            label, spelled, keep_label, tail = "Personal dictionary", "is spelled", "keep", "Use exactly these spellings."
        parts = [f"'{a}' {spelled} '{b}'" for a, b in pairs[:12]]
        if keep:
            parts.append(f"{keep_label} " + ", ".join(f"'{w}'" for w in keep[:12]))
        return f"\n{label}: " + "; ".join(parts) + f". {tail}"

    def apply_casing(self, text):
        """
        Deterministic pass after the LLM: enforce the dictionary's exact spelling
        for entries the model reproduced with different casing ('sap' -> 'SAP',
        'øresund bridge' -> 'Øresund Bridge', 'powerbi' -> 'PowerBI').

        Single words whose only capital is the first letter ('Monday', 'Annika')
        are left to the LLM hint: forcing them would also capitalize the
        ordinary word 'monday' mid-sentence. All-lowercase entries leave a
        capital at the start of a sentence alone.
        """
        if not text or not self.entries:
            return text
        with self.lock:
            words = [e["word"] for e in self.entries.values()]
        for w in words:
            if not w:
                continue
            if not self.casing_is_enforced(w):
                continue
            text = _word_pattern(w).sub(
                lambda m, w=w: self._expected_form(w, m.group(0), m.string, m.start()), text)
        return text
