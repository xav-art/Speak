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
        self.entries = {}      # fold(word)  -> {"word", "source", "uses", "added"}
        self.corrections = {}  # fold(heard) -> {"heard", "word", "count", "contexts": [...]}
        self.load()

    # ---- persistence -------------------------------------------------------

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        for e in data.get("words", []):
            if isinstance(e, str):
                e = {"word": e}
            w = str(e.get("word", "")).strip()
            if w:
                self.entries[_fold(w)] = {
                    "word": w,
                    "source": e.get("source", "manual"),
                    "uses": int(e.get("uses", 0)),
                    "added": e.get("added", time.strftime("%Y-%m-%d")),
                }
        for c in data.get("corrections", []):
            heard = str(c.get("heard", "")).strip()
            word = str(c.get("word", "")).strip()
            if heard and word and _fold(heard) != _fold(word):
                self.corrections[_fold(heard)] = {
                    "heard": heard,
                    "word": word,
                    "count": int(c.get("count", 1)),
                    "contexts": [str(x) for x in c.get("contexts", []) if x],
                }

    def save(self):
        with self.lock:
            words = sorted(self.entries.values(), key=lambda e: _fold(e["word"]))
            corrections = sorted(self.corrections.values(), key=lambda c: _fold(c["heard"]))
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"words": words, "corrections": corrections}, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
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

    def prepare(self, raw_text, lang):
        """
        Run before the LLM. Returns (text, hint, applied):
        - text   : raw transcript with confirmed mishearings already replaced
        - hint   : one line for the cleanup prompt, or "" when nothing is relevant
        - applied: list of (heard, word) replaced deterministically
        """
        if not raw_text:
            return raw_text, "", []
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

        # Dictionary words present in the text: exact form -> "keep", other casing -> "is spelled"
        keep, respell = [], []
        for e in entries:
            m = _word_pattern(e["word"]).search(text)
            if not m:
                continue
            if m.group(0) == e["word"]:
                keep.append(e["word"])
            else:
                respell.append((m.group(0), e["word"]))

        hint = self._build_hint(lang, pending + respell, keep)
        return text, hint, applied

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
        ordinary word 'monday' mid-sentence.
        """
        if not text or not self.entries:
            return text
        with self.lock:
            words = [e["word"] for e in self.entries.values()]
        for w in words:
            if not w:
                continue
            if " " not in w and w[1:] == w[1:].lower():
                continue
            text = _word_pattern(w).sub(w, text)
        return text
