"""
Speak - text rules shared by dictation and meeting mode.

Pure functions only (no Windows, audio or model calls), so every rule here is
unit tested in test_speak.py:

- strip_leading_fillers / needs_llm: the dictation fast path. A short phrase
  that is already clean is inserted without waiting for Qwen.
- has_disfluency: which meeting turns are worth sending to Qwen.
- prepare_for_llm / cleanup_is_faithful: what code does before Qwen (spoken
  commands, filler sounds) and the check that Qwen only cleaned up.
- is_hallucination: text Whisper invents on silence or noise.
"""

from collections import Counter

import re
import unicodedata

# Sounds that are never a word in English, Danish, German or French, so they
# can be removed without asking the language model. "er" and "eh" are NOT
# here: "er" means "is" in Danish and "he" in German. Nor is "um": it is a
# German word ("um zehn Uhr").
HARD_FILLERS = {
    "umm", "uh", "uhh", "uhm", "hmm", "hm", "erm", "ehm",
    "øh", "øhm", "øhh", "äh", "ähm", "ähh", "euh",
}

# Fillers that are not a word in that language only.
LANGUAGE_FILLERS = {"en": {"um", "er", "eh"}, "da": {"um"}, "fr": {"um"}, "de": set()}
# Whisper writes a German "äh" as "E," at times. Only that form, at the start
# of a sentence, is a filler: "Vitamin E" is a word.
_LEADING_ONLY = {"de": r"e(?=[ \t]*,)"}


def fillers_for(lang):
    """The filler sounds of a language. Unknown language: those that are no
    word in any of the four ("um" included, as before languages were told apart)."""
    if lang in LANGUAGE_FILLERS:
        return HARD_FILLERS | LANGUAGE_FILLERS[lang]
    return HARD_FILLERS | {"um"}

# Words that are fillers only when spoken as one ("it was, like, huge") but
# ordinary words otherwise ("I like that"). They send a phrase to the LLM
# only when set off by a comma, which is how Whisper writes them as fillers.
SOFT_FILLERS = {
    "en": {"like", "basically", "actually", "literally"},
    "da": {"altså", "sådan", "ikk", "nå"},
    "de": {"also", "quasi", "sozusagen", "halt", "eigentlich", "naja"},
    "fr": {"ben", "bah", "genre", "quoi", "bon"},
}

# Phrases that are fillers wherever they appear.
FILLER_PHRASES = {
    "en": {"you know", "i mean", "sort of", "kind of"},
    "da": {"du ved", "ikke også", "hvad hedder det", "hvad hedder han", "hvad hedder hun"},
    "de": {"weißt du", "sag ich mal", "na ja"},
    "fr": {"en fait", "du coup", "tu sais", "tu vois", "je veux dire"},
}

# Spoken formatting and correction commands the LLM must act on. Checked in
# every language: people mix ("ny linje" in an English email).
COMMAND_PHRASES = {
    # English
    "bullet point", "new line", "new paragraph", "next paragraph", "full stop", "period",
    "comma", "question mark", "exclamation mark", "exclamation point", "scratch that", "delete that",
    "no wait",
    # Danish
    "punkt", "linje", "ny linje", "afsnit", "nyt afsnit", "linjeskift", "punktum", "komma",
    "spørgsmålstegn", "udråbstegn", "slet det", "nej vent",
    # German
    "aufzählungspunkt", "zeile", "neue zeile", "absatz", "neuer absatz", "fragezeichen",
    "ausrufezeichen", "streich das", "nein warte",
    # French
    "puce", "point", "ligne", "à la ligne", "nouvelle ligne", "nouveau paragraphe", "virgule",
    "point d'interrogation", "point d'exclamation", "non attends",
}
# A missed command is pasted as a literal word into the user's document; a
# false alarm ("trial period", "on that point") only costs the LLM's time.

FAST_PATH_MAX_WORDS = 8

def _filler_alternation(lang):
    return "|".join(sorted((re.escape(f) for f in fillers_for(lang)), key=len, reverse=True))


_LEADING_FILLER, _INNER_FILLER = {}, {}


def _leading_filler_re(lang):
    """Fillers at the start of a sentence, with the punctuation after them. The
    filler must end at a separator, so "Uh-huh" and "Øh-" are left alone."""
    if lang not in _LEADING_FILLER:
        alternation = _filler_alternation(lang) + (f"|{_LEADING_ONLY[lang]}" if lang in _LEADING_ONLY else "")
        _LEADING_FILLER[lang] = re.compile(
            rf"^(?:(?:{alternation})(?=[\s,.…!?;:]|$)[\s,.…!?;:\-–]*)+", re.IGNORECASE)
    return _LEADING_FILLER[lang]


def _inner_filler_re(lang):
    """A filler anywhere, with the commas Whisper puts around it."""
    if lang not in _INNER_FILLER:
        _INNER_FILLER[lang] = re.compile(
            rf"(?:,[ \t]*)?(?<![\w\-])(?:{_filler_alternation(lang)})(?![\w\-])(?:[ \t]*,)?", re.IGNORECASE)
    return _INNER_FILLER[lang]


def words_of(text):
    """Lowercase words without surrounding punctuation."""
    out = []
    for w in unicodedata.normalize("NFC", text).lower().split():
        w = w.strip(".,;:!?\"'()[]…-–")
        if w:
            out.append(w)
    return out


def _contains_phrase(words, phrases):
    joined = f" {' '.join(words)} "
    return any(f" {p} " in joined for p in phrases)


def strip_leading_fillers(text, lang=None):
    """Removes hard fillers at the start of the text or of a sentence:
    "Um, I think so." -> "I think so.", "Fine. Øh, I'll go." -> "Fine. I'll go."
    Fillers in the middle of a sentence are left for the LLM: removing them
    can leave a stray comma ("we should, uh, move" -> "we should, move")."""
    # Split after sentence punctuation, keeping the whitespace separators.
    parts = re.split(r"(?<=[.!?])(\s+)", text.strip())
    out = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        cleaned = _leading_filler_re(lang).sub("", sentence)
        # Capitalize what now starts a sentence, or the text ("Um... yes."),
        # but never a word with its own casing ("iPhone", "eBay").
        if cleaned and (cleaned != sentence or (i > 0 and not out)) and not _has_inner_capital(cleaned):
            cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned:
            if out:
                out.append(parts[i - 1])
            out.append(cleaned)
    return "".join(out)


def _has_inner_capital(text):
    first = text.split(maxsplit=1)[0] if text.split() else ""
    return any(ch.isupper() for ch in first[1:])


def has_soft_filler(text, lang):
    """A filler phrase ("you know", "du ved") or a comma-delimited soft
    filler ("it was, like, huge") in the given language."""
    if _contains_phrase(words_of(text), FILLER_PHRASES.get(lang, ())):
        return True
    soft = SOFT_FILLERS.get(lang, set())
    if not soft:
        return False
    # "Like, it was" / "it was, like, huge" / "it was huge, like." (comma or
    # sentence start right before, or a comma right after).
    pattern = r"(?:^|[,.!?]\s*)(?:{0})(?=[\s,.!?]|$)|(?:^|\s)(?:{0}),".format(
        "|".join(re.escape(w) for w in soft))
    return re.search(pattern, unicodedata.normalize("NFC", text).lower()) is not None


def has_disfluency(text, lang=None):
    """A sound that is never a word ("um", "øh", English "er") or a stutter
    ("the the"). Meeting mode sends only these turns to Qwen: on a clean
    turn the model can only reword it."""
    words = words_of(text)
    fillers = fillers_for(lang)
    if any(w in fillers for w in words):
        return True
    return any(a == b for a, b in zip(words, words[1:]))


def needs_llm(text, lang=None, max_words=FAST_PATH_MAX_WORDS):
    """Whether a dictation needs Qwen. Call it on text after
    strip_leading_fillers. Long text, numbers (dates and amounts need
    formatting), spoken commands and remaining fillers need it."""
    words = words_of(text)
    if len(words) > max_words:
        return True
    if any(ch.isdigit() for ch in text):
        return True
    if _contains_phrase(words, COMMAND_PHRASES):
        return True
    return has_disfluency(text, lang) or has_soft_filler(text, lang)


def light_cleanup(text):
    """Whitespace, a capital first letter and closing punctuation. Line
    breaks are kept: they come from spoken commands. A text ending in a comma
    is part of a longer one (a chunk of a long dictation) and gets no stop."""
    lines = [" ".join(line.split()) for line in text.split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip("\n"))
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?:,;":
        text += "."
    return text


def keep_question_mark(before, after):
    """after, with the question mark back if the transcript ended in one and
    the cleanup ended in a full stop instead. Whisper's "?" comes from the
    intonation; Qwen3 1.7B turned it into "." in a fifth of the questions."""
    tail = after.rstrip()
    if before.rstrip().endswith("?") and tail.endswith(".") and not tail.endswith(".."):   # "..." trails off
        after = after.rstrip()[:-1] + "?"
    return after


# ---- Before and after the LLM ------------------------------------------------
#
# Code does what it can do reliably, so Qwen 1.5B does not have to: in testing
# Qwen dropped half the spoken line breaks and left "um" in a third of the
# phrases. Code applies only commands that cannot be ordinary words where they
# stand; "punkt" and "puce" ("et vigtigt punkt"), and commands Whisper wrote
# without a pause, stay with Qwen.

NEWLINE_COMMANDS = ("new line", "ny linje", "neue zeile", "à la ligne", "nouvelle ligne")
PARAGRAPH_COMMANDS = ("new paragraph", "next paragraph", "nyt afsnit", "neuer absatz", "nouveau paragraphe")
BULLET_COMMANDS = ("bullet point", "aufzählungspunkt")
# A sentence of its own that deletes the sentence before it.
SCRATCH_COMMANDS = ("scratch that", "delete that", "slet det", "streich das", "efface ça")
# Cues that the speaker takes back the words just before them.
CORRECTION_CUES = ("no wait", "sorry", "i mean", "nej vent", "undskyld", "jeg mener", "nein warte",
                   "ich meine", "non attends", "pardon", "je veux dire") + SCRATCH_COMMANDS


def _phrase_alternation(phrases):
    return "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in sorted(phrases, key=len, reverse=True))


# A command is dictated after a pause, so Whisper writes punctuation (or a
# line break) before it: "Hi Anna, new line thanks". "Our new line of
# candles" and "the new line, which" are product lines, not commands.
_AFTER_PAUSE = r"(?:^|(?<=[,.;:!?\n]))[ \t]*"
_NOT_A_NOUN = r"(?![ \t]+(?:of|af|von|for|til|für|de|du|des|der)\b)"


def _command_re(phrases):
    return re.compile(rf"{_AFTER_PAUSE}(?<!\w)(?:{_phrase_alternation(phrases)})(?!\w){_NOT_A_NOUN}[ \t,.;:]*",
                      re.IGNORECASE)


_PARAGRAPH_CMD = _command_re(PARAGRAPH_COMMANDS)
_NEWLINE_CMD = _command_re(NEWLINE_COMMANDS)
_BULLET_CMD = _command_re(BULLET_COMMANDS)
_SENTENCE_GAP = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’»)]))[ \t]+|[ \t]*(?:\n[ \t]*)+")
# Abbreviations whose full stop does not end a sentence ("Mr. Smith", "z.B. Kassen").
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "eg", "ie", "etc", "no", "hr", "fru", "frk", "nr",
                  "ca", "bla", "evt", "kl", "fx", "jf", "dvs", "osv", "inkl", "ekskl", "tlf", "str", "zb", "bzw",
                  "usw", "ggf", "inkl", "m", "mme", "mlle", "env", "cf"}


_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december", "januar", "februar", "märz", "mai", "juni", "juli", "oktober", "dezember",
           "marts", "maj", "januari", "janvier", "février", "mars", "avril", "juin", "juillet", "août",
           "septembre", "octobre", "novembre", "décembre"}


def _starts_sentence(rest):
    """Whether text can start a sentence: a capital, a digit, a quote or a
    bracket, or a word with its own casing ("iPhone works")."""
    if not rest:
        return False
    if rest[0].isupper() or rest[0].isdigit() or rest[0] in "\"'“”‘’„«»(":
        return True
    first = rest.split(maxsplit=1)[0]
    return any(ch.isupper() for ch in first[1:])


def sentences(text, lang=None):
    """(separator, sentence) pairs that join back to the text. A sentence ends
    at a line break, or at . ! ? before the start of a new one. In German a
    number with a full stop is an ordinal ("im 2. Quartal"). Elsewhere a number
    ("den 14. marts") or an abbreviation ("Mr. Smith", "e.g. the") does not
    end one, unless the next sentence is "scratch that"."""
    out, start, sep = [], 0, ""
    for m in _SENTENCE_GAP.finditer(text):
        if "\n" not in m.group(0):
            sentence, rest = text[start:m.start()], text[m.end():]
            words = sentence.split()
            last = words[-1] if words else ""
            nxt = rest.split(maxsplit=1)[0] if rest.split() else ""
            # "am 14. März" and "den 14. marts" go on; "kl. 10. Hun venter" ends.
            date = re.fullmatch(r"\d+\.", last) and (lang == "de" or nxt[:1].islower() or _norm_word(nxt) in _MONTHS)
            ends = _starts_sentence(rest) and not date and _norm_word(last) not in _ABBREVIATIONS
            if not ends and not _is_scratch(sentence) and not _is_scratch(re.split(r"(?<=[.!?])\s", rest, maxsplit=1)[0]):
                continue
        out.append((sep, text[start:m.start()]))
        sep, start = m.group(0), m.end()
    out.append((sep, text[start:]))
    return out


def _is_scratch(sentence):
    return " ".join(sentence.strip().rstrip(".!?").lower().split()) in SCRATCH_COMMANDS


# "Punkt opdater prislisten." / "Puce envoyer le devis." as a sentence of its
# own is a list item; "Punkt 3", "Punkt to" and "Punkt nummer" are not.
# Only in Danish and French: German "Punkt acht treffen wir uns" is a word.
_NOT_AN_ITEM = {
    "da": "en|et|to|tre|fire|fem|seks|syv|otte|ni|ti|elleve|tolv|nummer|for|og",
    "fr": "un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|numéro|par|électronique|pour|et",
}
_LIST_ITEM = {lang: re.compile(rf"^{word}[ \t,:]+(?!(?:\d|(?:{_NOT_AN_ITEM[lang]})(?!\w)))(?=\w)", re.IGNORECASE)
              for lang, word in (("da", "punkt"), ("fr", "puce"))}


def _capitalize_lines(text):
    """A capital at the start of the text, of each line and of each list item,
    but never in a word with its own casing ("iPhone", "eBay")."""
    def cap(m):
        word = m.group(3)
        if any(ch.isupper() for ch in word[1:]):
            return m.group(0)
        return m.group(1) + (m.group(2) or "") + word[0].upper() + word[1:]
    return re.sub(r"(^|\n)(- )?(\w+)", cap, text)


def _more_breaks(own, dropped):
    """The separator of a sentence after deleted ones: its own when the user
    dictated a line break there, else the larger break of the deleted ones."""
    return own if "\n" in own else max((own, dropped), key=lambda s: s.count("\n"))


def apply_spoken_commands(text, lang=None):
    """"new line", "new paragraph" and "bullet point" (and their Danish,
    German and French forms) become line breaks and "- " list items; then a
    Danish sentence "Punkt ..." or a French "Puce ..." becomes a list item,
    and "scratch that" deletes the sentence before it."""
    item = _LIST_ITEM.get(lang)
    for command, replacement in ((_PARAGRAPH_CMD, "\n\n"), (_NEWLINE_CMD, "\n"), (_BULLET_CMD, "\n- ")):
        # Again until nothing changes: "new line, new line" is a blank line,
        # and the first match takes the comma the second one needs before it.
        previous = None
        while previous != text:
            previous, text = text, command.sub(replacement, text)
    kept, dropped_sep = [], ""
    for sep, sentence in sentences(text, lang):
        if _is_scratch(sentence):
            if kept:
                # "One. Two. Scratch that. Scratch that." deletes both; a line
                # break before the deleted sentence stays for the next one.
                dropped_sep = max((dropped_sep, kept.pop()[0], sep), key=lambda s: s.count("\n"))
            continue
        if item and item.match(sentence):
            sentence, sep = "- " + item.sub("", sentence), ("\n" if kept else "")
        kept.append((_more_breaks(sep, dropped_sep) if kept else "", sentence))
        dropped_sep = ""
    text = "".join(sep + s for sep, s in kept)
    return re.sub(r"\n\n(?=- )", "\n", text)   # "new line, bullet point": no empty line


# Punctuation commands refer to what came before too. A false alarm costs one
# more cleanup of the two pieces after the stop, so the short words that
# start ordinary sentences as well ("No problem", "Wait for me", "Point
# taken") count only when Whisper wrote a pause after them ("No, Wednesday").
_CONTINUING = tuple(sorted({*NEWLINE_COMMANDS, *PARAGRAPH_COMMANDS, *BULLET_COMMANDS, *SCRATCH_COMMANDS,
                            *CORRECTION_CUES, "punkt", "puce", "full stop", "question mark", "punktum", "komma",
                            "spørgsmålstegn", "virgule"}, key=len, reverse=True))
_CONTINUING_WITH_PAUSE = re.compile(r"^(?:no|wait|actually|period|comma|point|nej|vent|nein|warte|non|attends)[,.;:!?]",
                                    re.IGNORECASE)


def continues_previous(previous, text, lang=None):
    """Whether `text`, the next piece of a dictation cut at a pause, must be
    cleaned together with `previous`: the previous piece did not end its
    sentence, or this one starts with a command or a correction that refers
    to what came before ("No wait, Wednesday", "Scratch that", "New line")."""
    end = previous.rstrip() if previous else ""
    if end and (end[-1] not in ".!?" or end.endswith("...")):   # "...": trailing off mid-sentence
        return True
    start = strip_leading_fillers(text, lang).lstrip(" \"'“„«(")
    if _CONTINUING_WITH_PAUSE.match(start):
        return True
    words = " ".join(words_of(start))
    return any(words == cue or words.startswith(cue + " ") for cue in _CONTINUING)


def remove_fillers(text, lang=None):
    """Filler sounds anywhere, with the commas around them: "Can you, uh,
    check it" -> "Can you check it". The punctuation is left to the LLM."""
    return _inner_filler_re(lang).sub(" ", strip_leading_fillers(text, lang))


def tidy(text):
    """Spaces left by removed words, empty lines, a capital on each line."""
    text = re.sub(r"[ \t]+([,.])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [re.sub(r"^[,.;:]+[ \t]*", "", line.strip()) for line in text.split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return _capitalize_lines(text)


def prepare_for_llm(text, lang=None, commands=True):
    """The text Qwen gets: spoken commands applied (dictation only: "new line"
    said in a meeting is just words) and filler sounds removed."""
    if commands:
        text = apply_spoken_commands(text, lang)
    return tidy(remove_fillers(text, lang))


# Words a cleanup may drop: fillers that are also ordinary words ("like",
# "altså") and filler phrases. "So", "okay" and "ja" are not here: they carry
# meaning often enough ("..., so I will ask") that dropping them is rewording.
_DROPPABLE = {"like", "basically", "actually", "literally", "altså", "sådan", "ikk", "nå", "also", "quasi",
              "sozusagen", "halt", "eigentlich", "naja", "ben", "bah", "genre", "quoi", "ah", "eh", "oh"}
_DROPPABLE_PHRASES = set().union(*FILLER_PHRASES.values())
# A meeting turn may also lose these at the start of a sentence ("Yeah, I
# will do it" -> "I will do it"), as before Phase 2.
_DISCOURSE_WORDS = {"so", "okay", "ok", "yeah", "well", "right", "ja", "jo", "nå", "nu", "altså", "bon",
                    "voilà", "alors", "na", "naja", "gut"}
# Phrases that become a line break or "- " when the LLM does the command.
_LINE_PHRASES = ("punkt", "puce") + NEWLINE_COMMANDS + PARAGRAPH_COMMANDS + BULLET_COMMANDS
# The LLM may never add these: "Send it." -> "Don't send it." is one word.
_NEGATIONS = {"not", "no", "never", "dont", "doesnt", "didnt", "cant", "cannot", "wont", "isnt", "arent",
              "wasnt", "shouldnt", "couldnt", "wouldnt", "nothing", "nobody", "none", "neither", "hardly",
              "ikke", "aldrig", "nej", "intet", "ingen", "ingenting", "nicht", "kein", "keine", "keinen",
              "keiner", "nie", "niemals", "nein", "nichts", "niemand", "kaum", "pas", "ne", "jamais", "non",
              "rien", "personne", "aucun", "aucune"}


def _norm_word(word):
    return "".join(ch for ch in unicodedata.normalize("NFC", word).lower() if ch.isalnum())


def _norm_words(text):
    return [w for w in map(_norm_word, text.split()) if w]


def _content_lines(text):
    return [line for line in text.split("\n") if line.strip()]


def _phrase_count(words, phrase):
    joined = f" {' '.join(words)} "
    return joined.count(f" {' '.join(_norm_words(phrase))} ")


_CUE_WORDS = {w for cue in CORRECTION_CUES for w in cue.split()}


def _taken_back(before, after_words, lang=None):
    """The words a speaker took back: for each correction cue the LLM removed,
    the cue and the words right before it that are gone from the output ("on
    Tuesday, no wait, Wednesday": "Tuesday"). That is the end of the text
    before the cue, in its sentence or, when the cue starts one ("Monday. No
    wait, Tuesday."), in the sentence before. Nothing without a correction
    after the cue ("Sorry for the delay", "I'm sorry."), and nothing when no
    word before the cue is left ("Ask Sarah, sorry, Peter first" -> "Peter
    first": that is the LLM dropping the sentence)."""
    taken = Counter()
    present = Counter(after_words)
    after_joined = f" {' '.join(after_words)} "
    parts = sentences(before, lang)
    for i, (_, sentence) in enumerate(parts):
        for cue in CORRECTION_CUES:
            if after_joined.count(f" {cue} ") >= f" {' '.join(_norm_words(sentence))} ".count(f" {cue} "):
                continue   # the cue is still there: nothing was taken back
            pattern = r"(?<!\w)" + r"[\s,]+".join(map(re.escape, cue.split())) + r"(?!\w)"
            for m in re.finditer(pattern, sentence, re.IGNORECASE):
                after_cue = _norm_words(re.split(r"[.;:!?]", sentence[m.end():], maxsplit=1)[0])
                if not after_cue and i + 1 < len(parts):   # "Monday. No, wait. Tuesday."
                    after_cue = _norm_words(re.split(r"[,.;:!?]", parts[i + 1][1], maxsplit=1)[0])
                extra_cues = []   # "fifty, no wait, no, sixty"
                while after_cue and after_cue[0] in _CUE_WORDS:
                    extra_cues.append(after_cue.pop(0))
                if not after_cue:
                    continue   # no correction follows
                head = _norm_words(sentence[:m.start()])
                if not head and i:
                    head = _norm_words(parts[i - 1][1])
                if all(not present[w] for w in head) and len(head) > len(after_cue) + 1:
                    continue   # nothing before the cue is left: the LLM dropped the sentence
                # The end of the text before the cue, up to the first word
                # that is still there ("to Peter, Paul and Anna, sorry, Lars")...
                gone = []
                for w in reversed(head):
                    if present[w]:
                        break
                    gone.append(w)
                # ...or words gone from just before it, about as many as the
                # correction, at most four ("Ruf Anna an. Nein warte, ruf Lars an.").
                window = head[-min(len(after_cue) + 2, 4):]
                gone += [w for w in window[:max(0, len(window) - len(gone))] if not present[w]]
                # A correction never takes back a "not": "I will not come on
                # Monday, sorry, on Tuesday" -> "I will come on Tuesday" reverses it.
                taken.update([w for w in gone if w not in _NEGATIONS] + cue.split() + extra_cues)
    return taken


def cleanup_is_faithful(before, after, lang=None, replacements=(), meeting=False):
    """True when the LLM output `after` only cleaned up `before`: fillers,
    stutters and self-corrections removed, punctuation and casing fixed, line
    breaks kept. Dropping any other word, adding a negation or more than about
    one word in ten, or translating is rewording: in testing Qwen dropped "I
    think" and "Please" from dictations and turned "Jeg giver butikkerne
    besked" into "Du giver besked". replacements: (heard, word) pairs the
    dictionary asked the LLM to make. meeting: a turn may also lose a
    discourse word at the start of a sentence ("Yeah, ...")."""
    lines_added = len(_content_lines(after)) - len(_content_lines(before))
    if lines_added < 0:
        return False   # a spoken line break or list item was merged away
    a, b = _norm_words(before), _norm_words(after)
    if not a or not b:
        return not a and not b
    removed, added = Counter(a) - Counter(b), Counter(b) - Counter(a)

    def credit(gone, new):
        if gone and new and all(removed[w] for w in gone) and all(added[w] for w in new):
            for w in gone:
                removed[w] -= 1
            for w in new:
                added[w] -= 1

    for heard, word in replacements:
        credit(_norm_words(heard), _norm_words(word))
    # A compound written as one word ("fertig stellen" -> "fertigstellen").
    for x, y in zip(a, a[1:]):
        if added[x + y]:
            credit([x, y], [x + y])
    removed, added = +removed, +added
    if any(w in _NEGATIONS for w in added):
        return False

    droppable = Counter(w for w in a if w in _DROPPABLE or w in fillers_for(lang))
    for phrase in _DROPPABLE_PHRASES:
        n = _phrase_count(a, phrase)
        for w in phrase.split():
            droppable[w] += n
    if meeting:
        sounds = fillers_for(lang)
        for _, sentence in sentences(before, lang):
            first = [w for w in _norm_words(sentence) if w not in sounds][:1]   # "Um, so I will"
            if first and first[0] in _DISCOURSE_WORDS:
                droppable[first[0]] += 1
    if lines_added:
        # The LLM did a spoken command ("Punkt priser", or a "new line"
        # Whisper wrote without a pause): the phrase became the line break.
        budget = lines_added
        # Longest first, and only phrases the output no longer has: a noun
        # "punkt" that stays must not use up the credit of a "new line".
        for phrase in sorted(_LINE_PHRASES, key=len, reverse=True):
            n = min(_phrase_count(a, phrase) - _phrase_count(b, phrase), budget)
            if n <= 0:
                continue
            budget -= n
            for w in _norm_words(phrase):
                droppable[w] += n
    droppable.update(_taken_back(before, b, lang))
    # A word that is still in the output was only said twice (a stutter).
    content_removed = sum(n for w, n in (removed - droppable).items() if w not in b)
    return content_removed == 0 and sum(added.values()) <= max(1, len(a) // 10)


# Text Whisper learned from subtitled video and "hears" in silence or
# breathing. Dropping a real dictation is far worse than letting one of these
# through, so the rules are strict:
#  - _INVENTED_ALONE: a segment that is exactly one of these sign-offs is
#    dropped. Nobody dictates "Thanks for watching!" as a whole sentence.
#  - _CREDIT_LIKE: credit lines and names ("Undertekster af ...", "Nordisk
#    Film", "Amara.org") are dropped only when Whisper itself doubts there was
#    speech: "Vi mødes hos Nordisk Film i morgen" is a real sentence.
#  - _AMBIGUOUS_SIGNOFFS: everyday words ("Thank you.") likewise.
_INVENTED_ALONE = re.compile(
    r"^(?:(?:thanks|thank you)(?: (?:so|very) much)? for watching(?: (?:this|the) video)?"
    r"(?: and)?(?: i'll)?(?: see you (?:in the )?next (?:time|video))?"
    r"|(?:please )?(?:like and )?subscribe(?: to (?:my|our|the) channel)?"
    r"|tak fordi (?:du|i) (?:så|kiggede|lyttede) med"
    r"|tak for at (?:se|lytte) med"
    r"|(?:vielen )?dank (?:fürs|für das|für's|furs) zu(?:schauen|sehen|hören)(?: und bis zum nächsten mal)?"
    r"|merci d'avoir regardé(?: cette vidéo)?(?: à bientôt)?"
    r"|abonnez vous(?: à la chaîne)?)"
    r"(?: bye(?: bye)?| hej hej| tschüss| au revoir)?$")
_CREDIT_LIKE = re.compile(
    r"^(?:undertekst\w*|(?:danske )?tekst(?:er|et|ning)? (?:af|by|fra)|untertitel\w*|sous titr\w*"
    r"|(?:subtitles|captions|transcription|transcribed) (?:by|from))"
    r"|nordisk film|amara ?org|im auftrag des zdf")
_AMBIGUOUS_SIGNOFFS = {
    "thank you", "thanks", "thank you very much", "bye", "you", "tak", "tak skal du have",
    "danke", "danke schön", "tschüss", "merci", "au revoir",
}
AMBIGUOUS_NO_SPEECH = 0.4


def _norm_segment(text):
    text = unicodedata.normalize("NFC", text).lower().replace("-", " ").replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def is_hallucination(text, no_speech_prob=0.0):
    """True when a Whisper segment is text it invented rather than heard.
    no_speech_prob is Whisper's own estimate that the segment has no speech."""
    norm = _norm_segment(text)
    if not norm:
        return True
    if _INVENTED_ALONE.match(norm):
        return True
    if no_speech_prob < AMBIGUOUS_NO_SPEECH or len(norm.split()) > 12:
        return False
    return bool(_CREDIT_LIKE.search(norm)) or norm in _AMBIGUOUS_SIGNOFFS
