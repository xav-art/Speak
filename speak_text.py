"""
Speak - text rules shared by dictation and meeting mode.

Pure functions only (no Windows, audio or model calls), so every rule here is
unit tested in test_speak.py:

- strip_leading_fillers / needs_llm: the dictation fast path. A short phrase
  that is already clean is inserted without waiting for Qwen.
- has_disfluency: which meeting turns are worth sending to Qwen.
- is_hallucination: text Whisper invents on silence or noise.
"""

import re
import unicodedata

# Sounds that are never a word in English, Danish, German or French, so they
# can be removed without asking the language model. "er" and "eh" are NOT
# here: "er" means "is" in Danish and "he" in German.
HARD_FILLERS = {
    "um", "umm", "uh", "uhh", "uhm", "hmm", "hm", "erm", "ehm",
    "øh", "øhm", "øhh", "äh", "ähm", "ähh", "euh",
}

# Fillers that are not a word in that language only.
LANGUAGE_FILLERS = {"en": {"er", "eh"}}

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
    "bullet point", "new line", "next line", "new paragraph", "next paragraph", "full stop", "period",
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

_HARD_FILLER_RE = "|".join(sorted((re.escape(f) for f in HARD_FILLERS), key=len, reverse=True))
# Fillers at the start of a sentence, with the punctuation after them. The
# filler must end at a separator, so "Uh-huh" and "Øh-" are left alone.
_LEADING_FILLER = re.compile(
    rf"^(?:(?:{_HARD_FILLER_RE})(?=[\s,.…!?;:]|$)[\s,.…!?;:\-–]*)+", re.IGNORECASE)


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


def strip_leading_fillers(text):
    """Removes hard fillers at the start of the text or of a sentence:
    "Um, I think so." -> "I think so.", "Fine. Øh, I'll go." -> "Fine. I'll go."
    Fillers in the middle of a sentence are left for the LLM: removing them
    can leave a stray comma ("we should, uh, move" -> "we should, move")."""
    # Split after sentence punctuation, keeping the whitespace separators.
    parts = re.split(r"(?<=[.!?])(\s+)", text.strip())
    out = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        cleaned = _LEADING_FILLER.sub("", sentence)
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
    fillers = HARD_FILLERS | LANGUAGE_FILLERS.get(lang, set())
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
    """Whitespace, a capital first letter and closing punctuation."""
    text = " ".join(text.split())
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?:":
        text += "."
    return text


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
