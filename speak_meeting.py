r"""
Speak - Meeting mode: record a meeting, then write a transcript that shows who said what.

Everything runs offline, after the recording stops:
1. Recording. The microphone (you, plus anyone in the room) and the system audio
   (everyone in a Teams, Zoom or Meet call, captured with WASAPI loopback) are
   written to two 16 kHz WAV files.
2. Diarization. sherpa-onnx (pyannote segmentation + NeMo TitaNet speaker
   embeddings) splits each stream into speakers.
3. Labelling. Mic speakers that match the enrolled voice profile become "Me".
   Other mic speech that overlaps call audio is the call leaking into the mic
   (no headset) and is dropped, so remote people are not transcribed twice.
4. Transcription. Each speaker's language is detected from a few of their
   turns, then Whisper transcribes every stretch of same-language speech in one
   pass with word timestamps, and each word goes to whoever was speaking.
   (One Whisper call per turn was 4x slower: every call pays for a full 30 s
   encoder window, even for a 3 s turn.)
5. Cleanup and summary. Qwen removes fillers per turn and writes a summary,
   decisions and action items.
"""

import bisect
import datetime
import gc
import json
import os
import queue
import re
import shutil
import tarfile
import threading
import time
import warnings
import wave

import numpy as np

import speak_text

SAMPLE_RATE = 16000
SHERPA_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# Benchmarked on a 98 s test meeting with 4 voices in English and Danish (Core
# Ultra 7 265U). TitaNet-small was the only embedding model that found exactly 4
# speakers with every turn right: ResNet34 split one voice in two and both CAM++
# models fragmented badly. It scores the enrolled voice 0.85 against <= 0.35 for
# everyone else, which leaves a wide margin for the "Me" threshold.
SEGMENTATION_MODEL = {
    "url": f"{SHERPA_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "sha256": "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    "member": "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx",
    "file": "pyannote-segmentation-3-0.int8.onnx",
}
EMBEDDING_MODEL = {
    # The upstream release tag really is spelled "recongition".
    "url": f"{SHERPA_RELEASES}/speaker-recongition-models/nemo_en_titanet_small.onnx",
    "sha256": "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
    "file": "nemo_en_titanet_small.onnx",
}

LANGUAGE_NAMES = {"en": "English", "da": "Danish", "de": "German", "fr": "French"}

# Meeting turns are cleaned far more conservatively than dictation: with the
# dictation prompt, Qwen 1.5B rewrote "Jeg giver butikkerne besked" into
# "Butikkerne har beskrevet" (wrong meaning) in testing. Written in each
# language so the model does not drift into English.
MEETING_CLEANUP_PROMPTS = {
    "en": ("Clean up this spoken meeting turn. Remove filler words (um, uh, you know, like) and accidental "
           "repetitions. Fix punctuation and capitalization. Do NOT rephrase, shorten, summarize or translate: "
           "keep every other word exactly as spoken. Output only the cleaned text."),
    "da": ("Ryd op i denne talte replik fra et møde. Fjern fyldord (øh, altså, ikke også) og utilsigtede "
           "gentagelser. Ret tegnsætning og store bogstaver. Du må IKKE omformulere, forkorte, opsummere eller "
           "oversætte: behold alle andre ord præcis som de blev sagt. Udskriv kun den rensede tekst."),
    "de": ("Bereinige diesen gesprochenen Redebeitrag aus einem Meeting. Entferne Füllwörter (äh, ähm, also, "
           "quasi) und versehentliche Wiederholungen. Korrigiere Zeichensetzung und Groß-/Kleinschreibung. "
           "Formuliere NICHT um, kürze nicht, fasse nicht zusammen und übersetze nicht: Behalte alle anderen "
           "Wörter genau so, wie sie gesagt wurden. Gib nur den bereinigten Text aus."),
    "fr": ("Nettoie cette intervention orale d'une réunion. Supprime les mots de remplissage (euh, ben, genre, "
           "en fait) et les répétitions involontaires. Corrige la ponctuation et les majuscules. Ne reformule PAS, "
           "ne raccourcis pas, ne résume pas et ne traduis pas : garde tous les autres mots exactement tels qu'ils "
           "ont été dits. Renvoie uniquement le texte nettoyé."),
}

ECHO_CORRELATION = 0.3          # above this, the call is audible in the mic (no headset)
ENROLL_SECONDS = 30
ENROLL_MIN_SPEECH = 12          # seconds of actual speech needed for a reliable profile
MAX_PAUSE_IN_TURN = 5.0         # a longer pause starts a new paragraph for the same speaker
SUMMARY_CTX = 8192              # Qwen 2.5 supports 32k; 8k keeps RAM use near 250 MB extra
SUMMARY_CHUNK_TOKENS = 4500

ENROLL_TEXT = (
    "Read this aloud at a normal pace, or just talk about your week in any language:\n\n"
    "  \"I am recording a short voice sample so Speak can recognise me in meetings.\n"
    "   In a typical week I plan work with my team, follow up on open tasks, and\n"
    "   join calls with colleagues and partners. I try to keep meetings short,\n"
    "   agree on clear next steps, and write down who does what and by when.\n"
    "   The quick brown fox jumps over the lazy dog, and the weather today is fine.\""
)


# ---- Pure helpers (unit tested) --------------------------------------------

def format_timestamp(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denom if denom else 0.0


def match_me(cluster_embeddings, profile, threshold):
    """Clusters whose voice matches the enrolled profile. Every cluster above the
    threshold counts: diarization sometimes splits one voice into two clusters."""
    if profile is None:
        return set()
    return {c for c, emb in cluster_embeddings.items() if cosine(emb, profile) >= threshold}


def drop_call_echo(mic_segments, system_segments, me_clusters, min_ratio=0.5):
    """Without a headset the call plays through the speakers and the mic hears it
    again. Mic speech from anyone but "Me" that mostly overlaps call speech is
    that echo. Returns (kept segments, number dropped)."""
    kept, dropped = [], 0
    for seg in mic_segments:
        start, end, cluster = seg[0], seg[1], seg[2]
        if cluster not in me_clusters and end > start:
            covered = sum(overlap(start, end, s[0], s[1]) for s in system_segments)
            if min(covered, end - start) >= min_ratio * (end - start):
                dropped += 1
                continue
        kept.append(seg)
    return kept, dropped


def label_speakers(keys_in_time_order, me_keys):
    """'Me' for the enrolled voice, then Person 1, 2, ... in order of first appearance."""
    labels, n = {}, 0
    for key in keys_in_time_order:
        if key in labels:
            continue
        if key in me_keys:
            labels[key] = "Me"
        else:
            n += 1
            labels[key] = f"Person {n}"
    return labels


def language_ranges(segments, pad=0.2):
    """{lang: [(start, end)]}: the time ranges to transcribe in each language,
    padded a little and merged where they touch or overlap. All the English of
    a stream becomes one Whisper call, all the Danish another, however often
    the speakers alternate. (Transcribing each stretch between two language
    switches separately cost a full 30 s Whisper window per stretch.)"""
    by_lang = {}
    for start, end, lang in sorted(segments, key=lambda s: s[0]):
        start, end = max(0.0, start - pad), end + pad
        ranges = by_lang.setdefault(lang, [])
        if ranges and start <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    return {lang: [tuple(r) for r in ranges] for lang, ranges in by_lang.items()}


TRACK_GAP = 0.6     # seconds of silence between glued ranges, so Whisper sees a pause


def build_track(audio, ranges, gap=TRACK_GAP):
    """Glues the given time ranges of audio into one track with short silences
    between them. Returns (track, pieces); pieces map track time back to the
    meeting timeline: [(track_start, meeting_start, length)]."""
    parts, pieces, t = [], [], 0.0
    silence = np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)
    for start, end in ranges:
        chunk = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        if not len(chunk):
            continue
        if parts:
            parts.append(silence)
            t += gap
        parts.append(chunk)
        pieces.append((t, start, len(chunk) / SAMPLE_RATE))
        t += len(chunk) / SAMPLE_RATE
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), pieces


def track_to_meeting_time(t, pieces, is_start=True):
    """Maps a time on a glued track back to the meeting. A word that starts in
    the silence between two pieces belongs to the speech after it (Whisper
    often stamps the first word after a pause early); a word that ends there
    belongs to the speech before it."""
    starts = [p[0] for p in pieces]
    i = max(0, bisect.bisect_right(starts, t) - 1)
    track_start, meeting_start, length = pieces[i]
    offset = t - track_start
    if offset > length and is_start and i + 1 < len(pieces):
        return pieces[i + 1][1]
    return meeting_start + min(max(offset, 0.0), length)


def assign_words(words, segments, max_distance=1.0, lookback=64):
    """Gives each word to the speaker segment it overlaps most (or the nearest one
    within max_distance seconds). Words far from any kept segment, such as call
    echo picked up by the mic, are dropped.

    words:    [(start, end, text, lang)]
    segments: [(start, end, key)]
    returns   [(start, end, text, lang, key)]
    """
    segs = sorted(segments, key=lambda s: s[0])
    starts = [s[0] for s in segs]
    out = []
    for w_start, w_end, text, lang in words:
        hi = bisect.bisect_right(starts, w_end + max_distance)
        best_key, best_ov, best_dist = None, 0.0, max_distance + 1e-9
        for j in range(hi - 1, max(-1, hi - 1 - lookback), -1):
            s_start, s_end, key = segs[j]
            ov = overlap(w_start, w_end, s_start, s_end)
            if ov > best_ov:
                best_key, best_ov = key, ov
            elif best_ov == 0.0:
                dist = max(s_start - w_end, w_start - s_end, 0.0)
                if dist < best_dist:
                    best_key, best_dist = key, dist
        if best_key is not None:
            out.append((w_start, w_end, text, lang, best_key))
    return out


def build_turns(assigned_words, max_pause=MAX_PAUSE_IN_TURN):
    """Consecutive words of one speaker become a turn. Whisper words carry their
    own leading space, so they are joined without a separator."""
    turns = []
    for start, end, text, lang, key in sorted(assigned_words, key=lambda w: w[0]):
        if turns and turns[-1]["speaker"] == key and start - turns[-1]["end"] <= max_pause:
            turns[-1]["end"] = max(turns[-1]["end"], end)
            turns[-1]["parts"].append(text)
        else:
            turns.append({"start": start, "end": end, "speaker": key, "lang": lang, "parts": [text]})
    for t in turns:
        t["text"] = " ".join("".join(t.pop("parts")).split())
    return [t for t in turns if t["text"]]


def fix_split_sentences(turns, max_tail_words=4, max_gap=1.5):
    """At a speaker change, diarization boundaries are a fraction of a second
    off, so the first words of the next speaker can land at the end of the
    previous turn ("... næste uge. One more" / "thing, the newsletter ...").
    Whisper capitalizes sentence starts, so a turn that starts in lowercase is
    a continuation: a short tail after the previous turn's last sentence end
    is moved over to it. So is a dangling word or two with no closing
    punctuation right before the next speaker ("... before lunch. I" /
    "Checked the stock report"), where Whisper started a new sentence."""
    import re
    def stream(speaker):
        return speaker[0] if isinstance(speaker, tuple) else None

    for prev, nxt in zip(turns, turns[1:]):
        gap = nxt["start"] - prev["end"]
        if prev["speaker"] == nxt["speaker"] or gap > max_gap:
            continue
        if stream(prev["speaker"]) != stream(nxt["speaker"]):
            continue    # a word the call audio heard never belongs to someone at the mic

        ends = [m.end() for m in re.finditer(r"[.!?]\s+", prev["text"])]
        cut = ends[-1] if ends else 0
        tail = prev["text"][cut:].strip()
        if not tail or cut == 0:
            continue
        continuation = nxt["text"][:1].islower() and len(tail.split()) <= max_tail_words
        dangling = len(tail.split()) <= 2 and tail[-1] not in ".!?,:;" and gap <= 0.3
        if not (continuation or dangling):
            continue
        rest = nxt["text"]
        if not continuation and rest[:1].isupper() and not rest.split()[0].isupper():
            rest = rest[0].lower() + rest[1:]    # "I" + "Checked the..." -> "I checked the..."
        prev["text"] = prev["text"][:cut].strip()
        nxt["text"] = f"{tail} {rest}"
    return [t for t in turns if t["text"]]


def needs_cleanup(text, lang=None):
    """Only turns with a filler sound or a stutter ("the the") go to Qwen:
    Whisper already leaves out most "um"s, and on a clean turn the model can
    only reword it. The language matters: "er" is a filler in English but
    means "is" in Danish, which once sent nearly every Danish turn to Qwen."""
    return speak_text.has_disfluency(text, lang)


def chunk_lines(lines, count_tokens, budget):
    """Splits transcript lines into chunks of at most ~budget tokens."""
    chunks, current, used = [], [], 0
    for line in lines:
        n = count_tokens(line) + 1
        if current and used + n > budget:
            chunks.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def render_markdown(info, turns, summary=None):
    """info keys: title, date, duration, sources, speakers, languages, notes."""
    lines = [f"# {info['title']}", "", "| | |", "| :--- | :--- |"]
    lines.append(f"| Date | {info['date']} |")
    lines.append(f"| Length | {info['duration']} |")
    lines.append(f"| Audio | {info['sources']} |")
    lines.append(f"| Speakers | {info['speakers'] or 'None detected'} |")
    if info.get("languages"):
        lines.append(f"| Languages | {info['languages']} |")
    lines.append("")
    for note in info.get("notes") or []:
        lines += [f"> {note}", ""]
    if summary:
        lines += [summary.strip(), ""]
    lines += ["## Transcript", ""]
    for t in turns:
        lines += [f"**[{format_timestamp(t['start'])}] {t['label']}:** {t['text']}", ""]
    if not turns:
        lines += ["_No speech was detected in this recording._", ""]
    lines += ["---", "_Transcribed offline by Speak. Speaker labels are automatic: "
              "rename them with find and replace (for example, Person 1 to Anna)._", ""]
    return "\n".join(lines)


# ---- Minutes: the code finds the moments, the model phrases them -----------
#
# Asked to write minutes from a whole transcript, Qwen 1.5B got the gist but
# gave tasks to the wrong people and missed decisions. Asked to cite line
# numbers, it cited the wrong lines; asked about a whole line, it skipped
# items and copied them from neighbouring lines. Small models are reliable
# at one narrow question about one short sentence, so:
# 1. The code cuts every line into clauses at clear cues, in all four
#    languages: promises ("I will"), requests ("can you") and agreements
#    ("let us"). The cue decides what the clause is.
# 2. For each clause the model only says whether it commits anyone to
#    anything, and phrases it as a to-do with any deadline, as a small JSON
#    object (parse_clause_answer).
# 3. The owner comes from the transcript, never from the model: a promise
#    belongs to its speaker, a request to whoever says yes right after it.
#    Every item shows when it was said, and items whose words are not in
#    their clause are dropped.

# The prompt is the same for every clause, so llama.cpp caches it after the
# first call and the examples cost almost nothing. They also keep the JSON
# well-formed: 48 of 48 answers parsed in the 10-minute test without
# llama.cpp's JSON grammar, which made every answer 50% slower (it checks
# each step against Qwen's 151k-token vocabulary).
MINUTES_EXTRACT_PROMPT = """You turn one sentence from a meeting into a to-do line. The sentence is marked >>. A line before it, if shown, only explains words like "it" or "them".
Return JSON:
- keep: false if the sentence does not commit anyone to do anything (an opinion, a question, small talk, a thank-you), otherwise true.
- text: a short instruction that makes sense on its own: start with a verb, say what it is about (replace "it" and "them" using the line before), leave out people's names, and write it in the language of the sentence.
- due: the deadline the sentence gives, or "" if none.

Example:
>> [Ben] Can you fix the two checkout bugs before the release?
Answer: {"keep": true, "text": "Fix the two checkout bugs", "due": "before the release"}

Example:
[Anna] The customer is waiting for the offer.
>> [Ben] Yes, I will send it tomorrow.
Answer: {"keep": true, "text": "Send the offer to the customer", "due": "tomorrow"}

Example:
>> [Ben] Jeg giver butikkerne besked om de nye priser i næste uge.
Answer: {"keep": true, "text": "Give butikkerne besked om de nye priser", "due": "i næste uge"}

Example:
>> [Anna] Let us move the launch to May.
Answer: {"keep": true, "text": "Move the launch to May", "due": ""}

Example:
>> [Ben] I can see why customers like the new design.
Answer: {"keep": false, "text": "", "due": ""}"""

_REQUEST = re.compile(r"\b(can you|could you|would you|please|kan du|vil du|kunne du|kannst du|könntest du|"
                      r"bitte|peux-tu|pouvez-vous|pourrais-tu|s'il te plaît|s'il vous plaît)\b", re.IGNORECASE)
_COMMIT = re.compile(r"\b(i will|i'll|i can|i am going to|i'm going to|let me|"
                     r"jeg vil|jeg skal|jeg kan|jeg giver|jeg sender|jeg tager|jeg ringer|"
                     r"ich werde|ich kann|ich mache|ich schicke|ich rufe|je vais|je peux|je m'en occupe)\b",
                     re.IGNORECASE)
_DECIDE = re.compile(r"\b(let us|let's|we will|we'll|we should|we need to|we agreed|agreed|decided|"
                     r"lad os|vi skal|vi gør|vi er enige|enige|besluttet|"
                     r"lass uns|wir werden|wir sollten|einverstanden|beschlossen|"
                     r"on va|nous allons|il faut|d'accord|décidé)\b", re.IGNORECASE)
_AGREE = {"yes", "yeah", "yep", "sure", "ok", "okay", "will", "absolutely", "certainly", "fine",
          "ja", "jep", "jo", "klart", "fint", "selvfølgelig", "jawohl", "klar", "gerne", "oui", "d'accord", "bien"}


def split_clauses(text):
    """[(kind, clause)] with kind 'promise', 'request' or 'decision'. A clause
    starts at its sentence start (or at its cue when an earlier cue already
    took that sentence) and ends at the end of the cue's sentence or where the
    next clause starts, whichever comes first."""
    hits = sorted((m.start(), m.end(), kind) for rx, kind in
                  ((_COMMIT, "promise"), (_REQUEST, "request"), (_DECIDE, "decision")) for m in rx.finditer(text))
    clauses, last_end = [], -1
    for pos, end, kind in hits:
        if pos < last_end:
            continue    # "we agreed" also contains "agreed"
        last_end = end
        sentence = max((m.end() for m in re.finditer(r"[.!?]\s+", text[:pos])), default=0)
        start = sentence if not clauses or sentence > clauses[-1][0] else pos
        stop = re.search(r"[.!?](\s|$)", text[end:])
        clauses.append([start, end + stop.end() if stop else len(text), kind])
    for k in range(len(clauses) - 1):
        clauses[k][1] = min(clauses[k][1], clauses[k + 1][0])
    return [(kind, text[s:e].strip()) for s, e, kind in clauses]


def minute_clauses(turns):
    """[(turn index, kind, clause)] for every cue in the meeting."""
    return [(i, kind, clause) for i, t in enumerate(turns) for kind, clause in split_clauses(t["text"])]


def parse_clause_answer(text):
    """{"keep", "text", "due"} from the model's answer, or None when it is not
    usable. Tolerates text around the JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return {"keep": data.get("keep") is True, "text": str(data.get("text") or ""), "due": str(data.get("due") or "")}


def context_for(turns, i, clause):
    """The marked clause, after the previous line only when the clause answers
    it ("Yes, I will call them"). More context made the model copy items from
    the context instead of reading the marked sentence."""
    first = clause.split()[0] if clause.split() else ""
    lines = []
    if i > 0 and _norm_word(first) in _AGREE:
        lines.append(f"[{turns[i - 1]['label']}] {turns[i - 1]['text']}")
    lines.append(f">> [{turns[i]['label']}] {clause}")
    return "\n".join(lines)


def _stems(text):
    return {w[:5] for w in (_norm_word(x) for x in text.split()) if len(w) >= 4}


def grounded(item_text, source_text):
    """True when the item shares a word with the text it was taken from
    (compared on 5-letter stems, which works the same in all four languages)."""
    return bool(_stems(item_text) & _stems(source_text))


def agreed_by_next(turns, i):
    """The next speaker if they answer turn i with a yes, else None."""
    nxt = turns[i + 1] if i + 1 < len(turns) else None
    if nxt and nxt["label"] != turns[i]["label"] and nxt["text"].split():
        if _norm_word(nxt["text"].split()[0]) in _AGREE or _COMMIT.match(nxt["text"]):
            return nxt["label"]
    return None


def render_minutes(summary, items, turns):
    """Markdown for the summary, decisions and action items.

    items: [{"line", "kind", "clause", "keep", "text", "due"}], one per clause.
    A request that the next speaker accepts with a promise of their own is
    listed once, under that promise ("Yes, I will call them")."""
    out = []
    summary = " ".join(str(summary or "").split())
    if summary:
        out += ["## Summary", "", summary, ""]
    promised = {it["line"] for it in items if it.get("keep") and it["kind"] == "promise"}
    decisions, tasks, listed = [], [], []
    for it in sorted(items, key=lambda x: x["line"]):
        i = it["line"]
        text = " ".join(str(it.get("text") or "").split()).rstrip(".")
        if (not it.get("keep") or len(text.split()) < 2 or not 0 <= i < len(turns)
                or not grounded(text, it["clause"])):
            continue    # dropped, ungrounded, or a bare "Agreed"
        stems = _stems(text)
        if any(len(stems & s) >= 0.6 * max(1, min(len(stems), len(s))) for s in listed):
            continue    # the same item, worded a little differently
        listed.append(stems)
        t = turns[i]
        text = text[0].upper() + text[1:]
        when = format_timestamp(t["start"])
        due = " ".join(str(it.get("due") or "").split())
        if due and not (grounded(due, it["clause"]) or due.lower() in it["clause"].lower()):
            due = ""    # a deadline this clause does not say
        if it["kind"] == "decision":
            when_due = f" ({due})" if due and due.lower() not in text.lower() else ""
            decisions.append(f"- {text}{when_due} *({t['label']}, {when})*")
            continue
        owner, raised_by = t["label"], None
        if it["kind"] == "request":
            owner = agreed_by_next(turns, i)
            if owner and i + 1 in promised:
                listed.pop()
                continue    # accepted: the answer's own promise is listed instead
            raised_by = None if owner else t["label"]
        details = ", ".join(x for x in (f"due: {due}" if due else "", f"said at {when}") if x)
        if owner:
            tasks.append(f"- [ ] **{owner}:** {text} *({details})*")
        else:
            tasks.append(f"- [ ] {text} *(no owner agreed; raised by {raised_by}, {details})*")
    out += ["## Decisions", ""] + (decisions or ["_None recorded._"]) + [""]
    out += ["## Action items", ""] + (tasks or ["_None recorded._"]) + [""]
    return "\n".join(out).strip()


def _norm_word(word):
    return "".join(ch for ch in word.lower() if ch.isalnum())


# Words a cleanup may drop without changing what was said.
SPOKEN_FILLERS = {
    "um", "uh", "uhm", "umm", "hmm", "hm", "mm", "erm", "er", "ah", "eh", "oh", "like", "you", "know", "i", "mean",
    "so", "well", "okay", "ok", "basically", "actually", "right", "yeah",
    "øh", "øhm", "øhh", "altså", "sådan", "ikke", "også", "ikk", "jo", "nå", "ja", "nu",
    "äh", "ähm", "also", "quasi", "sozusagen", "halt", "naja",
    "euh", "ben", "bah", "genre", "en", "fait", "du", "coup", "bon", "voilà",
}


def faithful(original, cleaned, max_changed=0.1):
    """True when the cleanup only dropped fillers and repeats and fixed
    punctuation. Removing or adding content words is rewording: in testing
    Qwen turned "Jeg giver butikkerne besked" into "Du giver besked", which
    a plain length check let through."""
    from collections import Counter
    a = [_norm_word(w) for w in original.split() if _norm_word(w)]
    b = [_norm_word(w) for w in cleaned.split() if _norm_word(w)]
    if not a or not b:
        return not a and not b
    removed = Counter(a) - Counter(b)
    added = Counter(b) - Counter(a)
    content_removed = sum(n for w, n in removed.items() if w not in SPOKEN_FILLERS)
    allowance = max(1, int(max_changed * len(a)))
    return content_removed <= allowance and sum(added.values()) <= allowance


def echo_correlation(mic, system, segments, frame=400, max_lag_frames=12):
    """How strongly the mic follows the call audio while remote people talk.
    Correlates 25 ms log-energy envelopes inside the call's speech segments,
    for delays up to 300 ms. Through a headset this stays near 0; when the
    call plays through laptop speakers the mic hears every syllable again and
    it rises well above ECHO_CORRELATION."""
    n = min(len(mic), len(system)) // frame
    if n < 40 or not segments:
        return 0.0

    def envelope(a):
        x = a[:n * frame].reshape(n, frame).astype(np.float64)
        return np.log10(np.mean(x * x, axis=1) + 1e-10)

    em, es = envelope(mic), envelope(system)
    active = np.zeros(n, dtype=bool)
    for s in segments:
        active[int(s[0] * SAMPLE_RATE) // frame:int(s[1] * SAMPLE_RATE) // frame + 1] = True
    best = 0.0
    for lag in range(max_lag_frames + 1):
        idx = np.nonzero(active[:n - lag])[0]
        if len(idx) < 40:
            break
        x, y = es[idx], em[idx + lag]
        if x.std() > 0 and y.std() > 0:
            best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def mute_segments(audio, segments, before=0.1, after=0.3):
    """Silences the given time ranges (padded for the speaker-to-mic delay)."""
    out = audio.copy()
    for s in segments:
        out[max(0, int((s[0] - before) * SAMPLE_RATE)):int((s[1] + after) * SAMPLE_RATE)] = 0.0
    return out


def drop_echo_words(mic_words, system_words, window=0.6):
    """Removes mic words that repeat a call word heard within `window` seconds:
    that is the call coming out of the speakers, not someone in the room."""
    sys_sorted = sorted((w[0], _norm_word(w[2])) for w in system_words)
    starts = [w[0] for w in sys_sorted]
    kept = []
    for w in mic_words:
        norm = _norm_word(w[2])
        lo = bisect.bisect_left(starts, w[0] - window)
        hi = bisect.bisect_right(starts, w[0] + window)
        if norm and any(sys_sorted[i][1] == norm for i in range(lo, hi)):
            continue
        kept.append(w)
    return kept


def merge_similar_clusters(cluster_embeddings, threshold):
    """Diarization sometimes splits one voice into several clusters. Joins any
    two whose voice fingerprints are this similar. Returns {cluster: kept id}."""
    ids = sorted(cluster_embeddings)
    parent = {c: c for c in ids}

    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if cosine(cluster_embeddings[a], cluster_embeddings[b]) >= threshold:
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {c: root(c) for c in ids}


# ---- Audio files -----------------------------------------------------------

def read_wav(path):
    """16 kHz mono int16 WAV -> float32. Survives a recording whose header was
    never finalized (Speak or Windows crashed mid-meeting): the header then says
    0 frames, so the samples are read straight from the file."""
    with wave.open(path, "rb") as w:
        n, rate, channels = w.getnframes(), w.getframerate(), w.getnchannels()
        data = w.readframes(n) if n else b""
    if not data:
        with open(path, "rb") as f:
            data = f.read()[44:]
        data = data[:len(data) // 2 * 2]
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio[:len(audio) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(audio):
        x = np.arange(0, len(audio), rate / SAMPLE_RATE)
        audio = np.interp(x, np.arange(len(audio)), audio).astype(np.float32)
    return audio


class _WavWriter:
    def __init__(self, path):
        self.w = wave.open(path, "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(SAMPLE_RATE)
        self.frames = 0

    def write(self, block):
        pcm = (np.clip(block.reshape(-1), -1.0, 1.0) * 32767).astype(np.int16)
        self.w.writeframes(pcm.tobytes())
        self.frames += len(pcm)

    def close(self):
        try:
            self.w.close()
        except Exception:
            pass


class MeetingRecorder:
    """Records the mic and the system audio side by side into a folder.

    The mic uses sounddevice (same as dictation). System audio uses the soundcard
    library, because sounddevice's PortAudio build has no WASAPI loopback. Each
    stream notes when its first block arrived so both files can be lined up on
    one timeline later."""

    def __init__(self, folder, log, mic=None, speaker=None, max_seconds=4 * 3600, on_limit=None):
        self.folder = folder
        self.log = log
        self.mic_device = mic or None
        self.speaker_name = speaker or None
        self.max_seconds = max_seconds
        self.on_limit = on_limit
        self.stop_event = threading.Event()
        self.mic_queue = queue.Queue()
        self.first_block = {}
        self.threads = []
        self.mic_stream = None
        self.system_error = None
        self.system_device = None
        self.started = None

    def start(self):
        import sounddevice as sd
        os.makedirs(self.folder, exist_ok=True)
        self.started = datetime.datetime.now()
        self.mic_writer = _WavWriter(os.path.join(self.folder, "mic.wav"))
        self.system_writer = _WavWriter(os.path.join(self.folder, "system.wav"))

        def callback(indata, frames, time_info, status):
            if "mic" not in self.first_block:
                self.first_block["mic"] = time.monotonic()
            self.mic_queue.put(indata.copy())

        try:
            self.mic_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                             device=self.mic_device, callback=callback)
            self.mic_stream.start()
        except Exception as e:
            self.mic_writer.close()
            self.system_writer.close()
            raise RuntimeError(f"could not open the microphone ({e})")

        for target, name in ((self._mic_writer_loop, "speak-meeting-mic"),
                             (self._system_loop, "speak-meeting-system")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self.threads.append(th)

    def _mic_writer_loop(self):
        t0 = time.monotonic()
        next_note = 600
        while not (self.stop_event.is_set() and self.mic_queue.empty()):
            try:
                block = self.mic_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.mic_writer.write(block)
            elapsed = time.monotonic() - t0
            if elapsed >= next_note:
                self.log(f"[MEETING] Still recording ({int(elapsed // 60)} min). Press the meeting hotkey to stop.")
                next_note += 600
            if elapsed >= self.max_seconds and self.on_limit and not self.stop_event.is_set():
                self.log(f"[MEETING] Maximum length ({self.max_seconds // 60} min) reached. Stopping automatically.")
                threading.Thread(target=self.on_limit, daemon=True).start()
                self.on_limit = None

    def _system_loop(self):
        try:
            with warnings.catch_warnings():
                # soundcard 0.4.x uses an array API NumPy 2.5 deprecates, and warns
                # on every gap in loopback data (normal when nothing is playing).
                warnings.simplefilter("ignore")
                import soundcard as sc
                speaker = sc.get_speaker(self.speaker_name) if self.speaker_name else sc.default_speaker()
                device = sc.get_microphone(id=str(speaker.name), include_loopback=True)
                self.system_device = speaker.name
                with device.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=1600) as rec:
                    while not self.stop_event.is_set():
                        block = rec.record(numframes=1600)
                        if "system" not in self.first_block:
                            self.first_block["system"] = time.monotonic()
                        self.system_writer.write(block[:, 0] if block.ndim > 1 else block)
        except Exception as e:
            self.system_error = str(e)
            self.log(f"[Meeting Warning] System audio cannot be recorded ({e}). Recording the microphone only.")

    def stop(self):
        self.stop_event.set()
        try:
            self.mic_stream.stop()
            self.mic_stream.close()
        except Exception:
            pass
        for th in self.threads:
            th.join(timeout=5)
        self.mic_writer.close()
        self.system_writer.close()
        t0 = min(self.first_block.values()) if self.first_block else 0.0
        meta = {
            "started": self.started.isoformat(timespec="seconds"),
            "stopped": datetime.datetime.now().isoformat(timespec="seconds"),
            "offsets": {k: round(v - t0, 3) for k, v in self.first_block.items()},
            "system_device": self.system_device,
            "system_error": self.system_error,
        }
        with open(os.path.join(self.folder, "meeting.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta


# ---- Manager: recording, voice profile, processing ---------------------------

class MeetingManager:
    def __init__(self, app, base_dir, models_dir, download_file, log, allowed_languages):
        self.app = app
        self.config = app.config
        self.log = log
        self.download_file = download_file
        self.allowed = set(allowed_languages)
        self.models_dir = models_dir
        self.speaker_dir = os.path.join(models_dir, "speakers")
        self.recordings_dir = os.path.join(base_dir, "recordings")
        self.profile_path = os.path.join(base_dir, "speak_voice.json")
        self.recorder = None
        self.lock = threading.Lock()
        self.jobs = queue.Queue()
        self.current_job = None
        self.worker = threading.Thread(target=self._worker_loop, name="speak-meetings", daemon=True)
        self.worker.start()

    # -- settings --
    @property
    def hotkey(self):
        return str(self.config.get("meeting_hotkey", "CTRL+ALT+M")).upper().replace(" ", "")

    @property
    def is_recording(self):
        return self.recorder is not None

    def output_folder(self):
        folder = self.config.get("meeting_folder")
        if folder:
            return os.path.expandvars(os.path.expanduser(folder))
        # Deliberately the local profile folder (C:\Users\<name>\Documents), not
        # the Windows "Documents" known folder: on company laptops that one is
        # often redirected to OneDrive, which would upload every transcript.
        return os.path.join(os.path.expanduser("~"), "Documents", "Speak Meetings")

    # -- models --
    def ensure_speaker_models(self, need_segmentation=True):
        """Downloads the speaker models once (about 47 MB). Returns (seg, emb) paths or None."""
        os.makedirs(self.speaker_dir, exist_ok=True)
        emb = os.path.join(self.speaker_dir, EMBEDDING_MODEL["file"])
        if not os.path.exists(emb):
            self.log("[Meeting] First-time setup: downloading the speaker recognition model (40 MB, one time only)...")
            if not self.download_file(EMBEDDING_MODEL["url"], emb, EMBEDDING_MODEL["sha256"], EMBEDDING_MODEL["file"]):
                return None
        seg = os.path.join(self.speaker_dir, SEGMENTATION_MODEL["file"])
        if need_segmentation and not os.path.exists(seg):
            self.log("[Meeting] First-time setup: downloading the speaker segmentation model (7 MB, one time only)...")
            archive = seg + ".tar.bz2"
            if not self.download_file(SEGMENTATION_MODEL["url"], archive, SEGMENTATION_MODEL["sha256"], "speaker segmentation"):
                return None
            try:
                # Read the one member we need by name; never extract the archive
                # as a whole (no path traversal, no stray files).
                with tarfile.open(archive, "r:bz2") as tar:
                    src = tar.extractfile(SEGMENTATION_MODEL["member"])
                    with open(seg + ".part", "wb") as out:
                        shutil.copyfileobj(src, out)
                os.replace(seg + ".part", seg)
            except Exception as e:
                self.log(f"[Meeting Error] Could not unpack the segmentation model: {e}")
                return None
            finally:
                for p in (archive, seg + ".part"):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        return seg, emb

    def ensure_whisper_model(self):
        size = self.config.get("meeting_model", "small")
        if os.path.exists(os.path.join(self.models_dir, f"models--Systran--faster-whisper-{size}")):
            return True
        self.log(f"[Meeting] First-time setup: downloading speech model 'faster-whisper-{size}' for meetings (one time only)...")
        try:
            from faster_whisper.utils import download_model
            download_model(size, cache_dir=self.models_dir)
            return True
        except Exception as e:
            self.log(f"[Meeting Error] Speech model could not be downloaded: {e}")
            return False

    def ensure_all_models(self):
        return bool(self.ensure_speaker_models()) and self.ensure_whisper_model()

    def _embedding_extractor(self, emb_path):
        import sherpa_onnx
        return sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=emb_path, num_threads=int(self.config.get("whisper_threads", 4))))

    @staticmethod
    def _embed(extractor, audio):
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        v = np.array(extractor.compute(stream), dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    # -- voice profile --
    def load_profile(self):
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model") != EMBEDDING_MODEL["file"]:
                self.log("[Meeting] Your voice sample was made with an older speaker model. Record it again (r, then 2).")
                return None
            return np.array(data["embedding"], dtype=np.float32)
        except FileNotFoundError:
            return None
        except Exception as e:
            self.log(f"[Meeting Warning] Voice sample could not be read ({e}). Record it again (r, then 2).")
            return None

    def profile_info(self):
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f).get("created", "yes")
        except Exception:
            return None

    def save_profile_from_audio(self, audio):
        """Builds and saves the voice profile from a recording. Returns seconds of speech used, or 0."""
        from faster_whisper.vad import get_speech_timestamps
        speech = [audio[c["start"]:c["end"]] for c in get_speech_timestamps(audio, sampling_rate=SAMPLE_RATE)]
        speech = np.concatenate(speech) if speech else np.zeros(0, np.float32)
        seconds = len(speech) / SAMPLE_RATE
        if seconds < ENROLL_MIN_SPEECH:
            self.log(f"[Meeting] Only {seconds:.0f}s of speech was heard; {ENROLL_MIN_SPEECH}s is needed. "
                     "Please try again and keep talking for the whole 30 seconds.")
            return 0
        paths = self.ensure_speaker_models(need_segmentation=False)
        if not paths:
            return 0
        embedding = self._embed(self._embedding_extractor(paths[1]), speech)
        data = {"model": EMBEDDING_MODEL["file"],
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "speech_seconds": round(seconds, 1),
                "embedding": [round(float(x), 6) for x in embedding]}
        tmp = self.profile_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.profile_path)
        return seconds

    def enroll_interactive(self):
        """Console flow for the 30 s voice sample."""
        import sounddevice as sd
        if self.is_recording:
            self.log("[Meeting] Stop the meeting recording first.")
            return
        self.log("\n--- RECORD MY VOICE SAMPLE ---")
        self.log("Speak uses this sample to label you as \"Me\" in meeting transcripts.")
        self.log("It is stored on this PC only, as numbers that describe your voice (no audio is kept).")
        self.log("Use the microphone you use in meetings, in a quiet room.\n")
        self.log(ENROLL_TEXT)
        try:
            input(f"\nPress Enter to start recording ({ENROLL_SECONDS} seconds)...")
        except (EOFError, KeyboardInterrupt):
            return
        try:
            audio = sd.rec(int(ENROLL_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                           dtype="float32", device=self.config.get("meeting_mic") or None)
            self.app.play_sound("start")
            for left in range(ENROLL_SECONDS, 0, -5):
                print(f"\r[Recording] Keep talking... {left:2d}s left ", end="", flush=True)
                time.sleep(5)
            sd.wait()
            print(flush=True)
            self.app.play_sound("stop")
        except Exception as e:
            self.log(f"\n[Audio Error] Could not record from the microphone: {e}")
            return
        seconds = self.save_profile_from_audio(audio[:, 0])
        if seconds:
            self.log(f"[Meeting] Voice sample saved ({seconds:.0f}s of speech). You will appear as \"Me\" in transcripts.")
            self.app.play_sound("success")
        else:
            self.app.play_sound("error")

    def delete_profile(self):
        try:
            os.remove(self.profile_path)
            self.log("[Meeting] Voice sample deleted.")
        except FileNotFoundError:
            self.log("[Meeting] There is no voice sample to delete.")

    # -- recording --
    def toggle(self):
        if self.is_recording:
            self.stop()
        else:
            self.start()

    def start(self):
        with self.lock:
            if self.recorder:
                return
            folder = os.path.join(self.recordings_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            rec = MeetingRecorder(folder, self.log,
                                  mic=self.config.get("meeting_mic"),
                                  speaker=self.config.get("meeting_speaker"),
                                  max_seconds=int(self.config.get("meeting_max_minutes", 240)) * 60,
                                  on_limit=self.stop)
            try:
                rec.start()
            except Exception as e:
                self.log(f"[Meeting Error] {e}")
                self.app.play_sound("error")
                shutil.rmtree(folder, ignore_errors=True)
                return
            self.recorder = rec
        _set_title("Speak - RECORDING MEETING")
        self.app.play_sound("start")
        self.log(f"\n[MEETING] Recording the microphone and system audio. Press {self.hotkey} again to stop.")
        self.log("          Remember to tell the other participants that you are recording.")
        if not os.path.exists(self.profile_path):
            self.log("          Tip: record a voice sample (type r, then 2) so the transcript can label you as \"Me\".")
        # Fetch any missing models while the meeting runs, not after it.
        threading.Thread(target=self.ensure_all_models, name="speak-meeting-models", daemon=True).start()

    def stop(self):
        with self.lock:
            rec, self.recorder = self.recorder, None
        if not rec:
            return
        meta = rec.stop()
        _set_title("Speak - Local Voice Dictation")
        self.app.play_sound("stop")
        minutes = (datetime.datetime.fromisoformat(meta["stopped"]) - datetime.datetime.fromisoformat(meta["started"])).total_seconds() / 60
        self.log(f"[MEETING] Recording stopped ({minutes:.0f} min). Transcribing in the background; "
                 "dictation keeps working meanwhile.")
        self.submit(rec.folder)

    def shutdown(self):
        """On quit: finalize a running recording so it can be processed later."""
        if self.is_recording:
            with self.lock:
                rec, self.recorder = self.recorder, None
            rec.stop()
            self.log(f"[MEETING] Recording saved but not transcribed: {rec.folder}")
            self.log("          Start Speak again and type r, then 4, to transcribe it.")
        elif self.current_job:
            self.log(f"[MEETING] Transcription was interrupted. The recording is kept in: {self.current_job}")

    def leftover_recordings(self):
        if not os.path.isdir(self.recordings_dir):
            return []
        queued = set(self._queued)
        return [os.path.join(self.recordings_dir, d) for d in sorted(os.listdir(self.recordings_dir))
                if os.path.exists(os.path.join(self.recordings_dir, d, "mic.wav"))
                and os.path.join(self.recordings_dir, d) not in queued
                and os.path.join(self.recordings_dir, d) != self.current_job
                and not (self.recorder and self.recorder.folder == os.path.join(self.recordings_dir, d))]

    @property
    def _queued(self):
        with self.jobs.mutex:
            return list(self.jobs.queue)

    def submit(self, path):
        self.jobs.put(path)

    def _worker_loop(self):
        while True:
            path = self.jobs.get()
            self.current_job = path
            try:
                self.process(path)
            except Exception as e:
                self.log(f"[Meeting Error] Transcription failed: {e}")
                self.log(f"              The recording is kept in: {path}")
                self.app.play_sound("error")
            finally:
                self.current_job = None
                self.jobs.task_done()

    # -- processing --
    def _load_streams(self, path):
        """Returns ([(name, audio)], meta, is_own_recording)."""
        if os.path.isdir(path):
            meta = {}
            try:
                with open(os.path.join(path, "meeting.json"), "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
            offsets = meta.get("offsets", {})
            streams = []
            for name in ("mic", "system"):
                p = os.path.join(path, f"{name}.wav")
                if not os.path.exists(p):
                    continue
                audio = read_wav(p)
                pad = int(offsets.get(name, 0.0) * SAMPLE_RATE)
                if pad > 0:
                    audio = np.concatenate([np.zeros(pad, np.float32), audio])
                streams.append((name, audio))
            own = os.path.normcase(os.path.abspath(os.path.dirname(path))) == os.path.normcase(os.path.abspath(self.recordings_dir))
            return streams, meta, own
        from faster_whisper import decode_audio
        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        started = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
        return [("mic", audio)], {"started": started, "source_file": os.path.basename(path)}, False

    def _diarize(self, audio, seg_path, emb_path, label):
        import sherpa_onnx
        n = self.config.get("meeting_num_speakers")
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=seg_path,
                    # 0.2 instead of the default 0.1 window shift: 2.4x faster on
                    # the benchmark meeting with identical speaker turns.
                    window_shift_ratio=float(self.config.get("meeting_window_shift", 0.2))),
                num_threads=int(self.config.get("whisper_threads", 4))),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=emb_path, num_threads=int(self.config.get("whisper_threads", 4))),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=int(n) if n else -1,
                threshold=float(self.config.get("meeting_cluster_threshold", 0.5))),
            min_duration_on=0.3, min_duration_off=0.5)
        if not cfg.validate():
            raise RuntimeError("speaker diarization settings are invalid")
        sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
        shown = [-1]

        def progress(done, total):
            pct = done * 100 // max(1, total)
            if pct // 25 != shown[0]:
                shown[0] = pct // 25
                print(f"\r[Meeting] Finding speakers ({label}): {pct:3d}%", end="", flush=True)
            return 0

        result = sd.process(np.ascontiguousarray(audio, dtype=np.float32), callback=progress).sort_by_start_time()
        print(flush=True)
        return [(float(r.start), float(r.end), int(r.speaker)) for r in result]

    def _cluster_embeddings(self, extractor, audio, segments, max_seconds=30.0):
        """One voice fingerprint per cluster, from its longest segments."""
        out = {}
        for c in sorted({s[2] for s in segments}):
            segs = sorted((s for s in segments if s[2] == c), key=lambda s: s[1] - s[0], reverse=True)
            parts, total = [], 0.0
            for s in segs:
                parts.append(audio[int(s[0] * SAMPLE_RATE):int(s[1] * SAMPLE_RATE)])
                total += s[1] - s[0]
                if total >= max_seconds:
                    break
            clip = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            if len(clip) >= SAMPLE_RATE // 2:
                out[c] = self._embed(extractor, clip)
        return out

    def _detect(self, model, clip):
        try:
            _, _, probs = model.detect_language(audio=clip)
        except Exception:
            return None, 0.0
        probs = [(lang, p) for lang, p in probs if lang in self.allowed]
        return probs[0] if probs else (None, 0.0)

    def _speaker_languages(self, model, audio_by_stream, segments_by_key, fallback):
        """Language per segment: {(speaker key, segment start): lang}.

        One detection per speaker, on up to 25 s of their speech glued together.
        Each detection costs a full 30 s encoder pass (3.6 s with 'small' on a
        laptop), so detecting every turn was most of the transcription time. An
        unsure result means the person switches languages; only then are their
        turns of 3 s or more detected one by one. Speakers with under 2 s of
        speech take the meeting's main language."""
        locked = self.app.language if self.app.language in self.allowed else None
        langs, pending, weight = {}, [], {}
        for key, segs in segments_by_key.items():
            if locked:
                langs.update({(key, s[0]): locked for s in segs})
                continue
            audio = audio_by_stream[key[0]]
            clip = lambda s: audio[int(s[0] * SAMPLE_RATE):int(s[1] * SAMPLE_RATE)]
            total = sum(s[1] - s[0] for s in segs)
            if total < 2.0:
                pending.append(key)
                continue
            parts, used = [], 0.0
            for s in sorted(segs, key=lambda s: s[1] - s[0], reverse=True):
                parts.append(clip(s))
                used += s[1] - s[0]
                if used >= 25:
                    break
            lang, prob = self._detect(model, np.concatenate(parts)[:28 * SAMPLE_RATE])
            lang = lang or fallback
            weight[lang] = weight.get(lang, 0.0) + total
            for s in segs:
                own = lang
                if prob < 0.6 and s[1] - s[0] >= 3.0:
                    own = self._detect(model, clip(s))[0] or lang
                langs[(key, s[0])] = own
        main = max(weight, key=weight.get) if weight else (locked or fallback)
        for key in pending:
            langs.update({(key, s[0]): main for s in segments_by_key[key]})
        return langs

    def _transcribe_ranges(self, model, audio, ranges_by_lang, progress, prompt):
        """One Whisper call per language: that language's speech glued into one
        track. Returns [(start, end, word, lang)] on the meeting timeline.
        progress: {'done', 'total', 'shown'} in track seconds, shared by both streams."""
        words = []
        beam = int(self.config.get("meeting_beam_size", 1))
        for lang, ranges in ranges_by_lang.items():
            track, pieces = build_track(audio, ranges)
            if not pieces:
                continue
            segments, _ = model.transcribe(
                track, language=lang, beam_size=beam, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=True, condition_on_previous_text=False, initial_prompt=prompt)
            for seg in segments:
                if speak_text.is_hallucination(seg.text, seg.no_speech_prob):
                    if seg.text.strip():
                        self.log(f"\n[Meeting] Ignored \"{seg.text.strip()}\" (Whisper invents this on silence).")
                    continue
                for w in seg.words or []:
                    start = track_to_meeting_time(w.start, pieces)
                    end = track_to_meeting_time(w.end, pieces, is_start=False)
                    words.append((start, max(start, end), w.word, lang))
                pct = int(min(1.0, (progress["done"] + seg.end) / max(1.0, progress["total"])) * 100)
                if pct != progress["shown"]:
                    progress["shown"] = pct
                    print(f"\r[Meeting] Transcribing: {pct:3d}%", end="", flush=True)
            progress["done"] += len(track) / SAMPLE_RATE
        return words

    def process(self, path):
        t_all = time.time()
        if not self.ensure_all_models():
            raise RuntimeError("the meeting models are not available (no internet during first use?)")
        seg_path, emb_path = self.ensure_speaker_models()
        streams, meta, own = self._load_streams(path)
        audio_by_stream = dict(streams)
        duration = max((len(a) for _, a in streams), default=0) / SAMPLE_RATE
        self.log(f"[Meeting] Processing {duration / 60:.1f} min of audio from {path}")

        def has_signal(name):
            a = audio_by_stream.get(name)
            return a is not None and len(a) >= SAMPLE_RATE and float(np.max(np.abs(a))) >= 1e-3

        # 1. Who spoke when. The call audio goes first: its speech segments tell
        #    when the mic may be hearing the call through the speakers. A silent
        #    system stream (in-room meeting) is skipped outright.
        t0 = time.time()
        diar, notes = {}, []
        if has_signal("system"):
            diar["system"] = self._diarize(audio_by_stream["system"], seg_path, emb_path, "call audio")
        call_mode = bool(diar.get("system"))
        if call_mode and has_signal("mic"):
            echo = echo_correlation(audio_by_stream["mic"], audio_by_stream["system"], diar["system"])
            if echo >= ECHO_CORRELATION:
                # Without this the call's voices also show up as extra people in
                # the mic and their words are written twice.
                audio_by_stream["mic"] = mute_segments(audio_by_stream["mic"], diar["system"])
                self.log(f"[Meeting] The call is audible in your microphone (echo {echo:.2f}, no headset?). "
                         "The microphone is ignored while remote people talk.")
                notes.append("The call was audible in the microphone (no headset?), so the microphone was "
                             "ignored while remote people talked. Use a headset to also capture people talking "
                             "over each other.")
        if has_signal("mic"):
            diar["mic"] = self._diarize(audio_by_stream["mic"], seg_path, emb_path, "microphone")
        t_diar = time.time() - t0

        # 2. Voices: join clusters that are one person, find "Me", drop echo.
        extractor = self._embedding_extractor(emb_path)
        merge_threshold = float(self.config.get("meeting_merge_threshold", 0.6))
        embs_by_stream = {}
        for name, segs in diar.items():
            embs = self._cluster_embeddings(extractor, audio_by_stream[name], segs)
            mapping = merge_similar_clusters(embs, merge_threshold)
            n_joined = sum(k != v for k, v in mapping.items())
            if n_joined:
                diar[name] = [(s, e, mapping.get(c, c)) for s, e, c in segs]
                embs = self._cluster_embeddings(extractor, audio_by_stream[name], diar[name])
                self.log(f"[Meeting] Joined {n_joined} split voice(s) in the "
                         f"{'microphone' if name == 'mic' else 'call audio'}.")
            embs_by_stream[name] = embs
        profile = self.load_profile()
        me, me_guessed = set(), False
        mic_segments = diar.get("mic", [])
        if mic_segments and profile is not None:
            embs = embs_by_stream["mic"]
            threshold = float(self.config.get("meeting_me_threshold", 0.5))
            me = match_me(embs, profile, threshold)
            scores = ", ".join(f"voice {c}: {cosine(e, profile):.2f}" for c, e in embs.items())
            self.log(f"[Meeting] Match with your voice sample ({scores}; 'Me' from {threshold:.2f}).")
        if mic_segments and call_mode:
            mic_segments, dropped = drop_call_echo(mic_segments, diar["system"], me)
            if dropped:
                self.log(f"[Meeting] Ignored {dropped} microphone segment(s) that were the call playing "
                         "through your speakers.")
        if mic_segments and profile is None and call_mode:
            # No voice sample: in a call, the mic is mostly you. Guessed only after
            # the echo is gone, so a remote voice is never picked.
            totals = {}
            for s in mic_segments:
                totals[s[2]] = totals.get(s[2], 0.0) + s[1] - s[0]
            me, me_guessed = {max(totals, key=totals.get)}, True
        del extractor

        segments_by_key = {}
        for s in mic_segments:
            segments_by_key.setdefault(("mic", s[2]), []).append(s)
        for s in diar.get("system", []):
            segments_by_key.setdefault(("system", s[2]), []).append(s)

        # 3. Languages, then one Whisper pass per same-language stretch.
        t0 = time.time()
        size = self.config.get("meeting_model", "small")
        self.log(f"[Meeting] Loading speech model 'faster-whisper-{size}'...")
        model = self.app._load_whisper(size)
        fallback = self.config.get("last_language") if self.config.get("last_language") in self.allowed else "en"
        langs = self._speaker_languages(model, audio_by_stream, segments_by_key, fallback)
        prompt = self.app.dictionary.whisper_prompt()
        keyed, ranges = {}, {}
        for name in ("system", "mic"):
            keyed[name] = [(s[0], s[1], key) for key, segs in segments_by_key.items() if key[0] == name for s in segs]
            ranges[name] = language_ranges([(s, e, langs[(k, s)]) for s, e, k in keyed[name]])
        total = sum(sum(e - s for s, e in r) + TRACK_GAP * (len(r) - 1)
                    for by_lang in ranges.values() for r in by_lang.values())
        progress = {"done": 0.0, "total": total, "shown": -1}
        words = {name: self._transcribe_ranges(model, audio_by_stream[name], ranges[name], progress, prompt)
                 for name in ("system", "mic") if ranges[name]}
        print(flush=True)
        del model
        gc.collect()
        if call_mode and words.get("mic"):
            before = len(words["mic"])
            words["mic"] = drop_echo_words(words["mic"], words.get("system", []))
            if before > len(words["mic"]):
                self.log(f"[Meeting] Removed {before - len(words['mic'])} word(s) the microphone heard from the call.")
        assigned = []
        for name, w in words.items():
            assigned += assign_words(w, keyed[name])
        t_whisper = time.time() - t0

        # 4. Turns, labels, cleanup.
        turns = fix_split_sentences(build_turns(assigned))
        labels = label_speakers([t["speaker"] for t in turns], {("mic", c) for c in me})
        for t in turns:
            t["label"] = labels[t["speaker"]]
        if profile is None and any(t["speaker"][0] == "mic" for t in turns):
            notes.append("\"Me\" was guessed as the main microphone voice. Record a voice sample "
                         "(in Speak, type r then 2) for reliable labelling." if me_guessed else
                         "No voice sample is recorded, so you appear as a numbered person. "
                         "Record one in Speak (type r, then 2).")
        t0 = time.time()
        ai_on = self.app.ai_mode != "off" and self.app.local_llm is not None
        clean = ai_on and self.config.get("meeting_cleanup", True)
        todo = {id(t) for t in turns if clean and needs_cleanup(t["text"], t["lang"])}
        rejected = done = 0
        for t in turns:
            text = self.app.clean_text(t["text"], t["lang"], use_llm=False, verbose=False)
            if id(t) in todo:
                done += 1
                print(f"\r[Meeting] Cleaning up turns: {done}/{len(todo)}", end="", flush=True)
                # The dictation LLM is shared: hold the engine lock one turn at a
                # time so a dictation never waits longer than a single turn.
                with self.app.engine_lock:
                    cleaned = self.app.clean_text(
                        t["text"], t["lang"], verbose=False,
                        instructions=MEETING_CLEANUP_PROMPTS.get(t["lang"], MEETING_CLEANUP_PROMPTS["en"]))
                cleaned = cleaned.lstrip("-*• ").strip()
                if faithful(text, cleaned):
                    text = cleaned
                else:
                    rejected += 1   # the model reworded it: keep what was said
            t["text"] = speak_text.light_cleanup(text)
        if todo:
            print(flush=True)
            self.log(f"[Meeting] AI cleanup: {len(todo) - rejected} of {len(turns)} turns had fillers or stutters "
                     f"removed{f'; {rejected} kept as spoken because the AI reworded them' if rejected else ''}.")
        t_clean = time.time() - t0

        # 5. Summary.
        t0 = time.time()
        summary = None
        if ai_on and self.config.get("meeting_summary", True) and turns:
            summary = self._summarize(turns)
        t_summary = time.time() - t0

        # 6. Write the Markdown file.
        started = datetime.datetime.fromisoformat(meta.get("started") or datetime.datetime.now().isoformat())
        word_counts = {}
        for t in turns:
            word_counts[t["lang"]] = word_counts.get(t["lang"], 0) + len(t["text"].split())
        ordered_labels = []
        for t in turns:
            if t["label"] not in ordered_labels:
                ordered_labels.append(t["label"])
        if meta.get("source_file"):
            sources = f"Imported file: {meta['source_file']}"
        elif call_mode:
            sources = "Microphone + system audio (call)"
        else:
            sources = "Microphone (in-room)"
        if meta.get("system_error"):
            notes.append(f"System audio could not be recorded ({meta['system_error']}), so remote participants may be missing.")
        info = {
            "title": f"Meeting {started:%Y-%m-%d %H:%M}",
            "date": f"{started:%A %d %B %Y, %H:%M}",
            "duration": f"{duration / 60:.0f} min",
            "sources": sources,
            "speakers": ", ".join(ordered_labels),
            "languages": ", ".join(LANGUAGE_NAMES.get(k, k) for k, _ in sorted(word_counts.items(), key=lambda kv: -kv[1])),
            "notes": notes,
        }
        folder = self.output_folder()
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, f"{started:%Y-%m-%d %H-%M} Meeting.md")
        n = 2
        while os.path.exists(out):
            out = os.path.join(folder, f"{started:%Y-%m-%d %H-%M} Meeting ({n}).md")
            n += 1
        with open(out, "w", encoding="utf-8") as f:
            f.write(render_markdown(info, turns, summary))

        if own and not self.config.get("meeting_keep_audio", False):
            shutil.rmtree(path, ignore_errors=True)
        self.log(f"[Meeting] Timings: speakers {t_diar:.0f}s, transcription {t_whisper:.0f}s, "
                 f"cleanup {t_clean:.0f}s, summary {t_summary:.0f}s (total {time.time() - t_all:.0f}s "
                 f"for {duration / 60:.1f} min of audio).")
        self.log(f"[MEETING DONE] {len(turns)} turns, speakers: {info['speakers'] or 'none'}.")
        self.log(f"               Saved to: {out}\n")
        self.app.play_sound("success")
        return out

    def _summarize(self, turns):
        """Summary, decisions and action items (see the Minutes section above),
        on an 8k-context Qwen instance of its own: the dictation instance keeps
        its 2k context and stays free for dictation."""
        llm = self.app.load_llm(SUMMARY_CTX)
        if llm is None:
            return None
        counts = {}
        for t in turns:
            counts[t["lang"]] = counts.get(t["lang"], 0) + len(t["text"].split())
        language = LANGUAGE_NAMES.get(max(counts, key=counts.get), "English")

        def ask(system, user, max_tokens):
            prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
                      f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
            res = llm(prompt, max_tokens=max_tokens, temperature=0.1, stop=["<|im_end|>"])
            return res["choices"][0]["text"].strip()

        def tokens(text):
            return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

        try:
            # 1. Tasks and decisions, one clause at a time.
            clauses = minute_clauses(turns)
            items = []
            for n, (i, kind, clause) in enumerate(clauses):
                print(f"\r[Meeting] Finding tasks and decisions: {n + 1}/{len(clauses)}", end="", flush=True)
                answer = parse_clause_answer(ask(MINUTES_EXTRACT_PROMPT, context_for(turns, i, clause), 120))
                if answer:
                    items.append({"line": i, "kind": kind, "clause": clause, **answer})
            if clauses:
                print(flush=True)

            # 2. The summary paragraph, condensed from parts for long meetings.
            print("[Meeting] Writing the summary...", flush=True)
            lines = [f"{t['label']}: {t['text']}" for t in turns]
            summaries = [ask(
                f"Summarize this part of a meeting transcript in 2 to 4 sentences in {language}: what was "
                "discussed and what came out of it. Only state what is said. Output only the summary.",
                chunk, 250) for chunk in chunk_lines(lines, tokens, SUMMARY_CHUNK_TOKENS)]
            summary = summaries[0] if len(summaries) == 1 else ask(
                f"Combine these partial summaries of one meeting into one summary of 3 to 5 sentences in "
                f"{language}. Output only the summary.", "\n\n".join(summaries), 300)
            return render_minutes(summary, items, turns)
        except Exception as e:
            self.log(f"[Meeting Warning] The summary could not be written ({e}).")
            return None
        finally:
            llm.close()     # free the 8k KV cache now, not at the next garbage collection
            gc.collect()


def _set_title(title):
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    except Exception:
        pass
