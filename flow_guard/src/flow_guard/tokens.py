"""Text normalization and token extraction.

Tokens are the things a rewrite must never silently lose: placeholders,
numbers, currency amounts, times and URLs.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")
_ESCAPED_BRACKET_RE = re.compile(r"\\([\[\]])")

_CURLY_VAR_RE = re.compile(r"\{\{\s*([^{}\s]+?)\s*\}\}")
_SQUARE_VAR_RE = re.compile(r"\[\[\s*([^\[\]\s]+?)\s*\]\]")
_URL_RE = re.compile(r"https?://(?:\{\{[^{}\s]*\}\}|[^\s<>\"'`)\]{}])+", re.IGNORECASE)
_CURRENCY_RE = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s*(\d[\d,]*(?:\.\d+)?)"
    r"|(\d[\d,]*(?:\.\d+)?)\s*(?:rupees|rupee|रुपये|रुपए|రూపాయలు|ರೂಪಾಯಿ|ரூபாய்)",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)(?![a-z])"
    r"|\b([01]?\d|2[0-3]):([0-5]\d)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# A markdown ordered-list marker at the start of a line ("1. ", "  2) ").
# Renumbering or turning the list into bullets is formatting, not content.
_LIST_MARKER_RE = re.compile(r"(?m)^(\s*)(\d+[.)])(?=\s)")

# Mojibake: UTF-8 bytes decoded as latin-1/cp1252 show up as these pairs.
_MOJIBAKE_RE = re.compile(r"[ÃÂàáâäåæçèéêëìíîïðñòóôõö][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc\u2013-\u203a\u20ac\u2122]")

TOKEN_KINDS = ("placeholder", "number", "currency", "time", "url")


def to_ascii_digits(text: str) -> str:
    """Map any Unicode decimal digit (Devanagari, Telugu, ...) to ASCII."""
    return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() and not ch.isascii() else ch for ch in text)


def unescape_markers(text: str) -> str:
    r"""`\[\[x]]` (markdown-escaped) and `[[x]]` are the same token."""
    return _ESCAPED_BRACKET_RE.sub(r"\1", text)


def collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_text(text: str | None) -> str:
    """Canonical form for equality and similarity: NFC, unescaped markers,
    collapsed whitespace."""
    if not text:
        return ""
    return collapse_ws(unescape_markers(unicodedata.normalize("NFC", text)))


def text_hash(text: str | None) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _prepare(text: str | None) -> str:
    return to_ascii_digits(unescape_markers(unicodedata.normalize("NFC", text or "")))


def _blank(m: re.Match) -> str:
    return " " * len(m.group(0))  # same length, so later spans still line up


def token_occurrences(text: str | None) -> tuple[str, list[tuple[str, str, int, int]]]:
    """(prepared_text, [(kind, token, start, end), ...]) in document order.

    Every occurrence is kept: a token that appears 9 times is counted 9 times.
    Spans index into prepared_text (NFC, unescaped markers, ASCII digits).
    """
    text = _prepare(text)
    occ: list[tuple[str, str, int, int]] = []
    if not text:
        return text, occ
    for m in _CURLY_VAR_RE.finditer(text):
        occ.append(("placeholder", "{{" + m.group(1) + "}}", m.start(), m.end()))
    for m in _SQUARE_VAR_RE.finditer(text):
        occ.append(("placeholder", "[[" + m.group(1) + "]]", m.start(), m.end()))
    for m in _URL_RE.finditer(text):
        occ.append(("url", m.group(0).rstrip(".,;:!?"), m.start(), m.end()))
    # Numbers inside URLs or placeholder names are not "numbers" the agent says.
    scrubbed = _URL_RE.sub(_blank, text)
    scrubbed = _CURLY_VAR_RE.sub(_blank, scrubbed)
    scrubbed = _SQUARE_VAR_RE.sub(_blank, scrubbed)
    for m in _CURRENCY_RE.finditer(scrubbed):
        amount = (m.group(1) or m.group(2)).replace(",", "")
        occ.append(("currency", "₹" + amount, m.start(), m.end()))
    for m in _TIME_RE.finditer(scrubbed):
        if m.group(1):
            meridiem = "AM" if m.group(3).lower().startswith("a") else "PM"
            minutes = m.group(2)
            # "10 AM" == "10:00 AM" == "10am" == "10 a.m."; "10:30 AM" stays distinct.
            token = f"{int(m.group(1))}{':' + minutes if minutes and minutes != '00' else ''} {meridiem}"
        else:
            token = f"{int(m.group(4))}:{m.group(5)}"
        occ.append(("time", token, m.start(), m.end()))
    # Digits inside a time are guarded by the time token itself, so "10:00 AM"
    # does not also count "10" and "00" as loose numbers.
    scrubbed = _TIME_RE.sub(_blank, scrubbed)
    scrubbed = _LIST_MARKER_RE.sub(lambda m: m.group(1) + " " * len(m.group(2)), scrubbed)
    for m in _NUMBER_RE.finditer(scrubbed):
        occ.append(("number", m.group(0).replace(",", ""), m.start(), m.end()))
    occ.sort(key=lambda o: (o[2], TOKEN_KINDS.index(o[0])))
    return text, occ


def token_counts(text: str | None) -> dict[str, Counter]:
    """Multiset of tokens per kind."""
    out: dict[str, Counter] = {kind: Counter() for kind in TOKEN_KINDS}
    for kind, token, _s, _e in token_occurrences(text)[1]:
        out[kind][token] += 1
    return out


def extract_tokens(text: str | None) -> dict[str, set[str]]:
    """Distinct tokens per kind (use token_counts when repetitions matter)."""
    return {kind: set(counter) for kind, counter in token_counts(text).items()}


def strip_tokens(text: str | None) -> str:
    """Normalized text with every token removed, to tell whether a change is
    fully explained by token edits."""
    text = to_ascii_digits(normalize_text(text))
    for pattern in (_URL_RE, _CURLY_VAR_RE, _SQUARE_VAR_RE, _CURRENCY_RE, _TIME_RE, _NUMBER_RE):
        text = pattern.sub(" ", text)
    return collapse_ws(text)


@dataclass(frozen=True)
class TokenChange:
    kind: str
    token: str
    before: int          # occurrences in the input text
    after: int           # occurrences in the output text
    contexts: tuple[str, ...] = ()
    positions: tuple[int, ...] = ()   # where each context starts, for de-duplication

    @property
    def delta(self) -> int:
        return abs(self.before - self.after)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "token": self.token, "before": self.before, "after": self.after,
                "contexts": list(self.contexts)}


def _contexts(src: str, spans: list[tuple[int, int]], other: str, limit: int = 2) -> list[tuple[int, str]]:
    """Snippets around the occurrences most likely to be the changed ones: the
    ones whose neighbourhood no longer appears in the other text."""
    other_norm = collapse_ws(other)
    picked: list[tuple[int, str]] = []
    for s, e in spans:
        probe = collapse_ws(src[max(0, s - _PROBE): e + _PROBE])
        if probe not in other_norm:
            picked.append((s, _snippet(src, s, e)))
        if len(picked) >= limit:
            break
    return picked or [(s, _snippet(src, s, e)) for s, e in spans[:1]]


_PROBE, _SHOW = 30, 60


def _snippet(text: str, s: int, e: int) -> str:
    left, right = max(0, s - _SHOW), min(len(text), e + _SHOW)
    return ("…" if left else "") + collapse_ws(text[left:right]) + ("…" if right < len(text) else "")


def token_delta(before: str | None, after: str | None) -> tuple[list[TokenChange], list[TokenChange]]:
    """Compare token MULTISETS. (lost, added): a token is lost when it occurs
    fewer times in `after` than in `before`, added when it occurs more times.
    Reordering the same tokens changes nothing."""
    a_text, a_occ = token_occurrences(before)
    b_text, b_occ = token_occurrences(after)
    a_spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    b_spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for kind, token, s, e in a_occ:
        a_spans.setdefault((kind, token), []).append((s, e))
    for kind, token, s, e in b_occ:
        b_spans.setdefault((kind, token), []).append((s, e))
    lost, added = [], []
    for key in sorted(set(a_spans) | set(b_spans), key=lambda k: (TOKEN_KINDS.index(k[0]), k[1])):
        n_a, n_b = len(a_spans.get(key, [])), len(b_spans.get(key, []))
        if n_b < n_a:
            ctx = _contexts(a_text, a_spans[key], b_text)
            lost.append(TokenChange(key[0], key[1], n_a, n_b, tuple(c for _, c in ctx), tuple(p for p, _ in ctx)))
        elif n_b > n_a:
            ctx = _contexts(b_text, b_spans[key], a_text)
            added.append(TokenChange(key[0], key[1], n_a, n_b, tuple(c for _, c in ctx), tuple(p for p, _ in ctx)))
    return lost, added


# --- corruption --------------------------------------------------------------

def script_of(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    return name.split(" ")[0]  # LATIN, DEVANAGARI, TELUGU, KANNADA, TAMIL, ...


def script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text or "":
        script = script_of(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
    return counts


def mojibake_score(text: str) -> int:
    return len(_MOJIBAKE_RE.findall(text or ""))


def corruption_reasons(before: str | None, after: str | None) -> list[str]:
    """Signs that `after` is a damaged copy of `before` (not a rewrite)."""
    before, after = before or "", after or ""
    reasons = []
    if after.count("\ufffd") > before.count("\ufffd"):
        reasons.append("contains U+FFFD replacement characters (bytes were lost in decoding)")
    if mojibake_score(after) > mojibake_score(before):
        reasons.append("contains mojibake (UTF-8 text decoded with the wrong encoding)")
    a_scripts, b_scripts = script_counts(before), script_counts(after)
    for script, count in sorted(a_scripts.items()):
        if script != "LATIN" and count >= 3 and b_scripts.get(script, 0) == 0 and after.strip():
            reasons.append(f"all {script.title()} script text is gone")
    return reasons
