"""Text normalisation, tokenisation and sentence splitting for English and Japanese."""

import re
import unicodedata

# Hiragana, katakana, CJK ideographs and half-width katakana (after NFKC this becomes full-width).
_CJK = "぀-ヿ㐀-䶿一-鿿"
_TOKEN_RE = re.compile(rf"[a-z0-9]+(?:[.\-/][a-z0-9]+)*|[{_CJK}]+")
_CJK_RE = re.compile(rf"[{_CJK}]")

# English labels split sentences at ". " followed by a capital, digit or bullet.
# Japanese text has no spaces, so split right after 。！？.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•(\[])|(?<=[。！？])|\s+(?=•\s)")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "my",
        "me",
        "your",
        "about",
        "any",
        "there",
        "tell",
    ]
)


def normalize(text: str) -> str:
    """NFKC folds full-width ASCII and half-width katakana, then lowercase."""
    return unicodedata.normalize("NFKC", text).lower()


def is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def tokenize(text: str) -> list[str]:
    """Word tokens for English, character bigrams for Japanese runs.

    Bigrams need no dictionary and still match katakana drug names such as
    タグリッソ exactly, which is what keyword search is for here.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalize(text)):
        tok = match.group()
        if _CJK_RE.match(tok):
            if len(tok) == 1:
                tokens.append(tok)
            else:
                tokens.extend(tok[i : i + 2] for i in range(len(tok) - 1))
        elif tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_NUMBER_WORDS = {
    "once": "1",
    "one": "1",
    "twice": "2",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b")


def numbers_in(text: str, words: bool = False) -> set[str]:
    """Digits in the text. With words=True, English number words count too ("once daily" gives 1),
    so a Japanese answer saying 1日1回 can be checked against a label that says "once daily"."""
    norm = normalize(text)
    found = set(_NUMBER_RE.findall(norm))
    if words:
        found |= {_NUMBER_WORDS[w] for w in _NUMBER_WORD_RE.findall(norm)}
    return found
