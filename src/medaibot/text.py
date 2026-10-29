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


def numbers_in(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(normalize(text)))
