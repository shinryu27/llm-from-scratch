"""Stage 1a — Tokenization.

Two interchangeable tokenizers:

* ``BPETokenizer``      — a byte-pair-encoding tokenizer trainable from scratch,
                          so the merge algorithm itself is transparent.
* ``get_gpt2_tokenizer`` — OpenAI's pretrained GPT-2 BPE via ``tiktoken``
                          (50,257-token vocabulary), when available.

Both expose ``encode(text) -> list[int]`` and ``decode(ids) -> str``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

END_OF_TEXT = "<|endoftext|>"

# GPT-2-style pre-tokenization: split into words w/ leading space, numbers, punct.
_PRETOKEN_RE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+")


class BPETokenizer:
    """A minimal but complete byte-pair-encoding tokenizer.

    Training: start from single characters and iteratively merge the most
    frequent adjacent symbol pair until ``vocab_size`` is reached — the same
    algorithm used (at byte level) by GPT-2.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[str, str], int] = {}   # pair -> merge rank
        self.vocab: dict[str, int] = {}                # token string -> id
        self.inverse_vocab: dict[int, str] = {}

    # ------------------------------------------------------------------ train
    def train(self, text: str, vocab_size: int = 1000) -> None:
        words = [tuple(w) + ("</w>",) for w in _PRETOKEN_RE.findall(text)]
        word_freq = Counter(words)

        # Base vocabulary: every symbol occurring in the corpus.
        symbols = {ch for word in word_freq for ch in word}
        self.vocab = {s: i for i, s in enumerate(sorted(symbols))}

        while len(self.vocab) < vocab_size:
            pair_freq: Counter = Counter()
            for word, freq in word_freq.items():
                for pair in zip(word, word[1:]):
                    pair_freq[pair] += freq
            if not pair_freq:
                break
            best = pair_freq.most_common(1)[0][0]
            self.merges[best] = len(self.merges)
            merged_symbol = best[0] + best[1]
            self.vocab[merged_symbol] = len(self.vocab)
            word_freq = Counter(
                {self._merge_word(w, best): f for w, f in word_freq.items()}
            )

        self.vocab.setdefault(END_OF_TEXT, len(self.vocab))
        self.inverse_vocab = {i: s for s, i in self.vocab.items()}

    @staticmethod
    def _merge_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ----------------------------------------------------------- encode/decode
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in _PRETOKEN_RE.findall(text):
            symbols = tuple(w) + ("</w>",)
            # Apply learned merges in rank order.
            while len(symbols) > 1:
                pairs = list(zip(symbols, symbols[1:]))
                ranked = [(self.merges[p], p) for p in pairs if p in self.merges]
                if not ranked:
                    break
                _, best = min(ranked)
                symbols = self._merge_word(symbols, best)
            for s in symbols:
                if s in self.vocab:
                    ids.append(self.vocab[s])
                else:  # unseen symbol -> fall back to characters
                    ids.extend(self.vocab[c] for c in s if c in self.vocab)
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.inverse_vocab[i] for i in ids)
        return text.replace("</w>", "")

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        payload = {
            "vocab": self.vocab,
            "merges": [[a, b, r] for (a, b), r in self.merges.items()],
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.vocab = payload["vocab"]
        tok.merges = {(a, b): r for a, b, r in payload["merges"]}
        tok.inverse_vocab = {i: s for s, i in tok.vocab.items()}
        return tok


def get_gpt2_tokenizer():
    """Return OpenAI's GPT-2 BPE tokenizer (requires ``tiktoken``)."""
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def get_tokenizer(train_text: str | None = None, vocab_size: int = 1000):
    """Prefer the pretrained GPT-2 BPE; fall back to a from-scratch BPE."""
    try:
        return get_gpt2_tokenizer()
    except Exception:
        if train_text is None:
            raise RuntimeError(
                "tiktoken unavailable and no training text supplied "
                "for the fallback BPE tokenizer."
            )
        tok = BPETokenizer()
        tok.train(train_text, vocab_size=vocab_size)
        return tok
