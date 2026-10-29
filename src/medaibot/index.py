"""Hybrid retrieval: BM25 keyword scores and FAISS dense scores, merged with Reciprocal Rank Fusion."""

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import faiss
import numpy as np

from medaibot.embeddings import Embedder
from medaibot.models import Chunk, SearchHit
from medaibot.text import tokenize

Mode = Literal["bm25", "dense", "hybrid"]

INDEX_FORMAT = 1


class IndexMismatch(Exception):
    """The saved index was built with a different embedder or format than the one loading it."""


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(documents)
        self.doc_len = np.array([len(d) for d in documents], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: each list adds 1 / (k + rank). Only ranks matter, so BM25
    scores and cosine similarities never have to be put on the same scale."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] += 1.0 / (k + rank)
    return dict(fused)


@dataclass
class HybridIndex:
    chunks: list[Chunk]
    embedder: Embedder
    faiss_index: faiss.Index
    bm25: BM25
    meta: dict[str, object]

    @classmethod
    def build(
        cls, chunks: list[Chunk], embedder: Embedder, label_versions: dict[str, str] | None = None
    ) -> "HybridIndex":
        texts = [c.index_text for c in chunks]
        vectors = embedder.embed_documents(texts)
        faiss_index = faiss.IndexFlatIP(embedder.dim)
        if len(vectors):
            faiss_index.add(vectors)
        meta: dict[str, object] = {
            "format": INDEX_FORMAT,
            "embedder": embedder.name,
            "dim": embedder.dim,
            "n_chunks": len(chunks),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "label_versions": label_versions or {},
        }
        return cls(chunks, embedder, faiss_index, BM25([tokenize(t) for t in texts]), meta)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        faiss.write_index(self.faiss_index, str(directory / "dense.faiss"))
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "HybridIndex":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != INDEX_FORMAT:
            raise IndexMismatch(f"index format {meta.get('format')} != {INDEX_FORMAT}; rebuild with `medaibot ingest`")
        if meta.get("embedder") != embedder.name:
            raise IndexMismatch(
                f"index was built with {meta.get('embedder')!r} but the current embedder is {embedder.name!r}; "
                "rebuild with `medaibot ingest`"
            )
        with (directory / "chunks.jsonl").open(encoding="utf-8") as fh:
            chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        faiss_index = faiss.read_index(str(directory / "dense.faiss"))
        bm25 = BM25([tokenize(c.index_text) for c in chunks])
        return cls(chunks, embedder, faiss_index, bm25, meta)

    @property
    def drugs(self) -> set[str]:
        return {c.drug for c in self.chunks}

    def sections_for(self, drug: str) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            if c.drug == drug:
                seen.setdefault(c.section)
        return list(seen)

    def get_section(self, drug: str, section: str) -> list[Chunk]:
        return [c for c in self.chunks if c.drug == drug and c.section == section]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def _allowed(self, drug: str | None, section: str | None) -> np.ndarray:
        allowed = np.ones(len(self.chunks), dtype=bool)
        if drug:
            allowed &= np.array([c.drug == drug for c in self.chunks])
        if section:
            allowed &= np.array([c.section == section for c in self.chunks])
        return allowed

    def _bm25_ranking(self, query: str, allowed: np.ndarray, depth: int) -> list[int]:
        scores = self.bm25.scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")
        return [int(i) for i in order if allowed[i] and scores[i] > 0][:depth]

    def _dense_ranking(self, query: str, allowed: np.ndarray, depth: int) -> tuple[list[int], dict[int, float]]:
        if self.faiss_index.ntotal == 0:
            return [], {}
        q = self.embedder.embed_query(query)[None, :]
        # Exact search over everything, then filter. Fine for a few thousand chunks;
        # a larger corpus would use FAISS ID selectors or a vector DB with metadata filters.
        sims, ids = self.faiss_index.search(q, self.faiss_index.ntotal)
        ranking: list[int] = []
        sim_by_id: dict[int, float] = {}
        for sim, i in zip(sims[0], ids[0], strict=True):
            if i >= 0 and allowed[i]:
                ranking.append(int(i))
                sim_by_id[int(i)] = float(sim)
                if len(ranking) >= depth:
                    break
        return ranking, sim_by_id

    def search(
        self,
        query: str,
        k: int = 5,
        drug: str | None = None,
        section: str | None = None,
        mode: Mode = "hybrid",
        rrf_k: int = 60,
        depth: int = 50,
    ) -> list[SearchHit]:
        allowed = self._allowed(drug, section)
        bm25_rank = self._bm25_ranking(query, allowed, depth) if mode in ("bm25", "hybrid") else []
        dense_rank, sims = self._dense_ranking(query, allowed, depth) if mode in ("dense", "hybrid") else ([], {})
        bm25_pos = {doc: r for r, doc in enumerate(bm25_rank, start=1)}
        dense_pos = {doc: r for r, doc in enumerate(dense_rank, start=1)}

        if mode == "bm25":
            scores = self.bm25.scores(tokenize(query))
            ordered = [(doc, float(scores[doc])) for doc in bm25_rank]
        elif mode == "dense":
            ordered = [(doc, sims[doc]) for doc in dense_rank]
        else:
            fused = rrf([bm25_rank, dense_rank], k=rrf_k)
            ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

        return [
            SearchHit(
                chunk=self.chunks[doc],
                score=round(score, 6),
                bm25_rank=bm25_pos.get(doc),
                dense_rank=dense_pos.get(doc),
            )
            for doc, score in ordered[:k]
        ]
