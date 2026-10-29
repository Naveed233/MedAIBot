import pytest

from medaibot.embeddings import HashingEmbedder
from medaibot.index import BM25, HybridIndex, IndexMismatch, rrf


def test_bm25_ranks_the_document_with_the_rare_term_first():
    bm25 = BM25([["take", "dose", "daily"], ["avoid", "grapefruit", "juice"], ["dose", "daily", "daily"]])
    scores = bm25.scores(["grapefruit"])
    assert scores.argmax() == 1
    assert scores[0] == scores[2] == 0


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert max(fused, key=fused.get) == 1
    assert fused[4] < fused[3]


def test_search_filters_by_drug_and_section(index):
    hits = index.search("recommended dosage", k=5, drug="farxiga")
    assert hits and all(h.chunk.drug == "farxiga" for h in hits)
    hits = index.search("dose", k=5, drug="tagrisso", section="contraindications")
    assert [h.chunk.chunk_id for h in hits] == ["tagrisso:contraindications:000"]


def test_modes_report_their_own_ranks(index):
    hybrid = index.search("strong CYP3A inducer", k=3, drug="tagrisso", mode="hybrid")
    assert hybrid[0].bm25_rank is not None or hybrid[0].dense_rank is not None
    bm25_only = index.search("strong CYP3A inducer", k=3, drug="tagrisso", mode="bm25")
    assert all(h.dense_rank is None for h in bm25_only)


def test_save_and_load_round_trip(tmp_path, index):
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, HashingEmbedder())
    query = "difficulty swallowing"
    assert [h.chunk.chunk_id for h in loaded.search(query)] == [h.chunk.chunk_id for h in index.search(query)]


def test_loading_with_a_different_embedder_fails_loudly(tmp_path, index):
    index.save(tmp_path)
    with pytest.raises(IndexMismatch):
        HybridIndex.load(tmp_path, HashingEmbedder(dim=128))
