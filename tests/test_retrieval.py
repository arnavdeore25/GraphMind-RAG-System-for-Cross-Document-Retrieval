"""retrieval.py: RRF fusion and vector / graph / hybrid modes (no services needed)."""
import pytest

import retrieval
from retrieval import Retriever, rrf
from conftest import FakeDB, FakeGraph, chunk, triple


def make(db_ids, graph, monkeypatch, query_entities=("X",)):
    r = Retriever.__new__(Retriever)
    r.db = FakeDB(db_ids)
    r.graph = graph
    monkeypatch.setattr(retrieval, "extract_query_entities", lambda q: list(query_entities))
    return r


# ---------------- rrf ----------------
class TestRRF:
    def test_empty_inputs(self):
        assert rrf([]) == []
        assert rrf([[], []]) == []

    def test_single_list_keeps_order(self):
        assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]

    def test_item_in_both_lists_wins(self):
        assert rrf([["a", "b", "c"], ["d", "b"]])[0] == "b"

    def test_score_formula(self):
        scores_two_lists = 2 / 61          # rank 0 in two lists, k=60
        scores_one_list = 1 / 61
        assert scores_two_lists > scores_one_list
        assert rrf([["x"], ["x"]]) == ["x"]

    def test_no_duplicates_in_output(self):
        out = rrf([["a", "b"], ["b", "a"], ["a"]])
        assert len(out) == len(set(out)) == 2


# ---------------- modes ----------------
class TestModes:
    def test_invalid_mode_raises(self, monkeypatch):
        r = make([], FakeGraph(), monkeypatch)
        with pytest.raises(ValueError):
            r.retrieve("q", mode="semantic")

    def test_vector_mode_marks_via_and_preserves_order(self, monkeypatch):
        g = FakeGraph(chunks=[chunk("a"), chunk("b"), chunk("c")])
        r = make(["b", "a", "c"], g, monkeypatch)
        res = r.retrieve("q", mode="vector", k=3)
        assert [c["chunk_id"] for c in res.chunks] == ["b", "a", "c"]
        assert {c["via"] for c in res.chunks} == {"vector"}
        assert res.triples == [] and res.mode == "vector"
        assert r.db.calls == [("q", 3)]

    def test_vector_mode_drops_chunks_missing_from_neo4j(self, monkeypatch):
        """README note: chunk text is read from Neo4j even in vector mode, so a
        --no-graph ingestion makes every mode return nothing."""
        r = make(["a", "b"], FakeGraph(chunks=[]), monkeypatch)
        assert r.retrieve("q", mode="vector").chunks == []

    def test_graph_mode_no_entities_linked_returns_empty(self, monkeypatch):
        r = make(["a"], FakeGraph(seeds=[], chunks=[chunk("a")]), monkeypatch)
        res = r.retrieve("q", mode="graph")
        assert res.chunks == [] and res.triples == []
        assert res.debug["linked_seeds"] == []

    def test_graph_mode_ranking_triples_count_double_plus_mentions(self, monkeypatch):
        g = FakeGraph(
            seeds=["e1"],
            chunks=[chunk("c1"), chunk("c2"), chunk("c3")],
            triples=[triple("A", "R", "B", "c1"), triple("B", "R", "C", "c1"), triple("C", "R", "D", "c2")],
            mentions=[{"chunk_id": "c2", "n": 3}, {"chunk_id": "c3", "n": 1}],
        )
        r = make([], g, monkeypatch)
        # scores: c1 = 2+2 = 4, c2 = 2+3 = 5, c3 = 1
        res = r.retrieve("q", mode="graph", k=3)
        assert [c["chunk_id"] for c in res.chunks] == ["c2", "c1", "c3"]
        assert {c["via"] for c in res.chunks} == {"graph"}

    def test_graph_mode_respects_k_and_hops(self, monkeypatch):
        g = FakeGraph(seeds=["e1"], chunks=[chunk("c1"), chunk("c2")],
                      mentions=[{"chunk_id": "c1", "n": 2}, {"chunk_id": "c2", "n": 1}])
        r = make([], g, monkeypatch)
        res = r.retrieve("q", mode="graph", k=1, hops=3)
        assert [c["chunk_id"] for c in res.chunks] == ["c1"]
        assert g.expand_calls == [(["e1"], 3)]

    def test_hybrid_fuses_and_labels_sources(self, monkeypatch):
        g = FakeGraph(
            seeds=["e1"], from_vec=["e2"],
            chunks=[chunk(i) for i in "abcd"],
            triples=[triple("A", "R", "B", "b")],
            mentions=[{"chunk_id": "d", "n": 1}],
        )
        r = make(["a", "b", "c"], g, monkeypatch)
        res = r.retrieve("q", mode="hybrid", k=5)
        # graph ranking = [d(1), b(2)] -> b(2) first;  fused: b (both) > a,d (tie, a first) > c
        assert [c["chunk_id"] for c in res.chunks] == ["b", "a", "d", "c"]
        via = {c["chunk_id"]: c["via"] for c in res.chunks}
        assert via == {"b": "both", "a": "vector", "d": "graph", "c": "vector"}

    def test_hybrid_expands_from_query_and_vector_seeds_deduplicated(self, monkeypatch):
        g = FakeGraph(seeds=["e1", "e2"], from_vec=["e2", "e3"], chunks=[chunk("a")])
        r = make(["a"], g, monkeypatch)
        res = r.retrieve("q", mode="hybrid")
        assert g.expand_calls[0][0] == ["e1", "e2", "e3"]
        assert res.debug["linked_seeds"] == ["e1", "e2"]
        assert res.debug["seeds_from_vector_hits"] == ["e2", "e3"]

    def test_hybrid_without_graph_seeds_falls_back_to_vector(self, monkeypatch):
        g = FakeGraph(seeds=[], from_vec=[], chunks=[chunk("a"), chunk("b")])
        r = make(["a", "b"], g, monkeypatch)
        res = r.retrieve("q", mode="hybrid")
        assert [c["chunk_id"] for c in res.chunks] == ["a", "b"]
        assert {c["via"] for c in res.chunks} == {"vector"}
        assert g.expand_calls == []

    def test_hybrid_returns_at_most_k_plus_3(self, monkeypatch):
        vec = [f"v{i}" for i in range(5)]
        gids = [f"g{i}" for i in range(5)]
        g = FakeGraph(seeds=["e1"], chunks=[chunk(i) for i in vec + gids],
                      mentions=[{"chunk_id": gid, "n": 10 - i} for i, gid in enumerate(gids)])
        r = make(vec, g, monkeypatch)
        assert len(r.retrieve("q", mode="hybrid", k=5).chunks) == 8

    def test_triples_only_for_returned_chunks(self, monkeypatch):
        g = FakeGraph(seeds=["e1"], chunks=[chunk("c1")],
                      triples=[triple("A", "R", "B", "c1"), triple("X", "R", "Y", "not_returned")],
                      mentions=[{"chunk_id": "c1", "n": 1}])
        r = make([], g, monkeypatch)
        res = r.retrieve("q", mode="graph", k=1)
        assert [t["chunk_id"] for t in res.triples] == ["c1"]

    def test_triples_capped_at_25(self, monkeypatch):
        many = [triple(f"h{i}", "R", f"t{i}", "c1") for i in range(40)]
        g = FakeGraph(seeds=["e1"], chunks=[chunk("c1")], triples=many,
                      mentions=[{"chunk_id": "c1", "n": 1}])
        res = make([], g, monkeypatch).retrieve("q", mode="graph")
        assert len(res.triples) == 25

    def test_latency_recorded(self, monkeypatch):
        g = FakeGraph(chunks=[chunk("a")])
        res = make(["a"], g, monkeypatch).retrieve("q", mode="vector")
        assert isinstance(res.latency, float) and res.latency >= 0
