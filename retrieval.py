"""Vector, graph and hybrid retrieval behind one interface."""
import time
from dataclasses import dataclass, field

from config import TOP_K, GRAPH_HOPS
from graph_extraction import extract_query_entities
from graph_store import GraphStore
from ingestion import get_vectorstore


@dataclass
class Retrieved:
    mode: str
    chunks: list = field(default_factory=list)   # [{chunk_id, source, text, via}]
    triples: list = field(default_factory=list)  # [{head, rel, tail, chunk_id, source}]
    latency: float = 0.0


def rrf(rank_lists, k=60):
    scores = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self):
        self.db = get_vectorstore()
        self.graph = GraphStore()

    # ---- vector ----
    def _vector(self, query, k):
        res = self.db.similarity_search_with_score(query, k=k)
        return [d.metadata["chunk_id"] for d, _ in res]

    # ---- graph ----
    def _graph(self, seed_entity_ids, hops, k):
        if not seed_entity_ids:
            return [], []
        triples = self.graph.expand(seed_entity_ids, hops=hops)
        counts = {}
        for t in triples:                                   # chunks that evidence traversed edges
            counts[t["chunk_id"]] = counts.get(t["chunk_id"], 0) + 2
        for r in self.graph.chunks_mentioning(seed_entity_ids):  # chunks mentioning seeds
            counts[r["chunk_id"]] = counts.get(r["chunk_id"], 0) + r["n"]
        ranked = sorted(counts, key=counts.get, reverse=True)[:k]
        return ranked, triples

    def retrieve(self, query, mode="hybrid", k=TOP_K, hops=GRAPH_HOPS) -> Retrieved:
        t0 = time.perf_counter()
        via = {}
        triples = []
        if mode == "vector":
            ids = self._vector(query, k)
            via = {i: "vector" for i in ids}
        elif mode == "graph":
            seeds = self.graph.find_entities(extract_query_entities(query))
            ids, triples = self._graph(seeds, hops, k)
            via = {i: "graph" for i in ids}
        elif mode == "hybrid":
            vec_ids = self._vector(query, k)
            seeds = self.graph.find_entities(extract_query_entities(query))
            seeds += self.graph.entities_of_chunks(vec_ids)      # expand from vector hits too
            graph_ids, triples = self._graph(list(dict.fromkeys(seeds)), hops, k)
            ids = rrf([vec_ids, graph_ids])[:k + 3]
            via = {i: ("both" if i in vec_ids and i in graph_ids else "vector" if i in vec_ids else "graph") for i in ids}
        else:
            raise ValueError(mode)

        chunks = self.graph.get_chunks(ids)
        for c in chunks:
            c["via"] = via.get(c["chunk_id"], mode)
        kept = {c["chunk_id"] for c in chunks}
        triples = [t for t in triples if t["chunk_id"] in kept][:25]  # only facts we can cite
        return Retrieved(mode, chunks, triples, time.perf_counter() - t0)
