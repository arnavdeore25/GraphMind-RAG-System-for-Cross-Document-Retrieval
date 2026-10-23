"""Shared test setup.

Unit tests run fully offline: the `llm` module (Gemini client) is replaced by a
stub, and Neo4j / Chroma are replaced by in-memory fakes inside each test.
Set GRAPHMIND_LIVE=1 to run the `live` tests against real services.
"""
import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("GEMINI_API_KEY", "test-key")
LIVE = os.getenv("GRAPHMIND_LIVE") == "1"

if not LIVE:
    stub = types.ModuleType("llm")
    stub.generate = lambda *a, **k: ""
    stub.generate_json = lambda *a, **k: {}
    sys.modules["llm"] = stub


def pytest_collection_modifyitems(config, items):
    if LIVE:
        return
    skip = pytest.mark.skip(reason="set GRAPHMIND_LIVE=1 to run live tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def real_llm():
    """The real llm.py (not the stub), with a fake Gemini client injected per test."""
    spec = importlib.util.spec_from_file_location("real_llm", os.path.join(ROOT, "llm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- reusable fakes ----------
class FakeDB:
    """Stands in for Chroma: returns the first k configured chunk ids."""
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def similarity_search_with_score(self, query, k):
        self.calls.append((query, k))
        return [(types.SimpleNamespace(metadata={"chunk_id": i}), 0.1) for i in self.ids[:k]]


class FakeGraph:
    """Stands in for GraphStore (read side)."""
    def __init__(self, chunks=(), seeds=(), from_vec=(), triples=(), mentions=()):
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.seeds, self.from_vec = list(seeds), list(from_vec)
        self.triples, self.mentions = list(triples), list(mentions)
        self.expand_calls = []

    def find_entities(self, names):
        return list(self.seeds)

    def entities_of_chunks(self, ids):
        return list(self.from_vec)

    def expand(self, seed_ids, hops=2):
        self.expand_calls.append((list(seed_ids), hops))
        return [dict(t) for t in self.triples]

    def chunks_mentioning(self, seed_ids):
        return [dict(m) for m in self.mentions]

    def get_chunks(self, ids):
        return [dict(self.chunks[i]) for i in ids if i in self.chunks]


def chunk(cid, source="a.txt", text=None):
    return {"chunk_id": cid, "source": source, "text": text or f"text of {cid}"}


def triple(h, r, t, cid, source="a.txt"):
    return {"head": h, "rel": r, "tail": t, "chunk_id": cid, "source": source}
