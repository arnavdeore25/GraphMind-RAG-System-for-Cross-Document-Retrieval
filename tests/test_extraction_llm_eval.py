"""graph_extraction.py, llm.py (real module, fake Gemini client) and evaluate.py."""
import json
import types

import pandas as pd
import pytest

import graph_extraction
from retrieval import Retrieved
from conftest import chunk


# ---------------- graph_extraction ----------------
class TestExtraction:
    def test_text_truncated_to_4000_chars(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(graph_extraction, "generate_json", lambda p: seen.setdefault("p", p) and {})
        graph_extraction.extract_chunk("x" * 5000 + "TAIL")
        assert "TAIL" not in seen["p"] and "x" * 4000 in seen["p"]

    def test_missing_keys_default_to_empty_lists(self, monkeypatch):
        monkeypatch.setattr(graph_extraction, "generate_json", lambda p: {})
        assert graph_extraction.extract_chunk("t") == {"entities": [], "relations": []}

    def test_query_entities_returned_as_list(self, monkeypatch):
        monkeypatch.setattr(graph_extraction, "generate_json", lambda p: {"entities": ["ZeroProof", "Meera Iyer"]})
        assert graph_extraction.extract_query_entities("who?") == ["ZeroProof", "Meera Iyer"]

    def test_query_entities_missing_key_gives_empty_list(self, monkeypatch):
        monkeypatch.setattr(graph_extraction, "generate_json", lambda p: {})
        assert graph_extraction.extract_query_entities("who?") == []

    @pytest.mark.xfail(reason="BUG PROBE: if Gemini returns a JSON list instead of an object, "
                              ".get() raises and chat crashes at query time", strict=False)
    def test_query_entities_survives_non_dict_json(self, monkeypatch):
        monkeypatch.setattr(graph_extraction, "generate_json", lambda p: ["ZeroProof"])
        assert graph_extraction.extract_query_entities("who?") == []


class FakeStore:
    def __init__(self):
        self.chunks, self.extractions, self.setup_called = [], [], False

    def setup(self): self.setup_called = True
    def upsert_chunk(self, cid, source, text): self.chunks.append((cid, source, text))
    def upsert_extraction(self, cid, source, ents, rels): self.extractions.append((cid, source, ents, rels))
    def stats(self): return {"chunks": len(self.chunks), "entities": 0, "relations": 0}


class TestBuildGraph:
    def docs(self, n, with_source=True):
        return [(f"c{i}", types.SimpleNamespace(page_content=f"t{i}", metadata={"source": "a.txt"} if with_source else {}))
                for i in range(n)]

    def test_all_chunks_stored_and_setup_called(self, monkeypatch):
        monkeypatch.setattr(graph_extraction, "generate_json", lambda p: {"entities": [{"name": "A"}], "relations": []})
        st = FakeStore()
        graph_extraction.build_graph(self.docs(5), st)
        assert st.setup_called and len(st.chunks) == 5 and len(st.extractions) == 5

    def test_one_failed_extraction_does_not_abort_the_run(self, monkeypatch, capsys):
        def flaky(prompt):
            if "t2" in prompt:
                raise RuntimeError("rate limit")
            return {"entities": [{"name": "A"}], "relations": []}
        monkeypatch.setattr(graph_extraction, "generate_json", flaky)
        st = FakeStore()
        graph_extraction.build_graph(self.docs(4), st)
        assert len(st.extractions) == 4
        failed = next(e for e in st.extractions if e[0] == "c2")
        assert failed[2] == [] and failed[3] == []
        assert "[extract fail] c2" in capsys.readouterr().out

    def test_missing_source_becomes_unknown(self, monkeypatch):
        monkeypatch.setattr(graph_extraction, "generate_json", lambda p: {})
        st = FakeStore()
        graph_extraction.build_graph(self.docs(1, with_source=False), st)
        assert st.chunks[0][1] == "unknown" and st.extractions[0][1] == "unknown"


# ---------------- llm.py ----------------
class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0
        self.models = types.SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return types.SimpleNamespace(text=o)


class TestLLM:
    def use(self, real_llm, monkeypatch, outcomes):
        c = FakeClient(outcomes)
        monkeypatch.setattr(real_llm, "client", c)
        monkeypatch.setattr(real_llm.time, "sleep", lambda s: None)
        return c

    def test_generate_strips_text(self, real_llm, monkeypatch):
        self.use(real_llm, monkeypatch, ["  hi \n"])
        assert real_llm.generate("p") == "hi"

    def test_generate_none_text_becomes_empty_string(self, real_llm, monkeypatch):
        self.use(real_llm, monkeypatch, [None])
        assert real_llm.generate("p") == ""

    def test_generate_retries_then_succeeds(self, real_llm, monkeypatch):
        c = self.use(real_llm, monkeypatch, [RuntimeError("429"), RuntimeError("429"), "ok"])
        assert real_llm.generate("p") == "ok" and c.calls == 3

    def test_generate_raises_after_all_retries(self, real_llm, monkeypatch):
        c = self.use(real_llm, monkeypatch, [RuntimeError("down")] * 4)
        with pytest.raises(RuntimeError):
            real_llm.generate("p")
        assert c.calls == 4

    def test_json_plain(self, real_llm, monkeypatch):
        self.use(real_llm, monkeypatch, ['{"a": 1}'])
        assert real_llm.generate_json("p") == {"a": 1}

    def test_json_inside_markdown_fence(self, real_llm, monkeypatch):
        self.use(real_llm, monkeypatch, ['```json\n{"a": 1}\n```'])
        assert real_llm.generate_json("p") == {"a": 1}

    def test_invalid_json_returns_empty_dict(self, real_llm, monkeypatch):
        self.use(real_llm, monkeypatch, ["Sorry, I cannot do that"])
        assert real_llm.generate_json("p") == {}

    def test_empty_response_returns_empty_dict(self, real_llm, monkeypatch):
        self.use(real_llm, monkeypatch, [""])
        assert real_llm.generate_json("p") == {}

    @pytest.mark.xfail(reason="BUG PROBE: valid JSON that is not an object (e.g. a list) is returned as-is; "
                              "callers then crash on .get()", strict=False)
    def test_json_list_is_normalised_to_dict(self, real_llm, monkeypatch):
        self.use(real_llm, monkeypatch, ['["a", "b"]'])
        assert isinstance(real_llm.generate_json("p"), dict)


# ---------------- evaluate.py ----------------
class TestEvaluate:
    def test_metrics_and_csv(self, tmp_path, monkeypatch, capsys):
        import evaluate

        by_mode = {
            "vector": [chunk("1", "teams.csv")],
            "graph": [chunk("2", "mentors.csv")],
            "hybrid": [chunk("1", "teams.csv"), chunk("2", "mentors.csv")],
        }

        class FakeRetriever:
            def retrieve(self, q, mode):
                return Retrieved(mode, by_mode[mode], [], 0.01)

        monkeypatch.setattr(evaluate, "Retriever", FakeRetriever)
        monkeypatch.setattr(evaluate, "answer", lambda q, r: f"ans-{r.mode}")
        monkeypatch.setattr(evaluate, "generate_json", lambda p: {"correct": 1 if "ans-hybrid" in p else 0})

        data = [
            {"question": "q1", "answer": "a", "gold_sources": ["teams.csv", "mentors.csv"], "hops": 2},
            {"question": "q2", "answer": "b"},                         # no gold_sources, no hops
        ]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        evaluate.run(str(path))

        df = pd.read_csv(tmp_path / "eval_results.csv")
        assert len(df) == 6 and set(df["mode"]) == {"vector", "graph", "hybrid"}
        q1 = df[df["hops"] == 2].set_index("mode")
        assert q1.loc["vector", "evidence_recall"] == 0.5 and q1.loc["vector", "full_coverage"] == 0
        assert q1.loc["hybrid", "evidence_recall"] == 1.0 and q1.loc["hybrid", "full_coverage"] == 1
        assert q1.loc["hybrid", "correct"] == 1 and q1.loc["vector", "correct"] == 0
        assert q1.loc["hybrid", "context_chars"] == sum(len(c["text"]) for c in by_mode["hybrid"])
        q2 = df[df["hops"] == 1]                                        # default hops
        assert q2["evidence_recall"].isna().all()                       # no gold -> not scored
        assert "=== Overall ===" in capsys.readouterr().out
