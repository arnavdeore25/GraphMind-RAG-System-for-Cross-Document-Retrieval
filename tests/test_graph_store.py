"""graph_store.py without a database: _run is replaced by a recorder."""
import pytest

from graph_store import GraphStore, _lucene_escape, norm


def store(rows=None):
    s = GraphStore.__new__(GraphStore)          # skip driver creation
    s.calls = []

    def fake_run(cypher, **params):
        s.calls.append((cypher, params))
        return rows(cypher, params) if callable(rows) else (rows or [])
    s._run = fake_run
    return s


class TestHelpers:
    def test_norm_lowercases_and_collapses_whitespace(self):
        assert norm("  Dr.   Meera \n Iyer ") == "dr. meera iyer"

    def test_norm_is_idempotent(self):
        assert norm(norm("  A  B ")) == norm("  A  B ")

    @pytest.mark.parametrize("raw,expected", [
        ("C++", r"C\+\+"),
        ("a:b", r"a\:b"),
        ("(x)", r"\(x\)"),
        ('say "hi"', r'say \"hi\"'),
        ("plain text", "plain text"),
    ])
    def test_lucene_escape(self, raw, expected):
        assert _lucene_escape(raw) == expected


class TestUpsertExtraction:
    def test_blank_entities_dropped_and_names_stripped(self):
        s = store()
        s.upsert_extraction("c1", "a.txt", [{"name": "  Alice "}, {"name": "   "}, {"name": ""}], [])
        ents = s.calls[0][1]["ents"]
        assert [e["id"] for e in ents] == ["alice"] and ents[0]["name"] == "Alice"
        assert ents[0]["type"] == "Other"           # default type

    def test_no_relations_means_no_relation_query(self):
        s = store()
        s.upsert_extraction("c1", "a.txt", [{"name": "A"}], [])
        assert len(s.calls) == 1

    def test_relation_type_normalised(self):
        s = store()
        s.upsert_extraction("c1", "a.txt", [{"name": "A"}, {"name": "B"}],
                            [{"head": "A", "relation": "works at", "tail": "B"}])
        rel = s.calls[1][1]["rels"][0]
        assert rel == {"h": "a", "t": "b", "type": "WORKS_AT"}
        assert s.calls[1][1]["cid"] == "c1" and s.calls[1][1]["src"] == "a.txt"

    def test_incomplete_relations_skipped(self):
        s = store()
        s.upsert_extraction("c1", "a.txt", [{"name": "A"}], [
            {"head": "A", "relation": "", "tail": "B"},
            {"head": "", "relation": "R", "tail": "B"},
            {"head": "A", "relation": "R", "tail": "  "},
        ])
        assert len(s.calls) == 1

    def test_relation_endpoints_missing_from_entities_are_created(self):
        s = store()
        s.upsert_extraction("c1", "a.txt", [{"name": "A"}],
                            [{"head": "A", "relation": "R", "tail": "Ghost"}])
        ids = {e["id"] for e in s.calls[0][1]["ents"]}
        assert ids == {"a", "ghost"}

    def test_same_entity_different_case_merges_to_one_id(self):
        s = store()
        s.upsert_extraction("c1", "a.txt", [{"name": "OpenAI"}, {"name": "openai"}], [])
        assert len({e["id"] for e in s.calls[0][1]["ents"]}) == 1

    @pytest.mark.xfail(reason="BUG PROBE: LLM may return non-string names; .strip() raises AttributeError", strict=False)
    def test_non_string_entity_name_does_not_crash(self):
        store().upsert_extraction("c1", "a.txt", [{"name": 2024}], [])

    @pytest.mark.xfail(reason="BUG PROBE: auto-created endpoints get the lower-cased id as display name", strict=False)
    def test_auto_created_entity_keeps_original_casing(self):
        s = store()
        s.upsert_extraction("c1", "a.txt", [], [{"head": "Dr. Meera Iyer", "relation": "R", "tail": "X"}])
        names = {e["name"] for e in s.calls[0][1]["ents"]}
        assert "Dr. Meera Iyer" in names


class TestReads:
    def test_find_entities_dedups_and_preserves_order(self):
        s = store(lambda c, p: [{"id": "meera"}, {"id": "zeroproof"}] if "meera" in p["q"].lower() else [{"id": "zeroproof"}])
        assert s.find_entities(["Meera", "ZeroProof"]) == ["meera", "zeroproof"]

    def test_find_entities_skips_blank_names_without_querying(self):
        s = store([{"id": "x"}])
        assert s.find_entities(["", "   "]) == []
        assert s.calls == []

    def test_find_entities_builds_exact_or_fuzzy_query_and_limit(self):
        s = store([])
        s.find_entities(["Meera Iyer"], per_name=3)
        _, params = s.calls[0]
        assert params["q"] == '"Meera Iyer" OR Meera Iyer~' and params["k"] == 3

    def test_find_entities_escapes_lucene_syntax(self):
        s = store([])
        s.find_entities(["C++ (lang)"])
        assert r"C\+\+ \(lang\)" in s.calls[0][1]["q"]

    @pytest.mark.xfail(reason="BUG PROBE: LLM can return null/number in the entities list; re.sub raises TypeError", strict=False)
    def test_find_entities_tolerates_non_string_names(self):
        store([]).find_entities([None, 42])

    def test_expand_rejects_non_integer_hops_before_touching_db(self):
        s = store([])
        with pytest.raises(ValueError):
            s.expand(["a"], hops="2; MATCH (n) DETACH DELETE n")
        assert s.calls == []

    def test_expand_inlines_integer_hops(self):
        s = store([])
        s.expand(["a"], hops="3")
        assert "*1..3" in s.calls[0][0]

    def test_get_chunks_keeps_requested_order_and_skips_missing(self):
        rows = [{"chunk_id": "b", "source": "s", "text": "B"}, {"chunk_id": "a", "source": "s", "text": "A"}]
        s = store(rows)
        assert [c["chunk_id"] for c in s.get_chunks(["a", "missing", "b"])] == ["a", "b"]
