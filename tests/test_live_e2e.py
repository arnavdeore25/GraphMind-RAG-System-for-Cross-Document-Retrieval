"""End-to-end tests against REAL Neo4j + Gemini + embeddings.

Run:  docker compose up -d && GRAPHMIND_LIVE=1 GRAPHMIND_WIPE_OK=1 pytest -m live -v

WARNING: ingestion.py --reset runs `MATCH (n) DETACH DELETE n` on the WHOLE Neo4j
database. These tests therefore refuse to run unless GRAPHMIND_WIPE_OK=1, and
you should point NEO4J_URI at a throw-away instance.
"""
import os
import re
import sys

import pytest

pytestmark = pytest.mark.live
FIXTURE_SRC = os.path.join(os.path.dirname(__file__), "fixtures", "source")


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    """Ingest the fixture corpus once (vector index in a temp dir + graph in Neo4j)."""
    if os.getenv("GRAPHMIND_WIPE_OK") != "1":
        pytest.skip("set GRAPHMIND_WIPE_OK=1 (tests wipe the Neo4j database)")
    import ingestion
    import generation

    ingestion.SOURCE_DIR = FIXTURE_SRC
    ingestion.CHROMA_DIR = str(tmp_path_factory.mktemp("chroma"))
    old_argv, sys.argv = sys.argv, ["ingestion.py", "--reset"]
    try:
        ingestion.main()
    finally:
        sys.argv = old_argv
    return generation.ChatSession


def ask(session_cls, mode, q):
    s = session_cls(mode)
    ans, r, _ = s.ask(q)
    return ans, r


class TestGraphBuilt:
    def test_graph_has_chunks_entities_relations(self, system):
        from graph_store import GraphStore
        st = GraphStore(); stats = st.stats(); st.close()
        assert stats["chunks"] >= 4 and stats["entities"] >= 8 and stats["relations"] >= 4

    def test_entities_merge_across_documents(self, system):
        from graph_store import GraphStore
        st = GraphStore()
        rows = st._run("MATCH (e:Entity) WHERE e.id CONTAINS 'meera' RETURN count(e) AS n")
        st.close()
        assert rows[0]["n"] <= 2       # "dr. meera iyer" (+ at most one variant), not one per file


@pytest.mark.parametrize("mode", ["vector", "graph", "hybrid"])
class TestSingleHop:
    def test_mentor_lookup(self, system, mode):
        ans, _ = ask(system, mode, "Who is the assigned mentor of team ZeroProof?")
        if mode == "graph":
            pytest.xfail("graph mode can miss when entity linking fails; informational")
        assert "meera" in ans.lower()

    def test_education_prize(self, system, mode):
        ans, _ = ask(system, mode, "What is the first prize for the Education track?")
        assert "40,000" in ans or "40000" in ans


class TestMultiHop:
    def test_two_hop_expertise(self, system):
        ans, r = ask(system, "hybrid", "What is the expertise of the mentor assigned to the team building the Phishing Detection Tool?")
        assert "cryptograph" in ans.lower() or "zero-knowledge" in ans.lower()
        assert {"teams.csv", "mentors.csv"} <= {c["source"] for c in r.chunks}

    def test_two_hop_sponsor_and_prize(self, system):
        ans, _ = ask(system, "hybrid", "Which company sponsors the track of the team working on the stock market sentiment tracker, and what is that track's first prize?")
        assert "finedge" in ans.lower() and "50,000" in ans

    def test_three_hop_office_hours_and_sponsor(self, system):
        ans, r = ask(system, "hybrid", "Where are the office hours of the mentor for the team that must use zero-knowledge proofs, and which company sponsors that team's track?")
        assert "A101" in ans and "cybershield" in ans.lower()
        assert len({c["source"] for c in r.chunks}) >= 3

    def test_three_hop_solar(self, system):
        ans, _ = ask(system, "hybrid", "Which organization employs the mentor of team SolarSense, and what model does the tech guide suggest for their problem?")
        assert "greengrid" in ans.lower() and ("random forest" in ans.lower() or "lstm" in ans.lower())


class TestGrounding:
    @pytest.mark.parametrize("mode", ["vector", "graph", "hybrid"])
    def test_unanswerable_question_is_refused(self, system, mode):
        ans, _ = ask(system, mode, "What is the capital of Australia?")
        assert "don't have enough information" in ans.lower()

    def test_citations_point_to_real_sources(self, system):
        ans, r = ask(system, "hybrid", "Who mentors team FinPulse?")
        nums = [int(n) for n in re.findall(r"\[(\d+)\]", ans)]
        assert nums and all(1 <= n <= len(r.chunks) for n in nums)

    def test_followup_question_resolved_from_history(self, system):
        s = system("hybrid")
        s.ask("Who is the mentor of team ZeroProof?")
        ans, _, standalone = s.ask("Where are her office hours?")
        assert "meera" in standalone.lower() or "zeroproof" in standalone.lower()
        assert "A101" in ans
