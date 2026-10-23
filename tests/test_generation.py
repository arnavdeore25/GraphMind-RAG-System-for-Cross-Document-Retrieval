"""generation.py: prompt building, grounding fallback, history-aware chat."""
import generation
from generation import NO_INFO, ChatSession, answer, build_prompt
from retrieval import Retrieved
from conftest import chunk, triple


def retrieved(chunks, triples=()):
    return Retrieved("hybrid", list(chunks), list(triples))


class TestBuildPrompt:
    def test_sources_numbered_in_order_with_source_names(self):
        p = build_prompt("Q?", retrieved([chunk("a", "teams.csv", "alpha"), chunk("b", "mentors.csv", "beta")]))
        assert "[1] (source: teams.csv)\nalpha" in p
        assert "[2] (source: mentors.csv)\nbeta" in p

    def test_triples_cite_the_right_source_number(self):
        r = retrieved([chunk("a"), chunk("b")], [triple("Team", "MENTORED_BY", "Meera", "b")])
        assert "- Team --MENTORED_BY--> Meera [2]" in build_prompt("Q?", r)

    def test_no_triples_placeholder(self):
        assert "(none)" in build_prompt("Q?", retrieved([chunk("a")]))

    def test_contains_question_and_exact_refusal_string(self):
        p = build_prompt("Who is X?", retrieved([chunk("a")]))
        assert "Question: Who is X?" in p and NO_INFO in p

    def test_grounding_instructions_present(self):
        p = build_prompt("Q?", retrieved([chunk("a")]))
        assert "ONLY" in p and "[n]" in p


class TestAnswer:
    def test_no_chunks_returns_refusal_without_calling_llm(self, monkeypatch):
        monkeypatch.setattr(generation, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
        assert answer("Q?", retrieved([])) == NO_INFO

    def test_with_chunks_calls_llm_with_built_prompt(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(generation, "generate", lambda prompt, *a, **k: seen.setdefault("p", prompt) and "ANSWER [1]")
        assert answer("Q?", retrieved([chunk("a", text="evidence")])) == "ANSWER [1]"
        assert "evidence" in seen["p"]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, q, mode="hybrid"):
        self.queries.append(q)
        return retrieved([chunk("a")])


class TestChatSession:
    def make(self, monkeypatch, replies):
        monkeypatch.setattr(generation, "Retriever", FakeRetriever)
        prompts = []

        def fake_generate(prompt, *a, **k):
            prompts.append(prompt)
            return replies.pop(0)
        monkeypatch.setattr(generation, "generate", fake_generate)
        return ChatSession("hybrid"), prompts

    def test_first_question_is_not_rewritten(self, monkeypatch):
        s, prompts = self.make(monkeypatch, ["answer 1"])
        ans, r, sq = s.ask("Who mentors ZeroProof?")
        assert sq == "Who mentors ZeroProof?" and ans == "answer 1"
        assert len(prompts) == 1                      # only the answer call, no rewrite call
        assert s.retriever.queries == ["Who mentors ZeroProof?"]

    def test_followup_is_rewritten_using_history(self, monkeypatch):
        s, prompts = self.make(monkeypatch, ["a1", "Where are Meera Iyer's office hours?", "a2"])
        s.ask("Who mentors ZeroProof?")
        _, _, sq = s.ask("Where are her office hours?")
        assert sq == "Where are Meera Iyer's office hours?"
        assert "Who mentors ZeroProof?" in prompts[1] and "History:" in prompts[1]
        assert s.retriever.queries[-1] == sq

    def test_history_records_original_questions_and_answers(self, monkeypatch):
        s, _ = self.make(monkeypatch, ["a1", "standalone", "a2"])
        s.ask("q1"); s.ask("q2")
        assert s.history == [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]

    def test_rewrite_window_is_last_six_messages(self, monkeypatch):
        replies = ["a1", "s2", "a2", "s3", "a3", "s4", "a4", "s5", "a5"]
        s, prompts = self.make(monkeypatch, replies)
        for q in ["q1", "q2", "q3", "q4", "q5"]:
            s.ask(q)
        rewrite_prompt_for_q5 = prompts[-2]
        assert "q1" not in rewrite_prompt_for_q5      # oldest turn dropped
        assert "q4" in rewrite_prompt_for_q5
