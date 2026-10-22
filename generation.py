"""History-aware, citation-grounded answer generation.

Usage: python generation.py [--mode vector|graph|hybrid]
"""
import argparse
from llm import generate
from retrieval import Retriever, Retrieved

NO_INFO = "I don't have enough information to answer that question."


def build_prompt(question, r: Retrieved):
    idx = {c["chunk_id"]: i for i, c in enumerate(r.chunks, 1)}
    sources = "\n\n".join(f"[{idx[c['chunk_id']]}] (source: {c['source']})\n{c['text']}" for c in r.chunks)
    facts = "\n".join(f"- {t['head']} --{t['rel']}--> {t['tail']} [{idx[t['chunk_id']]}]" for t in r.triples)
    return f"""Answer the question using ONLY the sources and graph facts below.
- Cite evidence inline as [n] after each claim. For multi-hop answers, cite every source used.
- If the answer is not supported, reply exactly: "{NO_INFO}"

Question: {question}

Knowledge graph facts (relations extracted from the sources):
{facts or '(none)'}

Sources:
{sources}
"""


def answer(question, r: Retrieved):
    if not r.chunks:
        return NO_INFO
    return generate(build_prompt(question, r))


class ChatSession:
    def __init__(self, mode="hybrid"):
        self.retriever = Retriever()
        self.mode = mode
        self.history = []   # [(role, text)]

    def _standalone(self, q):
        if not self.history:
            return q
        hist = "\n".join(f"{role}: {txt}" for role, txt in self.history[-6:])
        return generate(f"Rewrite the new question as a standalone, searchable question using the chat history. "
                        f"Return ONLY the question.\n\nHistory:\n{hist}\n\nNew question: {q}")

    def ask(self, question):
        sq = self._standalone(question)
        r = self.retriever.retrieve(sq, mode=self.mode)
        ans = answer(sq, r)
        self.history += [("user", question), ("assistant", ans)]
        return ans, r, sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["vector", "graph", "hybrid"])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    s = ChatSession(args.mode)
    print(f"GraphMind [{args.mode}] - type 'quit' to exit")
    while True:
        q = input("\nYou> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        ans, r, sq = s.ask(q)
        print(f"\n(standalone: {sq}) | retrieval {r.latency:.2f}s")
        if args.debug:
            print("\n[debug]", r.debug)
            print(f"[debug] {len(r.triples)} graph triples used")
            for t in r.triples[:10]:
                print(f"   {t['head']} --{t['rel']}--> {t['tail']}  ({t['source']})")
            for i, c in enumerate(r.chunks, 1):
                print(f"   chunk [{i}] {c['source']} via {c['via']}: {c['text'][:80]!r}")
        print("\n" + ans)
        print("\nSources:")
        for i, c in enumerate(r.chunks, 1):
            print(f"  [{i}] {c['source']}  (via {c['via']})")


if __name__ == "__main__":
    main()