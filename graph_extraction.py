"""LLM-based entity / relation extraction with evidence (chunk_id) preserved."""
from concurrent.futures import ThreadPoolExecutor
from llm import generate_json

ENTITY_TYPES = "Person, Organization, Concept, Technology, Method, Dataset, Metric, Location, Event, Product, Other"

EXTRACT_PROMPT = """Extract a knowledge graph from the text.

Rules:
- Entities: canonical, specific names (use full names, expand acronyms when the text does). Types: {types}.
- Relations: short verb-style labels (e.g. "FOUNDED", "USES", "PART_OF", "EVALUATED_ON", "CAUSES").
- Only include facts explicitly stated in the text. No outside knowledge.
- Max 15 entities and 20 relations.

Return JSON: {{"entities":[{{"name":"","type":""}}],"relations":[{{"head":"","relation":"","tail":""}}]}}

Text:
\"\"\"{text}\"\"\""""

QUERY_ENTITY_PROMPT = """List the key named entities/concepts in this question (as they would appear in documents).
Return JSON: {{"entities":["..."]}}

Question: {q}"""


def extract_chunk(text: str) -> dict:
    out = generate_json(EXTRACT_PROMPT.format(types=ENTITY_TYPES, text=text[:4000]))
    return {"entities": out.get("entities", []), "relations": out.get("relations", [])}


def extract_query_entities(question: str) -> list[str]:
    return generate_json(QUERY_ENTITY_PROMPT.format(q=question)).get("entities", [])


def build_graph(chunks_with_ids, store, workers=4):
    """chunks_with_ids: list of (chunk_id, Document)."""
    store.setup()
    for cid, doc in chunks_with_ids:
        store.upsert_chunk(cid, doc.metadata.get("source", "unknown"), doc.page_content)

    def work(item):
        cid, doc = item
        try:
            return cid, doc, extract_chunk(doc.page_content)
        except Exception as e:
            print(f"[extract fail] {cid}: {e}")
            return cid, doc, {"entities": [], "relations": []}

    with ThreadPoolExecutor(workers) as ex:
        for i, (cid, doc, ext) in enumerate(ex.map(work, chunks_with_ids), 1):
            store.upsert_extraction(cid, doc.metadata.get("source", "unknown"), ext["entities"], ext["relations"])
            if i % 10 == 0:
                print(f"  graph: {i}/{len(chunks_with_ids)} chunks")
    print("Graph stats:", store.stats())
