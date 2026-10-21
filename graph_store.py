"""Neo4j wrapper.

Schema
  (:Chunk {id, source, text})
  (:Entity {id, name, type})
  (:Chunk)-[:MENTIONS]->(:Entity)
  (:Entity)-[:RELATED {type, chunk_id, source}]->(:Entity)   # one edge per piece of evidence
"""
import re
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _lucene_escape(s: str) -> str:
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", s)


class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def _run(self, q, **params):
        with self.driver.session() as s:
            return [r.data() for r in s.run(q, **params)]

    def setup(self):
        self._run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        self._run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        self._run("CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]")

    def clear(self):
        self._run("MATCH (n) DETACH DELETE n")

    # ---------- write ----------
    def upsert_chunk(self, chunk_id, source, text):
        self._run("MERGE (c:Chunk {id:$id}) SET c.source=$source, c.text=$text",
                  id=chunk_id, source=source, text=text)

    def upsert_extraction(self, chunk_id, source, entities, relations):
        ents = [{"id": norm(e["name"]), "name": e["name"].strip(), "type": e.get("type", "Other")}
                for e in entities if e.get("name", "").strip()]
        rels = [{"h": norm(r["head"]), "t": norm(r["tail"]), "type": r["relation"].strip().upper().replace(" ", "_")}
                for r in relations
                if r.get("head", "").strip() and r.get("tail", "").strip() and r.get("relation", "").strip()]
        # make sure relation endpoints exist as entities
        known = {e["id"] for e in ents}
        for r in rels:
            for k in ("h", "t"):
                if r[k] not in known:
                    ents.append({"id": r[k], "name": r[k], "type": "Other"})
                    known.add(r[k])
        self._run("""
            MATCH (c:Chunk {id:$cid})
            UNWIND $ents AS e
              MERGE (n:Entity {id:e.id})
                ON CREATE SET n.name=e.name, n.type=e.type
              MERGE (c)-[:MENTIONS]->(n)
        """, cid=chunk_id, ents=ents)
        if rels:
            self._run("""
                UNWIND $rels AS r
                MATCH (a:Entity {id:r.h}), (b:Entity {id:r.t})
                MERGE (a)-[x:RELATED {type:r.type, chunk_id:$cid}]->(b)
                SET x.source=$src
            """, rels=rels, cid=chunk_id, src=source)

    # ---------- read ----------
    def find_entities(self, names, per_name=2):
        ids = []
        for n in names:
            q = _lucene_escape(n)
            if not q.strip():
                continue
            rows = self._run(
                "CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node, score "
                "RETURN node.id AS id ORDER BY score DESC LIMIT $k", q=f'"{q}" OR {q}~', k=per_name)
            ids += [r["id"] for r in rows]
        return list(dict.fromkeys(ids))

    def entities_of_chunks(self, chunk_ids, limit=8):
        rows = self._run("""
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.id IN $ids
            WITH e, count(c) AS n ORDER BY n DESC LIMIT $limit RETURN e.id AS id
        """, ids=chunk_ids, limit=limit)
        return [r["id"] for r in rows]

    def expand(self, entity_ids, hops=2, limit=60):
        """Multi-hop traversal; returns evidence-carrying triples."""
        hops = int(hops)
        return self._run(f"""
            MATCH (s:Entity) WHERE s.id IN $ids
            MATCH p=(s)-[:RELATED*1..{hops}]-(:Entity)
            WITH DISTINCT relationships(p) AS rs
            UNWIND rs AS r
            WITH DISTINCT r
            RETURN startNode(r).name AS head, r.type AS rel, endNode(r).name AS tail,
                   r.chunk_id AS chunk_id, r.source AS source
            LIMIT $limit
        """, ids=entity_ids, limit=limit)

    def chunks_mentioning(self, entity_ids, limit=20):
        return self._run("""
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $ids
            RETURN c.id AS chunk_id, count(e) AS n ORDER BY n DESC LIMIT $limit
        """, ids=entity_ids, limit=limit)

    def get_chunks(self, chunk_ids):
        rows = self._run("MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS chunk_id, c.source AS source, c.text AS text",
                         ids=chunk_ids)
        by_id = {r["chunk_id"]: r for r in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]

    def stats(self):
        return self._run("""
            MATCH (c:Chunk) WITH count(c) AS chunks
            MATCH (e:Entity) WITH chunks, count(e) AS entities
            OPTIONAL MATCH ()-[r:RELATED]->() RETURN chunks, entities, count(r) AS relations
        """)[0]
