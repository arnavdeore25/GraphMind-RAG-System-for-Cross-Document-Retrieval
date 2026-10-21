"""Ingest -> chunk -> embed (Chroma) -> extract entities/relations (Neo4j).

Usage:
  python ingestion.py            # vector + graph
  python ingestion.py --no-graph # vector only
  python ingestion.py --reset    # wipe both stores first
"""
import argparse
import hashlib
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import *
from loaders import load_documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    ids = []
    for i, c in enumerate(chunks):
        cid = hashlib.sha1(f"{c.metadata.get('source')}|{c.metadata.get('page', '')}|{c.page_content}".encode()).hexdigest()[:16]
        c.metadata["chunk_id"] = cid     # same id in Chroma AND Neo4j -> evidence tracing
        ids.append(cid)
    print(f"Created {len(chunks)} chunks")
    return chunks, ids


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    docs = load_documents(SOURCE_DIR)
    chunks, ids = split_documents(docs)

    print("Embedding into Chroma...")
    get_vectorstore().add_documents(chunks, ids=ids)   # upsert by id -> re-runs don't duplicate

    if not args.no_graph:
        from graph_store import GraphStore
        from graph_extraction import build_graph
        store = GraphStore()
        if args.reset:
            store.clear()
        print("Building knowledge graph in Neo4j...")
        build_graph(list(zip(ids, chunks)), store)
        store.close()
    print("Done.")


if __name__ == "__main__":
    main()
