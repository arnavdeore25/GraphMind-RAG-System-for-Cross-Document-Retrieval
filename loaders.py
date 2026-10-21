"""Heterogeneous loaders: txt/md, pdf, csv, docx, html and URLs (source/urls.txt)."""
import os
from langchain_core.documents import Document


def _load_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(path, encoding="utf-8").load()
    elif ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(path).load()          # one Document per page -> page metadata
    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        docs = CSVLoader(path, encoding="utf-8").load()  # one Document per row
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        docs = Docx2txtLoader(path).load()
    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import BSHTMLLoader
        docs = BSHTMLLoader(path).load()
    else:
        return []
    for d in docs:
        d.metadata["source"] = os.path.basename(path)
        d.metadata["doc_type"] = ext.lstrip(".")
    return docs


def _load_urls(urls_file: str):
    from langchain_community.document_loaders import WebBaseLoader
    urls = [u.strip() for u in open(urls_file, encoding="utf-8") if u.strip()]
    docs = WebBaseLoader(urls).load()
    for d in docs:
        d.metadata["doc_type"] = "web"
    return docs


def load_documents(docs_path: str = "source"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} does not exist")
    docs = []
    for root, _, files in os.walk(docs_path):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f == "urls.txt":
                docs += _load_urls(p)
                continue
            try:
                docs += _load_file(p)
            except Exception as e:
                print(f"[skip] {p}: {e}")
    if not docs:
        raise FileNotFoundError(f"No supported files found in {docs_path}")
    print(f"Loaded {len(docs)} document units")
    return docs
