"""ingestion.split_documents and loaders.load_documents."""
import os

import pytest
from langchain_core.documents import Document

import config
import loaders
from ingestion import split_documents


# ---------------- chunking ----------------
class TestSplitDocuments:
    def test_every_chunk_gets_an_id_matching_the_returned_list(self):
        chunks, ids = split_documents([Document(page_content="hello world " * 200, metadata={"source": "a.txt"})])
        assert len(chunks) == len(ids) > 1
        assert [c.metadata["chunk_id"] for c in chunks] == ids

    def test_chunk_size_is_respected(self):
        chunks, _ = split_documents([Document(page_content="word " * 1000, metadata={"source": "a.txt"})])
        assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)

    def test_ids_are_deterministic_so_reingest_upserts(self):
        d = lambda: [Document(page_content="stable text " * 100, metadata={"source": "a.txt"})]
        assert split_documents(d())[1] == split_documents(d())[1]

    def test_id_depends_on_source_and_page(self):
        base = "same text"
        ids = {
            split_documents([Document(page_content=base, metadata={"source": s, "page": p})])[1][0]
            for s, p in [("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1)]
        }
        assert len(ids) == 3

    def test_id_format_is_16_hex_chars(self):
        _, ids = split_documents([Document(page_content="x", metadata={"source": "a"})])
        assert len(ids[0]) == 16 and int(ids[0], 16) >= 0

    def test_empty_input_gives_no_chunks(self):
        assert split_documents([]) == ([], [])

    def test_whitespace_only_document_gives_no_chunks(self):
        chunks, _ = split_documents([Document(page_content="   \n\n ", metadata={"source": "a"})])
        assert chunks == []

    @pytest.mark.xfail(reason="BUG PROBE: identical text in the same source/page gives duplicate chunk ids "
                              "(Chroma/Neo4j keys collide)", strict=False)
    def test_ids_unique_when_text_repeats_in_one_source(self):
        docs = [Document(page_content="Page footer: Confidential", metadata={"source": "a.pdf"})] * 2
        _, ids = split_documents(docs)
        assert len(set(ids)) == len(ids)


# ---------------- loaders ----------------
def write(p, name, content, mode="w"):
    path = p / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "wb":
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return str(path)


class TestLoaders:
    def test_txt_and_md_loaded_with_metadata(self, tmp_path):
        write(tmp_path, "a.txt", "alpha")
        write(tmp_path, "b.md", "# beta")
        docs = loaders.load_documents(str(tmp_path))
        by_src = {d.metadata["source"]: d for d in docs}
        assert set(by_src) == {"a.txt", "b.md"}
        assert by_src["a.txt"].metadata["doc_type"] == "txt"

    def test_csv_yields_one_document_per_row(self, tmp_path):
        write(tmp_path, "teams.csv", "team,mentor\nZeroProof,Meera\nSolarSense,Kavita\n")
        docs = loaders.load_documents(str(tmp_path))
        assert len(docs) == 2 and all(d.metadata["doc_type"] == "csv" for d in docs)
        assert "mentor: Meera" in docs[0].page_content

    def test_html_loaded_as_text(self, tmp_path):
        write(tmp_path, "p.html", "<html><head><title>T</title></head><body><p>Hello HTML</p></body></html>")
        docs = loaders.load_documents(str(tmp_path))
        assert "Hello HTML" in docs[0].page_content and docs[0].metadata["doc_type"] == "html"

    def test_extension_match_is_case_insensitive(self, tmp_path):
        write(tmp_path, "UPPER.TXT", "caps")
        assert len(loaders.load_documents(str(tmp_path))) == 1

    def test_nested_directories_are_walked(self, tmp_path):
        write(tmp_path, "sub/deeper/x.txt", "deep")
        assert loaders.load_documents(str(tmp_path))[0].metadata["source"] == "x.txt"

    def test_unsupported_extension_returns_nothing(self, tmp_path):
        p = write(tmp_path, "image.png", b"\x89PNG", mode="wb")
        assert loaders._load_file(p) == []

    def test_unreadable_file_is_skipped_not_fatal(self, tmp_path, capsys):
        write(tmp_path, "good.txt", "ok")
        write(tmp_path, "bad.txt", b"\xff\xfe\x00\x80 not utf8", mode="wb")
        docs = loaders.load_documents(str(tmp_path))
        assert [d.metadata["source"] for d in docs] == ["good.txt"]
        assert "[skip]" in capsys.readouterr().out

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            loaders.load_documents(str(tmp_path / "nope"))

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No supported files"):
            loaders.load_documents(str(tmp_path))

    def test_only_unsupported_files_raises(self, tmp_path):
        write(tmp_path, "x.png", b"123", mode="wb")
        with pytest.raises(FileNotFoundError):
            loaders.load_documents(str(tmp_path))

    def test_corrupt_pdf_is_skipped(self, tmp_path):
        write(tmp_path, "broken.pdf", b"%PDF-1.4 garbage", mode="wb")
        write(tmp_path, "ok.txt", "fine")
        assert [d.metadata["source"] for d in loaders.load_documents(str(tmp_path))] == ["ok.txt"]


class FakeWebLoader:
    seen = []
    fail = False

    def __init__(self, urls):
        type(self).seen = list(urls)

    def load(self):
        if type(self).fail:
            raise RuntimeError("network down")
        return [Document(page_content=f"page {u}", metadata={"source": u}) for u in type(self).seen]


class TestUrlLoading:
    @pytest.fixture(autouse=True)
    def patch_web(self, monkeypatch):
        import langchain_community.document_loaders as dl
        FakeWebLoader.fail = False
        monkeypatch.setattr(dl, "WebBaseLoader", FakeWebLoader, raising=False)

    def test_urls_file_loaded_with_web_doc_type_and_blank_lines_ignored(self, tmp_path):
        write(tmp_path, "urls.txt", "https://a.example\n\n   \nhttps://b.example\n")
        docs = loaders.load_documents(str(tmp_path))
        assert FakeWebLoader.seen == ["https://a.example", "https://b.example"]
        assert {d.metadata["doc_type"] for d in docs} == {"web"}

    def test_urls_txt_is_not_also_ingested_as_a_text_document(self, tmp_path):
        write(tmp_path, "urls.txt", "https://a.example\n")
        docs = loaders.load_documents(str(tmp_path))
        assert all(d.metadata.get("source") != "urls.txt" for d in docs)

    @pytest.mark.xfail(reason="BUG PROBE: '# comment' lines in urls.txt are treated as URLs", strict=False)
    def test_comment_lines_in_urls_file_are_ignored(self, tmp_path):
        write(tmp_path, "urls.txt", "# my sources\nhttps://a.example\n")
        loaders.load_documents(str(tmp_path))
        assert FakeWebLoader.seen == ["https://a.example"]

    @pytest.mark.xfail(reason="BUG PROBE: a failing URL download aborts the whole ingestion "
                              "(local files use try/except, URLs do not)", strict=False)
    def test_failing_url_does_not_lose_local_documents(self, tmp_path):
        write(tmp_path, "local.txt", "keep me")
        write(tmp_path, "urls.txt", "https://down.example\n")
        FakeWebLoader.fail = True
        docs = loaders.load_documents(str(tmp_path))
        assert [d.metadata["source"] for d in docs] == ["local.txt"]
