# GraphMind

GraphMind is a cross-document question-answering system that combines vector search with a knowledge graph. It ingests local documents, extracts entities and relationships, and retrieves evidence for grounded answers with inline source citations.

## How it works

1. Documents are loaded and split into overlapping chunks.
2. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored locally in Chroma.
3. Gemini extracts entities and relationships from each chunk; Neo4j stores the graph alongside links back to the source chunks.
4. At question time, GraphMind can retrieve by vector similarity, graph traversal, or a hybrid of both.
5. Gemini generates an answer using only the retrieved evidence and cites its sources. If the evidence is insufficient, it says so.

## Requirements

- Python 3.10 or newer
- Docker with Docker Compose, for the Neo4j graph store
- A Google Gemini API key

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Create a local environment file from the example:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux, use `cp .env.example .env`. Edit `.env` and set `GEMINI_API_KEY` to your Gemini API key. The example Neo4j settings match the included Docker Compose configuration; change them together if you use different credentials or a different Neo4j instance.

Start Neo4j:

```bash
docker compose up -d
```

Neo4j Browser is available at <http://localhost:7474>; the Bolt connection uses `bolt://localhost:7687`.

## Add documents and ingest

Put files in the `source/` directory. Supported formats are:

- `.txt` and `.md`
- `.pdf`
- `.csv`
- `.docx`
- `.html` and `.htm`

To ingest web pages, create `source/urls.txt` and put one URL on each line. The loader downloads and indexes those pages along with local documents.

Build both the vector index and knowledge graph:

```bash
python ingestion.py
```

Useful options:

```bash
python ingestion.py --no-graph  # Build only the vector index
python ingestion.py --reset     # Clear existing indexes, then ingest again
```

Ingestion stores the vector index under `db/chroma_db/` and the graph in Neo4j. The first run downloads the embedding model and may take a while. Re-running ingestion upserts chunks by ID; use `--reset` when you want to rebuild from scratch.

## Ask questions

Start the interactive chat; the default retrieval mode is hybrid:

```bash
python generation.py
```

Choose a retrieval mode explicitly, and optionally print retrieval details:

```bash
python generation.py --mode hybrid --debug
python generation.py --mode vector
python generation.py --mode graph
```

Type `quit` or `exit` to end the chat. Answers include numbered citations, and the chat keeps recent conversation context so follow-up questions can be rewritten as standalone queries.

## Evaluate retrieval modes

Copy `eval_set.example.json` to `eval_set.json` and adapt it to your documents. Each item should include a question and reference answer; `gold_sources` and `hops` are optional:

```json
[
  {
    "question": "Which organization sponsors the track?",
    "answer": "Example Organization",
    "gold_sources": ["sponsors.txt"],
    "hops": 1
  }
]
```

Run the evaluation:

```bash
python evaluate.py eval_set.json
```

The evaluator compares vector, graph, and hybrid retrieval, asks Gemini to judge answer correctness, and reports evidence recall, full source coverage, latency, and context size. It writes per-question results to `eval_results.csv`. If no path is provided, it looks for `eval_set.json` in the project root.

## Project structure

```text
config.py             Application settings and model names
loaders.py            Local file and URL loaders
ingestion.py          Chunking, embedding, and index creation
graph_extraction.py   Entity and relationship extraction
graph_store.py        Neo4j persistence and graph queries
retrieval.py          Vector, graph, and hybrid retrieval
generation.py         Evidence-grounded chat interface
evaluate.py           Retrieval-mode evaluation
eval_set.example.json Example evaluation questions
docker-compose.yml    Local Neo4j service
```

## Notes

- Keep `.env` private; it contains your API key and service credentials.
- The local document directory (`source/`) and generated Chroma data (`db/`) are excluded from Git. Add your own source documents locally before ingesting.
- Answer generation and graph extraction require a valid Gemini API key. The Neo4j service must be running for retrieval in any mode; retrieved chunk text is loaded from Neo4j even when using vector search.
