# DocuRAG
A private, locally-run web application that answers questions about your own collection of PDF documents (of any domain) and backs every answer with inline citations to the exact source passages it used.

## Purpose
Modern LLMs pose two problems:
1. Lack of access to new, private, or confidential data, and
2. Hallucination

Retrieval Augmented Generation (RAG) solves those problems by retrieving the relevant passages from your documents first, then asks the model to answer using only those passages, with numbered citations, so you can verify every claim against the original text.

## Features
- **Grounded answers with verifiable citations.** Each answer carries inline `[n]` markers that map to the retrieved passages shown beneath it, so you can check any claim against its source.
- **Two ways to answer questions.** Single-pass mode always retrieves a fixed number of passages. Agentic mode lets the model decide whether to search, what to search for, and whether it needs another search, while recording each decision in a step-by-step trace. (Agentic mode is currently available through the Python API and CLI; the web UI still uses single-pass mode.)
- **Any PDF, any domain.** Drop unencrypted PDFs into a folder or upload them (multiple at once) through the interface.
- **Incremental ingestion.** A SHA-256 manifest tracks which files have already been indexed, so unchanged docs are never re-embedded on subsequent runs.
- **Multi-LLM provider compatible.** Switch between cloud and fully local models at runtime from the Model tab, without touching the code.
- **Session tools.** Query history with per-entry copy and clear or export to Markdown or plain text.
- **Privacy first.** API keys entered in the UI are held in session memory only — never written to disk or logged. With a local model, no data leaves your machine.

## Under the Hood
Below are the exact steps the application follows to process the questions.

**Ingestion (both modes):**
1. **Ingest.** PDFs are parsed and split into overlapping ~500-character chunks.
2. **Embed & store.** Chunks are embedded with a sentence-transformer model and stored in a persistent ChromaDB index. A manifest records file hashes so re-running only processes new or changed files.

**Single-pass mode** (`rag_pipeline.answer_with_citations`):
3. **Retrieve.** A question is embedded and the most similar chunks are pulled from the index.
4. **Generate.** The retrieved passages and the question go to the chosen LLM, which is instructed to answer only from the provided context and cite each claim by passage number.

**Agentic mode** (`agent_pipeline.answer_with_agent`):
3. **Decide.** The model is given a `search_documents` tool and chooses whether to call it, which query to run, and how many passages to retrieve. Its search query may differ from the user's original wording.
4. **Loop.** Tool results are fed back to the model, which is then invoked again. It may search again to improve a weak query or address a separate part of the question. This continues until the model answers or reaches the iteration limit.
5. **Trace.** Each step is recorded, including the tool used, its arguments, and the iteration in which it was called.

## Getting Started
### Tested Environment & Dependencies
- **OS:** MacOS
- **IDE:** VSCodium
- **Interpreter:** Python 3.13
- **Core dependencies:**
```
langchain                   # orchestration
langchain-community         # PyPDFLoader
langchain-core              # prompts, documents
langchain-text-splitters    # chunking
langchain-huggingface       # embeddings
langchain-chroma            # vector store wrapper
langchain-anthropic         # Claude
langchain-ollama            # local models
chromadb                    # vector database engine
sentence-transformers       # embedding model
pypdf                       # PDF parsing
gradio                      # web UI
python-dotenv               # .env loading
```

### Setup
```zsh
# 1. Clone and enter the repo
git clone <repo-url>
cd <repo>

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# (or install the core packages directly as shown below)
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface langchain-chroma langchain-anthropic langchain-ollama chromadb sentence-transformers pypdf python-dotenv gradio

# 4. Configure credentials
cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY (or set USE_OLLAMA=1 for local)
# alternatively (if you prefer to avoid using .env), from the Model tab: 
#   - select the provider and enter API, or 
#   - download local model (like llama3.2), select the provider (Ollama), and enter the model name
```
If a PDF fails to parse with an AES-encryption error, install crypto support:
`pip install "pypdf[crypto]"`.

### Usage
Put some PDFs in `./pdf_db/` (or upload them later through the UI), then pick one of the two:
**Command line (pipeline test only):**
```zsh
python rag_pipeline.py
```
**Web interface:**
```zsh
python gradio_app.py
```
Then open the local URL it prints.
To run fully offline with Ollama:
```zsh
# install Ollama, then:
ollama pull llama3.2
# in the app's Model tab: provider = ollama, model = llama3.2
```

## Tech stack
Python | LangChain (including tool calling) | ChromaDB | sentence-transformers (Hugging Face) | Gradio | Anthropic/Ollama and other LLM providers.

## Known limitations
- **Agentic mode is not yet in the web UI.** The Gradio Interface currently runs the single-pass pipeline only.
- **Fixed-size chunking.** Chunks are split at ~500 characters regardless of meaning, which can cut sentences awkwardly or separate a table from its header.
- **Citation numbers reset after each search.** In agentic mode, `[n]` refers to the passages returned by the latest search. If the model searches twice, `[1]` refers to the first passage in the second set of results rather than the first passage retrieved overall.
- **Single-turn.** Each question is answered independently; conversational style is a probable planned addition.
- **Local model reliability.** Small local models follow citation instructions less reliably than frontier models, and more capable local models demand significant storage and RAM. For example, on an identical query with identical retrieval, Claude Sonnet ran two searches, refining the query after reviewing the first set of results, and supported every claim with a retrieved passage. Llama 3.2 (3B) searched only once, asked for twice as many passages, and still made three claims with no citation to any retrieved passage. Both models followed the required tool-calling format, but only Claude followed the grounding requirement.
- **PDF-only**. Other document formats (DOCX, HTML, plain text, etc.) are not yet ingested.

## License
Licensed under the Apache License 2.0 — see LICENSE.