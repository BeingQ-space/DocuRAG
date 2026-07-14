# Copyright 2026 BeingQ-space
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Console ingestion + retrieval Q&A with citations for a document-based RAG pipeline.

- Loads/updates a persistent Chroma vector index from local PDFs (optionally downloading
    public PDFs), chunking with overlap and using a SHA-256 manifest to skip unchanged files.
- Retrieves top-k relevant chunks for a query.
- Prompts the selected LLM to answer using ONLY the provided context, citing inline as [n].
- Prints a retrieval preview and the final cited answer along with the document references used.

P.S. File contains 5 sections, 16 functions, and one main.
"""


import chromadb
import hashlib
import json
import os
import urllib.request

from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


########################################################################
load_dotenv() # loads the .env file

# Configs 
PDF_PATH = Path("./pdf_db")
PDF_PATH.mkdir(parents=True, exist_ok=True)
CHROMA_PATH = Path("./chroma_db")
COLLECTION_NAME = "multi_domain_pdfs"
MANIFEST_PATH = PDF_PATH / "manifest.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

PUBLIC_PDF_URLS = [
    # "https://example.org/some_guide.pdf"
]

USE_OLLAMA = os.environ.get("USE_OLLAMA", "0") == "1"
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

PROVIDER_DEFAULT_MODEL = {  # per-provider fallback model when user gives none
    "anthropic": ANTHROPIC_MODEL,
    "ollama": OLLAMA_MODEL,
    "openai": "gpt-4o-mini",
    "openai_compatible": "gpt-3.5-turbo",
    "google": "gemini-1.5-flash",
    "mistral": "mistral-small-latest",
    "groq": "llama-3.1-8b-instant",
}
########################################################################


########################################################################
# ___ SECTION 1: MANIFEST & FILESYSTEM ___
# FUNCTION 1
def sha256_file(path, chunk_size: int = 1024 * 1024):
    """
    Computes the SHA-256 hex digest of a file, read in fixed-size blocks.

    Args:
        path: Path to the file to hash.
        chunk_size: No. of bytes read per iteration (default: 1 MiB)
    
    Returns:
        The file's SHA-256 digest as a hex string.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    
    return h.hexdigest()


# FUNCTION 2
def load_manifest() -> Dict[str, dict]:
    """
    Loads the ingestion manifest mapping filename -> {sha256}.

    Returns:
        The manifest dictionary, or an empty dictionary if no manifest file exists yet.
    """

    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    
    return {}


# FUNCTION 3
def save_manifest(manifest: Dict[str, dict]):
    """
    Writes the ingestion manifest to disk as indented JSON.

    Args:
        manifest: The filename -> {sha256} mapping to persist.
    """

    PDF_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# FUNCTION 4
def ensure_paths():
    """Creates the PDF and Chroma directories if they don't already exist."""

    PDF_PATH.mkdir(parents=True, exist_ok=True)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)


# ___ SECTION 2: PDF INGESTION & INDEXING ___
# FUNCTION 5
def download_pdfs(urls, dest_path):
    """
    Downloads any configured public PDF URLs into the destination folder. Skips URLs whose
    target file already exists and is non-empty.

    Args:
        urls: Iterable of PDF URLs to fetch.
        dest_path: Directory to save downloaded PDFs into.
    
    Returns:
        List of Paths for files actually downloaded this call.
    """

    downloaded = []
    for url in urls:
        name = url.split("/")[-1]
        if not name.lower().endswith(".pdf"):
            name = name + ".pdf"
        out_path = dest_path / name

        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, out_path)
        downloaded.append(out_path)
    
    return downloaded


# FUNCTION 6
def load_and_chunk(pdf_paths: List[Path]):
    """
    Loads PDFs, splits them into overlapping chunks, and tags citation metadata. Files
    that fail to parse are skipped (logged) rather than aborting the run.

    Args:
        pdf_paths: List of PDF file Paths to process.
    
    Returns:
        A tuple (all_chunks, chunked_pdf_names) where all_chunks is the list of all
        chunk Documents across every file, and chunked_pdf_names lists the names of
        the PDFs that produced at least one chunk.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,                      # character limit per chunk (~70-100 words)
        chunk_overlap=CHUNK_OVERLAP,                # preserves context across splits
        length_function=len,                        # use standard character length
        separators=["\n\n", "\n", " ", ""]          # split markers
    ) # create text splitter object

    pdf_files = sorted(pdf_paths)       # sorted list of PDFs
    all_chunks = []                      # list of all chunks
    chunked_pdf_names= []                # names of PDFs that got chunked

    for pf in pdf_files:
        print(f"Loading and parsing file: {pf.name}...")
        try:
            loader = PyPDFLoader(pf)
            docs = loader.load()    # extract text from each page
        except Exception as e:
            print(f"Error parsing {pf} with PyPDFLoader: {e}")
            continue

        print(f"\n Loaded {len(docs)} total pages from {pf.name}.")
        
        chunks = text_splitter.split_documents(docs)

        # normalize metadata to support citations
        for c in chunks:
            if "source" in c.metadata:
                c.metadata["source"] = Path(c.metadata["source"]).name
            else:
                c.metadata["source"] = pf.name

            c.metadata.setdefault("page", c.metadata.get("page", "unknown"))
        
        if chunks:
            all_chunks.extend(chunks)
            chunked_pdf_names.append(pf.name)
        
        print(f"\n Total chunks: {len(all_chunks)}")
    
    return all_chunks, chunked_pdf_names


# FUNCTION 7
def get_vectordb(embeddings):
    """
    Opens (or creates) the persistent Chroma collection.

    Args:
        embeddings: The embedding function used to embed queries/documents.
    
    Returns:
        A Chroma vector store bound to the configured collection and directory.
    """

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )


# FUNCTION 8
def chroma_collection_is_nonempty():
    """
    Check whether the persisted Chroma collection contains any documents.
    Uses a direct PersistentClient read so it works without building the LangChain
    wrapper or an embedding model.

    Returns:
        True if the collection exists and holds at least one record, else False.
    """

    if not CHROMA_PATH.exists():
        return False
    
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        return False
    
    data = col.get(include=["metadatas"], limit=1)
    ids = data.get("ids") or []

    return len(ids) > 0


# FUNCTION 9
def add_new_pdfs_if_needed():
    """
    Ingests new or changed PDFs into the vector store, using the manifest to skip 
    unchanged ones.
    Hashes every PDF in PDF_PATH, compares against the manifest, embeds and adds
    only new /changed files, and records their hashes so later runs skip them.

    Returns:
        No. of chunks added this run (0 if nothing added).
    """

    ensure_paths()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = get_vectordb(embeddings)
    manifest = load_manifest()

    # add public downloads, if user requests
    if PUBLIC_PDF_URLS:
        downloaded = download_pdfs(urls=PUBLIC_PDF_URLS, dest_path=PDF_PATH)
        if downloaded:
            print(f"Downloaded {len(downloaded)} public PDFs into {PDF_PATH.resolve()}")
    
    pdf_files = sorted(PDF_PATH.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_PATH.resolve()}")
        return 0
    
    # determine which pdfs are new/changed
    to_add = []                                         # PDFs to add (updated after for-loop)
    digests = {}                                   # the digest cache
    for pf in pdf_files:
        digest = sha256_file(pf)
        digests[pf.name] = digest                                  # stashed for reuse
        entry = manifest.get(pf.name)
        if not entry or entry.get("sha256") != digest:
            to_add.append(pf)

    if not to_add and chroma_collection_is_nonempty():
        print("No new/changed PDFs detected and Chroma is non-empty. Skipping PDF addition.")
        return 0

    if not to_add and not chroma_collection_is_nonempty():          # empty chroma, but manifest says nothing
        to_add = pdf_files                              # add all PDFs
        
    print(f"Adding {len(to_add)} PDF(s)...")
    print(f"PDF addition status: to_add={len(to_add)} files={[p.name for p in to_add]}")
    chunks, chunked_pdf_names = load_and_chunk(to_add)    # chunks & PDFs' names that got chunked
    print(
        f"Chunking results: chunks={len(chunks)} "
        f"chunked_pdfs={len(chunked_pdf_names)} "
        f"chunked_pdf_names={chunked_pdf_names}"
        )                                                           # manifest update will be based on this
    if chunks:                                                      # add (& persistance is already default)
        vector_db.add_documents(chunks)
        print(f"Chroma index stored at: {CHROMA_PATH.resolve()}")
    else:
        print("No chunks produced; nothing to add. Manifest will not be updated")
        return 0

    # update manifest fingerprints ONLY for successfully chunked PDFs
    for pf_name in chunked_pdf_names:
        manifest[pf_name] = {"sha256": digests[pf_name]}
    
    save_manifest(manifest)

    return len(chunks) if chunks else 0


# ___ SECTION 3: LLM SELECTION ___
# FUNCTION 10
def describe_active_model(llm_kwargs=None):
    """
    Resolve the provider/model that a query would use, without building the LLM.

    Args:
        llm_kwargs: Optional dict with any of 'provider'/'model'; falls back to the
            module defaults (Ollama if USE_OLLAMA else Anthropic).

    Returns: 
        A tuple (provider, model) of display strings.
    """

    kw = llm_kwargs or {}
    provider = (kw.get("provider") or ("ollama" if USE_OLLAMA else "anthropic")).lower()
    model = kw.get("model") or PROVIDER_DEFAULT_MODEL.get(provider, "(default)")

    return provider, model


# FUNCTION 11
def build_llm(provider=None, model=None, api_key=None, base_url=None):
    """
    Builds a chat LLM. If no args, falls back to module env defaults.
    Integration packages are imported lazily so an uninstalled provider
    doesn't crash the app at import time.

    Args:
        provider: Provider key (e.g. 'anthropic', 'ollama'); defaults to
            Ollama if USE_OLLAMA is set, otherwise Anthropic.
        model: Model name; falls back to the provider's default when omitted.
        api_key: Session API key; when omitted the provider reads its env variable.
        base_url: Endpoint URL, used only by 'openai_compatible'.
    
    Returns:
        A LangChain chat model instance for the selected provider.
    
    Raises:
        ValueError: If the provider key is not recognized.
    """

    provider = (provider or ("ollama" if USE_OLLAMA else "anthropic")).lower()

    if provider == "ollama":
        return ChatOllama(model=model or OLLAMA_MODEL)
    
    if provider == "anthropic":
        kwargs = {"model": model or ANTHROPIC_MODEL}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model or "gpt-4o-mini"}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)
    
    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-3.5-turbo",
            api_key=api_key or "not-needed",
            base_url=base_url,
        )
    
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": model or "gemini-1.5-flash"}
        if api_key:
            kwargs["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)
    
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        kwargs = {"model": model or "mistral-small-latest"}
        if api_key:
            kwargs["mistral_api_key"] = api_key
        return ChatMistralAI(**kwargs)
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        kwargs = {"model": model or "llama-3.1-8b-instant"}
        if api_key:
            kwargs["groq_api_key"] = api_key
        return ChatGroq(**kwargs)
    
    raise ValueError(f"Unknown provider: {provider}")


# ___ SECTION 4: RETRIEVAL & ANSWERING ___
# FUNCTION 12
def format_retrieval_context(docs):
    """
    Formats retrieved chunks into a numbered context block for the prompt.
    Each chunk is prefixed with a [n] marker so the model can cite by number;
    chunk text is bounded to keep the prompt size in check.

    Args:
        docs: Retrieved chunk Documents, in ranked order.
    
    Returns:
        A single string with one numbered, source-tagged block per chunk.
    """

    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        snippet = (doc.page_content or "").strip()
        snippet = snippet[:2000]                # bounds the prompts
        blocks.append(f"[{i}] (source={src} page={page})\n {snippet}")

    return "\n\n".join(blocks)


# FUNCTION 13
def docs_citations(docs):
    """
    Builds the numbered citation list aligned with the prompt's [n] markers.

    Args:
        docs: Retrieved chunk Documents, in the same order shown to the model.
    
    Returns:
        List of (index, "source (page N)") tuples, numbered from 1.
    """

    out = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        out.append((i, f"{src} (page {page})"))

    return out


# FUNCTION 14
def retrieve_top_k(query: str, k: int = 5):
    """
    Embeds the query and returns the k most similar chunks from the vector store.

    Args:
        query: The user's question.
        k: No. of chunks to retrieve (default: 5)
    
    Returns:
        A list of the top-k matching chunk Documents.
    """

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = get_vectordb(embeddings)

    return vector_db.similarity_search(query, k=k)


# FUNCTION 15
def answer_with_citations(query: str, top_k: int = 5, llm_kwargs=None):
    """
    Retrieves context, asks the LLM to answer using only that context, and returns citations.

    Args:
        query: The user's question.
        top_k: No. of chunks to retrieve as context (default: 5)
        llm_kwargs: Optional provider/model/key overrides passed to build_llm
    
    Returns:
        A tuple (answer_text, citations, docs): the model's answer string, the numbered
        citation list from docs_citations, and the retrieved chunks.
    """

    docs = retrieve_top_k(query, top_k)
    context = format_retrieval_context(docs)
    citations = docs_citations(docs)

    llm = build_llm(**(llm_kwargs or {}))

    prompt = ChatPromptTemplate.from_messages(messages=[
        ("system",
        "Answer the user's question using ONLY the provided context. "
        "If the context is insufficient, say you don't know. "
        "Each context item is numbered (1, 2, 3, ...). When a claim comes from a "
        "context item, cite it inline with that item's number in square brackets, "
        "e.g. [1] or [2]. Use the actual number, never the literal letter n. "
        "Do not include the source filename or page in your answer text; just the "
        "bracketed number. Do not cite sources you don't/didn't use."),
        ("user",
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])

    chain = prompt | llm
    resp = chain.invoke({"question": query, "context": context})

    content = getattr(resp, "content", str(resp))
    if isinstance(content, list):   # some models return content blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", str(block)))
        answer_text = "".join(parts).strip()
    else:
        answer_text = str(content)

    
    return answer_text, citations, docs


# ___ SECTION 5: CLI HELPERS & MAIN ___
# FUNCTION 16
def print_retrieval_preview(docs, limit_chars: int = 200):
    """
    Prints a short, truncated preview of retrieved chunks (mainly for CLI debugging aid).

    Args:
        docs: Retrieved chunk Documents to preview.
        limit_chars: Max characters of each chunk to show (default: 200).
    """

    print("\nRETRIEVAL RESULTS (top-k chunks)")
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        snippet = snippet[:limit_chars] + ("..." if len(snippet) > limit_chars else "")
        print(f"\n{i}) source={src} page={page}\n   {snippet}")


# MAIN FUNCTION
def main():
    """Runs a full pass: ingests PDFs, previews retrieval, and prints a cited answer."""

    added = add_new_pdfs_if_needed()
    print(f"Addition result: added_chunks={added}")

    # retrieval test
    query = os.environ.get(
        "QUERY",
        "What are the key considerations when setting up a trust for estate planning?"
    )
    top_k = int(os.environ.get("TOP_K", "5"))

    docs = retrieve_top_k(query, top_k)
    print_retrieval_preview(docs)

    # retrievalQA with citations
    answer, citations, _ = answer_with_citations(query, top_k)

    print("\nANSWER \n")
    print(answer)

    print("\nDOCUMENTS USED (INDEXED)")
    for i, c in citations:
        print(f"- [{i}], {c}")


if __name__ == "__main__":
    main()
########################################################################
