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
REST API for the DocuRAG pipeline.

Exposes the same retrieval and answering functions the Gradio UI uses, over HTTP, 
allowing other programs to query the document index.

P.S. File contains 3 sections, 5 classes, and 7 functions.
"""


from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException  # framework and error helper
from pydantic import BaseModel, Field  # request/response validation

from . import agent
from . import pipeline as rag


########################################################################
# ___ SECTION 1: SCHEMAS ___
# CLASS 1
class AskRequest(BaseModel):
    """Body for POST/ask."""

    question: str = Field(..., min_length=1, description="The question to asnwer.")
    top_k: int = Field(5, ge=1, le=10, description="Passages to retrieve.")
    mode: Literal["single_pass", "agentic"] = Field(
        "single_pass", description="Fixed retrieval, or let the model decide when to search."
    )
    provider: Optional[str] = Field(None, description="LLM provider override.")
    model: Optional[str] = Field(None, description="Model name override.")


# CLASS 2
class Citation(BaseModel):
    """One numbered source reference."""

    index: int
    label: str


# CLASS 3
class Passage(BaseModel):
    """One retrieved passage with its provennace."""

    index: int
    source: str
    page: str
    text: str


# CLASS 4
class TraceStep(BaseModel):
    """One decision the agent made."""

    step: int
    type: str
    tool: Optional[str] = None
    query: Optional[str] = None
    k: Optional[int] = None


# CLASS 5
class AskResponse(BaseModel):
    """Body returned by POST /ask."""

    answer: str
    mode: str
    citations: List[Citation]
    passages: List[Passage]
    trace: List[TraceStep] = [] # empty in single-pass mode


########################################################################
# ___ SECTON 2: HELPERS ___
# FUNCTION 1
def _to_passages(docs):
    """
    Converts retrieved documents into API response objects.

    Args:
        docs: LangChain Document objects from retrieval.
    
    Returns:
        A list of Passage models, numbered to match the citation markers.
    """

    return [
        Passage(
            index=i + 1,
            source=str(d.metadata.get("source", "unknown")),
            page=str(d.metadata.get("page", "unknown")),
            text=(d.page_content or "").strip(),
        )
        for i, d in enumerate(docs)
    ]


# FUNCTION 2
def _to_trace(trace):
    """
    Flattens agent trace dictionaries into API response objects.

    Args:
        trace: Step dicitonaries from answer_with_agent.
    
    Returns:
        A list of TraceStep models.
    """

    steps = []
    for t in trace or []:
        args = t.get("args", {}) or {}
        steps.append(
            TraceStep(
                step=t.get("step", 0),
                type=t.get("type", "unknown"),
                tool=t.get("took"),
                query=args.get("query"),
                k=args.get("k"),
            )
        )
    
    return steps


# FUNCTION 3
def _llm_kwargs(req):
    """
    Builds the provider override dicitonary from a request.

    Args:
        req: An AskRequest.
    
    Returns:
        A dictionary for build_llm, or None to use defaults.
    """

    kwargs = {}
    if req.provider:
        kwargs["provider"] = req.provider
    if req.model:
        kwargs["model"] = req.model
    
    return kwargs or None # None will use environment defaults


########################################################################
# ___ SECTON 3: APP & ENDPOINTS ___

app = FastAPI(
    title="DocuRAG API",
    description="Ask questions against an indexded PDF corpus and get cited answers.",
    version="0.1.0",
)


# FUNCTION 4
@app.get("/health")
def health():
    """Liveness check and index status."""

    return {
        "status": "ok",
        "index_ready": rag.chroma_collection_is_nonempty(),
    }


# FUNCTION 5
@app.get("/documents")
def list_documents():
    """
    Lists the PDFs currently available to the index.

    Returns:
        The document directory and the filenames in it.
    """
    
    rag.ensure_paths()
    files = sorted(p.name for p in rag.PDF_PATH.glob("*pdf"))

    return {"count": len(files), "documents": files}


# FUNCTION 6
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Answers a question from the indexed documents, with citations.

    Args:
        req: The validated request body.
    
    Returns: 
        An AskResponse with the answer, citations, passages, and (in agentic 
        mode) the sequence of decisions the model made.

    Raises:
        HTTPException: 409 if nothing is indexed, 500 if answering fails.
    """

    if not rag.chroma_collection_is_nonempty():
        raise HTTPException(
            status_code=409,
            detail="No documents are indexed. Add PDFs and ingest before asking.",
        )
    
    try:
        if req.mode == "agentic":
            answer, citations, docs, trace = agent.answer_with_agent(
                req.question, req.top_k, _llm_kwargs(req)
            )
        else:
            answer, citations, docs = rag.answer_with_citations(
                req.question, req.top_k, _llm_kwargs(req)
            )
            trace = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answering failed: {e}")
            # convert an internal exception into a proper HTTP error
    
    return AskResponse(
        answer=answer,
        mode=req.mode,
        citations=[Citation(index=i, label=label) for i, label in citations],
        passages=_to_passages(docs),
        trace=_to_trace(trace),
    )


# FUNCTION 7
@app.post("/ingest")
def ingest():
    """
    Indexes any new or changed PDFs in the document directory.

    Returns:
        How many chunks were added and whether the index is now usable.
    """

    try:
        added = rag.add_new_pdfs_if_needed()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    
    return {
        "chunks_added": added,
        "index_ready": rag.chroma_collection_is_nonempty(),
    }