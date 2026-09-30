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
Tool-calling agent loop.

This module adds a tool and a loop to allow the model to decide
whether and how often to search before answering.

P.S. File contains 2 sections and 3 functions.
"""


from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import rag_pipeline as rag


########################################################################

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. You have one tool, `search_documents`, "
    "which retrieves passages from the user's indexed PDFs. Call it whenever "
    "you need information from the documents to answer. You may call it more "
    "than once (to refine a query that returned weak results) or to look up "
    "a separate sub-question. When you have enough context, answer using ONLY "
    "the retrieved passages, citing each claim inline with that passage's "
    "bracketed number, e.g. [1]. If nothing you retrieved answers the "
    "question, say you don't know rather than guessing. Do not include "
    "the source filename or page in your answer text; just the "
    "bracketed number. Do not cite sources you don't/didn't use."
)

MAX_ITERATIONS = 5 # to prevent indefinite tool call

########################################################################
# ___ SECTION 1: TOOLS ___
# FUNCTION 1
@tool
def search_documents(query: str, k: int = 5) -> str:
    """
    Search the indexed PDF documents for passages relevant to a query.

    Args:
        query: The natural-language query. Rephrase the user's question
            for retrieval if that gets better results.
        k: Number of passages to retrieve (5 default, 10 max).

    Returns:
        A numbered block of retrieved passages, each tagged with source
        filename and page number.
    """

    k = max(1, min(int(k), 10))
    docs = rag.retrieve_top_k(query, k)

    return rag.format_retrieval_context(docs)

########################################################################
# ___ SECTION 2: AGENT LOOP ___
# FUNCTION 2
def _extract_text(response):
    """
    Extracts plain text from a model response, whether it's provided as
    a string or a collection of content blocks.

    Args:
        response: An AIMessage from the model.
    
    Returns:
        The response's text content as a single string.
    """

    content = getattr(response, "content", str(response))
    if not isinstance(content, list): # if content is not list
        return str(content)
    
    parts = [] # to store text extracted from each item (block) in content list
    for block in content:
        if isinstance(block, dict): # if current block is a dictionary
            parts.append(block.get("text", ""))
        else: # if current block is maybe custom object
            parts.append(getattr(block, "text", str(block)))
    
    return "".join(parts).strip()


# FUNCTION 3
def answer_with_agent(query: str, top_k: int = 5, llm_kwargs=None, 
                        max_iterations: int = MAX_ITERATIONS):
    """
    Runs the tool-calling loop where the model decides when to search and
    when to answer, and may search more than once.

    Args:
        query: The user's question.
        top_k: Default k when the model's tool call omits it.
        llm_kwargs: Provider/model/key overrides for build_llm.
        max_iterations: Cap/limit on model<->tool round trips.
    
    Returns:
        A tuple (answer_text, citations, docs, trace).
    """
    
    llm = rag.build_llm(**(llm_kwargs or {}))
    llm_with_tools = llm.bind_tools([search_documents])

    messages = [SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query)] # initialize conversation history
    trace = [] # to store the execution steps, incl. model and tool calls
    all_docs = [] # to store all documents returned by searches
    seen = set() # to track duplicate passages returned by multiple searches

    # run agent loop until stated limit
    for step in range(1, max_iterations + 1):
        response = llm_with_tools.invoke(messages) # get model's response using conv. history
        messages.append(response) 

        tool_calls = getattr(response, "tool_calls", None) or [] 
            # get model's tool calls, or empty list if there are none

        if not tool_calls:
            trace.append({"step": step, "type": "final_answer"}) # record that the final answer was produced
            return _extract_text(response), rag.docs_citations(all_docs), all_docs, trace
                # return answer, citations, docs, and execution trace
        
        # tool execution: process each tool call
        for call in tool_calls:
            name = call.get("name") # tool's name
            args = call.get("args", {}) or {} # tool arguments or empty dictionary
            q = args.get("query", query) # requested or original user query
            k = int(args.get("k") or top_k) # no. of top similar docs to retrieve

            trace.append({"step": step, "type": "tool_call", 
                            "tool": name, "args": {"query": q, "k": k}})
                # record tool call and its arguments
            
            # handle document-search requests
            if name == "search_documents":
                k = max(1, min(k, 10)) # limit range 1 through 10
                docs = rag.retrieve_top_k(q, k) # get matching docs

                # check each retrieved doc
                for d in docs:
                    key = (d.metadata.get("source"), 
                                d.metadata.get("page"), 
                                d.page_content[:50])
                        # key for identifying duplicate docs
                    
                    # add only unseen docs
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(d)
                
                result = rag.format_retrieval_context(docs)
                    # formats search results for model
            else:
                result = f"Unknown tool requested: {name}"
                    # handles unsupported tool names
            
            messages.append(ToolMessage(content=result, 
                                        tool_call_id=call.get("id")))
                # add tool result to the conversation
    
    # handling when the iteration limit is reached
    trace.append({"step": max_iterations, "type": "max_iterations_reached"})
    fallback = (
        "I couldn't settle on an answer within the allowed number of search "
        "steps. Here is the most recent context retrieved, unverified:\n\n"
        + rag.format_retrieval_context(all_docs[-top_k:])
    )
    return fallback, rag.docs_citations(all_docs), all_docs, trace
        # returns fallback, citations, collected docs, and execution trace