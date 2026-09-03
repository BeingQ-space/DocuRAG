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

This module adds tools and a loop to allow the model to decide
whether and how often to search before answering.

P.S. File contains x sections and y functions.
"""


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from typing import Any, Dict, List, Tuple

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
    "question, say you don't know rather than guessing."
)

MAX_ITERATIONS = 5 # to prevent indefinite tool call

########################################################################
# ___ SECTION 1: TOOLS ___
# FUNCTION 1
@tool
def search_documents(query: str, k: int = 5) :
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