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
Gradio interface for Document-based Q&A RAG pipeline.

Wraps the rag_pipeline module in a web UI with three tabs:
- Ask (query + grounded, cited answers with retrieved passages)
- History (session Q&A with copy/clear/export), and
- Model (runtime provider/model selection). API keys entered here
    are held in session state only, never written to the disk.

P.S. File contains 6 sections and 12 functions.
"""


import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr

import rag_pipeline as rag


########################################################################
MAX_CHUNKS = 10             # max passage boxes pre-built
MAX_HISTORY = 20            # max history slots pre-built
PROVIDERS = [    # list of supported providers
    "anthropic",
    "openai",
    "google",
    "mistral",
    "groq",
    "openai_compatible",
    "ollama",
]
########################################################################


########################################################################
# ___ SECTION 1: HELPERS (BADGE, FORMATTING) ___
# FUNCTION 1
def _badge(llm_kwargs):
    """
    Visual green pill showing the current provider/model being used.

    Args:
        llm_kwargs: Session LLM override dictionary (may be empty for defaults).
    
    Returns:
        An HTML string for the gr.HTML badge component.
    """

    provider, model = rag.describe_active_model(llm_kwargs)

    return (
        "<div style='display:inline-block;padding:4px 12px;border-radius:12px;"
        "background:#e6f4ea;color:#137333;font-weight:600;font-size:0.85rem;"
        "border:1px solid #137333;'>"
        f"&#9679; CURRENTLY ACTIVE: {provider} &middot; {model}</div>"
    )


# FUNCTION 2
def _sources_to_plain(sources_md):
    """
    Strips Markdown markers from a sources block for plain-text/copy output.

    Args:
        sources_md: The Markdown-formatted sources string.
    
    Returns:
        The same content with heading/bold markers removed.
    """

    return sources_md.replace("### ", "").replace("**", "")


# ___ SECTION 2: INGESTION & SETTINGS HANDLERS ___
# FUNCTION 3
def add_uploaded_files(files):
    """
    Copies uploaded PDFs into PDF_PATH and (re-)index any new/changed ones.
    Non-PDF uploads are skipped and reported; indexing reuses the pipeline's
    manifest logic so unchanged files aren't re-embedded.

    Args:
        files: List of uploaded file objects/paths from the gr.File component.
    
    Returns:
        A status string summarizing what was added, skipped, or indexed.
    """

    if not files:
        return "No files selected. Add PDF(s), or simply ask a question against the indexed doc."

    rag.ensure_paths()
    copied = []
    skipped = []

    for f in files:
        src = Path(f.name if hasattr(f, "name") else f)
        if src.suffix.lower() != ".pdf":
            skipped.append(f"{src.name} (not a PDF)")
            continue
        dest = rag.PDF_PATH / src.name
        shutil.copy(src, dest)
        copied.append(src.name)

    if not copied:
        detail = "; ".join(skipped) if skipped else "nothing usable"
        return f"Nothing added - {detail}."
    
    added = rag.add_new_pdfs_if_needed()
    msg = f"Added {len(copied)} file(s): {', '.join(copied)}. Added {added} new chunk(s) to the index."
    if skipped:
        msg += f" Skipped: {', '.join(skipped)}."
    
    return msg


# FUNCTION 4
def apply_settings(provider, model, api_key, base_url):
    """
    Builds the session LLM-override dictionary from the Model tab inputs.

    Args:
        provider: Selected provider key.
        model: Optional model name (blank uses the provider default).
        api_key: Optional session API key (blank uses the environment).
        base_url: Endpoint URL, required only for 'openai_compatible'.
    
    Returns:
        A tuple (llm_kwargs, status_markdown, badge_html) for the bound outputs.
    """

    provider = (provider or "anthropic").strip()
    llm_kwargs = {"provider": provider}

    if model and model.strip():
        llm_kwargs["model"] = model.strip()
    if api_key and api_key.strip():
        llm_kwargs["api_key"] = api_key.strip()
    if base_url and base_url.strip():
        llm_kwargs["base_url"] = base_url.strip()
    
    if provider == "openai_compatible" and "base_url" not in llm_kwargs:
        return (
            llm_kwargs, 
            "Set a Base URL for an OpenAI-compatible endpoint.",
            _badge(llm_kwargs),
            )

    shown_model = llm_kwargs.get("model", "(provider default)")
    key_note = "key set for this session" if "api_key" in llm_kwargs else "using environment key"
    status = f"Active: **{provider}**, model {shown_model} - {key_note}."

    return llm_kwargs, status, _badge(llm_kwargs)


# FUNCTION 5
def clear_session_keys(provider, model, base_url):
    """
    Resets session LLM overrides to defaults and clears the API key textbox.

    Args:
        provider: Current provider selection (unused; kept for binding).
        model: Current model text (unused; kept for binding).
        base_url: Current base URL text (unused; kept for binding).
    
    Returns:
        A tuple (empty_kwargs, status_markdown, default_badge, cleared_key_box).
    """

    return (
        {},             # llm_kwargs_state reset to defaults
        "Session keys cleared. Back to environment/default credentials.",
        _badge({}),     # badge back to default
        "",             # clears the API key textbox
    )


# ___ SECTION 3: QUERY HANDLER ___
# FUNCTION 6
def ask(question, top_k, history, llm_kwargs):
    """
    Runs a query; returns answer, sources, passage boxes, and updated history.

    Args:
        question: The user's question text.
        top_k: No. of passages to retrieve.
        history: Current history list (newest first).
        llm_kwargs: Session provider/model/key overrides (may be empty).
    
    Returns:
        A tuple of (answer_md, sources_md, *MAX_CHUNKS passage-box updates, history,
        badge_html) matching the ask_outputs binding.
    """

    question = (question or "").strip()
    if not question:
        return ( 
            "Enter a question to get started.", 
            "",
            *[gr.update(visible=False, value="") for _ in range(MAX_CHUNKS)],
            history,
            _badge(llm_kwargs),
        )
    
    if not rag.chroma_collection_is_nonempty():
        return (
            "No documents are indexed yet. Upload a PDF above, or place PDFs in "
            f"`{rag.PDF_PATH}/` and ask again.",
            "",
            *[gr.update(visible=False, value="") for _ in range(MAX_CHUNKS)],
            history,
            _badge(llm_kwargs),
        )
    
    try:
        answer, citations, docs = rag.answer_with_citations(
            question, int(top_k), llm_kwargs or None
            )
    except Exception as e:
        return (
            f"Something went wrong while answering: {e}", 
            "",
            *[gr.update(visible=False, value="") for _ in range(MAX_CHUNKS)],
            history,
            _badge(llm_kwargs),
            )
    
    # build the numbered source list
    source_lines = [f"**[{i}]** {label}" for i, label in citations]
    sources_md = "### Sources\n\n" + "\n\n".join(source_lines) if source_lines else ""

    box_updates = []
    for i in range(MAX_CHUNKS):
        if i < len(docs):
            d = docs[i]
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "unknown")
            box_updates.append(
                gr.update(
                    visible=True,
                    label=f"[{i + 1}] {src} - page {page}",
                    value=(d.page_content or "").strip(),
                )
            )
        else:
            box_updates.append(gr.update(visible=False, value=""))
    
    entry = {
        "question": question,
        "answer": answer,
        "sources_md": sources_md,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    history = ([entry] + history) if history else [entry]

    return answer, sources_md, *box_updates, history, _badge(llm_kwargs)


# ___ SECTION 4: HISTORY HANDLERS ___
# FUNCTION 7
def render_history(history):
    """
    Produces accordion/body/copy-box updates for each pre-built history slot.
    
    Args:
        history: Session history list (newest first).
    
    Returns:
        A flat list of gr.update objects, three per slot (accordion, body, copy box),
        for MAX_HISTORY slots.
    """

    updates = []
    for i in range(MAX_HISTORY):
        if history and i < len(history):
            e = history[i]
            label = f"{e['timestamp']} - {e['question'][:70]}"
            body = (
                f"**Q:** {e['question']}\n\n"
                f"**A:**\n\n{e['answer']}\n\n"
                f"{e['sources_md']}"
            )
            copy_text = (
                f"Q: {e['question']}\n\n"
                f"A:\n{e['answer']}\n\n"
                f"{_sources_to_plain(e['sources_md'])}"
            )
            updates.append(gr.update(visible=True, label=label, open=(i == 0)))
            updates.append(gr.update(value=body))
            updates.append(gr.update(value=copy_text))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=""))
            updates.append(gr.update(value=""))
        
    return updates


# FUNCTION 8
def clear_history():
    """
    Empties the history and collapses all history slots.

    Returns:
        A list [empty_history, status_markdown, *slot_updates] matching the
        clear button's bound outputs.
    """

    updates = []
    for _ in range(MAX_HISTORY):
        updates.append(gr.update(visible=False))
        updates.append(gr.update(value=""))
        updates.append(gr.update(value=""))
    
    return [[], "History cleared.", *updates]


# ___ SECTION 5: EXPORT HANDLERS ___
# FUNCTION 9
def _history_to_text(history, scope):
    """
    Renders history to a plain-text export string.

    Args:
        history: Session history list (newest first).
        scope: "Latest Q&A only" for just the newest entry, else all entries.
    
    Returns:
        The formatted plain-text export (or a placeholder if empty).
    """

    entries = history[:1] if scope == "Latest Q&A only" else history
    blocks = []
    
    for e in entries:
        blocks.append(
            f"[{e['timestamp']}]\n"
            f"Q: {e['question']}\n\n"
            f"A:\n{e['answer']}\n\n"
            f"{_sources_to_plain(e['sources_md'])}\n"
            + ("-" * 60)
        )
    
    return "\n\n".join(blocks) if blocks else "No history yet."


# FUNCTION 10
def _history_to_markdown(history, scope):
    """
    Renders history to a Markdown export string.

    Args:
        history: Session history list (newest first).
        scope: "Latest Q&A only" for just the newest entry, else all entries.
    
    Returns:
        The formatted Markdown export (or a placeholder if empty).
    """

    entries = history[:1] if scope == "Latest Q&A only" else history
    blocks = ["# Q&A History\n"]
    
    for e in entries:
        blocks.append(
            f"## {e['question']}\n\n"
            f"[*{e['timestamp']}*]\n\n"
            f"{e['answer']}\n\n"
            f"{e['sources_md']}\n"
        )
    
    return "\n\n".join(blocks) if len(blocks) > 1 else "# Q&A History\n\n_No history yet._"


# FUNCTION 11
def export_text(history, scope, fmt):
    """
    Writes the selected history to a temp .txt or .md file for download.

    Args:
        history: Session history list.
        scope: Export scope ("All history" or "Latest Q&A only").
        fmt: Format being "Markdown (.md)" or "Plain text (.txt)".
    
    Returns:
        The temp file path to serve for download, or None if history is empty.
    """

    if not history:
        return None
    
    if fmt == "Markdown (.md)":
        content, suffix = _history_to_markdown(history, scope), ".md"
    else:
        content, suffix = _history_to_text(history, scope), ".txt"
    
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8", prefix="qa_history_"
    )
    tmp.write(content)
    tmp.close()

    return tmp.name


# ___ SECTION 6: UI ASSEMBLY & LAUNCH ___
# FUNCTION 12
def build_ui():
    """
    Constructs and returns the Gradio Blocks app (tabs, components, binding).

    Returns:
        The assembled gr.Blocks demo, ready to .launch().
    """

    with gr.Blocks(title="Document Q&A RAG") as demo:
        history_state = gr.State(value=[])
        llm_kwargs_state = gr.State({}) # provider/mode/key for session

        gr.Markdown(
            "# Document Q&A\n"
            "Ask questions and get answers grounded in the indexed documents "
            "with citations that link back to the exact retrieved passages."
        )

        with gr.Tabs():
            # Tab for Q&A
            with gr.Tab("Ask"):
                model_badge = gr.HTML(_badge({}))

                with gr.Accordion("Add documents (optional)", open=False):
                    uploader = gr.File(
                        label="Upload PDFs to index",
                        file_count="multiple",
                        file_types=[".pdf"],
                    )
                    add_btn = gr.Button("Add to index")
                    addition_status = gr.Markdown()
                
                with gr.Row():
                    question = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What is the difference between a revocable and irrevocable trust?",
                        scale=4,
                        lines=2,
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=MAX_CHUNKS, value=5, step=1,
                        label="Passages to retrieve",
                        scale=1,
                    )
                ask_btn = gr.Button("Ask", variant="primary")

                answer_out = gr.Markdown(label="Answer")

                gr.Markdown("---")
                with gr.Accordion("Sources & retrieved passages", open=True):
                    sources_out = gr.Markdown()
                    gr.Markdown(
                        "_Each box below is a passage the retriever returned. "
                        "Its number matches the bracketed citation (e.g. [1]) in the answer above._"
                    )
                    chunk_boxes = [
                        gr.Textbox(visible=False, lines=6, interactive=False, buttons=["copy"])
                        for _ in range(MAX_CHUNKS)
                    ]
            
            # Tab for history
            with gr.Tab("History"):
                with gr.Row():
                    export_scope = gr.Radio(
                        choices=["All history", "Latest Q&A only"],
                        value="All history",
                        label="Export scope",
                    )
                    export_fmt = gr.Radio(
                        choices=["Markdown (.md)", "Plain text (.txt)"],
                        value="Markdown (.md)",
                        label="Text format",
                    )
                with gr.Row():
                    export_text_btn = gr.Button("Export text")
                    clear_btn = gr.Button("Clear history", variant="stop")
                export_file = gr.File(label="Download", interactive=False)
                history_status = gr.Markdown()

                gr.Markdown("---")

                hist_accordions, hist_bodies, hist_copies = [], [], []
                for i in range(MAX_HISTORY):
                    with gr.Accordion(visible=False, open=False, label="") as acc:
                        body = gr.Markdown()
                        copy = gr.Textbox(
                            label="Copy this entry",
                            lines=4,
                            interactive=False,
                            buttons=["copy"],
                        )
                    hist_accordions.append(acc)
                    hist_bodies.append(body)
                    hist_copies.append(copy)
            
            # Tab for model selection
            with gr.Tab("Model"):
                gr.Markdown(
                    "### Model provider\n"
                    "Choose a provider and (optionally) a model. Keys are held for "
                    "this session only — never written to disk or logged or sent outside "
                    "this app. Leave the key blank to use whatever is already in your "
                    "environment.\n\n"
                    "For **openai_compatible**, set the Base URL (works with Together, Fireworks, "
                    "Groq, DeepSeek, OpenRouter, local vLLM / LM Studio, etc.). For **ollama**, "
                    "no key is needed — just the local model name.\n\n"
                    "_A provider only works if its integration package is installed (e.g. `pip "
                    "install langchain-openai`). If it isn't, you'll get an actionable error when you ask._"
                )
                provider_dd = gr.Dropdown(
                    choices=PROVIDERS, value="anthropic", label="Provider"
                )
                model_tb = gr.Textbox(
                    label="Model (optional - blank uses your provider default)",
                    placeholder="e.g. claude-sonnet-5 / gpt-4o-mini / gemini-1.5-flash",
                )
                base_url_tb = gr.Textbox(
                    label="Base URL (openai_compatible only)",
                    placeholder="e.g. https://api.together.xyz/v1",
                )
                api_key_tb = gr.Textbox(
                    label="API key (session only)", type="password",
                    placeholder="Leave blank to use environment variable",
                )

                with gr.Row():
                    apply_btn = gr.Button("Apply Changes", variant="primary")
                    clear_keys_btn = gr.Button("Clear session keys", variant="stop")
                settings_status = gr.Markdown("Active: **anthropic** (environment key).")

        hist_outputs = []
        for i in range(MAX_HISTORY):
            hist_outputs += [hist_accordions[i], hist_bodies[i], hist_copies[i]]


        # Binding
        add_btn.click(add_uploaded_files, inputs=[uploader], outputs=[addition_status])

        apply_btn.click(
            apply_settings,
            inputs=[provider_dd, model_tb, api_key_tb, base_url_tb],
            outputs=[llm_kwargs_state, settings_status, model_badge]
        )
        clear_keys_btn.click(
            clear_session_keys,
            inputs=[provider_dd, model_tb, base_url_tb],
            outputs=[llm_kwargs_state, settings_status, model_badge, api_key_tb],
        )
        
        ask_outputs = [answer_out, sources_out, *chunk_boxes, history_state, model_badge]
        ask_btn.click(
            ask, 
            inputs=[question, top_k, history_state, llm_kwargs_state], 
            outputs=ask_outputs,
        )
        question.submit(
            ask, 
            inputs=[question, top_k, history_state, llm_kwargs_state], 
            outputs=ask_outputs,
        )

        ask_btn.click(render_history, inputs=[history_state], outputs=hist_outputs)
        question.submit(render_history, inputs=[history_state], outputs=hist_outputs)

        export_text_btn.click(
            export_text,
            inputs=[history_state, export_scope, export_fmt],
            outputs=[export_file],
        )
        clear_btn.click(
            clear_history, 
            inputs=[], 
            outputs=[history_state, history_status, *hist_outputs]
        )
    
    return demo


if __name__ == "__main__":
    build_ui().launch()
    