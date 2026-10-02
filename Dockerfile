# DocuRAG container image
#
# Produces one image that serves either the Gradio UI or the 
# FastAPI layer, selected by the command at runtime.
#
# Uses the ONNX embedding backend (fastembed) rather than PyTorch
# to keep image under ~1GB instead of ~5GB.


########################################################################
# ___ SECTION 1: BUILDING DEPENDENCIES INTO VIRTUALENV ___
FROM python:3.13-slim AS builder

# building tools for compiling wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy to take advantage of docker layer caching
COPY requirements.txt .

# install dependencies and disable pip cache
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ___ SECTION 2: RUNTIME ___
FROM python:3.13-slim

COPY --from=builder /opt/venv /opt/venv
    # cross over the installed packages only, keeping the build essentials behind
ENV PATH="/opt/venv/bin:$PATH"

# create non-root user
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# show logs immediately, prevent .pyc cluttering, onnx backend, 
# gradio host, cache inside appuser's home, prevent telemetry
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDING_BACKEND=onnx \
    GRADIO_HOST=0.0.0.0 \
    FASTEMBED_CACHE_PATH=/home/appuser/.cache/fastembed \
    LANGCHAIN_TRACING_V2=false \
    ANONYMIZED_TELEMETRY=false \
    CHROMA_TELEMETRY_ENABLED=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

# changing ownership at copy time to avoid a separate chown layer
COPY --chown=appuser:appuser docurag/ ./docurag/
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser demo_docs/ ./demo_docs/

RUN mkdir -p /app/pdf_db /app/chroma_db /home/appuser/.cache/fastembed \ 
    && chown -R appuser:appuser /app /home/appuser/.cache

# switch to non-root user for subsequent prefetch run
USER appuser

# model in at build time
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; \
    FastEmbedEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# gradio UI, fastAPI
EXPOSE 7860 8000

# 0.0.0.0 inside container for ui (overridden by api service in compose)
CMD ["python", "app.py"]
