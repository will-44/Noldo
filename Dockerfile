FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv : résolveur Rust ultra-rapide, évite le backtracking pathologique de pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

# PDF.js prébuilt (servi localement pour l'UI ChatPDF → aucun CDN, respecte l'offline)
ARG PDFJS_VERSION=4.0.379
RUN curl -L -o /tmp/pdfjs.zip \
      "https://github.com/mozilla/pdf.js/releases/download/v${PDFJS_VERSION}/pdfjs-${PDFJS_VERSION}-dist.zip" \
    && python -c "import zipfile; zipfile.ZipFile('/tmp/pdfjs.zip').extractall('/app/webapp/static/pdfjs')" \
    && rm /tmp/pdfjs.zip

# Marked (rendu Markdown de la réponse de l'agent) + DOMPurify (assainissement du HTML
# généré, avant injection via innerHTML) : mêmes principes que PDF.js, vendorés en dur.
ARG MARKED_VERSION=13.0.3
ARG DOMPURIFY_VERSION=3.1.6
RUN mkdir -p /app/webapp/static/vendor \
    && curl -L -o /app/webapp/static/vendor/marked.esm.js \
         "https://unpkg.com/marked@${MARKED_VERSION}/lib/marked.esm.js" \
    && curl -L -o /app/webapp/static/vendor/purify.min.js \
         "https://unpkg.com/dompurify@${DOMPURIFY_VERSION}/dist/purify.min.js"

# Docling télécharge ses modèles ML au premier lancement → cache dans un volume
ENV DOCLING_ARTIFACTS_PATH=/app/data/docling_models

# Modèle de reranking (sentence-transformers) : redirige le cache HuggingFace vers le
# volume data/ pour n'être téléchargé qu'une fois (~1,1 Go, BAAI/bge-reranker-base)
ENV HF_HOME=/app/data/hf_cache

# 7860 = Gradio (serve) · 7862 = interface web ChatPDF (webserve)
EXPOSE 7860 7862

CMD ["python", "main.py", "serve"]
