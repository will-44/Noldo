# Zotero RAG — Interrogation en langage naturel de la bibliothèque Zotero

Pipeline RAG local sur Jetson AGX Orin : interroge en français une bibliothèque
d'articles scientifiques Zotero, avec citations (auteur, année, page) et surlignage
des passages dans le PDF original. 100 % local (Ollama), aucune clé API cloud pour
l'inférence.

## Architecture

```
Zotero (API web, métadonnées)  ─┐
PDFs locaux (/data/zotero/files)─┴─> Docling (parse) ─> chunks ─> nomic-embed-text
                                                                        │
                                                                        v
   Réponse + sources <─ mistral-small3.2 <─ retrieval <─ ChromaDB (vecteurs)
                                                                        │
                                            highlighter (PyMuPDF) <──────┘
```

- **Ollama natif** (hors Docker) sert les modèles sur `http://host.docker.internal:11434`
  (configuré pour écouter sur `0.0.0.0`, modèles stockés sur `/data/ollama/models`).
- **Le conteneur RAG** lit les PDFs directement depuis `/data/zotero/files` (montés en
  lecture seule) — pas de re-téléchargement quand le fichier est déjà sur le Jetson.

## Prérequis (déjà en place sur ce Jetson)

- Ollama natif avec les modèles : `mistral-small3.2`, `nomic-embed-text`
- Image Docker `zotero-rag:latest` (torch **CPU-only**, ~2.75 Go)
- Modèles Docling pré-téléchargés dans `data/docling_models/` (layout + tableformer)

## Configuration — `config.yaml`

Points clés (déjà renseignés) :

```yaml
zotero:
  mode: "web"                    # métadonnées via api.zotero.org
  web_api_key: "<clé>"
  library_id: "12014782"
  local_files_dir: "/zotero_files"  # PDFs locaux montés dans le conteneur
ollama:
  base_url: "http://host.docker.internal:11434"
  llm_model: "mistral-small3.2"
  embed_model: "nomic-embed-text"
  timeout: 300
```

> **Changer de modèle LLM** = une ligne (`llm_model`). Ex. `llama3.1:8b` pour des
> réponses plus rapides (~5-10 tok/s vs ~2 tok/s pour mistral-small3.2 24B).
> Ne pas utiliser `llama3.1:70b` : ~13 min/réponse sur ce Jetson (GPU intégré).

## Lancer le service

```bash
cd /data/zotero-rag
docker compose up -d            # UI Gradio sur http://<jetson>:7860
docker compose logs -f          # suivre les logs
docker compose down             # arrêter
```

Accès UI :
- LAN : `http://192.168.50.12:7860`
- Tailscale : `http://100.100.84.81:7860`

## Construire / mettre à jour l'index

Via l'UI (onglet « Gestion de l'index ») ou en CLI :

```bash
docker compose exec zotero-rag python main.py index    # index complet (~237 PDFs)
docker compose exec zotero-rag python main.py update   # mise à jour incrémentale
docker compose exec zotero-rag python main.py stats    # statistiques
docker compose exec zotero-rag python main.py query "ta question"
```

> Le parsing Docling tourne sur **CPU** (~1-2 min/PDF). L'index complet des 237 articles
> prend donc un certain temps ; il est persistant (`data/chroma_db/`) et incrémental
> ensuite (seuls les items modifiés sont reparsés).

## Tests

```bash
docker run --rm -v /data/zotero-rag:/app -w /app zotero-rag:latest \
  sh -c "pip install -q pytest && python -m pytest tests/ -q"
# 10 passed
```

## Reconstruire l'image

```bash
cd /data/zotero-rag && docker build -t zotero-rag:latest .
```

Le Dockerfile utilise **uv** (résolveur rapide) et épingle `torch==2.8.0` /
`torchvision==0.23.0` en **CPU-only** : sur aarch64 ces wheels n'entraînent aucune
dépendance `nvidia-*` (gated `platform_machine=="x86_64"`). Ne pas monter torch ≥ 2.9
(wheels aarch64 CUDA ~10 Go inutiles ici, Docling tourne sur CPU).

## Structure

```
zotero-rag/
├── config.yaml          docker-compose.yml   Dockerfile   requirements.txt
├── main.py              CLI (click)
├── app.py               UI Gradio (2 onglets)
├── zotero_rag/
│   ├── zotero_client.py  API Zotero + extraction des zips locaux
│   ├── pdf_parser.py     Docling → documents structurés (page, section)
│   ├── indexer.py        ChromaDB + LlamaIndex + état incrémental
│   ├── retriever.py      query engine + prompt FR + déduplication sources
│   ├── highlighter.py    surlignage PyMuPDF du passage dans le PDF
│   └── utils.py
├── data/
│   ├── pdfs/  chroma_db/  highlighted/  docling_models/
└── tests/
```

## Notes / pièges rencontrés

- **torch CPU-only** : indispensable de pinner ≤ 2.8.x (voir « Reconstruire »).
- **Ollama** : doit écouter sur `0.0.0.0` (override systemd) et stocker ses modèles
  sur `/data` (l'eMMC `/` de 54 Go sature) — déjà configuré.
- **Modèles Docling** : pré-téléchargés dans `data/docling_models` car
  `DOCLING_ARTIFACTS_PATH` impose un dossier déjà peuplé.
