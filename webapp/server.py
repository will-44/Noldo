"""Service FastAPI — interface ChatPDF (chat central | historique | visionneuse PDF | sync).

Réutilise le package zotero_rag (RAGRetriever, RAGAgent, RAGIndexer) ; ne modifie pas le
pipeline d'indexation existant. Lancé via `python main.py webserve`.
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from zotero_rag.agent import RAGAgent
from zotero_rag.indexer import RAGIndexer
from zotero_rag.retriever import MAX_HISTORY_TURNS, RAGRetriever

from .store import ConversationStore
from .sync_job import SyncJob

STATIC_DIR = Path(__file__).parent / "static"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


class QueryRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    item_key: str | None = None
    # Champs purement diagnostiques (journal de debug côté agent) : état réel de l'UI,
    # n'influencent aucune logique de récupération. item_key reste lui un filtre dur (voir
    # RAGAgent.run_stream : prime sur tout item_key choisi par le LLM).
    scope_locked: bool = False
    selected_item_key: str | None = None


class RenameRequest(BaseModel):
    title: str


def create_app(config: dict) -> FastAPI:
    app = FastAPI(title="Zotero Chat")
    retriever = RAGRetriever(config)
    agent = RAGAgent(config, retriever)
    pdf_dir = Path(config["rag"]["pdf_cache_dir"])
    store = ConversationStore(Path(config["rag"]["persist_dir"]).parent / "conversations.db")
    sync_job = SyncJob()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/docs")
    def list_docs():
        return retriever.list_documents()

    @app.get("/api/pdf/{item_key}")
    def serve_pdf(item_key: str):
        pdf_path = pdf_dir / f"{item_key}.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF non disponible")
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )

    # ── Conversations ────────────────────────────────────────────────────

    @app.get("/api/conversations")
    def list_conversations():
        return store.list()

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: int):
        conv = store.get(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation introuvable")
        return conv

    @app.patch("/api/conversations/{conversation_id}")
    def rename_conversation(conversation_id: int, req: RenameRequest):
        if store.get(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation introuvable")
        store.rename(conversation_id, req.title.strip() or "Sans titre")
        return {"ok": True}

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: int):
        store.delete(conversation_id)
        return {"ok": True}

    @app.post("/api/query")
    def query(req: QueryRequest):
        conv_id = req.conversation_id
        is_new = conv_id is None

        def event_stream():
            nonlocal conv_id
            if is_new:
                conv_id = store.create()
                yield _sse({
                    "type": "conversation", "id": conv_id, "title": store.get(conv_id)["title"],
                })

            # Chargé AVANT d'ajouter la question courante : recent_history() ignore un message
            # user sans réponse assistant appariée, donc l'ordre importe peu ici en pratique,
            # mais lire avant écrire évite toute ambiguïté si cette logique évolue.
            history = store.recent_history(conv_id, n=MAX_HISTORY_TURNS)
            is_first_turn = not history
            store.add_message(conv_id, "user", req.question)

            # Générateur synchrone : Starlette l'exécute dans un threadpool
            # (iterate_in_threadpool), donc les appels bloquants d'ollama.Client ne gèlent
            # pas la boucle uvloop — même raison que use_async=False sur QueryFusionRetriever.
            for event in agent.run_stream(
                req.question,
                history=history,
                item_key=req.item_key,
                ui_scope_checked=req.scope_locked,
                ui_selected_item_key=req.selected_item_key,
            ):
                yield _sse(event)
                if event["type"] == "done":
                    store.add_message(
                        conv_id, "assistant", event["answer"],
                        meta={
                            "sources": event["sources"],
                            "external_sources": event["external_sources"],
                        },
                    )
                    if is_first_turn:
                        # Après le "done" (déjà envoyé) : ne retarde pas la réponse visible.
                        title = agent.generate_title(req.question, event["answer"])
                        store.rename(conv_id, title)
                        yield _sse({"type": "title", "id": conv_id, "title": title})

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    # ── Synchronisation Zotero (indexation incrémentale depuis l'UI) ────────
    # Job en mémoire (SyncJob), découplé de toute requête HTTP : survit à la fermeture de
    # l'onglet qui l'a lancé, et son état reste interrogeable (GET) après coup — voir
    # webapp/sync_job.py pour le pourquoi (l'ancien mécanisme SSE ne permettait ni l'un ni
    # l'autre, et laissait deux syncs concurrentes se chevaucher sans garde-fou).

    def _run_sync(progress_cb) -> int:
        indexer = RAGIndexer(config)
        added = indexer.update_index(progress_cb=progress_cb)
        if added > 0:
            # Le retriever hybride garde son BM25Retriever figé sur le contenu chargé au
            # démarrage : sans reconstruction, les nouveaux chunks resteraient invisibles à
            # search_corpus/scan_corpus jusqu'au prochain redémarrage du conteneur. Fait ici
            # (dans le job, pas dans une réponse HTTP) pour que ça se produise même si personne
            # n'est connecté pour le voir.
            retriever.fused_retriever = retriever._build_fused_retriever()
        return added

    @app.post("/api/index/sync")
    def start_sync():
        sync_job.start(_run_sync)
        return sync_job.snapshot()

    @app.get("/api/index/sync")
    def get_sync_status():
        return sync_job.snapshot()

    return app
