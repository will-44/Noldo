import threading
from pathlib import Path

import gradio as gr

from zotero_rag import PDFHighlighter, RAGIndexer, RAGRetriever
from zotero_rag.utils import get_logger, load_config

logger = get_logger(__name__)


def build_app(config: dict) -> gr.Blocks:
    indexer = RAGIndexer(config)
    retriever = RAGRetriever(config)
    highlighter = PDFHighlighter()
    highlighted_dir = Path(config["rag"]["highlighted_dir"])

    # ------------------------------------------------------------------ #
    #  Onglet 1 — Interroger la bibliothèque                              #
    # ------------------------------------------------------------------ #

    def do_query(question: str):
        if not question.strip():
            return "Veuillez saisir une question.", "", []

        response = retriever.query(question)

        # Format sources
        sources_md = ""
        file_list = []
        for src in response.sources:
            authors_short = src.authors.split(",")[0] if src.authors else "?"
            year = src.year or "n.d."
            score_pct = f"{src.score * 100:.1f}%"
            sources_md += (
                f"**{src.title}**  \n"
                f"{src.authors} ({year})  —  page {src.page_number}  —  score {score_pct}\n\n"
            )
            if src.doi:
                sources_md += f"DOI: {src.doi}  \n"
            sources_md += "---\n"

        return response.answer, sources_md, response.sources

    def do_highlight(sources_state):
        if not sources_state:
            return "Aucune source à surligner.", []
        paths = highlighter.highlight_sources(sources_state, highlighted_dir)
        files = [str(p) for p in paths.values() if p.exists()]
        msg = f"{len(files)} PDF(s) surligné(s) généré(s)."
        return msg, files

    # ------------------------------------------------------------------ #
    #  Onglet 2 — Gestion de l'index                                     #
    # ------------------------------------------------------------------ #

    _indexing_lock = threading.Lock()

    def get_stats() -> str:
        s = indexer.get_index_stats()
        return (
            f"**Articles indexés :** {s['nb_documents']}  \n"
            f"**Chunks vectoriels :** {s['nb_chunks']}  \n"
            f"**Dernière mise à jour :** {s['last_update'] or 'Jamais'}  \n"
            f"**Version bibliothèque Zotero :** {s['library_version']}"
        )

    def run_update(log_box: str):
        if not _indexing_lock.acquire(blocking=False):
            return log_box + "\n⚠️  Une indexation est déjà en cours."
        log_lines = [log_box, "🔄 Mise à jour incrémentale en cours…"]
        try:
            messages = []

            def cb(current, total, msg):
                messages.append(msg)

            n = indexer.update_index(progress_cb=cb)
            log_lines += messages
            log_lines.append(f"✅ {n} article(s) mis à jour.")
        except Exception as e:
            log_lines.append(f"❌ Erreur : {e}")
        finally:
            _indexing_lock.release()
        return "\n".join(log_lines)

    def run_full_rebuild(log_box: str, confirm: bool):
        if not confirm:
            return log_box + "\n⚠️  Cochez la case de confirmation d'abord."
        if not _indexing_lock.acquire(blocking=False):
            return log_box + "\n⚠️  Une indexation est déjà en cours."
        log_lines = [log_box, "🔨 Reconstruction complète de l'index…"]
        try:
            messages = []

            def cb(current, total, msg):
                messages.append(msg)

            items = indexer.zotero_client.get_items_with_pdfs()
            # Force re-index by clearing state
            indexer.state_file.write_text('{"library_version":0,"indexed_items":{},"last_update":null}')
            indexer.build_index(items=items, progress_cb=cb)
            log_lines += messages
            log_lines.append("✅ Index reconstruit.")
        except Exception as e:
            log_lines.append(f"❌ Erreur : {e}")
        finally:
            _indexing_lock.release()
        return "\n".join(log_lines)

    # ------------------------------------------------------------------ #
    #  Assemblage Gradio                                                  #
    # ------------------------------------------------------------------ #

    with gr.Blocks(title="Zotero RAG", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📚 Zotero RAG — Interroger votre bibliothèque")

        with gr.Tab("💬 Interroger la bibliothèque"):
            with gr.Row():
                question_box = gr.Textbox(
                    label="Question",
                    placeholder="Ex : Quelles méthodes de planification de mouvement sont utilisées pour les bras robotiques ?",
                    lines=3,
                    scale=4,
                )
                send_btn = gr.Button("Envoyer", variant="primary", scale=1)

            answer_box = gr.Markdown(label="Réponse")

            with gr.Accordion("Sources utilisées", open=True):
                sources_md_box = gr.Markdown()

            with gr.Row():
                highlight_btn = gr.Button("📄 Générer PDFs surlignés")
                highlight_status = gr.Textbox(label="Statut", interactive=False)

            highlighted_files = gr.File(
                label="PDFs surlignés",
                file_count="multiple",
                interactive=False,
            )

            sources_state = gr.State([])

            send_btn.click(
                fn=do_query,
                inputs=[question_box],
                outputs=[answer_box, sources_md_box, sources_state],
            )
            question_box.submit(
                fn=do_query,
                inputs=[question_box],
                outputs=[answer_box, sources_md_box, sources_state],
            )
            highlight_btn.click(
                fn=do_highlight,
                inputs=[sources_state],
                outputs=[highlight_status, highlighted_files],
            )

        with gr.Tab("⚙️ Gestion de l'index"):
            stats_box = gr.Markdown(value=get_stats)

            with gr.Row():
                refresh_btn = gr.Button("🔃 Rafraîchir les stats")
                update_btn = gr.Button("📥 Mise à jour incrémentale", variant="primary")
                rebuild_btn = gr.Button("💥 Reconstruire l'index complet", variant="stop")

            confirm_cb = gr.Checkbox(
                label="Je confirme vouloir supprimer et reconstruire tout l'index",
                value=False,
            )

            index_log = gr.Textbox(
                label="Progression",
                lines=15,
                interactive=False,
                value="",
            )

            refresh_btn.click(fn=get_stats, outputs=[stats_box])
            update_btn.click(fn=run_update, inputs=[index_log], outputs=[index_log])
            rebuild_btn.click(
                fn=run_full_rebuild,
                inputs=[index_log, confirm_cb],
                outputs=[index_log],
            )

    return app


if __name__ == "__main__":
    cfg = load_config()
    app = build_app(cfg)
    app.launch(
        server_name="0.0.0.0",
        server_port=cfg["ui"]["port"],
        share=cfg["ui"]["share"],
    )
