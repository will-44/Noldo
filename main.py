import click

from zotero_rag import RAGIndexer, RAGRetriever
from zotero_rag.utils import get_logger, load_config

logger = get_logger("main")


@click.group()
@click.option("--config", default="config.yaml", help="Path to config.yaml")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.pass_context
def index(ctx):
    """Build the complete index from scratch."""
    cfg = ctx.obj["config"]
    indexer = RAGIndexer(cfg)
    items = indexer.zotero_client.get_items_with_pdfs()
    click.echo(f"Found {len(items)} items with PDFs.")

    def progress(current, total, msg):
        click.echo(f"  [{current}/{total}] {msg}")

    indexer.build_index(items=items, progress_cb=progress)
    click.echo("✅ Index build complete.")


@cli.command()
@click.pass_context
def update(ctx):
    """Incremental index update (only modified items)."""
    cfg = ctx.obj["config"]
    indexer = RAGIndexer(cfg)

    def progress(current, total, msg):
        click.echo(f"  [{current}/{total}] {msg}")

    n = indexer.update_index(progress_cb=progress)
    click.echo(f"✅ Updated {n} item(s).")


@cli.command()
@click.argument("question")
@click.pass_context
def query(ctx, question):
    """Query the index from the command line."""
    cfg = ctx.obj["config"]
    retriever = RAGRetriever(cfg)
    response = retriever.query(question)

    click.echo("\n" + "=" * 60)
    click.echo("RÉPONSE")
    click.echo("=" * 60)
    click.echo(response.answer)

    if response.sources:
        click.echo("\n" + "-" * 60)
        click.echo("SOURCES")
        click.echo("-" * 60)
        for src in response.sources:
            click.echo(
                f"  [{src.score:.3f}] {src.authors} ({src.year}) — "
                f"{src.title[:60]} — p.{src.page_number}"
            )


@cli.command()
@click.pass_context
def stats(ctx):
    """Display index statistics."""
    cfg = ctx.obj["config"]
    indexer = RAGIndexer(cfg)
    s = indexer.get_index_stats()
    click.echo(f"Articles indexés  : {s['nb_documents']}")
    click.echo(f"Chunks vectoriels : {s['nb_chunks']}")
    click.echo(f"Dernière MAJ      : {s['last_update'] or 'Jamais'}")
    click.echo(f"Version Zotero    : {s['library_version']}")


@cli.command()
@click.pass_context
def serve(ctx):
    """Launch the Gradio UI."""
    from app import build_app
    cfg = ctx.obj["config"]
    app = build_app(cfg)
    app.launch(
        server_name="0.0.0.0",
        server_port=cfg["ui"]["port"],
        share=cfg["ui"]["share"],
    )


@cli.command()
@click.option("--port", default=7862, help="Port du service web ChatPDF")
@click.pass_context
def webserve(ctx, port):
    """Launch the ChatPDF-style FastAPI web interface."""
    import uvicorn

    from webapp.server import create_app
    cfg = ctx.obj["config"]
    app = create_app(cfg)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    cli()
