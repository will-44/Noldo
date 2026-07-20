import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from .pdf_parser import PDFParser
from .utils import get_logger
from .zotero_client import ZoteroClient, ZoteroItem

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class RAGIndexer:
    COLLECTION_NAME = "zotero_rag"

    def __init__(self, config: dict):
        self.config = config
        self.persist_dir = Path(config["rag"]["persist_dir"])
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.persist_dir / ".index_state.json"

        self._configure_llamaindex(config)

        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            self.COLLECTION_NAME
        )

        self.zotero_client = ZoteroClient(config)
        self.pdf_parser = PDFParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(
        self,
        items: list[ZoteroItem] | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        if items is None:
            items = self.zotero_client.get_items_with_pdfs()

        state = self._load_state()
        total = len(items)
        logger.info(f"Building index for {total} items…")

        for i, item in enumerate(items):
            _cb(progress_cb, i, total, f"[{i+1}/{total}] {item.title[:60]}")

            # Skip if already indexed and unchanged
            if self._is_up_to_date(item, state):
                logger.debug(f"Skipping (up-to-date): {item.item_key}")
                continue

            # Remove stale chunks if item existed before
            if item.item_key in state["indexed_items"]:
                self._delete_item_chunks(item.item_key)

            try:
                pdf_path = self.zotero_client.download_pdf(item)
                docs = self.pdf_parser.parse(pdf_path, item)
                if not docs:
                    logger.warning(f"No content extracted from {item.item_key}")
                    continue

                self._index_documents(docs)
                state["indexed_items"][item.item_key] = item.date_modified

            except Exception as e:
                logger.error(f"Failed to index {item.item_key} ({item.title[:40]}): {e}")

        state["last_update"] = datetime.now().isoformat()
        state["library_version"] = self.zotero_client.get_library_version()
        self._save_state(state)
        _cb(progress_cb, total, total, "Index build complete.")
        logger.info("Index build complete.")

    def update_index(self, progress_cb: ProgressCallback | None = None) -> int:
        state = self._load_state()
        since = state.get("library_version", 0)
        modified = self.zotero_client.get_new_or_modified_items(since)
        if not modified:
            logger.info("Nothing to update.")
            return 0
        logger.info(f"Updating {len(modified)} modified items…")
        self.build_index(items=modified, progress_cb=progress_cb)
        return len(modified)

    def get_index_stats(self) -> dict:
        state = self._load_state()
        return {
            "nb_documents": len(state.get("indexed_items", {})),
            "nb_chunks": self.chroma_collection.count(),
            "last_update": state.get("last_update", "Never"),
            "library_version": state.get("library_version", 0),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_llamaindex(config: dict) -> None:
        ollama_cfg = config["ollama"]
        rag_cfg = config["rag"]
        Settings.llm = Ollama(
            model=ollama_cfg["llm_model"],
            base_url=ollama_cfg["base_url"],
            request_timeout=ollama_cfg["timeout"],
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=ollama_cfg["embed_model"],
            base_url=ollama_cfg["base_url"],
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=rag_cfg["chunk_size"],
            chunk_overlap=rag_cfg["chunk_overlap"],
        )

    def _index_documents(self, docs) -> None:
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    def _delete_item_chunks(self, item_key: str) -> None:
        try:
            self.chroma_collection.delete(where={"item_key": {"$eq": item_key}})
        except Exception as e:
            logger.warning(f"Could not delete chunks for {item_key}: {e}")

    def _is_up_to_date(self, item: ZoteroItem, state: dict) -> bool:
        saved = state.get("indexed_items", {}).get(item.item_key)
        return saved is not None and saved == item.date_modified

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {"library_version": 0, "indexed_items": {}, "last_update": None}

    def _save_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, indent=2))


def _cb(fn: ProgressCallback | None, current: int, total: int, msg: str) -> None:
    if fn:
        fn(current, total, msg)
