import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zotero_rag.zotero_client import ZoteroItem

CONFIG = {
    "zotero": {
        "mode": "web",
        "web_api_key": "fake",
        "library_id": "123",
        "library_type": "user",
        "collection_key": "",
        "local_files_dir": "",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "llm_model": "llama3.1:8b",
        "embed_model": "nomic-embed-text",
        "timeout": 30,
    },
    "rag": {
        "chunk_size": 256,
        "chunk_overlap": 32,
        "similarity_top_k": 3,
        "persist_dir": "",
        "pdf_cache_dir": "",
        "highlighted_dir": "",
    },
}

FAKE_ITEM = ZoteroItem(
    item_key="TEST01",
    title="Robot Motion Planning Survey",
    authors=["Smith John", "Lee Ann"],
    year=2022,
    doi="10.0000/test",
    tags=["robotics"],
    pdf_attachment_key="ATT01",
    date_modified="2022-01-01",
)


@pytest.fixture
def indexer(tmp_path):
    cfg = json.loads(json.dumps(CONFIG))
    cfg["rag"]["persist_dir"] = str(tmp_path / "chroma")
    cfg["rag"]["pdf_cache_dir"] = str(tmp_path / "pdfs")
    cfg["rag"]["highlighted_dir"] = str(tmp_path / "highlighted")
    cfg["zotero"]["local_files_dir"] = str(tmp_path / "files")
    (tmp_path / "files").mkdir()

    from llama_index.core.llms import MockLLM
    from llama_index.core import MockEmbedding

    with (
        patch("zotero_rag.indexer.ZoteroClient"),
        patch("zotero_rag.indexer.PDFParser"),
        patch("zotero_rag.indexer.Ollama", return_value=MockLLM()),
        patch("zotero_rag.indexer.OllamaEmbedding", return_value=MockEmbedding(embed_dim=8)),
        patch("zotero_rag.indexer.VectorStoreIndex"),
        patch("zotero_rag.indexer.chromadb.PersistentClient") as mock_chroma,
    ):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        from zotero_rag.indexer import RAGIndexer
        idx = RAGIndexer(cfg)
        idx.zotero_client = MagicMock()
        idx.zotero_client.get_library_version.return_value = 1
        idx.pdf_parser = MagicMock()
        yield idx, tmp_path, mock_collection


def test_build_index_skips_up_to_date(indexer):
    idx, tmp_path, _ = indexer

    # Pre-populate state as if item is already indexed
    state = {
        "library_version": 1,
        "indexed_items": {"TEST01": "2022-01-01"},
        "last_update": "2022-01-01T00:00:00",
    }
    idx.state_file.parent.mkdir(parents=True, exist_ok=True)
    idx.state_file.write_text(json.dumps(state))

    idx.build_index(items=[FAKE_ITEM])

    idx.pdf_parser.parse.assert_not_called()


def test_build_index_processes_new_item(indexer):
    idx, tmp_path, _coll = indexer

    mock_doc = MagicMock()
    idx.pdf_parser.parse.return_value = [mock_doc]

    fake_pdf = tmp_path / "pdfs" / "TEST01.pdf"
    fake_pdf.parent.mkdir(parents=True, exist_ok=True)
    fake_pdf.write_bytes(b"%PDF fake")
    idx.zotero_client.download_pdf.return_value = fake_pdf

    with patch("zotero_rag.indexer.VectorStoreIndex") as mock_vi:
        idx.build_index(items=[FAKE_ITEM])

    idx.pdf_parser.parse.assert_called_once()
    state = json.loads(idx.state_file.read_text())
    assert "TEST01" in state["indexed_items"]


def test_get_index_stats(indexer):
    idx, _, mock_coll = indexer
    state = {
        "library_version": 42,
        "indexed_items": {"A": "x", "B": "y"},
        "last_update": "2023-01-01T00:00:00",
    }
    idx.state_file.parent.mkdir(parents=True, exist_ok=True)
    idx.state_file.write_text(json.dumps(state))

    s = idx.get_index_stats()
    assert s["nb_documents"] == 2
    assert s["library_version"] == 42
