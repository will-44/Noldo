import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zotero_rag.zotero_client import ZoteroClient, ZoteroItem

CONFIG = {
    "zotero": {
        "mode": "web",
        "web_api_key": "fake_key",
        "library_id": "123",
        "library_type": "user",
        "collection_key": "",
        "local_files_dir": "/tmp/zotero_test_files",
    },
    "rag": {
        "pdf_cache_dir": "/tmp/zotero_test_cache",
        "chunk_size": 512,
        "chunk_overlap": 64,
    },
}

FAKE_ATTACHMENT = {
    "key": "ATT001",
    "data": {
        "itemType": "attachment",
        "contentType": "application/pdf",
        "linkMode": "imported_file",
        "parentItem": "ITEM001",
    },
}

FAKE_PARENT = {
    "key": "ITEM001",
    "data": {
        "itemType": "journalArticle",
        "title": "Test Article on Robotics",
        "creators": [
            {"creatorType": "author", "lastName": "Dupont", "firstName": "Jean"},
            {"creatorType": "author", "lastName": "Martin", "firstName": "Claire"},
        ],
        "date": "2023-05-12",
        "DOI": "10.1234/test",
        "tags": [{"tag": "robotics"}, {"tag": "planning"}],
        "dateModified": "2023-06-01T10:00:00Z",
    },
    "meta": {"parsedDate": "2023-06-01"},
}


@pytest.fixture
def client(tmp_path):
    cfg = dict(CONFIG)
    cfg["zotero"] = dict(CONFIG["zotero"])
    cfg["rag"] = dict(CONFIG["rag"])
    cfg["rag"]["pdf_cache_dir"] = str(tmp_path / "cache")
    cfg["zotero"]["local_files_dir"] = str(tmp_path / "files")
    (tmp_path / "files").mkdir()
    (tmp_path / "cache").mkdir()

    with patch("zotero_rag.zotero_client.zotero.Zotero") as mock_zot_cls:
        mock_zot = MagicMock()
        mock_zot_cls.return_value = mock_zot
        c = ZoteroClient(cfg)
        c.zot = mock_zot
        yield c, mock_zot, tmp_path


def test_get_items_with_pdfs(client):
    c, mock_zot, tmp_path = client
    mock_zot.everything.return_value = [FAKE_ATTACHMENT]
    mock_zot.items.return_value = [FAKE_PARENT]

    items = c.get_items_with_pdfs()

    assert len(items) == 1
    item = items[0]
    assert item.item_key == "ITEM001"
    assert item.title == "Test Article on Robotics"
    assert "Dupont Jean" in item.authors
    assert item.year == 2023
    assert item.doi == "10.1234/test"
    assert item.pdf_attachment_key == "ATT001"


def test_download_pdf_from_local_zip(client):
    c, mock_zot, tmp_path = client

    # Create a fake zip with a PDF
    zip_path = tmp_path / "files" / "ATT001.zip"
    fake_pdf_content = b"%PDF-1.4 fake content"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("article.pdf", fake_pdf_content)

    item = ZoteroItem(
        item_key="ITEM001",
        title="Test",
        authors=["Dupont Jean"],
        year=2023,
        doi=None,
        tags=[],
        pdf_attachment_key="ATT001",
    )

    pdf_path = c.download_pdf(item)

    assert pdf_path.exists()
    assert pdf_path.read_bytes() == fake_pdf_content


def test_download_pdf_cache_hit(client):
    c, mock_zot, tmp_path = client

    item = ZoteroItem(
        item_key="ITEM001",
        title="Test",
        authors=[],
        year=None,
        doi=None,
        tags=[],
        pdf_attachment_key="ATT001",
    )

    # Pre-populate cache
    cache_path = Path(c.pdf_cache_dir) / "ITEM001.pdf"
    cache_path.write_bytes(b"cached")

    pdf_path = c.download_pdf(item)
    assert pdf_path == cache_path
    mock_zot.dump.assert_not_called()


def test_build_item_filters_non_articles(client):
    c, _, _ = client
    note = {"key": "N1", "data": {"itemType": "note", "title": "note"}, "meta": {}}
    result = c._build_item(note, "ATT")
    assert result is None
