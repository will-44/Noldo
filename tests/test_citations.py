import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from zotero_rag.citations import SemanticScholarClient, SemanticScholarError


def _ok_response(payload: dict):
    cm = MagicMock()
    cm.__enter__.return_value = cm
    cm.read.return_value = json.dumps(payload).encode("utf-8")
    return cm


@pytest.fixture
def client():
    return SemanticScholarClient()


def test_resolve_paper_by_doi_success(client):
    paper = {"paperId": "P1", "title": "Reuleaux: Robot Base Placement", "year": 2018}
    with patch("zotero_rag.citations.urllib.request.urlopen", return_value=_ok_response(paper)) as m:
        result = client.resolve_paper(doi="10.1109/IROS.2018.1234")

    assert result == paper
    url = m.call_args[0][0].full_url
    assert "DOI%3A10.1109%2FIROS.2018.1234" in url or "DOI:10.1109" in url


def test_resolve_paper_falls_back_to_title_on_doi_failure(client):
    http_err = urllib.error.HTTPError("url", 404, "Not Found", None, None)
    title_payload = {"data": [{"paperId": "P2", "title": "Reuleaux: Robot Base Placement"}]}

    with patch(
        "zotero_rag.citations.urllib.request.urlopen",
        side_effect=[http_err, _ok_response(title_payload)],
    ):
        result = client.resolve_paper(doi="10.0/bad", title="Reuleaux: Robot Base Placement")

    assert result["paperId"] == "P2"


def test_resolve_paper_title_search_no_results_returns_none(client):
    with patch(
        "zotero_rag.citations.urllib.request.urlopen",
        return_value=_ok_response({"data": []}),
    ):
        result = client.resolve_paper(title="Un titre introuvable")
    assert result is None


def test_resolve_paper_without_doi_or_title_returns_none(client):
    result = client.resolve_paper()
    assert result is None


def test_get_citations_returns_contexts_and_intents(client):
    payload = {
        "data": [
            {
                "contexts": ["This extends the base placement method of [12]."],
                "intents": ["methodology"],
                "citingPaper": {"paperId": "C1", "title": "Mobile Manipulation Survey", "year": 2022},
            }
        ]
    }
    with patch("zotero_rag.citations.urllib.request.urlopen", return_value=_ok_response(payload)):
        citations = client.get_citations("P1")

    assert len(citations) == 1
    assert citations[0]["citingPaper"]["title"] == "Mobile Manipulation Survey"
    assert "extends the base placement" in citations[0]["contexts"][0]


def test_429_raises_clear_semantic_scholar_error(client):
    http_err = urllib.error.HTTPError("url", 429, "Too Many Requests", None, None)
    with patch("zotero_rag.citations.urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(SemanticScholarError, match="429|limité"):
            client.get_citations("P1")


def test_url_error_raises_semantic_scholar_error(client):
    with patch(
        "zotero_rag.citations.urllib.request.urlopen",
        side_effect=urllib.error.URLError("Name or service not known"),
    ):
        with pytest.raises(SemanticScholarError, match="injoignable"):
            client.get_citations("P1")


def test_api_key_sent_as_header_when_provided():
    client = SemanticScholarClient(api_key="secret-key")
    with patch(
        "zotero_rag.citations.urllib.request.urlopen", return_value=_ok_response({"data": []})
    ) as m:
        client.get_citations("P1")

    req = m.call_args[0][0]
    assert req.headers.get("X-api-key") == "secret-key"


def test_no_api_key_omits_header():
    client = SemanticScholarClient()
    with patch(
        "zotero_rag.citations.urllib.request.urlopen", return_value=_ok_response({"data": []})
    ) as m:
        client.get_citations("P1")

    req = m.call_args[0][0]
    assert "X-api-key" not in req.headers
