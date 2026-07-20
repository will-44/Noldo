import json
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

CONFIG = {
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
        "persist_dir": "/tmp/chroma_test",
        "pdf_cache_dir": "/tmp/pdfs_test",
        "highlighted_dir": "/tmp/highlighted_test",
    },
}


@pytest.fixture
def retriever(tmp_path):
    cfg = dict(CONFIG)
    cfg["rag"] = dict(CONFIG["rag"])
    cfg["rag"]["persist_dir"] = str(tmp_path / "chroma")

    from llama_index.core.llms import MockLLM
    from llama_index.core import MockEmbedding

    with (
        patch("zotero_rag.retriever.Ollama", return_value=MockLLM()),
        patch("zotero_rag.retriever.OllamaEmbedding", return_value=MockEmbedding(embed_dim=8)),
        patch("zotero_rag.retriever.chromadb.PersistentClient") as mock_chroma,
        patch("zotero_rag.retriever.VectorStoreIndex") as mock_vi,
        patch("zotero_rag.retriever.BM25Retriever") as mock_bm25_cls,
        patch("zotero_rag.retriever.QueryFusionRetriever") as mock_fusion_cls,
        patch("zotero_rag.retriever.SentenceTransformerRerank") as mock_reranker_cls,
    ):
        mock_collection = MagicMock()
        # _load_all_nodes() attend un dict réel avec des listes (pas un sous-Mock) :
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_index = MagicMock()
        mock_vi.from_vector_store.return_value = mock_index

        mock_bm25_cls.from_defaults.return_value = MagicMock(name="bm25_retriever")
        # Pass-through par défaut : le reranker (cross-encoder) est mocké, mais les VRAIS
        # postprocessors (section_filter, diversity_cap, doc_trim) doivent quand même opérer
        # sur les NodeWithScore réels qui traversent la chaîne — sans ça, la plupart des tests
        # de _retrieve_and_rerank ne testeraient rien.
        mock_reranker_cls.return_value.postprocess_nodes.side_effect = (
            lambda nodes, query_bundle=None, **kw: nodes
        )

        from zotero_rag.retriever import RAGRetriever
        r = RAGRetriever(cfg)
        yield r, mock_index, mock_bm25_cls, mock_fusion_cls, mock_reranker_cls


def _node(item_key, score, meta=None):
    meta = {"item_key": item_key, **(meta or {})}
    return NodeWithScore(node=TextNode(text=f"texte-{item_key}", metadata=meta), score=score)


def _full_node(item_key, text, score, page=1, section="Methods", **meta_overrides):
    meta = {
        "item_key": item_key,
        "title": f"Title {item_key}",
        "authors": "Author",
        "year": 2022,
        "doi": None,
        "page_number": page,
        "section": section,
        "pdf_path": f"/tmp/{item_key}.pdf",
    }
    meta.update(meta_overrides)
    return NodeWithScore(node=TextNode(text=text, metadata=meta), score=score)


# ── query() : bout-en-bout (retrieve mocké, synthèse via MockLLM en mode echo) ──────────────

def test_query_returns_rag_response(retriever):
    r, mock_index, *_ = retriever
    r.fused_retriever.retrieve.return_value = [
        _full_node("ITEM01", "Passage pertinent sur la planification.", 0.92, page=3),
        _full_node("ITEM02", "Autre extrait similaire.", 0.85, page=7),
    ]

    response = r.query("Comment fonctionne la planification de trajectoire ?")

    assert len(response.sources) == 2
    assert response.sources[0].score >= response.sources[1].score
    assert response.sources[0].item_key == "ITEM01"
    assert response.sources[0].page_number == 3
    # MockLLM.complete(prompt) renvoie le prompt tel quel (echo) : l'important est que le
    # contexte récupéré ait bien été injecté dans le prompt de synthèse.
    assert "Passage pertinent sur la planification." in response.answer


def test_query_deduplicates_same_page(retriever):
    r, *_ = retriever
    r.fused_retriever.retrieve.return_value = [
        _full_node("ITEM01", "chunk A", 0.9, page=5),
        _full_node("ITEM01", "chunk B", 0.8, page=5),  # même item+page → doublon
    ]

    response = r.query("question")
    assert len(response.sources) == 1


def test_query_handles_engine_error(retriever):
    r, *_ = retriever
    r.fused_retriever.retrieve.side_effect = Exception("Ollama unreachable")

    response = r.query("test")
    assert "Erreur" in response.answer
    assert response.sources == []


def test_query_passes_history_into_composite_question(retriever):
    r, *_ = retriever
    r.fused_retriever.retrieve.return_value = []

    r.query("Et pour les autres ?", history=[{"question": "Q1", "answer": "R1"}])

    called_bundle = r.fused_retriever.retrieve.call_args[0][0]
    assert "Q1" in called_bundle.query_str
    assert "Et pour les autres ?" in called_bundle.query_str


def test_query_writes_debug_log_entry(retriever):
    """Le journal de debug doit refléter fidèlement les paramètres reçus — c'est tout
    l'intérêt : pouvoir comparer item_key_received (ce qui a vraiment filtré) à
    ui_scope_checked/ui_selected_item_key (ce que l'interface affichait), et voir la trace en
    3 étapes de la récupération (bruts → après filtre sections → après reranking)."""
    r, mock_index, *_ = retriever
    # item_key fourni → _retrieve_and_rerank passe par self.index.as_retriever(...), pas
    # self.fused_retriever (réservé au mode global).
    mock_index.as_retriever.return_value.retrieve.return_value = []

    r.query(
        "Une question ?",
        item_key="TG2BGSCK",
        ui_scope_checked=True,
        ui_selected_item_key="TG2BGSCK",
    )

    assert r.debug_log_path.exists()
    lines = r.debug_log_path.read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[-1])

    assert entry["item_key_received"] == "TG2BGSCK"
    assert entry["ui_scope_checked"] is True
    assert entry["ui_selected_item_key"] == "TG2BGSCK"
    assert entry["scope_resolved"] == "doc"
    assert entry["answer"] is not None
    assert entry["error"] is None
    assert "candidates_raw" in entry
    assert "candidates_before_rerank" in entry
    assert "candidates_after_rerank" in entry


def test_query_debug_log_survives_engine_error(retriever):
    """Une erreur du moteur RAG doit quand même produire une ligne de log exploitable
    (avec le message d'erreur), et ne jamais empêcher la réponse d'erreur d'être renvoyée."""
    r, *_ = retriever
    r.fused_retriever.retrieve.side_effect = Exception("Ollama unreachable")

    r.query("test", ui_scope_checked=False, ui_selected_item_key=None)

    lines = r.debug_log_path.read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["error"] == "Ollama unreachable"
    assert entry["answer"] is None


# ── _build_composite_question (encore utilisé par query(), pas par l'agent) ─────────────────

def test_build_composite_question_without_history_is_unchanged():
    from zotero_rag.retriever import RAGRetriever

    result = RAGRetriever._build_composite_question("Quelle est la contribution ?", None)
    assert result == "Quelle est la contribution ?"

    result_empty = RAGRetriever._build_composite_question("Une question", [])
    assert result_empty == "Une question"


def test_build_composite_question_includes_recent_history():
    from zotero_rag.retriever import RAGRetriever

    history = [
        {"question": "Q1", "answer": "R1"},
        {"question": "Q2", "answer": "R2"},
    ]
    result = RAGRetriever._build_composite_question("Et pour les autres ?", history)

    assert "Q1" in result and "R1" in result
    assert "Q2" in result and "R2" in result
    assert "Et pour les autres ?" in result
    assert result.index("R2") < result.index("Et pour les autres ?")


def test_build_composite_question_caps_history_length():
    from zotero_rag.retriever import RAGRetriever

    history = [{"question": f"Q{i}", "answer": f"R{i}"} for i in range(10)]
    result = RAGRetriever._build_composite_question("Question finale", history)

    # Seuls les 3 derniers tours (MAX_HISTORY_TURNS) doivent apparaître.
    assert "Q9" in result and "Q8" in result and "Q7" in result
    assert "Q0" not in result and "Q5" not in result


# ── _retrieve_and_rerank : pipeline partagé query()/agent ───────────────────────────────────

def test_fused_retriever_built_with_num_queries_one(retriever):
    """Garde-fou latence : QueryFusionRetriever reformule la requête via un appel LLM si
    num_queries != 1 (défaut de la lib : 4), ce qui multiplierait le temps de réponse sur ce
    Jetson. use_async=False évite "Detected nested async" sous uvicorn/uvloop."""
    r, mock_index, mock_bm25_cls, mock_fusion_cls, mock_reranker_cls = retriever

    assert mock_bm25_cls.from_defaults.called
    assert mock_fusion_cls.called
    _, fusion_kwargs = mock_fusion_cls.call_args
    assert fusion_kwargs["num_queries"] == 1
    assert fusion_kwargs["mode"] == "reciprocal_rerank"
    assert len(fusion_kwargs["retrievers"]) == 2  # vecteur + BM25
    assert fusion_kwargs["use_async"] is False
    assert r.fused_retriever is mock_fusion_cls.return_value


def test_retrieve_and_rerank_global_mode_filters_reference_sections(retriever):
    """Mode global : la chaîne est filtre_sections → reranker → diversity_cap, dans cet ordre
    (le filtre retire le bruit bibliographique AVANT de faire tourner le cross-encoder dessus)."""
    r, *_ = retriever
    r.fused_retriever.retrieve.return_value = [
        _full_node("REF", "Bibliographie...", 0.99, section="References"),
        _full_node("ITEM01", "Contenu utile.", 0.5, section="Methods"),
    ]

    nodes = r._retrieve_and_rerank("question", item_key=None)

    assert [n.metadata["item_key"] for n in nodes] == ["ITEM01"]


def test_retrieve_and_rerank_global_mode_caps_diversity(retriever):
    r, *_ = retriever
    r.fused_retriever.retrieve.return_value = (
        [_full_node("SELF_CITED", f"t{i}", 0.9 - i * 0.01, page=i) for i in range(4)]
        + [_full_node(f"OTHER{i}", f"t{i}", 0.5 - i * 0.01, page=i) for i in range(4)]
    )

    nodes = r._retrieve_and_rerank("question", item_key=None)

    # r.diversity_cap.top_k == r.similarity_top_k == 3 (fixture) : sans plafond, les 3
    # premiers par score seraient tous SELF_CITED. Avec max_per_document=2, au moins un
    # autre article doit être repêché — la couverture complète du plafond (respect strict
    # de max_per_document, préservation de l'ordre) est testée isolément plus bas.
    keys = [n.metadata["item_key"] for n in nodes]
    assert keys.count("SELF_CITED") <= 2
    assert len(set(keys)) >= 2


def test_retrieve_and_rerank_records_trace_stages(retriever):
    r, *_ = retriever
    r.fused_retriever.retrieve.return_value = [
        _full_node("REF", "Bibliographie...", 0.99, section="References"),
        _full_node("ITEM01", "Contenu utile.", 0.5, section="Methods"),
    ]

    trace: dict = {}
    r._retrieve_and_rerank("question", item_key=None, trace=trace)

    assert len(trace["candidates_raw"]) == 2
    assert len(trace["candidates_before_rerank"]) == 1  # REF déjà filtré
    assert len(trace["candidates_after_rerank"]) == 1


def test_retrieve_and_rerank_item_key_mode_uses_filtered_vector_retriever(retriever):
    r, mock_index, mock_bm25_cls, mock_fusion_cls, mock_reranker_cls = retriever
    bm25_calls_before = mock_bm25_cls.from_defaults.call_count
    fusion_calls_before = mock_fusion_cls.call_count
    mock_index.as_retriever.return_value.retrieve.return_value = [
        _full_node("SOME_ITEM_KEY", "texte", 0.5, page=1),
    ]

    nodes = r._retrieve_and_rerank("question", item_key="SOME_ITEM_KEY")

    assert len(nodes) == 1
    _, kwargs = mock_index.as_retriever.call_args
    assert "filters" in kwargs
    assert kwargs["similarity_top_k"] == r.fetch_k
    # Le filtre par item_key ne doit pas déclencher de nouvelle construction BM25/fusion.
    assert mock_bm25_cls.from_defaults.call_count == bm25_calls_before
    assert mock_fusion_cls.call_count == fusion_calls_before


def test_retrieve_and_rerank_item_key_mode_truncates_no_diversity_cap(retriever):
    """Mode "cet article" : simple troncature à similarity_top_k (doc_trim), pas de plafond
    de diversité — un seul article dans le pool, la diversité inter-documents n'a pas de sens."""
    r, mock_index, *_ = retriever
    mock_index.as_retriever.return_value.retrieve.return_value = [
        _full_node("K", f"texte {i}", 1.0 - i * 0.1, page=i) for i in range(10)
    ]

    nodes = r._retrieve_and_rerank("question", item_key="K")
    assert len(nodes) == r.similarity_top_k


# ── _load_all_nodes : exhaustif, alimente BM25 pour tous les section_scope ──────────────────

def test_load_all_nodes_is_exhaustive_includes_references(retriever):
    """_load_all_nodes ne filtre plus les références : le filtrage par portée se fait en aval
    dans _retrieve_and_rerank (section_scope). Sinon section_scope="references" ne pourrait
    jamais remonter de résultats côté BM25 (bug réintroduit par la 1ère version du filtre)."""
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "ids": ["1", "2", "3"],
        "documents": ["Contenu méthode.", "Liste de références...", "Remerciements..."],
        "metadatas": [
            {"item_key": "A", "section": "Methods"},
            {"item_key": "B", "section": "References"},
            {"item_key": "C", "section": "Acknowledgments"},
        ],
    }

    nodes = r._load_all_nodes()
    assert [n.metadata["item_key"] for n in nodes] == ["A", "B", "C"]


# ── Postprocessors (mode global) ─────────────────────────────────────────────────────────────

def test_top_k_postprocessor_truncates():
    from zotero_rag.retriever import TopKPostprocessor

    nodes = [_node(f"K{i}", 1.0 - i * 0.1) for i in range(5)]
    result = TopKPostprocessor(top_k=2)._postprocess_nodes(nodes)
    assert len(result) == 2
    assert result == nodes[:2]


def test_diversity_cap_limits_per_document():
    """Reproduit le bug observé : le reranker faisait remonter 4/5 passages du même article
    (celui qui se cite lui-même) alors que le pool contenait 8 articles distincts avant
    reranking. Le plafond doit forcer la diversité tout en respectant l'ordre de pertinence."""
    from zotero_rag.retriever import DiversityCapPostprocessor

    nodes = (
        [_node("SELF_CITED", 0.08 - i * 0.01) for i in range(4)]
        + [_node(f"OTHER{i}", 0.02 - i * 0.001) for i in range(4)]
    )

    result = DiversityCapPostprocessor(max_per_document=2, top_k=5)._postprocess_nodes(nodes)

    assert len(result) == 5
    keys = [n.metadata["item_key"] for n in result]
    assert keys.count("SELF_CITED") <= 2
    assert len(set(keys)) >= 3  # diversité restaurée, plusieurs articles distincts présents


def test_diversity_cap_preserves_order_within_quota():
    from zotero_rag.retriever import DiversityCapPostprocessor

    nodes = [_node("A", 0.9), _node("A", 0.8), _node("A", 0.7), _node("B", 0.6)]
    result = DiversityCapPostprocessor(max_per_document=2, top_k=5)._postprocess_nodes(nodes)
    keys = [n.metadata["item_key"] for n in result]
    assert keys == ["A", "A", "B"]  # 3e "A" exclu par le plafond, "B" repêché


# ── _is_reference_section : robustesse aux entêtes "espacés" (PDF anciens/scannés) ──────────

def test_is_reference_section_handles_letter_spaced_headers():
    """Cas réel trouvé sur ce corpus (papier de 1955, PNWN5AD7) : l'entête de bibliographie
    est extraite comme 'B I B L I O G R A P H Y' (une lettre par token) — invisible à un
    simple "bibliograph" in section.lower() sans normalisation préalable."""
    from zotero_rag.retriever import _is_reference_section

    assert _is_reference_section("B I B L I O G R A P H Y") is True
    assert _is_reference_section("R E F E R E N C E S") is True


def test_is_reference_section_letter_spacing_heuristic_does_not_over_trigger():
    """L'heuristique de désespacement ne doit jamais fusionner un entête normal à mots courts
    (peu de tokens, ou tokens de plus de 2 caractères)."""
    from zotero_rag.retriever import _is_reference_section

    assert _is_reference_section("A. Results and Analysis") is False
    assert _is_reference_section("I. INTRODUCTION") is False
    assert _is_reference_section("3 Related Work") is False


# ── _in_section_scope : la frontière references/content, seule fiable (voir plan Lot 2) ─────

def test_in_section_scope_content_excludes_references():
    from zotero_rag.retriever import _in_section_scope

    assert _in_section_scope("Methods", "content") is True
    assert _in_section_scope("References", "content") is False
    assert _in_section_scope("Bibliography", "content") is False
    assert _in_section_scope("Acknowledgments", "content") is False
    assert _in_section_scope(None, "content") is True  # pas de section = pas une biblio


def test_in_section_scope_references_keeps_only_references():
    from zotero_rag.retriever import _in_section_scope

    assert _in_section_scope("References", "references") is True
    assert _in_section_scope("REFERENCES", "references") is True  # insensible à la casse
    assert _in_section_scope("Methods", "references") is False


def test_in_section_scope_all_keeps_everything():
    from zotero_rag.retriever import _in_section_scope

    assert _in_section_scope("References", "all") is True
    assert _in_section_scope("Methods", "all") is True
    assert _in_section_scope(None, "all") is True


def test_in_section_scope_substring_match():
    from zotero_rag.retriever import _in_section_scope

    # "3. Related Work" ne contient aucun mot-clé référence → reste "content".
    assert _in_section_scope("3. Related Work", "content") is True


# ── _retrieve_and_rerank : paramètre section_scope (nouveau, Lot 2) ─────────────────────────

def test_retrieve_and_rerank_section_scope_references_keeps_only_biblio(retriever):
    r, *_ = retriever
    r.fused_retriever.retrieve.return_value = [
        _full_node("REF", "Makhal A. Reuleaux...", 0.9, section="References"),
        _full_node("ITEM01", "Contenu utile.", 0.8, section="Methods"),
    ]

    nodes = r._retrieve_and_rerank("Makhal", item_key=None, section_scope="references")

    assert [n.metadata["item_key"] for n in nodes] == ["REF"]


def test_retrieve_and_rerank_section_scope_all_keeps_everything(retriever):
    r, *_ = retriever
    r.fused_retriever.retrieve.return_value = [
        _full_node("REF", "Bibliographie...", 0.9, section="References"),
        _full_node("ITEM01", "Contenu utile.", 0.8, section="Methods"),
    ]

    nodes = r._retrieve_and_rerank("question", item_key=None, section_scope="all")

    assert {n.metadata["item_key"] for n in nodes} == {"REF", "ITEM01"}


def test_retrieve_and_rerank_section_scope_defaults_to_content(retriever):
    """Comportement inchangé pour les appelants existants (query(), search_corpus sans
    section_scope explicite) : les références restent exclues par défaut."""
    r, *_ = retriever
    r.fused_retriever.retrieve.return_value = [
        _full_node("REF", "Bibliographie...", 0.9, section="References"),
        _full_node("ITEM01", "Contenu utile.", 0.8, section="Methods"),
    ]

    nodes = r._retrieve_and_rerank("question", item_key=None)

    assert [n.metadata["item_key"] for n in nodes] == ["ITEM01"]


# ── scan_corpus : balayage exhaustif groupé par article (nouveau, Lot 2) ────────────────────

def _chunk_meta(item_key, title="T", authors="A", year=2020, doi=None, section="Methods", page=1):
    return {
        "item_key": item_key, "title": title, "authors": authors, "year": year,
        "doi": doi, "section": section, "page_number": page,
    }


def test_scan_corpus_finds_all_matching_articles_not_just_top_k(retriever):
    """Le cas exact du bug diagnostiqué : un classement par pertinence plafonne à quelques
    résultats, scan_corpus doit tous les trouver (26 articles réels sur le vrai corpus,
    ici simulés en réduit)."""
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "documents": [f"... Makhal and Goins, Reuleaux ... [{i}]" for i in range(10)],
        "metadatas": [_chunk_meta(f"ART{i}", section="References", page=8) for i in range(10)],
    }

    result = r.scan_corpus("Makhal")

    assert result["total_articles"] == 10
    assert len(result["articles"]) == 10
    assert {a["item_key"] for a in result["articles"]} == {f"ART{i}" for i in range(10)}


def test_scan_corpus_groups_multiple_hits_per_article(retriever):
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "documents": [
            "Makhal et al. introduced Reuleaux for base placement.",
            "We compare against Makhal's Reuleaux baseline in Table 2.",
            "Unrelated content about something else entirely.",
        ],
        "metadatas": [
            _chunk_meta("SAME", section="Introduction", page=1),
            _chunk_meta("SAME", section="Results", page=5),
            _chunk_meta("OTHER", section="Methods", page=2),
        ],
    }

    result = r.scan_corpus("Makhal")

    assert result["total_articles"] == 1
    assert result["articles"][0]["item_key"] == "SAME"
    assert result["articles"][0]["n_hits"] == 2


def test_scan_corpus_prefers_body_snippet_over_references_only(retriever):
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "documents": [
            "Makhal, A. Reuleaux: Robot Base Placement. In IRC, 2018.",  # biblio
            "Compared to Reuleaux (Makhal et al.), our method is 250x faster.",  # discussion
        ],
        "metadatas": [
            _chunk_meta("ART", section="References", page=8),
            _chunk_meta("ART", section="A. Results", page=5),
        ],
    }

    result = r.scan_corpus("Makhal")

    article = result["articles"][0]
    assert article["in_references_only"] is False
    assert article["page"] == 5
    assert "250x faster" in article["snippet"]


def test_scan_corpus_reports_references_only_when_no_body_mention(retriever):
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "documents": ["Makhal, A. Reuleaux: Robot Base Placement. In IRC, 2018."],
        "metadatas": [_chunk_meta("ART", section="References", page=8)],
    }

    result = r.scan_corpus("Makhal")
    assert result["articles"][0]["in_references_only"] is True


def test_scan_corpus_sections_filter_restricts_to_references(retriever):
    """sections=["references"] : n'inclut PAS un article qui ne mentionne le mot-clé que dans
    son corps (pas en bibliographie) — utile pour ne garder QUE les citations avérées."""
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "documents": ["Discussion of Makhal's approach in the introduction."],
        "metadatas": [_chunk_meta("ART", section="Introduction", page=1)],
    }

    result = r.scan_corpus("Makhal", sections=["references"])
    assert result["total_articles"] == 0


def test_scan_corpus_sorts_by_n_hits_descending(retriever):
    """n_hits compte les CHUNKS contenant le mot-clé (pas les occurrences littérales dans un
    chunk) : un article présent dans plus de chunks est mieux classé."""
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "documents": ["Makhal mention.", "Makhal mention.", "Makhal mention."],
        "metadatas": [
            _chunk_meta("MANY", page=1), _chunk_meta("MANY", page=2), _chunk_meta("FEW", page=1),
        ],
    }

    result = r.scan_corpus("Makhal")
    assert [a["item_key"] for a in result["articles"]] == ["MANY", "FEW"]


def test_scan_corpus_truncates_to_max_articles_but_keeps_exact_total(retriever):
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "documents": [f"Makhal {i}" for i in range(5)],
        "metadatas": [_chunk_meta(f"ART{i}", page=1) for i in range(5)],
    }

    result = r.scan_corpus("Makhal", max_articles=2)
    assert result["total_articles"] == 5
    assert len(result["articles"]) == 2


def test_scan_corpus_empty_keyword_returns_empty(retriever):
    r, *_ = retriever
    assert r.scan_corpus("") == {"total_articles": 0, "articles": []}
    assert r.scan_corpus("   ") == {"total_articles": 0, "articles": []}


def test_scan_corpus_case_insensitive(retriever):
    r, *_ = retriever
    r.chroma_collection.get.return_value = {
        "documents": ["REULEAUX is discussed here."],
        "metadatas": [_chunk_meta("ART", page=1)],
    }
    result = r.scan_corpus("reuleaux")
    assert result["total_articles"] == 1


# ── _snippet_around : ne coupe pas un mot en deux ────────────────────────────────────────────

def test_snippet_around_does_not_cut_words():
    from zotero_rag.retriever import RAGRetriever

    text = "This is a long sentence about Makhal and his Reuleaux library for robotics."
    snippet = RAGRetriever._snippet_around(text, "makhal", radius=10)

    # Chaque mot du snippet (hors ellipses) doit être un mot ENTIER du texte d'origine —
    # jamais un fragment coupé en milieu de mot.
    original_words = set(text.split())
    snippet_words = snippet.replace("…", "").split()
    assert snippet_words  # non vide
    assert all(w in original_words for w in snippet_words)
    assert "Makhal" in snippet
