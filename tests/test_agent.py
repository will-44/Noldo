import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from zotero_rag.citations import SemanticScholarError
from zotero_rag.retriever import SourceChunk

CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "llm_model": "gpt-oss:20b",
        "embed_model": "nomic-embed-text",
        "timeout": 300,
    },
    "agent": {"max_steps": 3},
}


def _chunk(content="", thinking="", tool_calls=None, done=False):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, thinking=thinking, tool_calls=tool_calls),
        done=done,
    )


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _source(item_key="ITEM01", score=0.9, page=3):
    return SourceChunk(
        text="Passage pertinent.", score=score, item_key=item_key, title="Titre",
        authors="Smith", year=2022, doi="10.0/x", page_number=page, section="Methods",
        pdf_path="/tmp/x.pdf",
    )


@pytest.fixture
def agent(tmp_path):
    with patch("zotero_rag.agent.ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_retriever = MagicMock()
        mock_retriever.debug_log_path = tmp_path / "debug" / "query_log.jsonl"
        mock_retriever.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Défaut : aucun article résolu (les tests qui vérifient le contexte article le
        # surchargent). Évite qu'un MagicMock truthy soit formaté dans le prompt système.
        mock_retriever.get_document.return_value = None

        mock_s2 = MagicMock()

        from zotero_rag.agent import RAGAgent
        a = RAGAgent(CONFIG, mock_retriever, s2_client=mock_s2)
        yield a, mock_client, mock_retriever, mock_s2


def _collect(gen):
    return list(gen)


def test_direct_answer_without_tool_call_streams_tokens(agent):
    """Cas simple : le LLM répond directement, sans outil. Vérifie que content est bien
    streamé token par token (events "token") et que "done" porte la réponse complète."""
    a, mock_client, *_ = agent
    mock_client.chat.return_value = iter([
        _chunk(thinking="Je"), _chunk(thinking=" réfléchis"),
        _chunk(content="Un"), _chunk(content=" PID"), _chunk(content=" controller."),
        _chunk(done=True),
    ])

    events = _collect(a.run_stream("C'est quoi un PID ?"))

    token_events = [e for e in events if e["type"] == "token"]
    assert "".join(e["text"] for e in token_events) == "Un PID controller."
    thinking_events = [e for e in events if e["type"] == "thinking"]
    assert "".join(e["text"] for e in thinking_events) == "Je réfléchis"

    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Un PID controller."
    assert done["sources"] == []
    mock_client.chat.assert_called_once()


def test_tool_call_then_final_answer(agent):
    """Cas nominal agentique : 1er tour → tool_calls (search_corpus), 2e tour → réponse finale.
    Vérifie l'événement "step", l'exécution réelle de la recherche via le retriever, et que
    le résultat est bien réinjecté comme message role=tool avant le 2e appel."""
    a, mock_client, mock_retriever, _ = agent
    src = _source()
    mock_retriever._retrieve_and_rerank.return_value = ["node"]
    mock_retriever._nodes_to_sources.return_value = [src]

    tc = _tool_call("search_corpus", {"query": "MPC model predictive control"})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="Réponse finale."), _chunk(done=True)]),
    ]

    events = _collect(a.run_stream("Comment fonctionne un MPC ?"))

    step_events = [e for e in events if e["type"] == "step"]
    assert len(step_events) == 1
    assert step_events[0]["tool"] == "search_corpus"
    assert step_events[0]["args"]["query"] == "MPC model predictive control"

    sources_events = [e for e in events if e["type"] == "sources"]
    assert len(sources_events) == 1
    assert sources_events[0]["kind"] == "corpus"
    assert sources_events[0]["items"][0]["item_key"] == "ITEM01"

    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Réponse finale."
    assert len(done["sources"]) == 1

    assert mock_client.chat.call_count == 2
    second_call_messages = mock_client.chat.call_args_list[1].kwargs["messages"]
    tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "ITEM01" in tool_msgs[0]["content"]

    # La question de recherche envoyée au retriever est celle formulée par le LLM (MPC),
    # pas la question brute de l'utilisateur ni un historique concaténé — c'est tout l'intérêt
    # de l'agent par rapport à l'ancienne composite_question. section_scope="content" par
    # défaut quand le LLM ne le précise pas dans ses arguments.
    mock_retriever._retrieve_and_rerank.assert_called_once_with(
        "MPC model predictive control", None, section_scope="content", trace={}
    )


def test_search_corpus_passes_llm_chosen_section_scope(agent):
    """Quand le LLM demande explicitement section_scope="references" (ex: pour trouver des
    citations), il doit être transmis tel quel à _retrieve_and_rerank."""
    a, mock_client, mock_retriever, _ = agent
    mock_retriever._retrieve_and_rerank.return_value = []
    mock_retriever._nodes_to_sources.return_value = []

    tc = _tool_call("search_corpus", {"query": "Makhal", "section_scope": "references"})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="ok"), _chunk(done=True)]),
    ]

    _collect(a.run_stream("question"))

    _, kwargs = mock_retriever._retrieve_and_rerank.call_args
    assert kwargs["section_scope"] == "references"


def test_search_corpus_rejects_invalid_section_scope(agent):
    """Une valeur de section_scope hors SECTION_SCOPES (le LLM peut halluciner un argument)
    ne doit jamais crasher l'agent : repli silencieux sur le défaut "content"."""
    a, mock_client, mock_retriever, _ = agent
    mock_retriever._retrieve_and_rerank.return_value = []
    mock_retriever._nodes_to_sources.return_value = []

    tc = _tool_call("search_corpus", {"query": "q", "section_scope": "methodology"})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="ok"), _chunk(done=True)]),
    ]

    _collect(a.run_stream("question"))

    _, kwargs = mock_retriever._retrieve_and_rerank.call_args
    assert kwargs["section_scope"] == "content"


def test_scan_corpus_tool_executes_and_emits_sources(agent):
    """Cas nominal : l'agent appelle scan_corpus, le résultat groupé par article devient des
    SourceChunk (même forme que search_corpus, réutilise l'UI existante sans modification)."""
    a, mock_client, mock_retriever, _ = agent
    mock_retriever.scan_corpus.return_value = {
        "total_articles": 2,
        "articles": [
            {
                "item_key": "ART1", "title": "Paper A", "authors": "Smith", "year": 2022,
                "doi": None, "n_hits": 3, "page": 5, "snippet": "250x faster than Reuleaux",
                "in_references_only": False,
            },
            {
                "item_key": "ART2", "title": "Paper B", "authors": "Jones", "year": 2021,
                "doi": "10.0/x", "n_hits": 1, "page": 8, "snippet": "Makhal, A. Reuleaux...",
                "in_references_only": True,
            },
        ],
    }

    tc = _tool_call("scan_corpus", {"keyword": "Makhal", "sections": ["references"]})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="26 articles trouvés."), _chunk(done=True)]),
    ]

    events = _collect(a.run_stream("Qui cite Reuleaux dans ma bibliothèque ?"))

    mock_retriever.scan_corpus.assert_called_once_with("Makhal", sections=["references"])

    step_events = [e for e in events if e["type"] == "step"]
    assert step_events[0]["tool"] == "scan_corpus"

    sources_events = [e for e in events if e["type"] == "sources"]
    assert len(sources_events) == 1
    assert sources_events[0]["kind"] == "corpus"
    items = sources_events[0]["items"]
    assert {i["item_key"] for i in items} == {"ART1", "ART2"}

    done = events[-1]
    assert len(done["sources"]) == 2

    # Le texte réinjecté au LLM doit porter le compte total et les deux articles.
    second_call_messages = mock_client.chat.call_args_list[1].kwargs["messages"]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "2 article" in tool_msg["content"]
    assert "ART1" in tool_msg["content"] and "ART2" in tool_msg["content"]


def test_scan_corpus_tool_no_results(agent):
    a, mock_client, mock_retriever, _ = agent
    mock_retriever.scan_corpus.return_value = {"total_articles": 0, "articles": []}

    tc = _tool_call("scan_corpus", {"keyword": "UnknownAuthor"})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="Aucun résultat."), _chunk(done=True)]),
    ]

    events = _collect(a.run_stream("question"))

    assert not [e for e in events if e["type"] == "sources"]
    second_call_messages = mock_client.chat.call_args_list[1].kwargs["messages"]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "Aucun article" in tool_msg["content"]


def test_scan_corpus_tool_empty_keyword_short_circuits(agent):
    """Un mot-clé vide ne doit jamais atteindre RAGRetriever.scan_corpus (garde-fou côté
    agent, cohérent avec search_corpus sur une requête vide)."""
    a, mock_client, mock_retriever, _ = agent

    tc = _tool_call("scan_corpus", {"keyword": ""})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="ok"), _chunk(done=True)]),
    ]

    _collect(a.run_stream("question"))
    mock_retriever.scan_corpus.assert_not_called()


def test_scan_corpus_filters_invalid_sections_values(agent):
    """Des valeurs de sections hallucinées par le LLM (hors SECTION_SCOPES) sont filtrées
    plutôt que transmises telles quelles."""
    a, mock_client, mock_retriever, _ = agent
    mock_retriever.scan_corpus.return_value = {"total_articles": 0, "articles": []}

    tc = _tool_call("scan_corpus", {"keyword": "q", "sections": ["references", "bogus"]})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="ok"), _chunk(done=True)]),
    ]

    _collect(a.run_stream("question"))
    mock_retriever.scan_corpus.assert_called_once_with("q", sections=["references"])


def test_ui_item_key_overrides_llm_choice(agent):
    """La case 'Limiter à cet article' (item_key transmis par l'UI) doit primer sur tout
    item_key que le LLM tenterait de choisir lui-même — contrat déjà établi côté UI."""
    a, mock_client, mock_retriever, _ = agent
    mock_retriever._retrieve_and_rerank.return_value = []
    mock_retriever._nodes_to_sources.return_value = []

    tc = _tool_call("search_corpus", {"query": "q", "item_key": "LLM_CHOSEN"})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="ok"), _chunk(done=True)]),
    ]

    _collect(a.run_stream("question", item_key="UI_LOCKED"))

    args, kwargs = mock_retriever._retrieve_and_rerank.call_args
    assert args[1] == "UI_LOCKED"


def test_open_article_context_injected_when_item_key_set(agent):
    """Bug réel : « résume cet article » avec un article ouvert (scope verrouillé) → l'agent
    demandait de quel article il s'agit, car item_key ne servait que de filtre en aval et
    n'était jamais dit au LLM. Le prompt système doit désormais nommer l'article ouvert."""
    a, mock_client, mock_retriever, _ = agent
    mock_retriever.get_document.return_value = {
        "item_key": "GXTCF5JL", "title": "3D U-Net", "authors": "Çiçek", "year": 2016, "doi": None,
    }
    mock_client.chat.return_value = iter([_chunk(content="Résumé..."), _chunk(done=True)])

    _collect(a.run_stream("résume cet article", item_key="GXTCF5JL"))

    system_msg = mock_client.chat.call_args.kwargs["messages"][0]
    assert system_msg["role"] == "system"
    assert "3D U-Net" in system_msg["content"]
    assert "cet article" in system_msg["content"].lower()
    mock_retriever.get_document.assert_called_once_with("GXTCF5JL")


def test_no_article_context_when_no_item_key(agent):
    """Question globale (aucun article ouvert) : le prompt système reste inchangé, pas de
    lookup d'article inutile."""
    a, mock_client, mock_retriever, _ = agent
    mock_client.chat.return_value = iter([_chunk(content="ok"), _chunk(done=True)])

    _collect(a.run_stream("question globale"))

    system_msg = mock_client.chat.call_args.kwargs["messages"][0]
    assert "CONTEXTE : l'utilisateur consulte" not in system_msg["content"]
    mock_retriever.get_document.assert_not_called()


def test_open_article_context_falls_back_without_metadata(agent):
    """Si l'article n'est pas retrouvé (get_document renvoie None), le contexte doit quand
    même signaler qu'un article est ouvert (via son identifiant) plutôt que de planter."""
    a, mock_client, mock_retriever, _ = agent
    mock_retriever.get_document.return_value = None
    mock_client.chat.return_value = iter([_chunk(content="ok"), _chunk(done=True)])

    _collect(a.run_stream("résume cet article", item_key="UNKNOWN99"))

    system_msg = mock_client.chat.call_args.kwargs["messages"][0]
    assert "UNKNOWN99" in system_msg["content"]
    assert "cet article" in system_msg["content"].lower()


def test_get_external_citations_tool(agent):
    a, mock_client, _, mock_s2 = agent
    mock_s2.resolve_paper.return_value = {"paperId": "P1", "title": "Reuleaux", "year": 2018}
    mock_s2.get_citations.return_value = [
        {
            "contexts": ["This extends [12]."],
            "intents": ["methodology"],
            "citingPaper": {"paperId": "C1", "title": "Survey", "year": 2022, "authors": [{"name": "Ang"}]},
        }
    ]

    tc = _tool_call("get_external_citations", {"title": "Reuleaux"})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="Cité par Survey (2022)."), _chunk(done=True)]),
    ]

    events = _collect(a.run_stream("Qui cite Reuleaux ?"))

    sources_events = [e for e in events if e["type"] == "sources" and e["kind"] == "external"]
    assert len(sources_events) == 1
    assert sources_events[0]["items"][0]["title"] == "Survey"
    assert "This extends" in sources_events[0]["items"][0]["context"]

    done = events[-1]
    assert len(done["external_sources"]) == 1


def test_external_citations_handles_semantic_scholar_error_gracefully(agent):
    """Une erreur S2 (429, réseau...) ne doit jamais crasher l'agent : elle devient un
    résultat d'outil que le LLM peut lire et sur lequel il peut réagir."""
    a, mock_client, _, mock_s2 = agent
    mock_s2.resolve_paper.side_effect = SemanticScholarError("429, pool anonyme")

    tc = _tool_call("get_external_citations", {"title": "Reuleaux"})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="Je n'ai pas pu vérifier les citations externes."), _chunk(done=True)]),
    ]

    events = _collect(a.run_stream("Qui cite Reuleaux ?"))

    assert events[-1]["type"] == "done"
    second_call_messages = mock_client.chat.call_args_list[1].kwargs["messages"]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "429" in tool_msg["content"]


def test_max_steps_triggers_forced_synthesis_without_tools(agent):
    """Si le modèle continue d'appeler des outils au-delà de max_steps, l'agent doit forcer
    une synthèse (dernier appel SANS tools) plutôt que de boucler indéfiniment."""
    a, mock_client, mock_retriever, _ = agent
    mock_retriever._retrieve_and_rerank.return_value = []
    mock_retriever._nodes_to_sources.return_value = []

    tc = _tool_call("search_corpus", {"query": "q"})
    # max_steps=3 (CONFIG) : 3 tours avec tool_calls, puis repli forcé.
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(tool_calls=[tc], done=True)]),
        SimpleNamespace(message=SimpleNamespace(content="Synthèse forcée.", tool_calls=None)),
    ]

    events = _collect(a.run_stream("question sans fin"))

    assert mock_client.chat.call_count == 4
    last_call_kwargs = mock_client.chat.call_args_list[-1].kwargs
    assert "tools" not in last_call_kwargs  # le repli ne doit PAS offrir d'outils au modèle

    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Synthèse forcée."


def test_ollama_connection_error_yields_error_event_not_exception(agent):
    a, mock_client, *_ = agent
    mock_client.chat.side_effect = ConnectionError("Ollama unreachable")

    events = _collect(a.run_stream("question"))

    assert events == [{"type": "error", "message": "Ollama unreachable"}]


def test_debug_log_records_steps_and_final_answer(agent):
    a, mock_client, mock_retriever, _ = agent
    mock_retriever._retrieve_and_rerank.return_value = []
    mock_retriever._nodes_to_sources.return_value = []

    tc = _tool_call("search_corpus", {"query": "MPC"})
    mock_client.chat.side_effect = [
        iter([_chunk(tool_calls=[tc], done=True)]),
        iter([_chunk(content="Réponse."), _chunk(done=True)]),
    ]

    _collect(a.run_stream("Et le MPC ?", ui_scope_checked=False, ui_selected_item_key="X"))

    lines = a.debug_log_path.read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["kind"] == "agent_run"
    assert entry["ui_scope_checked"] is False
    assert entry["ui_selected_item_key"] == "X"
    assert entry["final_answer"] == "Réponse."
    assert len(entry["steps"]) == 1
    assert entry["steps"][0]["tool"] == "search_corpus"


def test_generate_title_returns_stripped_llm_title(agent):
    a, mock_client, *_ = agent
    mock_client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content='"MPC en robotique mobile"')
    )

    title = a.generate_title("Comment fonctionne un MPC ?", "Un MPC est...")

    assert title == "MPC en robotique mobile"
    _, kwargs = mock_client.chat.call_args
    assert kwargs["stream"] is False


def test_generate_title_falls_back_to_truncated_question_on_empty_content(agent):
    a, mock_client, *_ = agent
    mock_client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=""))

    title = a.generate_title("Comment fonctionne un MPC en robotique mobile ?", "Réponse.")

    assert title == "Comment fonctionne un MPC en robotique mobile ?"[:50]


def test_generate_title_falls_back_on_exception(agent):
    a, mock_client, *_ = agent
    mock_client.chat.side_effect = ConnectionError("Ollama unreachable")

    title = a.generate_title("Une question précise", "Réponse.")

    assert title == "Une question précise"


def test_generate_title_falls_back_to_default_for_empty_question(agent):
    a, mock_client, *_ = agent
    mock_client.chat.side_effect = ConnectionError("down")

    title = a.generate_title("", "Réponse.")
    assert title == "Nouvelle conversation"


def test_generate_title_truncates_overly_long_titles(agent):
    a, mock_client, *_ = agent
    long_title = "Un titre beaucoup trop long que le modèle a généré malgré la consigne de rester court et concis"
    mock_client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=long_title))

    title = a.generate_title("question", "answer")

    assert len(title) <= 60
    assert title.endswith("…")


def test_history_becomes_user_assistant_turns(agent):
    """L'historique doit devenir de vrais tours user/assistant dans les messages envoyés au
    LLM (pas une concaténation en une seule chaîne comme l'ancienne composite_question) :
    c'est ce qui permet à l'agent de formuler une requête de recherche non diluée."""
    a, mock_client, *_ = agent
    mock_client.chat.return_value = iter([_chunk(content="ok"), _chunk(done=True)])

    history = [{"question": "Parle-moi de Reuleaux", "answer": "C'est un article sur..."}]
    _collect(a.run_stream("Et pour le MPC ?", history=history))

    sent_messages = mock_client.chat.call_args.kwargs["messages"]
    assert {"role": "user", "content": "Parle-moi de Reuleaux"} in sent_messages
    assert {"role": "assistant", "content": "C'est un article sur..."} in sent_messages
    assert sent_messages[-1] == {"role": "user", "content": "Et pour le MPC ?"}
