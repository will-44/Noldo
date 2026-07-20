"""Agent ReAct léger : le LLM (gpt-oss:20b, tool-calling natif) décide lui-même d'appeler
search_corpus (corpus Zotero local) et/ou get_external_citations (Semantic Scholar), en
boucle bornée (MAX_STEPS). Remplace la condensation par concaténation de RAGRetriever.query()
en mode conversationnel : c'est l'agent qui formule sa propre requête de recherche à partir du
contexte, ce qui règle la dilution observée avec l'ancienne approche (question noyée dans
l'historique).

Boucle manuelle via le client `ollama` (pas les abstractions d'agent de LlamaIndex) : contrôle
complet sur le streaming SSE, le logging de debug par étape, et l'arrêt. Format de streaming
validé empiriquement sur ce modèle (voir Annexe du plan) : gpt-oss (harmony) sépare proprement
le canal "thinking" (raisonnement, streamé token par token), le canal "content" (réponse finale,
streamé token par token) et les tool_calls (livrés groupés dans le chunk terminal) — jamais de
contenu mélangé à un appel d'outil, donc streamer "content" en direct sans le retenir est sûr.
"""
import json
import os
from datetime import datetime, timezone

import ollama

from .citations import SemanticScholarClient, SemanticScholarError
from .retriever import MAX_HISTORY_TURNS, SECTION_SCOPES, RAGRetriever, SourceChunk
from .utils import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_STEPS = 6

TITLE_SYSTEM_PROMPT = (
    "Génère un titre court (3 à 6 mots, en français, sans guillemets ni ponctuation finale) "
    "résumant le sujet de l'échange suivant. Réponds UNIQUEMENT avec le titre, rien d'autre."
)

AGENT_SYSTEM_PROMPT = (
    "Tu es un assistant de recherche scientifique expert en robotique, avec accès à trois outils :\n"
    "- search_corpus : recherche PAR PERTINENCE (quelques meilleurs passages) dans la bibliothèque "
    "Zotero locale de l'utilisateur. Pour \"que dit la littérature sur X\", \"explique la méthode Y\".\n"
    "- scan_corpus : balayage EXHAUSTIF (tous les articles, pas seulement les plus pertinents) de la "
    "bibliothèque locale pour un mot-clé littéral. Pour \"quels/combien d'articles citent/mentionnent "
    "X\", \"qui, dans MA bibliothèque, cite l'article Y\". search_corpus ne peut PAS répondre à ce "
    "genre de question : il plafonne à quelques résultats, jamais une liste exhaustive.\n"
    "- get_external_citations : cherche, dans la littérature scientifique MONDIALE (Semantic "
    "Scholar, pas seulement la bibliothèque locale), quels articles CITENT un article donné, avec "
    "la phrase de citation.\n\n"
    "Utilise ces outils autant de fois que nécessaire pour rassembler l'information avant de "
    "répondre — n'invente jamais un résultat. Formule pour search_corpus une requête autonome et "
    "ciblée : si la question de l'utilisateur dépend du contexte de la conversation (ex: \"et pour "
    "le MPC ?\"), cherche le sujet réel (\"model predictive control\"), pas la question brute ni "
    "l'historique complet.\n\n"
    "Portée de section (search_corpus et scan_corpus) : \"content\" (défaut, exclut les "
    "bibliographies — bon pour les questions de contenu), \"references\" (uniquement les "
    "bibliographies — pour trouver qui CITE un article), \"all\". Pour \"qui cite l'article X dans "
    "ma bibliothèque\", utilise scan_corpus avec le NOM D'UN AUTEUR de X comme mot-clé (plus fiable "
    "qu'un titre complet, ex: \"Makhal\" plutôt que le titre) et sections=[\"references\"].\n\n"
    "Pour get_external_citations, préfère fournir le DOI si tu le connais déjà (via un résultat "
    "précédent de search_corpus/scan_corpus) : cela lève les ambiguïtés de titre. Si "
    "get_external_citations renvoie zéro résultat, ne cherche PAS la solution en relançant "
    "search_corpus ou scan_corpus (aucun des deux ne contient de données de citations dans la "
    "littérature mondiale) : réessaie au plus une fois get_external_citations avec un titre "
    "reformulé, sinon accepte le résultat et dis-le à l'utilisateur.\n\n"
    "Quand tu as assez d'information, réponds en français, de façon précise et technique, en "
    "Markdown (tableaux/listes bienvenus si utiles). Pour chaque affirmation issue d'une source, "
    "cite-la entre crochets [Auteur, Année]. Si tu n'as rien trouvé de pertinent après avoir "
    "cherché, dis-le explicitement plutôt que d'inventer une réponse."
)

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Recherche PAR PERTINENCE (quelques meilleurs passages, pas exhaustif) dans le "
                "corpus local de PDFs scientifiques (bibliothèque Zotero de l'utilisateur, environ "
                "236 articles). À utiliser pour toute question sur le CONTENU d'un ou plusieurs "
                "articles : méthode, résultats, définitions, comparaisons. Pour énumérer/compter "
                "TOUS les articles vérifiant un critère (ex: qui cite un article donné), utiliser "
                "scan_corpus à la place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête de recherche en langage naturel, autonome et ciblée sur un sujet précis.",
                    },
                    "item_key": {
                        "type": "string",
                        "description": (
                            "Optionnel. Identifiant d'article pour limiter la recherche à un seul "
                            "article précis. Laisser vide pour chercher dans toute la bibliothèque."
                        ),
                    },
                    "section_scope": {
                        "type": "string",
                        "enum": list(SECTION_SCOPES),
                        "description": (
                            "Optionnel, défaut 'content'. 'content' = exclut les bibliographies "
                            "(questions de contenu). 'references' = uniquement les bibliographies. "
                            "'all' = tout."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_corpus",
            "description": (
                "Balaie EXHAUSTIVEMENT tout le corpus local (pas une recherche par pertinence "
                "limitée à quelques résultats) pour un mot-clé littéral : renvoie TOUS les articles "
                "qui le contiennent, groupés par article, avec un compte total exact. À utiliser "
                "pour toute question d'énumération ou de comptage : \"quels/combien d'articles "
                "citent/mentionnent X\", \"qui, dans ma bibliothèque, cite l'article Y\"."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": (
                            "Mot-clé LITTÉRAL à chercher (nom d'auteur, terme technique) — pas une "
                            "phrase en langage naturel. Pour chercher qui cite un article, utiliser "
                            "le nom d'un de ses auteurs (plus fiable qu'un titre complet)."
                        ),
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SECTION_SCOPES)},
                        "description": (
                            "Optionnel, défaut : tout le corpus. Utiliser [\"references\"] pour "
                            "chercher des citations (qui mentionne cet auteur/titre en bibliographie)."
                        ),
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_external_citations",
            "description": (
                "Cherche, dans la littérature scientifique mondiale (Semantic Scholar), quels "
                "articles citent un article donné, avec la phrase de citation et l'intention "
                "(background/méthode/comparaison). À utiliser pour toute question du type \"qui "
                "cite X\", \"comment X a été repris dans la littérature\", \"quelles limites ont "
                "été soulevées par d'autres travaux\" — le corpus local ne suffit pas à répondre "
                "à ce type de question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Titre exact (ou aussi précis que possible) de l'article dont on cherche les citations.",
                    },
                    "doi": {
                        "type": "string",
                        "description": "Optionnel mais recommandé si connu : lève l'ambiguïté entre articles de titres proches ou homonymes.",
                    },
                },
                "required": ["title"],
            },
        },
    },
]


class RAGAgent:
    def __init__(
        self,
        config: dict,
        retriever: RAGRetriever,
        s2_client: SemanticScholarClient | None = None,
    ):
        self.config = config
        self.retriever = retriever
        ollama_cfg = config["ollama"]
        self.model = ollama_cfg["llm_model"]
        self.client = ollama.Client(host=ollama_cfg["base_url"])

        agent_cfg = config.get("agent") or {}
        self.max_steps = agent_cfg.get("max_steps", DEFAULT_MAX_STEPS)

        s2_cfg = config.get("semantic_scholar") or {}
        self.s2_client = s2_client or SemanticScholarClient(
            api_key=s2_cfg.get("api_key") or os.environ.get("S2_API_KEY")
        )

        # Même journal que RAGRetriever (data/debug/query_log.jsonl) : entrées distinguées par
        # "kind" ("agent_run" vs les entrées historiques de query()), pour un seul `tail -f`.
        self.debug_log_path = retriever.debug_log_path

    def run_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        item_key: str | None = None,
        ui_scope_checked: bool | None = None,
        ui_selected_item_key: str | None = None,
    ):
        """Générateur d'événements SSE-ready (dicts). Types émis :
        thinking {text} · token {text} · step {step, tool, args} ·
        sources {kind: corpus|external, items} · done {answer, sources, external_sources} ·
        error {message}.

        item_key, si fourni (case "Limiter à cet article" cochée côté UI), est un filtre DUR :
        il prime sur tout item_key que le LLM tenterait de passer à search_corpus — cohérent avec
        le contrat déjà établi côté UI (verrou = garantie, pas juste une préférence). Il est AUSSI
        injecté dans le prompt système (voir _open_article_context) : sans ça, le LLM ignore
        qu'un article est ouvert et à quoi « cet article » fait référence (il demandait lequel)."""
        system_content = AGENT_SYSTEM_PROMPT
        if item_key:
            system_content += "\n\n" + self._open_article_context(
                item_key, self.retriever.get_document(item_key)
            )
        messages = [{"role": "system", "content": system_content}]
        for h in (history or [])[-MAX_HISTORY_TURNS:]:
            if h.get("question"):
                messages.append({"role": "user", "content": h["question"]})
            if h.get("answer"):
                messages.append({"role": "assistant", "content": h["answer"]})
        messages.append({"role": "user", "content": question})

        debug_entry = {
            "kind": "agent_run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_raw": question,
            "item_key_received": item_key,
            "ui_scope_checked": ui_scope_checked,
            "ui_selected_item_key": ui_selected_item_key,
            "history_turns_received": len(history) if history else 0,
            "llm_model": self.model,
            "max_steps": self.max_steps,
            "steps": [],
        }

        all_sources: list[SourceChunk] = []
        all_external: list[dict] = []

        try:
            for step_idx in range(self.max_steps):
                thinking_buf, content_buf, tool_calls = "", "", None
                for chunk in self.client.chat(
                    model=self.model, messages=messages, tools=TOOL_DEFS, stream=True
                ):
                    if chunk.message.thinking:
                        thinking_buf += chunk.message.thinking
                        yield {"type": "thinking", "text": chunk.message.thinking}
                    if chunk.message.content:
                        content_buf += chunk.message.content
                        yield {"type": "token", "text": chunk.message.content}
                    if chunk.message.tool_calls:
                        # Observé empiriquement : gpt-oss livre les tool_calls groupés dans
                        # l'unique chunk terminal. Accumule quand même par sécurité (delta,
                        # comme "content") au cas où un modèle streamerait plusieurs tool_calls
                        # sur des chunks distincts.
                        tool_calls = (tool_calls or []) + list(chunk.message.tool_calls)
                    if chunk.done:
                        break

                if not tool_calls:
                    debug_entry["final_answer"] = content_buf
                    debug_entry["error"] = None
                    self._write_debug(debug_entry)
                    yield {
                        "type": "done",
                        "answer": content_buf,
                        "sources": [self._source_to_dict(s) for s in all_sources],
                        "external_sources": all_external,
                    }
                    return

                messages.append(
                    {"role": "assistant", "content": content_buf, "tool_calls": tool_calls}
                )

                for tc in tool_calls:
                    name = tc.function.name
                    args = dict(tc.function.arguments or {})
                    yield {"type": "step", "step": step_idx + 1, "tool": name, "args": args}

                    step_trace = {"tool": name, "args": args}
                    try:
                        result_text, events = self._execute_tool(
                            name, args, item_key, step_trace, all_sources, all_external
                        )
                        for event in events:
                            yield event
                    except Exception as e:
                        logger.warning(f"Tool {name} failed: {e}")
                        result_text = f"Erreur lors de l'exécution de l'outil {name} : {e}"
                        step_trace["error"] = str(e)

                    debug_entry["steps"].append(step_trace)
                    messages.append(
                        {"role": "tool", "content": result_text, "tool_name": name}
                    )

            fallback_answer = self._forced_synthesis(messages)
            debug_entry["final_answer"] = fallback_answer
            debug_entry["error"] = None
            debug_entry["max_steps_exceeded"] = True
            self._write_debug(debug_entry)
            yield {
                "type": "done",
                "answer": fallback_answer,
                "sources": [self._source_to_dict(s) for s in all_sources],
                "external_sources": all_external,
            }
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            debug_entry["error"] = str(e)
            self._write_debug(debug_entry)
            yield {"type": "error", "message": str(e)}

    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        name: str,
        args: dict,
        ui_item_key: str | None,
        step_trace: dict,
        all_sources: list[SourceChunk],
        all_external: list[dict],
    ) -> tuple[str, list[dict]]:
        """Exécute un outil, enrichit step_trace et les accumulateurs de sources en place, et
        renvoie (texte à réinjecter au LLM, événements SSE à émettre par l'appelant — cette
        méthode n'est pas elle-même un générateur, pour rester facilement testable seule)."""
        events: list[dict] = []
        if name == "search_corpus":
            result_text, sources, retr_trace = self._execute_search_corpus(args, ui_item_key)
            all_sources.extend(sources)
            step_trace["retrieval"] = retr_trace
            step_trace["n_sources"] = len(sources)
            if sources:
                events.append({
                    "type": "sources", "kind": "corpus",
                    "items": [self._source_to_dict(s) for s in sources],
                })
        elif name == "scan_corpus":
            result_text, sources, scan_trace = self._execute_scan_corpus(args)
            all_sources.extend(sources)
            step_trace["scan"] = scan_trace
            step_trace["n_sources"] = len(sources)
            if sources:
                events.append({
                    "type": "sources", "kind": "corpus",
                    "items": [self._source_to_dict(s) for s in sources],
                })
        elif name == "get_external_citations":
            result_text, ext_sources = self._execute_get_external_citations(args)
            all_external.extend(ext_sources)
            step_trace["n_external_sources"] = len(ext_sources)
            if ext_sources:
                events.append({"type": "sources", "kind": "external", "items": ext_sources})
        else:
            result_text = f"Outil « {name} » inconnu."
            step_trace["error"] = result_text
        return result_text, events

    def _execute_search_corpus(
        self, args: dict, ui_item_key: str | None
    ) -> tuple[str, list[SourceChunk], dict]:
        query = (args.get("query") or "").strip()
        if not query:
            return "Requête de recherche vide — aucun résultat.", [], {}
        # Le verrou UI (case "Limiter à cet article") prime sur l'item_key choisi par le LLM.
        item_key = ui_item_key or (args.get("item_key") or None)
        section_scope = args.get("section_scope") or "content"
        if section_scope not in SECTION_SCOPES:
            section_scope = "content"

        trace: dict = {}
        nodes = self.retriever._retrieve_and_rerank(
            query, item_key, section_scope=section_scope, trace=trace
        )
        sources = self.retriever._nodes_to_sources(nodes)
        if not sources:
            return "Aucun passage pertinent trouvé dans le corpus pour cette requête.", [], trace

        lines = [
            f'- [{s.item_key}] {s.authors} ({s.year}) — "{s.title}", p.{s.page_number} : '
            f"{s.text[:500]}"
            for s in sources
        ]
        return "\n".join(lines), sources, trace

    def _execute_scan_corpus(self, args: dict) -> tuple[str, list[SourceChunk], dict]:
        """Convertit le résultat de RAGRetriever.scan_corpus en SourceChunk (même forme que
        search_corpus) pour réutiliser telle quelle la restitution UI existante (cartes
        cliquables PDF) sans aucune modification frontend."""
        keyword = (args.get("keyword") or "").strip()
        if not keyword:
            return "Mot-clé vide — aucun résultat.", [], {}
        sections = args.get("sections") or None
        if sections is not None:
            sections = [s for s in sections if s in SECTION_SCOPES] or None

        result = self.retriever.scan_corpus(keyword, sections=sections)
        total = result["total_articles"]
        articles = result["articles"]
        trace = {"keyword": keyword, "sections": sections, "total_articles": total}
        if total == 0:
            return f"Aucun article du corpus local ne contient « {keyword} ».", [], trace

        lines = [f"{total} article(s) du corpus local contiennent « {keyword} » :"]
        sources: list[SourceChunk] = []
        for a in articles:
            ref_note = " (bibliographie uniquement, pas de discussion)" if a["in_references_only"] else ""
            lines.append(
                f'- [{a["item_key"]}] {a["authors"]} ({a["year"]}) — "{a["title"]}", '
                f'{a["n_hits"]} occurrence(s), p.{a["page"]}{ref_note} : {a["snippet"]}'
            )
            sources.append(SourceChunk(
                text=a["snippet"], score=float(a["n_hits"]), item_key=a["item_key"],
                title=a["title"], authors=a["authors"], year=a["year"], doi=a["doi"],
                page_number=a["page"], section="references" if a["in_references_only"] else "",
                pdf_path="",
            ))
        return "\n".join(lines), sources, trace

    def _execute_get_external_citations(self, args: dict) -> tuple[str, list[dict]]:
        title = (args.get("title") or "").strip()
        doi = args.get("doi") or None
        if not title and not doi:
            return "Ni titre ni DOI fournis — impossible de résoudre l'article.", []

        try:
            paper = self.s2_client.resolve_paper(doi=doi, title=title)
        except SemanticScholarError as e:
            return f"Erreur Semantic Scholar : {e}", []

        if not paper:
            return f"Aucun article correspondant à « {title or doi} » trouvé sur Semantic Scholar.", []

        try:
            citations = self.s2_client.get_citations(paper["paperId"])
        except SemanticScholarError as e:
            return (
                f"Article trouvé (« {paper.get('title')} ») mais erreur lors de la récupération "
                f"des citations : {e}",
                [],
            )

        if not citations:
            return f"« {paper.get('title')} » trouvé mais aucune citation recensée sur Semantic Scholar.", []

        lines = [f"Citations de « {paper.get('title')} » ({paper.get('year')}) :"]
        external_sources = []
        for c in citations:
            citing = c.get("citingPaper") or {}
            contexts = c.get("contexts") or []
            ctx = " / ".join(contexts) or "(pas de phrase de citation disponible)"
            intents = ", ".join(c.get("intents") or []) or "non précisé"
            authors = ", ".join(a.get("name", "") for a in (citing.get("authors") or [])[:3])
            lines.append(
                f'- {authors} ({citing.get("year")}) — "{citing.get("title")}" '
                f"[intention : {intents}] : {ctx}"
            )
            paper_id = citing.get("paperId")
            external_sources.append({
                "title": citing.get("title"),
                "year": citing.get("year"),
                "authors": authors,
                "intents": c.get("intents") or [],
                "context": ctx,
                "url": f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None,
            })
        return "\n".join(lines), external_sources

    def _forced_synthesis(self, messages: list[dict]) -> str:
        """Repli si MAX_STEPS est dépassé : un dernier appel SANS tools (le modèle ne peut
        alors plus décider d'en appeler un) pour forcer une synthèse à partir de ce qui a déjà
        été rassemblé, plutôt que de couper la conversation sans réponse."""
        forced_messages = messages + [{
            "role": "user",
            "content": (
                "Réponds maintenant directement, en synthèse, à partir des informations déjà "
                "obtenues ci-dessus (n'appelle plus aucun outil)."
            ),
        }]
        try:
            resp = self.client.chat(model=self.model, messages=forced_messages, stream=False)
            return resp.message.content or "Je n'ai pas pu formuler de réponse dans le nombre d'étapes imparti."
        except Exception as e:
            logger.error(f"Forced synthesis failed: {e}")
            return f"Erreur lors de la synthèse finale : {e}"

    @staticmethod
    def _open_article_context(item_key: str, doc: dict | None) -> str:
        """Bloc ajouté au prompt système quand un article est ouvert côté UI (scope verrouillé) :
        donne au LLM l'identité de l'article pour que « cet article » se résolve, et lui indique
        que la recherche est déjà limitée à celui-ci (donc appeler search_corpus, pas demander
        lequel)."""
        if doc and doc.get("title"):
            ref = f'« {doc["title"]} »'
            extra = [x for x in [doc.get("authors"), str(doc["year"]) if doc.get("year") else None] if x]
            if extra:
                ref += f' ({", ".join(extra)})'
        else:
            ref = f"l'article d'identifiant {item_key}"
        return (
            f"CONTEXTE : l'utilisateur consulte actuellement un article précis dans la visionneuse : "
            f"{ref}. Toute mention de « cet article », « ce papier » ou « ce document » désigne "
            f"CELUI-CI — ne demande jamais de quel article il s'agit. Ta recherche dans le corpus "
            f"est automatiquement limitée à cet article : utilise search_corpus pour récupérer son "
            f"contenu et répondre (par exemple, pour un résumé, cherche ses contributions, sa "
            f"méthode et ses résultats)."
        )

    def generate_title(self, question: str, answer: str) -> str:
        """Titre court pour une conversation, généré à partir de son 1er échange — appelé par
        le serveur APRÈS la réponse complète (n'ajoute donc aucune latence perçue). Jamais
        fatal : repli sur la question tronquée si l'appel échoue ou ne renvoie rien.

        num_predict généreux (500) : gpt-oss consacre une partie de son budget de tokens au
        canal "thinking" même pour une tâche aussi simple qu'un titre (observé empiriquement,
        un budget trop serré peut renvoyer un content vide) — think=False n'est pas respecté
        par ce modèle, la seule protection fiable est un budget large + repli."""
        fallback = question.strip()[:50] or "Nouvelle conversation"
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": TITLE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question : {question}\nRéponse : {answer[:500]}"},
                ],
                stream=False,
                options={"num_predict": 500},
            )
            title = (resp.message.content or "").strip().strip('"').strip("«»").strip()
            if len(title) > 60:
                title = title[:57].rsplit(" ", 1)[0].rstrip() + "…"
            return title or fallback
        except Exception as e:
            logger.warning(f"generate_title failed, falling back to truncated question: {e}")
            return fallback

    @staticmethod
    def _source_to_dict(s: SourceChunk) -> dict:
        return {
            "item_key": s.item_key,
            "title": s.title,
            "authors": s.authors,
            "year": s.year,
            "doi": s.doi,
            "page": s.page_number,
            "text": s.text,
            "score": round(s.score, 5),
        }

    def _write_debug(self, entry: dict) -> None:
        try:
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Could not write debug log (non-fatal): {e}")
