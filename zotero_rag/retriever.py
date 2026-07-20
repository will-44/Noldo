import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from .utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Tu es un assistant de recherche scientifique expert en robotique.\n"
    "Réponds en français à la question posée en te basant UNIQUEMENT sur les extraits "
    "fournis. Si l'information n'est pas dans les extraits, dis-le explicitement.\n"
    "Pour chaque affirmation, indique entre crochets le nom de l'auteur et l'année : "
    "[Nom, Année]. Sois précis et technique."
)

QA_TEMPLATE = PromptTemplate(
    "Contexte :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    f"{SYSTEM_PROMPT}\n\n"
    "Question : {query_str}\n"
    "Réponse :"
)

RERANKER_MODEL = "BAAI/bge-reranker-base"  # 278M, CPU-friendly (~130ms/lot de 16 paires)
MAX_HISTORY_TURNS = 3  # borne la taille du prompt et la dilution de la recherche
DIVERSITY_MAX_PER_DOC = 2  # mode global : nb max de passages retenus par article après reranking

# Mots-clés (comparaison sur section.lower()) identifiant une section références/bibliographie/
# remerciements : ~24% des chunks du corpus (mesuré sur ce projet). Pour une question de
# *contenu*, c'est du bruit truffé de noms propres qui polluent BM25 et faussent le reranker ;
# mais pour une question de *citation* ("qui cite X"), c'est au contraire là que vit la réponse.
# D'où le scope explicite (SECTION_SCOPES) plutôt qu'un filtre inconditionnel.
REFERENCE_SECTION_KEYWORDS = {
    "references", "bibliograph", "acknowledg", "remerciement", "funding", "publisher's note",
}

# Portées de section exposées à l'agent (search_corpus) : "content" = tout sauf références
# (défaut, questions de contenu) · "references" = uniquement la biblio (questions de citation) ·
# "all" = aucun filtre. Seule la frontière références/contenu est fiable : les labels de section
# sont trop spécifiques au papier (1988 valeurs distinctes) pour un filtrage fin méthodo/résultats.
SECTION_SCOPES = ("content", "references", "all")


def _despace_section(section: str) -> str:
    """Normalise les entêtes extraites avec une lettre par "mot" (artefact d'extraction sur
    certains PDF anciens/scannés, ex: "B I B L I O G R A P H Y") en un mot compact, pour que
    REFERENCE_SECTION_KEYWORDS les reconnaisse. Heuristique restrictive (>3 tokens, tous ≤2
    caractères) pour ne jamais fusionner un entête normal ("3 Related Work", "A. Results")."""
    tokens = section.split()
    if len(tokens) > 3 and all(len(t) <= 2 for t in tokens):
        return "".join(tokens)
    return section


def _is_reference_section(section: str | None) -> bool:
    s = _despace_section((section or "")).lower()
    return any(kw in s for kw in REFERENCE_SECTION_KEYWORDS)


def _in_section_scope(section: str | None, scope: str) -> bool:
    """True si un passage de section donnée entre dans la portée demandée."""
    if scope == "all":
        return True
    if scope == "references":
        return _is_reference_section(section)
    return not _is_reference_section(section)  # "content" (défaut)


class TopKPostprocessor(BaseNodePostprocessor):
    """Simple troncature au top_k après reranking. Utilisé en mode "cet article" : la
    diversité inter-documents n'a pas de sens quand le pool ne contient qu'un seul article."""

    top_k: int = 5

    @classmethod
    def class_name(cls) -> str:
        return "TopKPostprocessor"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        return nodes[: self.top_k]


class DiversityCapPostprocessor(BaseNodePostprocessor):
    """Plafonne le nombre de passages retenus par article après reranking (mode global
    uniquement). Sans ça, le cross-encoder peut faire remonter presque exclusivement
    l'article qui se cite lui-même : son propre titre/résumé est toujours le plus "pertinent"
    textuellement à une question qui le mentionne — observé concrètement sur ce projet
    (4/5 sources finales venant du même article après reranking, alors que le pool
    pré-reranking en contenait 8 distincts). Ceci écrase la diversité que BM25+vecteur avaient
    pourtant trouvée ; ce postprocessor la restaure en gardant le classement du reranker mais
    en limitant la contribution de chaque article."""

    max_per_document: int = DIVERSITY_MAX_PER_DOC
    top_k: int = 5

    @classmethod
    def class_name(cls) -> str:
        return "DiversityCapPostprocessor"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        counts: dict[str, int] = {}
        kept: list[NodeWithScore] = []
        for node in nodes:  # déjà triés par pertinence par le reranker en amont
            key = (node.metadata or {}).get("item_key", "")
            if counts.get(key, 0) >= self.max_per_document:
                continue
            counts[key] = counts.get(key, 0) + 1
            kept.append(node)
            if len(kept) >= self.top_k:
                break
        return kept


@dataclass
class SourceChunk:
    text: str
    score: float
    item_key: str
    title: str
    authors: str
    year: int | None
    doi: str | None
    page_number: int
    section: str
    pdf_path: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)


class RAGRetriever:
    COLLECTION_NAME = "zotero_rag"

    def __init__(self, config: dict):
        self.config = config
        ollama_cfg = config["ollama"]

        Settings.llm = Ollama(
            model=ollama_cfg["llm_model"],
            base_url=ollama_cfg["base_url"],
            request_timeout=ollama_cfg["timeout"],
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=ollama_cfg["embed_model"],
            base_url=ollama_cfg["base_url"],
        )

        persist_dir = config["rag"]["persist_dir"]
        self.pdf_cache_dir = Path(config["rag"]["pdf_cache_dir"])
        self.similarity_top_k = config["rag"]["similarity_top_k"]
        # Journal de debug (JSON Lines) : un objet par requête, params reçus + candidats
        # avant/après reranking + réponse. `tail -f` pendant les tests en conditions réelles.
        # Partagé avec RAGAgent (même fichier, entrées "kind":"agent_step" entrelacées).
        self.debug_log_path = Path(persist_dir).parent / "debug" / "query_log.jsonl"
        self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Nombre de candidats remontés par la récupération AVANT reranking : plus large que
        # similarity_top_k (le filtre références en écarte ~24%, et le reranker/diversity_cap
        # ont besoin de matière à trier).
        self.fetch_k = min(self.similarity_top_k * 6, 30)
        # top_n=fetch_k : le reranker retrie TOUT le pool sans le tronquer — la troncature
        # finale est déléguée à un second postprocessor (différent selon le mode), pour ne
        # charger le modèle cross-encoder qu'une seule fois en mémoire (~1,1 Go) et réutilisé
        # par les deux moteurs.
        self.reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=self.fetch_k)
        # Mode "cet article" : simple troncature, un seul document dans le pool.
        self.doc_trim = TopKPostprocessor(top_k=self.similarity_top_k)
        # Mode global : plafonne les passages par article pour éviter qu'un document ne
        # monopolise le résultat après reranking (voir DiversityCapPostprocessor).
        self.diversity_cap = DiversityCapPostprocessor(
            max_per_document=DIVERSITY_MAX_PER_DOC, top_k=self.similarity_top_k
        )

        chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = chroma_client.get_or_create_collection(
            self.COLLECTION_NAME
        )

        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

        # Retriever global : fusion vecteur + BM25 (mots-clés). Le mode par-document reste un
        # filtre vectoriel simple, construit à la volée dans _retrieve_and_rerank. Ni l'un ni
        # l'autre ne fait de synthèse — _synthesize() est un point unique, partagé par query()
        # (CLI/tests) et par l'agent (agent.py), pour ne jamais diverger sur le prompt LLM.
        self.fused_retriever = self._build_fused_retriever()

    # ------------------------------------------------------------------
    # Récupération (partagée entre query() et l'outil search_corpus de l'agent)
    # ------------------------------------------------------------------

    def _retrieve_and_rerank(
        self,
        query: str,
        item_key: str | None = None,
        section_scope: str = "content",
        trace: dict | None = None,
    ) -> list[NodeWithScore]:
        """Pipeline de récupération complet : retriever adéquat → filtre de portée de section
        → reranker (cross-encoder) → troncature (diversity_cap en mode global, doc_trim en mode
        "cet article"). Point d'entrée unique utilisé par query() et par l'outil search_corpus
        de l'agent — garantit qu'ils ne divergent jamais.

        section_scope ∈ SECTION_SCOPES : "content" (défaut, exclut références/biblio — bruit pour
        une question de contenu), "references" (uniquement la biblio — questions de citation),
        "all" (aucun filtre). Voir _in_section_scope.

        trace (optionnel) : si fourni, rempli avec les 3 étapes intermédiaires
        (candidats_bruts / avant_rerank / après_rerank) pour le journal de debug — n'affecte
        jamais le résultat retourné."""
        query_bundle = QueryBundle(query)
        if item_key:
            retriever = self.index.as_retriever(
                similarity_top_k=self.fetch_k, filters=self._filters_for(item_key)
            )
            trim = self.doc_trim
        else:
            retriever = self.fused_retriever
            trim = self.diversity_cap

        nodes = retriever.retrieve(query_bundle)
        if trace is not None:
            trace["candidates_raw"] = self._nodes_to_debug(nodes)

        nodes = [
            n for n in nodes if _in_section_scope((n.metadata or {}).get("section"), section_scope)
        ]
        if trace is not None:
            trace["candidates_before_rerank"] = self._nodes_to_debug(nodes)

        nodes = self.reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
        nodes = trim.postprocess_nodes(nodes, query_bundle=query_bundle)
        if trace is not None:
            trace["candidates_after_rerank"] = self._nodes_to_debug(nodes)

        return nodes

    def scan_corpus(
        self, keyword: str, sections: list[str] | None = None, max_articles: int = 40
    ) -> dict:
        """Balayage EXHAUSTIF (pas top-k) du corpus entier pour un mot-clé littéral (substring,
        insensible à la casse), groupé par article. Complète _retrieve_and_rerank (pertinence
        sémantique, plafonné à ~top_k) pour les questions d'énumération/comptage — "qui cite X",
        "combien d'articles mentionnent Y" — qu'un classement par pertinence ne peut
        structurellement pas résoudre : vérifié sur ce corpus, 26 articles citent un papier donné
        mais le meilleur classement sémantique n'en remonte que 2-3.

        sections : sous-ensemble de SECTION_SCOPES à inclure (défaut : tout le corpus, comme
        "all"). Pour une recherche de citations, passer sections=["references"].

        Renvoie {"total_articles": int, "articles": [...]}, trié par n_hits décroissant, tronqué
        à max_articles (total_articles reste le compte exact même si la liste est tronquée).
        Chaque article : item_key/title/authors/year/doi/n_hits/page/snippet/
        in_references_only — le snippet privilégie une occurrence hors-références (discussion
        substantielle) quand elle existe, sinon une occurrence en bibliographie (confirme la
        citation sans discussion)."""
        if not keyword or not keyword.strip():
            return {"total_articles": 0, "articles": []}
        needle = keyword.strip().lower()
        scopes = sections or ["all"]

        records = self.chroma_collection.get(include=["documents", "metadatas"])
        by_article: dict[str, dict] = {}
        for text, meta in zip(records["documents"], records["metadatas"]):
            if not text or needle not in text.lower():
                continue
            meta = meta or {}
            section = meta.get("section")
            if not any(_in_section_scope(section, s) for s in scopes):
                continue
            key = meta.get("item_key")
            if not key:
                continue

            state = by_article.setdefault(key, {
                "item_key": key,
                "title": meta.get("title", "Sans titre"),
                "authors": meta.get("authors", ""),
                "year": meta.get("year"),
                "doi": meta.get("doi") or None,
                "n_hits": 0,
                "_best_body": None,  # (page, snippet) hors références
                "_best_ref": None,   # (page, snippet) en référence seulement
            })
            state["n_hits"] += 1
            snippet = self._snippet_around(text, needle)
            page = meta.get("page_number", 1)
            if _is_reference_section(section):
                if state["_best_ref"] is None:
                    state["_best_ref"] = (page, snippet)
            elif state["_best_body"] is None:
                state["_best_body"] = (page, snippet)

        articles = []
        for state in by_article.values():
            page, snippet = state["_best_body"] or state["_best_ref"]
            articles.append({
                "item_key": state["item_key"],
                "title": state["title"],
                "authors": state["authors"],
                "year": state["year"],
                "doi": state["doi"],
                "n_hits": state["n_hits"],
                "page": page,
                "snippet": snippet,
                "in_references_only": state["_best_body"] is None,
            })
        articles.sort(key=lambda a: a["n_hits"], reverse=True)
        return {"total_articles": len(articles), "articles": articles[:max_articles]}

    @staticmethod
    def _snippet_around(text: str, needle: str, radius: int = 100) -> str:
        low = text.lower()
        idx = low.find(needle)
        if idx == -1:
            return text[:200].strip()
        start = max(0, idx - radius)
        end = min(len(text), idx + len(needle) + radius)
        # Recule/avance jusqu'à la frontière de mot la plus proche : ne coupe pas un mot en
        # deux (lisibilité, et le surlignage PDF côté frontend compare par mots entiers).
        while start > 0 and text[start - 1] != " ":
            start -= 1
        while end < len(text) and text[end] != " ":
            end += 1
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(text) else ""
        return prefix + text[start:end].strip().replace("\n", " ") + suffix

    @staticmethod
    def _synthesize(question: str, nodes: list[NodeWithScore]) -> str:
        """Synthèse LLM simple (un seul appel, pas de map-reduce) à partir de passages déjà
        récupérés — utilisée par query() et par l'étape finale de l'agent."""
        context_str = (
            "\n\n".join(n.text or "" for n in nodes)
            if nodes
            else "(Aucun extrait pertinent trouvé.)"
        )
        prompt = QA_TEMPLATE.format(context_str=context_str, query_str=question)
        return str(Settings.llm.complete(prompt))

    def query(
        self,
        question: str,
        item_key: str | None = None,
        history: list[dict] | None = None,
        ui_scope_checked: bool | None = None,
        ui_selected_item_key: str | None = None,
    ) -> RAGResponse:
        """Interroge l'index en un coup (pas d'agent) : utilisé par le CLI `main.py query` et
        les tests. Si item_key est fourni, la recherche est limitée à cet article ; sinon elle
        porte sur toute la bibliothèque. history (optionnel) est une liste de tours précédents
        {"question": ..., "answer": ...} : injectée dans le prompt pour le suivi de
        conversation, sans appel LLM de reformulation supplémentaire — contrairement à
        RAGAgent, qui formule lui-même sa requête de recherche à partir du contexte.
        ui_scope_checked / ui_selected_item_key sont purement diagnostiques (journal de debug
        uniquement) : ils n'influencent en rien la récupération ou la réponse."""
        logger.info(f"Query: {question[:80]} (item_key={item_key or 'global'})")
        composite_question = self._build_composite_question(question, history)
        debug_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_raw": question,
            "item_key_received": item_key,
            "ui_scope_checked": ui_scope_checked,
            "ui_selected_item_key": ui_selected_item_key,
            "scope_resolved": "doc" if item_key else "global",
            "history_turns_received": len(history) if history else 0,
            "composite_question": composite_question,
            "engine_used": "per_document_filtered_vector" if item_key else "global_fused_bm25_vector",
            "fetch_k": self.fetch_k,
            "similarity_top_k": self.similarity_top_k,
            "llm_model": self.config["ollama"]["llm_model"],
        }
        trace: dict = {}

        try:
            nodes = self._retrieve_and_rerank(composite_question, item_key, trace=trace)
            answer = self._synthesize(composite_question, nodes)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            debug_entry.update(trace)
            debug_entry["error"] = str(e)
            debug_entry["answer"] = None
            self._write_debug_log(debug_entry)
            return RAGResponse(answer=f"Erreur lors de la requête : {e}")

        sources = self._nodes_to_sources(nodes)
        debug_entry.update(trace)
        debug_entry["error"] = None
        debug_entry["answer"] = answer
        self._write_debug_log(debug_entry)
        return RAGResponse(answer=answer, sources=sources)

    @staticmethod
    def _nodes_to_debug(nodes: list[NodeWithScore]) -> list[dict]:
        return [
            {
                "item_key": (n.metadata or {}).get("item_key"),
                "page": (n.metadata or {}).get("page_number"),
                "score": float(n.score or 0.0),
                "text_preview": (n.text or "")[:120],
            }
            for n in nodes
        ]

    def _write_debug_log(self, entry: dict) -> None:
        try:
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not write debug log (non-fatal): {e}")

    @staticmethod
    def _build_composite_question(question: str, history: list[dict] | None) -> str:
        """Préfixe la question courante avec les derniers tours de conversation (question
        composite unique, utilisée à la fois pour la récupération et la synthèse — pas de
        second appel LLM). Sans historique, renvoie la question telle quelle (comportement
        inchangé)."""
        if not history:
            return question
        recent = history[-MAX_HISTORY_TURNS:]
        turns = "\n".join(
            f"Q: {h.get('question', '')}\nR: {h.get('answer', '')}" for h in recent
        )
        return (
            f"[Historique de la conversation]\n{turns}\n\n"
            f"[Question actuelle]\n{question}"
        )

    @staticmethod
    def _filters_for(item_key: str) -> MetadataFilters:
        return MetadataFilters(filters=[ExactMatchFilter(key="item_key", value=item_key)])

    def _build_fused_retriever(self) -> QueryFusionRetriever:
        """Retriever hybride global : fusion recherche vectorielle + BM25 (mots-clés), utile
        pour les noms propres et citations littérales que la similarité vectorielle seule
        manque souvent (ex: "qui cite l'article X ?"). num_queries=1 est essentiel : par défaut
        QueryFusionRetriever fait reformuler la requête par le LLM (4 appels supplémentaires),
        ce qui multiplierait la latence sur ce Jetson. Ici la fusion se limite à un
        reciprocal-rank-fusion pur, sans appel LLM additionnel."""
        nodes = self._load_all_nodes()
        vector_retriever = self.index.as_retriever(similarity_top_k=self.fetch_k)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=self.fetch_k
        )
        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=self.fetch_k,
            num_queries=1,
            mode="reciprocal_rerank",
            # use_async=True (défaut) orchestre les retrievers via asyncio en interne, ce qui
            # entre en conflit avec la boucle d'événements d'uvicorn ("Detected nested async").
            # On ne fusionne que 2 retrievers sur un petit corpus : le gain de parallélisme
            # serait de toute façon négligeable.
            use_async=False,
        )

    def _load_all_nodes(self) -> list[TextNode]:
        """Recharge tous les chunks depuis ChromaDB en TextNode : nécessaire pour BM25Retriever,
        car VectorStoreIndex.from_vector_store ne peuple pas de docstore local. Exhaustif
        (inclut les sections références) : le filtrage par portée de section se fait en aval,
        dans _retrieve_and_rerank (section_scope) — sinon section_scope="references" ne pourrait
        jamais remonter de résultats côté BM25."""
        records = self.chroma_collection.get(include=["documents", "metadatas"])
        nodes = []
        for doc_id, text, meta in zip(
            records["ids"], records["documents"], records["metadatas"]
        ):
            if not text:
                continue
            nodes.append(TextNode(id_=doc_id, text=text, metadata=meta or {}))
        return nodes

    def get_document(self, item_key: str) -> dict | None:
        """Métadonnées d'un seul article (title/authors/year/doi) par lookup ciblé — sans
        rebalayer tout le corpus comme list_documents. Utilisé par l'agent pour dire au LLM
        quel article est ouvert ("cet article" → celui-ci)."""
        records = self.chroma_collection.get(
            where={"item_key": item_key}, include=["metadatas"], limit=1
        )
        metas = records.get("metadatas") or []
        if not metas:
            return None
        m = metas[0]
        return {
            "item_key": item_key,
            "title": m.get("title", ""),
            "authors": m.get("authors", ""),
            "year": m.get("year"),
            "doi": m.get("doi") or None,
        }

    def list_documents(self) -> list[dict]:
        """Liste les articles indexés (dédupliqués par item_key), avec présence PDF."""
        records = self.chroma_collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in records.get("metadatas") or []:
            key = meta.get("item_key")
            if not key or key in docs:
                continue
            docs[key] = {
                "item_key": key,
                "title": meta.get("title", "Sans titre"),
                "authors": meta.get("authors", ""),
                "year": meta.get("year"),
                "doi": meta.get("doi") or None,
                "has_pdf": (self.pdf_cache_dir / f"{key}.pdf").exists(),
            }
        return sorted(docs.values(), key=lambda d: (d["title"] or "").lower())

    # ------------------------------------------------------------------

    @staticmethod
    def _nodes_to_sources(nodes: list[NodeWithScore]) -> list[SourceChunk]:
        seen: set[tuple] = set()
        chunks: list[SourceChunk] = []

        for node in nodes:
            meta = node.metadata or {}
            key = (meta.get("item_key"), meta.get("page_number"))
            if key in seen:
                continue
            seen.add(key)

            chunks.append(SourceChunk(
                text=node.text or "",
                score=float(node.score or 0.0),
                item_key=meta.get("item_key", ""),
                title=meta.get("title", ""),
                authors=meta.get("authors", ""),
                year=meta.get("year"),
                doi=meta.get("doi") or None,
                page_number=int(meta.get("page_number", 1)),
                section=meta.get("section", ""),
                pdf_path=meta.get("pdf_path", ""),
            ))

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks
