"""Client Semantic Scholar Graph API — citations externes (littérature mondiale), en
complément du corpus Zotero local (236 PDFs). Utilisé par l'outil get_external_citations de
RAGAgent. urllib (stdlib) plutôt qu'une dépendance HTTP : usage ponctuel, pas de session à
maintenir. Jamais fatal : toute erreur réseau/HTTP est levée comme SemanticScholarError, à
charge de l'appelant (agent.py) de la transformer en résultat d'outil que le LLM peut lire."""
import json
import urllib.error
import urllib.parse
import urllib.request

from .utils import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,year,authors,externalIds"
CITATION_FIELDS = (
    "contexts,intents,citingPaper.paperId,citingPaper.title,citingPaper.year,"
    "citingPaper.authors,citingPaper.externalIds"
)
DEFAULT_TIMEOUT = 15  # secondes ; ne doit jamais bloquer longtemps la boucle de l'agent


class SemanticScholarError(Exception):
    """Erreur non fatale : message destiné à être renvoyé au LLM comme résultat d'outil
    (il doit pouvoir l'expliquer à l'utilisateur ou changer de stratégie), pas seulement loggé."""


class SemanticScholarClient:
    def __init__(self, api_key: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        self.api_key = api_key
        self.timeout = timeout

    def resolve_paper(self, doi: str | None = None, title: str | None = None) -> dict | None:
        """Résout un papier par DOI (exact, prioritaire — évite les collisions de titre, ex.
        l'article "Reuleaux" (2018) vs l'ingénieur Franz Reuleaux) sinon par recherche de
        titre (meilleur résultat retourné par l'API). Renvoie None si rien n'est trouvé
        (pas une erreur : un papier absent de Semantic Scholar est un cas normal)."""
        if doi:
            try:
                return self._get(
                    f"/paper/DOI:{urllib.parse.quote(doi, safe='')}", {"fields": PAPER_FIELDS}
                )
            except SemanticScholarError as e:
                logger.warning(f"DOI lookup failed, falling back to title search: {e}")
        if not title:
            return None
        data = self._get("/paper/search", {"query": title, "fields": PAPER_FIELDS, "limit": 1})
        results = data.get("data") or []
        return results[0] if results else None

    def get_citations(self, paper_id: str, limit: int = 10) -> list[dict]:
        """Renvoie les articles qui citent paper_id, avec la phrase de citation (contexts)
        et l'intention (intents : background/methodology/result comparison...) quand
        disponibles — c'est contexts qui permet de dire COMMENT l'article a été cité, pas
        seulement PAR QUI."""
        data = self._get(
            f"/paper/{urllib.parse.quote(paper_id, safe='')}/citations",
            {"fields": CITATION_FIELDS, "limit": limit},
        )
        return data.get("data") or []

    def _get(self, path: str, params: dict) -> dict:
        url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise SemanticScholarError(
                    "Semantic Scholar a limité les requêtes (429, pool anonyme). "
                    "Réessayez plus tard ou configurez semantic_scholar.api_key."
                ) from e
            raise SemanticScholarError(f"Semantic Scholar a répondu {e.code} : {e.reason}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise SemanticScholarError(f"Semantic Scholar injoignable : {e}") from e
        except json.JSONDecodeError as e:
            raise SemanticScholarError(f"Réponse Semantic Scholar invalide : {e}") from e
