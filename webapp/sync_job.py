"""SyncJob — état d'un job de synchronisation en arrière-plan, découplé de toute requête HTTP.

Motivation : l'ancien endpoint de sync streamait la progression en SSE. Une fois l'onglet
fermé, le flux mourait — impossible de savoir où en était le job (qui, lui, continuait en
thread daemon). Ici l'état vit dans le process du conteneur webapp et s'interroge par polling
(GET), donc survit et reste consultable après déconnexion / réouverture de page.

Générique : ne connaît rien de RAGIndexer. start() reçoit une fonction run_fn(progress_cb) -> int
(nombre d'items ajoutés). Un seul job à la fois : start() renvoie False si un job tourne déjà —
c'est le garde-fou contre les syncs concurrentes (bug observé : deux syncs simultanées traitant
le même lot)."""
import threading
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SyncJob:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # status : "idle" (jamais lancé) | "running" | "done" | "error". done/error persistent
        # jusqu'au prochain start() réussi → une page rouverte après coup voit le dernier résultat.
        self._state = {
            "status": "idle",
            "current": 0,
            "total": 0,
            "message": "",
            "added": 0,
            "error": None,
            "started_at": None,
            "finished_at": None,
        }

    def start(self, run_fn) -> bool:
        """Démarre run_fn dans un thread daemon si aucun job n'est en cours. Renvoie True si un
        nouveau job a démarré, False si un job tournait déjà (l'appelant se contente alors de
        suivre le job existant via snapshot())."""
        with self._lock:
            if self._state["status"] == "running":
                return False
            self._state = {
                "status": "running",
                "current": 0,
                "total": 0,
                "message": "Démarrage…",
                "added": 0,
                "error": None,
                "started_at": _now(),
                "finished_at": None,
            }
        self._thread = threading.Thread(target=self._run, args=(run_fn,), daemon=True)
        self._thread.start()
        return True

    def _run(self, run_fn) -> None:
        def progress_cb(current, total, message):
            with self._lock:
                self._state["current"] = current
                self._state["total"] = total
                self._state["message"] = message

        try:
            added = run_fn(progress_cb)
            with self._lock:
                self._state["status"] = "done"
                self._state["added"] = added
                self._state["finished_at"] = _now()
        except Exception as e:  # jamais fatal pour le process : l'erreur devient un état lisible
            with self._lock:
                self._state["status"] = "error"
                self._state["error"] = str(e)
                self._state["finished_at"] = _now()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)
