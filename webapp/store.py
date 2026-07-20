"""ConversationStore — persistance SQLite des conversations (data/conversations.db, sur le
volume /app/data : survit aux redémarrages du conteneur). Connexions courtes par appel plutôt
qu'une connexion partagée : sqlite3 interdit nativement le partage d'une connexion entre threads,
et Starlette exécute les endpoints synchrones dans un threadpool — le coût d'ouverture est
négligeable à ce volume (usage mono-utilisateur, quelques requêtes/minute)."""
# Nécessaire : la méthode list() ci-dessous masque le builtin `list` pour le reste du corps de
# la classe — sans ceci, l'annotation -> list[dict] d'une méthode suivante (recent_history)
# tenterait de souscrire la méthode list elle-même (TypeError à l'import).
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(self, title: str = "Nouvelle conversation") -> int:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            return cur.lastrowid

    def list(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get(self, conversation_id: int) -> dict | None:
        with self._connect() as conn:
            conv = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not conv:
                return None
            msgs = conn.execute(
                "SELECT role, content, meta_json FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        return {
            **dict(conv),
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "meta": json.loads(m["meta_json"]) if m["meta_json"] else None,
                }
                for m in msgs
            ],
        }

    def add_message(
        self, conversation_id: int, role: str, content: str, meta: dict | None = None
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, meta_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(meta) if meta else None, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )

    def recent_history(self, conversation_id: int, n: int = 3) -> list[dict]:
        """Derniers n tours {question, answer} — format attendu par RAGAgent.run_stream
        (history=...). Suppose une alternance user/assistant ; un message user final sans
        réponse (ex: requête en cours) n'est pas compté comme un tour complet."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        turns = []
        pending_question = None
        for r in rows:
            if r["role"] == "user":
                pending_question = r["content"]
            elif r["role"] == "assistant" and pending_question is not None:
                turns.append({"question": pending_question, "answer": r["content"]})
                pending_question = None
        return turns[-n:]

    def rename(self, conversation_id: int, title: str) -> None:
        # updated_at délibérément inchangé : renommer n'est pas une activité de conversation,
        # ça ne doit pas faire remonter artificiellement l'entrée en tête de liste.
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
            )

    def delete(self, conversation_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
