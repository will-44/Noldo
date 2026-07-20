import pytest

from webapp.store import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(tmp_path / "conversations.db")


def test_create_returns_id_and_appears_in_list(store):
    conv_id = store.create("Ma conversation")
    assert isinstance(conv_id, int)

    convs = store.list()
    assert len(convs) == 1
    assert convs[0]["id"] == conv_id
    assert convs[0]["title"] == "Ma conversation"


def test_create_default_title(store):
    conv_id = store.create()
    conv = store.get(conv_id)
    assert conv["title"] == "Nouvelle conversation"


def test_get_returns_none_for_unknown_id(store):
    assert store.get(999) is None


def test_get_includes_messages_in_insertion_order(store):
    conv_id = store.create()
    store.add_message(conv_id, "user", "Question 1")
    store.add_message(conv_id, "assistant", "Réponse 1")
    store.add_message(conv_id, "user", "Question 2")

    conv = store.get(conv_id)
    roles = [m["role"] for m in conv["messages"]]
    contents = [m["content"] for m in conv["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert contents == ["Question 1", "Réponse 1", "Question 2"]


def test_add_message_stores_and_returns_meta(store):
    conv_id = store.create()
    meta = {"sources": [{"item_key": "ABC", "page": 3}], "external_sources": []}
    store.add_message(conv_id, "assistant", "Réponse.", meta=meta)

    conv = store.get(conv_id)
    assert conv["messages"][0]["meta"] == meta


def test_add_message_without_meta_returns_none(store):
    conv_id = store.create()
    store.add_message(conv_id, "user", "Question sans meta")

    conv = store.get(conv_id)
    assert conv["messages"][0]["meta"] is None


def test_add_message_bumps_updated_at(store):
    conv_id = store.create()
    before = store.get(conv_id)["updated_at"]
    store.add_message(conv_id, "user", "Une question")
    after = store.get(conv_id)["updated_at"]
    assert after >= before


def test_recent_history_pairs_user_assistant_turns(store):
    conv_id = store.create()
    store.add_message(conv_id, "user", "Q1")
    store.add_message(conv_id, "assistant", "R1")
    store.add_message(conv_id, "user", "Q2")
    store.add_message(conv_id, "assistant", "R2")

    history = store.recent_history(conv_id)
    assert history == [
        {"question": "Q1", "answer": "R1"},
        {"question": "Q2", "answer": "R2"},
    ]


def test_recent_history_ignores_trailing_unanswered_question(store):
    """Un message user sans réponse assistant (requête en cours ou erreur) ne doit pas
    apparaître comme un tour complet dans l'historique renvoyé à l'agent."""
    conv_id = store.create()
    store.add_message(conv_id, "user", "Q1")
    store.add_message(conv_id, "assistant", "R1")
    store.add_message(conv_id, "user", "Q2 sans réponse")

    history = store.recent_history(conv_id)
    assert history == [{"question": "Q1", "answer": "R1"}]


def test_recent_history_caps_to_n(store):
    conv_id = store.create()
    for i in range(5):
        store.add_message(conv_id, "user", f"Q{i}")
        store.add_message(conv_id, "assistant", f"R{i}")

    history = store.recent_history(conv_id, n=2)
    assert history == [
        {"question": "Q3", "answer": "R3"},
        {"question": "Q4", "answer": "R4"},
    ]


def test_recent_history_empty_conversation(store):
    conv_id = store.create()
    assert store.recent_history(conv_id) == []


def test_rename_updates_title_without_touching_updated_at(store):
    conv_id = store.create("Ancien titre")
    before = store.get(conv_id)["updated_at"]

    store.rename(conv_id, "Nouveau titre")

    conv = store.get(conv_id)
    assert conv["title"] == "Nouveau titre"
    assert conv["updated_at"] == before


def test_delete_removes_conversation(store):
    conv_id = store.create()
    store.delete(conv_id)
    assert store.get(conv_id) is None


def test_delete_cascades_messages(store):
    """Après suppression, les messages orphelins ne doivent pas rester en base (vérifié
    indirectement : recréer une conversation avec le même schéma ne doit pas planter et
    la liste ne doit plus référencer l'ancienne conversation)."""
    conv_id = store.create()
    store.add_message(conv_id, "user", "Question")
    store.delete(conv_id)

    assert store.list() == []


def test_list_orders_by_updated_at_descending(store):
    first = store.create("Première")
    second = store.create("Deuxième")
    # Réactive "first" : doit repasser devant "second" dans le tri par updated_at.
    store.add_message(first, "user", "nouvelle activité")

    convs = store.list()
    assert convs[0]["id"] == first
    assert convs[1]["id"] == second


def test_multiple_conversations_are_independent(store):
    conv1 = store.create("Conv 1")
    conv2 = store.create("Conv 2")
    store.add_message(conv1, "user", "Question conv1")
    store.add_message(conv2, "user", "Question conv2")

    assert len(store.get(conv1)["messages"]) == 1
    assert len(store.get(conv2)["messages"]) == 1
    assert store.get(conv1)["messages"][0]["content"] == "Question conv1"
