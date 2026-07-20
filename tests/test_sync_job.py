import threading
import time

import pytest

from webapp.sync_job import SyncJob


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture
def job():
    return SyncJob()


def test_initial_status_is_idle(job):
    snap = job.snapshot()
    assert snap["status"] == "idle"
    assert snap["added"] == 0
    assert snap["error"] is None


def test_start_runs_to_completion(job):
    def run_fn(progress_cb):
        progress_cb(1, 2, "en cours")
        return 3

    started = job.start(run_fn)
    assert started is True
    assert _wait_until(lambda: job.snapshot()["status"] == "done")

    snap = job.snapshot()
    assert snap["status"] == "done"
    assert snap["added"] == 3
    assert snap["error"] is None
    assert snap["finished_at"] is not None


def test_progress_callback_updates_snapshot(job):
    release = threading.Event()

    def run_fn(progress_cb):
        progress_cb(2, 5, "[2/5] Un article")
        release.wait(timeout=2)
        return 5

    job.start(run_fn)
    assert _wait_until(lambda: job.snapshot()["current"] == 2)

    snap = job.snapshot()
    assert snap["status"] == "running"
    assert snap["current"] == 2
    assert snap["total"] == 5
    assert snap["message"] == "[2/5] Un article"

    release.set()
    assert _wait_until(lambda: job.snapshot()["status"] == "done")


def test_exception_in_run_fn_becomes_error_state(job):
    def run_fn(progress_cb):
        raise RuntimeError("Zotero API injoignable")

    job.start(run_fn)
    assert _wait_until(lambda: job.snapshot()["status"] == "error")

    snap = job.snapshot()
    assert snap["status"] == "error"
    assert "Zotero API injoignable" in snap["error"]


def test_second_start_while_running_is_rejected(job):
    """Reproduit le bug observé en production : deux appels à /api/index/sync rapprochés
    traitaient le même lot d'articles en parallèle. Un job déjà en cours doit bloquer tout
    nouveau démarrage — l'appelant doit se contenter de suivre le job existant."""
    release = threading.Event()
    second_run_fn_called = threading.Event()

    def slow_run_fn(progress_cb):
        release.wait(timeout=2)
        return 1

    def second_run_fn(progress_cb):
        second_run_fn_called.set()
        return 99

    first_started = job.start(slow_run_fn)
    assert first_started is True
    assert _wait_until(lambda: job.snapshot()["status"] == "running")

    second_started = job.start(second_run_fn)
    assert second_started is False
    assert not second_run_fn_called.is_set()

    release.set()
    assert _wait_until(lambda: job.snapshot()["status"] == "done")
    assert job.snapshot()["added"] == 1  # bien le résultat du 1er job, pas du 2e (jamais lancé)


def test_status_persists_after_completion_until_next_start(job):
    """Un job terminé (done/error) reste visible tel quel — une page rouverte plus tard doit
    voir le dernier résultat, pas un état réinitialisé silencieusement."""
    def run_fn(progress_cb):
        return 7

    job.start(run_fn)
    assert _wait_until(lambda: job.snapshot()["status"] == "done")

    time.sleep(0.05)
    snap = job.snapshot()
    assert snap["status"] == "done"
    assert snap["added"] == 7


def test_new_start_after_completion_resets_state(job):
    def first(progress_cb):
        return 1

    job.start(first)
    assert _wait_until(lambda: job.snapshot()["status"] == "done")

    release = threading.Event()

    def second(progress_cb):
        release.wait(timeout=2)
        return 2

    started = job.start(second)
    assert started is True
    snap = job.snapshot()
    assert snap["status"] == "running"
    assert snap["error"] is None

    release.set()
    assert _wait_until(lambda: job.snapshot()["status"] == "done")
    assert job.snapshot()["added"] == 2
