import threading

import pytest
from pydantic import BaseModel

from pipeline_builder.core.state import ArtifactStore, State


class Item(BaseModel):
    value: str


def test_session_id_auto_generated():
    s = State()
    assert s.session_id
    assert len(s.session_id) == 8


def test_session_id_custom():
    s = State(session_id="abc123")
    assert s.session_id == "abc123"


def test_set_and_get_nodes():
    s = State()
    items = [Item(value="a"), Item(value="b")]
    s.set_nodes("item", items)
    assert s.get_nodes("item") == items


def test_get_nodes_missing_level_returns_empty():
    s = State()
    assert s.get_nodes("nonexistent") == []


def test_extend_nodes_appends():
    s = State()
    s.set_nodes("item", [Item(value="a")])
    s.extend_nodes("item", [Item(value="b")])
    assert len(s.get_nodes("item")) == 2


def test_extend_nodes_creates_level_if_absent():
    s = State()
    s.extend_nodes("item", [Item(value="x")])
    assert len(s.get_nodes("item")) == 1


def test_get_nodes_returns_copy():
    s = State()
    s.set_nodes("item", [Item(value="a")])
    copy = s.get_nodes("item")
    copy.append(Item(value="b"))
    assert len(s.get_nodes("item")) == 1  # original unchanged


def test_artifact_store_set_and_get():
    store = ArtifactStore()
    store.set("key", 42)
    assert store.get("key") == 42


def test_artifact_store_default():
    store = ArtifactStore()
    assert store.get("missing", "default") == "default"


def test_artifact_store_has():
    store = ArtifactStore()
    store.set("x", 1)
    assert store.has("x")
    assert not store.has("y")


def test_state_thread_safety():
    """Concurrent writes to extend_nodes must not lose data."""
    s = State()
    errors = []

    def worker(i: int) -> None:
        try:
            s.extend_nodes("item", [Item(value=str(i))])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(s.get_nodes("item")) == 20
