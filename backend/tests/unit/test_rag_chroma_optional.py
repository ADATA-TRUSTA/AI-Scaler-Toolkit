"""
chromadb is the opt-in `chroma` extra, so a host can ask for Chroma without having it.

What must hold then: the service still starts, RAG still answers from SQLite, and the log
says why Chroma is not in use. Before the extra existed every install had chromadb, and a
failed import was swallowed silently -- harmless then, a confusing dead setting now.
"""

import logging
import sys
from pathlib import Path

import pytest

from service.rag_manager import RagManager

pytestmark = pytest.mark.inference


@pytest.mark.parametrize(
    "env", [{"RAG_BACKEND": "chroma"}, {"RAG_USE_CHROMA": "true"}], ids=["backend", "flag"]
)
def test_chroma_requested_without_the_extra_warns_and_stays_on_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env: dict[str, str],
) -> None:
    """Either switch, with chromadb absent: a warning naming the extra, and SQLite still works."""
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.delenv("RAG_USE_CHROMA", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    # A None entry makes `import chromadb` raise ImportError, as on a host without the extra.
    monkeypatch.setitem(sys.modules, "chromadb", None)

    with caplog.at_level(logging.WARNING, logger="service.rag_manager"):
        rag = RagManager(base_dir=str(tmp_path))

    assert rag._use_chroma is False
    assert any("chroma" in r.getMessage() and "extra" in r.getMessage() for r in caplog.records)

    rag.add_document(content="the quick brown fox", doc_id="fox")
    assert [hit["doc_id"] for hit in rag.search("fox", k=1)] == ["fox"]


def test_chroma_not_requested_logs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The default path never mentions Chroma, installed or not."""
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.delenv("RAG_USE_CHROMA", raising=False)
    monkeypatch.setitem(sys.modules, "chromadb", None)

    with caplog.at_level(logging.WARNING, logger="service.rag_manager"):
        RagManager(base_dir=str(tmp_path))

    assert not caplog.records
