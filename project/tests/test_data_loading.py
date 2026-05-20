"""
Тесты загрузки и обработки данных.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь поиска
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_documents, clean_text, split_into_chunks, load_and_chunk_documents

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_load_documents_returns_list():
    """Функция должна вернуть список."""
    docs = load_documents(str(DATA_DIR))
    assert isinstance(docs, list)


def test_load_documents_not_empty():
    """Должен загрузить хотя бы один документ."""
    docs = load_documents(str(DATA_DIR))
    assert len(docs) > 0


def test_load_documents_count():
    """Должно быть ровно 14 документов."""
    docs = load_documents(str(DATA_DIR))
    assert len(docs) == 14


def test_document_has_required_fields():
    """Каждый документ должен содержать нужные поля."""
    docs = load_documents(str(DATA_DIR))
    for doc in docs:
        assert "text" in doc
        assert "source" in doc
        assert "filename" in doc
        assert "category" in doc


def test_document_categories():
    """Категории должны быть только ml_docs или python_docs."""
    docs = load_documents(str(DATA_DIR))
    for doc in docs:
        assert doc["category"] in ("ml_docs", "python_docs")


def test_document_text_not_empty():
    """Текст каждого документа не должен быть пустым."""
    docs = load_documents(str(DATA_DIR))
    for doc in docs:
        assert len(doc["text"].strip()) > 0


def test_clean_text_strips_whitespace():
    """clean_text должна убирать лишние пробелы."""
    dirty = "  привет  \n\n\n  мир  \n"
    result = clean_text(dirty)
    assert not result.startswith(" ")
    assert not result.endswith(" ")
    assert "\n\n\n" not in result


def test_split_into_chunks_returns_list():
    """split_into_chunks должна вернуть список."""
    text = " ".join(["слово"] * 300)
    chunks = split_into_chunks(text, chunk_size=100, overlap=20)
    assert isinstance(chunks, list)


def test_split_into_chunks_not_empty():
    """Чанки не должны быть пустыми."""
    text = " ".join(["слово"] * 300)
    chunks = split_into_chunks(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_split_respects_chunk_size():
    """Каждый чанк не должен превышать chunk_size слов."""
    text = " ".join(["слово"] * 500)
    chunk_size = 100
    chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=20)
    for chunk in chunks:
        assert len(chunk.split()) <= chunk_size


def test_load_and_chunk_documents():
    """Главная функция должна вернуть непустой список чанков с нужными полями."""
    chunks = load_and_chunk_documents(str(DATA_DIR))
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "source" in chunk
        assert "filename" in chunk
        assert "category" in chunk
        assert "chunk_id" in chunk
