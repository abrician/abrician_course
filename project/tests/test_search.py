"""
Тесты поисковых моделей: TF-IDF и Embeddings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.data.loader import load_and_chunk_documents
from src.models.tfidf_search import TFIDFSearch
from src.models.embedding_search import EmbeddingSearch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Фикстура — загружаем данные один раз для всех тестов
# Фикстура в pytest — это функция которая готовит данные перед тестом
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chunks():
    return load_and_chunk_documents(str(DATA_DIR))


@pytest.fixture(scope="module")
def tfidf_model(chunks):
    model = TFIDFSearch()
    model.fit(chunks)
    return model


@pytest.fixture(scope="module")
def embedding_model(chunks):
    model = EmbeddingSearch(model_name="all-MiniLM-L6-v2")
    model.fit(chunks)
    return model


# ---------------------------------------------------------------------------
# Тесты TF-IDF
# ---------------------------------------------------------------------------

def test_tfidf_fit(chunks):
    """Модель должна обучиться без ошибок."""
    model = TFIDFSearch()
    model.fit(chunks)
    assert model.is_fitted


def test_tfidf_search_returns_list(tfidf_model):
    """Поиск должен вернуть список."""
    results = tfidf_model.search("переобучение", top_k=3)
    assert isinstance(results, list)


def test_tfidf_search_top_k(tfidf_model):
    """Поиск не должен вернуть больше top_k результатов."""
    for k in (1, 3, 5):
        results = tfidf_model.search("python функции", top_k=k)
        assert len(results) <= k


def test_tfidf_result_fields(tfidf_model):
    """Каждый результат должен содержать нужные поля."""
    results = tfidf_model.search("классификация метрики", top_k=3)
    for r in results:
        assert "rank" in r
        assert "score" in r
        assert "text" in r
        assert "filename" in r
        assert "category" in r


def test_tfidf_scores_in_range(tfidf_model):
    """Score должен быть от 0 до 1."""
    results = tfidf_model.search("нейронные сети", top_k=3)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_tfidf_scores_descending(tfidf_model):
    """Результаты должны быть отсортированы по убыванию score."""
    results = tfidf_model.search("регрессия линейная", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_tfidf_finds_relevant(tfidf_model):
    """Поиск по явным ключевым словам должен найти правильный файл в топ-3."""
    results = tfidf_model.search("классификация accuracy precision recall", top_k=3)
    filenames = [r["filename"] for r in results]
    assert "classification.txt" in filenames


def test_tfidf_raises_without_fit():
    """Поиск без обучения должен бросить ошибку."""
    model = TFIDFSearch()
    with pytest.raises(RuntimeError):
        model.search("тест")


# ---------------------------------------------------------------------------
# Тесты Embeddings
# ---------------------------------------------------------------------------

def test_embedding_fit(chunks):
    """Модель должна обучиться без ошибок."""
    model = EmbeddingSearch(model_name="all-MiniLM-L6-v2")
    model.fit(chunks)
    assert model.is_fitted


def test_embedding_search_returns_list(embedding_model):
    """Поиск должен вернуть список."""
    results = embedding_model.search("переобучение", top_k=3)
    assert isinstance(results, list)


def test_embedding_search_top_k(embedding_model):
    """Поиск не должен вернуть больше top_k результатов."""
    for k in (1, 3, 5):
        results = embedding_model.search("python функции", top_k=k)
        assert len(results) <= k


def test_embedding_result_fields(embedding_model):
    """Каждый результат должен содержать нужные поля."""
    results = embedding_model.search("классификация метрики", top_k=3)
    for r in results:
        assert "rank" in r
        assert "score" in r
        assert "text" in r
        assert "filename" in r
        assert "category" in r


def test_embedding_scores_descending(embedding_model):
    """Результаты должны быть отсортированы по убыванию score."""
    results = embedding_model.search("нейронные сети", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_embedding_finds_relevant(embedding_model):
    """Поиск по смыслу должен найти правильный файл."""
    # Запрос перефразирован — нет точных слов из документа
    results = embedding_model.search("модель запоминает данные вместо обобщения", top_k=3)
    filenames = [r["filename"] for r in results]
    assert "overfitting.txt" in filenames


def test_embedding_raises_without_fit():
    """Поиск без обучения должен бросить ошибку."""
    model = EmbeddingSearch()
    with pytest.raises(RuntimeError):
        model.search("тест")
