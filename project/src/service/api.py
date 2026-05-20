"""
FastAPI сервис для RAG-поиска по документации Python и ML.

Эндпоинты:
  GET  /health  — проверка что сервис живой (статус, счётчики, время работы)
  POST /ask     — поиск по запросу пользователя (+ опциональная генерация через LLM)
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_and_chunk_documents
from src.models.tfidf_search import TFIDFSearch
from src.models.embedding_search import EmbeddingSearch
from src.models.llm_generator import generate_answer, is_ollama_available


# ---------------------------------------------------------------------------
# Счётчики для наблюдаемости — хранятся в памяти, сбрасываются при рестарте
# ---------------------------------------------------------------------------

_stats = {
    "requests_total": 0,       # общее число запросов к /ask
    "requests_tfidf": 0,       # запросов через tfidf
    "requests_embeddings": 0,  # запросов через embeddings
    "errors_total": 0,         # число ошибок
    "started_at": time.time(), # время запуска сервиса
}


# ---------------------------------------------------------------------------
# Схемы запроса и ответа
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Запрос пользователя."""
    query: str
    top_k: int = 3
    method: str = "embeddings"
    use_llm: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "query": "что такое переобучение и как с ним бороться",
                "top_k": 3,
                "method": "embeddings",
                "use_llm": False,
            }
        }


class SearchResult(BaseModel):
    """Один найденный фрагмент."""
    rank: int
    score: float
    text: str
    source: str
    filename: str
    category: str
    chunk_id: int


class AskResponse(BaseModel):
    """Ответ сервиса."""
    query: str
    method: str
    results: List[SearchResult]
    total_found: int
    response_time_ms: float          # время обработки запроса в миллисекундах
    llm_answer: Optional[str] = None
    llm_available: bool = False


# ---------------------------------------------------------------------------
# Приложение FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG-помощник по документации Python и ML",
    description=(
        "Сервис для поиска информации по документации Python и материалам по МО. "
        "Поиск работает без LLM. LLM (Ollama) подключается опционально."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Загрузка моделей при старте
# ---------------------------------------------------------------------------

DATA_DIR = ROOT / "data"

print("Загрузка данных...")
chunks = load_and_chunk_documents(str(DATA_DIR))
print(f"Загружено чанков: {len(chunks)}")

print("Инициализация TF-IDF...")
tfidf_model = TFIDFSearch()
tfidf_model.fit(chunks)

print("Инициализация Embeddings...")
embedding_model = EmbeddingSearch(model_name="all-MiniLM-L6-v2")
embedding_model.fit(chunks)

ollama_status = "доступна" if is_ollama_available() else "не запущена (поиск работает без неё)"
print(f"Ollama: {ollama_status}")

print("Сервис готов к работе!")


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Проверка работоспособности сервиса.
    Возвращает статус, счётчики запросов и время работы сервиса.
    """
    uptime_seconds = int(time.time() - _stats["started_at"])
    return {
        "status": "ok",
        "chunks_indexed": len(chunks),
        "methods_available": ["tfidf", "embeddings"],
        "llm_available": is_ollama_available(),
        "metrics": {
            "requests_total": _stats["requests_total"],
            "requests_tfidf": _stats["requests_tfidf"],
            "requests_embeddings": _stats["requests_embeddings"],
            "errors_total": _stats["errors_total"],
            "uptime_seconds": uptime_seconds,
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Поиск по запросу пользователя.

    - **query**: вопрос на русском языке в свободной форме
    - **top_k**: сколько результатов вернуть (1–10, по умолчанию 3)
    - **method**: "tfidf" или "embeddings" (по умолчанию "embeddings")
    - **use_llm**: генерировать ответ через Ollama (если запущена)
    """
    t_start = time.time()

    if not request.query.strip():
        _stats["errors_total"] += 1
        raise HTTPException(status_code=400, detail="Запрос не может быть пустым")

    if request.method not in ("tfidf", "embeddings"):
        _stats["errors_total"] += 1
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный метод: {request.method}. Используйте 'tfidf' или 'embeddings'",
        )

    if not (1 <= request.top_k <= 10):
        _stats["errors_total"] += 1
        raise HTTPException(status_code=400, detail="top_k должно быть от 1 до 10")

    # Обновляем счётчики
    _stats["requests_total"] += 1
    if request.method == "tfidf":
        _stats["requests_tfidf"] += 1
    else:
        _stats["requests_embeddings"] += 1

    # Поиск
    if request.method == "tfidf":
        raw_results = tfidf_model.search(request.query, top_k=request.top_k)
    else:
        raw_results = embedding_model.search(request.query, top_k=request.top_k)

    results = [
        SearchResult(
            rank=r["rank"],
            score=r["score"],
            text=r["text"],
            source=r["source"],
            filename=r["filename"],
            category=r["category"],
            chunk_id=r["chunk_id"],
        )
        for r in raw_results
    ]

    # Опциональная LLM-генерация
    llm_answer = None
    llm_available = is_ollama_available()

    if request.use_llm:
        llm_answer = generate_answer(
            query=request.query,
            context_chunks=[r["text"] for r in raw_results],
        )

    response_time_ms = round((time.time() - t_start) * 1000, 2)

    return AskResponse(
        query=request.query,
        method=request.method,
        results=results,
        total_found=len(results),
        response_time_ms=response_time_ms,
        llm_answer=llm_answer,
        llm_available=llm_available,
    )
