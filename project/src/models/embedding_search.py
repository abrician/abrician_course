"""
Поиск на основе sentence-transformers (семантический поиск).

Принцип работы:
1. Каждый чанк преобразуется в вектор-эмбеддинг (384 числа).
2. Запрос пользователя тоже преобразуется в вектор.
3. Через cosine similarity находим ближайшие векторы.
4. Возвращаем топ-K результатов.

Отличие от TF-IDF: модель понимает смысл, а не только слова.
"""

import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Загружаем предобученную модель.
        # При первом запуске скачается с интернета (~90 МБ).
        print(f"Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.chunks = []
        self.embeddings = None  # матрица эмбеддингов: (кол-во чанков × 384)
        self.is_fitted = False

    def fit(self, chunks: List[Dict]) -> None:
        """
        Создаёт эмбеддинги для всех чанков и сохраняет их.
        Это занимает несколько секунд — модель обрабатывает каждый чанк.
        """
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        print(f"Создание эмбеддингов для {len(texts)} чанков...")
        # encode() — преобразует список текстов в матрицу эмбеддингов
        # show_progress_bar=True — показывает прогресс
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.is_fitted = True

        print(f"Готово! Размер матрицы эмбеддингов: {self.embeddings.shape}")
        # Например: (29, 384) — 29 чанков, каждый вектор из 384 чисел

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Ищет top_k наиболее релевантных чанков для запроса query.
        """
        if not self.is_fitted:
            raise RuntimeError("Сначала вызови метод fit()!")

        # Создаём эмбеддинг для запроса
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Считаем cosine similarity между запросом и всеми чанками
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Берём топ-K индексов с наибольшей схожестью
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            result = dict(self.chunks[idx])
            result["score"] = round(score, 4)
            result["rank"] = rank
            results.append(result)

        return results

    def save(self, path: str) -> None:
        """Сохраняет эмбеддинги и чанки на диск (модель не сохраняем — она скачивается)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "embeddings": self.embeddings,
                "model_name": self.model_name,
            }, f)
        print(f"Эмбеддинги сохранены: {path}")

    def load(self, path: str) -> None:
        """Загружает эмбеддинги и чанки с диска."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        self.model_name = data["model_name"]
        self.is_fitted = True
        print(f"Эмбеддинги загружены: {path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.loader import load_and_chunk_documents

    chunks = load_and_chunk_documents("data")

    model = EmbeddingSearch()
    model.fit(chunks)

    test_queries = [
        "как бороться с переобучением",
        "что такое функции в python",
        "классификация алгоритмы метрики",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Запрос: {query}")
        print(f"{'='*50}")
        results = model.search(query, top_k=3)
        for r in results:
            print(f"  #{r['rank']} score={r['score']:.4f} [{r['category']}] {r['filename']}")
            print(f"     {r['text'][:120]}...")
