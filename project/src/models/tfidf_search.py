"""
TF-IDF поиск по корпусу документов.

Принцип работы:
1. Все чанки преобразуются в TF-IDF матрицу (каждый чанк — вектор чисел).
2. Запрос пользователя тоже преобразуется в вектор.
3. Через cosine similarity находим чанки, ближайшие к запросу.
4. Возвращаем топ-K результатов.
"""

import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSearch:
    def __init__(self):
        # TfidfVectorizer — умеет превращать список текстов в матрицу TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=10000,   # использовать максимум 10000 уникальных слов
            ngram_range=(1, 2),   # учитывать одиночные слова И пары слов ("random forest")
            min_df=1,             # слово должно встречаться хотя бы в 1 документе
            sublinear_tf=True,    # сглаживание: log(tf) вместо tf (уменьшает влияние очень частых слов)
        )
        self.chunks = []          # список всех чанков
        self.tfidf_matrix = None  # матрица TF-IDF (строки = чанки, столбцы = слова)
        self.is_fitted = False    # True если модель обучена

    def fit(self, chunks: List[Dict]) -> None:
        """
        Обучает модель на списке чанков.
        chunks — список словарей из load_and_chunk_documents().
        """
        self.chunks = chunks

        # Берём только текст из каждого чанка
        texts = [chunk["text"] for chunk in chunks]

        # fit_transform: сначала учится (fit), потом преобразует (transform)
        # Результат — разреженная матрица размером (кол-во чанков × кол-во слов)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

        print(f"TF-IDF модель обучена.")
        print(f"  Чанков в индексе: {len(chunks)}")
        print(f"  Размер словаря: {len(self.vectorizer.vocabulary_)} слов")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Ищет top_k наиболее релевантных чанков для запроса query.

        Возвращает список словарей с ключами:
        - text, source, filename, category, chunk_id (из исходного чанка)
        - score: оценка схожести от 0 до 1
        - rank: позиция в результатах (1 = лучший)
        """
        if not self.is_fitted:
            raise RuntimeError("Сначала вызови метод fit()!")

        # Преобразуем запрос в вектор TF-IDF (используем тот же векторизатор)
        query_vector = self.vectorizer.transform([query])

        # Считаем cosine similarity между запросом и каждым чанком
        # Результат — массив из N чисел (по одному на каждый чанк)
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Сортируем индексы по убыванию схожести, берём top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            if score == 0.0:
                # Если схожесть 0 — запрос не нашёл ничего общего с чанком
                continue
            result = dict(self.chunks[idx])  # копируем чанк
            result["score"] = round(score, 4)
            result["rank"] = rank
            results.append(result)

        return results

    def save(self, path: str) -> None:
        """Сохраняет обученную модель на диск."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer,
                         "chunks": self.chunks,
                         "tfidf_matrix": self.tfidf_matrix}, f)
        print(f"Модель сохранена: {path}")

    def load(self, path: str) -> None:
        """Загружает обученную модель с диска."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.chunks = data["chunks"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.is_fitted = True
        print(f"Модель загружена: {path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.loader import load_and_chunk_documents

    # Загружаем данные и обучаем модель
    chunks = load_and_chunk_documents("data")
    model = TFIDFSearch()
    model.fit(chunks)

    # Тестируем несколько запросов
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
