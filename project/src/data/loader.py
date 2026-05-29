"""
Модуль для загрузки, очистки и разбиения текстовых документов на чанки.
"""

import os
import re
from pathlib import Path
from typing import List, Dict


def load_documents(data_dir: str = "data") -> List[Dict]:
    """
    Загружает все .txt файлы из папки data/ рекурсивно.

    Возвращает список словарей, каждый из которых содержит:
    - text: текст документа
    - source: путь к файлу
    - category: ml_docs или python_docs
    """
    data_path = Path(data_dir)
    documents = []

    # rglob("*.txt") — находит все .txt файлы во всех подпапках
    for file_path in sorted(data_path.rglob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        # Определяем категорию по пути к файлу
        parts = file_path.parts
        if "ml_docs" in parts:
            category = "ml_docs"
        elif "python_docs" in parts:
            category = "python_docs"
        else:
            category = "other"

        documents.append({
            "text": text,
            "source": str(file_path),
            "filename": file_path.name,
            "category": category,
        })

    return documents


def clean_text(text: str) -> str:
    """
    Очищает текст: убирает лишние пробелы и пустые строки.
    """
    # Убираем лишние пробелы в начале и конце каждой строки
    lines = [line.strip() for line in text.splitlines()]

    # Убираем подряд идущие пустые строки (оставляем максимум одну)
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if line == "":
            if not prev_empty:
                cleaned_lines.append(line)
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False

    return "\n".join(cleaned_lines).strip()


def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Разбивает текст на чанки (фрагменты) по словам.

    chunk_size — сколько слов в одном чанке (по умолчанию 200)
    overlap    — сколько слов перекрываются между соседними чанками (по умолчанию 50)

    Перекрытие нужно чтобы ответ не "разрезался" на границе двух чанков.
    """
    words = text.split()
    chunks = []

    # Идём по тексту с шагом (chunk_size - overlap)
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        chunk = " ".join(chunk_words)
        if chunk:
            chunks.append(chunk)

    return chunks


def load_and_chunk_documents(data_dir: str = "data",
                              chunk_size: int = 200,
                              overlap: int = 50) -> List[Dict]:
    """
    Главная функция: загружает документы и разбивает их на чанки.

    Возвращает список чанков. Каждый чанк — словарь:
    - text: текст фрагмента
    - source: файл-источник
    - category: ml_docs или python_docs
    - chunk_id: номер чанка внутри документа
    """
    raw_docs = load_documents(data_dir)
    all_chunks = []

    for doc in raw_docs:
        clean = clean_text(doc["text"])
        chunks = split_into_chunks(clean, chunk_size=chunk_size, overlap=overlap)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "filename": doc["filename"],
                "category": doc["category"],
                "chunk_id": i,
            })

    return all_chunks


if __name__ == "__main__":
    # Быстрая проверка: запусти этот файл напрямую → увидишь статистику
    import sys
    # Если запускаем из папки project/ — data/ найдётся автоматически
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    print(f"Загрузка документов из: {data_dir}")
    docs = load_documents(data_dir)
    print(f"Найдено документов: {len(docs)}")
    for doc in docs:
        words = len(doc["text"].split())
        print(f"  [{doc['category']}] {doc['filename']} — {words} слов")

    print()
    chunks = load_and_chunk_documents(data_dir)
    print(f"Всего чанков: {len(chunks)}")
    print()
    print("Пример первого чанка:")
    print("-" * 40)
    print(chunks[0]["text"][:300])
    print("-" * 40)
    print(f"Источник: {chunks[0]['source']}")
