"""
Опциональная генерация ответа через локальную LLM (Ollama).

Ollama — это локальный инструмент для запуска LLM без интернета и API-ключей.
Сайт: https://ollama.com

Установка Ollama (один раз):
  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
  Windows: скачать установщик с ollama.com

Скачать модель (один раз после установки Ollama):
  ollama pull qwen2.5:1.5b

Запустить Ollama перед использованием:
  ollama serve

Если Ollama не запущена — этот модуль возвращает None,
сервис продолжает работать в режиме только поиска.
"""

import requests
from typing import Optional, List


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:1.5b"


def is_ollama_available(url: str = OLLAMA_URL) -> bool:
    """Проверяет доступность Ollama."""
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def generate_answer(
    query: str,
    context_chunks: List[str],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
) -> Optional[str]:
    """
    Генерирует ответ на основе найденных фрагментов.

    Возвращает строку с ответом или None если Ollama недоступна.

    query          — вопрос пользователя
    context_chunks — список найденных текстовых фрагментов
    model          — название модели в Ollama
    max_tokens     — максимальная длина ответа
    """
    if not is_ollama_available():
        return None

    # Собираем контекст из найденных фрагментов
    context = "\n\n---\n\n".join(context_chunks)

    # Промпт: объясняем модели её задачу
    prompt = (
        "Ты — помощник по документации Python и машинному обучению. "
        "Используй только информацию из предоставленных фрагментов. "
        "Отвечай кратко и по существу на русском языке.\n\n"
        f"Фрагменты документации:\n{context}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,          # получаем весь ответ сразу
                "options": {
                    "num_predict": max_tokens,  # максимум токенов в ответе
                    "temperature": 0.3,          # низкая температура = более точный ответ
                },
            },
            timeout=60,  # ждём ответа максимум 60 секунд
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except Exception as e:
        # Не ломаем сервис — просто возвращаем None
        print(f"LLM недоступна: {e}")
        return None


if __name__ == "__main__":
    print(f"Ollama доступна: {is_ollama_available()}")

    if is_ollama_available():
        answer = generate_answer(
            query="Что такое переобучение?",
            context_chunks=[
                "Переобучение — ситуация, когда модель слишком хорошо "
                "подстраивается под обучающие данные, но плохо работает на новых.",
                "Методы борьбы: регуляризация L1/L2, dropout, ранняя остановка.",
            ],
        )
        print(f"Ответ LLM:\n{answer}")
    else:
        print("Ollama не запущена. Запустите: ollama serve")
