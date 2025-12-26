# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.
```md
Дополнительные параметры команды `report`:

- `--max-hist-columns` – максимальное число числовых колонок,
  для которых строятся гистограммы;
- `--title` – заголовок отчёта (используется как `# ...` в `report.md`).

Пример запуска с дополнительными параметрами:

```bash
uv run eda-cli report data/example.csv \
  --out-dir reports_example \
  --max-hist-columns 4 \
  --title "EDA по учебному датасету"

## Тесты

```bash
uv run pytest -q
```
# EDA CLI + HTTP Dataset Quality Service

Проект оценивает качество датасетов для обучения моделей. 
Содержит:
- CLI для анализа CSV (EDA, отчёты, визуализации)
- HTTP API для удалённой оценки качества и получения флагов

## HTTP API

### GET /health
Простейший health-check сервиса.

### POST /quality
Принимает агрегированные признаки и возвращает оценку качества.

### POST /quality-from-csv
Принимает CSV-файл и возвращает оценку качества с использованием EDA-ядра.

### POST /quality-flags-from-csv

Новый дополнительный эндпоинт HW04.

**Путь:** `POST /quality-flags-from-csv`

Принимает CSV-файл и возвращает полный набор флагов качества,
включая эвристики, реализованные в HW03:
- too_few_rows
- too_many_missing
- has_constant_columns
- has_high_missing_columns
- penalties
- quality_score


## Запуск

```bash
cd homeworks/HW04/eda-cli
uv sync
uv run pytest -q
uv run eda-cli report data/example.csv --out-dir reports_example

## Запуск HTTP-сервиса

Для запуска HTTP API используется FastAPI + Uvicorn:

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000

