# textvec — пайплайн векторизации текстов и анализа семантического пространства

Исследовательский прототип для НИР (Семестр II): подготовка корпуса научных статей,
конвейер предобработки, построение эмбеддингов разными методами и анализ семантического
пространства текстов.

Конвейер: **сбор корпуса → предобработка → векторизация → анализ семантического пространства**
(снижение размерности PCA / UMAP + визуализация).

> Корпус — англоязычные научные статьи из открытых источников **ArXiv** и **PubMed**.
> Лемматизация реализована для английского языка (spaCy `en_core_web_sm`); язык
> вынесен в конфиг (`project.language`) и легко переключается на русский.

## Структура проекта

```
Project/
├── config/default.yaml          # все параметры эксперимента (источники, предобработка, модели)
├── src/textvec/
│   ├── corpus/                   # сбор корпуса: ArXiv, PubMed, единая схема, сохранение csv/json
│   ├── preprocessing/            # очистка, нормализация, токенизация, лемматизация, стоп-слова
│   ├── vectorization/            # TF-IDF, FastText, Doc2Vec, BERT, SBERT, BGE-M3, E5, GTE
│   ├── analysis/                 # статистика корпуса, PCA/UMAP, визуализация, метрики
│   ├── cli.py                    # единый интерфейс запуска этапов
│   └── config.py / utils.py
├── data/                         # raw / interim / processed (не коммитится)
├── results/                      # embeddings / figures / reports (не коммитится)
└── tests/                        # unit- и smoke-тесты
```

## Установка

```bash
cd Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Языковые ресурсы:
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```

Требуется Python ≥ 3.10. Тяжёлые модели (BERT, BGE-M3, E5, GTE) запускаются на CPU,
а при наличии GPU/MPS используют его автоматически (`project.device: auto`).

## Запуск конвейера

Каждый этап запускается отдельно или целиком:

```bash
python -m textvec collect       # 1. собрать корпус из ArXiv/PubMed -> data/processed/corpus.csv|json
python -m textvec stats         # 2. первичный анализ корпуса -> results/reports/corpus_stats.json
python -m textvec preprocess    # 3. предобработка -> data/processed/corpus_clean.csv
python -m textvec vectorize     # 4. эмбеддинги -> results/embeddings/*.npy (+ manifest.json)
python -m textvec analyze       # 5. PCA/UMAP + метрики -> results/figures, results/reports

# Весь конвейер сразу (подмножество методов):
python -m textvec all --methods tfidf doc2vec sbert
```

Выбор методов векторизации:

```bash
python -m textvec vectorize --methods tfidf fasttext doc2vec   # классические (без сети)
python -m textvec vectorize --methods sbert e5 bge-m3 gte bert # нейросетевые (скачиваются модели)
```

## Методы векторизации

| Группа        | Метод     | Реализация |
|---------------|-----------|------------|
| Разрежённые   | TF-IDF    | scikit-learn |
| Плотные (классич.) | FastText, Doc2Vec | gensim |
| Нейросетевые  | BERT, Sentence-BERT, BGE-M3, E5, GTE | sentence-transformers |

Модели нейросетевых методов задаются в `config/default.yaml`
(`vectorization.transformers.models`) и заменяются без изменения кода.

## Выходные артефакты

- `data/processed/corpus.csv` / `corpus.json` — собранный корпус с метаданными
  (название, источник, дата публикации, тематика).
- `data/processed/corpus_clean.csv` — очищенный/лемматизированный корпус.
- `results/embeddings/<method>.npy` + `manifest.json` — векторные представления и параметры эксперимента.
- `results/figures/<method>_{pca,umap}.png` — проекции семантического пространства.
- `results/reports/corpus_stats.json`, `semantic_analysis.md|json` — статистика и метрики разделимости тем.

## Воспроизводимость

Все запуски детерминированы через `project.seed` (random / numpy / torch / gensim / UMAP).
Параметры каждого эксперимента сохраняются в `manifest.json` и `*.meta.json`.

## Тесты

```bash
pytest tests/ -q
```

## Расширение

- Новый источник корпуса: добавить загрузчик в `src/textvec/corpus/`, вернуть `list[Document]`.
- Новый метод векторизации: наследовать `BaseVectorizer`, зарегистрировать в `vectorization/registry.py`.
- Новый метод снижения размерности: добавить в `analysis/reduce.py`.
