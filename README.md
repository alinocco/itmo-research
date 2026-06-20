# textvec — пайплайн векторизации текстов и анализа семантического пространства

Исследовательский прототип для НИР (Семестр II): подготовка корпуса научных статей,
конвейер предобработки, построение эмбеддингов разными методами и анализ семантического
пространства текстов.

Конвейер: **сбор корпуса → предобработка → векторизация → анализ семантического пространства**
(снижение размерности PCA / UMAP + визуализация).

> Корпус — англоязычные научные статьи из открытых источников **ArXiv** и **PubMed**.
> Лемматизация реализована для английского языка (spaCy `en_core_web_sm`); язык
> вынесен в конфиг (`project.language`) и легко переключается на русский.

### Варианты эксперимента (abstract vs full_text)

Что считать «текстом документа» задаётся через `experiment.variants` в конфиге.
Каждый вариант имеет свой исходный CSV, свой набор полей и **изолированные папки
выходов**, поэтому результаты не перетирают друг друга:

| Вариант | Источник | Поля текста | Назначение |
|---|---|---|---|
| `abstract` | весь корпус (`corpus.csv`) | title + abstract | основной прогон |
| `full_text` | подвыборка (`corpus_fulltext.csv`) | title + full_text | полнотекстовый прогон |
| `abstract_subset` | та же подвыборка | title + abstract | честное сравнение на тех же документах |

Полный текст недоступен через метаданные-API: для ArXiv качается PDF (парсинг `pypdf`),
для PubMed — только подмножество **PMC Open Access** (свежие статьи часто под эмбарго,
поэтому полнотекстовое покрытие PubMed низкое — это ограничение источника, не пайплайна).

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
python -m textvec collect                      # 1. корпус из ArXiv/PubMed -> data/processed/corpus.csv|json
python -m textvec stats                        # 2. первичный анализ корпуса -> results/reports/corpus_stats.json
python -m textvec fulltext --per-topic 50      # (опц.) собрать полнотекстовую подвыборку -> corpus_fulltext.csv
python -m textvec preprocess --variant abstract
python -m textvec vectorize  --variant abstract
python -m textvec analyze    --variant abstract

# Полнотекстовый прогон и честное сравнение на той же подвыборке:
python -m textvec preprocess --variant full_text       && python -m textvec vectorize --variant full_text && python -m textvec analyze --variant full_text
python -m textvec preprocess --variant abstract_subset && python -m textvec vectorize --variant abstract_subset && python -m textvec analyze --variant abstract_subset

# Весь конвейер для варианта сразу (подмножество методов):
python -m textvec all --variant abstract --methods tfidf doc2vec sbert
```

Выходы каждого варианта лежат в `results/embeddings/<variant>/`, `results/figures/<variant>/`,
`results/reports/<variant>/`, а очищенный корпус — в `data/processed/clean_<variant>.csv`.

Выбор методов векторизации:

```bash
python -m textvec vectorize --methods tfidf fasttext doc2vec   # классические (без сети)
python -m textvec vectorize --methods sbert e5 bge-m3 gte bert # нейросетевые (скачиваются модели)
```

## Методы векторизации

| Группа        | Метод     | Реализация |
|---------------|-----------|------------|
| Разрежённые   | TF-IDF (+ опц. TruncatedSVD/LSA) | scikit-learn |
| Плотные (классич.) | FastText, Doc2Vec | gensim |
| Нейросетевые  | BERT, Sentence-BERT, BGE-M3, E5, GTE | sentence-transformers |

Модели нейросетевых методов задаются в `config/default.yaml`
(`vectorization.transformers.models`) и заменяются без изменения кода.

Масштабируемость:
- **TF-IDF** не материализует плотную матрицу — при `svd_components` идёт через
  `TruncatedSVD` (LSA), что позволяет работать с десятками тысяч документов.
- **Длинные документы** (full_text > 512 токенов): при `transformers.chunk_long_docs: true`
  документ режется на окна по `chunk_size_words` слов, эмбеддинги окон усредняются (mean-pooling).
- **Silhouette** на больших корпусах считается по случайной выборке (`sample_size`).

## Выходные артефакты

- `data/processed/corpus.csv` / `corpus.json` — собранный корпус с метаданными
  (название, источник, дата публикации, тематика).
- `data/processed/corpus_fulltext.csv` — полнотекстовая подвыборка (если собрана).
- `data/processed/clean_<variant>.csv` — очищенный/лемматизированный корпус варианта.
- `results/embeddings/<variant>/<method>.npy` + `manifest.json` — векторы и параметры эксперимента.
- `results/figures/<variant>/<method>_{pca,umap}.png` — проекции семантического пространства.
- `results/reports/<variant>/semantic_analysis.md|json`, `results/reports/corpus_stats.json` — метрики и статистика.

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
