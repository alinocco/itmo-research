# Semantic-space analysis

Descriptive metrics computed against known topic labels (higher silhouette / purity = topics are better separated in the embedding space).

| Method | Silhouette (cosine) | kNN topic purity | #topics |
|--------|---------------------|------------------|---------|
| tfidf | 0.0748 | 0.7329 | 17 |
| e5 | 0.0364 | 0.7136 | 17 |
| bge-m3 | 0.0254 | 0.6839 | 17 |
| doc2vec | -0.0058 | 0.4294 | 17 |
| sbert | -0.0128 | 0.5772 | 17 |
| bert | -0.0409 | 0.5596 | 17 |
| gte | -0.046 | 0.3423 | 17 |
| fasttext | -0.1071 | 0.6519 | 17 |
