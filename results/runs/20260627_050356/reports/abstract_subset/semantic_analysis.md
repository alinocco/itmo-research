# Semantic-space analysis

Descriptive metrics computed against known topic labels (higher silhouette / purity = topics are better separated in the embedding space).

| Method | Silhouette (cosine) | kNN topic purity | #topics |
|--------|---------------------|------------------|---------|
| doc2vec | 0.2599 | 0.652 | 6 |
| sbert | 0.1807 | 0.755 | 6 |
| bert | 0.1795 | 0.6907 | 6 |
| fasttext | 0.1644 | 0.478 | 6 |
| gte | 0.1251 | 0.755 | 6 |
| e5 | 0.1244 | 0.7253 | 6 |
| bge-m3 | 0.0756 | 0.6653 | 6 |
| tfidf | 0.0278 | 0.6317 | 6 |
