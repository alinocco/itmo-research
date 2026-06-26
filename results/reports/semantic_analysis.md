# Semantic-space analysis

Descriptive metrics computed against known topic labels (higher silhouette / purity = topics are better separated in the embedding space).

| Method | Silhouette (cosine) | kNN topic purity | #topics |
|--------|---------------------|------------------|---------|
| doc2vec | 0.2869 | 0.7587 | 6 |
| sbert | 0.1968 | 0.8221 | 6 |
| tfidf | 0.031 | 0.72 | 6 |
