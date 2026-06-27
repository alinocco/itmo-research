# Semantic-space analysis

Descriptive metrics computed against known topic labels (higher silhouette / purity = topics are better separated in the embedding space).

| Method | Silhouette (cosine) | kNN topic purity | #topics |
|--------|---------------------|------------------|---------|
| tfidf | 0.0595 | 0.7123 | 17 |
| doc2vec | -0.009 | 0.4197 | 17 |
| fasttext | -0.044 | 0.5729 | 17 |
