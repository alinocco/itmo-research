# Semantic-space analysis

Descriptive metrics computed against known topic labels (higher silhouette / purity = topics are better separated in the embedding space).

| Method | Silhouette (cosine) | kNN topic purity | #topics |
|--------|---------------------|------------------|---------|
| sbert | 0.1632 | 0.849 | 8 |
| gte | 0.1184 | 0.844 | 8 |
| bert | 0.1055 | 0.7526 | 8 |
| e5 | 0.1041 | 0.819 | 8 |
| tfidf | 0.0785 | 0.7947 | 8 |
| bge-m3 | 0.0713 | 0.802 | 8 |
| fasttext | 0.0704 | 0.78 | 8 |
| doc2vec | 0.0025 | 0.5627 | 8 |
