# Semantic-space analysis

Descriptive metrics computed against known topic labels (higher silhouette / purity = topics are better separated in the embedding space).

| Method | Silhouette (cosine) | kNN topic purity | #topics |
|--------|---------------------|------------------|---------|
| sbert | 0.238 | 0.77 | 6 |
| bert | 0.2078 | 0.679 | 6 |
| gte | 0.1973 | 0.761 | 6 |
| e5 | 0.1805 | 0.7137 | 6 |
| bge-m3 | 0.1279 | 0.6797 | 6 |
| fasttext | 0.1165 | 0.6067 | 6 |
| doc2vec | 0.0717 | 0.6157 | 6 |
| tfidf | 0.0437 | 0.6983 | 6 |
