# Document-based RAG Chatbot

Manually upload the document to create your database and instantly talk with chatbot. Get only relevant answers. No hallucinating, no irrelevant answers!

Tackles major retrieval issues in production. Mainly:
1. Unclear queries
2. Confused retrieval, when too many documents exist
3. Missing out Keywords, Acronyms
4. Only some of the context considered to answer. Information Loss


### Improvement Strategy

1. Query reframing
2. Source Citation
3. Semantic Chunking with hyperparameter tuning (Using Golden QA)
4. Hybrid Search with Dense and Sparse(BM25)
5. Re-Ranking

## Tech Stack

- Python
- Database : Chromadb
- Frontend: Streamlit
- Locally hosted
