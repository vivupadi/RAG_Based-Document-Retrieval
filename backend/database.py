from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


class VectorDatabase:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.vectorstore = None
        self.bm25_retriever = None
        self.chunks = None
        self.raw_docs = None

    def load_document(self, uploaded_file):
        if uploaded_file.type == "application/pdf":
            import pdfplumber
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    text += " | ".join(clean_row) + "\n"
                        text += "\n"
                    
                    # Extract regular text
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if not text.strip():
                raise ValueError(f"No text extracted from PDF: {uploaded_file.name}")

            self.raw_docs = [Document(page_content=text.strip(), metadata={"source": uploaded_file.name})]
            print(f"Loaded: {uploaded_file.name} ({len(text)} chars)")

        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode('utf-8')
            if not text.strip():
                raise ValueError(f"Empty file: {uploaded_file.name}")

            self.raw_docs = [Document(page_content=text.strip(), metadata={"source": uploaded_file.name})]
            print(f"Loaded: {uploaded_file.name}")

        else:
            raise ValueError(f"Unsupported type: {uploaded_file.type}")

    def build_index(self, chunk_size=600, chunk_overlap=100, persist_directory="./chroma_db"):
        if not self.raw_docs:
            raise ValueError("No documents loaded")

        total_chars = sum(len(doc.page_content) for doc in self.raw_docs)
        if total_chars == 0:
            raise ValueError("Empty document")

        print(f"Building index (chunk={chunk_size}, overlap={chunk_overlap})")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

        self.chunks = text_splitter.split_documents(self.raw_docs)

        if not self.chunks:
            raise ValueError(f"0 chunks created from {total_chars} chars. Try smaller chunk_size")

        print(f"Created {len(self.chunks)} chunks")

        # Build vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )

        # Build BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = 4

        print(f"Index ready ({len(self.chunks)} chunks)")
        return self.vectorstore

    def retrieve_documents(self, query, top_k=4, use_hybrid=False, dense_weight=0.5):
        if not self.vectorstore:
            raise ValueError("No index built")

        if use_hybrid and self.bm25_retriever:
            return self._hybrid_search(query, top_k, dense_weight)
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            return retriever.invoke(query)

    def _hybrid_search(self, query, top_k=4, dense_weight=0.5):
        """Hybrid search using Weighted Reciprocal Rank Fusion (RRF)
        """
        # Get more candidates than needed for better RRF results
        k_candidates = top_k * 2

        # Get results from both retrievers
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_candidates})
        dense_docs = dense_retriever.invoke(query)

        self.bm25_retriever.k = k_candidates
        sparse_docs = self.bm25_retriever.invoke(query)

        # Weighted RRF scoring: score = w_dense * (1/(k+rank_dense)) + w_sparse * (1/(k+rank_sparse))
        # k=60 is a standard constant from the RRF paper
        rrf_k = 60
        sparse_weight = 1.0 - dense_weight

        doc_scores = {}
        doc_objects = {}

        # Score documents from dense (semantic) retriever with weight
        for rank, doc in enumerate(dense_docs, 1):
            content = doc.page_content
            if content not in doc_scores:
                doc_scores[content] = 0
                doc_objects[content] = doc
            doc_scores[content] += dense_weight * (1 / (rrf_k + rank))

        # Score documents from sparse (BM25) retriever with weight
        for rank, doc in enumerate(sparse_docs, 1):
            content = doc.page_content
            if content not in doc_scores:
                doc_scores[content] = 0
                doc_objects[content] = doc
            doc_scores[content] += sparse_weight * (1 / (rrf_k + rank))

        # Sort by weighted RRF score (descending) and return top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Return Document objects in order of RRF score
        return [doc_objects[content] for content, _ in sorted_docs]

    def rebuild_bm25_from_vectorstore(self):
        """Rebuild BM25 retriever from existing vectorstore (for auto-load scenarios)"""
        if not self.vectorstore:
            print("No vectorstore to rebuild BM25 from")
            return

        # Get all documents from ChromaDB
        all_data = self.vectorstore.get()

        if not all_data or not all_data.get('documents'):
            print("No documents found in vectorstore")
            return

        # Convert to LangChain Document format
        self.chunks = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(all_data['documents'], all_data['metadatas'] or [{}] * len(all_data['documents']))
        ]

        # Rebuild BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = 4

        print(f"BM25 retriever rebuilt with {len(self.chunks)} chunks")

