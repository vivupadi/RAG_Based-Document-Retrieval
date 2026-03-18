#Vector Database Management - Document ingestion, chunking, and retrieval

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from typing import List
from sentence_transformers import CrossEncoder


class VectorDatabase:
    """Manages vector store operations with data-driven parameter optimization."""

    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.vectorstore = None
        self.raw_docs = None
        self.chunks = None  # Store chunks for BM25 retriever
        self.bm25_retriever = None
        self.evaluator = None  # Cross-encoder for evaluation (lazy loading)

    # Document Loading 

    def load_document(self, uploaded_file):
        """Load document from Streamlit file upload."""
        # Handle PDF files
        if uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
            except ImportError:
                raise ImportError(
                    "PyPDF2 is required for PDF processing. "
                    "Install it with: pip install PyPDF2"
                )

            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            self.raw_docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]

        # Handle text files
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode('utf-8')
            self.raw_docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]

        print(f"Loaded document: {uploaded_file.name}")

    # Chunking Strategies

    def chunking(self, docs, chunk_size=600, chunk_overlap=100):
        separators = [
            "\n\n", "\n", ". ", " ", ""       
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            add_start_index=True,
        )

        all_splits = text_splitter.split_documents(docs)
        print(f"Created {len(all_splits)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        return all_splits

    # Index Management

    def build_index(self, chunk_size=600, chunk_overlap=100, persist_directory="./chroma_db"):
        if not hasattr(self, 'raw_docs') or not self.raw_docs:
            raise ValueError("No documents loaded. Load documents first.")

        print(f"Building index with default config (chunk_size={chunk_size}, overlap={chunk_overlap})")

        # Apply chunking
        splits = self.chunking(self.raw_docs, chunk_size, chunk_overlap)

        # Store chunks for BM25 retriever
        self.chunks = splits

        # Build dense vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )

        # Build BM25 (sparse) retriever
        self.bm25_retriever = BM25Retriever.from_documents(splits)
        self.bm25_retriever.k = 4  # Default top_k

        print(f"✓ Index built with {len(splits)} chunks (dense + sparse)")
        return self.vectorstore

    def rebuild_index(self, config, persist_directory="./chroma_db"):
        """
        Rebuild index with optimized configuration from tune_chunk_params().

        """
        if not hasattr(self, 'raw_docs') or not self.raw_docs:
            raise ValueError("No documents loaded.")

        chunk_size = config.get('chunk_size', 600)
        overlap = config.get('overlap', 100)

        print(f"Rebuilding index with optimized config (chunk_size={chunk_size}, overlap={overlap})")

        splits = self.chunking(self.raw_docs, chunk_size, overlap)

        # Store chunks for BM25 retriever
        self.chunks = splits

        # Rebuild dense vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )

        # Rebuild BM25 (sparse) retriever
        self.bm25_retriever = BM25Retriever.from_documents(splits)
        self.bm25_retriever.k = 4  # Default top_k

        print(f"✓ Index rebuilt with {len(splits)} optimized chunks (dense + sparse)")
        return self.vectorstore

    def ingest_documents(self, docs, chunk_size=600, chunk_overlap=100, persist_directory="./chroma_db"):
        """
        Ingest pre-loaded Langchain documents into the vector store.

        Args:
            docs: List of Langchain Document objects
            chunk_size: Character count per chunk
            chunk_overlap: Characters overlapping between chunks
            persist_directory: Where to store the vector database
        """
        splits = self.chunking(docs, chunk_size, chunk_overlap)

        # Store chunks for BM25 retriever
        self.chunks = splits

        # Build dense vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )

        # Build BM25 (sparse) retriever
        self.bm25_retriever = BM25Retriever.from_documents(splits)
        self.bm25_retriever.k = 4  # Default top_k

        print(f"Ingested {len(splits)} document chunks into vector store (dense + sparse).")
        return self.vectorstore

    # Retrieval 

    def _ensemble_retrieval(self, query, top_k=4, dense_weight=0.5):
        """
        Custom ensemble retrieval combining dense and sparse results.
        """
        # Get results from both retrievers
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k * 2})
        dense_docs = dense_retriever.invoke(query)

        self.bm25_retriever.k = top_k * 2
        sparse_docs = self.bm25_retriever.invoke(query)

        # Score documents based on rank (reciprocal rank fusion)
        doc_scores = {}

        # Score dense results
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.page_content
            score = dense_weight / (rank + 1)  # Reciprocal rank
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

        # Score sparse results
        sparse_weight = 1 - dense_weight
        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.page_content
            score = sparse_weight / (rank + 1)  # Reciprocal rank
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

        # Combine all unique documents
        all_docs = {doc.page_content: doc for doc in dense_docs + sparse_docs}

        # Sort by score and return top_k
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        result = [all_docs[doc_id] for doc_id in sorted_doc_ids[:top_k]]

        return result

    def retrieve_documents(self, query, top_k=4, use_hybrid=False, dense_weight=0.5):
        """
        Retrieve relevant documents for a query.

        """
        if self.vectorstore is None:
            return []

        if use_hybrid and self.bm25_retriever is not None:
            # Use hybrid search: combine dense (semantic) + sparse (keyword)
            docs = self._ensemble_retrieval(query, top_k, dense_weight)
            return docs
        else:
            # Use dense-only search
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(query)
            return docs

    def get_retriever(self, top_k=4, use_hybrid=False, dense_weight=0.5):
        """
        Get a retriever instance for the vector store.
        """
        if self.vectorstore is None:
            return None

        # For hybrid mode, users should call retrieve_documents() directly
        # This method returns dense-only retriever for compatibility
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Evaluation & Tuning 

    def generate_test_questions(self, num_questions=5):
        """
        Generate test questions from the document for evaluation.
        """
        if not hasattr(self, 'raw_docs') or not self.raw_docs:
            return []

        # Sample from document
        sample_text = "\n".join([doc.page_content[:500] for doc in self.raw_docs[:3]])

        prompt = f"""Generate {num_questions} diverse questions that can be answered using this document.
        Return only the questions, one per line.

        Document sample:
        {sample_text}

        Questions:"""

        response = self.llm.invoke(prompt)
        questions = [q.strip() for q in response.content.split('\n') if q.strip() and '?' in q]
        return questions[:num_questions]

    def evaluate_chunks(self, test_questions, top_k=4):
        """
        Evaluate chunk quality using Context Precision metric.
        """
        if self.vectorstore is None:
            return 0.0

        # Lazy load cross-encoder for evaluation
        if self.evaluator is None:
            print("Loading cross-encoder for evaluation...")
            self.evaluator = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        precision_scores = []

        for question in test_questions:
            try:
                docs = retriever.invoke(question)

                if docs:
                    # Score all retrieved documents
                    pairs = [[question, doc.page_content] for doc in docs]
                    relevance_scores = self.evaluator.predict(pairs)

                    # Count how many documents are relevant (threshold: score > 0)
                    relevant_count = sum(1 for score in relevance_scores if score > 0)

                    # Context Precision = relevant docs / total docs retrieved
                    # Scale to 0-10 for consistency with other scores
                    precision = (relevant_count / len(docs)) * 10
                    precision_scores.append(precision)
                else:
                    precision_scores.append(0.0)

            except Exception as e:
                print(f"Warning: Evaluation error for question '{question[:50]}...': {e}")
                precision_scores.append(0.0)

        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        return avg_precision

    def tune_chunk_params(self, test_questions=None, chunk_sizes=None, overlaps=None, top_ks=None):
        """
        Find optimal chunk parameters through grid search evaluation.
        """
        if not hasattr(self, 'raw_docs') or not self.raw_docs:
            print("No documents loaded. Using default config.")
            return {'chunk_size': 600, 'overlap': 100, 'top_k': 4}

        # Generate test questions if not provided
        if test_questions is None:
            print("Generating test questions...")
            test_questions = self.generate_test_questions()

        if not test_questions:
            print("Could not generate test questions. Using default config.")
            return {'chunk_size': 600, 'overlap': 100, 'top_k': 4}

        print(f"Testing configurations with {len(test_questions)} questions...")
        print("This may take a few minutes...\n")

        # Grid search parameters
        chunk_sizes = chunk_sizes or [100, 300, 600, 800]
        overlaps = overlaps or [50, 100, 150, 200]
        top_ks = top_ks or [3, 4, 5]

        best_score = 0
        best_config = None

        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= chunk_size:
                    continue

                for top_k in top_ks:
                    # Rebuild index with these parameters
                    splits = self.chunking(
                        self.raw_docs,
                        chunk_size,
                        overlap
                    )

                    self.vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=self.embeddings,
                        persist_directory=f"./chroma_temp"
                    )

                    # Evaluate
                    score = self.evaluate_chunks(test_questions, top_k)

                    print(f"  chunk_size={chunk_size:4d}, overlap={overlap:3d}, top_k={top_k} → score: {score:.2f}/10")

                    if score > best_score:
                        best_score = score
                        best_config = {
                            'chunk_size': chunk_size,
                            'overlap': overlap,
                            'top_k': top_k,
                            'score': score
                        }

        print(f"\n✓ Best configuration found:")
        print(f"  chunk_size: {best_config['chunk_size']}")
        print(f"  overlap: {best_config['overlap']}")
        print(f"  top_k: {best_config['top_k']}")
        print(f"  score: {best_config['score']:.2f}/10")

        return best_config
