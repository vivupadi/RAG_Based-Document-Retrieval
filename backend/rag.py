from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from backend.database import VectorDatabase
from sentence_transformers import CrossEncoder

load_dotenv()


class RAGPipeline:
    def __init__(self, use_reranking=False, use_query_reframing=True, use_hybrid_search=False, dense_weight=0.5):

        MISTRAL_API = os.getenv("MISTRAL_API_KEY")

        # LLM model
        model = "mistral-large-latest"

        self.llm = ChatMistralAI(
            model=model,
            temperature=0.1,
            api_key= MISTRAL_API
        )

        #Embedding model
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            api_key= MISTRAL_API
        )

        # Initialize Vector Database
        self.db = VectorDatabase(self.embeddings, self.llm)

        # RAG settings
        self.use_reranking_flag = use_reranking
        self.use_query_reframing = use_query_reframing
        self.use_hybrid_search = use_hybrid_search
        self.dense_weight = dense_weight
        self.tuned_top_k = 3  

        # Initialize cross-encoder for reranking (lazy loading)
        self.reranker = None
        if use_reranking:
            print("Loading cross-encoder model for reranking...")
            self.reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
            print("✓ Cross-encoder loaded")

    def rag_chain(self, question, retrieved_docs):
        if self.use_query_reframing:
            reframed_question = self.query_reframing(question)
            print(f"Original: {question}")
            print(f"Reframed: {reframed_question}")
        else:
            reframed_question = question

        # Number the sources so LLM can cite them
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Source {i}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        prompt_template = """You are answering a single question. The question language determines your response language.

Question: {question}

LANGUAGE RULE: Detect the language of the question above. Write your ENTIRE answer in EXACTLY that language. If the question is in English, answer in English. If the question is in German, answer in German. Do NOT mix languages under any circumstances.

Context:
{context}

Instructions:
- Answer using only the provided context
- If multiple sources are relevant, combine them
- Cite which sources you used (e.g., "Source 1", "Source 2")
- If you cannot answer based on the context, say so clearly

Answer (in the SAME language as the question):"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": reframed_question})

        return answer, retrieved_docs

    def use_reranking(self, query, docs, top_k=3):
        if not self.reranker or not docs:
            return docs[:top_k]

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:top_k]]

    def query_reframing(self, question):
        prompt = f"""Rewrite this question to be clearer
        and more specific for document search.
        Keep the same language as the input.
        Return only the rewritten question, nothing else.

        Question: {question}"""

        response = self.llm.invoke(prompt)
        return response.content

    def query(self, question):
        if not self.db.vectorstore:
            return {"answer": "Error: No documents ingested.", "sources": []}

        search_query = question
        if self.use_query_reframing:
            search_query = self.query_reframing(question)

        # Determine retrieval strategy
        if self.use_reranking_flag:
            initial_k = 10  # Get 10 candidates for reranking
            final_k = 3      # Rerank to top 3
        else:
            initial_k = self.tuned_top_k  # tuned top_k
            final_k = self.tuned_top_k

        docs = self.db.retrieve_documents(
            search_query,
            top_k=initial_k,
            use_hybrid=self.use_hybrid_search,
            dense_weight=self.dense_weight
        )

        # Rerank to get best 3
        if self.use_reranking_flag:
            docs = self.use_reranking(search_query, docs, top_k=final_k)
        else:
            docs = docs[:final_k]

        answer, _ = self.rag_chain(question, docs)

        sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

        return {"answer": answer, "sources": sources}

    # Delegation Methods to VectorDatabase

    def load_document(self, uploaded_file):
        """Load document from file upload. Delegates to VectorDatabase."""
        return self.db.load_document(uploaded_file)

    def build_index(self, chunk_size=600, chunk_overlap=100, persist_directory="./chroma_db"):
        """Build vector index from loaded documents. Delegates to VectorDatabase."""
        return self.db.build_index(chunk_size, chunk_overlap, persist_directory)

