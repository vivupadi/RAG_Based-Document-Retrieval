from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder


class ChunkTuner:
    def __init__(self, vectordb):
        self.db = vectordb
        self.evaluator = None

    def generate_test_questions(self, num_questions=5):
        if not self.db.raw_docs:
            return []

        sample_text = "\n".join([doc.page_content[:500] for doc in self.db.raw_docs[:3]])

        prompt = f"""Generate {num_questions} diverse questions that can be answered using this document.
        Use the same language as the document.
Return only the questions, one per line.

Document sample:
{sample_text}

Questions:"""

        response = self.db.llm.invoke(prompt)
        questions = [q.strip() for q in response.content.split('\n') if q.strip() and '?' in q]
        return questions[:num_questions]

    def evaluate_chunks(self, test_questions, vectorstore, top_k=4):
        if not vectorstore:
            return 0.0

        if not self.evaluator:
            print("Loading cross-encoder...")
            self.evaluator = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        scores = []

        for question in test_questions:
            try:
                docs = retriever.invoke(question)
                if not docs:
                    scores.append(0.0)
                    continue

                pairs = [[question, doc.page_content] for doc in docs]
                relevance_scores = self.evaluator.predict(pairs)

                relevant_count = sum(1 for s in relevance_scores if s > 0)
                precision = (relevant_count / len(docs)) * 10    # Retrieval relevance score
                scores.append(precision)

            except Exception as e:
                print(f"Error: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def tune_parameters(self, test_questions=None, chunk_sizes=None, overlaps=None, top_ks=None):
        if not self.db.raw_docs:
            return {'chunk_size': 600, 'overlap': 100, 'top_k': 4}

        if not test_questions:
            print("Generating test questions...")
            test_questions = self.generate_test_questions()

        if not test_questions:
            return {'chunk_size': 600, 'overlap': 100, 'top_k': 4}

        print(f"Testing with {len(test_questions)} questions: {test_questions}\n")

        chunk_sizes = chunk_sizes or [150, 300, 600, 800]
        overlaps = overlaps or [50, 100, 150, 200]
        top_ks = top_ks or [3, 4, 5]

        best_score = 0
        best_config = None
        results = []

        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= chunk_size:
                    continue

                for top_k in top_ks:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )

                    splits = splitter.split_documents(self.db.raw_docs)

                    vs = Chroma.from_documents(
                        documents=splits,
                        embedding=self.db.embeddings,
                        persist_directory=None
                    )

                    score = self.evaluate_chunks(test_questions, vs, top_k)

                    config = {
                        'chunk_size': chunk_size,
                        'overlap': overlap,
                        'top_k': top_k,
                        'score': score,
                        'num_chunks': len(splits)
                    }
                    results.append(config)

                    print(f"  chunk={chunk_size}, overlap={overlap}, top_k={top_k} -> {score:.2f}/10 ({len(splits)} chunks)")

                    if score > best_score:
                        best_score = score
                        best_config = config
                
        if best_config is None:
            print("\nUsing default configuration: chunk_size=600, overlap=100, top_k=3")
            return {'chunk_size': 600, 'overlap': 100, 'top_k': 3, 'score': 0.0}

        print(f"\nBest: chunk={best_config['chunk_size']}, overlap={best_config['overlap']}, top_k={best_config['top_k']}, score={best_config['score']:.2f}/10")

        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        print("\nTop 3:")
        for i, r in enumerate(sorted_results[:3], 1):
            print(f"{i}. chunk={r['chunk_size']}, overlap={r['overlap']}, top_k={r['top_k']} -> {r['score']:.2f}/10")

        return best_config

    def apply_best_config(self, config):
        print(f"\nApplying config: chunk_size={config['chunk_size']}, overlap={config['overlap']}")
        self.db.build_index(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['overlap'],
            persist_directory="./chroma_db"
        )
        print("Done")
        return config['top_k']
