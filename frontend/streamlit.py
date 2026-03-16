#Frontend

import streamlit as st
import time
#from backend.rag import RAGPipeline

st.set_page_config(
    page_title="RAG Prototype",
    layout="wide"
)

st.title("RAG Document Q&A")


# ── File Upload ──────────────────────
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt"]
)

# ── Settings Sidebar ─────────────────
with st.sidebar:
    st.header("Settings")

    use_reranking = st.toggle(
        "Reranking",
        value=True,
        help="Rerank retrieved chunks using a cross-encoder model."
    )

    use_query_reframing = st.toggle(
        "Query Reframing",
        value=True,
        help="Rewrite unclear queries before retrieval."
    )

    use_guardrail = st.toggle(
        "Guardrail",
        value=True,
        help="Check if question is relevant to document before answering."
    )


# ── Main Area ────────────────────────
if uploaded_file:

    # Build index
    with st.spinner("Processing document..."):
        pipeline = RAGPipeline(
            use_reranking=use_reranking,
            use_query_reframing=use_query_reframing,
            use_guardrail=use_guardrail
        )
        pipeline.load_document(uploaded_file)
        pipeline.build_index()
    st.success("Document ready!")

    # Chunk tuning
    if run_tuning:
        with st.spinner("Running chunk parameter tuning..."):
            best_config = pipeline.tune_chunk_params()
        st.info(
            f"Best config — Chunk size: {best_config['chunk_size']}, "
            f"Overlap: {best_config['overlap']}, "
            f"Top-k: {best_config['top_k']}"
        )
        # Rebuild index with best config
        pipeline.rebuild_index(best_config)

    # Question input
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Thinking..."):
            start = time.time()
            result = pipeline.query(question)
            latency = time.time() - start

        # Answer
        st.subheader("Answer")
        st.write(result["answer"])

        # Latency
        st.caption(f"Response time: {latency:.2f}s")

        # Sources
        st.subheader("Sources")
        for source in result["sources"]:
            with st.expander(
                f"Chunk — page {source['metadata'].get('page', '?')}"
            ):
                st.write(source["content"])

        # Evaluation scores
        if use_evaluation and "evaluation" in result:
            st.subheader("Evaluation")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Faithfulness",
                f"{result['evaluation']['faithfulness']:.2f}"
            )
            col2.metric(
                "Answer Relevancy",
                f"{result['evaluation']['answer_relevancy']:.2f}"
            )
            col3.metric(
                "Context Precision",
                f"{result['evaluation']['context_precision']:.2f}"
            )
            col4.metric(
                "Context Recall",
                f"{result['evaluation']['context_recall']:.2f}"
            )

else:
    st.info("Please upload a document to get started.")