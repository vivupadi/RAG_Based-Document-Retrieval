import streamlit as st
import time
from backend.rag import RAGPipeline

st.set_page_config(page_title="RAG Q&A", layout="wide")

st.title("RAG Document Q&A")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    st.subheader("Retrieval")
    use_hybrid_search = st.checkbox("Enable Hybrid Search (Dense + Sparse)", value=False, help="Combines semantic (embedding) and keyword (BM25) search")

    if use_hybrid_search:
        dense_weight = st.slider("Dense Weight", 0.0, 1.0, 0.5, 0.1, help="Weight for dense retriever. Sparse weight = 1 - dense_weight")
    else:
        dense_weight = 0.5

    st.subheader("Advanced")
    use_reranking = st.checkbox("Enable Reranking", value=False)
    use_query_reframing = st.checkbox("Enable Query Reframing", value=True)
    use_guardrail = st.checkbox("Enable Safety Guardrail", value=True)

    run_tuning = st.checkbox("Auto-tune parameters", value=False)

    st.divider()

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# File upload
uploaded_file = st.file_uploader("Upload document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    # Initialize or reuse pipeline (clear chat on new document)
    if 'pipeline' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        st.session_state.chat_history = []  # Clear chat on new document

        with st.spinner("Loading document..."):
            pipeline = RAGPipeline(
                use_reranking=use_reranking,
                use_query_reframing=use_query_reframing,
                use_guardrail=use_guardrail,
                use_hybrid_search=use_hybrid_search,
                dense_weight=dense_weight
            )
            pipeline.load_document(uploaded_file)
            pipeline.build_index(chunk_size=600, chunk_overlap=100)

        st.session_state.pipeline = pipeline
        st.success(f"Document '{uploaded_file.name}' loaded successfully!")
    else:
        pipeline = st.session_state.pipeline

    # Optional tuning
    if run_tuning and st.button("Run Parameter Tuning"):
        with st.spinner("Tuning parameters..."):
            best_config = pipeline.tune_chunk_params()

        if best_config:
            st.success(f"Best config: chunk_size={best_config['chunk_size']}, overlap={best_config['overlap']}, top_k={best_config['top_k']}, score={best_config['score']:.2f}")

            with st.spinner("Rebuilding index..."):
                pipeline.rebuild_index(best_config)
            st.success("Index rebuilt with optimal parameters")

    st.divider()

    # Chat interface
    st.subheader("Chat with your document")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            # User message
            with st.chat_message("user"):
                st.write(chat["question"])

            # Assistant message
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                st.caption(f"Response time: {chat['latency']:.2f}s")

                # Show sources in expander
                with st.expander("View Sources"):
                    for idx, source in enumerate(chat["sources"], 1):
                        st.markdown(f"**Source {idx}:**")
                        st.text(source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
                        st.divider()

    # Chat input at the bottom
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Add user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.write(question)

        # Generate response
        with st.spinner("Thinking..."):
            start = time.time()
            result = pipeline.query(question)
            latency = time.time() - start

        # Display assistant response
        with chat_container:
            with st.chat_message("assistant"):
                st.write(result["answer"])
                st.caption(f"Response time: {latency:.2f}s")

                # Show sources in expander
                with st.expander("View Sources"):
                    for idx, source in enumerate(result["sources"], 1):
                        st.markdown(f"**Source {idx}:**")
                        st.text(source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
                        st.divider()

        # Add to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "latency": latency
        })

        # Rerun to update the display
        st.rerun()

else:
    st.info("Upload a document to get started")
