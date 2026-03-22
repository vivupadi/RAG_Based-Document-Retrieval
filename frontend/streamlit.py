import streamlit as st
import time
from backend.rag import RAGPipeline
from backend.chunk_tuning import ChunkTuner

st.set_page_config(page_title="RAG Q&A", layout="wide")

st.title("RAG Document Q&A")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    st.subheader("Retrieval")
    use_query_reframing = st.checkbox("Enable Query Reframing", value=True)
    use_hybrid_search = st.checkbox("Enable Hybrid Search (Dense + Sparse)", value=False, help="Combines semantic (embedding) and keyword (BM25) search")

    if use_hybrid_search:
        dense_weight = st.slider("Dense Weight", 0.0, 1.0, 0.5, 0.1, help="Weight for dense retriever. Sparse weight = 1 - dense_weight")
    else:
        dense_weight = 0.5

    use_reranking = st.checkbox("Enable Reranking", value=True, help="Use cross-encoder to rerank results for better accuracy")

    st.divider()

    st.subheader("Chunk Tuning")
    run_chunk_tuning = st.checkbox("Auto-tune chunk parameters", value=False,
                                    help="Find optimal chunk size, overlap, and top_k using grid search")

    # Clear chat button
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# File upload - Always visible
uploaded_file = st.file_uploader("Upload document (PDF or TXT)", type=["pdf", "txt"])

# Handle file upload
if uploaded_file:
    # Initialize or update pipeline with new document
    if 'pipeline' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        st.session_state.chat_history = []  # Clear chat on new document

        with st.spinner("Loading document..."):
            pipeline = RAGPipeline(
                use_reranking=use_reranking,
                use_query_reframing=use_query_reframing,
                use_hybrid_search=use_hybrid_search,
                dense_weight=dense_weight
            )
            pipeline.load_document(uploaded_file)
            pipeline.build_index(chunk_size=600, chunk_overlap=100)

        st.session_state.pipeline = pipeline
        st.session_state.best_config = None  # Reset tuning config
        st.success(f"Document '{uploaded_file.name}' loaded successfully!")
    else:
        pipeline = st.session_state.pipeline

    # Chunk tuning section (only when file is uploaded)
    if run_chunk_tuning:
        st.divider()
        st.subheader("Chunk Parameter Tuning")

        # Show current configuration if tuned
        if 'best_config' in st.session_state and st.session_state.best_config:
            st.success("✓ Using optimized configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunk Size", st.session_state.best_config['chunk_size'])
                st.metric("Overlap", st.session_state.best_config['overlap'])
            with col2:
                st.metric("Top K", st.session_state.best_config['top_k'])
                st.metric("Score", f"{st.session_state.best_config['score']:.2f}/10")
        else:
            st.write("**Current settings:** chunk_size=600, overlap=100, top_k=3")

        if st.button("Run Grid Search & Apply"):
            with st.spinner("Running hyperparameter tuning... This may take 2-5 minutes"):
                tuner = ChunkTuner(pipeline.db)

                # Run grid search with smaller search space for faster results
                best_config = tuner.tune_parameters(
                    chunk_sizes=[150, 300, 600, 800],
                    overlaps=[50, 100, 150, 200],
                    top_ks=[3, 4, 5]
                )

                st.session_state.best_config = best_config

            if st.session_state.best_config:
                # Automatically apply the best configuration
                with st.spinner("Applying best configuration and rebuilding index..."):
                    tuner = ChunkTuner(pipeline.db)
                    tuner.apply_best_config(st.session_state.best_config)
                    pipeline.tuned_top_k = st.session_state.best_config['top_k']

                st.success("✓ Tuning complete and applied!")
                st.info(f"Now using: chunk_size={st.session_state.best_config['chunk_size']}, "
                        f"overlap={st.session_state.best_config['overlap']}, "
                        f"top_k={st.session_state.best_config['top_k']}")
                st.rerun()

# If no file uploaded but pipeline exists (from previous session), load it
elif 'pipeline' not in st.session_state:
    # Try to load existing vectorstore
    import os
    if os.path.exists("./chroma_db"):
        try:
            with st.spinner("Loading existing vectorstore..."):
                from langchain_community.vectorstores import Chroma
                
                pipeline = RAGPipeline(
                    use_reranking=use_reranking,
                    use_query_reframing=use_query_reframing,
                    use_hybrid_search=use_hybrid_search,
                    dense_weight=dense_weight
                )
                
                # Load existing vectorstore
                pipeline.db.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=pipeline.embeddings
                )

                # Rebuild BM25 retriever for hybrid search
                pipeline.db.rebuild_bm25_from_vectorstore()

                st.session_state.pipeline = pipeline
                st.info("Loaded existing vectorstore from previous session")
        except Exception as e:
            st.warning(f"⚠️ Could not load existing vectorstore: {str(e)}")
            st.info("Please upload a document to create a new vectorstore")

# Divider before chat
st.divider()

# Chat interface - Always visible
st.subheader("Chat with your document")

# Check if vectorstore exists
if 'pipeline' in st.session_state and st.session_state.pipeline.db.vectorstore is not None:
    pipeline = st.session_state.pipeline

    # Sync pipeline settings with sidebar
    pipeline.use_query_reframing = use_query_reframing
    pipeline.use_reranking_flag = use_reranking
    pipeline.use_hybrid_search = use_hybrid_search
    pipeline.dense_weight = dense_weight
    
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
                        # Add document name
                        doc_name = source.get("metadata", {}).get("source", "Unknown")
                        st.markdown(f"**Source {idx}** (from: *{doc_name}*)")
                        
                        content_preview = source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"]
                        st.text(content_preview)
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
                        # Add document name
                        doc_name = source.get("metadata", {}).get("source", "Unknown")
                        st.markdown(f"**Source {idx}** (from: *{doc_name}*)")
                        
                        content_preview = source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"]
                        st.text(content_preview)
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
    # No vectorstore available
    st.info("📤 Upload a document above to get started, or a saved vectorstore will load automatically if available")
