import streamlit as st
from pathlib import Path
import time
from datetime import datetime
import shutil

from src.document_qa_system.document_loader import load_documents, DocumentLoader
from src.document_qa_system.text_processor import TextProcessor
from src.document_qa_system.vectorstore import VectorStoreManager
from src.document_qa_system.rag_chain import RAGChain
from src.document_qa_system.config import Config

st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .stat-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeeba;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_indexed' not in st.session_state:
        st.session_state.documents_indexed = False
    
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'k_documents': 4,
            'temperature': 0.1,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'streaming': True
        }


def load_vector_store():
    """Load or create vector store."""
    if st.session_state.vector_store is None:
        with st.spinner("Loading vector store..."):
            st.session_state.vector_store = VectorStoreManager(
                persist_directory=Config.CHROMA_PERSIST_DIR
            )
            
            doc_count = st.session_state.vector_store.get_document_count()
            st.session_state.documents_indexed = doc_count > 0


def load_rag_chain():
    """Initialize RAG chain."""
    if st.session_state.rag_chain is None and st.session_state.vector_store is not None:
        if st.session_state.vector_store.get_document_count() > 0:
            with st.spinner("Initializing RAG chain..."):
                st.session_state.rag_chain = RAGChain(
                    vector_store_manager=st.session_state.vector_store,
                    k_documents=st.session_state.settings['k_documents'],
                    temperature=st.session_state.settings['temperature']
                )


def sidebar():
    """Render sidebar with settings and document management."""
    with st.sidebar:
        st.markdown("###  Settings")
        
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_statistics()
            
            st.markdown("####  System Status")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Documents", stats['total_documents'])
            
            with col2:
                st.metric("Source Files", stats['unique_source_files'])
            
            if st.session_state.documents_indexed:
                st.success(" System Ready")
            else:
                st.warning(" No documents indexed")
        
        st.divider()
        
        st.markdown("####  RAG Configuration")
        
        k_docs = st.slider(
            "Documents to Retrieve (k)",
            min_value=1,
            max_value=10,
            value=st.session_state.settings['k_documents'],
            help="Number of relevant chunks to retrieve for each query"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings['temperature'],
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        streaming = st.checkbox(
            "Enable Streaming",
            value=st.session_state.settings['streaming'],
            help="Stream responses in real-time"
        )
        
        if (k_docs != st.session_state.settings['k_documents'] or 
            temperature != st.session_state.settings['temperature']):
            
            st.session_state.settings['k_documents'] = k_docs
            st.session_state.settings['temperature'] = temperature
            st.session_state.rag_chain = None  
            st.rerun()
        
        st.session_state.settings['streaming'] = streaming
        
        st.divider()
        
        st.markdown("####  Document Management")
        
        if st.button(" Clear All Documents", type="secondary", use_container_width=True):
            if st.session_state.vector_store:
                with st.spinner("Clearing documents..."):
                    st.session_state.vector_store.clear_vector_store()
                    st.session_state.documents_indexed = False
                    st.session_state.rag_chain = None
                    st.session_state.chat_history = []
                st.success("Documents cleared!")
                time.sleep(1)
                st.rerun()
        
        if st.button(" Refresh System", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.rag_chain = None
            st.rerun()
        
        st.divider()
        
        st.markdown("####  Model Information")
        st.text(f"LLM: {Config.GROQ_MODEL}")
        st.text(f"Embeddings: {Config.EMBEDDING_MODEL.split('/')[-1]}")


def upload_documents_tab():
    """Document upload and indexing interface."""
    st.markdown("###  Upload Documents")
    st.markdown("Upload PDF, TXT, or DOCX files to build your knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload one or more documents"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        process_button = st.button(
            " Process & Index",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True
        )
    
    if process_button and uploaded_files:
        process_uploaded_files(uploaded_files)
    
    if st.session_state.vector_store:
        stats = st.session_state.vector_store.get_statistics()
        
        if stats['total_documents'] > 0:
            st.divider()
            st.markdown("###  Indexed Documents")
            
            if stats['source_files']:
                for i, file in enumerate(stats['source_files'], 1):
                    st.markdown(f"{i}. **{file}**")
            else:
                st.info("No source file information available")


def process_uploaded_files(uploaded_files):
    """Process and index uploaded files."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(" Saving files...")
        saved_files = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_path = Config.UPLOAD_DIR / uploaded_file.name
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            saved_files.append(file_path)
            progress_bar.progress((i + 1) / (len(uploaded_files) * 4))
        
        status_text.text(f" Saved {len(saved_files)} file(s)")
        
        status_text.text(" Loading documents...")
        loader = DocumentLoader()
        all_documents = []
        
        for i, file_path in enumerate(saved_files):
            docs = loader.load_document(file_path)
            all_documents.extend(docs)
            progress_bar.progress((len(uploaded_files) + i + 1) / (len(uploaded_files) * 4))
        
        status_text.text(f" Loaded {len(all_documents)} document(s)")
        
        status_text.text(" Chunking documents...")
        processor = TextProcessor(
            chunk_size=st.session_state.settings['chunk_size'],
            chunk_overlap=st.session_state.settings['chunk_overlap']
        )
        chunks = processor.process_documents(all_documents)
        progress_bar.progress(0.75)
        
        status_text.text(f" Created {len(chunks)} chunks")
        
        status_text.text(" Indexing in vector store...")
        if st.session_state.vector_store is None:
            load_vector_store()
        
        result = st.session_state.vector_store.add_documents(chunks)
        progress_bar.progress(1.0)
        
        if result['success']:
            status_text.empty()
            progress_bar.empty()
            
            st.success(
                f"Successfully indexed {result['documents_added']} chunks "
                f"in {result['duration_seconds']:.2f} seconds!"
            )
            
            st.session_state.documents_indexed = True
            st.session_state.rag_chain = None  
            
            time.sleep(2)
            st.rerun()
        else:
            st.error(f" Indexing failed: {result.get('error')}")
    
    except Exception as e:
        st.error(f" Error processing files: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def qa_interface_tab():
    """Q&A interface with chat history."""
    st.markdown("###  Ask Questions")
    
    if not st.session_state.documents_indexed:
        st.warning(" Please upload and index documents first!")
        return
    
    if st.session_state.rag_chain is None:
        load_rag_chain()
    
    chat_container = st.container()
    
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            
            with st.chat_message("user"):
                st.markdown(chat['question'])
            
            with st.chat_message("assistant"):
                st.markdown(chat['answer'])
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    if chat.get('sources'):
                        with st.expander(f"📚 View {len(chat['sources'])} sources"):
                            for source in chat['sources']:
                                st.markdown(f"**[{source['index']}] {source['file']}**")
                                st.caption(source['preview'])
                                st.divider()
                
                with col2:
                    st.caption(f"⏱️ {chat.get('processing_time', 0):.2f}s")
    
    st.divider()
    
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        process_question(question)


def process_question(question):
    """Process user question and display answer."""
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        if st.session_state.settings['streaming']:
            response_placeholder = st.empty()
            full_response = ""
            
            start_time = time.time()
            
            for chunk in st.session_state.rag_chain.stream_query(question):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            processing_time = time.time() - start_time
            
            retrieved_docs = st.session_state.rag_chain.retriever.invoke(question)
            sources = st.session_state.rag_chain._format_sources(retrieved_docs)
            
            with st.expander(f"📚 View {len(sources)} sources"):
                for source in sources:
                    st.markdown(f"**[{source['index']}] {source['file']}**")
                    st.caption(source['preview'])
                    st.divider()
            
            st.caption(f"⏱️ {processing_time:.2f}s")
            
            st.session_state.chat_history.append({
                'question': question,
                'answer': full_response,
                'sources': sources,
                'processing_time': processing_time,
                'timestamp': datetime.now()
            })
        
        else:
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.query(question, return_sources=True)
            
            if result['success']:
                st.markdown(result['answer'])
                
                if result.get('sources'):
                    with st.expander(f"📚 View {len(result['sources'])} sources"):
                        for source in result['sources']:
                            st.markdown(f"**[{source['index']}] {source['file']}**")
                            st.caption(source['preview'])
                            st.divider()
                
                st.caption(f"⏱️ {result['processing_time']:.2f}s")
                
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'processing_time': result['processing_time'],
                    'timestamp': datetime.now()
                })
            else:
                st.error(f"Error: {result.get('error')}")


def analytics_tab():
    """Display system analytics and statistics."""
    st.markdown("### 📊 Analytics")
    
    if not st.session_state.vector_store:
        st.info("No data available yet. Upload and index documents first.")
        return
    
    stats = st.session_state.vector_store.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("Total Chunks", stats['total_documents'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("Source Files", stats['unique_source_files'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("Questions Asked", len(st.session_state.chat_history))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        if st.session_state.chat_history:
            avg_time = sum(c['processing_time'] for c in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        else:
            st.metric("Avg Response Time", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    if stats['source_files']:
        st.markdown("#### 📁 Indexed Documents")
        
        for i, file in enumerate(stats['source_files'], 1):
            st.markdown(f"{i}. **{file}**")
    
    st.divider()
    
    if st.session_state.chat_history:
        st.markdown("#### 💬 Recent Queries")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Query {i}: {chat['question'][:50]}..."):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer'][:200]}...")
                st.caption(f"Time: {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | "
                          f"Processing: {chat['processing_time']:.2f}s")
    
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


def main():
    """Main application."""

    init_session_state()
    load_vector_store()
    
    st.markdown('<p class="main-header">📚 Enterprise Document Q&A System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions, get answers with source citations</p>', unsafe_allow_html=True)
    
    sidebar()
    
    tab1, tab2, tab3 = st.tabs(["💬 Q&A", "📤 Upload Documents", "📊 Analytics"])
    
    with tab1:
        qa_interface_tab()
    
    with tab2:
        upload_documents_tab()
    
    with tab3:
        analytics_tab()
    
    st.divider()
    st.caption("Built with LangChain, Groq, Chroma, and Streamlit | Powered by RAG")


if __name__ == "__main__":
    main()