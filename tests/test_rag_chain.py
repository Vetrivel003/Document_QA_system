from pathlib import Path
from langchain_core.documents import Document
from document_qa_system.document_loader import load_documents
from document_qa_system.text_processor import TextProcessor
from document_qa_system.vectorstore import VectorStoreManager
from document_qa_system.rag_chain import RAGChain
from document_qa_system.config import Config
import time

def setup_test_environment():
    print("=" * 70)
    print("SETUP: Preparing Test Environment")
    print("=" * 70)

    vector_store = VectorStoreManager(
        persist_directory=Config.CHROMA_PERSIST_DIR / "test_db"
    )

    doc_count = vector_store.get_document_count()

    if doc_count == 0:
        print("\nNo documents in vector store. Indexing now...")
        
        print("\nLoading documents from uploads...")
        upload_dir = Config.UPLOAD_DIR
        results = load_documents(upload_dir)
        
        if not results:
            print("ERROR: No documents found in data/uploads/")
            print("Please add some documents (PDF, TXT, DOCX) to test with.")
            return None
        
        all_documents = []
        for filename, docs in results.items():
            all_documents.extend(docs)
        
        print(f"Loaded {len(all_documents)} document(s) from {len(results)} file(s)")
        
        print("\nChunking documents...")
        processor = TextProcessor()
        chunks = processor.process_documents(all_documents)
        print(f"Created {len(chunks)} chunks")
        
        print("\nIndexing documents...")
        result = vector_store.add_documents(chunks)
        
        if result['success']:
            print(f" Indexed {result['documents_added']} chunks")
            print(f" Duration: {result['duration_seconds']:.2f}s")
        else:
            print(f" Indexing failed: {result.get('error')}")
            return None
    else:
        print(f"\nUsing existing vector store with {doc_count} documents")
    
    return vector_store

def test_basic_qa():
    print("\n" + "=" * 70)
    print("TEST 1: Basic Question Answering")
    print("=" * 70)

    vector_store = setup_test_environment()
    if not vector_store:
        return None
    
    print("\n Initializing RAG chain...")
    rag_chain = RAGChain(
        vector_store_manager=vector_store,
        k_documents=4
    )

    info = rag_chain.get_chain_info()
    print(f"\nChain Configuration:")
    print(f"   Model: {info['model']}")
    print(f"   Chain Type: {info['chain_type']}")
    print(f"   Documents to retrieve: {info['k_documents']}")
    print(f"   Total indexed documents: {info['vector_store_documents']}")
    print(f"   Embedding model: {info['embedding_model']}")

    test_questions = [
        "What is the main topic of these documents?",
        "Can you summarize the key points?",
        "What are the important concepts explained?",
        "Are there any examples or use cases mentioned?",
    ]

    print(f"\n Testing {len(test_questions)} questions...\n")

    results = []
    for i, question in enumerate(test_questions, 1):
        print("=" * 70)
        print(f"Question {i}/{len(test_questions)}: {question}")
        print("=" * 70)
        
        result = rag_chain.query(question, return_sources=True)
        results.append(result)

        if result['success']:
            print(f"\nAnswer:")
            print(f"{result['answer']}\n")
            
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Model: {result['model']}")
            
            if 'sources' in result and result['sources']:
                print(f"\nSources ({result['num_sources']} documents retrieved):")
                for source in result['sources']:
                    print(f"\n   [{source['index']}] {source['file']}")
                    if 'page' in source:
                        print(f"       Page: {source['page']}")
                    print(f"       Preview: {source['preview']}")
        else:
            print(f"\nError: {result.get('error')}")
        
        print()  

    print("=" * 70)
    print("TEST 1 SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results if r['success'])
    avg_time = sum(r.get('processing_time', 0) for r in results if r['success']) / max(successful, 1)
    
    print(f"Successful queries: {successful}/{len(test_questions)}")
    print(f"Average processing time: {avg_time:.2f}s")
    
    return rag_chain


def test_streaming():
    print("\n" + "=" * 70)
    print("TEST 2: Streaming Responses")
    print("=" * 70)

    vector_store = VectorStoreManager(
        persist_directory=Config.CHROMA_PERSIST_DIR / "test_db"
    )

    if vector_store.get_document_count() == 0:
        print("\n No documents in vector store.")
        print("   Run test_basic_qa first or add documents to data/uploads/")
        return
    
    print("\nInitializing RAG chain...")
    rag_chain = RAGChain(vector_store_manager=vector_store)

    test_questions = [
        "What are the main topics in these documents?",
        "Explain the key concepts briefly",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"Streaming Question {i}: {question}")
        print('=' * 70)
        print("\nAnswer (streaming): ", end="", flush=True)
        
        start_time = time.time()
        
        for chunk in rag_chain.stream_query(question):
            print(chunk, end="", flush=True)
        
        duration = time.time() - start_time
        print(f"\n\nStreaming completed in {duration:.2f}s")
    
    print("\nStreaming test complete!")


if __name__ == "__main__":
    # setup_test_environment()

    test_streaming()