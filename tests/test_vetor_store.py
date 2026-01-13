from pathlib import Path
from document_qa_system.document_loader import load_documents
from document_qa_system.text_processor import TextProcessor
from document_qa_system.vectorstore import VectorStoreManager
from document_qa_system.config import Config

def test_vector_store():
    print("=" * 70)
    print("TESTING VECTOR STORE")
    print("=" * 70)

    print("\nStep 1: Loading documents...")
    upload_dir = Config.UPLOAD_DIR
    results = load_documents(upload_dir)

    if not results:
        print("No documents found. Add files to data/uploads/")
        return
    
    all_documents = []
    for  filename,docs in results.items():
        all_documents.extend(docs)

    print(f"Loaded {len(all_documents)} document(s)")

    print("\nStep 2: Chunking documents...")
    processor = TextProcessor()
    chunks = processor.process_documents(all_documents)
    print(f"Created {len(chunks)} chunks")

    print("\nStep 3: Initializing vector store...")
    vector_store = VectorStoreManager(
        persist_directory=Config.CHROMA_PERSIST_DIR / "test_db"
    )

    print("Clearing existing data...")
    vector_store.clear_vector_store()

    print("\nStep 4: Indexing documents...")
    result = vector_store.add_documents(chunks)

    if result['success']:
        print(f"Indexed {result['documents_added']} chunks")
        print(f"   Duration: {result['duration_seconds']:.2f} seconds")
        print(f"   Total in store: {result['total_documents']}")
    else:
        print(f"Indexing failed: {result.get('error')}")
        return
    
    print("\nStep 5: Testing similarity search...")

    test_queries = [
        "What is Python?",
        "How does machine learning work?",
        "Explain databases",
        "What is LangChain?"
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = vector_store.similarity_search(query, k=3)
        
        if results:
            print(f"   Found {len(results)} results:")
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source_file', 'unknown')
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"      {i}. {source}: {preview}...")
        else:
            print("   No results found")

    print("\nStep 6: Testing search with similarity scores...")
    query = "Langchain chains"
    print(f"   Query: '{query}'")

    results_with_scores = vector_store.similarity_search_with_score(query, k=5)

    if results_with_scores:
        print(f"   Found {len(results_with_scores)} results with scores:")
        for i, (doc, score) in enumerate(results_with_scores, 1):
            source = doc.metadata.get('source_file', 'unknown')
            preview = doc.page_content[:80].replace('\n', ' ')
            
            if score > 0.8:
                quality = "Excellent"
            elif score > 0.7:
                quality = "Good"
            else:
                quality = "Fair"
            
            print(f"      {i}. [{quality}] Score: {score:.3f}")
            print(f"         Source: {source}")
            print(f"         Preview: {preview}...")

    print("\nStep 7: Testing metadata filtering...")

    stats = vector_store.get_statistics()
    source_files = stats.get('source_files', [])

    if source_files:
        test_file = source_files[0]
        print(f"   Filtering by source_file: '{test_file}'")
        
        filtered_results = vector_store.similarity_search(
            query="what is this about?",
            k=3,
            filter_dict={"source_file": test_file}
        )
        
        print(f"   Found {len(filtered_results)} results from {test_file}")
        for i, doc in enumerate(filtered_results, 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"      {i}. {preview}...")

    print("\nStep 8: Vector Store Statistics")
    stats = vector_store.get_statistics()
    
    print(f"\n   Total documents: {stats['total_documents']}")
    print(f"   Unique source files: {stats['unique_source_files']}")
    print(f"   Embedding model: {stats['embedding_model']}")
    print(f"   Persist directory: {stats['persist_directory']}")
    print(f"   \n   Source files:")
    for file in stats['source_files']:
        print(f"      - {file}")
    
    print("\n" + "=" * 70)
    print("VECTOR STORE TEST COMPLETE!")
    print("=" * 70)
    
    return vector_store

def test_persistence():
    print("\n" + "=" * 70)
    print("TESTING PERSISTENCE")
    print("=" * 70)

    print("\nLoading existing vector store...")

    vector_store = VectorStoreManager(
        persist_directory=Config.CHROMA_PERSIST_DIR / "test_db"
    )

    count = vector_store.get_document_count()

    if count > 0:
        print(f"Found {count} documents from previous session!")
        print("   Testing search on persisted data...")
        
        results = vector_store.similarity_search("test query", k=2)
        if results:
            print(f"   Search works! Found {len(results)} results")
        else:
            print("   Search returned no results")
    else:
        print("ℹNo persisted data found (run test_vector_store first)")
    
    print("\nPersistence test complete!")

if __name__ == "__main__":
    vector_store = test_vector_store()
    test_persistence()
    