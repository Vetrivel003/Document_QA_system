from pathlib import Path
from document_qa_system.document_loader import load_documents
from document_qa_system.text_processor import TextProcessor
from document_qa_system.config import Config

print("Starting test_chunking...")

def test_chunking():
    print("=" * 70)
    print("TESTING TEXT CHUNKING")
    print("=" * 70)

    upload_dir = Config.UPLOAD_DIR
    print(f"\nLoading documents from {upload_dir.name}...")

    results = load_documents(upload_dir)

    if not results:
        print("No documents found. Add files to data/uploads/")
        return
    
    all_documents = []
    for filename,docs in results.items():
        if isinstance(docs[0], list):
                print(" WARNING: Found nested list! Flattening...")
                for doc_list in docs:
                    all_documents.extend(doc_list)
        else:
            all_documents.extend(docs)

    print(f"Loaded {len(all_documents)} document(s) from {len(results)} file(s)")

    print(f"\nInitializing TextProcessor...")
    print(f"   Chunk size: {Config.CHUNK_SIZE}")
    print(f"   Chunk overlap: {Config.CHUNK_OVERLAP}")

    processor = TextProcessor()

    print(f"\nChunking documents...")
    chunks = processor.process_documents(all_documents)

    print(f"\nCHUNK ANALYSIS:")
    analysis = processor.analyze_chunk_quality(chunks)

    print(f"\n   Total chunks: {analysis['total_chunks']}")
    print(f"   Average size: {analysis['average_size']} characters")
    print(f"   Size range: {analysis['size_range'][0]} - {analysis['size_range'][1]} chars")

    print(f"\n   Quality Metrics:")
    metrics = analysis['quality_metrics']
    print(f"      Chunks too small (<300 chars): {metrics['chunks_too_small']}")
    print(f"      Chunks too large (>1500 chars): {metrics['chunks_too_large']}")
    print(f"      Chunks starting mid-sentence: {metrics['chunks_with_low_context']}")

    print(f"\n   Recommendations:")
    for rec in analysis['recommendations']:
        print(f"      {rec}")

    print(f"\nSAMPLE CHUNKS (showing first 3):")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n   --- Chunk {i + 1} ---")
        print(f"   Source: {chunk.metadata.get('source_file', 'unknown')}")
        print(f"   Size: {chunk.metadata.get('chunk_size', 0)} chars")
        print(f"   Words: {chunk.metadata.get('word_count', 0)}")
        print(f"   Preview: {chunk.metadata.get('content_preview', '')}")
        
        content = chunk.page_content[:300].replace('\n', ' ')
        print(f"   Content: {content}...")

    print(f"\nTESTING DIFFERENT CHUNK SIZES:")
    test_sizes = [500, 1000, 1500]

    for size in test_sizes:
        test_processor = TextProcessor(chunk_size=size, chunk_overlap=int(size * 0.2))
        test_chunks = test_processor.process_documents(all_documents)
        test_analysis = test_processor.analyze_chunk_quality(test_chunks)
        
        print(f"\n   Chunk size {size}:")
        print(f"      Total chunks: {test_analysis['total_chunks']}")
        print(f"      Avg size: {test_analysis['average_size']} chars")
        print(f"      Quality issues: {sum(test_analysis['quality_metrics'].values())}")
    
    print("\n" + "=" * 70)
    print("CHUNKING TEST COMPLETE!")
    print("=" * 70)
    
    return chunks, analysis

def test_edge_cases():
    print("\n" + "=" * 70)
    print("TESTING EDGE CASES")
    print("=" * 70)

    from langchain_core.documents import Document
    processor = TextProcessor()

    print("\n1️ Testing very short document...")
    short_doc = Document(
        page_content="This is a very short document.",
        metadata={"source": "test"}
    )
    chunks = processor.process_documents([short_doc])
    print(f"   Input: {len(short_doc.page_content)} chars")
    print(f"   Output: {len(chunks)} chunk(s)")

    print("\n2️ Testing document with no natural breaks...")
    no_breaks = Document(
        page_content="word" * 500,  
        metadata={"source": "test"}
    )
    chunks = processor.process_documents([no_breaks])
    print(f"   Input: {len(no_breaks.page_content)} chars (no spaces)")
    print(f"   Output: {len(chunks)} chunk(s)")

    print("\n3️ Testing document with excessive whitespace...")
    whitespace_doc = Document(
        page_content="Word.\n\n\n\n\nAnother word.\n\n\n\nMore text." * 50,
        metadata={"source": "test"}
    )
    chunks = processor.process_documents([whitespace_doc])
    print(f"   Input: {len(whitespace_doc.page_content)} chars")
    print(f"   Output: {len(chunks)} chunk(s)")


    print("\n4️ Testing empty document...")
    empty_doc = Document(page_content="", metadata={"source": "test"})
    chunks = processor.process_documents([empty_doc])
    print(f"   Input: 0 chars")
    print(f"   Output: {len(chunks)} chunk(s)")
    
    print("\nEdge case testing complete!")

if __name__ == "__main__":

    chunks, analysis = test_chunking()
    
    test_edge_cases()
    
    print("\nTIP: Review the chunk samples above.")
    print("   If chunks look broken or incomplete, adjust Config.CHUNK_SIZE")