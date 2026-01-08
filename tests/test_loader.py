from pathlib import Path
from document_qa_system.document_loader import DocumentLoader
from document_qa_system.config import Config

loader = DocumentLoader()

def test_loader():
    """Test document loading with sample files."""
    
    print("=" * 60)
    print("TESTING DOCUMENT LOADER")
    print("=" * 60)

    upload_dir = Config.UPLOAD_DIR

    files = list(upload_dir.glob("*"))
    supported_files = [
        f for f in files 
        if f.suffix.lower() in Config.SUPPORTED_FORMATS
    ]

    if not supported_files:
        print(f"\nNo files found in {upload_dir}")
        print("Please add some test files (.pdf, .txt, .docx) to:")
        print(f"   {upload_dir.absolute()}")
        return
    
    print(f"\nFound {len(supported_files)} supported files:")
    for f in supported_files:
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")

    print(f"\nLoading documents...")
    results = loader.load_directory(upload_dir)

    print(f"\nRESULTS:")
    print(f"   Total files processed: {len(results)}")

    total_docs = sum(len(docs) for docs in results.values())
    print(f"   Total document chunks: {total_docs}")

    print(f"\nDOCUMENT DETAILS:")
    for filename, docs in results.items():
        print(f"\n   {filename}:")
        print(f"      Chunks: {len(docs)}")
        print(f"      First chunk length: {len(docs[0].page_content)} chars")
        print(f"      Metadata keys: {list(docs[0].metadata.keys())}")
        
        preview = docs[0].page_content
        print(f"      Preview: {preview}...")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_loader()