from pathlib import Path 
from typing import List,Dict,Optional
import logging
from datetime import datetime

from langchain_community.document_loaders import(
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_core.documents import Document

from document_qa_system.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.supported_formats = Config.SUPPORTED_FORMATS
        self.max_file_size = Config.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def load_document(self,file_path : str | Path) -> List[Document]:
        file_path = Path(file_path)

        # Validation
        self._validate_file(file_path)

        #Load based on File Types
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                documents = self._load_pdf(file_path)
            elif suffix == '.txt':
                documents = self._load_txt(file_path)
            elif suffix == '.docx':
                documents = self._load_docx(file_path)
            else: 
                raise ValueError(f"Unsupported file format: {suffix}")
            
            documents = self._enrich_metadata(documents,file_path)

            logger.info(
                f"Successfully loaded {file_path.name}: "
                f"{len(documents)} document(s)"
            )

            return documents
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {str(e)}")
            raise

    def load_directory(self,directory_path : str | Path) -> Dict[str,List[Document]]:
        directory_path = Path(directory_path)

        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        results = {}

        files = [
            f for f in directory_path.iterdir()
            if f.suffix.lower() in self.supported_formats
        ]

        logger.info(f"Found {len(files)} supported files in {directory_path.name}")

        for file_path in files:
            try:
                documents = self.load_document(file_path)
                results[file_path.name] = documents
            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: {str(e)}")
                continue
        logger.info(
            f"Successfully loaded {len(results)}/{len(files)} files"
        )
        return results
    
    def _validate_file(self,file_path : Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found : {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {suffix}."
                f"Supported : {', '.join(self.supported_formats)}"
            )
        
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            raise ValueError(
                f"File too large: {size_mb:.2f}MB "
                f"(max: {Config.MAX_FILE_SIZE_MB}MB)"
            )
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    
    def _load_txt(self , file_path : Path) -> List[Document]:
        encodings = ['utf-8','latin-1','cp1252']

        for encoding in encodings:
            try:
                loader = TextLoader(str(file_path),encoding=encoding)
                return loader.load()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {file_path.name} with supported encodings")
    
    def _load_docx(self,file_path:Path) -> List[Document]:
        loader = Docx2txtLoader(str(file_path))
        return loader.load()
    
    def _enrich_metadata(self,documents:List[Document],file_path:Path) -> List[Document]:
        for i , doc in enumerate(documents):
            doc.metadata.update({
                'source_file': file_path.name,
                'file_path': str(file_path.absolute()),
                'file_type': file_path.suffix.lower(),
                'loaded_at': datetime.now().isoformat(),
                'chunk_index': i,
                'total_chunks': len(documents)
            })

        return documents
    
def load_documents(path: str | Path) -> Dict[str, List[Document]] | List[Document]:
    loader = DocumentLoader()
    path = Path(path)
    
    if path.is_file():
        return loader.load_document(path)
    elif path.is_dir():
        return loader.load_directory(path)
    else:
        raise ValueError(f"Invalid path: {path}")