from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from document_qa_system.config import Config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(
        self,
        embedding_model_name: str = Config.EMBEDDING_MODEL,
        persist_directory: str | Path = Config.CHROMA_PERSIST_DIR,
        collection_name: str = "Langchain"
    ):
        self.embedding_model_name = embedding_model_name
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing VectorStoreManager...")
        logger.info(f"Embedding model: {embedding_model_name}")
        logger.info(f"Persist directory: {self.persist_directory}")
        logger.info(f"Collection: {collection_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  
            encode_kwargs={'normalize_embeddings': True} 
        )

        logger.info("Embeddings model loaded")

        self.vector_store: Optional[Chroma] = None

        self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self) -> None:
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )

            collection = self.vector_store._collection
            count = collection.count()

            if count > 0:
                logger.info(f"Loaded existing vector store with {count} documents")
            else:
                logger.info("Vector store exists but is empty")

        except Exception as e:
            logger.info(f"Creating new vector store (no existing store found)")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )

    def add_documents(
        self, documents: List[Document],batch_size: int = 100) -> Dict[str, any]:
        if not documents:
            logger.warning(" No documents provided to add")
            return {'success': False, 'error': 'No documents'}
        
        logger.info(f"Adding {len(documents)} documents to vector store...")

        start_time = datetime.now()

        try:
            initial_count = self.get_document_count()

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                
                logger.info(f"   Processing batch {batch_num}/{total_batches}...")
                
                self.vector_store.add_documents(batch)

            final_count = self.get_document_count()
            added_count = final_count - initial_count

            duration = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Successfully added {added_count} documents "
                f"in {duration:.2f} seconds"
            )

            return {
                'success': True,
                'documents_added': added_count,
                'total_documents': final_count,
                'duration_seconds': duration
            }
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def similarity_search(self,query: str,k: int = 4,filter_dict: Optional[Dict] = None) -> List[Document]:
        #filter_dict -> for metadata filtering
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            logger.info(f"Searching for: '{query}' (top {k} results)")

            if filter_dict:
                logger.info(f" Applying filters: {filter_dict}")
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )

            logger.info(f"Found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
        
    def similarity_search_with_score(
        self,query: str,k: int = 4,filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        #scores > 0.7 are good matches,higher = more similar

        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            logger.info(f"Searching with scores: '{query}' (top {k})")
            
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )

            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get('source_file', 'unknown')
                logger.info(f"Result {i}: {source} (score: {score:.3f})")

            return results
        
        except Exception as e:
            logger.error(f"Search with scores failed: {str(e)}")
            return []
        
    def get_document_count(self) -> int:
        if not self.vector_store:
            return 0
        
        try:
            return self.vector_store._collection.count()
        except:
            return 0
        
    def clear_vector_store(self) -> bool:
        try:
            logger.warning("Clearing vector store...")

            self.vector_store.delete_collection()

            self._load_or_create_vector_store()

            logger.info("Vector store cleared")
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")
            return False
        
    def get_statistics(self) -> Dict:
        if not self.vector_store:
            return {'error': 'Vector store not initialized'}
        
        try:
            doc_count = self.get_document_count()

            if doc_count > 0:
                sample = self.vector_store.similarity_search("test", k=min(100, doc_count))
                source_files = set(doc.metadata.get('source_file', 'unknown') for doc in sample)
            else:
                source_files = set()

            return {
                'total_documents': doc_count,
                'unique_source_files': len(source_files),
                'source_files': list(source_files),
                'embedding_model': self.embedding_model_name,
                'persist_directory': str(self.persist_directory),
                'collection_name': self.collection_name
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {'error': str(e)}
        
def create_vector_store(
    documents: List[Document],
    embedding_model: str = Config.EMBEDDING_MODEL,
    persist_directory: str | Path = Config.CHROMA_PERSIST_DIR
) -> VectorStoreManager:
    
    manager = VectorStoreManager(
        embedding_model_name=embedding_model,
        persist_directory=persist_directory
    )

    manager.add_documents(documents)

    return manager

def search_documents(
    query: str,k: int = 4,persist_directory: str | Path = Config.CHROMA_PERSIST_DIR
) -> List[Document]:
    manager = VectorStoreManager(persist_directory=persist_directory)
    return manager.similarity_search(query, k=k)

        