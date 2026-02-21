from typing import List,Dict,Optional
import logging 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from document_qa_system.config import Config

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(
            self,
            chunk_size : int = Config.CHUNK_SIZE,
            chunk_overlap : int = Config.CHUNK_OVERLAP,
            separators: Optional[List[str]] = None
    ):

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

        self.separators = separators or [
            "\n\n", # Paragraph breaks 
            "\n", # Line break
            ". ", # Sentence break
            "! ", "? ","; ",", "," ","" # Character level break
        ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=self.separators,
            length_function = len,
            is_separator_regex=False
        )

        logger.info(
            f"TextProcessor initialized: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def process_documents(
            self,
            documents : list[Document],
            add_chunk_metadata : bool = True
    ) -> List[Document]:
         if not documents:
             logger.warning("No documents provided for processing")
             return []
         
         logger.info(f"Processing {len(documents)} document(s)...")

         chunks = self.splitter.split_documents(documents)

         if add_chunk_metadata:
             chunks = self._enhance_chunk_metadata(chunks)

         stats = self._calculate_statistics(chunks)

         logger.info(
            f"Created {len(chunks)} chunks | "
            f"Avg size: {stats['avg_chunk_size']:.0f} chars | "
            f"Min: {stats['min_chunk_size']} | Max: {stats['max_chunk_size']}"
        )
         return chunks
    
    def process_single_document(
        self,
        document: Document,
        add_chunk_metadata: bool = True
    ) -> List[Document]:
        return self.process_documents([document], add_chunk_metadata)
    
    def _enhance_chunk_metadata(self,chunks:List[Document]) -> List[Document]:
        for id,chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = id
            chunk.metadata['chunk_size'] = len(chunk.page_content)

            word_count = len(chunk.page_content.split())
            chunk.metadata['word_count'] = word_count
            chunk.metadata['reading_time_seconds'] = int((word_count/200)*60)

            preview = chunk.page_content[:100].replace('\n', ' ')
            chunk.metadata['content_preview'] = preview + "..."
        
        return chunks
    
    def _calculate_statistics(self,chunks: List[Document]) -> Dict[str,float]:
        if not chunks:
            return{
                'total_chunks' : 0,
                'avg_chunk_size' : 0,
                'min_chunk_size' : 0,
                'max_chunk_size' : 0,
                'total_characters':0
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]

        return{
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }

    def analyze_chunk_quality(self,chunks: List[Document]) -> Dict:
        if not chunks:
            return {'error' : 'No chunk to analyze'}
        
        stats = self._calculate_statistics(chunks)

        too_small = [
            c for c in chunks
            if len(c.page_content) < self.chunk_size * 0.3
        ]

        too_large = [
            c for c in chunks 
            if len(c.page_content) > self.chunk_size * 1.5
        ]

        low_context_chunks = []
        for i in range(1,len(chunks)):
            content = chunks[i].page_content
            if content and content[0].islower():
                low_context_chunks.append(i)

        analysis = {
            'total_chunks':stats['total_chunks'],
            'average_size':int(stats['avg_chunk_size']),
            'size_range':(stats['min_chunk_size'],stats['max_chunk_size']),
            'quality_metrics':{
                'chunks_too_small': len(too_small),
                'chunks_too_large':len(too_large),
                'chunks_with_low_context': len(low_context_chunks),
            },
            'recommendations':[]
        }

        if len(too_small) > len(chunks) * 0.1:  
            analysis['recommendations'].append(
                f"{len(too_small)} chunks are very small. "
                "Consider reducing chunk_size or adjusting separators."
            )

        if len(too_large) > len(chunks) * 0.1:  
            analysis['recommendations'].append(
                f"{len(too_large)} chunks are very large. "
                "Consider increasing chunk_size or reviewing document structure."
            )

        if len(low_context_chunks) > len(chunks) * 0.15: 
            analysis['recommendations'].append(
                f"{len(low_context_chunks)} chunks start mid-sentence. "
                "Consider increasing chunk_overlap for better context."
            )

        if not analysis['recommendations']:
            analysis['recommendations'].append("Chunk quality looks good!")

        return analysis
        

    def chunk_documents(
            documents : List[Document],
            chunk_size: int = Config.CHUNK_SIZE,
            chunk_overlap : int = Config.CHUNK_OVERLAP
    )->List[Document]:
         processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
         return processor.process_documents(documents)
        
        
            