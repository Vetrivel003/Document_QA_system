from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from document_qa_system.config import Config
from document_qa_system.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(
        self,vector_store_manager: VectorStoreManager,
        model_name: str = Config.GROQ_MODEL,
        temperature: float = 0.1,
        k_documents: int = 4
    ):
        self.vector_store_manager = vector_store_manager
        self.k_documents = k_documents
        self.model_name = model_name 

        logger.info("    Initializing RAG Chain with pure LCEL...")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Temperature: {temperature}")
        logger.info(f"   Documents to retrieve: {k_documents}")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=Config.GROQ_API_KEY
        )

        self.retriever = vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": k_documents}
        )

        self.prompt = self._create_prompt()

        self.chain = self._build_rag_chain()

        logger.info("RAG Chain initialized successfully")

    def _create_prompt(self) -> ChatPromptTemplate:
        template = """You are a helpful AI assistant answering questions based on provided documents.

        Context from documents:
        {context}

        Question: {question}

        Instructions:
        1. Answer the question using ONLY the information from the context above
        2. If the context doesn't contain enough information, say "I don't have enough information to answer this question based on the provided documents."
        3. Be specific and cite which document you're referencing when possible
        4. Keep your answer clear and concise
        5. Do not make up information not present in the context

        Answer:"""
        
        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            content = doc.page_content
            formatted.append(f"[Source {i}: {source}]\n{content}\n")

        return "\n".join(formatted)
    
    def _build_rag_chain(self):
        rag_chain = (
            RunnableParallel(
                context=self.retriever | self._format_docs,
                question=RunnablePassthrough()
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain
    
    def query(
        self, question: str,return_sources: bool = True) -> Dict[str, any]:

        if not question or not question.strip():
            return {
                'success': False,
                'error': 'Empty question provided'
            }
        
        logger.info(f"Query: '{question}'")
        
        start_time = datetime.now()

        try:
            if return_sources:
                retrieved_docs = self.retriever.invoke(question)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")

            answer = self.chain.invoke(question)

            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f" Answer generated in {duration:.2f} seconds")

            response = {
                'success': True,
                'question': question,
                'answer': answer,
                'processing_time': duration,
                'model': self.model_name
            }

            if return_sources:
                sources = self._format_sources(retrieved_docs)
                response['sources'] = sources
                response['num_sources'] = len(sources)
            
            return response
        
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'question': question,
                'error': str(e)
            }
        
    def stream_query(self, question: str):
        if not question or not question.strip():
            yield "Error: Empty question provided"
            return
        
        logger.info(f"Streaming query: '{question}'")
        
        try:
            for chunk in self.chain.stream(question):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            yield f"Error: {str(e)}"

    def batch_query(
        self, questions: List[str],return_sources: bool = False
    ) -> List[Dict[str, any]]:
        
        logger.info(f"Processing batch of {len(questions)} questions...")
        
        try:
            answers = self.chain.batch(questions)
        
            results = []
            for question, answer in zip(questions, answers):
                result = {
                    'success': True,
                    'question': question,
                    'answer': answer,
                    'model': self.model_name
                }
                
                if return_sources:
                    docs = self.retriever.invoke(question)
                    result['sources'] = self._format_sources(docs)
                    result['num_sources'] = len(docs)
                
                results.append(result)
            
            logger.info(f"Batch processing complete")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return [
                {
                    'success': False,
                    'question': q,
                    'error': str(e)
                }
                for q in questions
            ]
        
    def _format_sources(self, documents: List[Document]) -> List[Dict]:
        sources = []
        
        for i, doc in enumerate(documents, 1):
            source_info = {
                'index': i,
                'file': doc.metadata.get('source_file', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                'preview': doc.page_content[:200].replace('\n', ' ') + "...",
                'full_content': doc.page_content
            }
            
            if 'page' in doc.metadata:
                source_info['page'] = doc.metadata['page']
            
            sources.append(source_info)
        
        return sources
    
    def get_chain_info(self) -> Dict:
        return {
            'model': self.model_name,
            'k_documents': self.k_documents,
            'vector_store_documents': self.vector_store_manager.get_document_count(),
            'embedding_model': self.vector_store_manager.embedding_model_name,
            'chain_type': 'LCEL (LangChain Expression Language)'
        }
    
    def inspect_chain(self):
        print("\nRAG Chain Structure:")
        print("=" * 60)
        print(self.chain)
        print("=" * 60)

def create_rag_chain(
    vector_store_manager: VectorStoreManager,
    model_name: str = Config.GROQ_MODEL,
    k_documents: int = 4
) -> RAGChain:
   
    return RAGChain(
        vector_store_manager=vector_store_manager,
        model_name=model_name,
        k_documents=k_documents
    )


            