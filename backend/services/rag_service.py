"""
RAG (Retrieval Augmented Generation) service - FIXED VERSION
Handles vector database with conversation-specific document isolation
"""
import os
import shutil
import logging
from typing import List, Tuple, Optional, Dict

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import Config

logger = logging.getLogger(__name__)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

# FIXED: Track which conversation has documents
conversation_documents = {}  # {conversation_id: [doc_ids]}

def get_vector_db():
    """Get vector database instance"""
    return Chroma(
        persist_directory=str(Config.VECTOR_DB_DIR),
        embedding_function=embeddings
    )

def has_documents() -> bool:
    """DEPRECATED: Check if vector database has documents (global check)"""
    try:
        if not os.path.exists(Config.VECTOR_DB_DIR):
            return False
        db = get_vector_db()
        results = db.similarity_search("test", k=1)
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error checking documents: {str(e)}")
        return False

def has_documents_for_conversation(conversation_id: int) -> bool:
    """
    FIXED: Check if a specific conversation has documents
    """
    try:
        # Check in-memory tracking
        if conversation_id not in conversation_documents:
            return False
        
        doc_list = conversation_documents[conversation_id]
        return len(doc_list) > 0
    except Exception as e:
        logger.error(f"Error checking documents for conversation {conversation_id}: {str(e)}")
        return False

def clear_conversation_documents(conversation_id: int):
    """
    FIXED: Clear documents for a specific conversation
    For simplicity, we clear the entire vector DB when switching conversations
    In production, you'd want conversation-specific collections
    """
    try:
        logger.info(f"🗑️ Clearing documents for conversation {conversation_id}")
        
        # Clear the tracking
        if conversation_id in conversation_documents:
            del conversation_documents[conversation_id]
        
        # Clear all conversations from tracking (fresh start)
        conversation_documents.clear()
        
        # Clear the vector database
        if os.path.exists(Config.VECTOR_DB_DIR):
            shutil.rmtree(Config.VECTOR_DB_DIR)
            os.makedirs(Config.VECTOR_DB_DIR, exist_ok=True)
            logger.info("✓ Vector database cleared for new conversation")
        
        return True
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        return False

def clear_vector_database():
    """Clear the entire vector database"""
    try:
        if os.path.exists(Config.VECTOR_DB_DIR):
            shutil.rmtree(Config.VECTOR_DB_DIR)
            os.makedirs(Config.VECTOR_DB_DIR, exist_ok=True)
            conversation_documents.clear()
            logger.info("🗑️ Vector database completely cleared")
            return True
    except Exception as e:
        logger.error(f"Error clearing vector database: {str(e)}")
        return False

def index_document(pages: List[Document], filename: str, conversation_id: int = None) -> List[Document]:
    """
    FIXED: Index document pages into vector database with conversation tracking
    
    Args:
        pages: List of document pages
        filename: Source filename
        conversation_id: ID of the conversation this document belongs to
        
    Returns:
        List of text chunks created
    """
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = text_splitter.split_documents(pages)
        
        # FIXED: Add conversation_id to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            if conversation_id:
                chunk.metadata['conversation_id'] = conversation_id
        
        # Create/update vector database
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=str(Config.VECTOR_DB_DIR)
        )
        db.persist()
        
        # FIXED: Track documents for this conversation
        if conversation_id:
            if conversation_id not in conversation_documents:
                conversation_documents[conversation_id] = []
            conversation_documents[conversation_id].append(filename)
            logger.info(f"✓ Indexed {len(pages)} pages into {len(chunks)} chunks for conversation {conversation_id}")
        else:
            logger.info(f"✓ Indexed {len(pages)} pages into {len(chunks)} chunks (no conversation)")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        raise

def get_document_context(question: str, conversation_id: int = None, k: int = 5) -> Tuple[Optional[str], List[str]]:
    """
    FIXED: Retrieve relevant document context for a question from specific conversation
    
    Args:
        question: User question
        conversation_id: Filter by conversation ID
        k: Number of chunks to retrieve
        
    Returns:
        Tuple of (context_text, source_files_list)
    """
    try:
        # Check if this conversation has documents
        if conversation_id and not has_documents_for_conversation(conversation_id):
            logger.warning(f"No documents found for conversation {conversation_id}")
            return None, []
        
        # Ensure question is a string
        if not isinstance(question, str):
            question = str(question)
        
        db = get_vector_db()
        
        # Get more documents than needed, then filter by conversation
        docs = db.similarity_search(question, k=k*3)
        
        # FIXED: Filter by conversation_id if provided
        if conversation_id:
            filtered_docs = []
            for doc in docs:
                metadata = doc.metadata if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else {}
                doc_conv_id = metadata.get('conversation_id')
                
                # Include doc if it belongs to this conversation or has no conversation ID (legacy)
                if doc_conv_id is None or doc_conv_id == conversation_id:
                    filtered_docs.append(doc)
                    if len(filtered_docs) >= k:
                        break
            docs = filtered_docs
        else:
            docs = docs[:k]
        
        if not docs:
            logger.warning("No relevant documents found after filtering")
            return None, []
        
        # Build context with page references
        context_parts = []
        sources = set()
        
        for i, doc in enumerate(docs):
            metadata = doc.metadata if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else {}
            
            page_num = metadata.get('page', 'Unknown')
            source = metadata.get('source', 'Unknown')
            
            context_parts.append(f"--- Section {i+1} (Page {page_num}) ---")
            
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            context_parts.append(content.strip())
            
            if source != 'Unknown':
                sources.add(f"{source} (Page {page_num})")
        
        context = "\n\n".join(context_parts)
        sources_list = sorted(list(sources))
        
        logger.info(f"✓ Retrieved {len(docs)} chunks from {len(sources_list)} sources for conversation {conversation_id}")
        
        return context, sources_list
        
    except Exception as e:
        logger.error(f"Error getting document context: {str(e)}", exc_info=True)
        return None, []

def get_document_chunks(filename: str = None, conversation_id: int = None) -> List[Dict]:
    """
    FIXED: Get all chunks from the vector database, optionally filtered by conversation
    
    Args:
        filename: Optional filter by source filename
        conversation_id: Optional filter by conversation ID
        
    Returns:
        List of chunk dictionaries with metadata
    """
    try:
        db = get_vector_db()
        
        # Get all documents
        all_docs = db.similarity_search("", k=100)
        
        chunks = []
        for doc in all_docs:
            metadata = doc.metadata if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else {}
            
            # Filter by conversation if specified
            if conversation_id:
                doc_conv_id = metadata.get('conversation_id')
                if doc_conv_id and doc_conv_id != conversation_id:
                    continue
            
            # Filter by filename if specified
            if filename and metadata.get('source') != filename:
                continue
            
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            chunks.append({
                'content': content,
                'metadata': metadata,
                'page': metadata.get('page', 'Unknown'),
                'source': metadata.get('source', 'Unknown'),
                'conversation_id': metadata.get('conversation_id')
            })
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error getting document chunks: {str(e)}")
        return []

def search_in_documents(query: str, conversation_id: int = None, k: int = 10) -> List[Dict]:
    """
    FIXED: Search for specific content in documents, optionally filtered by conversation
    
    Args:
        query: Search query
        conversation_id: Optional filter by conversation ID
        k: Number of results
        
    Returns:
        List of matching chunks with scores
    """
    try:
        if not isinstance(query, str):
            query = str(query)
        
        db = get_vector_db()
        results = db.similarity_search_with_score(query, k=k*2)
        
        matches = []
        for doc, score in results:
            metadata = doc.metadata if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else {}
            
            # Filter by conversation if specified
            if conversation_id:
                doc_conv_id = metadata.get('conversation_id')
                if doc_conv_id and doc_conv_id != conversation_id:
                    continue
            
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            matches.append({
                'content': content,
                'score': float(score),
                'page': metadata.get('page', 'Unknown'),
                'source': metadata.get('source', 'Unknown'),
                'conversation_id': metadata.get('conversation_id'),
                'metadata': metadata
            })
            
            if len(matches) >= k:
                break
        
        return matches
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}", exc_info=True)
        return []

def get_conversation_info(conversation_id: int) -> Dict:
    """
    Get information about documents in a conversation
    """
    try:
        return {
            'conversation_id': conversation_id,
            'has_documents': has_documents_for_conversation(conversation_id),
            'document_list': conversation_documents.get(conversation_id, []),
            'document_count': len(conversation_documents.get(conversation_id, []))
        }
    except Exception as e:
        logger.error(f"Error getting conversation info: {str(e)}")
        return {
            'conversation_id': conversation_id,
            'has_documents': False,
            'document_list': [],
            'document_count': 0
        }