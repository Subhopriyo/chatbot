"""
Configuration settings for AI Assistant Pro
"""
import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    PLOTS_FOLDER = BASE_DIR / 'static' / 'plots'
    VECTOR_DB_DIR = BASE_DIR / 'db_miniLM'
    DB_PATH = BASE_DIR / 'chat_history.db'
    
    # API Configuration - Groq (Free and Fast!)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("GROK_API_KEY"))  # Fallback to GROK_API_KEY
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Backward compatibility - keep old names working
    GROK_API_KEY = GROQ_API_KEY
    GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROK_MODEL = GROQ_MODEL
    
    # File Upload Settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'pptx', 'ppt', 'xlsx', 'xls', 'csv'}
    
    # RAG Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Chart Settings
    CHART_DPI = 300
    CHART_STYLE = 'dark_background'
    
    # Cleanup Settings
    FILE_MAX_AGE = 24 * 60 * 60  # 24 hours in seconds
    
    # Keywords for intent detection
    GRAPH_KEYWORDS = [
        'graph', 'chart', 'plot', 'visualize', 'visualization',
        'bar', 'line', 'pie', 'scatter', 'histogram', 'show', 'draw'
    ]
    
    DOCUMENT_KEYWORDS = [
        'summarise', 'summarize', 'summary', 'document', 'content',
        'explain', 'analyze', 'findings'
    ]
    
    # Action button types
    ACTION_TYPES = {
        'summarize': 'Create a concise summary',
        'notes': 'Generate structured notes',
        'mcq': 'Create multiple choice questions',
        'flashcards': 'Generate flashcards',
        'key_points': 'Extract key points',
        'expand': 'Provide detailed explanation'
    }

# Create directories on import
for directory in [Config.UPLOAD_FOLDER, Config.PLOTS_FOLDER, Config.VECTOR_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)