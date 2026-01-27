"""
AI Assistant Pro - Main Flask Application
Modern RAG system with Grok AI integration
"""

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import logging
import os

from config import Config
from routes.chat_routes import chat_bp
from routes.file_routes import file_bp
from routes.analysis_routes import analysis_bp
from database.db_manager import init_database
from utils.helpers import cleanup_old_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

def create_app():
    """Application factory pattern"""
    app = Flask(__name__, static_folder=None)
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(file_bp, url_prefix='/api/files')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
    
    # Serve frontend HTML
    @app.route('/')
    def index():
        logger.info(f"Serving index from: {FRONTEND_DIR}")
        try:
            return send_from_directory(FRONTEND_DIR, 'index1.html')
        except Exception as e:
            logger.error(f"Error: {e}")
            return jsonify({'error': str(e), 'frontend_dir': FRONTEND_DIR}), 404
    
    # Serve CSS files
    @app.route('/styles.css')
    @app.route('/<path:filename>.css')
    def serve_css(filename='styles'):
        return send_from_directory(FRONTEND_DIR, f'{filename}.css')
    
    # Serve JS files
    @app.route('/script.js')
    @app.route('/<path:filename>.js')
    def serve_js(filename='script'):
        return send_from_directory(FRONTEND_DIR, f'{filename}.js')
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        from services.rag_service import has_documents
        return jsonify({
            'status': 'healthy',
            'grok_api': bool(Config.GROK_API_KEY),
            'database': os.path.exists(Config.DB_PATH),
            'has_documents': has_documents(),
            'version': '2.0.0',
            'frontend_path': FRONTEND_DIR,
            'frontend_exists': os.path.exists(FRONTEND_DIR)
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        logger.warning(f"404 error: {error}")
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {str(error)}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(413)
    def too_large(error):
        return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413
    
    return app

def initialize_app():
    """Initialize application components"""
    logger.info("🚀 Starting AI Assistant Pro...")
    
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.PLOTS_FOLDER, exist_ok=True)
    os.makedirs(Config.VECTOR_DB_DIR, exist_ok=True)
    
    # Initialize database
    init_database()
    logger.info("✓ Database initialized")
    
    # Cleanup old files
    cleanup_old_files()
    logger.info("✓ Cleanup completed")
    
    # Check API configuration
    if Config.GROK_API_KEY:
        logger.info("✓ Grok API configured")
    else:
        logger.warning("⚠️ GROK_API_KEY not set")
    
    logger.info(f"📁 Upload folder: {Config.UPLOAD_FOLDER}")
    logger.info(f"📊 Plots folder: {Config.PLOTS_FOLDER}")
    logger.info(f"💾 Database: {Config.DB_PATH}")
    logger.info(f"🌐 Frontend path: {FRONTEND_DIR}")
    logger.info(f"📄 index1.html exists: {os.path.exists(os.path.join(FRONTEND_DIR, 'index1.html'))}")

if __name__ == '__main__':
    # Initialize
    initialize_app()
    
    # Create app
    app = create_app()
    
    # Run
    logger.info("=" * 60)
    logger.info("🚀 Server running at: http://localhost:5000")
    logger.info("📊 Health check: http://localhost:5000/health")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )