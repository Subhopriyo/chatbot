"""
File upload and management routes - FIXED VERSION
"""
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import logging

from config import Config
from services.pdf_service import process_document
from services.chart_service import excel_data_store, process_data_file
from services.rag_service import (
    index_document, 
    clear_vector_database, 
    clear_conversation_documents,
    has_documents_for_conversation,
    get_conversation_info
)

logger = logging.getLogger(__name__)

file_bp = Blueprint('files', __name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# ==================== FILE UPLOAD ====================

@file_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process files - FIXED VERSION"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        # FIXED: Get conversation_id from form data
        conversation_id = request.form.get('conversation_id')
        if conversation_id:
            conversation_id = int(conversation_id)
            logger.info(f"📎 Uploading file to conversation {conversation_id}")
        else:
            logger.warning("⚠️ No conversation_id provided, uploading to global context")
        
        filename = secure_filename(file.filename)
        filepath = Config.UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        logger.info(f"📁 File saved: {filepath}")
        
        ext = filename.rsplit('.', 1)[1].lower()
        
        # Process data files (Excel/CSV)
        if ext in ['xls', 'xlsx', 'csv']:
            result = process_data_file(filepath, filename)
            
            # FIXED: Handle NaN values in result before sending JSON
            import json
            import math
            
            def clean_nan(obj):
                """Recursively replace NaN with None for JSON serialization"""
                if isinstance(obj, dict):
                    return {k: clean_nan(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan(item) for item in obj]
                elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                else:
                    return obj
            
            result = clean_nan(result)
            
            # Add conversation info to result
            if conversation_id:
                result['conversation_id'] = conversation_id
            
            return jsonify(result)
        
        # Process document files (PDF, DOCX, TXT)
        else:
            # FIXED: Only clear documents for THIS conversation, not all
            if conversation_id:
                logger.info(f"🗑️ Clearing old documents for conversation {conversation_id}")
                clear_conversation_documents(conversation_id)
            else:
                # If no conversation provided, clear all (legacy behavior)
                logger.info("🗑️ No conversation_id - clearing all documents")
                clear_vector_database()
            
            # Process document
            pages, error = process_document(filepath, filename)
            if error:
                return jsonify({'error': error}), 500
            
            # FIXED: Index document with conversation_id
            chunks = index_document(pages, filename, conversation_id)
            
            response_data = {
                'message': f'Document uploaded and processed. {len(pages)} pages into {len(chunks)} chunks.',
                'pages': len(pages),
                'chunks': len(chunks),
                'file_type': ext,
                'filename': filename,
                'mode': 'document_uploaded'
            }
            
            # Add conversation info
            if conversation_id:
                response_data['conversation_id'] = conversation_id
                response_data['message'] += f' Linked to conversation {conversation_id}.'
            
            return jsonify(response_data)
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@file_bp.route('/plots/<filename>')
def serve_plot(filename):
    """Serve generated plot images"""
    try:
        return send_from_directory(str(Config.PLOTS_FOLDER), filename)
    except Exception as e:
        logger.error(f"Error serving plot: {str(e)}")
        return jsonify({'error': 'Plot not found'}), 404

@file_bp.route('/clear', methods=['POST'])
def clear_files():
    """Clear all uploaded files - FIXED VERSION"""
    try:
        data = request.get_json() or {}
        conversation_id = data.get('conversation_id')
        
        if conversation_id:
            # FIXED: Clear only for specific conversation
            logger.info(f"🧹 Clearing files for conversation {conversation_id}")
            clear_conversation_documents(conversation_id)
            
            return jsonify({
                'message': f'Files cleared for conversation {conversation_id}',
                'conversation_id': conversation_id
            })
        else:
            # Clear everything (legacy behavior)
            logger.info("🧹 Clearing all files globally")
            
            # Clear vector database
            clear_vector_database()
            
            # Clear data store
            excel_data_store.clear()
            
            # Clear upload folder
            for filename in os.listdir(Config.UPLOAD_FOLDER):
                file_path = Config.UPLOAD_FOLDER / filename
                if file_path.is_file():
                    file_path.unlink()
            
            # Clear plots
            for filename in os.listdir(Config.PLOTS_FOLDER):
                file_path = Config.PLOTS_FOLDER / filename
                if file_path.is_file():
                    file_path.unlink()
            
            return jsonify({'message': 'All files cleared successfully'})
        
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/info', methods=['GET'])
def get_file_info():
    """Get information about uploaded files - FIXED VERSION"""
    try:
        # FIXED: Support conversation-specific queries
        conversation_id = request.args.get('conversation_id')
        if conversation_id:
            conversation_id = int(conversation_id)
        
        from services.rag_service import has_documents
        
        response_data = {
            'has_data': len(excel_data_store) > 0,
            'data_files': list(excel_data_store.keys()),
            'total_data_files': len(excel_data_store)
        }
        
        # Add conversation-specific info if conversation_id provided
        if conversation_id:
            conv_info = get_conversation_info(conversation_id)
            response_data.update({
                'conversation_id': conversation_id,
                'has_documents': conv_info['has_documents'],
                'document_list': conv_info['document_list'],
                'document_count': conv_info['document_count']
            })
        else:
            # Global info (legacy)
            response_data['has_documents'] = has_documents()
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_bp.route('/conversation/<int:conversation_id>/info', methods=['GET'])
def get_conversation_file_info(conversation_id):
    """Get file information for a specific conversation"""
    try:
        conv_info = get_conversation_info(conversation_id)
        
        return jsonify({
            'conversation_id': conversation_id,
            'has_documents': conv_info['has_documents'],
            'documents': conv_info['document_list'],
            'document_count': conv_info['document_count'],
            'has_data': len(excel_data_store) > 0,
            'data_files': list(excel_data_store.keys())
        })
        
    except Exception as e:
        logger.error(f"Error getting conversation file info: {str(e)}")
        return jsonify({'error': str(e)}), 500