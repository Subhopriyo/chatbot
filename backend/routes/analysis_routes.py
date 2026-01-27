"""
Data analysis and suggestion routes
"""
from flask import Blueprint, request, jsonify
import logging

from services.chart_service import excel_data_store, get_numeric_columns
from services.groq_service import grok
from services.rag_service import has_documents, search_in_documents
from config import Config

logger = logging.getLogger(__name__)

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/data-info', methods=['GET'])
def get_data_info():
    """Get information about uploaded data"""
    if not excel_data_store:
        return jsonify({
            'has_data': False,
            'message': 'No data files uploaded'
        })
    
    data_info = {}
    for filename, df in excel_data_store.items():
        data_info[filename] = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'numeric_columns': get_numeric_columns(df),
            'data_types': df.dtypes.astype(str).to_dict(),
            'sample': df.head(2).to_dict()
        }
    
    return jsonify({
        'has_data': True,
        'files': data_info,
        'total_files': len(excel_data_store)
    })

@analysis_bp.route('/analyze-data', methods=['POST'])
def analyze_data():
    """Provide statistical analysis of uploaded data"""
    if not excel_data_store:
        return jsonify({'error': 'No data files uploaded'}), 400
    
    data = request.get_json()
    filename = data.get('filename')
    
    try:
        # Get dataframe
        if filename and filename in excel_data_store:
            df = excel_data_store[filename]
        else:
            filename, df = next(iter(excel_data_store.items()))
        
        # Basic analysis
        analysis = {
            'filename': filename,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'numeric_columns': get_numeric_columns(df),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Statistics for numeric columns
        numeric_cols = get_numeric_columns(df)
        if numeric_cols:
            analysis['statistics'] = df[numeric_cols].describe().to_dict()
        
        # Get AI insights
        df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'sample': df.head(5).to_string()
        }
        
        ai_analysis = grok.analyze_data(df_info, "Analyze this dataset")
        analysis['ai_insights'] = ai_analysis
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get smart question suggestions"""
    suggestions = []
    
    try:
        # Data-based suggestions
        if excel_data_store:
            for filename, df in excel_data_store.items():
                numeric_cols = get_numeric_columns(df)
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if numeric_cols and categorical_cols:
                    suggestions.extend([
                        f"Create a bar chart of {numeric_cols[0]} by {categorical_cols[0]}",
                        f"Show distribution of {numeric_cols[0]}",
                        f"Analyze trends in {filename}",
                        f"Compare {numeric_cols[0]} across different {categorical_cols[0]}"
                    ])
                elif numeric_cols:
                    suggestions.extend([
                        f"Show histogram of {numeric_cols[0]}",
                        f"What are the statistics for {numeric_cols[0]}?"
                    ])
        
        # Document-based suggestions
        if has_documents():
            suggestions.extend([
                "📄 Summarize the main points",
                "🔍 What are the key findings?",
                "📝 Extract important quotes",
                "❓ Create study questions",
                "📊 Analyze the conclusions"
            ])
        
        # General suggestions if no files
        if not excel_data_store and not has_documents():
            suggestions.extend([
                "Upload a document to get started",
                "Upload data for visualization",
                "Ask me anything!"
            ])
        
        return jsonify({
            'suggestions': suggestions[:8],
            'has_data': len(excel_data_store) > 0,
            'has_documents': has_documents()
        })
        
    except Exception as e:
        logger.error(f"Suggestions error: {str(e)}")
        return jsonify({
            'suggestions': ["Ask me anything!"],
            'error': str(e)
        })

@analysis_bp.route('/search-documents', methods=['POST'])
def search_documents():
    """Search within uploaded documents"""
    if not has_documents():
        return jsonify({'error': 'No documents uploaded'}), 400
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        results = search_in_documents(query, k=10)
        return jsonify({
            'query': query,
            'results': results,
            'total': len(results)
        })
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500