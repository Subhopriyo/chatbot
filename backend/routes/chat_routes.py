"""
Chat and conversation management routes - FIXED VERSION
"""
from flask import Blueprint, request, jsonify
import logging
import json
import re

from database.db_manager import db
from services.groq_service import grok
from services.rag_service import get_document_context, has_documents_for_conversation, clear_conversation_documents
from services.chart_service import excel_data_store, generate_smart_chart
from config import Config

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)

# Greeting patterns - skip document search for these
GREETING_PATTERNS = [
    r'^(hi|hello|hey|hola|greetings?|good\s+(morning|afternoon|evening|day))[\s\W]*$',
    r'^(what\'?s?\s+up|wassup|sup)[\s\W]*$',
    r'^(how\s+(are|r)\s+you|how\s+do\s+you\s+do)[\s\W]*$',
    r'^(yo|howdy)[\s\W]*$',
]

# FIXED: More specific document keywords
DOCUMENT_SPECIFIC_KEYWORDS = [
    'document', 'pdf', 'file', 'upload', 'paper',
    'according to', 'in the document', 'in this file',
    'from the document', 'what does it say', 'summarize this',
    'analyze this', 'explain this document', 'in the text'
]

def is_greeting(message: str) -> bool:
    """Check if message is a casual greeting"""
    message_lower = message.lower().strip()
    
    # Check against patterns
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, message_lower, re.IGNORECASE):
            return True
    
    return False

def is_question_about_documents(message: str, conversation_id: int) -> bool:
    """
    FIXED: Check if message is specifically asking about uploaded documents
    Now requires explicit document context OR checks if docs exist for this conversation
    """
    message_lower = message.lower()
    
    # First check: Does this conversation have documents?
    if not has_documents_for_conversation(conversation_id):
        return False
    
    # Second check: Is the question explicitly about documents?
    has_doc_keyword = any(keyword in message_lower for keyword in DOCUMENT_SPECIFIC_KEYWORDS)
    
    if has_doc_keyword:
        return True
    
    # Third check: General questions should NOT use documents unless explicitly asked
    # Questions like "what's the capital of France" should NOT search documents
    general_question_patterns = [
        r'^what\s+is\s+(the\s+)?capital',
        r'^who\s+is\s+(the\s+)?(president|ceo|founder)',
        r'^when\s+(was|is|did)',
        r'^where\s+is',
        r'^how\s+do\s+(i|you)\s+(?!.*document)',  # "how do I" questions not about documents
        r'^define\s+\w+$',  # Single word definitions
        r'^what\s+does\s+\w+\s+mean$',  # Word meanings
    ]
    
    for pattern in general_question_patterns:
        if re.search(pattern, message_lower):
            logger.info(f"🌍 Detected general knowledge question - skipping document search")
            return False
    
    # If message has question words but no document keywords, it's likely general
    # Only use documents if explicitly mentioned or if it's clearly about document content
    return False

# ==================== CONVERSATION MANAGEMENT ====================

@chat_bp.route('/conversations', methods=['GET'])
def get_conversations():
    """Get all conversations"""
    try:
        conversations = db.get_all_conversations()
        return jsonify({
            'conversations': conversations,
            'total': len(conversations)
        })
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/conversations', methods=['POST'])
def create_conversation():
    """Create new conversation - FIXED: Clear old documents"""
    try:
        data = request.get_json()
        
        if isinstance(data, str):
            data = json.loads(data)
        
        title = data.get('title', 'New Chat')
        
        conversation_id = db.create_conversation(title)
        
        # FIXED: Clear documents when creating new conversation
        logger.info(f"🆕 Creating new conversation {conversation_id} - clearing old documents")
        clear_conversation_documents(conversation_id)
        
        conversation = db.get_conversation(conversation_id)
        
        return jsonify({
            'conversation': conversation,
            'message': 'New conversation created - ready for fresh documents'
        })
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/conversations/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get specific conversation with messages"""
    try:
        conversation = db.get_conversation(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        messages = db.get_messages(conversation_id)
        
        return jsonify({
            'conversation': conversation,
            'messages': messages
        })
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/conversations/<int:conversation_id>', methods=['PUT'])
def update_conversation(conversation_id):
    """Update conversation title"""
    try:
        data = request.get_json()
        
        if isinstance(data, str):
            data = json.loads(data)
        
        title = data.get('title')
        
        if not title:
            return jsonify({'error': 'Title required'}), 400
        
        db.update_conversation_title(conversation_id, title)
        conversation = db.get_conversation(conversation_id)
        
        return jsonify({
            'conversation': conversation,
            'message': 'Conversation updated'
        })
    except Exception as e:
        logger.error(f"Error updating conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/conversations/<int:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete conversation - FIXED: Also clear documents"""
    try:
        # FIXED: Clear documents associated with this conversation
        clear_conversation_documents(conversation_id)
        
        db.delete_conversation(conversation_id)
        return jsonify({'message': 'Conversation and associated documents deleted'})
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/conversations/search', methods=['GET'])
def search_conversations():
    """Search conversations"""
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({'conversations': [], 'total': 0})
        
        conversations = db.search_conversations(query)
        return jsonify({
            'conversations': conversations,
            'total': len(conversations),
            'query': query
        })
    except Exception as e:
        logger.error(f"Error searching conversations: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ==================== MESSAGE MANAGEMENT ====================

@chat_bp.route('/messages', methods=['POST'])
def send_message():
    """Send message and get AI response - FIXED VERSION"""
    try:
        data = request.get_json()
        
        if isinstance(data, str):
            data = json.loads(data)
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        conversation_id = data.get('conversation_id')
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message required'}), 400
        
        # Create conversation if not exists
        if not conversation_id:
            conversation_id = db.create_conversation()
            logger.info(f"🆕 Auto-created conversation {conversation_id}")
        
        logger.info(f"💬 Message in conversation {conversation_id}: {message[:100]}...")
        
        # Save user message
        db.add_message(conversation_id, 'user', message)
        
        # Check if it's a simple greeting
        if is_greeting(message):
            logger.info("🤝 Detected greeting - responding directly")
            greeting_response = grok.simple_prompt(message)
            
            if isinstance(greeting_response, dict):
                response_text = greeting_response.get('text', greeting_response.get('answer', str(greeting_response)))
            else:
                response_text = str(greeting_response)
            
            db.add_message(conversation_id, 'assistant', response_text)
            
            return jsonify({
                'conversation_id': conversation_id,
                'answer': response_text,
                'mode': 'greeting'
            })
        
        # Check for graph request
        is_graph_req = any(kw in message.lower() for kw in Config.GRAPH_KEYWORDS)
        
        if is_graph_req and excel_data_store:
            filename, df = next(iter(excel_data_store.items()))
            chart_url, chart_info = generate_smart_chart(df, filename, message)
            
            if chart_url:
                chart_response = grok.generate_chart_description({
                    'type': chart_info,
                    'columns': df.columns.tolist(),
                    'context': message
                })
                
                if isinstance(chart_response, dict):
                    response_text = chart_response.get('text', chart_response.get('answer', str(chart_response)))
                else:
                    response_text = str(chart_response)
                
                db.add_message(conversation_id, 'assistant', response_text, {
                    'chart_url': chart_url,
                    'chart_info': chart_info
                })
                
                return jsonify({
                    'conversation_id': conversation_id,
                    'answer': response_text,
                    'chart_url': chart_url,
                    'chart_info': chart_info,
                    'mode': 'chart_generated'
                })
        
        # FIXED: Check if question is specifically about documents
        if is_question_about_documents(message, conversation_id):
            logger.info("📄 Question is about documents - using RAG mode")
            context, sources = get_document_context(message, conversation_id, k=5)
            
            if context:
                rag_response = grok.rag_prompt(message, context, sources)
                
                if isinstance(rag_response, dict):
                    response_text = rag_response.get('text', rag_response.get('answer', str(rag_response)))
                else:
                    response_text = str(rag_response)
                
                confidence = grok.calculate_confidence(message, context)
                
                db.add_message(conversation_id, 'assistant', response_text, {
                    'sources': sources,
                    'confidence': confidence
                })
                
                return jsonify({
                    'conversation_id': conversation_id,
                    'answer': response_text,
                    'sources': sources,
                    'confidence': confidence,
                    'mode': 'rag'
                })
        
        # FIXED: General chat mode (for all general knowledge questions)
        logger.info("🌍 Using general knowledge mode (no document search)")
        general_response = grok.simple_prompt(message)
        
        if isinstance(general_response, dict):
            response_text = general_response.get('text', general_response.get('answer', str(general_response)))
        else:
            response_text = str(general_response)
        
        db.add_message(conversation_id, 'assistant', response_text)
        
        return jsonify({
            'conversation_id': conversation_id,
            'answer': response_text,
            'mode': 'general'
        })
        
    except AttributeError as e:
        logger.error(f"AttributeError in send_message: {str(e)}")
        return jsonify({'error': 'Invalid request format'}), 400
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/actions/<action_type>', methods=['POST'])
def execute_action(action_type):
    """Execute smart action on content"""
    try:
        data = request.get_json()
        
        if isinstance(data, str):
            data = json.loads(data)
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        content = data.get('content', '')
        conversation_id = data.get('conversation_id')
        
        if not content:
            return jsonify({'error': 'Content required'}), 400
        
        if action_type not in Config.ACTION_TYPES:
            return jsonify({'error': 'Invalid action type'}), 400
        
        action_response = grok.action_prompt(action_type, content)
        
        if isinstance(action_response, dict):
            result = action_response.get('text', action_response.get('answer', str(action_response)))
        else:
            result = str(action_response)
        
        if conversation_id:
            db.add_message(
                conversation_id, 
                'assistant', 
                result,
                {'action_type': action_type}
            )
        
        return jsonify({
            'action': action_type,
            'result': result,
            'description': Config.ACTION_TYPES[action_type]
        })
        
    except Exception as e:
        logger.error(f"Error executing action: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """Get chat statistics"""
    try:
        stats = db.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500