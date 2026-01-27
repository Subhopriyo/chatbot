"""
Groq AI service for intelligent responses
"""
import requests
import logging
from typing import List, Dict, Optional, Union

from config import Config

logger = logging.getLogger(__name__)

class GroqService:
    """Handles all Groq AI API interactions"""
    
    def __init__(self):
        # Use GROQ_API_KEY but fallback to GROK_API_KEY for backward compatibility
        self.api_key = getattr(Config, 'GROQ_API_KEY', None) or getattr(Config, 'GROK_API_KEY', None)
        # Groq API endpoint
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        # Default to llama model (fast and free)
        self.model = getattr(Config, 'GROQ_MODEL', 'llama-3.3-70b-versatile')
        
        # Validate configuration on init
        if not self.api_key:
            logger.warning("⚠️ GROQ_API_KEY not found in environment variables")
        else:
            logger.info(f"✓ Groq service initialized with model: {self.model}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Optional[str]:
        """
        Send chat completion request to Groq
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length
            stream: Enable streaming responses
        
        Returns:
            Response text or None on error
        """
        if not self.api_key:
            error_msg = "⚠️ Groq API key not configured. Please set GROQ_API_KEY in your .env file."
            logger.error(error_msg)
            return error_msg
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            logger.info(f"🤖 Calling Groq API with {len(messages)} messages")
            logger.debug(f"API URL: {self.api_url}")
            logger.debug(f"Model: {self.model}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Log response details for debugging
            logger.debug(f"Response Status: {response.status_code}")
            
            # Check for specific error codes
            if response.status_code == 400:
                try:
                    error_detail = response.json() if response.text else {}
                    logger.error(f"Bad Request (400): {error_detail}")
                    
                    # Handle both dict and string error responses
                    if isinstance(error_detail, dict):
                        error_msg = error_detail.get('error', 'No details')
                        if isinstance(error_msg, dict):
                            error_msg = error_msg.get('message', 'No details')
                        elif not isinstance(error_msg, str):
                            error_msg = str(error_msg)
                    else:
                        error_msg = str(error_detail)
                    
                    return f"⚠️ API Error: Invalid request. Please check your API key and model name. Details: {error_msg}"
                except Exception as e:
                    logger.error(f"Error parsing 400 response: {str(e)}")
                    return "⚠️ API Error: Invalid request. Please check your API key and model name."
            
            elif response.status_code == 401:
                logger.error("Unauthorized (401): Invalid API key")
                return "⚠️ Invalid API key. Please check your GROQ_API_KEY in .env file."
            
            elif response.status_code == 429:
                logger.error("Rate Limited (429)")
                return "⚠️ Rate limit exceeded. Please try again in a moment."
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                logger.info(f"✓ Groq response received ({len(content)} chars)")
                return content.strip()
            else:
                logger.error(f"No valid response from Groq. Response: {result}")
                return "⚠️ No response generated from Groq API."
                
        except requests.exceptions.Timeout:
            logger.error("Groq API timeout after 60 seconds")
            return "⚠️ Request timed out. The API took too long to respond. Please try again."
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error details: {error_detail}")
                    
                    # Safely extract error message
                    if isinstance(error_detail, dict):
                        error_msg = error_detail.get('error', str(e))
                        if isinstance(error_msg, dict):
                            error_msg = error_msg.get('message', str(e))
                    else:
                        error_msg = str(error_detail)
                    
                    return f"⚠️ Groq API Error: {error_msg}"
                except Exception as parse_error:
                    logger.error(f"Error parsing error response: {str(parse_error)}")
                    return f"⚠️ Error communicating with Groq: {str(e)}"
            return f"⚠️ Error communicating with Groq: {str(e)}"
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return f"⚠️ Unexpected error: {str(e)}"
    
    def simple_prompt(self, prompt: str, temperature: float = 0.7) -> str:
        """Simple single prompt completion"""
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages, temperature=temperature)
        return str(response) if response else "⚠️ No response generated"
    
    def rag_prompt(self, question: str, context: str, sources: Union[List[str], str, None] = None) -> str:
        """Generate response based on document context (RAG)"""
        # Handle sources properly
        sources_list = []
        if sources:
            if isinstance(sources, list):
                sources_list = sources
            elif isinstance(sources, str):
                sources_list = [sources]
            else:
                logger.warning(f"Unexpected sources type: {type(sources)}")
        
        prompt = f"""Based on the following document content, answer the question comprehensively and accurately.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Provide a detailed, accurate response using the document information. If the context doesn't contain enough information, acknowledge that."""

        if sources_list:
            prompt += f"\n\nSOURCES: {', '.join(sources_list)}"
        
        response = self.simple_prompt(prompt, temperature=0.3)
        return str(response) if response else "⚠️ No response generated"
    
    def action_prompt(self, action_type: str, content: str) -> str:
        """Execute smart action on content"""
        prompts = {
            'summarize': f"""Summarize the following content concisely, highlighting key points:

{content}

Provide a clear, structured summary.""",
            
            'notes': f"""Create structured study notes from the following content:

{content}

Format as organized notes with headings, bullet points, and key concepts.""",
            
            'mcq': f"""Create 5 multiple choice questions based on this content:

{content}

Format: Question, 4 options (A-D), and indicate the correct answer.""",
            
            'flashcards': f"""Create flashcards from this content:

{content}

Format: Front (question/term) | Back (answer/definition)""",
            
            'key_points': f"""Extract the key points from this content:

{content}

List the most important points in bullet format.""",
            
            'expand': f"""Provide a detailed explanation of this content:

{content}

Expand with examples, context, and thorough explanations."""
        }
        
        prompt = prompts.get(action_type, prompts['summarize'])
        response = self.simple_prompt(prompt, temperature=0.5)
        return str(response) if response else "⚠️ No response generated"
    
    def analyze_data(self, df_info: Dict, question: str) -> str:
        """Analyze data and answer questions"""
        prompt = f"""Analyze this dataset and answer the question:

Dataset Information:
- Shape: {df_info.get('shape', 'Unknown')}
- Columns: {', '.join(df_info.get('columns', []))}
- Sample Data: {df_info.get('sample', 'Not available')}

Question: {question}

Provide insights about the data and answer the question."""

        response = self.simple_prompt(prompt, temperature=0.5)
        return str(response) if response else "⚠️ No response generated"
    
    def generate_chart_description(self, chart_info: Dict) -> str:
        """Generate description for a chart"""
        prompt = f"""Describe and analyze this data visualization:

Chart Type: {chart_info.get('type', 'Unknown')}
Data Columns: {chart_info.get('columns', 'Unknown')}
Context: {chart_info.get('context', 'No additional context')}

Provide insights about what the visualization shows and any patterns or trends."""

        response = self.simple_prompt(prompt, temperature=0.5)
        return str(response) if response else "⚠️ No response generated"
    
    def calculate_confidence(self, question: str, context: str) -> Dict:
        """Calculate confidence score for an answer"""
        try:
            context = str(context) if context else ""
            question = str(question) if question else ""
            
            # Simple heuristic - can be improved with embeddings
            context_lower = context.lower()
            question_words = question.lower().split()
            
            # Count matching words
            matches = sum(1 for word in question_words if len(word) > 3 and word in context_lower)
            
            # Calculate confidence
            if len(question_words) > 0:
                confidence = min(100, int((matches / len(question_words)) * 100))
            else:
                confidence = 50
            
            # Determine level
            if confidence >= 80:
                level = "Strong"
                color = "success"
            elif confidence >= 50:
                level = "Medium"
                color = "warning"
            else:
                level = "Weak"
                color = "error"
            
            return {
                'score': confidence,
                'level': level,
                'color': color
            }
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return {
                'score': 0,
                'level': 'Weak',
                'color': 'error'
            }

# Global service instance - keeping old name for backward compatibility
groq = GroqService()
# Also export as grok for backward compatibility
grok = groq