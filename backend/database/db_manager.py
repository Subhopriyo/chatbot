"""
Database manager for chat history and conversations
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

from config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Config.DB_PATH)
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor
    
    # ==================== CONVERSATION MANAGEMENT ====================
    
    def create_conversation(self, title: str = "New Chat") -> int:
        """Create a new conversation"""
        query = """
            INSERT INTO conversations (title, created_at, updated_at)
            VALUES (?, ?, ?)
        """
        timestamp = datetime.now().isoformat()
        cursor = self.execute_query(query, (title, timestamp, timestamp))
        logger.info(f"✓ Created conversation: {title}")
        return cursor.lastrowid
    
    def get_all_conversations(self) -> List[Dict]:
        """Get all conversations"""
        query = """
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.deleted = 0
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Get a specific conversation"""
        query = "SELECT * FROM conversations WHERE id = ? AND deleted = 0"
        with self.get_connection() as conn:
            cursor = conn.execute(query, (conversation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_conversation_title(self, conversation_id: int, title: str):
        """Update conversation title"""
        query = """
            UPDATE conversations 
            SET title = ?, updated_at = ?
            WHERE id = ?
        """
        self.execute_query(query, (title, datetime.now().isoformat(), conversation_id))
        logger.info(f"✓ Updated conversation {conversation_id} title to: {title}")
    
    def delete_conversation(self, conversation_id: int):
        """Soft delete a conversation"""
        query = "UPDATE conversations SET deleted = 1 WHERE id = ?"
        self.execute_query(query, (conversation_id,))
        logger.info(f"🗑️ Deleted conversation: {conversation_id}")
    
    def search_conversations(self, search_term: str) -> List[Dict]:
        """Search conversations by title or content"""
        query = """
            SELECT DISTINCT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.deleted = 0 AND (
                c.title LIKE ? OR
                m.content LIKE ?
            )
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """
        search_pattern = f"%{search_term}%"
        with self.get_connection() as conn:
            cursor = conn.execute(query, (search_pattern, search_pattern))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== MESSAGE MANAGEMENT ====================
    
    def add_message(self, conversation_id: int, role: str, content: str, 
                   metadata: Dict = None) -> int:
        """Add a message to a conversation"""
        query = """
            INSERT INTO messages (conversation_id, role, content, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.execute_query(
            query, 
            (conversation_id, role, content, metadata_json, timestamp)
        )
        
        # Update conversation timestamp
        self.execute_query(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id)
        )
        
        return cursor.lastrowid
    
    def get_messages(self, conversation_id: int) -> List[Dict]:
        """Get all messages in a conversation"""
        query = """
            SELECT * FROM messages 
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (conversation_id,))
            messages = []
            for row in cursor.fetchall():
                msg = dict(row)
                if msg['metadata']:
                    msg['metadata'] = json.loads(msg['metadata'])
                messages.append(msg)
            return messages
    
    def delete_message(self, message_id: int):
        """Delete a specific message"""
        query = "DELETE FROM messages WHERE id = ?"
        self.execute_query(query, (message_id,))
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}
            
            # Total conversations
            cursor = conn.execute("SELECT COUNT(*) FROM conversations WHERE deleted = 0")
            stats['total_conversations'] = cursor.fetchone()[0]
            
            # Total messages
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            stats['total_messages'] = cursor.fetchone()[0]
            
            # Most active conversation
            cursor = conn.execute("""
                SELECT c.title, COUNT(m.id) as msg_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.deleted = 0
                GROUP BY c.id
                ORDER BY msg_count DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                stats['most_active'] = {'title': row[0], 'messages': row[1]}
            
            return stats

# Global database instance
db = DatabaseManager()

def init_database():
    """Initialize database with schema"""
    schema = """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted INTEGER DEFAULT 0
    );
    
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_conversation_id ON messages(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp);
    CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
    """
    
    with sqlite3.connect(str(Config.DB_PATH)) as conn:
        conn.executescript(schema)
        logger.info("✓ Database schema initialized")