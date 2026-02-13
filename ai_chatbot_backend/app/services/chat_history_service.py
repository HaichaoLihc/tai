import logging
from typing import List, Optional
from app.core.mongodb_client import get_mongodb_client
from app.core.models.chat_completion import Message

logger = logging.getLogger(__name__)

class ChatHistoryService:
    """Service for interacting with chat history in MongoDB"""
    
    def __init__(self):
        self.mongodb_client = get_mongodb_client()
        self.database_name = "main"
        self.collection_name = "chats"
        
    async def get_messages_by_sid(self, sid: str) -> List[Message]:
        """
        Retrieve messages for a given chat session ID (SID).
        
        Args:
            sid: The session ID (maps to 'id' field in MongoDB)
            
        Returns:
            List of Message objects
        """
        if not self.mongodb_client.connect():
            logger.error("❌ Failed to connect to MongoDB for chat history retrieval")
            return []
            
        try:
            collection = self.mongodb_client.get_collection(self.database_name, self.collection_name)
            if collection is None:
                logger.error(f"❌ Collection {self.database_name}.{self.collection_name} not found")
                return []
                
            # Query by the 'id' field as shown in the user's schema
            chat_doc = collection.find_one({"id": sid})
            
            if not chat_doc:
                logger.warning(f"⚠️ No chat history found for SID: {sid}")
                return []
                
            messages_data = chat_doc.get("messages", [])
            messages = []
            
            for msg in messages_data:
                print("role ---> ", msg.get("role"))
                role = msg.get("role")
                content = msg.get("content")
                if role and content:
                    messages.append(Message(role=role, content=content))
                elif isinstance(msg.get("content"), list):
                    # Handle cases where content is a list (e.g., multimodal)
                    # For now, we'll just extract text parts or simplify
                    text_content = ""
                    for part in msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_content += part.get("text", "")
                    if text_content:
                       messages.append(Message(role=role, content=text_content))
            
            logger.info(f"✅ Successfully retrieved {len(messages)} messages for SID: {sid}")
            return messages
            
        except Exception as e:
            logger.error(f"❌ Error retrieving chat history for SID {sid}: {e}")
            return []

# Global instance
chat_history_service = ChatHistoryService()
