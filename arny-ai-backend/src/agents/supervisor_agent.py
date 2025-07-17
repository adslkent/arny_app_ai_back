"""
Supervisor Agent Module - ENHANCED VERSION with Unified Conversation Context

This module provides a supervisor agent that routes requests to specialized agents
and handles general conversation with unified context management.

Key Features:
1. Ultra-fast keyword-based routing (no LLM calls for routing decisions)
2. Direct routing to specialized agents (flight/hotel)
3. Unified conversation context approach (50 messages)
4. Enhanced general conversation handling with conversation history
5. NO TIMEOUT LIMITS for better reliability

Usage example:
```python
from supervisor_agent import SupervisorAgent

# Create and use the agent
agent = SupervisorAgent()
result = await agent.process_message(user_id, "Find flights", session_id, {}, [])
```
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception_message,
    retry_any,
    retry_if_result,
    before_sleep_log,
    retry_if_exception
)
import requests

from ..utils.config import config
from .flight_agent import FlightAgent
from .hotel_agent import HotelAgent

# Configure logging
logger = logging.getLogger(__name__)

# ==================== OPENAI API RETRY CONDITIONS ====================

def retry_on_openai_api_error_result(result):
    """Condition 2: Retry if OpenAI API result contains error/warning fields"""
    if hasattr(result, 'error') and result.error:
        return True
    if isinstance(result, dict):
        return (
            result.get("error") is not None or
            "warning" in result or
            result.get("success") is False
        )
    return False

def retry_on_openai_http_status_error(result):
    """Condition 1: Retry on HTTP status errors for OpenAI API calls"""
    if hasattr(result, 'status_code'):
        return result.status_code >= 400
    if hasattr(result, 'response') and hasattr(result.response, 'status_code'):
        return result.response.status_code >= 400
    return False

def retry_on_openai_validation_failure(result):
    """Condition 5: Retry on validation failures"""
    return result is None or (isinstance(result, str) and len(result.strip()) == 0)

def retry_on_openai_api_exception(exception):
    """Condition 3: Check if exception is related to OpenAI API timeouts or rate limits"""
    exception_str = str(exception).lower()
    return (
        "timeout" in exception_str or
        "rate limit" in exception_str or
        "429" in exception_str or
        "502" in exception_str or
        "503" in exception_str or
        "504" in exception_str
    )

# ==================== RETRY DECORATORS ====================

openai_api_retry = retry(
    reraise=True,
    retry=retry_any(
        # Condition 3: OpenAI API exceptions (timeouts, rate limits, server errors)
        retry_if_exception_message(match=r".*(?i)(timeout|rate.limit|429|502|503|504).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_openai_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_openai_api_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_openai_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_openai_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

class SupervisorAgent:
    """
    ENHANCED: Supervisor agent with fast routing, NO TIMEOUT LIMITS, and unified conversation context
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize the specialized agents
        self.flight_agent = FlightAgent()
        self.hotel_agent = HotelAgent()
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                            user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ENHANCED: Process user message with ultra-fast routing and NO TIMEOUT LIMITS
        """
        
        try:
            print(f"🤖 SUPERVISOR: Processing message: '{message[:50]}...'")
            
            # OPTIMIZATION 1: Ultra-fast keyword-based routing (no LLM call)
            routing_decision = self._ultra_fast_routing(message)
            
            print(f"⚡ INSTANT routing decision: {routing_decision}")
            
            # OPTIMIZATION 2: Direct routing without delays
            if routing_decision == "flight_search":
                print("✈️ Routing to FlightAgent")
                return await self.flight_agent.process_message(
                    user_id, message, session_id, user_profile, conversation_history
                )
            elif routing_decision == "hotel_search":
                print("🏨 Routing to HotelAgent")
                return await self.hotel_agent.process_message(
                    user_id, message, session_id, user_profile, conversation_history
                )
            else:
                print("💬 Handling as general conversation")
                return await self._handle_general_conversation_no_timeout(
                    user_id, message, session_id, user_profile, conversation_history
                )
        
        except Exception as e:
            print(f"❌ Error in supervisor agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": "I'm sorry, I encountered an error processing your request. Please try again.",
                "agent_type": "supervisor",
                "requires_action": False,
                "error": str(e)
            }
    
    def _ultra_fast_routing(self, message: str) -> str:
        """
        ULTRA-OPTIMIZATION: Instant keyword-based routing (no LLM calls)
        """
        
        message_lower = message.lower()
        
        # Flight keywords - prioritized list
        flight_keywords = [
            'flight', 'flights', 'fly', 'flying', 'plane', 'airline', 'airport', 
            'departure', 'arrival', 'ticket', 'book flight', 'find flight',
            'airfare', 'air travel', 'aviation'
        ]
        
        # Hotel keywords - prioritized list  
        hotel_keywords = [
            'hotel', 'hotels', 'accommodation', 'stay', 'room', 'rooms',
            'check-in', 'check-out', 'booking', 'book hotel', 'find hotel',
            'resort', 'motel', 'inn', 'lodge', 'hostel'
        ]
        
        # Quick flight detection
        if any(keyword in message_lower for keyword in flight_keywords):
            return "flight_search"
        
        # Quick hotel detection
        elif any(keyword in message_lower for keyword in hotel_keywords):
            return "hotel_search"
        
        # Default to general conversation
        else:
            return "general"
    
    async def _handle_general_conversation_no_timeout(self, user_id: str, message: str, session_id: str,
                                                    user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ENHANCED: Fast general conversation with NO TIMEOUT LIMITS and unified conversation context
        """
        
        try:
            print(f"💬 Fast general conversation with unified context - NO TIMEOUT LIMITS")
            
            # Build minimal user context
            user_context = ""
            if user_profile.get("name"):
                user_context += f"User's name: {user_profile['name']}\n"
            if user_profile.get("city"):
                user_context += f"User's city: {user_profile['city']}\n"
            
            # UNIFIED CONTEXT: Build conversation context from history (last 50 messages)
            context_messages = []
            for msg in conversation_history[-50:]:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            print(f"🔧 Processing general conversation with {len(context_messages)} previous messages")
            
            # Ultra-short system prompt for efficiency
            system_prompt = f"""You are Arny, a helpful AI travel assistant. 

{user_context}

For specific searches, ask users to request "flights" or "hotels" with dates and destinations.
Keep responses friendly and brief."""

            # ENHANCED: Direct OpenAI call with conversation context and NO TIMEOUT LIMITS
            try:
                if context_messages:
                    # Include conversation history for context-aware responses
                    assistant_message = await self._make_openai_call_with_context(
                        system_prompt, message, context_messages
                    )
                else:
                    # First message, no history
                    assistant_message = await self._make_openai_call_no_timeout(
                        system_prompt, message
                    )
                    
            except Exception as openai_error:
                print(f"⚠️ OpenAI error: {openai_error}")
                assistant_message = "I'm here to help with your travel planning! You can ask me to search for flights or hotels with your travel dates."
            
            return {
                "message": assistant_message,
                "agent_type": "supervisor",
                "requires_action": False,
                "metadata": {
                    "agent_type": "supervisor",
                    "conversation_type": "general",
                    "context_messages_count": len(context_messages)
                }
            }
            
        except Exception as e:
            print(f"❌ Error in general conversation: {e}")
            return {
                "message": "I'm here to help with your travel planning! You can ask me to search for flights or hotels.",
                "agent_type": "supervisor",
                "requires_action": False,
                "metadata": {
                    "agent_type": "supervisor",
                    "conversation_type": "general",
                    "error": str(e)
                }
            }
    
    @openai_api_retry
    async def _make_openai_call_no_timeout(self, system_prompt: str, user_message: str) -> str:
        """ENHANCED: Make OpenAI call for general conversation with NO TIMEOUT LIMITS and Tenacity retry strategies"""
        
        # Use ultra-short prompt for efficiency
        input_prompt = f"""System: {system_prompt}

User: {user_message}

Assistant (be brief and helpful):"""
        
        response = self.openai_client.responses.create(
            model="o4-mini",
            input=input_prompt
        )
        
        # Extract response quickly
        assistant_message = "I'm here to help with your travel planning!"
        
        if response and hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if hasattr(output_item, 'content') and output_item.content:
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text') and content_item.text:
                            assistant_message = content_item.text.strip()
                            break
                    if assistant_message != "I'm here to help with your travel planning!":
                        break
        
        return assistant_message
    
    @openai_api_retry
    async def _make_openai_call_with_context(self, system_prompt: str, user_message: str, 
                                           context_messages: List[Dict[str, str]]) -> str:
        """ENHANCED: Make OpenAI call with conversation context for better continuity"""
        
        # Build conversation context for better responses
        conversation_context = ""
        if context_messages:
            # Include last few messages for context (limit to keep prompt manageable)
            recent_messages = context_messages[-10:]  # Last 10 for context building
            for msg in recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    conversation_context += f"{role.title()}: {content}\n"
        
        # Build prompt with context
        input_prompt = f"""System: {system_prompt}

Previous conversation:
{conversation_context}

User: {user_message}

Assistant (be brief and helpful, considering the conversation context):"""
        
        response = self.openai_client.responses.create(
            model="o4-mini",
            input=input_prompt
        )
        
        # Extract response
        assistant_message = "I'm here to help with your travel planning!"
        
        if response and hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if hasattr(output_item, 'content') and output_item.content:
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text') and content_item.text:
                            assistant_message = content_item.text.strip()
                            break
                    if assistant_message != "I'm here to help with your travel planning!":
                        break
        
        return assistant_message

    # ==================== ADDITIONAL METHODS FOR COMPATIBILITY ====================
    
    async def handle_general_conversation(self, user_id: str, message: str, session_id: str,
                                        user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Compatibility method for general conversation handling (calls the main method)
        """
        return await self._handle_general_conversation_no_timeout(
            user_id, message, session_id, user_profile, conversation_history
        )

# ==================== MODULE EXPORTS ====================

__all__ = [
    'SupervisorAgent'
]
