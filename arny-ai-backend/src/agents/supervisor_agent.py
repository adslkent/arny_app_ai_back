from typing import Dict, Any, Optional
import asyncio
import logging
import requests
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
from pydantic import BaseModel, ValidationError

from ..utils.config import config
from .flight_agent import FlightAgent
from .hotel_agent import HotelAgent

# Set up logging
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class OpenAIResponse(BaseModel):
    """Pydantic model for OpenAI response validation"""
    output: Optional[Any] = None

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
    """Condition 5: Retry if OpenAI result fails Pydantic validation"""
    try:
        if result:
            OpenAIResponse.model_validate(result.__dict__ if hasattr(result, '__dict__') else result)
        return False
    except (ValidationError, AttributeError):
        return True

def retry_on_openai_api_exception(exception):
    """Condition 4: Custom exception checker for OpenAI API calls"""
    exception_str = str(exception).lower()
    return any(keyword in exception_str for keyword in [
        'timeout', 'failed', 'unavailable', 'rate limit', 'api error',
        'connection', 'network', 'server error'
    ])

# OpenAI API retry decorator with all 5 conditions
openai_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|rate.limit|api.error|connection|network|server.error).*"),
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
    ENHANCED: Supervisor agent with fast routing and NO TIMEOUT LIMITS
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
        ENHANCED: Fast general conversation with NO TIMEOUT LIMITS
        """
        
        try:
            print(f"💬 Fast general conversation - NO TIMEOUT LIMITS")
            
            # Build minimal context
            user_context = ""
            if user_profile.get("name"):
                user_context += f"User's name: {user_profile['name']}\n"
            if user_profile.get("city"):
                user_context += f"User's city: {user_profile['city']}\n"
            
            # Ultra-short system prompt
            system_prompt = f"""You are Arny, a helpful AI travel assistant. 

{user_context}

For specific searches, ask users to request "flights" or "hotels" with dates and destinations.
Keep responses friendly and brief."""

            # ENHANCED: Direct OpenAI call with NO TIMEOUT LIMITS
            try:
                assistant_message = await self._make_openai_call_no_timeout(system_prompt, message)
                    
            except Exception as openai_error:
                print(f"⚠️ OpenAI error: {openai_error}")
                assistant_message = "I'm here to help with your travel planning! You can ask me to search for flights or hotels with your travel dates."
            
            return {
                "message": assistant_message,
                "agent_type": "supervisor",
                "requires_action": False,
                "metadata": {
                    "agent_type": "supervisor",
                    "conversation_type": "general"
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

# ==================== MODULE EXPORTS ====================

__all__ = [
    'SupervisorAgent'
]
