from typing import Dict, Any, Optional
import asyncio
from openai import OpenAI

from ..utils.config import config
from .flight_agent import FlightAgent
from .hotel_agent import HotelAgent

class SupervisorAgent:
    """
    ULTRA-OPTIMIZED: Supervisor agent with fast routing and timeout prevention
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize the specialized agents
        self.flight_agent = FlightAgent()
        self.hotel_agent = HotelAgent()
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                            user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ULTRA-OPTIMIZED: Process user message with ultra-fast routing
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
                return await self._handle_general_conversation_fast(
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
    
    async def _handle_general_conversation_fast(self, user_id: str, message: str, session_id: str,
                                             user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ULTRA-OPTIMIZATION: Fast general conversation with 5s timeout
        """
        
        try:
            print(f"💬 Fast general conversation")
            
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

            # OPTIMIZATION: 5-second timeout on OpenAI call
            try:
                timeout_task = asyncio.create_task(asyncio.sleep(5))
                openai_task = asyncio.create_task(self._make_openai_call_fast(system_prompt, message))
                
                done, pending = await asyncio.wait(
                    [timeout_task, openai_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                if openai_task in done:
                    assistant_message = openai_task.result()
                else:
                    print(f"⚠️ OpenAI call timed out after 5s")
                    assistant_message = "I'm here to help with your travel planning! You can ask me to search for flights or hotels with your travel dates."
                    
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
    
    async def _make_openai_call_fast(self, system_prompt: str, user_message: str) -> str:
        """ULTRA-FAST: Make OpenAI call for general conversation"""
        
        # Use ultra-short prompt for speed
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
    
    async def handle_general_conversation(self, user_id: str, message: str, session_id: str,
                                        user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Legacy method for backwards compatibility - delegates to process_message
        """
        return await self.process_message(user_id, message, session_id, user_profile, conversation_history)
