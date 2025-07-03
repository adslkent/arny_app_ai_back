from typing import Dict, Any, Optional
from openai import OpenAI

from ..utils.config import config
from .flight_agent import FlightAgent
from .hotel_agent import HotelAgent

class SupervisorAgent:
    """
    Supervisor agent that coordinates sub-agents and handles general queries
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize the specialized agents
        self.flight_agent = FlightAgent()
        self.hotel_agent = HotelAgent()
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                            user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Process user message with intelligent routing to specialized agents
        
        Args:
            user_id: User identifier
            message: User's message
            session_id: Session identifier
            user_profile: User's profile information
            conversation_history: Previous conversation history
            
        Returns:
            Dict containing agent response
        """
        
        try:
            print(f"🤖 Supervisor processing message: '{message[:50]}...'")
            
            # Analyze the message to determine routing
            routing_decision = await self._analyze_message_for_routing(message)
            
            print(f"🎯 Routing decision: {routing_decision}")
            
            # Route to appropriate agent based on analysis
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
                return await self._handle_general_conversation(
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
    
    async def _analyze_message_for_routing(self, message: str) -> str:
        """
        Analyze user message to determine which agent should handle it
        
        Args:
            message: User's message
            
        Returns:
            str: "flight_search", "hotel_search", or "general"
        """
        
        try:
            print(f"🔍 Analyzing message for routing: '{message}'")
            
            # Use OpenAI to analyze the message intent
            prompt = f"""Analyze this user message and determine what type of request it is.

User message: "{message}"

Determine if this is:
1. A flight search request (user wants to find, search, or book flights)
2. A hotel search request (user wants to find, search, or book hotels/accommodation)
3. A general travel question or conversation

Respond with exactly one of these three words:
- "flight_search" if it's about flights
- "hotel_search" if it's about hotels/accommodation
- "general" if it's general travel conversation

Look for keywords like:
- Flight search: flight, flights, fly, plane, airline, departure, arrival, ticket
- Hotel search: hotel, hotels, accommodation, stay, room, check-in, check-out, booking

Message: "{message}"
Classification:"""

            response = self.openai_client.responses.create(
                model="o4-mini",
                input=prompt
            )
            
            # Extract response
            classification = "general"  # Default
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                classification = content_item.text.strip().lower()
                                break
            
            # Validate classification
            valid_classifications = ["flight_search", "hotel_search", "general"]
            if classification not in valid_classifications:
                print(f"⚠️ Invalid classification '{classification}', defaulting to 'general'")
                classification = "general"
            
            print(f"✅ Message classified as: '{classification}'")
            return classification
            
        except Exception as e:
            print(f"❌ Error in message analysis: {e}")
            # Fallback to keyword-based routing
            return self._fallback_keyword_routing(message)
    
    def _fallback_keyword_routing(self, message: str) -> str:
        """
        Fallback keyword-based routing if LLM analysis fails
        
        Args:
            message: User's message
            
        Returns:
            str: "flight_search", "hotel_search", or "general"
        """
        
        message_lower = message.lower()
        
        # Flight keywords
        flight_keywords = [
            'flight', 'flights', 'fly', 'plane', 'airline', 'airport', 
            'departure', 'arrival', 'ticket', 'book flight', 'find flight'
        ]
        
        # Hotel keywords  
        hotel_keywords = [
            'hotel', 'hotels', 'accommodation', 'stay', 'room', 'rooms',
            'check-in', 'check-out', 'booking', 'book hotel', 'find hotel'
        ]
        
        # Check for flight keywords
        if any(keyword in message_lower for keyword in flight_keywords):
            print(f"🔄 Fallback: Detected flight keywords")
            return "flight_search"
        
        # Check for hotel keywords
        elif any(keyword in message_lower for keyword in hotel_keywords):
            print(f"🔄 Fallback: Detected hotel keywords")
            return "hotel_search"
        
        else:
            print(f"🔄 Fallback: No specific keywords detected, treating as general")
            return "general"
    
    async def _handle_general_conversation(self, user_id: str, message: str, session_id: str,
                                         user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Handle general travel conversation that doesn't require specialized agents
        
        Args:
            user_id: User identifier
            message: User's message
            session_id: Session identifier
            user_profile: User's profile information
            conversation_history: Previous conversation history
            
        Returns:
            Dict containing response
        """
        
        try:
            print(f"💬 Handling general conversation")
            
            # Build context from user profile
            user_context = ""
            if user_profile.get("name"):
                user_context += f"User's name: {user_profile['name']}\n"
            if user_profile.get("city"):
                user_context += f"User's city: {user_profile['city']}\n"
            
            # Build conversation context
            context_messages = []
            
            # Add recent conversation history (last 5 messages)
            for msg in conversation_history[-5:]:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            # Create prompt for general travel assistance
            system_prompt = f"""You are Arny, a helpful AI travel assistant. You provide general travel advice, recommendations, and friendly conversation.

{user_context}

You can help with:
- General travel advice and tips
- Travel destination recommendations  
- Travel planning guidance
- Answering travel-related questions
- Friendly conversation about travel

For specific flight searches or hotel searches, let the user know they can ask you to search for flights or hotels and you'll help them with that.

Always be friendly, helpful, and professional. Keep responses conversational and not too long."""

            # Build messages for OpenAI
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            messages.extend(context_messages)
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Get response from OpenAI
            response = self.openai_client.responses.create(
                model="o4-mini",
                input=f"""Context: {system_prompt}

Conversation history: {context_messages}

User message: {message}

Respond as Arny, the travel assistant:"""
            )
            
            # Extract response
            assistant_message = "I'm here to help with your travel planning! Feel free to ask me about flights, hotels, or any travel advice you need."
            
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                assistant_message = content_item.text.strip()
                                break
            
            print(f"✅ Generated general response: '{assistant_message[:50]}...'")
            
            return {
                "message": assistant_message,
                "agent_type": "supervisor",
                "requires_action": False,
                "metadata": {
                    "agent_type": "supervisor",
                    "conversation_type": "general",
                    "user_profile_used": bool(user_profile.get("name") or user_profile.get("city"))
                }
            }
            
        except Exception as e:
            print(f"❌ Error in general conversation: {e}")
            return {
                "message": "I'm here to help with your travel planning! You can ask me to search for flights, find hotels, or get travel advice.",
                "agent_type": "supervisor",
                "requires_action": False,
                "metadata": {
                    "agent_type": "supervisor",
                    "conversation_type": "general",
                    "error": str(e)
                }
            }
    
    async def handle_general_conversation(self, user_id: str, message: str, session_id: str,
                                        user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Legacy method for backwards compatibility - delegates to process_message
        """
        return await self.process_message(user_id, message, session_id, user_profile, conversation_history)
