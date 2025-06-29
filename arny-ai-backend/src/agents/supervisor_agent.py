from typing import Dict, Any, Optional
from agents import Agent, Runner
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from ..utils.config import config
from .flight_agent import FlightAgent
from .hotel_agent import HotelAgent

class SupervisorAgent:
    """
    Supervisor agent that coordinates sub-agents and handles general queries using OpenAI Agents SDK
    """
    
    def __init__(self):
        # Initialize the specialized agents
        self.flight_agent_instance = FlightAgent()
        self.hotel_agent_instance = HotelAgent()
        
        # Create SDK agents for handoffs
        self.flight_sdk_agent = Agent(
            name="Flight Assistant",
            handoff_description="Specialist agent for flight searches and bookings",
            instructions="You are Arny's flight search specialist. You help users find and book flights.",
            model="o4-mini"
        )
        
        self.hotel_sdk_agent = Agent(
            name="Hotel Assistant", 
            handoff_description="Specialist agent for hotel searches and bookings",
            instructions="You are Arny's hotel booking specialist. You help users find and book hotels.",
            model="o4-mini"
        )
        
        # Create the main supervisor agent with handoffs
        self.agent = Agent(
            name="Arny Travel Assistant",
            instructions=prompt_with_handoff_instructions(
                "You are Arny, a helpful AI travel assistant with expertise in travel planning. "
                "You can help with general questions and coordinate with specialized agents when needed.\n\n"
                "For specific requests:\n"
                "- If the user asks to search for flights, handoff to the Flight Assistant\n"
                "- If the user asks to search for hotels, handoff to the Hotel Assistant\n"
                "- For general travel advice, recommendations, or other questions, provide helpful responses directly\n\n"
                "Always be friendly, professional, and helpful. Understand the user's travel needs and "
                "provide appropriate guidance or delegate to the right specialist agent."
            ),
            model="o4-mini",
            handoffs=[self.flight_sdk_agent, self.hotel_sdk_agent]
        )
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                            user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Process user message using OpenAI Agents SDK with handoffs
        
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
            # Store context for potential handoff processing
            self.current_user_id = user_id
            self.current_session_id = session_id
            self.current_user_profile = user_profile
            
            # Build conversation context
            context_messages = []
            
            # Add recent conversation history (last 10 messages)
            for msg in conversation_history[-10:]:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            # Process with agent - the SDK will handle handoffs automatically
            if not context_messages:
                # First message in conversation
                result = await Runner.run(self.agent, message)
            else:
                # Continue conversation with context
                result = await Runner.run(self.agent, context_messages + [{"role": "user", "content": message}])
            
            # Check if a handoff occurred by examining the active agent
            final_agent_name = getattr(result, 'agent', self.agent).name if hasattr(result, 'agent') else self.agent.name
            
            # If handoff occurred to specialized agent, delegate to the actual implementation
            if final_agent_name == "Flight Assistant":
                return await self.flight_agent_instance.process_message(
                    user_id, message, session_id, user_profile, conversation_history
                )
            elif final_agent_name == "Hotel Assistant":
                return await self.hotel_agent_instance.process_message(
                    user_id, message, session_id, user_profile, conversation_history
                )
            else:
                # Supervisor handled the request directly
                return {
                    "message": result.final_output,
                    "agent_type": "supervisor",
                    "requires_action": False,
                    "metadata": {
                        "agent_type": "supervisor",
                        "conversation_type": "general",
                        "user_profile_used": bool(user_profile.get("name") or user_profile.get("city"))
                    }
                }
        
        except Exception as e:
            print(f"Error in supervisor agent: {e}")
            return {
                "message": "I'm sorry, I encountered an error processing your request. Please try again.",
                "agent_type": "supervisor",
                "requires_action": False,
                "error": str(e)
            }
    
    async def handle_general_conversation(self, user_id: str, message: str, session_id: str,
                                        user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Handle general conversation using the SDK agent (legacy method for backwards compatibility)
        
        This method now delegates to process_message since the SDK handles routing automatically.
        """
        return await self.process_message(user_id, message, session_id, user_profile, conversation_history)
