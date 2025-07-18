"""
Hotel Search Agent Module - ENHANCED VERSION to match flight agent improvements

This module provides a hotel search agent with enhanced capabilities:
1. Support for up to 50 hotel results from Amadeus API
2. Send all hotels to OpenAI for filtering
3. Return up to 10 filtered hotel results
4. Optimized for larger datasets

Usage example:
```python
from hotel_agent import HotelAgent

# Create and use the agent
agent = HotelAgent()
result = await agent.process_message(user_id, "Find hotels in Paris", session_id, {}, [])
```
"""

import json
import uuid
import logging
import asyncio
import concurrent.futures
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from agents import Agent, function_tool, Runner
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
from ..services.amadeus_service import AmadeusService
from ..database.operations import DatabaseOperations
from ..database.models import HotelSearch
from .user_profile_agent import UserProfileAgent

# Configure logging
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class AgentRunnerResponse(BaseModel):
    """Pydantic model for Agent Runner response validation"""
    final_output: Optional[str] = None

# ==================== OPENAI AGENTS SDK RETRY CONDITIONS ====================

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
    """Condition 5: Retry if validation fails on OpenAI result"""
    try:
        if isinstance(result, dict) and result.get("final_output"):
            AgentRunnerResponse(**result)
        return False  # Validation passed
    except ValidationError:
        return True  # Validation failed, retry
    except Exception as e:
        logger.warning(f"Unexpected validation error: {e}")
        return True
    return False

def retry_on_openai_api_exception(exception):
    """Custom exception checker for OpenAI-specific exceptions"""
    return "openai" in str(type(exception)).lower() or "api" in str(exception).lower()

# ==================== COMBINED RETRY STRATEGIES ====================

# Primary retry strategy for critical OpenAI Agents SDK operations
openai_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|api.?error|rate.?limit).*"),
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

# ===== City Code Mapping for Hotels =====
CITY_CODE_MAPPING = {
    # Major cities to city codes for Amadeus Hotel API
    "new york": "NYC",
    "los angeles": "LAX", 
    "san francisco": "SFO",
    "chicago": "CHI",
    "washington": "WAS",
    "washington dc": "WAS",
    "boston": "BOS",
    "miami": "MIA",
    "las vegas": "LAS",
    "seattle": "SEA",
    "denver": "DEN",
    "atlanta": "ATL",
    
    # International cities
    "london": "LON",
    "paris": "PAR",
    "tokyo": "TYO",
    "sydney": "SYD",
    "melbourne": "MEL",
    "brisbane": "BNE",
    "perth": "PER",
    "adelaide": "ADL",
    "canberra": "CBR",
    "dubai": "DXB",
    "rome": "ROM",
    "madrid": "MAD",
    "barcelona": "BCN",
    "amsterdam": "AMS",
    "berlin": "BER",
    "singapore": "SIN",
    "hong kong": "HKG",
    "bangkok": "BKK",
    "mumbai": "BOM",
    "delhi": "DEL",
    "toronto": "YTO",
    "vancouver": "YVR",
    "montreal": "YMQ"
}

def _convert_to_city_code(destination: str) -> str:
    """Convert destination name to city code"""
    destination_lower = destination.lower().strip()
    
    # Direct mapping
    if destination_lower in CITY_CODE_MAPPING:
        return CITY_CODE_MAPPING[destination_lower]
    
    # Partial matching
    for city, code in CITY_CODE_MAPPING.items():
        if city in destination_lower or destination_lower in city:
            return code
    
    # Fallback: return first 3 letters uppercase
    return destination_lower[:3].upper()

# ===== System Message =====
def get_hotel_system_message() -> str:
    """Generate the hotel agent system message with current date"""
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return f"""You are a specialized hotel search assistant powered by Amadeus API. 

Your main responsibilities are:
1. Understanding users' hotel needs and extracting key information from natural language descriptions
2. Using the search_hotels_tool to search for hotels that meet user requirements
3. Presenting hotel options in a clear, organized format

**IMPORTANT PRESENTATION RULES:**
- When you receive hotel search results, ALWAYS present ALL hotels in your response
- Do NOT truncate or summarize the hotel list - show complete details for every result
- Present hotels in an organized, easy-to-read format
- Include all key details: name, price, rating, amenities, location
- Number each hotel option for easy reference

Today's date: {today}
Tomorrow's date: {tomorrow}

**Key Guidelines:**
1. **Date Handling**: If user says "tomorrow", use {tomorrow}. If no year specified, assume current year.

2. **Hotel Search**: Always use search_hotels_tool for hotel requests. Extract:
   - Destination city/location
   - Check-in date (format: YYYY-MM-DD)
   - Check-out date (format: YYYY-MM-DD)
   - Number of adults (default: 1)
   - Number of rooms (default: 1)

3. **Response Style**: Be professional, helpful, and provide clear options. When presenting multiple hotels, organize them clearly and include all important details.

4. **No Booking**: You can search and provide hotel information, but cannot make actual bookings. Direct users to hotel websites or booking platforms for reservations.

5. **Missing Information**: If critical details are missing, ask for clarification before searching.

6. **Date Validation**: Ensure check-in date is in the future and check-out date is after check-in date.

Remember: You are a specialized hotel search agent. Focus on hotel-related queries and use your tools effectively to provide the best accommodation options for users."""

# ==================== ASYNC UTILITIES ====================

def run_async(coro):
    """Run async coroutine in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running loop, create a new one in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_in_new_loop, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return _run_in_new_loop(coro)

def _run_in_new_loop(coro):
    """Run coroutine in a new event loop"""
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
        asyncio.set_event_loop(None)

# Global variable to store the current agent instance
_current_hotel_agent = None

def _get_hotel_agent():
    """Get the current hotel agent instance"""
    global _current_hotel_agent
    return _current_hotel_agent

@function_tool
async def search_hotels_tool(destination: str, check_in_date: str, check_out_date: str,
                           adults: int = 1, rooms: int = 1) -> dict:
    """
    Search for hotels using Amadeus API with enhanced profile filtering and caching
    
    Args:
        destination: Destination city or hotel location
        check_in_date: Check-in date in YYYY-MM-DD format
        check_out_date: Check-out date in YYYY-MM-DD format
        adults: Number of adults (default 1)
        rooms: Number of rooms (default 1)
    
    Returns:
        Dict with search results and metadata
    """
    
    try:
        print(f"🏨 ENHANCED Hotel search started: {destination}")
        start_time = datetime.now()
        
        hotel_agent = _get_hotel_agent()
        if not hotel_agent:
            return {"success": False, "error": "Hotel agent not available"}
        
        # OPTIMIZATION 1: Check cache first
        search_key = f"{destination}_{check_in_date}_{check_out_date}_{adults}_{rooms}"
        if hasattr(hotel_agent, '_search_cache') and search_key in hotel_agent._search_cache:
            print(f"⚡ Cache hit! Returning cached results")
            cached_result = hotel_agent._search_cache[search_key]
            
            # Update instance variables for response
            hotel_agent.latest_search_results = cached_result.get('results', [])
            hotel_agent.latest_search_id = cached_result.get('search_id')
            hotel_agent.latest_filtering_info = cached_result.get('filtering_info', {})
            
            return cached_result
        
        # OPTIMIZATION 2: Validate dates
        try:
            checkin_datetime = datetime.strptime(check_in_date, "%Y-%m-%d")
            checkout_datetime = datetime.strptime(check_out_date, "%Y-%m-%d")
            
            if checkin_datetime < datetime.now():
                return {
                    "success": False,
                    "error": "Check-in date cannot be in the past",
                    "message": "Please provide a future check-in date."
                }
            
            if checkout_datetime <= checkin_datetime:
                return {
                    "success": False,
                    "error": "Check-out date must be after check-in date",
                    "message": "Please ensure check-out date is after check-in date."
                }
                
        except ValueError:
            return {
                "success": False,
                "error": "Invalid date format",
                "message": "Please provide dates in YYYY-MM-DD format."
            }
        
        # OPTIMIZATION 3: Get destination code using proper city mapping
        destination_code = _convert_to_city_code(destination)
        print(f"🔧 Using destination code: {destination_code}")

        # OPTIMIZATION 4: Search hotels using Amadeus with ENHANCED parameters for larger datasets
        try:
            search_results = await hotel_agent.amadeus_service.search_hotels(
                city_code=destination_code,
                check_in_date=check_in_date,
                check_out_date=check_out_date,
                adults=adults,
                rooms=rooms,
                max_results=50  # ENHANCED: Get up to 50 results for better filtering
            )
        except Exception as e:
            # Handle RetryError and other exceptions gracefully
            error_msg = str(e)
            if "RetryError" in error_msg:
                error_msg = "Unable to search hotels at this time. The hotel service is temporarily unavailable."
            
            return {
                "success": False,
                "error": error_msg,
                "message": f"I'm having trouble finding hotels in {destination}. Please try a different city or try again later."
            }
        
        if not search_results.get("success"):
            return {
                "success": False,
                "error": search_results.get("error", "Hotel search failed"),
                "message": f"I couldn't find any hotels in {destination} for your dates. Please try different dates or destinations."
            }
        
        hotels = search_results.get("results", [])
        print(f"🔍 Found {len(hotels)} raw hotels from Amadeus")
        
        # OPTIMIZATION 5: Store in database with enhanced metadata
        user_id = hotel_agent.current_user_id
        session_id = hotel_agent.current_session_id
        
        hotel_search = HotelSearch(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            city_code=destination_code,  # FIXED: Use city_code instead of destination
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            adults=adults,
            rooms=rooms,
            search_results=hotels,
            search_successful=True,  # ADDED: Set search_successful flag
            result_count=len(hotels)  # ADDED: Set result count
        )
        
        try:
            await hotel_agent.db.save_hotel_search(hotel_search)
            print(f"💾 Saved search with ID: {hotel_search.id}")
        except Exception as save_error:
            print(f"⚠️ Failed to save hotel search: {save_error}")
            # Continue processing even if save fails
        
        # OPTIMIZATION 6: ENHANCED filtering with profile agent (send ALL hotels for filtering)
        print(f"🧠 Filtering {len(hotels)} hotels with group profiles...")
        filtering_start = datetime.now()
        
        try:
            filtering_result = await hotel_agent.profile_agent.filter_hotel_results(
                user_id=user_id,
                hotel_results=hotels,  # Send ALL hotels for filtering
                search_params={
                    "city_code": destination_code,
                    "check_in_date": check_in_date,
                    "check_out_date": check_out_date,
                    "adults": adults,
                    "rooms": rooms
                }
            )
        except Exception as filter_error:
            print(f"⚠️ Filtering failed, using top 10 hotels: {filter_error}")
            filtering_result = {
                "filtered_results": hotels[:10],
                "total_results": len(hotels),
                "filtered_count": min(10, len(hotels)),
                "filter_applied": False,
                "filter_reason": "Profile filtering failed - using default"
            }
        
        filtered_hotels = filtering_result.get("filtered_results", hotels[:10])
        filtering_info = {
            "total_found": filtering_result.get("total_results", len(hotels)),
            "filtered_count": len(filtered_hotels),
            "filter_applied": filtering_result.get("filter_applied", False),
            "filter_reason": filtering_result.get("filter_reason", "Standard filtering")
        }
        
        filtering_time = (datetime.now() - filtering_start).total_seconds()
        print(f"🎯 Profile filtering completed in {filtering_time:.2f}s")
        print(f"✅ Filtered to {len(filtered_hotels)} hotels based on user preferences")
        
        # OPTIMIZATION 7: Store results for response
        hotel_agent.latest_search_results = filtered_hotels
        hotel_agent.latest_search_id = hotel_search.id
        hotel_agent.latest_filtering_info = filtering_info
        
        # OPTIMIZATION 8: Cache the result for future requests
        cache_result = {
            "success": True,
            "results": filtered_hotels,
            "search_id": hotel_search.id,
            "filtering_info": filtering_info,
            "message": f"Found {len(filtered_hotels)} hotels in {destination}"
        }
        hotel_agent._search_cache[search_key] = cache_result
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"🏨 ENHANCED hotel search completed in {total_time:.2f}s")
        
        return cache_result
        
    except Exception as e:
        print(f"❌ Error in hotel search tool: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for hotels: {str(e)}"
        }

# ==================== HOTEL AGENT CLASS ====================

class HotelAgent:
    """
    Hotel agent using OpenAI Agents SDK with Amadeus API tools and profile filtering - ENHANCED VERSION
    """
    
    def __init__(self):
        global _current_hotel_agent
        
        self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.amadeus_service = AmadeusService()
        self.db = DatabaseOperations()
        self.profile_agent = UserProfileAgent()
        
        # Store context for tool calls
        self.current_user_id = None
        self.current_session_id = None
        self.user_profile = None
        
        # Store latest search results for response
        self.latest_search_results = []
        self.latest_search_id = None
        self.latest_filtering_info = {}
        
        # Initialize search cache
        self._search_cache = {}
        
        # Set global instance
        _current_hotel_agent = self
        
        # ENHANCED: Create agent with tools - FIXED: Added required 'name' parameter
        self.agent = Agent(
            name="hotel_search_agent",  # FIXED: Added required name parameter
            model="o4-mini",
            instructions=get_hotel_system_message(),
            tools=[search_hotels_tool]
        )
        
        print("✅ HotelAgent initialized with enhanced tools and caching")
    
    # ==================== PROCESS MESSAGE ====================
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                             user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Process hotel search requests - ENHANCED VERSION with support for larger datasets
        """
        
        try:
            print(f"🏨 ENHANCED HotelAgent processing: '{message[:50]}...'")
            start_time = datetime.now()
            
            # Clear previous search results
            self.latest_search_results = []
            self.latest_search_id = None
            self.latest_filtering_info = {}
            
            # Store context for tool calls
            self.current_user_id = user_id
            self.current_session_id = session_id
            self.user_profile = user_profile
            
            # Update the global instance
            global _current_hotel_agent
            _current_hotel_agent.current_user_id = user_id
            _current_hotel_agent.current_session_id = session_id
            _current_hotel_agent.user_profile = user_profile
            _current_hotel_agent.latest_search_results = []
            _current_hotel_agent.latest_search_id = None
            _current_hotel_agent.latest_filtering_info = {}
            
            print(f"🔧 Context set: user_id={user_id}")
            
            # Build conversation context (optimized for performance)
            context_messages = []
            # UNIFIED CONTEXT: Use last 50 messages for context (matching flight agent)
            for msg in conversation_history[-50:]:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            print(f"🔧 Processing with {len(context_messages)} previous messages")
            
            # ENHANCED: Direct agent processing with improved efficiency
            if not context_messages:
                # First message in conversation
                print("🚀 Starting new hotel conversation")
                result = await self._run_agent_with_retry(self.agent, message)
            else:
                # Continue conversation with context
                print("🔄 Continuing hotel conversation with context")
                result = await self._run_agent_with_retry(self.agent, context_messages + [{"role": "user", "content": message}])
            
            # Extract response
            assistant_message = result.get("final_output") or "I'm sorry, but I encountered an unexpected error while searching for hotels. Would you like me to try again?"
            
            # Get search results from global instance
            global_agent = _get_hotel_agent()
            search_results = global_agent.latest_search_results if global_agent else []
            search_id = global_agent.latest_search_id if global_agent else None
            filtering_info = global_agent.latest_filtering_info if global_agent else {}
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ ENHANCED HotelAgent completed in {elapsed_time:.2f}s")
            print(f"📊 Retrieved search data: {len(search_results)} results, search_id: {search_id}")
            
            return {
                "message": assistant_message,
                "agent_type": "hotel",
                "requires_action": False,
                "search_results": search_results,
                "search_id": search_id,
                "filtering_info": filtering_info,
                "metadata": {
                    "agent_type": "hotel",
                    "conversation_type": "hotel_search",
                    "processing_time": elapsed_time
                }
            }
        
        except Exception as e:
            print(f"❌ Error in hotel agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": "I'm sorry, I encountered an error while searching for hotels. Please try again with different dates or destinations.",
                "agent_type": "hotel",
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {},
                "metadata": {
                    "agent_type": "hotel",
                    "conversation_type": "hotel_search",
                    "processing_time": 0,
                    "error": str(e)
                }
            }
    
    # ==================== AGENT EXECUTION WITH RETRY ====================
    
    @openai_api_retry
    async def _run_agent_with_retry(self, agent: Agent, messages) -> Dict[str, Any]:
        """Run agent with retry logic and proper error handling"""
        try:
            print("🔄 Running hotel agent with retry logic...")
            
            # Handle different message formats
            if isinstance(messages, str):
                # Single message - use the same pattern as flight agent
                result = await Runner.run(agent, messages)
            elif isinstance(messages, list):
                # List of messages - use the same pattern as flight agent
                result = await Runner.run(agent, messages)
            else:
                raise ValueError(f"Invalid messages format: {type(messages)}")
            
            # Extract clean response - handle RunResult object
            assistant_message = ""
            if hasattr(result, 'final_output'):
                assistant_message = result.final_output
            elif hasattr(result, 'messages') and result.messages:
                # Get the last message from the agent
                last_message = result.messages[-1]
                if hasattr(last_message, 'content'):
                    if isinstance(last_message.content, list) and len(last_message.content) > 0:
                        assistant_message = last_message.content[0].text if hasattr(last_message.content[0], 'text') else str(last_message.content[0])
                    else:
                        assistant_message = str(last_message.content)
                else:
                    assistant_message = str(last_message)
            else:
                # Fallback: convert result to string and clean it
                result_str = str(result)
                # Remove the RunResult debug information
                if "Final output (str):" in result_str:
                    parts = result_str.split("Final output (str):")
                    if len(parts) > 1:
                        # Get everything after "Final output (str):" and clean it
                        assistant_message = parts[1].strip()
                        # Remove any trailing debug info
                        if "\n- " in assistant_message:
                            assistant_message = assistant_message.split("\n- ")[0].strip()
                    else:
                        assistant_message = result_str
                else:
                    assistant_message = result_str
            
            # Clean up any remaining whitespace
            assistant_message = assistant_message.strip()
            
            return {
                "final_output": assistant_message,
                "messages": []  # Runner.run doesn't return messages like the old pattern
            }
            
        except Exception as e:
            print(f"❌ Hotel agent execution failed: {e}")
            # Return error response that will trigger retry
            return {
                "final_output": None,
                "error": str(e),
                "success": False
            }
