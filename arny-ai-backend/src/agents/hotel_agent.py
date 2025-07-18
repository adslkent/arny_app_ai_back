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
    "beijing": "PEK",
    "shanghai": "SHA",
    "hong kong": "HKG",
    "singapore": "SIN",
    "bangkok": "BKK",
    "sydney": "SYD",
    "melbourne": "MEL",
    "dubai": "DXB",
    "berlin": "BER",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
    "rome": "ROM",
    "madrid": "MAD",
    "barcelona": "BCN",
    "moscow": "MOW",
    "toronto": "YTO",
    "vancouver": "YVR",
    "montreal": "YMQ",
    "milan": "MIL",
    "vienna": "VIE",
    "zurich": "ZUR",
    "geneva": "GVA",
    "brussels": "BRU",
    "seoul": "SEL",
    "taipei": "TPE",
    "osaka": "OSA",
    "auckland": "AKL",
    "mumbai": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "kuala lumpur": "KUL",
    "manila": "MNL",
    "jakarta": "JKT",
    "cairo": "CAI",
    "istanbul": "IST",
    "athens": "ATH",
    "munich": "MUC",
    "copenhagen": "CPH",
    "stockholm": "STO",
    "oslo": "OSL",
    "helsinki": "HEL",
    "lisbon": "LIS",
    "dublin": "DUB",
    "sao paulo": "SAO",
    "rio de janeiro": "RIO",
    "buenos aires": "BUE",
    "johannesburg": "JNB",
}

def _convert_to_city_code(location: str) -> str:
    """Convert city names to city codes using mapping"""
    
    location_lower = location.lower().strip()
    
    # Use city code mapping
    if location_lower in CITY_CODE_MAPPING:
        return CITY_CODE_MAPPING[location_lower]
    
    # If already looks like city code (3 letters, uppercase)
    if len(location) == 3 and location.isupper():
        return location
    
    # If 3 letters but lowercase, convert to uppercase
    if len(location) == 3 and location.isalpha():
        return location.upper()
    
    # For longer names, try first 3 letters as fallback
    if len(location) > 3:
        return location.upper()[:3]
    
    # Default: return as-is and let Amadeus handle it
    return location.upper()

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
    """Condition 5: Retry if validation against OpenAI response schema fails"""
    if result is None:
        return True
    try:
        if isinstance(result, dict):
            AgentRunnerResponse(**result)
        elif hasattr(result, 'final_output'):
            AgentRunnerResponse(final_output=result.final_output)
        else:
            AgentRunnerResponse(final_output=str(result))
        return False  # Validation passed
    except ValidationError as e:
        logger.warning(f"OpenAI Agent response validation failed: {e}")
        return True  # Validation failed, retry

def retry_on_openai_exception_type(exception):
    """Custom exception checker for OpenAI-specific exceptions"""
    return isinstance(exception, (requests.exceptions.RequestException, ConnectionError, TimeoutError))

# ==================== OPENAI AGENTS SDK RETRY STRATEGY ====================

openai_agents_sdk_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|429|502|503|504|rate.?limit).*"),
        # Condition 4: Exception types
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        # Custom exception checker
        retry_if_exception(retry_on_openai_exception_type),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_openai_api_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_openai_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_openai_validation_failure)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# ==================== ASYNCIO HELPERS ====================

def run_async(coro):
    """Run async function in sync context"""
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
        
        if not search_results or "results" not in search_results:
            return {
                "success": False,
                "error": "No hotels found",
                "message": f"I couldn't find any hotels in {destination} for your dates. Please try different dates or destinations."
            }
        
        raw_hotels = search_results.get("results", [])
        
        if not raw_hotels:
            return {
                "success": False,
                "error": "No hotels found",
                "message": f"I couldn't find any hotels in {destination} for your dates. Please try different dates or destinations."
            }
        
        print(f"🔍 Found {len(raw_hotels)} raw hotels from Amadeus")
        
        # OPTIMIZATION 5: Get user profile for filtering (safely handle missing profile)
        user_profile = getattr(hotel_agent, 'user_profile', {}) or {}
        group_profiles = user_profile.get('group_profiles', [])
        
        # OPTIMIZATION 6: Enhanced hotel filtering for groups and preferences using OpenAI
        filtered_hotels = []
        filtering_info = {
            "filtering_applied": False,
            "original_count": len(raw_hotels),
            "group_size": len(group_profiles) + 1,  # +1 for primary user
            "rationale": ""
        }
        
        if len(raw_hotels) > 10:  # ENHANCED: Only filter if we have more than 10 hotels
            print(f"🧠 Filtering {len(raw_hotels)} hotels with group profiles...")
            
            try:
                # Build filter prompt for OpenAI
                filter_prompt = _build_hotel_filter_prompt(raw_hotels, user_profile, group_profiles, destination, adults, rooms)
                
                # Call OpenAI for filtering using direct API call for better control
                filtering_response = await hotel_agent.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a hotel filtering assistant. Analyze the provided hotels and return the top 10 that best match the user's profile and group preferences. Respond with valid JSON only."
                        },
                        {"role": "user", "content": filter_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                # Parse the filtering response
                filter_result = json.loads(filtering_response.choices[0].message.content)
                filtered_hotels = filter_result.get("filtered_hotels", raw_hotels[:10])
                
                filtering_info.update({
                    "filtering_applied": True,
                    "rationale": filter_result.get("rationale", "Filtered based on group preferences and user profile")
                })
                
                print(f"✅ OpenAI filtered to {len(filtered_hotels)} hotels")
                
            except Exception as e:
                print(f"⚠️ Filtering failed, using top 10 hotels: {e}")
                filtered_hotels = raw_hotels[:10]
        else:
            # If 10 or fewer hotels, use all of them
            filtered_hotels = raw_hotels
            print(f"📋 Using all {len(filtered_hotels)} hotels (no filtering needed)")
        
        # OPTIMIZATION 7: Generate search ID and store results
        search_id = str(uuid.uuid4())
        
        # OPTIMIZATION 8: Store search in database (async)
        if hotel_agent.current_user_id:
            try:
                search_record = HotelSearch(
                    user_id=hotel_agent.current_user_id,
                    session_id=hotel_agent.current_session_id,
                    search_id=search_id,
                    destination=destination,
                    destination_code=destination_code,
                    check_in_date=datetime.strptime(check_in_date, "%Y-%m-%d").date(),
                    check_out_date=datetime.strptime(check_out_date, "%Y-%m-%d").date(),
                    adults=adults,
                    rooms=rooms,
                    search_results=filtered_hotels,
                    result_count=len(filtered_hotels)
                )
                
                # Save asynchronously (don't wait for completion)
                asyncio.create_task(hotel_agent.db.create_hotel_search(search_record))
                print(f"💾 Hotel search saved asynchronously")
                
            except Exception as e:
                print(f"⚠️ Failed to save hotel search: {e}")
        
        # OPTIMIZATION 9: Update agent state
        hotel_agent.latest_search_results = filtered_hotels
        hotel_agent.latest_search_id = search_id
        hotel_agent.latest_filtering_info = filtering_info
        
        # Build response
        response = {
            "success": True,
            "results": filtered_hotels,
            "search_id": search_id,
            "filtering_info": filtering_info,
            "message": f"Found {len(filtered_hotels)} hotels in {destination}",
            "search_params": {
                "city_code": destination_code,
                "check_in_date": check_in_date,
                "check_out_date": check_out_date,
                "adults": adults,
                "rooms": rooms
            }
        }
        
        # OPTIMIZATION 10: Cache the response
        if not hasattr(hotel_agent, '_search_cache'):
            hotel_agent._search_cache = {}
        hotel_agent._search_cache[search_key] = response
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"🏨 ENHANCED hotel search completed in {elapsed_time:.2f}s")
        
        return response
        
    except Exception as e:
        print(f"❌ Error in hotel search: {e}")
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
            
            # ENHANCED: Create conversation with proper context handling
            if context_messages:
                print(f"🚀 Continuing hotel conversation with context")
                # Add the current message to context
                context_messages.append({
                    "role": "user",
                    "content": message
                })
                result = await self._run_agent_with_retry(self.agent, context_messages)
            else:
                print(f"🚀 Starting new hotel conversation")
                result = await self._run_agent_with_retry(self.agent, message)
            
            # Extract final response
            response_text = result.final_output if hasattr(result, 'final_output') else str(result)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ ENHANCED HotelAgent completed in {processing_time:.2f}s")
            
            # Return comprehensive response with all search data for supervisor
            return {
                "response": response_text,
                "agent_type": "hotel", 
                "requires_action": False,
                "search_results": self.latest_search_results,
                "search_id": self.latest_search_id,
                "filtering_info": self.latest_filtering_info,
                "metadata": {
                    "agent_type": "hotel",
                    "conversation_type": "hotel_search",
                    "processing_time": processing_time
                }
            }
            
        except Exception as e:
            logger.error(f"Hotel agent error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "response": "I'm sorry, but I encountered an unexpected error while searching for hotels. Would you like me to try again? If you'd prefer, I can also adjust the parameters or check availability for different dates. Let me know how you'd like to proceed!",
                "agent_type": "hotel",
                "error": str(e),
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {}
            }
    
    # ==================== HELPER METHODS ====================
        
    @openai_agents_sdk_retry
    async def _run_agent_with_retry(self, agent, messages):
        """Run agent with retry logic and validation - ENHANCED for NO TIMEOUT LIMITS"""
        
        try:
            print(f"🔄 Running hotel agent with retry logic...")
            
            # FIXED: Use Runner.run as static method instead of instantiating Runner
            if isinstance(messages, str):
                # Single message
                result = await Runner.run(agent, messages)
            else:
                # Multiple messages (conversation context)
                result = await Runner.run(agent, messages)
            
            # ENHANCED: Validate result and ensure it has the required structure
            if not result:
                raise ValueError("Agent returned empty result")
            
            # Convert to our expected format if needed
            if hasattr(result, 'final_output'):
                validated_result = AgentRunnerResponse(final_output=result.final_output)
            elif isinstance(result, dict) and 'final_output' in result:
                validated_result = AgentRunnerResponse(**result)
            else:
                # Try to extract message from different result formats
                if hasattr(result, 'content'):
                    validated_result = AgentRunnerResponse(final_output=str(result.content))
                elif hasattr(result, 'message'):
                    validated_result = AgentRunnerResponse(final_output=str(result.message))
                else:
                    validated_result = AgentRunnerResponse(final_output=str(result))
            
            print(f"✅ Hotel agent execution successful")
            return validated_result
            
        except Exception as e:
            print(f"❌ Hotel agent execution failed: {e}")
            # Re-raise for retry logic to catch
            raise e

# ==================== HELPER FUNCTIONS ====================

def _build_hotel_filter_prompt(hotels: List[Dict], user_profile: Dict, group_profiles: List[Dict], 
                              destination: str, adults: int, rooms: int) -> str:
    """Build filtering prompt for OpenAI to select best hotels for user/group"""
    
    # Extract relevant user preferences
    preferences = user_profile.get('preferences', {})
    travel_style = preferences.get('travel_style', 'standard')
    budget_range = preferences.get('budget_range', 'moderate')
    
    # Build group context
    group_context = ""
    if group_profiles:
        group_context = f"\n\nGroup Information:\n"
        group_context += f"- Total travelers: {len(group_profiles) + 1} people\n"
        group_context += f"- Requested adults: {adults}, rooms: {rooms}\n"
        
        # Add group preferences if available
        for i, profile in enumerate(group_profiles):
            name = profile.get('name', f'Traveler {i+1}')
            group_context += f"- {name}: {profile.get('preferences', {})}\n"
    
    # Build hotel list for analysis
    hotel_list = ""
    for i, hotel in enumerate(hotels[:30], 1):  # Limit to 30 for prompt size
        hotel_list += f"{i}. {json.dumps(hotel, indent=2)}\n\n"
    
    prompt = f"""Please analyze these {len(hotels)} hotels in {destination} and select the top 10 that best match the user's preferences.

User Profile:
- Travel Style: {travel_style}
- Budget Range: {budget_range}
- All Preferences: {preferences}
{group_context}

Hotels to analyze:
{hotel_list}

Please respond with valid JSON in this exact format:
{{
    "filtered_hotels": [
        // Array of the top 10 hotel objects (full objects, not just IDs)
    ],
    "rationale": "Brief explanation of why these hotels were selected"
}}

Consider factors like:
- Price range matching budget preferences
- Star rating and amenities matching travel style
- Location convenience
- Group accommodation needs if applicable
- Overall value for money"""

    return prompt

# ==================== HOTEL SYSTEM MESSAGE ====================

def get_hotel_system_message() -> str:
    """Generate hotel system message with current date"""
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return f"""You are Arny's professional hotel search specialist. You help users find and book hotels using the Amadeus hotel search system with intelligent group filtering.

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
