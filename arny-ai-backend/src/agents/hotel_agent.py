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
from datetime import datetime, timedelta, date as date_type

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
    """Condition 5: Retry if validation fails for OpenAI responses"""
    if isinstance(result, dict) and result.get("success"):
        try:
            AgentRunnerResponse(**result)
            return False  # Validation passed
        except ValidationError:
            return True  # Validation failed, retry
        except Exception as e:
            logger.warning(f"Unexpected validation error: {e}")
            return True
    return False

def retry_on_openai_exception_type(exception):
    """Custom exception checker for OpenAI-specific exceptions"""
    return isinstance(exception, (requests.exceptions.RequestException, ConnectionError, TimeoutError))

# ==================== COMBINED RETRY STRATEGIES FOR OPENAI ====================

# Primary retry strategy for OpenAI Agent operations
openai_agent_retry = retry(
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
    stop=stop_after_attempt(3),  # Fewer attempts for OpenAI API
    wait=wait_exponential(multiplier=1, min=1, max=10),  # Shorter wait times
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# ==================== ASYNC EXECUTION HELPERS ====================

def _run_sync_in_async(coro):
    """Run coroutine in sync context"""
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
        
        if not search_results or not search_results.get("success"):
            return {
                "success": False,
                "error": search_results.get("error", "No hotels found"),
                "message": f"I couldn't find any hotels in {destination} for your dates. Please try different dates or destinations."
            }
        
        hotels = search_results.get("results", [])  # Fixed: use 'results' from amadeus service
        if not hotels:
            return {
                "success": False,
                "error": "No hotels found",
                "message": f"I couldn't find any hotels in {destination} for your dates. Please try different dates or destinations."
            }
        
        print(f"🔍 Found {len(hotels)} raw hotels from Amadeus")
        
        # OPTIMIZATION 5: Store in database with enhanced metadata
        user_id = hotel_agent.current_user_id
        session_id = hotel_agent.current_session_id
        
        hotel_search = HotelSearch(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            city_code=destination_code,  # Fixed: use city_code instead of destination
            check_in_date=date_type.fromisoformat(check_in_date),  # Fixed: convert to date object
            check_out_date=date_type.fromisoformat(check_out_date),  # Fixed: convert to date object
            adults=adults,
            rooms=rooms,
            search_results=hotels,
            # search_timestamp removed - model uses created_at automatically
        )
        
        try:
            save_success = await hotel_agent.db.save_hotel_search(hotel_search)
            if save_success:
                print(f"💾 Saved search with ID: {hotel_search.id}")
            else:
                print(f"⚠️ Failed to save hotel search to database")
        except Exception as e:
            print(f"⚠️ Failed to save hotel search: {e}")
        
        # OPTIMIZATION 6: ENHANCED filtering with profile agent (send ALL hotels for filtering)
        print(f"🧠 Filtering {len(hotels)} hotels with group profiles...")
        filtering_start = datetime.now()
        
        try:
            filtering_result = await hotel_agent.profile_agent.filter_hotels_enhanced(
                hotels=hotels,  # Send ALL hotels for filtering
                user_profile=hotel_agent.user_profile
            )
            
            filtered_hotels = filtering_result.get('filtered_results', hotels[:10])  # Fallback to first 10
            filtering_info = filtering_result.get('filtering_info', {})
            
        except Exception as e:
            print(f"⚠️ Filtering failed, using top 10 hotels: {e}")
            filtered_hotels = hotels[:10]  # Use first 10 hotels as fallback
            filtering_info = {"filtering_applied": False, "error": str(e)}
        
        filtering_time = (datetime.now() - filtering_start).total_seconds()
        print(f"🎯 Profile filtering completed in {filtering_time:.2f}s")
        
        print(f"✅ Filtered to {len(filtered_hotels)} hotels based on user preferences")
        
        # OPTIMIZATION 7: Update instance variables for response (FIXED to use filtered results)
        hotel_agent.latest_search_results = filtered_hotels
        hotel_agent.latest_search_id = hotel_search.id
        hotel_agent.latest_filtering_info = filtering_info
        
        # OPTIMIZATION 8: Build response
        response = {
            "success": True,
            "results": filtered_hotels,
            "search_id": hotel_search.id,
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
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        self.agent = Agent(
            name="Hotel Search Assistant",
            model="o1-mini",
            instructions=f"""You help users find and book hotels using the Amadeus hotel search system with intelligent group filtering.

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

Remember: You are a specialized hotel search agent. Focus on hotel-related queries and use your tools effectively to provide the best accommodation options for users.""",
            functions=[search_hotels_tool]
        )
        
        logger.info("✅ HotelAgent initialized with enhanced tools and caching")
    
    def set_context(self, user_id: str, session_id: str = None, user_profile: dict = None):
        """Set context for tool calls"""
        self.current_user_id = user_id
        self.current_session_id = session_id or str(uuid.uuid4())
        self.user_profile = user_profile or {}
        
        logger.info(f"🔧 Context set: user_id={user_id}")
    
    def _format_hotel_results_for_agent(self, results: List[Dict[str, Any]], 
                                      destination: str, check_in_date: str, check_out_date: str,
                                      filtering_info: Dict[str, Any]) -> str:
        """Format hotel results for the AI agent to present naturally with filtering information"""
        
        if not results:
            return f"No hotels found in {destination} for {check_in_date} to {check_out_date}."
        
        # Start with filtering information if applied
        formatted = ""
        if filtering_info.get("filtering_applied"):
            group_size = filtering_info.get("group_size", 1)
            original_count = filtering_info.get("original_count", 0)
            filtered_count = len(results)
            
            if group_size > 1:
                formatted += f"🏠 **Group Travel Filtering Applied**\n"
                formatted += f"I analyzed {original_count} hotels for your group of {group_size} travelers and "
                formatted += f"selected the {filtered_count} best options based on your group's preferences.\n\n"
                formatted += f"*{filtering_info.get('rationale', 'Filtered for group compatibility')}*\n\n"
        
        formatted += f"I found {len(results)} hotels in {destination} for {check_in_date} to {check_out_date}:\n\n"
        
        # Show up to 10 results
        for i, hotel in enumerate(results[:10], 1):
            hotel_info = hotel.get('hotel', {})
            offers = hotel.get('offers', [])
            
            # Basic hotel information
            formatted += f"**{i}. {hotel_info.get('name', 'Hotel Name Not Available')}**\n"
            
            # Location and rating
            if hotel_info.get('cityCode'):
                formatted += f"📍 Location: {hotel_info.get('cityCode')}\n"
            
            if hotel_info.get('rating'):
                formatted += f"⭐ Rating: {hotel_info.get('rating')}/5\n"
            
            # Price information from offers
            if offers and len(offers) > 0:
                offer = offers[0]
                price = offer.get('price', {})
                if price.get('total'):
                    currency = price.get('currency', 'USD')
                    formatted += f"💰 Price: {price['total']} {currency}\n"
                
                # Room information
                room = offer.get('room', {})
                if room.get('type'):
                    formatted += f"🛏️ Room: {room['type']}\n"
                
                # Check if cancellation is free
                policies = offer.get('policies', {})
                if policies.get('cancellation', {}).get('numberOfNights') == 0:
                    formatted += f"✅ Free cancellation\n"
            
            # Hotel amenities
            amenities = hotel_info.get('amenities', [])
            if amenities:
                amenity_names = [amenity.get('name', '') for amenity in amenities[:3]]
                if amenity_names:
                    formatted += f"🏨 Amenities: {', '.join(filter(None, amenity_names))}\n"
            
            formatted += "\n"
        
        if len(results) > 10:
            formatted += f"... and {len(results) - 10} more hotels available.\n"
        
        return formatted
    
    @openai_agent_retry
    async def process_message(self, user_id: str, message: str, session_id: str = None, 
                            user_profile: dict = None, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process user message with enhanced retry strategies and profile filtering
        """
        
        try:
            # Set context for tools
            self.set_context(user_id, session_id, user_profile)
            
            # Reset search results
            self.latest_search_results = []
            self.latest_search_id = None
            self.latest_filtering_info = {}
            
            logger.info(f"🔧 Processing with {len(conversation_history) if conversation_history else 0} previous messages")
            
            # Prepare conversation history for the agent
            messages = []
            if conversation_history:
                for msg in conversation_history[-10:]:  # Limit to last 10 messages
                    role = "user" if msg.get("message_type") == "user" else "assistant"
                    messages.append({"role": role, "content": msg.get("content", "")})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Run the agent
            if conversation_history:
                logger.info("🔄 Continuing hotel conversation...")
                # For continuing conversations, use the existing thread
                response = await asyncio.to_thread(
                    self.agent.run, 
                    message, 
                    additional_messages=messages[:-1]  # Exclude the current message as it's passed separately
                )
            else:
                logger.info("🚀 Starting new hotel conversation")
                # For new conversations, start fresh
                response = await asyncio.to_thread(self.agent.run, message)
            
            # Validate response using Pydantic
            try:
                validated_response = AgentRunnerResponse(final_output=response.final_output)
                logger.info("Agent response validation successful")
            except ValidationError as ve:
                logger.warning(f"Agent response validation failed: {ve}")
                # Continue with unvalidated response as fallback
            
            # Get the final response
            final_output = response.final_output
            
            # If no specific response and we have search results, format them
            if not final_output and self.latest_search_results:
                final_output = "I found some hotels for you. Please see the search results above."
            
            return {
                "response": final_output,
                "search_results": self.latest_search_results,
                "search_id": self.latest_search_id,
                "filtering_info": self.latest_filtering_info,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Hotel agent error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "response": "I'm sorry, but I encountered an unexpected error while searching for hotels. Would you like me to try again? If you'd prefer, I can also adjust the parameters or check availability for different dates. Let me know how you'd like to proceed!",
                "search_results": [],
                "search_id": None,
                "filtering_info": {},
                "success": False,
                "error": str(e)
            }
