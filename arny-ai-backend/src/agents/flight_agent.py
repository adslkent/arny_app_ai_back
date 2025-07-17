"""
Flight Search Agent Module - ENHANCED VERSION with Cache Management

This module provides a flight search agent with enhanced capabilities including:
1. Support for up to 50 flight results from Amadeus API
2. Send all flights to OpenAI for filtering
3. Return up to 10 filtered flight results
4. Optimized for larger datasets
5. Cache management for improved performance

Usage example:
```python
from flight_agent import FlightAgent

# Create and use the agent
agent = FlightAgent()
result = await agent.process_message(user_id, "Find flights from Sydney to LA", session_id, {}, [])
```
"""

import uuid
import logging
import asyncio
import concurrent.futures
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from openai import OpenAI
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
from ..database.models import FlightSearch
from .user_profile_agent import UserProfileAgent

# Configure logger
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
    """Condition 5: Retry on validation failures"""
    if isinstance(result, AgentRunnerResponse):
        return result.final_output is None
    return False

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

openai_agents_sdk_retry = retry(
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

def _run_async_safely(coro):
    """Run async coroutine safely by using the current event loop or creating a new one"""
    try:
        loop = asyncio.get_running_loop()
        # If there's already a running loop, we need to run in a thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_in_new_loop, coro)
            return future.result()
    except RuntimeError:
        # No running loop, we can use asyncio.run
        return asyncio.run(coro)

def _run_in_new_loop(coro):
    """Run coroutine in a completely new event loop"""
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
        asyncio.set_event_loop(None)

# ===== ENHANCED Airport Code Mapping =====
AIRPORT_CODE_MAPPING = {
    "london": "LHR", "paris": "CDG", "newyork": "JFK", "new york": "JFK",
    "tokyo": "NRT", "beijing": "PEK", "shanghai": "PVG", "hongkong": "HKG",
    "hong kong": "HKG", "singapore": "SIN", "bangkok": "BKK", "sydney": "SYD",
    "dubai": "DXB", "losangeles": "LAX", "los angeles": "LAX", 
    "sanfrancisco": "SFO", "san francisco": "SFO", "berlin": "BER",
    "frankfurt": "FRA", "amsterdam": "AMS", "rome": "FCO", "madrid": "MAD",
    "barcelona": "BCN", "moscow": "SVO", "chicago": "ORD", "washington": "DCA",
    "boston": "BOS", "toronto": "YYZ", "vancouver": "YVR", "montreal": "YUL",
    "milan": "MXP", "vienna": "VIE", "melbourne": "MEL", "brisbane": "BNE",
    "perth": "PER", "adelaide": "ADL", "canberra": "CBR", "gold coast": "OOL",
    "cairns": "CNS", "osaka": "KIX", "miami": "MIA", "las vegas": "LAS",
    "seattle": "SEA", "zurich": "ZUR", "dublin": "DUB", "mumbai": "BOM",
    "delhi": "DEL"
}

def get_airport_code(city_name: str) -> str:
    """Get airport code for a city name"""
    city_lower = city_name.lower().replace(" ", "").replace("-", "")
    return AIRPORT_CODE_MAPPING.get(city_lower, city_name.upper()[:3])

# Global variable to store the current agent instance
_current_flight_agent = None

def _get_flight_agent():
    """Get the current flight agent instance"""
    global _current_flight_agent
    return _current_flight_agent

@function_tool
async def search_flights_tool(origin: str, destination: str, departure_date: str,
                             return_date: Optional[str] = None, passengers: int = 1) -> dict:
    """
    Search for flights using Amadeus API with enhanced profile filtering and caching
    
    Args:
        origin: Origin airport code or city name (e.g., 'SYD', 'Sydney')
        destination: Destination airport code or city name (e.g., 'LAX', 'Los Angeles')
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format (optional, for round trips)
        passengers: Number of passengers (default 1)
    
    Returns:
        Dict with search results and metadata
    """
    
    try:
        print(f"🚀 ENHANCED Flight search started: {origin} to {destination}")
        start_time = datetime.now()
        
        flight_agent = _get_flight_agent()
        if not flight_agent:
            return {"success": False, "error": "Flight agent not available"}
        
        # OPTIMIZATION 1: Check cache first
        search_key = f"{origin}_{destination}_{departure_date}_{return_date}_{passengers}"
        if hasattr(flight_agent, '_search_cache') and search_key in flight_agent._search_cache:
            print(f"⚡ Cache hit! Returning cached results for {search_key}")
            cached_result = flight_agent._search_cache[search_key]
            
            # Store cached results in agent for response
            flight_agent.latest_search_results = cached_result["filtered_results"]
            flight_agent.latest_search_id = cached_result["search_id"]
            flight_agent.latest_filtering_info = cached_result["filtering_info"]
            
            return cached_result["response"]
        
        # OPTIMIZATION 2: Convert city names to airport codes if needed
        origin_code = get_airport_code(origin) if len(origin) > 3 else origin.upper()
        destination_code = get_airport_code(destination) if len(destination) > 3 else destination.upper()
        
        print(f"🎯 Searching: {origin_code} → {destination_code} on {departure_date}")
        
        # OPTIMIZATION 3: Search flights using Amadeus API
        flight_response = await flight_agent.amadeus_service.search_flights(
            origin=origin_code,
            destination=destination_code,
            departure_date=departure_date,
            return_date=return_date,
            adults=passengers,
            max_results=50  # ENHANCED: Get up to 50 results
        )
        
        if not flight_response.get("success"):
            return {
                "success": False,
                "error": flight_response.get("error", "Flight search failed"),
                "message": "I couldn't find flights for those dates and locations. Please try different dates or destinations."
            }
        
        raw_flights = flight_response.get("results", [])
        print(f"✅ Amadeus returned {len(raw_flights)} flights")
        
        if not raw_flights:
            return {
                "success": True,
                "results": [],
                "message": f"No flights found from {origin} to {destination} on {departure_date}. Try different dates?",
                "search_params": {
                    "origin": origin_code,
                    "destination": destination_code,
                    "departure_date": departure_date,
                    "return_date": return_date,
                    "passengers": passengers
                }
            }
        
        # OPTIMIZATION 4: Save search to database (non-blocking)
        flight_search = FlightSearch(
            id=str(uuid.uuid4()),
            user_id=flight_agent.current_user_id,
            session_id=flight_agent.current_session_id,
            origin=origin_code,
            destination=destination_code,
            departure_date=departure_date,
            return_date=return_date,
            passengers=passengers,
            search_results=raw_flights,
            created_at=datetime.now()
        )
        
        # OPTIMIZATION 5: Save to database without blocking the response
        try:
            await flight_agent.db.save_flight_search(flight_search)
            print(f"💾 Saved flight search to database: {flight_search.id}")
        except Exception as e:
            print(f"⚠️ Database save failed (non-critical): {e}")
        
        # OPTIMIZATION 6: Apply user profile filtering to ALL flights
        print(f"🎯 Applying profile filtering to {len(raw_flights)} flights...")
        
        filtering_result = await flight_agent.profile_agent.filter_flight_results(
            user_id=flight_agent.current_user_id,
            flight_results=raw_flights[:50],  # ENHANCED: Process up to 50 flights
            search_params={
                "origin": origin_code,
                "destination": destination_code,
                "departure_date": departure_date,
                "return_date": return_date,
                "passengers": passengers
            }
        )
        
        print(f"✅ Profile filtering completed: {len(filtering_result['filtered_results'])} results")
        
        # OPTIMIZATION 7: Store results in agent instance for response
        flight_agent.latest_search_results = filtering_result["filtered_results"]
        flight_agent.latest_search_id = flight_search.id
        flight_agent.latest_filtering_info = {
            "original_count": len(raw_flights),
            "filtered_count": len(filtering_result["filtered_results"]),
            "filtering_applied": filtering_result.get("filtering_applied", True),
            "reasoning": filtering_result.get("reasoning", "Profile-based filtering applied")
        }
        
        # OPTIMIZATION 8: Prepare response for agent
        response_data = {
            "success": True,
            "results": filtering_result["filtered_results"],
            "message": f"Found {len(filtering_result['filtered_results'])} flights from {origin} to {destination}",
            "search_id": flight_search.id,
            "search_params": {
                "origin": origin_code,
                "destination": destination_code,
                "departure_date": departure_date,
                "return_date": return_date,
                "passengers": passengers
            },
            "filtering_info": flight_agent.latest_filtering_info
        }
        
        # OPTIMIZATION 9: Cache the results for future use
        if hasattr(flight_agent, '_search_cache'):
            flight_agent._search_cache[search_key] = {
                "response": response_data,
                "filtered_results": filtering_result["filtered_results"],
                "search_id": flight_search.id,
                "filtering_info": flight_agent.latest_filtering_info,
                "timestamp": datetime.now()
            }
            print(f"📦 Cached search results for key: {search_key}")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"⚡ ENHANCED flight search completed in {elapsed_time:.2f}s")
        
        return response_data
        
    except Exception as e:
        print(f"❌ Error in flight search: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for flights: {str(e)}"
        }

@function_tool
def search_airports_tool(query: str) -> dict:
    """
    Search for airport information by city or airport name
    
    Args:
        query: City name or airport name to search for
    
    Returns:
        Dict with airport information
    """
    try:
        print(f"🔍 Airport search for: {query}")
        
        # Check in our mapping first
        query_lower = query.lower().replace(" ", "").replace("-", "")
        airport_code = AIRPORT_CODE_MAPPING.get(query_lower)
        
        if airport_code:
            return {
                "success": True,
                "airports": [{
                    "code": airport_code,
                    "city": query.title(),
                    "name": f"{query.title()} Airport"
                }],
                "message": f"Found airport code {airport_code} for {query}"
            }
        else:
            # Return a generic response for unknown airports
            return {
                "success": True,
                "airports": [],
                "message": f"I don't have airport information for '{query}'. Please try a major city name or provide the 3-letter airport code directly."
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for airports: {str(e)}"
        }

@function_tool
def get_checkin_links_tool(airline_code: str) -> dict:
    """
    Get check-in links for specific airlines
    
    Args:
        airline_code: 2-letter airline code (e.g., 'QF', 'UA')
    
    Returns:
        Dict with check-in information
    """
    try:
        # Common airline check-in links
        checkin_links = {
            "QF": "https://www.qantas.com/au/en/flight-status/check-in.html",
            "UA": "https://www.united.com/en/us/checkin",
            "AA": "https://www.aa.com/checkin",
            "DL": "https://www.delta.com/us/en/check-in/overview",
            "BA": "https://www.britishairways.com/travel/managebooking/public/en_us",
            "LH": "https://www.lufthansa.com/us/en/online-check-in",
            "AF": "https://www.airfrance.us/US/en/common/transverse/check-in/",
            "KL": "https://www.klm.com/checkin",
            "EK": "https://www.emirates.com/us/english/manage-booking/online-check-in/"
        }
        
        airline_code_upper = airline_code.upper()
        
        if airline_code_upper in checkin_links:
            return {
                "success": True,
                "airline_code": airline_code_upper,
                "checkin_url": checkin_links[airline_code_upper],
                "message": f"Here's the check-in link for {airline_code_upper}: {checkin_links[airline_code_upper]}"
            }
        else:
            return {
                "success": True,
                "airline_code": airline_code_upper,
                "checkin_url": None,
                "message": f"I don't have a specific check-in link for airline {airline_code_upper}, but "
                          f"you can usually find it by searching '{airline_code_upper} check in' in your browser."
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error getting check-in links: {str(e)}"
        }

@function_tool
def get_flight_pricing_tool(flight_number: str) -> dict:
    """
    Get detailed pricing for a specific flight
    
    Args:
        flight_number: Flight number or position from search results (e.g., "1", "flight 2")
    
    Returns:
        Dict with detailed flight pricing information
    """
    try:
        # Extract number from flight selection
        import re
        numbers = re.findall(r'\d+', flight_number)
        if not numbers:
            return {
                "success": False,
                "error": "Invalid flight selection",
                "message": "I need a flight number to get pricing. Please specify like 'flight 1' or 'the first option'."
            }
        
        # For now, return a detailed pricing response
        # In production, this would call Amadeus Flight Offers Price API
        return {
            "success": True,
            "message": f"Getting accurate pricing for flight {flight_number}...",
            "flight_number": flight_number,
            "pricing_info": {
                "note": "This would contain detailed pricing from Amadeus Flight Offers Price API",
                "includes": ["taxes", "fees", "cancellation_policy", "baggage_allowance"]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error getting flight pricing: {str(e)}"
        }

# ==================== FLIGHT AGENT CLASS ====================

class FlightAgent:
    """
    Flight agent using OpenAI Agents SDK with Amadeus API tools, profile filtering, and cache management
    """
    
    def __init__(self):
        global _current_flight_agent
        
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.amadeus_service = AmadeusService()
        self.db = DatabaseOperations()
        self.profile_agent = UserProfileAgent()
        
        # FIXED: Initialize context variables
        self.current_user_id = None
        self.current_session_id = None
        self.user_profile = None
        
        # FIXED: Add attributes to store latest search results for response
        self.latest_search_results = []
        self.latest_search_id = None
        self.latest_filtering_info = {}
        
        # ADDED: Initialize search cache for enhanced performance (matching hotel agent)
        self._search_cache = {}
        
        # Store this instance globally for tool access
        _current_flight_agent = self
        
        # Create the agent with enhanced flight search tools
        self.agent = Agent(
            name="Arny Flight Assistant",
            instructions=self._get_system_instructions(),
            model="o4-mini",
            tools=[
                search_flights_tool,
                search_airports_tool,
                get_checkin_links_tool,
                get_flight_pricing_tool
            ]
        )
    
    @openai_agents_sdk_retry
    async def _run_agent_with_retry(self, agent, input_data):
        """Run agent with retry logic applied"""
        return await Runner.run(agent, input_data)
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                            user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Process flight search requests using OpenAI Agents SDK with profile filtering
        """
        
        try:
            print(f"✈️ FlightAgent processing message: '{message[:50]}...'")
            
            # FIXED: Clear previous search results at start of new message
            self.latest_search_results = []
            self.latest_search_id = None
            self.latest_filtering_info = {}
            
            # FIXED: Store context for tool calls on the current instance
            self.current_user_id = user_id
            self.current_session_id = session_id
            self.user_profile = user_profile
            
            # FIXED: Also update the global instance to ensure tools have access
            global _current_flight_agent
            _current_flight_agent.current_user_id = user_id
            _current_flight_agent.current_session_id = session_id
            _current_flight_agent.user_profile = user_profile
            # FIXED: Clear global instance results as well
            _current_flight_agent.latest_search_results = []
            _current_flight_agent.latest_search_id = None
            _current_flight_agent.latest_filtering_info = {}
            
            print(f"🔧 Context set: user_id={user_id}, session_id={session_id}")
            
            # Build conversation context
            context_messages = []
            
            # UNIFIED CONTEXT: Add recent conversation history (last 50 messages)
            for msg in conversation_history[-50:]:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            print(f"🔧 Processing with {len(context_messages)} previous messages")
            
            # Process with agent using retry logic
            if not context_messages:
                # First message in conversation
                print("🚀 Starting new flight conversation")
                result = await self._run_agent_with_retry(self.agent, message)
            else:
                # Continue conversation with context
                print("🔄 Continuing flight conversation with context")
                result = await self._run_agent_with_retry(self.agent, context_messages + [{"role": "user", "content": message}])
          
            # Extract response
            assistant_message = result.final_output
            
            print(f"✅ FlightAgent response generated: '{assistant_message[:50]}...'")
            
            # FIXED: Read search results from the global instance that tools updated
            global_agent = _get_flight_agent()
            search_results = global_agent.latest_search_results if global_agent else []
            search_id = global_agent.latest_search_id if global_agent else None
            filtering_info = global_agent.latest_filtering_info if global_agent else {}
            
            print(f"📊 Retrieved search data: {len(search_results)} results, search_id: {search_id}")
            
            # FIXED: Include search results and search ID in response from global instance
            return {
                "message": assistant_message,
                "agent_type": "flight",
                "requires_action": False,  # Will be set to True if flight selection is needed
                "search_results": search_results,
                "search_id": search_id,
                "filtering_info": filtering_info,
                "metadata": {
                    "agent_type": "flight",
                    "conversation_type": "flight_search"
                }
            }
        
        except Exception as e:
            print(f"❌ Error in flight agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": "I'm sorry, I encountered an error while searching for flights. Please try again.",
                "agent_type": "flight",
                "error": str(e),
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {}
            }
    
    # ==================== HELPER METHODS ====================
    
    def _get_system_instructions(self) -> str:
        """Generate enhanced system instructions with current date and airport code mapping"""
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Format airport code mapping as string (showing first 20 examples)
        airport_mappings = "\n".join([f"- {city}: {code}" for city, code in list(AIRPORT_CODE_MAPPING.items())[:20]])
        
        return f"""You are Arny's professional flight search specialist. You help users find and book flights using the Amadeus flight search system with intelligent group filtering.

Your main responsibilities are:
1. Understanding users' flight needs and extracting key information from natural language descriptions
2. Using the search_flights_tool to search for flights that meet user requirements
3. Using search_airports_tool when users need airport information or codes
4. Using get_checkin_links_tool when users ask about checking in for flights
5. Using get_flight_pricing_tool when users need detailed pricing information

Current date: {today}
Tomorrow's date: {tomorrow}

**Airport Code Mapping (top 20):**
{airport_mappings}

**Key Rules:**
1. ALWAYS use search_flights_tool for ANY flight request with dates and destinations
2. Convert city names to airport codes using the mapping above when possible
3. DEFAULT: passengers=1 for searches unless specified otherwise
4. For round trips, ask for return date if not provided
5. **IMPORTANT: Present ALL filtered flight results in your response - do not truncate the list**
6. Show flight details including airline, departure/arrival times, duration, and price for ALL results
7. If users ask about specific flight numbers or want pricing details, use get_flight_pricing_tool
8. For check-in requests, use get_checkin_links_tool with the airline code

**Response Style:**
- Be professional, helpful, and thorough
- Always show comprehensive flight options
- Explain any filtering that was applied based on user preferences
- Provide clear next steps (e.g., "Would you like me to get detailed pricing for any of these flights?")

Example: "Flights from Sydney to Los Angeles on March 15" → search_flights_tool(origin="SYD", destination="LAX", departure_date="2025-03-15")

Remember: Your goal is to find the perfect flights for each user's unique travel needs!"""

# ==================== MODULE EXPORTS ====================

__all__ = [
    'FlightAgent',
    'search_flights_tool',
    'search_airports_tool',
    'get_checkin_links_tool',
    'get_flight_pricing_tool'
]
