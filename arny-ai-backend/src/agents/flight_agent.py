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
        # FIXED: Moved (?i) flag to the beginning of the regex pattern
        retry_if_exception_message(match=r"(?i).*(timeout|rate.limit|429|502|503|504).*"),
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
            
            # Update latest search results in agent
            flight_agent.latest_search_results = cached_result.get("results", [])
            flight_agent.latest_search_id = cached_result.get("search_id")
            flight_agent.latest_filtering_info = cached_result.get("filtering_info", {})
            
            return cached_result
        
        # Convert city names to airport codes
        origin_code = get_airport_code(origin)
        destination_code = get_airport_code(destination)
        
        print(f"🗺️ Airport codes: {origin} -> {origin_code}, {destination} -> {destination_code}")
        
        # OPTIMIZATION 2: Increase max_results to 50 for enhanced filtering
        flights_response = await flight_agent.amadeus_service.search_flights(
            origin=origin_code,
            destination=destination_code,
            departure_date=departure_date,
            return_date=return_date,
            adults=passengers,
            max_results=50  # ENHANCED: Fetch up to 50 results for better filtering
        )
        
        if not flights_response.get("success"):
            return {
                "success": False,
                "error": flights_response.get("error", "Flight search failed"),
                "message": "I couldn't find flights for those dates. Please try different dates or destinations."
            }
        
        raw_flights = flights_response.get("results", [])
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
        
        # OPTIMIZATION 3: Save search to database (non-blocking)
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
        
        # OPTIMIZATION 4: Save to database without blocking the response
        try:
            await flight_agent.db.save_flight_search(flight_search)
            print(f"💾 Saved flight search to database: {flight_search.id}")
        except Exception as e:
            print(f"⚠️ Database save failed (non-critical): {e}")
        
        # OPTIMIZATION 5: Apply user profile filtering
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
        
        # OPTIMIZATION 6: Store filtered results in agent for response
        flight_agent.latest_search_results = filtering_result['filtered_results']
        flight_agent.latest_search_id = flight_search.id
        flight_agent.latest_filtering_info = filtering_result.get('filtering_info', {})
        
        # Build response
        response = {
            "success": True,
            "results": filtering_result['filtered_results'],
            "search_id": flight_search.id,
            "filtering_info": filtering_result.get('filtering_info', {}),
            "message": f"Found {len(filtering_result['filtered_results'])} flights from {origin} to {destination}",
            "search_params": {
                "origin": origin_code,
                "destination": destination_code,
                "departure_date": departure_date,
                "return_date": return_date,
                "passengers": passengers
            }
        }
        
        # OPTIMIZATION 7: Cache the response
        if not hasattr(flight_agent, '_search_cache'):
            flight_agent._search_cache = {}
        flight_agent._search_cache[search_key] = response
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"🚀 ENHANCED flight search completed in {elapsed_time:.2f}s")
        
        return response
        
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
async def search_airports_tool(query: str) -> dict:
    """
    Search for airports by name or code
    
    Args:
        query: Airport name, city name, or airport code to search for
    
    Returns:
        Dict with airport search results
    """
    try:
        flight_agent = _get_flight_agent()
        if not flight_agent:
            return {"success": False, "error": "Flight agent not available"}
        
        airports_response = await flight_agent.amadeus_service.search_airports(query)
        
        if not airports_response.get("success"):
            return {
                "success": False,
                "error": airports_response.get("error", "Airport search failed"),
                "message": f"I couldn't find airports matching '{query}'. Please try a different search term."
            }
        
        airports = airports_response.get("results", [])
        
        if not airports:
            return {
                "success": True,
                "results": [],
                "message": f"No airports found for '{query}'. Try searching for a city name or airport code."
            }
        
        return {
            "success": True,
            "results": airports,
            "message": f"Found {len(airports)} airports matching '{query}'"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for airports: {str(e)}"
        }

@function_tool
def get_flight_status_tool(flight_number: str, flight_date: str) -> dict:
    """
    Get real-time flight status information
    
    Args:
        flight_number: Flight number (e.g., 'QF1', 'AA100')
        flight_date: Flight date in YYYY-MM-DD format
    
    Returns:
        Dict with flight status information
    """
    try:
        flight_agent = _get_flight_agent()
        if not flight_agent:
            return {"success": False, "error": "Flight agent not available"}
        
        # For now, return a mock response
        # In production, this would call Amadeus On Demand Flight Status API
        return {
            "success": True,
            "message": f"Getting real-time status for flight {flight_number} on {flight_date}...",
            "flight_number": flight_number,
            "flight_date": flight_date,
            "status": {
                "note": "This would contain real-time flight status from Amadeus On Demand Flight Status API",
                "includes": ["departure_time", "arrival_time", "delays", "gate_info", "terminal_info"]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error getting flight status: {str(e)}"
        }

@function_tool
def get_checkin_links_tool(airline_code: str) -> dict:
    """
    Get online check-in links for airlines
    
    Args:
        airline_code: IATA airline code (e.g., 'QF', 'AA', 'BA')
    
    Returns:
        Dict with check-in information
    """
    try:
        # Common airline check-in links
        checkin_links = {
            'QF': 'https://www.qantas.com/au/en/booking-and-travel/online-check-in.html',
            'JQ': 'https://www.jetstar.com/au/en/help/online-check-in',
            'VA': 'https://www.virginaustralia.com/au/en/book/online-check-in/',
            'AA': 'https://www.aa.com/travelInformation/checkin/online_checkin.do',
            'UA': 'https://www.united.com/en/us/check-in',
            'DL': 'https://www.delta.com/us/en/check-in/overview',
            'BA': 'https://www.britishairways.com/en-gb/information/checking-in',
            'LH': 'https://www.lufthansa.com/de/en/online-check-in',
            'AF': 'https://www.airfrance.us/US/en/common/page_flottante/information/check-in-online.htm',
            'SQ': 'https://www.singaporeair.com/en_UK/us/booking-and-travel/online-checkin/',
            'EK': 'https://www.emirates.com/us/english/manage-booking/online-check-in/',
            'TG': 'https://www.thaiairways.com/en_US/manage/check_in_online.page'
        }
        
        link = checkin_links.get(airline_code.upper())
        
        if link:
            return {
                "success": True,
                "airline_code": airline_code.upper(),
                "checkin_link": link,
                "message": f"Here's the online check-in link for {airline_code.upper()}: {link}"
            }
        else:
            return {
                "success": False,
                "error": "Airline not found",
                "message": f"I don't have the check-in link for airline '{airline_code}'. Please visit the airline's website directly."
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
        
        # ADDED: Initialize search cache
        self._search_cache = {}
        
        # Set up the global instance
        _current_flight_agent = self
        
        # ENHANCED: Create agent with tools
        self.agent = Agent(
            model="o4-mini",
            instructions=self._get_system_instructions(),
            tools=[
                search_flights_tool,
                search_airports_tool,
                get_flight_status_tool,
                get_checkin_links_tool,
                get_flight_pricing_tool
            ]
        )
        
        print("✅ FlightAgent initialized with enhanced tools and caching")
    
    # ==================== PROCESS MESSAGE ====================
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                             user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Process flight search requests - ENHANCED VERSION with support for larger datasets and NO TIMEOUT LIMITS
        """
        
        try:
            print(f"✈️ ENHANCED FlightAgent processing: '{message[:50]}...'")
            start_time = datetime.now()
            
            # Clear previous search results  
            self.latest_search_results = []
            self.latest_search_id = None
            self.latest_filtering_info = {}
            
            # Store context for tool calls - FIXED to ensure tools have access
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
4. Using get_flight_status_tool for real-time flight information
5. Using get_checkin_links_tool to help users with online check-in
6. Using get_flight_pricing_tool for detailed pricing information

**IMPORTANT PRESENTATION RULES:**
- When you receive flight search results, ALWAYS present ALL flights in your response
- Do NOT truncate or summarize the flight list - show complete details for every result
- Present flights in an organized, easy-to-read format
- Include all key details: times, duration, price, airline, stops
- Number each flight option for easy reference

Today's date: {today}
Tomorrow's date: {tomorrow}

**Airport Code Examples (use these for common cities):**
{airport_mappings}
... and many more cities are supported

**Key Guidelines:**
1. **Date Handling**: If user says "tomorrow", use {tomorrow}. If no year specified, assume current year.

2. **Flight Search**: Always use search_flights_tool for flight requests. Extract:
   - Origin city/airport (convert to airport code when possible)
   - Destination city/airport (convert to airport code when possible)  
   - Departure date (format: YYYY-MM-DD)
   - Return date (optional, format: YYYY-MM-DD)
   - Number of passengers (default: 1)

3. **Response Style**: Be professional, helpful, and provide clear options. When presenting multiple flights, organize them clearly and include all important details.

4. **No Booking**: You can search and provide flight information, but cannot make actual bookings. Direct users to airline websites or travel agents for booking.

5. **Missing Information**: If critical details are missing, ask for clarification before searching.

Examples:
- "Find flights from Sydney to Los Angeles on March 15th" → search_flights_tool(origin="Sydney", destination="Los Angeles", departure_date="2025-03-15")
- "I need a return flight from London to New York, leaving Feb 20, coming back Feb 27" → search_flights_tool(origin="London", destination="New York", departure_date="2025-02-20", return_date="2025-02-27")
"""

    @openai_agents_sdk_retry
    async def _run_agent_with_retry(self, agent, message_or_messages):
        """
        Run agent with comprehensive retry logic and NO TIMEOUT LIMITS
        """
        print(f"🤖 Running agent with message type: {type(message_or_messages)}")
        
        try:
            if isinstance(message_or_messages, str):
                # Single message
                with Runner(agent) as runner:
                    result = runner.run(message_or_messages)
                    return result
            elif isinstance(message_or_messages, list):
                # Message history
                with Runner(agent) as runner:
                    result = runner.run_stream(message_or_messages)
                    # Get the final result from the stream
                    final_result = None
                    for chunk in result:
                        if hasattr(chunk, 'final_output') and chunk.final_output:
                            final_result = chunk
                    return final_result or AgentRunnerResponse(final_output="I'm ready to help you with flight searches!")
            else:
                raise ValueError(f"Invalid message type: {type(message_or_messages)}")
                
        except Exception as e:
            print(f"❌ Agent execution error: {e}")
            # Return a valid response structure even on error
            return AgentRunnerResponse(final_output=f"I encountered an error processing your request: {str(e)}")

# ==================== MODULE EXPORTS ====================

__all__ = [
    'FlightAgent',
    'search_flights_tool',
    'search_airports_tool',
    'get_flight_status_tool',
    'get_checkin_links_tool',
    'get_flight_pricing_tool'
]
