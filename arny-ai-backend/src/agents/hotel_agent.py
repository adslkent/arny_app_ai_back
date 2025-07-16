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
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from agents import Agent, function_tool, Runner

from ..utils.config import config
from ..services.amadeus_service import AmadeusService
from ..database.operations import DatabaseOperations
from ..database.models import HotelSearch
from .user_profile_agent import UserProfileAgent

# Configure logging
logger = logging.getLogger(__name__)

# ===== ENHANCED City Code Mapping =====
CITY_CODE_MAPPING = {
    "london": "LON", "paris": "PAR", "newyork": "NYC", "new york": "NYC",
    "tokyo": "TYO", "beijing": "PEK", "shanghai": "SHA", "hongkong": "HKG",
    "hong kong": "HKG", "singapore": "SIN", "bangkok": "BKK", "sydney": "SYD",
    "dubai": "DXB", "losangeles": "LAX", "los angeles": "LAX",
    "sanfrancisco": "SFO", "san francisco": "SFO", "berlin": "BER",
    "frankfurt": "FRA", "amsterdam": "AMS", "rome": "ROM", "madrid": "MAD",
    "barcelona": "BCN", "moscow": "MOW", "chicago": "CHI", "washington": "WAS",
    "boston": "BOS", "toronto": "YTO", "vancouver": "YVR", "montreal": "YMQ",
    "milan": "MIL", "vienna": "VIE", "melbourne": "MEL", "brisbane": "BNE",
    "perth": "PER", "adelaide": "ADL", "canberra": "CBR", "gold coast": "OOL",
    "cairns": "CNS", "osaka": "OSA", "miami": "MIA", "las vegas": "LAS",
    "seattle": "SEA", "zurich": "ZUR", "dublin": "DUB", "mumbai": "BOM",
    "delhi": "DEL"
}

# Global variable to store the current agent instance
_current_hotel_agent = None

def _get_hotel_agent():
    """Get the current hotel agent instance"""
    global _current_hotel_agent
    return _current_hotel_agent

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

def get_hotel_system_message() -> str:
    """
    Generate enhanced hotel search assistant system prompt
    """
    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Enhanced system message for better hotel search
    system_message = f"""You are Arny's professional hotel search assistant. Today is {today}.

Your task: Use search_hotels_tool to find hotels with dates and pricing for users.

Key rules:
1. ALWAYS use search_hotels_tool for ANY hotel request with dates
2. If no check-out date provided, use one day after check-in
3. Convert city names to codes: NYC, LON, PAR, etc.
4. DEFAULT: adults=1, rooms=1
5. NEVER call the same tool twice with different city variations
6. Present results clearly with hotel names, ratings, and prices

Example: "Hotels in New York July 15-18" → search_hotels_tool(destination="New York", check_in_date="2025-07-15", check_out_date="2025-07-18")

Be professional and helpful. Provide comprehensive hotel options."""
    
    return system_message

@function_tool
async def search_hotels_tool(destination: str, check_in_date: str, check_out_date: Optional[str] = None,
                            adults: int = 1, rooms: int = 1) -> dict:
    """
    Search for hotels - ENHANCED VERSION with support for larger datasets
    
    Args:
        destination: Destination city (e.g., 'Sydney', 'NYC')
        check_in_date: Check-in date in YYYY-MM-DD format
        check_out_date: Check-out date in YYYY-MM-DD format, defaults to one day after check-in
        adults: Number of adults (default 1)
        rooms: Number of rooms (default 1)
    
    Returns:
        Dict with hotel search results and profile filtering information
    """
    
    print(f"🚀 ENHANCED Hotel search started for {destination}")
    start_time = datetime.now()
    
    try:
        hotel_agent = _get_hotel_agent()
        if not hotel_agent:
            return {"success": False, "error": "Hotel agent not available"}
        
        # OPTIMIZATION 1: Check if this exact search was already done
        search_key = f"{destination}_{check_in_date}_{check_out_date}_{adults}_{rooms}"
        if hasattr(hotel_agent, '_search_cache') and search_key in hotel_agent._search_cache:
            print(f"⚡ CACHE HIT: Returning cached results for {search_key}")
            return hotel_agent._search_cache[search_key]
        
        # OPTIMIZATION 2: Check if context is properly set
        if not hasattr(hotel_agent, 'current_user_id') or not hotel_agent.current_user_id:
            return {"success": False, "error": "User context not available"}
        
        print(f"🔍 Processing: {destination} for {check_in_date} to {check_out_date}")
        
        # OPTIMIZATION 3: Enhanced city code conversion
        city_code = hotel_agent._convert_to_city_code_enhanced(destination)
        print(f"⚡ Enhanced conversion: {destination} → {city_code}")
        
        # OPTIMIZATION 4: Handle default check-out date
        if not check_out_date:
            try:
                check_in_dt = datetime.strptime(check_in_date, "%Y-%m-%d")
                check_out_dt = check_in_dt + timedelta(days=1)
                check_out_date = check_out_dt.strftime("%Y-%m-%d")
            except ValueError:
                return {
                    "success": False,
                    "error": "Invalid check_in_date format",
                    "message": "Please provide check-in date in YYYY-MM-DD format"
                }
        
        # ENHANCED: Amadeus API call with increased result limit
        print(f"🏨 Calling Amadeus API...")
        
        search_results = _run_async_safely(
            hotel_agent.amadeus_service.search_hotels(
                city_code=city_code,
                check_in_date=check_in_date,
                check_out_date=check_out_date,
                adults=adults,
                rooms=rooms,
                max_results=50  # CHANGED: Increased from 6 to 50 results
            )
        )
        
        print(f"📊 Amadeus API response: success={search_results.get('success')}, results={len(search_results.get('results', []))}")
        
        if not search_results.get("success"):
            print(f"❌ Amadeus API error: {search_results.get('error')}")
            return {
                "success": False,
                "error": search_results.get("error", "Hotel search failed"),
                "message": f"Sorry, I couldn't find hotels in {destination} for {check_in_date} to {check_out_date}."
            }
        
        # ENHANCED: Profile filtering with all results
        search_params = {
            "city_code": city_code,
            "destination": destination,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "rooms": rooms
        }
        
        print(f"🔧 Profile filtering with enhanced dataset...")
        
        filtering_result = _run_async_safely(
            hotel_agent.profile_agent.filter_hotel_results(
                user_id=hotel_agent.current_user_id,
                hotel_results=search_results["results"],  # CHANGED: Send all hotels (up to 50)
                search_params=search_params
            )
        )
        
        print(f"✅ Filtering complete: {filtering_result['filtered_count']} of {filtering_result['original_count']} results")
        
        # OPTIMIZATION 7: Enhanced database save
        hotel_search = HotelSearch(
            id=str(uuid.uuid4()),
            user_id=hotel_agent.current_user_id,
            city_code=city_code,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            adults=adults,
            rooms=rooms,
            search_results=search_results["results"],  # Save original results
            result_count=len(filtering_result["filtered_results"]),
            search_successful=True
        )
        
        print(f"💾 Saving search to database...")
        # Save asynchronously for better performance
        asyncio.create_task(hotel_agent.db.save_hotel_search(hotel_search))
        
        # OPTIMIZATION 8: Store search results on agent instance
        hotel_agent.latest_search_results = filtering_result["filtered_results"]
        hotel_agent.latest_search_id = hotel_search.id
        hotel_agent.latest_filtering_info = {
            "original_count": filtering_result["original_count"],
            "filtered_count": filtering_result["filtered_count"],
            "filtering_applied": filtering_result["filtering_applied"],
            "group_size": filtering_result.get("group_size", 1),
            "rationale": filtering_result["rationale"]
        }
        
        # OPTIMIZATION 9: Create enhanced result payload
        result_payload = {
            "success": True,
            "results": filtering_result["filtered_results"],
            "search_id": hotel_search.id,
            "search_params": search_params,
            "filtering_info": hotel_agent.latest_filtering_info
        }
        
        # OPTIMIZATION 10: Cache the result
        if not hasattr(hotel_agent, '_search_cache'):
            hotel_agent._search_cache = {}
        hotel_agent._search_cache[search_key] = result_payload
        
        # Keep cache manageable (max 10 entries)
        if len(hotel_agent._search_cache) > 10:
            oldest_key = list(hotel_agent._search_cache.keys())[0]
            del hotel_agent._search_cache[oldest_key]
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ ENHANCED Hotel search completed in {elapsed_time:.2f}s!")
        
        return result_payload
        
    except Exception as e:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"❌ Error in search_hotels_tool after {elapsed_time:.2f}s: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for hotels: {str(e)}"
        }

class HotelAgent:
    """
    Hotel agent with enhanced capabilities for larger datasets
    """
    
    def __init__(self):
        global _current_hotel_agent
        
        self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.amadeus_service = AmadeusService()
        self.db = DatabaseOperations()
        self.profile_agent = UserProfileAgent()
        
        # Initialize context variables
        self.current_user_id = None
        self.current_session_id = None
        self.user_profile = None
        
        # Initialize search result storage
        self.latest_search_results = []
        self.latest_search_id = None
        self.latest_filtering_info = {}
        
        # Initialize search cache for enhanced performance
        self._search_cache = {}
        
        # Set this instance as the global instance for tools
        _current_hotel_agent = self
        
        # Create the agent with enhanced settings
        self.agent = Agent(
            name="Arny Hotel Assistant", 
            instructions=get_hotel_system_message(),
            model="o4-mini",  # Use efficient model
            tools=[search_hotels_tool]
        )
    
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
            for msg in conversation_history[-8:]:  # Last 8 messages for context
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            print(f"🔧 Processing with {len(context_messages)} previous messages")
            
            # ENHANCED: Direct agent processing with improved efficiency
            if not context_messages:
                # First message in conversation
                print("🚀 Starting new hotel conversation")
                result = await Runner.run(self.agent, message)
            else:
                # Continue conversation with context
                print("🔄 Continuing hotel conversation with context")
                result = await Runner.run(self.agent, context_messages + [{"role": "user", "content": message}])
            
            # Extract response
            assistant_message = result.final_output
            
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
                "message": "I'm sorry, I encountered an error while searching for hotels. Please try again.",
                "agent_type": "hotel",
                "error": str(e),
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {}
            }
    
    def _convert_to_city_code_enhanced(self, location: str) -> str:
        """ENHANCED: Enhanced city code conversion with better mapping"""
        
        location_lower = location.lower().strip().replace(" ", "")
        
        # Direct lookup in enhanced mapping
        if location_lower in CITY_CODE_MAPPING:
            return CITY_CODE_MAPPING[location_lower]
        
        # Try with spaces for exact matches
        location_lower_with_spaces = location.lower().strip()
        if location_lower_with_spaces in CITY_CODE_MAPPING:
            return CITY_CODE_MAPPING[location_lower_with_spaces]
        
        # If already looks like city code (3 letters)
        if len(location) == 3 and location.isalpha():
            return location.upper()
        
        # Enhanced fallback with partial matching
        for city, code in CITY_CODE_MAPPING.items():
            if city in location_lower or location_lower in city:
                return code
        
        # Default: return as-is
        return location.upper()
