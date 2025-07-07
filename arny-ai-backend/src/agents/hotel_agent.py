"""
Hotel Search Agent Module - ULTRA-OPTIMIZED VERSION to prevent timeouts

This module provides a hotel search agent with timeout prevention optimizations:
1. Prevents duplicate API calls through smart caching
2. Ultra-fast city name conversion
3. 10-second timeout limits on all AI calls
4. Multiple fallback layers for reliability

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

# ===== FIXED: Updated City Code Mapping for Hotel Search =====
CITY_CODE_MAPPING = {
    # Major US cities
    "newyork": "NYC", "new york": "NYC", "nyc": "NYC", "manhattan": "NYC",
    "losangeles": "LAX", "los angeles": "LAX", "la": "LAX",
    "sanfrancisco": "SFO", "san francisco": "SFO", "sf": "SFO",
    "chicago": "CHI", "washington": "WAS", "washington dc": "WAS",
    "boston": "BOS", "miami": "MIA", "las vegas": "LAS", "seattle": "SEA",
    
    # Major European cities
    "london": "LON", "paris": "PAR", "berlin": "BER", "frankfurt": "FRA",
    "amsterdam": "AMS", "rome": "ROM", "madrid": "MAD", "barcelona": "BCN",
    "milan": "MIL", "vienna": "VIE", "zurich": "ZUR", "dublin": "DUB",
    "brussels": "BRU", "copenhagen": "CPH", "stockholm": "STO", "oslo": "OSL",
    "helsinki": "HEL", "lisbon": "LIS", "athens": "ATH", "munich": "MUC",
    
    # Major Asian cities
    "tokyo": "TYO", "beijing": "PEK", "shanghai": "SHA", "hongkong": "HKG",
    "hong kong": "HKG", "singapore": "SIN", "bangkok": "BKK", "seoul": "SEL",
    "taipei": "TPE", "osaka": "OSA", "mumbai": "BOM", "delhi": "DEL",
    "newdelhi": "DEL", "new delhi": "DEL", "kualalumpur": "KUL", "kuala lumpur": "KUL",
    "jakarta": "CGK", "manila": "MNL",
    
    # Major Australian cities
    "sydney": "SYD", "melbourne": "MEL", "brisbane": "BNE", "perth": "PER",
    "adelaide": "ADL", "canberra": "CBR", "gold coast": "OOL", "cairns": "CNS",
    
    # Other major cities
    "dubai": "DXB", "toronto": "YTO", "vancouver": "YVR", "montreal": "YMQ",
    "moscow": "MOW", "saopaulo": "SAO", "rio de janeiro": "RIO", "riodejaneiro": "RIO",
    "buenosaires": "BUE", "buenos aires": "BUE", "johannesburg": "JNB", "cairo": "CAI",
    "istanbul": "IST"
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
    Generate ultra-optimized hotel search assistant system prompt
    """
    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Ultra-short system message to reduce processing time
    system_message = f"""You are Arny's hotel search assistant. Today is {today}.

Your task: Use search_hotels_tool to find hotels with dates and pricing.

Key rules:
1. ALWAYS use search_hotels_tool for ANY hotel request with dates
2. If no check-out date provided, use one day after check-in
3. Convert city names to standard codes (NYC, LON, PAR, etc.)
4. DEFAULT: adults=1, rooms=1
5. NEVER call the same tool twice with different city variations

Example: "Hotels in New York July 15-18" → search_hotels_tool(destination="New York", check_in_date="2025-07-15", check_out_date="2025-07-18")

Be fast and efficient. One tool call per request."""
    
    return system_message

@function_tool
async def search_hotels_tool(destination: str, check_in_date: str, check_out_date: Optional[str] = None,
                            adults: int = 1, rooms: int = 1) -> dict:
    """
    ULTRA-OPTIMIZED: Search for hotels with timeout prevention and better error handling
    
    Args:
        destination: Destination city (e.g., 'Sydney', 'New York', 'London')
        check_in_date: Check-in date in YYYY-MM-DD format
        check_out_date: Check-out date in YYYY-MM-DD format, defaults to one day after check-in
        adults: Number of adults (default 1)
        rooms: Number of rooms (default 1)
    
    Returns:
        Dict with hotel search results and profile filtering information
    """
    
    print(f"🚀 ULTRA-OPTIMIZED: Hotel search started for {destination}")
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
        
        # OPTIMIZATION 3: Ultra-fast city code conversion with better fallbacks
        city_code = hotel_agent._convert_to_city_code_ultra_fast(destination)
        print(f"⚡ Fast conversion: {destination} → {city_code}")
        
        # OPTIMIZATION 4: Handle default check-out date instantly
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
        
        # OPTIMIZATION 5: Ultra-fast Amadeus API call with 8-second timeout
        print(f"🏨 Calling Amadeus API with 8s timeout...")
        
        try:
            # Set 8-second timeout for Amadeus API
            api_timeout_task = asyncio.create_task(asyncio.sleep(8))
            amadeus_task = asyncio.create_task(
                hotel_agent.amadeus_service.search_hotels(
                    city_code=city_code,
                    check_in_date=check_in_date,
                    check_out_date=check_out_date,
                    adults=adults,
                    rooms=rooms,
                    max_results=6  # Limit to 6 for ultra-speed
                )
            )
            
            done, pending = await asyncio.wait(
                [api_timeout_task, amadeus_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            if amadeus_task in done:
                search_results = amadeus_task.result()
            else:
                print(f"⚠️ Amadeus API timeout after 8s")
                return {
                    "success": False,
                    "error": "Amadeus API timeout",
                    "message": f"Hotel search timed out. Please try a different destination or check your dates."
                }
                
        except Exception as amadeus_error:
            print(f"❌ Amadeus API error: {amadeus_error}")
            # FIXED: Better error messaging based on error content
            error_message = str(amadeus_error)
            if "No hotels found for city code" in error_message:
                return {
                    "success": False,
                    "error": f"No hotels found for '{destination}'",
                    "message": f"I couldn't find hotels in {destination}. Please try:\n• A major city name (e.g., 'New York', 'London', 'Paris')\n• Different spelling or variation\n• A nearby major city"
                }
            elif "Could not find hotels for city" in error_message:
                return {
                    "success": False,
                    "error": f"Invalid destination '{destination}'",
                    "message": f"I couldn't recognize '{destination}' as a valid destination. Please try a major city like 'New York', 'London', 'Paris', 'Sydney', etc."
                }
            else:
                return {
                    "success": False,
                    "error": str(amadeus_error),
                    "message": f"Sorry, I encountered an issue searching for hotels in {destination}. Please try again or use a different destination."
                }
        
        print(f"📊 Amadeus API response: success={search_results.get('success')}, results={len(search_results.get('results', []))}")
        
        if not search_results.get("success"):
            print(f"❌ Amadeus API error: {search_results.get('error')}")
            amadeus_error = search_results.get("error", "Hotel search failed")
            
            # FIXED: Better error messaging for users
            if "No hotels found for city code" in amadeus_error:
                return {
                    "success": False,
                    "error": amadeus_error,
                    "message": f"I couldn't find hotels in {destination}. Please try:\n• A major city name (e.g., 'New York', 'London', 'Paris')\n• Different spelling or variation\n• A nearby major city"
                }
            elif "Could not find hotels for city" in amadeus_error:
                return {
                    "success": False,
                    "error": amadeus_error,
                    "message": f"I couldn't recognize '{destination}' as a valid destination. Please try a major city like 'New York', 'London', 'Paris', 'Sydney', etc."
                }
            else:
                return {
                    "success": False,
                    "error": amadeus_error,
                    "message": f"Sorry, I couldn't find hotels in {destination} for {check_in_date} to {check_out_date}. Please try different dates or destination."
                }
        
        # OPTIMIZATION 6: Ultra-fast profile filtering with 10s timeout
        search_params = {
            "city_code": city_code,
            "destination": destination,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "rooms": rooms
        }
        
        print(f"🔧 Ultra-fast profile filtering with 10s timeout...")
        
        try:
            # Set 10-second timeout for profile filtering
            filter_timeout_task = asyncio.create_task(asyncio.sleep(10))
            filter_task = asyncio.create_task(
                hotel_agent.profile_agent.filter_hotel_results(
                    user_id=hotel_agent.current_user_id,
                    hotel_results=search_results["results"],
                    search_params=search_params
                )
            )
            
            done, pending = await asyncio.wait(
                [filter_timeout_task, filter_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            if filter_task in done:
                filtering_result = filter_task.result()
                print(f"✅ Filtering complete: {filtering_result['filtered_count']} of {filtering_result['original_count']} results")
            else:
                print(f"⚠️ Profile filtering timeout after 10s, using original results")
                filtering_result = {
                    "filtered_results": search_results["results"][:3],  # Top 3 results
                    "original_count": len(search_results["results"]),
                    "filtered_count": min(3, len(search_results["results"])),
                    "filtering_applied": False,
                    "rationale": "Used top results due to timeout"
                }
                
        except Exception as filter_error:
            print(f"⚠️ Profile filtering error: {filter_error}, using original results")
            filtering_result = {
                "filtered_results": search_results["results"][:3],  # Top 3 results
                "original_count": len(search_results["results"]),
                "filtered_count": min(3, len(search_results["results"])),
                "filtering_applied": False,
                "rationale": f"Used top results due to filtering error: {str(filter_error)}"
            }
        
        # OPTIMIZATION 7: Ultra-fast database save (no await)
        hotel_search = HotelSearch(
            id=str(uuid.uuid4()),
            user_id=hotel_agent.current_user_id,
            city_code=city_code,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            adults=adults,
            rooms=rooms,
            search_results=search_results["results"],
            result_count=len(filtering_result["filtered_results"]),
            search_successful=True
        )
        
        print(f"💾 Saving search to database (async)...")
        # Save asynchronously without waiting
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
        
        # OPTIMIZATION 9: Create result payload
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
        
        # Keep cache small (max 10 entries)
        if len(hotel_agent._search_cache) > 10:
            oldest_key = list(hotel_agent._search_cache.keys())[0]
            del hotel_agent._search_cache[oldest_key]
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Hotel search completed in {elapsed_time:.2f}s!")
        
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
    ULTRA-OPTIMIZED Hotel agent to prevent timeouts
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
        
        # Initialize search cache for ultra-fast responses
        self._search_cache = {}
        
        # Set this instance as the global instance for tools
        _current_hotel_agent = self
        
        # Create the agent with ultra-optimized settings
        self.agent = Agent(
            name="Arny Hotel Assistant", 
            instructions=get_hotel_system_message(),
            model="gpt-4o-mini",  # Use faster model
            tools=[search_hotels_tool]
        )
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                             user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ULTRA-OPTIMIZED: Process hotel search requests with timeout prevention
        """
        
        try:
            print(f"🏨 ULTRA-OPTIMIZED: HotelAgent processing: '{message[:50]}...'")
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
            
            # Build minimal conversation context (last 5 messages only)
            context_messages = []
            for msg in conversation_history[-5:]:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            print(f"🔧 Processing with {len(context_messages)} previous messages")
            
            # ULTRA-OPTIMIZATION: Set 20-second timeout for entire agent processing
            try:
                agent_timeout_task = asyncio.create_task(asyncio.sleep(20))
                
                if not context_messages:
                    # First message in conversation
                    print("🚀 Starting new hotel conversation")
                    agent_task = asyncio.create_task(Runner.run(self.agent, message))
                else:
                    # Continue conversation with context
                    print("🔄 Continuing hotel conversation with context")
                    agent_task = asyncio.create_task(
                        Runner.run(self.agent, context_messages + [{"role": "user", "content": message}])
                    )
                
                done, pending = await asyncio.wait(
                    [agent_timeout_task, agent_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                if agent_task in done:
                    result = agent_task.result()
                    assistant_message = result.final_output
                else:
                    print(f"⚠️ Agent processing timeout after 20s")
                    assistant_message = "I'm still searching for hotels. This is taking longer than expected. Please try again with a more specific location."
                    
            except Exception as agent_error:
                print(f"❌ Agent processing error: {agent_error}")
                assistant_message = "I encountered an error while searching for hotels. Please try again or be more specific about your destination."
            
            # Get search results from global instance
            global_agent = _get_hotel_agent()
            search_results = global_agent.latest_search_results if global_agent else []
            search_id = global_agent.latest_search_id if global_agent else None
            filtering_info = global_agent.latest_filtering_info if global_agent else {}
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ HotelAgent completed in {elapsed_time:.2f}s")
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
    
    def _convert_to_city_code_ultra_fast(self, location: str) -> str:
        """ULTRA-OPTIMIZED: Ultra-fast city code conversion with better fallbacks"""
        
        # Clean the input
        location_clean = location.lower().strip()
        
        # Remove common words that might interfere
        location_clean = location_clean.replace("hotels in ", "").replace("hotel in ", "").replace("stay in ", "")
        
        # Direct lookup in optimized mapping
        if location_clean in CITY_CODE_MAPPING:
            return CITY_CODE_MAPPING[location_clean]
        
        # Try without spaces
        location_no_spaces = location_clean.replace(" ", "")
        if location_no_spaces in CITY_CODE_MAPPING:
            return CITY_CODE_MAPPING[location_no_spaces]
        
        # If already looks like city code (3 letters)
        if len(location) == 3 and location.isalpha():
            return location.upper()
        
        # Try partial matches for common city names
        for city_key, city_code in CITY_CODE_MAPPING.items():
            if city_key in location_clean or location_clean in city_key:
                return city_code
        
        # Default: return cleaned input as uppercase
        # This will let Amadeus API handle unknown cities
        cleaned_code = location.strip().replace(" ", "").upper()[:3]
        if len(cleaned_code) >= 2:
            return cleaned_code
        else:
            return location.upper()
