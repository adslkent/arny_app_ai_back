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
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from openai import OpenAI
from agents import Agent, function_tool, Runner

from ..utils.config import config
from ..services.amadeus_service import AmadeusService
from ..database.operations import DatabaseOperations
from ..database.models import FlightSearch
from .user_profile_agent import UserProfileAgent

# Configure logger
logger = logging.getLogger(__name__)

# Global variable to store the current agent instance
_current_flight_agent = None

def _get_flight_agent():
    """Get the current flight agent instance"""
    global _current_flight_agent
    return _current_flight_agent

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

# ===== Enhanced Airport Code Mapping =====
AIRPORT_CODE_MAPPING = {
    # Major cities to airport codes
    "london": "LON",
    "paris": "PAR",
    "newyork": "NYC",
    "tokyo": "TYO", 
    "beijing": "PEK",
    "shanghai": "SHA",
    "hongkong": "HKG",
    "singapore": "SIN",
    "bangkok": "BKK",
    "sydney": "SYD",
    "dubai": "DXB",
    "losangeles": "LAX",
    "sanfrancisco": "SFO",
    
    # Direct mappings
    "pek": "PEK",
    "sha": "SHA",
    "can": "CAN",
    "tyo": "TYO",
    "hkg": "HKG",
    "sin": "SIN",
    "bkk": "BKK",
    "lon": "LON",
    "par": "PAR",
    "nyc": "NYC",
    "lax": "LAX",
    "sfo": "SFO",
    
    # Extended cities
    "berlin": "BER",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
    "rome": "ROM",
    "madrid": "MAD",
    "barcelona": "BCN",
    "moscow": "MOW",
    "chicago": "CHI",
    "washington": "WAS",
    "boston": "BOS",
    "toronto": "YTO",
    "vancouver": "YVR",
    "montreal": "YMQ",
    "milan": "MIL",
    "vienna": "VIE",
    "zurich": "ZRH",
    "geneva": "GVA",
    "brussels": "BRU",
    "seoul": "SEL",
    "taipei": "TPE",
    "osaka": "OSA",
    "melbourne": "MEL",
    "auckland": "AKL",
    "saopaulo": "SAO",
    "riodejaneiro": "RIO",
    "buenosaires": "BUE",
    "johannesburg": "JNB",
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
    "newdelhi": "DEL",
    "mumbai": "BOM",
    "manila": "MNL",
    "kualalumpur": "KUL",
    "jakarta": "CGK",
    
    # Common variations
    "new york": "NYC",
    "los angeles": "LAX",
    "san francisco": "SFO",
    "washington dc": "WAS",
    "sao paulo": "SAO",
    "rio de janeiro": "RIO",
    "buenos aires": "BUE",
    "hong kong": "HKG",
    "new delhi": "DEL",
    "kuala lumpur": "KUL",
}

# ==================== STANDALONE TOOL FUNCTIONS ====================

@function_tool
def search_flights_tool(origin: str, destination: str, departure_date: str, 
                       return_date: Optional[str] = None, adults: int = 1, 
                       cabin_class: str = "ECONOMY") -> dict:
    """
    Search for flights using Amadeus API with group profile filtering and cache management
    
    Args:
        origin: Origin airport/city code (e.g., 'SYD', 'Sydney')
        destination: Destination airport/city code (e.g., 'LAX', 'Los Angeles')
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format (optional for one-way)
        adults: Number of adult passengers (default 1)
        cabin_class: Cabin class - ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
    
    Returns:
        Dict with flight search results and profile filtering information
    """
    
    try:
        agent = _get_flight_agent()
        if not agent:
            return {"success": False, "error": "Flight agent not available"}
        
        # ADDED: Cache management similar to hotel agent
        search_key = f"{origin}_{destination}_{departure_date}_{return_date}_{adults}_{cabin_class}"
        if hasattr(agent, '_search_cache') and search_key in agent._search_cache:
            print(f"⚡ CACHE HIT: Returning cached flight results for {search_key}")
            return agent._search_cache[search_key]
        
        # FIXED: Check if context is properly set
        if not hasattr(agent, 'current_user_id') or not agent.current_user_id:
            return {"success": False, "error": "User context not available"}
        
        print(f"🔍 Search flights tool called with: {origin} → {destination} on {departure_date}")
        
        # Convert city names to airport codes if needed
        origin_code = agent._convert_to_airport_code(origin)
        destination_code = agent._convert_to_airport_code(destination)
        
        print(f"✅ Converted codes: {origin_code} → {destination_code}")
        
        # Perform flight search using the safe async runner - INCREASED FROM 12 TO 50
        print(f"🛫 Calling Amadeus API...")
        search_results = _run_async_safely(
            agent.amadeus_service.search_flights(
                origin=origin_code,
                destination=destination_code,
                departure_date=departure_date,
                return_date=return_date,
                adults=adults,
                cabin_class=cabin_class,
                max_results=50  # CHANGED: Increased from 12 to 50 results
            )
        )
        
        print(f"📊 Amadeus API response: success={search_results.get('success')}, results={len(search_results.get('results', []))}")
        
        if not search_results.get("success"):
            print(f"❌ Amadeus API error: {search_results.get('error')}")
            return {
                "success": False,
                "error": search_results.get("error", "Flight search failed"),
                "message": f"Sorry, I couldn't find flights from {origin} to {destination} on {departure_date}. Please check your dates and destinations."
            }
        
        # Apply profile-based filtering
        search_params = {
            "origin": origin_code,
            "destination": destination_code,
            "departure_date": departure_date,
            "return_date": return_date,
            "adults": adults,
            "cabin_class": cabin_class
        }
        
        print(f"🔧 Applying profile filtering...")
        # Filter results based on group profiles - SENDING ALL 50 FLIGHTS
        filtering_result = _run_async_safely(
            agent.profile_agent.filter_flight_results(
                user_id=agent.current_user_id,
                flight_results=search_results["results"],  # CHANGED: Send all flights (up to 50)
                search_params=search_params
            )
        )
        
        # FIXED: Handle different filtering result formats
        if isinstance(filtering_result, dict):
            # Check if it has the expected structure
            if "filtered_results" in filtering_result:
                filtered_flights = filtering_result["filtered_results"]
                original_count = filtering_result.get("original_count", len(search_results["results"]))
                filtered_count = len(filtered_flights)
                filtering_applied = filtering_result.get("filtering_applied", True)
                rationale = filtering_result.get("rationale", "Applied AI filtering for best results")
            elif "results" in filtering_result:
                # Alternative structure
                filtered_flights = filtering_result["results"]
                original_count = filtering_result.get("total_found", len(search_results["results"]))
                filtered_count = len(filtered_flights)
                filtering_applied = True
                rationale = filtering_result.get("message", "Applied AI filtering for best results")
            else:
                # Fallback: assume the result is just the filtered flights list
                filtered_flights = filtering_result if isinstance(filtering_result, list) else search_results["results"]
                original_count = len(search_results["results"])
                filtered_count = len(filtered_flights)
                filtering_applied = True
                rationale = "Applied AI filtering for best results"
        else:
            # Fallback: use original results if filtering failed
            filtered_flights = search_results["results"]
            original_count = len(search_results["results"])
            filtered_count = len(filtered_flights)
            filtering_applied = False
            rationale = "No filtering applied"
        
        print(f"✅ Filtering complete: {filtered_count} of {original_count} results")
        
        # Save search to database (save original results for analytics)
        flight_search = FlightSearch(
            search_id=str(uuid.uuid4()),
            user_id=agent.current_user_id,
            origin=origin_code,
            destination=destination_code,
            departure_date=departure_date,
            return_date=return_date,
            passengers=adults,
            cabin_class=cabin_class,
            search_results=search_results["results"],  # Save original results
            result_count=filtered_count,
            search_successful=True
        )
        
        print(f"💾 Saving search to database...")
        _run_async_safely(agent.db.save_flight_search(flight_search))
        
        # FIXED: Store search results and ID on agent instance for access in response
        agent.latest_search_results = filtered_flights
        agent.latest_search_id = flight_search.search_id
        agent.latest_filtering_info = {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "filtering_applied": filtering_applied,
            "group_size": 1,  # Default for now
            "rationale": rationale
        }
        
        # Format results for presentation
        formatted_results = agent._format_flight_results_for_agent(
            filtered_flights, 
            origin_code, 
            destination_code, 
            departure_date,
            {
                "filtering_applied": filtering_applied,
                "original_count": original_count,
                "filtered_count": filtered_count,
                "group_size": 1,
                "rationale": rationale
            }
        )
        
        result_payload = {
            "success": True,
            "results": filtered_flights,
            "formatted_results": formatted_results,
            "search_id": flight_search.search_id,
            "search_params": search_params,
            "filtering_info": {
                "original_count": original_count,
                "filtered_count": filtered_count,
                "filtering_applied": filtering_applied,
                "group_size": 1,
                "rationale": rationale
            }
        }
        
        # ADDED: Cache the result (matching hotel agent pattern)
        if not hasattr(agent, '_search_cache'):
            agent._search_cache = {}
        agent._search_cache[search_key] = result_payload
        
        # ADDED: Keep cache manageable (same as hotel agent - max 10 entries)
        if len(agent._search_cache) > 10:
            oldest_key = list(agent._search_cache.keys())[0]
            del agent._search_cache[oldest_key]
        
        print(f"✅ Flight search completed successfully!")
        return result_payload
        
    except Exception as e:
        print(f"❌ Error in search_flights_tool: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for flights: {str(e)}"
        }

@function_tool
def search_airports_tool(keyword: str, subtype: str = "AIRPORT") -> dict:
    """
    Search for airports based on keyword.
    
    Args:
        keyword: Search keyword, such as city name or airport code
        subtype: Location type, defaults to AIRPORT
    """
    logger.info(f"Searching airports: {keyword}, type: {subtype}")
    
    try:
        agent = _get_flight_agent()
        if not agent:
            return {"success": False, "error": "Flight agent not available"}
        
        airports = _run_async_safely(agent.amadeus_service.search_airports(keyword, subtype))
        
        if not airports:
            return {
                "success": False,
                "message": f"No airports found matching '{keyword}'."
            }
        
        result = f"Found {len(airports)} airports matching '{keyword}':\n\n"
        
        for airport in airports:
            name = airport.get('name', 'Unknown')
            iata_code = airport.get('iataCode', 'Unknown')
            city = airport.get('address', {}).get('cityName', 'Unknown')
            country = airport.get('address', {}).get('countryName', 'Unknown')
            
            result += f"- {name} ({iata_code})\n"
            result += f"  City: {city}, Country: {country}\n\n"
        
        return {
            "success": True,
            "message": result,
            "airports": airports
        }
    except Exception as e:
        logger.error(f"Error searching airports: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error searching for airports: {str(e)}"
        }

@function_tool
def get_checkin_links_tool(airline_code: str) -> dict:
    """
    Get online check-in links for an airline.
    
    Args:
        airline_code: Airline code (e.g., 'CA')
    """
    logger.info(f"Getting check-in links for airline: {airline_code}")
    
    try:
        agent = _get_flight_agent()
        if not agent:
            return {"success": False, "error": "Flight agent not available"}
        
        links = _run_async_safely(agent.amadeus_service.get_flight_checkin_links(airline_code))
        
        if not links:
            return {
                "success": False,
                "message": f"No check-in links found for airline {airline_code}."
            }
        
        result = f"Found {len(links)} check-in links for {airline_code} airline:\n\n"
        
        for link in links:
            link_type = link.get('type', 'Unknown')
            airline = link.get('airline', {}).get('name', airline_code)
            url = link.get('href', 'Unknown')
            
            result += f"- {airline}\n"
            result += f"  Type: {link_type}\n"
            result += f"  Link: {url}\n\n"
        
        return {
            "success": True,
            "message": result,
            "links": links
        }
        
    except Exception as e:
        logger.error(f"Error getting check-in links: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error getting check-in links: {str(e)}"
        }

@function_tool
def get_flight_pricing_tool(flight_selection: str) -> dict:
    """
    Get accurate pricing for a selected flight
    
    Args:
        flight_selection: Description of which flight the user selected (e.g., "flight 1", "the morning flight")
    
    Returns:
        Dict with detailed pricing information
    """
    
    try:
        agent = _get_flight_agent()
        if not agent:
            return {"success": False, "error": "Flight agent not available"}
        
        # Extract flight number from selection
        flight_number = agent._extract_flight_number(flight_selection)
        
        if not flight_number:
            return {
                "success": False,
                "message": "I couldn't determine which flight you selected. Please specify like 'flight 1' or 'the first option'."
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
            
            # Add recent conversation history (last 10 messages)
            for msg in conversation_history[-10:]:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            print(f"🔧 Processing with {len(context_messages)} previous messages")
            
            # Process with agent
            if not context_messages:
                # First message in conversation
                print("🚀 Starting new flight conversation")
                result = await Runner.run(self.agent, message)
            else:
                # Continue conversation with context
                print("🔄 Continuing flight conversation with context")
                result = await Runner.run(self.agent, context_messages + [{"role": "user", "content": message}])
            
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
4. Using get_checkin_links_tool when users ask about airline check-in
5. Using get_flight_pricing_tool when users want detailed pricing for specific flights

Current date: {today}
Tomorrow: {tomorrow}

Airport code examples:
{airport_mappings}

Key rules:
1. ALWAYS use search_flights_tool for ANY flight search request
2. If no return date is provided, search for one-way flights
3. Default to 1 adult passenger unless specified otherwise
4. Default to ECONOMY class unless specified otherwise
5. Convert city names to airport codes automatically (e.g., Sydney → SYD, London → LHR)
6. Be specific about dates - ask for clarification if dates are unclear
7. **IMPORTANT: Present ALL filtered flight results in your response - do not truncate the list**
8. Show flight details including airlines, times, prices, and duration for ALL results
9. If multiple passengers, collect this information before searching

Example interactions:
- "Flights from Sydney to LA next Friday" → extract departure city, destination, and date
- "Round trip to Tokyo in March" → ask for specific dates and departure city
- "Business class to London" → search with travel_class="BUSINESS"

Always be helpful, professional, and efficient in finding the best flight options. When presenting search results, ensure you show ALL available flight options that were filtered for the user - never truncate or skip flights from your response."""
    
    def _convert_to_airport_code(self, location: str) -> str:
        """Convert city names to airport codes using enhanced mapping"""
        
        location_lower = location.lower().strip()
        
        # Use enhanced airport code mapping
        if location_lower in AIRPORT_CODE_MAPPING:
            return AIRPORT_CODE_MAPPING[location_lower]
        
        # If already looks like airport code (3 letters, uppercase)
        if len(location) == 3 and location.isupper():
            return location
        
        # If 3 letters but lowercase, convert to uppercase
        if len(location) == 3 and location.isalpha():
            return location.upper()
        
        # Default: return as-is and let Amadeus handle it
        return location.upper()
    
    def _format_flight_results_for_agent(self, results: List[Dict[str, Any]], 
                                       origin: str, destination: str, departure_date: str,
                                       filtering_info: Dict[str, Any]) -> str:
        """Format flight results for the AI agent to present naturally with filtering information"""
        
        if not results:
            return f"No flights found from {origin} to {destination} on {departure_date}."
        
        # Start with filtering information if applied
        formatted = ""
        if filtering_info.get("filtering_applied"):
            group_size = filtering_info.get("group_size", 1)
            original_count = filtering_info.get("original_count", 0)
            filtered_count = len(results)
            
            if group_size > 1:
                formatted += f"🏠 **Group Travel Filtering Applied**\n"
                formatted += f"I analyzed {original_count} flights for your group of {group_size} travelers and "
                formatted += f"selected the {filtered_count} best options based on your group's preferences.\n\n"
                formatted += f"*{filtering_info.get('rationale', 'Filtered for group compatibility')}*\n\n"
        
        formatted += f"I found {len(results)} flights from {origin} to {destination} on {departure_date}:\n\n"
        
        # CHANGED: Show up to 10 results instead of 5
        for i, flight in enumerate(results[:10], 1):  # CHANGED: Show top 10 results instead of 5
            price = flight.get("price", {})
            total_price = price.get("total", "N/A")
            currency = price.get("currency", "")
            
            # Get first itinerary details
            itineraries = flight.get("itineraries", [])
            if itineraries:
                first_itinerary = itineraries[0]
                duration = first_itinerary.get("duration", "N/A")
                segments = first_itinerary.get("segments", [])
                
                if segments:
                    first_segment = segments[0]
                    departure = first_segment.get("departure", {})
                    arrival = first_segment.get("arrival", {})
                    carrier = first_segment.get("carrierCode", "")
                    flight_number = first_segment.get("number", "")
                    
                    departure_time = departure.get("at", "").split("T")[-1][:5] if departure.get("at") else "N/A"
                    arrival_time = arrival.get("at", "").split("T")[-1][:5] if arrival.get("at") else "N/A"
                    
                    stops = len(segments) - 1
                    stops_text = "Direct" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"
                    
                    formatted += f"**Option {i}:**\n"
                    formatted += f"• {carrier}{flight_number}: {departure_time} → {arrival_time} ({stops_text})\n"
                    formatted += f"• Duration: {duration}\n"
                    formatted += f"• Price: {currency} {total_price}\n\n"
        
        # CHANGED: Update the "more options" message for 10 instead of 5
        if len(results) > 10:
            formatted += f"... and {len(results) - 10} more options available.\n\n"
        
        # Add group filtering note if applicable
        if filtering_info.get("filtering_applied") and filtering_info.get("group_size", 1) > 1:
            formatted += "✈️ These flights have been selected to work well for your entire group. "
            if filtering_info.get("excluded_count", 0) > 0:
                formatted += f"I excluded {filtering_info['excluded_count']} options that didn't match your group's preferences. "
            formatted += "\n\n"
        
        formatted += "Would you like detailed pricing for any specific flight? Just tell me which one interests you!"
        
        return formatted
    
    def _extract_flight_number(self, selection_text: str) -> Optional[int]:
        """Extract flight number from user selection text"""
        
        # Look for patterns like "flight 1", "option 2", "first one", etc.
        import re
        
        # Direct number patterns
        patterns = [
            r'flight\s+(\d+)',
            r'option\s+(\d+)',
            r'number\s+(\d+)',
            r'^(\d+)$',  # Just a number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, selection_text.lower())
            if match:
                return int(match.group(1))
        
        # Word patterns
        word_numbers = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5
        }
        
        for word, number in word_numbers.items():
            if word in selection_text.lower():
                return number
        
        return None
