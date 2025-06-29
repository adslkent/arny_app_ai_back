"""
Hotel Search Agent Module - Complete hotel search agent implementation using OpenAI Agents SDK

This module provides a complete hotel search agent with integrated tools for:
1. Searching hotels by city (basic hotel information)  
2. Searching hotel offers with dates and pricing
3. Natural language processing for hotel queries with profile filtering

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

# ===== Hotel ID Mapping Dictionary =====
# This mapping is used to convert city names to corresponding hotel IDs (for future use)
HOTEL_ID_MAPPING = {
    "london": ["MCLONGHM", "HSLONROT"],
    "lon": ["MCLONGHM", "HSLONROT"],
    "newyork": ["NYNYCTSC", "NYCNYCHM"],
    "nyc": ["NYNYCTSC", "NYCNYCHM"],
    "new york": ["NYNYCTSC", "NYCNYCHM"],
    "paris": ["PARPARHT", "HSPARPDG"],
    "par": ["PARPARHT", "HSPARPDG"],
    "tokyo": ["TYOPACTH", "TYOTYOKH"],
    "tyo": ["TYOPACTH", "TYOTYOKH"],
    "beijing": ["PEKPEKFS", "PEKBEKRG"],
    "pek": ["PEKPEKFS", "PEKBEKRG"],
    "shanghai": ["SHAPUDHD", "SHAPEKRG"],
    "sha": ["SHAPUDHD", "SHAPEKRG"],
    "hongkong": ["HKGHKGSH", "HKGHKGHR"],
    "hkg": ["HKGHKGSH", "HKGHKGHR"],
    "hong kong": ["HKGHKGSH", "HKGHKGHR"],
    "singapore": ["SINSINRH", "SINSINFS"],
    "sin": ["SINSINRH", "SINSINFS"],
    "bangkok": ["BKKBKKHB", "BKKBKKSH"],
    "bkk": ["BKKBKKHB", "BKKBKKSH"],
    "sydney": ["SYDSYDHP", "SYDSYDFS"],
    "syd": ["SYDSYDHP", "SYDSYDFS"],
    "dubai": ["DXBDUBCC", "DXBDUBIC"],
    "dxb": ["DXBDUBCC", "DXBDUBIC"],
    "losangeles": ["LAXLAXTL", "LAXBVRMC"],
    "lax": ["LAXLAXTL", "LAXBVRMC"],
    "los angeles": ["LAXLAXTL", "LAXBVRMC"],
    "sanfrancisco": ["SFOSFOLW", "SFOSFOSC"],
    "sfo": ["SFOSFOLW", "SFOSFOSC"],
    "san francisco": ["SFOSFOLW", "SFOSFOSC"],
}

# ===== City Code Mapping Dictionary =====
# Enhanced mapping for converting city names to city codes
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

# ===== Board Type Mapping =====
BOARD_TYPE_MAPPING = {
    "room only": "ROOM_ONLY",
    "breakfast": "BREAKFAST",
    "half board": "HALF_BOARD", 
    "full board": "FULL_BOARD",
    "all inclusive": "ALL_INCLUSIVE",
}


def get_hotel_system_message(hotel_mapping: Dict[str, List[str]] = None, city_mapping: Dict[str, str] = None) -> str:
    """
    Generate hotel search assistant system prompt with current date and hotel/city mappings
    
    Parameters:
        hotel_mapping: Hotel ID mapping dictionary  
        city_mapping: City code mapping dictionary
        
    Returns:
        str: System prompt
    """
    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Use default mapping if none provided
    if not hotel_mapping:
        hotel_mapping = HOTEL_ID_MAPPING
    
    if not city_mapping:
        city_mapping = CITY_CODE_MAPPING
    
    # Format hotel mapping as string (showing only first 10 examples)
    hotel_mappings = "\n".join([f"- {city}: {ids}" for city, ids in list(hotel_mapping.items())[:10]])
    
    # Format city mapping as string (showing only first 10 examples)
    city_mappings = "\n".join([f"- {city}: {code}" for city, code in list(city_mapping.items())[:10]])
    
    # Complete prompt
    system_message = f"""You are Arny's professional hotel search intelligent assistant using the Amadeus hotel search system with intelligent group filtering.

Your main responsibilities are:
1. Understanding users' accommodation needs and extracting key information from natural language descriptions
2. Using the search tools to find hotels that meet user requirements with group profile filtering
3. Providing clear, detailed hotel options and information to users
4. Assisting users in completing the entire hotel query process with personalized recommendations

Today's date is {today}. Remember this when explaining relative dates (like "tomorrow", which is {tomorrow}, "next week", which is {next_week}, etc.).

You have three powerful search tools available:
1. search_hotels_tool - Search for hotels with specific dates, pricing, and availability with group filtering
2. search_hotels_by_city_tool - Search for hotels in a city without specific dates (general discovery)
3. get_hotel_offers_tool - Get detailed room offers and pricing for a selected hotel

When searching for hotels, you need to pay attention to:
- Accurately identify the city or location, converting city names to the correct codes
- Correctly understand date information for check-in and check-out
- Extract information about the number of adults and rooms needed
- Apply intelligent group filtering for families and groups
- Highlight group-friendly features when applicable

Examples of city name to hotel ID mapping (reference only):
{hotel_mappings}
...and more.

Examples of city name to city code mapping (for search_hotels_by_city_tool):
{city_mappings}
...and more.

Workflow for hotel offers search with dates:
1. Extract the following parameters from user query:
   - destination: City name or airport code, e.g., 'london' or 'LON'
   - check_in_date: Check-in date, format YYYY-MM-DD
   - check_out_date: Check-out date, format YYYY-MM-DD (if not provided, defaults to one day after check-in)
   - adults: Number of adults (default 1)
   - rooms: Number of rooms (default 1)

2. Use the search_hotels_tool to find hotel offers with group filtering applied

Workflow for general hotel discovery:
1. Extract the following parameters from user query:
   - city: City name or airport code, e.g., 'london' or 'LON'
   - radius: Search radius in km (default 5)
   - radius_unit: Unit of radius (default "KM")

2. Use the search_hotels_by_city_tool to find hotels in the city

Key guidelines:
- Always be helpful and conversational
- If travel details are missing, ask for clarification
- Present hotel options clearly with prices, ratings, and locations
- Help users understand amenities and room types
- If they select a hotel, use get_hotel_offers_tool for detailed information
- Explain when group filtering has been applied to help families/groups
- Highlight group-friendly features when applicable

Always highlight:
- Hotel star rating and guest reviews
- Location relative to city center or attractions
- Key amenities (WiFi, breakfast, pool, gym)
- Cancellation policies
- Group-friendly features when applicable
- When group filtering has improved their results

For example, when a user says: "I need a hotel in London for next week from Monday to Wednesday", you should:
1. Extract parameters: destination="London", check_in_date=next Monday's date, check_out_date=next Wednesday's date
2. Call search_hotels_tool with these parameters
3. Present results with group filtering information if applied
4. Ask if they want details on any specific hotel

Always respond in English. When presenting hotel information, use clear formatting to make it easy to read.

Please say "Hotel search completed" after completing the task.
"""
    return system_message

class HotelAgent:
    """
    Hotel agent using OpenAI Agents SDK with Amadeus API tools and profile filtering
    """
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.amadeus_service = AmadeusService()
        self.db = DatabaseOperations()
        self.profile_agent = UserProfileAgent()
        
        # Store context for tool calls (will be set during process_message)
        self.current_user_id = None
        self.current_session_id = None
        self.user_profile = None
        
        # Set this instance as the global instance for tools
        global _hotel_agent_instance
        _hotel_agent_instance = self
        
        # Create the agent with hotel search tools
        self.agent = Agent(
            name="Arny Hotel Assistant", 
            instructions=get_hotel_system_message(
                hotel_mapping=HOTEL_ID_MAPPING,
                city_mapping=CITY_CODE_MAPPING
            ),
            model="gpt-4o-mini",
            tools=[
                search_hotels_tool,
                search_hotels_by_city_tool,
                get_hotel_offers_tool
            ]
        )
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                             user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Process hotel search requests using OpenAI Agents SDK with profile filtering
        """
        
        try:
            # Store context for tool calls
            self.current_user_id = user_id
            self.current_session_id = session_id
            self.user_profile = user_profile
            
            # Build conversation context
            context_messages = []
            
            # Add recent conversation history (last 10 messages)
            for msg in conversation_history[-10:]:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content
                })
            
            # Process with agent
            if not context_messages:
                # First message in conversation
                result = await Runner.run(self.agent, message)
            else:
                # Continue conversation with context
                result = await Runner.run(self.agent, context_messages + [{"role": "user", "content": message}])
            
            # Extract response
            assistant_message = result.final_output
            
            return {
                "message": assistant_message,
                "agent_type": "hotel",
                "requires_action": False,  # Will be set to True if hotel selection is needed
                "metadata": {
                    "agent_type": "hotel",
                    "conversation_type": "hotel_search"
                }
            }
        
        except Exception as e:
            print(f"Error in hotel agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": "I'm sorry, I encountered an error while processing your request. Please try again or contact support if the issue persists.",
                "agent_type": "hotel",
                "error": str(e),
                "requires_action": False
            }
    
    # ==================== AGENT TOOLS ====================
    # Tool functions are now defined at module level below the class
    
    # ==================== HELPER METHODS ====================
    
    def _convert_to_city_code(self, location: str) -> str:
        """Convert city names to city codes for Amadeus hotel search"""
        
        location_lower = location.lower().strip().replace(" ", "")
        
        # Use the enhanced global city mapping
        if location_lower in CITY_CODE_MAPPING:
            return CITY_CODE_MAPPING[location_lower]
        
        # Try without space replacement for exact matches
        location_lower_with_spaces = location.lower().strip()
        if location_lower_with_spaces in CITY_CODE_MAPPING:
            return CITY_CODE_MAPPING[location_lower_with_spaces]
        
        # If already looks like city code (3 letters, uppercase)
        if len(location) == 3 and location.isupper():
            return location
        
        # If 3 letters but lowercase, convert to uppercase
        if len(location) == 3 and location.isalpha():
            return location.upper()
        
        # Default: return as-is and let Amadeus handle it
        logger.warning(f"Unknown city location: {location}, using as-is")
        return location.upper()
    
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
        
        for i, hotel_data in enumerate(results[:8], 1):  # Show top 8 results
            hotel = hotel_data.get("hotel", {})
            offers = hotel_data.get("offers", [])
            
            hotel_name = hotel.get("name", "Hotel")
            rating = hotel.get("rating", "")
            address = hotel.get("address", {})
            
            # Get location info
            location_info = ""
            if address.get("cityName"):
                location_info = address.get("cityName", "")
            
            # Get best offer (usually first one is cheapest)
            best_offer = None
            if offers:
                best_offer = offers[0]
            
            formatted += f"**Hotel {i}: {hotel_name}**\n"
            
            if rating:
                stars = "⭐" * int(float(rating)) if rating.replace('.', '').isdigit() else ""
                formatted += f"• Rating: {rating}/5 {stars}\n"
            
            if location_info:
                formatted += f"• Location: {location_info}\n"
            
            if best_offer:
                price = best_offer.get("price", {})
                total_price = price.get("total", "N/A")
                currency = price.get("currency", "")
                
                room = best_offer.get("room", {})
                room_type = room.get("type", "")
                
                if room_type:
                    formatted += f"• Room: {room_type}\n"
                
                formatted += f"• Price: {currency} {total_price} per night\n"
                
                # Add policies if available
                policies = best_offer.get("policies", {})
                if policies.get("cancellation"):
                    cancellation = policies["cancellation"]
                    if cancellation.get("type") == "FREE_CANCELLATION":
                        formatted += f"• ✅ Free cancellation\n"
                
                # Add amenities if available
                amenities = hotel.get("amenities", [])
                if amenities:
                    amenity_list = ", ".join(amenities[:3])  # Show first 3 amenities
                    formatted += f"• Amenities: {amenity_list}\n"
            
            formatted += "\n"
        
        if len(results) > 8:
            formatted += f"... and {len(results) - 8} more options available.\n\n"
        
        # Add group filtering note if applicable
        if filtering_info.get("filtering_applied") and filtering_info.get("group_size", 1) > 1:
            formatted += "🏨 These hotels have been selected to work well for your entire group. "
            if filtering_info.get("excluded_count", 0) > 0:
                formatted += f"I excluded {filtering_info['excluded_count']} options that didn't match your group's preferences. "
            formatted += "\n\n"
        
        formatted += "Would you like to see detailed room options for any specific hotel? Just tell me which one interests you!"
        
        return formatted
    
    def _extract_hotel_number(self, selection_text: str) -> Optional[int]:
        """Extract hotel number from user selection text"""
        
        # Look for patterns like "hotel 1", "option 2", "first one", etc.
        import re
        
        # Direct number patterns
        patterns = [
            r'hotel\s+(\d+)',
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
            'sixth': 6, 'seventh': 7, 'eighth': 8,
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8
        }
        
        for word, number in word_numbers.items():
            if word in selection_text.lower():
                return number
        
        return None


# Global instance for tool functions to access
_hotel_agent_instance = None

def _get_hotel_agent():
    """Get the global hotel agent instance"""
    global _hotel_agent_instance
    if _hotel_agent_instance is None:
        raise RuntimeError("Hotel agent instance not initialized. Create a HotelAgent instance first.")
    return _hotel_agent_instance


@function_tool
async def search_hotels_tool(destination: str, check_in_date: str, check_out_date: Optional[str] = None,
                            adults: int = 1, rooms: int = 1, price_range: Optional[str] = None,
                            currency: Optional[str] = None, board_type: Optional[str] = None) -> dict:
    """
    Search for hotels using Amadeus API with group profile filtering
    
    Args:
        destination: Destination city or city code (e.g., 'Sydney', 'SYD')
        check_in_date: Check-in date in YYYY-MM-DD format
        check_out_date: Check-out date in YYYY-MM-DD format, defaults to one day after check-in
        adults: Number of adults (default 1)
        rooms: Number of rooms (default 1)
        price_range: Price range, e.g., '100-200'
        currency: Currency code, e.g., 'USD' or 'EUR'
        board_type: Board type, options: ROOM_ONLY, BREAKFAST, HALF_BOARD, FULL_BOARD, ALL_INCLUSIVE
    
    Returns:
        Dict with hotel search results and profile filtering information
    """
    
    # ---------------------------------------------------------------------
    # DEBUG LOGGING
    # ---------------------------------------------------------------------
    print("[DEBUG] search_hotels_tool called with:", {
        "destination": destination,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "rooms": rooms,
        "price_range": price_range,
        "currency": currency,
        "board_type": board_type,
    })
    
    try:
        hotel_agent = _get_hotel_agent()
        
        # Convert city names to city codes if needed
        city_code = hotel_agent._convert_to_city_code(destination)
        
        # Handle board type mapping
        board_type_processed = None
        if board_type:
            board_type_processed = BOARD_TYPE_MAPPING.get(board_type.lower(), board_type)
            logger.info(f"Board type processed: {board_type} -> {board_type_processed}")
        
        # Handle default check-out date (one day after check-in)
        if not check_out_date:
            try:
                check_in_dt = datetime.strptime(check_in_date, "%Y-%m-%d")
                check_out_dt = check_in_dt + timedelta(days=1)
                check_out_date = check_out_dt.strftime("%Y-%m-%d")
                logger.info(f"Auto-generated check_out_date: {check_out_date}")
            except ValueError:
                return {
                    "success": False,
                    "error": "Invalid check_in_date format",
                    "message": "Please provide check-in date in YYYY-MM-DD format"
                }
        
        # Perform hotel search
        search_results = await hotel_agent.amadeus_service.search_hotels(
            city_code=city_code,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            adults=adults,
            rooms=rooms,
            max_results=15  # Get more results for better filtering
        )
        
        if not search_results.get("success"):
            return {
                "success": False,
                "error": search_results.get("error", "Hotel search failed"),
                "message": f"Sorry, I couldn't find hotels in {destination} for {check_in_date} to {check_out_date}. Please check your dates and destination."
            }
        
        # Apply profile-based filtering
        search_params = {
            "city_code": city_code,
            "destination": destination,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "rooms": rooms
        }
        
        # Filter results based on group profiles
        filtering_result = await hotel_agent.profile_agent.filter_hotel_results(
            user_id=hotel_agent.current_user_id,
            hotel_results=search_results["results"],
            search_params=search_params
        )
        
        # Save search to database (save original results for analytics)
        hotel_search = HotelSearch(
            search_id=str(uuid.uuid4()),
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
        
        await hotel_agent.db.save_hotel_search(hotel_search)
        
        # Format results for presentation
        formatted_results = hotel_agent._format_hotel_results_for_agent(
            filtering_result["filtered_results"], 
            destination,
            check_in_date, 
            check_out_date,
            filtering_result
        )
        
        result_payload = {
            "success": True,
            "results": filtering_result["filtered_results"],
            "formatted_results": formatted_results,
            "search_id": hotel_search.search_id,
            "search_params": search_params,
            "filtering_info": {
                "original_count": filtering_result["original_count"],
                "filtered_count": filtering_result["filtered_count"],
                "filtering_applied": filtering_result["filtering_applied"],
                "group_size": filtering_result.get("group_size", 1),
                "rationale": filtering_result["rationale"]
            }
        }
        
        # DEBUG print summary of results
        print("[DEBUG] search_hotels_tool result_summary:", {
            "original_count": filtering_result["original_count"],
            "filtered_count": filtering_result["filtered_count"]
        })

        return result_payload
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for hotels: {str(e)}"
        }


@function_tool  
async def search_hotels_by_city_tool(
    city: str,
    radius: int = 5,
    radius_unit: str = "KM",
    amenities: Optional[List[str]] = None,
    ratings: Optional[List[str]] = None
) -> dict:
    """
    Search for hotels by city code, returning basic hotel information without specific dates.
    Useful for general hotel discovery in a destination.
    
    Args:
        city: City name or airport code, e.g., 'london' or 'LON'
        radius: Search radius in kilometers (default: 5)
        radius_unit: Radius unit, options: KM, MILE (default: KM)
        amenities: List of amenities, e.g., ['SWIMMING_POOL', 'WIFI']
        ratings: Hotel star ratings, e.g., ['4', '5']
        
    Returns:
        Dict with basic hotel information
    """
    logger.info(f"Search hotels by city: city={city}, radius={radius}{radius_unit}")
    
    print("[DEBUG] search_hotels_by_city_tool called with:", {
        "city": city,
        "radius": radius,
        "radius_unit": radius_unit,
        "amenities": amenities,
        "ratings": ratings,
    })
    
    try:
        hotel_agent = _get_hotel_agent()
        
        # Convert city names to city codes if needed
        city_code = hotel_agent._convert_to_city_code(city)
        
        # For now, we'll use a simple fallback approach since we're using Amadeus service
        # In a full implementation, this would call a separate city-based search endpoint
        
        # Use tomorrow as default check-in for basic search
        tomorrow = datetime.now() + timedelta(days=1)
        day_after = tomorrow + timedelta(days=1)
        
        search_results = await hotel_agent.amadeus_service.search_hotels(
            city_code=city_code,
            check_in_date=tomorrow.strftime("%Y-%m-%d"),
            check_out_date=day_after.strftime("%Y-%m-%d"),
            adults=1,
            rooms=1,
            max_results=10
        )
        
        if not search_results.get("success"):
            return {
                "success": False,
                "error": search_results.get("error", "Hotel search failed"),
                "message": f"No hotels found in city '{city}' ({city_code}). Please try other cities or increase search radius."
            }
        
        hotels = search_results.get("results", [])
        
        if not hotels:
            return {
                "success": False,
                "message": f"No hotels found in city '{city}' ({city_code}). Please try other cities."
            }
        
        # Format response for city-based search (focus on hotel info, not pricing)
        response = f"Found {len(hotels)} hotels in {city} ({city_code}) within {radius}{radius_unit} radius:\n\n"
        
        for i, hotel_data in enumerate(hotels[:10], 1):  # Show up to 10 results
            hotel = hotel_data.get("hotel", {})
            hotel_name = hotel.get("name", "Unknown Hotel")
            rating = hotel.get("rating", "")
            address = hotel.get("address", {})
            
            response += f"Hotel {i}: {hotel_name}\n"
            
            if rating:
                stars = "⭐" * int(float(rating)) if rating.replace('.', '').isdigit() else ""
                response += f"• Rating: {rating}/5 {stars}\n"
            
            if address.get("cityName"):
                response += f"• Location: {address.get('cityName', '')}\n"
            
            # Add amenities if available
            amenities = hotel.get("amenities", [])
            if amenities:
                amenity_list = ", ".join(amenities[:3])  # Show first 3 amenities
                response += f"• Amenities: {amenity_list}\n"
            
            response += "\n"
        
        response += "To see current prices and availability, please provide your travel dates!"
        
        return {
            "success": True,
            "message": response,
            "hotels_found": len(hotels),
            "city_code": city_code
        }
        
    except Exception as e:
        logger.error(f"Error in search_hotels_by_city_tool: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"An error occurred while searching for hotels in {city}: {str(e)}"
        }


@function_tool
async def get_hotel_offers_tool(hotel_selection: str) -> dict:
    """
    Get detailed offers for a selected hotel
    
    Args:
        hotel_selection: Description of which hotel the user selected (e.g., "hotel 1", "the Hilton")
    
    Returns:
        Dict with detailed hotel offers and room types
    """
    
    print("[DEBUG] get_hotel_offers_tool called with:", hotel_selection)
    
    try:
        hotel_agent = _get_hotel_agent()
        
        # Extract hotel number from selection
        hotel_number = hotel_agent._extract_hotel_number(hotel_selection)
        
        if not hotel_number:
            return {
                "success": False,
                "message": "I couldn't determine which hotel you selected. Please specify like 'hotel 1' or 'the first option'."
            }
        
        # For now, return a detailed offers response
        # In production, this would call Amadeus Hotel Offers API
        result_payload = {
            "success": True,
            "message": f"Getting detailed room offers for hotel {hotel_number}...",
            "hotel_number": hotel_number,
            "offers_info": {
                "note": "This would contain detailed room types and offers from Amadeus Hotel Offers API",
                "includes": ["room_types", "amenities", "cancellation_policy", "breakfast_options", "pricing_breakdown"]
            }
        }
        
        print("[DEBUG] get_hotel_offers_tool returning payload for hotel_number", hotel_number)

        return result_payload
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error getting hotel offers: {str(e)}"
        }


# ===== Utility Functions =====

def get_supported_cities() -> List[str]:
    """
    Get list of supported cities for hotel search
    
    Returns:
        List of supported city names
    """
    return list(CITY_CODE_MAPPING.keys())


def get_supported_board_types() -> List[str]:
    """
    Get list of supported board types
    
    Returns:
        List of supported board type names
    """
    return list(BOARD_TYPE_MAPPING.keys())


def create_hotel_agent(model: str = "gpt-4o-mini") -> HotelAgent:
    """
    Create hotel search intelligent assistant agent
    
    Args:
        model: Model name to use (default: gpt-4o-mini)
        
    Returns:
        HotelAgent instance
    """
    logger.info(f"Creating hotel search intelligent assistant agent with model: {model}")
    # Note: Model parameter could be used to customize the agent's model in future versions
    return HotelAgent()


async def process_hotel_query(query: str, user_id: str = "default", session_id: str = "default") -> str:
    """
    Process hotel query request - Use AI agent to extract parameters, then call hotel search API
    
    Args:
        query: User's natural language query
        user_id: User identifier
        session_id: Session identifier
        
    Returns:
        str: Processing result string containing hotel search results
    """
    try:
        logger.info(f"Processing hotel query: {query}")
        
        # Create hotel search assistant
        agent = create_hotel_agent()
        
        # Process the query
        result = await agent.process_message(
            user_id=user_id,
            message=query,
            session_id=session_id,
            user_profile={},
            conversation_history=[]
        )
        
        logger.info("Successfully processed hotel query")
        return result.get("message", "No response generated")
        
    except Exception as e:
        logger.error(f"Error processing hotel query: {str(e)}", exc_info=True)
        return f"Error processing hotel query: {str(e)}"