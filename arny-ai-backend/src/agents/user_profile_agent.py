"""
User Profile Agent for Arny AI

This agent filters and ranks flight/hotel search results based on the preferences
of all users in the same family or group. It uses LLM intelligence to understand
group dynamics and provide recommendations that work for everyone.

Key Features:
- Group preference analysis
- Budget consensus filtering
- Dietary and accessibility accommodation
- Travel style harmonization
- Intelligent ranking based on group satisfaction
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI

from ..utils.config import config
from ..database.operations import DatabaseOperations

# Set up logging
logger = logging.getLogger(__name__)

class UserProfileAgent:
    """
    User Profile Agent that filters search results based on group preferences
    
    This agent analyzes all user profiles in a group and uses AI to filter
    and rank flight/hotel options that best satisfy the group's needs.
    """
    
    def __init__(self):
        """Initialize the user profile agent with required services"""
        try:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.db = DatabaseOperations()
            
            logger.info("UserProfileAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize UserProfileAgent: {e}")
            raise Exception(f"UserProfileAgent initialization failed: {e}")
    
    async def filter_flight_results(self, user_id: str, flight_results: List[Dict[str, Any]], 
                                  search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter flight search results based on group preferences - ENHANCED WITH DEBUG LOGGING
        
        Args:
            user_id: ID of user who initiated the search
            flight_results: Raw flight results from Amadeus API
            search_params: Original search parameters
            
        Returns:
            Dict containing filtered results and filtering rationale
        """
        try:
            logger.info(f"Filtering flight results for user {user_id}")
            
            # DEBUG: Log all original results
            print(f"🔍 DEBUG: Original {len(flight_results)} flight results BEFORE filtering:")
            for i, flight_offer in enumerate(flight_results, 1):
                price = flight_offer.get("price", {})
                itineraries = flight_offer.get("itineraries", [])
                
                print(f"   Flight {i}: ID {flight_offer.get('id', 'Unknown')}")
                print(f"   - Price: {price.get('currency', '')} {price.get('total', 'N/A')}")
                
                if itineraries:
                    first_itinerary = itineraries[0]
                    duration = first_itinerary.get("duration", "N/A")
                    segments = first_itinerary.get("segments", [])
                    
                    print(f"   - Duration: {duration}")
                    print(f"   - Segments: {len(segments)}")
                    
                    if segments:
                        first_segment = segments[0]
                        carrier = first_segment.get("carrierCode", "N/A")
                        flight_number = first_segment.get("number", "N/A")
                        departure = first_segment.get("departure", {})
                        arrival = first_segment.get("arrival", {})
                        
                        print(f"   - Airline: {carrier}{flight_number}")
                        print(f"   - Route: {departure.get('iataCode', 'N/A')} → {arrival.get('iataCode', 'N/A')}")
                        
                        departure_time = departure.get("at", "").split("T")[-1][:5] if departure.get("at") else "N/A"
                        arrival_time = arrival.get("at", "").split("T")[-1][:5] if arrival.get("at") else "N/A"
                        print(f"   - Times: {departure_time} → {arrival_time}")
                
                print()

            # Get group member profiles
            group_profiles = await self._get_group_profiles(user_id)
            
            if not group_profiles:
                print(f"🔍 DEBUG: No group filtering applied - single user or no group found")
                # No group or single user - return original results
                return {
                    "filtered_results": flight_results,
                    "original_count": len(flight_results),
                    "filtered_count": len(flight_results),
                    "filtering_applied": False,
                    "rationale": "No group filtering applied - single user or no group found"
                }
            
            # Create group analysis prompt
            group_summary = self._create_group_summary(group_profiles)
            
            print(f"🔍 DEBUG: Group filtering applied for {len(group_profiles)} group members:")
            print(f"   - Group size: {len(group_profiles)}")
            print(f"   - Budget ranges: {group_summary.get('budget_ranges', [])}")
            print(f"   - Travel styles: {group_summary.get('travel_styles', [])}")
            print(f"   - Preferred airlines: {group_summary.get('preferred_airlines', [])}")
            print(f"   - Dietary restrictions: {group_summary.get('dietary_restrictions', [])}")
            print(f"   - Accessibility needs: {group_summary.get('accessibility_needs', [])}")

            # Filter flights using AI
            filtered_results = await self._ai_filter_flights(
                flight_results, group_summary, search_params
            )
            
            # DEBUG: Log filtering results
            recommended_flights = filtered_results.get("recommended_flights", flight_results)
            excluded_count = len(flight_results) - len(recommended_flights)
            
            print(f"🔍 DEBUG: Filtering results:")
            print(f"   - Original count: {len(flight_results)}")
            print(f"   - Filtered count: {len(recommended_flights)}")
            print(f"   - Excluded count: {excluded_count}")
            print(f"   - Filtering rationale: {filtered_results.get('filtering_rationale', 'N/A')}")
            
            if excluded_count > 0:
                print(f"🔍 DEBUG: Flights that were EXCLUDED by filtering:")
                excluded_flights = []
                recommended_ids = [f.get('id', '') for f in recommended_flights]
                
                for i, flight_offer in enumerate(flight_results, 1):
                    flight_id = flight_offer.get("id", "")
                    if flight_id not in recommended_ids:
                        price = flight_offer.get("price", {})
                        excluded_flights.append((i, flight_id, flight_offer))
                        print(f"   ❌ Flight {i}: ID {flight_id} - {price.get('currency', '')} {price.get('total', 'N/A')}")
                
                print(f"🔍 DEBUG: Recommended flights that PASSED filtering:")
                for i, flight_offer in enumerate(recommended_flights, 1):
                    flight_id = flight_offer.get("id", "")
                    price = flight_offer.get("price", {})
                    print(f"   ✅ Flight {i}: ID {flight_id} - {price.get('currency', '')} {price.get('total', 'N/A')}")

            return {
                "filtered_results": recommended_flights,
                "original_count": len(flight_results),
                "filtered_count": len(recommended_flights),
                "filtering_applied": True,
                "rationale": filtered_results.get("filtering_rationale", "AI filtering applied"),
                "group_size": len(group_profiles),
                "group_preferences": filtered_results.get("group_considerations", {}),
                "excluded_count": excluded_count
            }
            
        except Exception as e:
            logger.error(f"Error filtering flight results for user {user_id}: {e}")
            # Return original results if filtering fails
            return {
                "filtered_results": flight_results,
                "original_count": len(flight_results),
                "filtered_count": len(flight_results),
                "filtering_applied": False,
                "error": str(e),
                "rationale": "Filtering failed - returning original results"
            }
    
    async def filter_hotel_results(self, user_id: str, hotel_results: List[Dict[str, Any]], 
                                 search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter hotel search results based on group preferences - ENHANCED WITH DEBUG LOGGING
        
        Args:
            user_id: ID of user who initiated the search
            hotel_results: Raw hotel results from Amadeus API
            search_params: Original search parameters
            
        Returns:
            Dict containing filtered results and filtering rationale
        """
        try:
            logger.info(f"Filtering hotel results for user {user_id}")
            
            # DEBUG: Log all original results
            print(f"🔍 DEBUG: Original {len(hotel_results)} hotel results BEFORE filtering:")
            for i, hotel_data in enumerate(hotel_results, 1):
                hotel = hotel_data.get("hotel", {})
                offers = hotel_data.get("offers", [])
                best_offer = offers[0] if offers else {}
                price = best_offer.get("price", {})
                
                print(f"   Hotel {i}: {hotel.get('name', 'Unknown')}")
                print(f"   - Chain: {hotel.get('chainCode', 'N/A')}")
                print(f"   - Rating: {hotel.get('rating', 'N/A')}")
                print(f"   - Price: {price.get('currency', '')} {price.get('total', 'N/A')}")
                print(f"   - Room: {best_offer.get('room', {}).get('description', {}).get('text', 'N/A')}")
                print()

            # Get group member profiles
            group_profiles = await self._get_group_profiles(user_id)
            
            if not group_profiles:
                print(f"🔍 DEBUG: No group filtering applied - single user or no group found")
                # No group or single user - return original results
                return {
                    "filtered_results": hotel_results,
                    "original_count": len(hotel_results),
                    "filtered_count": len(hotel_results),
                    "filtering_applied": False,
                    "rationale": "No group filtering applied - single user or no group found"
                }
            
            # Create group analysis prompt
            group_summary = self._create_group_summary(group_profiles)
            
            print(f"🔍 DEBUG: Group filtering applied for {len(group_profiles)} group members:")
            print(f"   - Group size: {len(group_profiles)}")
            print(f"   - Budget ranges: {group_summary.get('budget_ranges', [])}")
            print(f"   - Travel styles: {group_summary.get('travel_styles', [])}")
            print(f"   - Preferred hotels: {group_summary.get('preferred_hotels', [])}")
            print(f"   - Dietary restrictions: {group_summary.get('dietary_restrictions', [])}")
            print(f"   - Accessibility needs: {group_summary.get('accessibility_needs', [])}")

            # Filter hotels using AI
            filtered_results = await self._ai_filter_hotels(
                hotel_results, group_summary, search_params
            )
            
            # DEBUG: Log filtering results
            recommended_hotels = filtered_results.get("recommended_hotels", hotel_results)
            excluded_count = len(hotel_results) - len(recommended_hotels)
            
            print(f"🔍 DEBUG: Filtering results:")
            print(f"   - Original count: {len(hotel_results)}")
            print(f"   - Filtered count: {len(recommended_hotels)}")
            print(f"   - Excluded count: {excluded_count}")
            print(f"   - Filtering rationale: {filtered_results.get('filtering_rationale', 'N/A')}")
            
            if excluded_count > 0:
                print(f"🔍 DEBUG: Hotels that were EXCLUDED by filtering:")
                excluded_hotels = []
                recommended_names = [h.get('hotel', {}).get('name', '') for h in recommended_hotels]
                
                for i, hotel_data in enumerate(hotel_results, 1):
                    hotel_name = hotel_data.get("hotel", {}).get("name", "")
                    if hotel_name not in recommended_names:
                        excluded_hotels.append((i, hotel_name, hotel_data))
                        print(f"   ❌ Hotel {i}: {hotel_name}")
                
                print(f"🔍 DEBUG: Recommended hotels that PASSED filtering:")
                for i, hotel_data in enumerate(recommended_hotels, 1):
                    hotel_name = hotel_data.get("hotel", {}).get("name", "")
                    print(f"   ✅ Hotel {i}: {hotel_name}")

            return {
                "filtered_results": recommended_hotels,
                "original_count": len(hotel_results),
                "filtered_count": len(recommended_hotels),
                "filtering_applied": True,
                "rationale": filtered_results.get("filtering_rationale", "AI filtering applied"),
                "group_size": len(group_profiles),
                "group_preferences": filtered_results.get("group_considerations", {}),
                "excluded_count": excluded_count
            }
            
        except Exception as e:
            logger.error(f"Error filtering hotel results for user {user_id}: {e}")
            # Return original results if filtering fails
            return {
                "filtered_results": hotel_results,
                "original_count": len(hotel_results),
                "filtered_count": len(hotel_results),
                "filtering_applied": False,
                "error": str(e),
                "rationale": "Filtering failed - returning original results"
            }
    
    async def _get_group_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all user profiles in the same group as the requesting user
        
        Args:
            user_id: User ID to find group for
            
        Returns:
            List of user profile dictionaries
        """
        try:
            # Get user's groups
            user_groups = await self.db.get_user_groups(user_id)
            
            if not user_groups:
                return []
            
            # For now, use the first group (users typically belong to one family group)
            primary_group = user_groups[0]
            
            # Get all group members
            group_members = await self.db.get_group_members(primary_group)
            
            # Get profiles for all group members
            group_profiles = []
            for member in group_members:
                profile = await self.db.get_user_profile(member.user_id)
                if profile:
                    # Convert to dict and add user preferences
                    profile_dict = profile.dict()
                    
                    # Get additional preferences
                    preferences = await self.db.get_user_preferences(member.user_id)
                    if preferences:
                        profile_dict["preferences"] = preferences.dict()
                    
                    # Add group role
                    profile_dict["group_role"] = member.role
                    group_profiles.append(profile_dict)
            
            logger.info(f"Found {len(group_profiles)} profiles in group {primary_group}")
            return group_profiles
            
        except Exception as e:
            logger.error(f"Error getting group profiles for user {user_id}: {e}")
            return []
    
    def _create_group_summary(self, group_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of group preferences for AI filtering
        
        Args:
            group_profiles: List of user profile dictionaries
            
        Returns:
            Group summary dictionary
        """
        group_summary = {
            "group_size": len(group_profiles),
            "budget_ranges": [],
            "travel_styles": [],
            "dietary_restrictions": [],
            "accessibility_needs": [],
            "preferred_airlines": [],
            "preferred_hotels": [],
            "holiday_preferences": [],
            "age_ranges": [],
            "cities": []
        }
        
        for profile in group_profiles:
            # Budget information
            if profile.get("annual_income"):
                group_summary["budget_ranges"].append(profile["annual_income"])
            
            # Travel style
            if profile.get("travel_style"):
                group_summary["travel_styles"].append(profile["travel_style"])
            
            # Holiday preferences
            if profile.get("holiday_preferences"):
                group_summary["holiday_preferences"].extend(profile["holiday_preferences"])
            
            # Demographics
            if profile.get("city"):
                group_summary["cities"].append(profile["city"])
            
            # Preferences from user_preferences table
            prefs = profile.get("preferences", {})
            if prefs.get("dietary_restrictions"):
                group_summary["dietary_restrictions"].extend(prefs["dietary_restrictions"])
            
            if prefs.get("accessibility_needs"):
                group_summary["accessibility_needs"].extend(prefs["accessibility_needs"])
            
            if prefs.get("preferred_airlines"):
                group_summary["preferred_airlines"].extend(prefs["preferred_airlines"])
            
            if prefs.get("preferred_hotels"):
                group_summary["preferred_hotels"].extend(prefs["preferred_hotels"])
        
        # Remove duplicates and clean up
        for key in ["dietary_restrictions", "accessibility_needs", "preferred_airlines", 
                   "preferred_hotels", "holiday_preferences", "travel_styles", "cities"]:
            if isinstance(group_summary[key], list):
                group_summary[key] = list(set(group_summary[key]))
        
        return group_summary
    
    async def _ai_filter_flights(self, flight_results: List[Dict[str, Any]], 
                               group_summary: Dict[str, Any], search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to filter flight results based on group preferences
        """
        try:
            # Create filtering prompt
            prompt = f"""You are an expert travel agent analyzing flight options for a group of {group_summary['group_size']} travelers.

Group Profile:
- Group size: {group_summary['group_size']} people
- Travel styles: {', '.join(group_summary['travel_styles']) if group_summary['travel_styles'] else 'Not specified'}
- Budget ranges: {', '.join(group_summary['budget_ranges']) if group_summary['budget_ranges'] else 'Not specified'}
- Preferred airlines: {', '.join(group_summary['preferred_airlines']) if group_summary['preferred_airlines'] else 'No preferences'}
- Holiday preferences: {', '.join(group_summary['holiday_preferences'][:5]) if group_summary['holiday_preferences'] else 'General travel'}
- Cities: {', '.join(group_summary['cities']) if group_summary['cities'] else 'Various'}

Search Parameters:
- Origin: {search_params.get('origin', 'N/A')}
- Destination: {search_params.get('destination', 'N/A')}
- Departure: {search_params.get('departure_date', 'N/A')}
- Passengers: {search_params.get('adults', 1)}
- Cabin Class: {search_params.get('cabin_class', 'ECONOMY')}

Flight Options: {json.dumps(flight_results[:10], indent=2)}

Please analyze these flights and recommend the best options for this group. Consider:
1. Budget compatibility across the group
2. Travel style preferences (budget/comfort/luxury)
3. Airline preferences if specified
4. Flight timing and duration
5. Group coordination needs

Return a JSON response with:
{{
    "recommended_flights": [list of recommended flight objects],
    "excluded_count": number_of_excluded_flights,
    "filtering_rationale": "explanation of why these flights work best for the group",
    "group_considerations": {{
        "budget_consensus": "analysis of budget fit",
        "style_match": "how well flights match travel styles",
        "special_needs": "any special accommodations considered"
    }}
}}

Focus on flights that work for the majority of the group while considering budget constraints and preferences."""

            # Get AI response
            response = self.openai_client.responses.create(
                model="o4-mini",
                input=prompt
            )
            
            # Extract and parse response
            response_text = ""
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                response_text = content_item.text.strip()
                                break
            
            try:
                # Try to parse as JSON
                filtered_result = json.loads(response_text)
                return filtered_result
            except json.JSONDecodeError:
                # If JSON parsing fails, return original results with explanation
                logger.warning("AI response was not valid JSON, returning original results")
                return {
                    "recommended_flights": flight_results,
                    "excluded_count": 0,
                    "filtering_rationale": "AI filtering encountered an error, showing all options",
                    "group_considerations": {"error": "JSON parsing failed"}
                }
                
        except Exception as e:
            logger.error(f"Error in AI flight filtering: {e}")
            return {
                "recommended_flights": flight_results,
                "excluded_count": 0,
                "filtering_rationale": f"AI filtering failed: {str(e)}",
                "group_considerations": {"error": str(e)}
            }
    
    async def _ai_filter_hotels(self, hotel_results: List[Dict[str, Any]], 
                              group_summary: Dict[str, Any], search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to filter hotel results based on group preferences
        """
        try:
            # Create filtering prompt
            prompt = f"""You are an expert travel agent analyzing hotel options for a group of {group_summary['group_size']} travelers.

Group Profile:
- Group size: {group_summary['group_size']} people
- Travel styles: {', '.join(group_summary['travel_styles']) if group_summary['travel_styles'] else 'Not specified'}
- Budget ranges: {', '.join(group_summary['budget_ranges']) if group_summary['budget_ranges'] else 'Not specified'}
- Preferred hotel chains: {', '.join(group_summary['preferred_hotels']) if group_summary['preferred_hotels'] else 'No preferences'}
- Dietary restrictions: {', '.join(group_summary['dietary_restrictions']) if group_summary['dietary_restrictions'] else 'None specified'}
- Accessibility needs: {', '.join(group_summary['accessibility_needs']) if group_summary['accessibility_needs'] else 'None specified'}
- Holiday preferences: {', '.join(group_summary['holiday_preferences'][:5]) if group_summary['holiday_preferences'] else 'General travel'}

Search Parameters:
- Destination: {search_params.get('city_code', 'N/A')}
- Check-in: {search_params.get('check_in_date', 'N/A')}
- Check-out: {search_params.get('check_out_date', 'N/A')}
- Adults: {search_params.get('adults', 1)}
- Rooms: {search_params.get('rooms', 1)}

Hotel Options: {json.dumps(hotel_results[:8], indent=2)}

Please analyze these hotels and recommend the best options for this group. Consider:
1. Budget compatibility across the group
2. Room capacity and configuration for group size
3. Amenities that match group preferences and needs
4. Location convenience for group activities
5. Dietary and accessibility accommodations
6. Hotel style matching travel preferences

Return a JSON response with:
{{
    "recommended_hotels": [list of recommended hotel objects],
    "excluded_count": number_of_excluded_hotels,
    "filtering_rationale": "explanation of why these hotels work best for the group",
    "group_considerations": {{
        "budget_consensus": "analysis of budget fit",
        "amenity_match": "how well amenities match group needs",
        "accessibility": "accessibility considerations",
        "location_benefits": "location advantages for the group"
    }}
}}

Prioritize hotels that can accommodate the group's size, budget, and special requirements."""

            # Get AI response
            response = self.openai_client.responses.create(
                model="o4-mini",
                input=prompt
            )
            
            # Extract and parse response
            response_text = ""
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                response_text = content_item.text.strip()
                                break
            
            try:
                # Try to parse as JSON
                filtered_result = json.loads(response_text)
                return filtered_result
            except json.JSONDecodeError:
                # If JSON parsing fails, return original results with explanation
                logger.warning("AI response was not valid JSON, returning original results")
                return {
                    "recommended_hotels": hotel_results,
                    "excluded_count": 0,
                    "filtering_rationale": "AI filtering encountered an error, showing all options",
                    "group_considerations": {"error": "JSON parsing failed"}
                }
                
        except Exception as e:
            logger.error(f"Error in AI hotel filtering: {e}")
            return {
                "recommended_hotels": hotel_results,
                "excluded_count": 0,
                "filtering_rationale": f"AI filtering failed: {str(e)}",
                "group_considerations": {"error": str(e)}
            }

# ==================== MODULE EXPORTS ====================

__all__ = [
    'UserProfileAgent'
]
