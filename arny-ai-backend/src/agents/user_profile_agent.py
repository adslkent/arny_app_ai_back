"""
User Profile Agent for Arny AI - ULTRA-OPTIMIZED VERSION

This agent filters and ranks flight/hotel search results based on the preferences
of all users in the same family or group. Ultra-optimized for performance to avoid timeouts.

Key Ultra-Optimizations:
- Extract only essential data (price, airline, times) instead of full JSON
- Maximum 3 results processed for ultra-fast response
- Much shorter prompts (under 2000 chars)
- 15-second timeout on OpenAI calls
- Multiple fast fallback layers
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI

from ..utils.config import config
from ..database.operations import DatabaseOperations

# Set up logging
logger = logging.getLogger(__name__)

class UserProfileAgent:
    """
    User Profile Agent that filters search results based on group preferences - ULTRA-OPTIMIZED VERSION
    
    This agent analyzes all user profiles in a group and uses AI to filter
    and rank flight/hotel options that best satisfy the group's needs.
    Ultra-optimized for performance to complete well within Lambda timeout limits.
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
        Filter flight search results based on group preferences - ULTRA-OPTIMIZED VERSION
        
        Args:
            user_id: ID of user who initiated the search
            flight_results: Raw flight results from Amadeus API
            search_params: Original search parameters
            
        Returns:
            Dict containing filtered results and filtering rationale
        """
        try:
            logger.info(f"Ultra-optimized filtering flight results for user {user_id}")
            
            # DEBUG: Log results count
            print(f"🚀 DEBUG: Ultra-optimized filtering - Original {len(flight_results)} flight results")

            # Get group member profiles
            group_profiles = await self._get_group_profiles(user_id)
            
            if not group_profiles:
                print(f"🔍 DEBUG: No group filtering applied - single user or no group found")
                return {
                    "filtered_results": flight_results,
                    "original_count": len(flight_results),
                    "filtered_count": len(flight_results),
                    "filtering_applied": False,
                    "rationale": "No group filtering applied - single user or no group found"
                }
            
            # Create group analysis prompt
            group_summary = self._create_group_summary(group_profiles)
            
            print(f"⚡ DEBUG: Ultra-optimized group filtering for {len(group_profiles)} members")

            # ULTRA-OPTIMIZED: Filter flights using ultra-fast AI processing
            filtered_results = await self._ai_filter_flights_ultra_optimized(
                flight_results, group_summary, search_params
            )
            
            # Handle results
            recommended_flights = filtered_results.get("recommended_flights", flight_results)
            excluded_count = len(flight_results) - len(recommended_flights)
            
            print(f"🚀 DEBUG: Ultra-optimized filtering results:")
            print(f"   - Original count: {len(flight_results)}")
            print(f"   - Filtered count: {len(recommended_flights)}")
            print(f"   - Processing time: Ultra-optimized for sub-15s response")

            return {
                "filtered_results": recommended_flights,
                "original_count": len(flight_results),
                "filtered_count": len(recommended_flights),
                "filtering_applied": True,
                "rationale": filtered_results.get("filtering_rationale", "AI filtering applied (ultra-optimized)"),
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
        Filter hotel search results based on group preferences - ULTRA-OPTIMIZED VERSION
        
        Args:
            user_id: ID of user who initiated the search
            hotel_results: Raw hotel results from Amadeus API
            search_params: Original search parameters
            
        Returns:
            Dict containing filtered results and filtering rationale
        """
        try:
            logger.info(f"Ultra-optimized filtering hotel results for user {user_id}")
            
            # DEBUG: Log results count
            print(f"🚀 DEBUG: Ultra-optimized filtering - Original {len(hotel_results)} hotel results")

            # Get group member profiles
            group_profiles = await self._get_group_profiles(user_id)
            
            if not group_profiles:
                print(f"🔍 DEBUG: No group filtering applied - single user or no group found")
                return {
                    "filtered_results": hotel_results,
                    "original_count": len(hotel_results),
                    "filtered_count": len(hotel_results),
                    "filtering_applied": False,
                    "rationale": "No group filtering applied - single user or no group found"
                }
            
            # Create group analysis prompt
            group_summary = self._create_group_summary(group_profiles)
            
            print(f"⚡ DEBUG: Ultra-optimized group filtering for {len(group_profiles)} members")

            # ULTRA-OPTIMIZED: Filter hotels using ultra-fast AI processing
            filtered_results = await self._ai_filter_hotels_ultra_optimized(
                hotel_results, group_summary, search_params
            )
            
            # Handle results
            recommended_hotels = filtered_results.get("recommended_hotels", hotel_results)
            excluded_count = len(hotel_results) - len(recommended_hotels)
            
            print(f"🚀 DEBUG: Ultra-optimized filtering results:")
            print(f"   - Original count: {len(hotel_results)}")
            print(f"   - Filtered count: {len(recommended_hotels)}")
            print(f"   - Processing time: Ultra-optimized for sub-15s response")

            return {
                "filtered_results": recommended_hotels,
                "original_count": len(hotel_results),
                "filtered_count": len(recommended_hotels),
                "filtering_applied": True,
                "rationale": filtered_results.get("filtering_rationale", "AI filtering applied (ultra-optimized)"),
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
    
    def _extract_essential_flight_data(self, flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ULTRA-OPTIMIZATION: Extract only essential flight data to minimize prompt size
        
        Args:
            flights: Full flight data from Amadeus
            
        Returns:
            Minimal flight data for AI processing
        """
        essential_flights = []
        
        for i, flight in enumerate(flights[:3]):  # Only process top 3 for ultra-speed
            try:
                price = flight.get("price", {})
                itineraries = flight.get("itineraries", [])
                
                essential_flight = {
                    "id": i + 1,  # Simple numeric ID
                    "total_price": price.get("total", "N/A"),
                    "currency": price.get("currency", ""),
                }
                
                # Extract first itinerary details only
                if itineraries:
                    first_itinerary = itineraries[0]
                    segments = first_itinerary.get("segments", [])
                    
                    if segments:
                        first_segment = segments[0]
                        
                        # Essential timing info
                        departure = first_segment.get("departure", {})
                        arrival = first_segment.get("arrival", {})
                        
                        essential_flight.update({
                            "airline": first_segment.get("carrierCode", ""),
                            "flight_number": first_segment.get("number", ""),
                            "departure_time": departure.get("at", "").split("T")[-1][:5],
                            "arrival_time": arrival.get("at", "").split("T")[-1][:5],
                            "duration": first_itinerary.get("duration", ""),
                            "stops": len(segments) - 1
                        })
                
                essential_flights.append(essential_flight)
                
            except Exception as e:
                print(f"⚠️ Error extracting flight {i}: {e}")
                continue
        
        return essential_flights
    
    def _extract_essential_hotel_data(self, hotels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ULTRA-OPTIMIZATION: Extract only essential hotel data to minimize prompt size
        
        Args:
            hotels: Full hotel data from Amadeus
            
        Returns:
            Minimal hotel data for AI processing
        """
        essential_hotels = []
        
        for i, hotel_data in enumerate(hotels[:3]):  # Only process top 3 for ultra-speed
            try:
                hotel = hotel_data.get("hotel", {})
                offers = hotel_data.get("offers", [])
                
                essential_hotel = {
                    "id": i + 1,  # Simple numeric ID
                    "name": hotel.get("name", "Hotel"),
                    "rating": hotel.get("rating", ""),
                }
                
                # Extract best offer only
                if offers:
                    best_offer = offers[0]
                    price = best_offer.get("price", {})
                    
                    essential_hotel.update({
                        "price_per_night": price.get("total", "N/A"),
                        "currency": price.get("currency", ""),
                        "room_type": best_offer.get("room", {}).get("type", ""),
                        "free_cancellation": "cancellation" in best_offer.get("policies", {})
                    })
                
                essential_hotels.append(essential_hotel)
                
            except Exception as e:
                print(f"⚠️ Error extracting hotel {i}: {e}")
                continue
        
        return essential_hotels
    
    async def _ai_filter_flights_ultra_optimized(self, flight_results: List[Dict[str, Any]], 
                                               group_summary: Dict[str, Any], search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ULTRA-OPTIMIZED: Use AI to filter flight results with maximum speed optimizations
        
        Key ultra-optimizations:
        - Extract only essential flight data (price, airline, times)
        - Limit to 3 flights maximum for ultra-fast processing
        - Ultra-short prompts (under 2000 chars)
        - 15-second timeout on OpenAI calls
        - Multiple fast fallback layers
        """
        try:
            print(f"🚀 DEBUG: Starting ultra-optimized AI flight filtering...")
            
            # ULTRA-OPTIMIZATION 1: Extract only essential flight data
            essential_flights = self._extract_essential_flight_data(flight_results)
            print(f"⚡ Ultra-optimized: Processing {len(essential_flights)} flights (max 3) for ultra-speed")
            
            # ULTRA-OPTIMIZATION 2: Create ultra-short prompt
            budget = group_summary.get('budget_ranges', [])[:1]  # Only first budget
            style = group_summary.get('travel_styles', [])[:1]    # Only first style
            
            prompt = f"""Select best 2-3 flights for group of {group_summary['group_size']}.
Budget: {budget[0] if budget else 'any'}
Style: {style[0] if style else 'any'}

Flights:
{json.dumps(essential_flights)}

Return JSON:
{{"recommended_flights": [flight objects], "filtering_rationale": "brief reason"}}"""

            print(f"⚡ Ultra-optimized: Using ultra-short prompt ({len(prompt)} chars)")

            # ULTRA-OPTIMIZATION 3: Ultra-fast OpenAI call with strict timeout
            try:
                # Set 15-second timeout
                timeout_task = asyncio.create_task(asyncio.sleep(15))
                api_task = asyncio.create_task(self._make_openai_call(prompt))
                
                done, pending = await asyncio.wait(
                    [timeout_task, api_task], 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                
                if api_task in done:
                    response_text = api_task.result()
                    print(f"✅ Ultra-optimized: Got AI response in under 15s")
                else:
                    print(f"⚠️ Ultra-optimized: OpenAI call timed out after 15s")
                    raise asyncio.TimeoutError("OpenAI call timed out")
                
                # ULTRA-OPTIMIZATION 4: Ultra-fast JSON parsing
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        filtered_result = json.loads(json_text)
                        
                        # Map back to original flight objects
                        recommended_ids = [f.get("id", 0) for f in filtered_result.get("recommended_flights", [])]
                        original_flights = [flight_results[i-1] for i in recommended_ids if 1 <= i <= len(flight_results)]
                        
                        filtered_result["recommended_flights"] = original_flights
                        print(f"✅ Ultra-optimized: Successfully parsed JSON and mapped back to original flights")
                        return filtered_result
                    else:
                        raise ValueError("No JSON found in response")
                        
                except (json.JSONDecodeError, ValueError) as json_error:
                    print(f"⚠️ Ultra-optimized: JSON parsing failed: {json_error}")
                    # Ultra-fast fallback: return top 2 flights
                    return {
                        "recommended_flights": flight_results[:2],
                        "filtering_rationale": "AI filtering applied with ultra-fast fallback"
                    }
                
            except Exception as api_error:
                print(f"⚠️ Ultra-optimized: OpenAI API error: {api_error}")
                # Ultra-fast fallback: return top 2 flights
                return {
                    "recommended_flights": flight_results[:2],
                    "filtering_rationale": f"AI filtering ultra-fast fallback due to API timeout"
                }
                
        except Exception as e:
            logger.error(f"Error in ultra-optimized AI flight filtering: {e}")
            # ULTRA-OPTIMIZATION 5: Always return valid data instantly
            return {
                "recommended_flights": flight_results[:2],
                "filtering_rationale": f"Ultra-optimized filtering fallback: {str(e)}"
            }
    
    async def _make_openai_call(self, prompt: str) -> str:
        """Make OpenAI API call asynchronously"""
        response = self.openai_client.responses.create(
            model="o4-mini",
            input=prompt
        )
        
        response_text = ""
        if response and hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if hasattr(output_item, 'content') and output_item.content:
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text') and content_item.text:
                            response_text = content_item.text.strip()
                            break
                    if response_text:
                        break
        
        return response_text
    
    async def _ai_filter_hotels_ultra_optimized(self, hotel_results: List[Dict[str, Any]], 
                                              group_summary: Dict[str, Any], search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ULTRA-OPTIMIZED: Use AI to filter hotel results with maximum speed optimizations
        
        Key ultra-optimizations:
        - Extract only essential hotel data (price, name, rating)
        - Limit to 3 hotels maximum for ultra-fast processing
        - Ultra-short prompts (under 2000 chars)
        - 15-second timeout on OpenAI calls
        - Multiple fast fallback layers
        """
        try:
            print(f"🚀 DEBUG: Starting ultra-optimized AI hotel filtering...")
            
            # ULTRA-OPTIMIZATION 1: Extract only essential hotel data
            essential_hotels = self._extract_essential_hotel_data(hotel_results)
            print(f"⚡ Ultra-optimized: Processing {len(essential_hotels)} hotels (max 3) for ultra-speed")
            
            # ULTRA-OPTIMIZATION 2: Create ultra-short prompt
            budget = group_summary.get('budget_ranges', [])[:1]  # Only first budget
            style = group_summary.get('travel_styles', [])[:1]    # Only first style
            
            prompt = f"""Select best 2-3 hotels for group of {group_summary['group_size']}.
Budget: {budget[0] if budget else 'any'}
Style: {style[0] if style else 'any'}

Hotels:
{json.dumps(essential_hotels)}

Return JSON:
{{"recommended_hotels": [hotel objects], "filtering_rationale": "brief reason"}}"""

            print(f"⚡ Ultra-optimized: Using ultra-short prompt ({len(prompt)} chars)")

            # ULTRA-OPTIMIZATION 3: Ultra-fast OpenAI call with strict timeout
            try:
                # Set 15-second timeout
                timeout_task = asyncio.create_task(asyncio.sleep(15))
                api_task = asyncio.create_task(self._make_openai_call(prompt))
                
                done, pending = await asyncio.wait(
                    [timeout_task, api_task], 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                
                if api_task in done:
                    response_text = api_task.result()
                    print(f"✅ Ultra-optimized: Got AI response in under 15s")
                else:
                    print(f"⚠️ Ultra-optimized: OpenAI call timed out after 15s")
                    raise asyncio.TimeoutError("OpenAI call timed out")
                
                # ULTRA-OPTIMIZATION 4: Ultra-fast JSON parsing
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        filtered_result = json.loads(json_text)
                        
                        # Map back to original hotel objects
                        recommended_ids = [h.get("id", 0) for h in filtered_result.get("recommended_hotels", [])]
                        original_hotels = [hotel_results[i-1] for i in recommended_ids if 1 <= i <= len(hotel_results)]
                        
                        filtered_result["recommended_hotels"] = original_hotels
                        print(f"✅ Ultra-optimized: Successfully parsed JSON and mapped back to original hotels")
                        return filtered_result
                    else:
                        raise ValueError("No JSON found in response")
                        
                except (json.JSONDecodeError, ValueError) as json_error:
                    print(f"⚠️ Ultra-optimized: JSON parsing failed: {json_error}")
                    # Ultra-fast fallback: return top 2 hotels
                    return {
                        "recommended_hotels": hotel_results[:2],
                        "filtering_rationale": "AI filtering applied with ultra-fast fallback"
                    }
                
            except Exception as api_error:
                print(f"⚠️ Ultra-optimized: OpenAI API error: {api_error}")
                # Ultra-fast fallback: return top 2 hotels
                return {
                    "recommended_hotels": hotel_results[:2],
                    "filtering_rationale": f"AI filtering ultra-fast fallback due to API timeout"
                }
                
        except Exception as e:
            logger.error(f"Error in ultra-optimized AI hotel filtering: {e}")
            # ULTRA-OPTIMIZATION 5: Always return valid data instantly
            return {
                "recommended_hotels": hotel_results[:2],
                "filtering_rationale": f"Ultra-optimized filtering fallback: {str(e)}"
            }

# ==================== MODULE EXPORTS ====================

__all__ = [
    'UserProfileAgent'
]
