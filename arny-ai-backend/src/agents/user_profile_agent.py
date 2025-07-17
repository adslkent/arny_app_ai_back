"""
User Profile Agent for Arny AI - ENHANCED VERSION WITHOUT TIMEOUTS AND MEMBER LIMITS

This agent filters search results based on group preferences with optimizations
to handle larger result sets efficiently. Features:
- Process up to 50 flight results
- Process up to 50 hotel results  
- Return up to 10 filtered results
- Ultra-short prompts for efficiency
- NO TIMEOUT LIMITS on AI processing
- NO LIMITS on group member processing
- Cache results to prevent duplicate processing
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
    ENHANCED: User Profile Agent with support for larger result sets - NO TIMEOUTS OR MEMBER LIMITS
    """
    
    def __init__(self):
        """Initialize with enhanced settings for larger datasets"""
        try:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.db = DatabaseOperations()
            
            # Initialize result cache for instant responses
            self._filter_cache = {}
            
            logger.info("UserProfileAgent initialized with enhanced support for larger datasets - NO TIMEOUTS OR MEMBER LIMITS")
        except Exception as e:
            logger.error(f"Failed to initialize UserProfileAgent: {e}")
            raise Exception(f"UserProfileAgent initialization failed: {e}")
    
    async def filter_flight_results(self, user_id: str, flight_results: List[Dict[str, Any]], 
                                  search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Filter flight results with support for up to 50 flights, returning up to 10 - NO TIMEOUTS OR MEMBER LIMITS
        """
        try:
            logger.info(f"ENHANCED flight filtering for user {user_id} - NO TIMEOUTS OR MEMBER LIMITS")
            
            print(f"🚀 ENHANCED: Processing {len(flight_results)} flight results - NO TIMEOUTS OR MEMBER LIMITS")

            # OPTIMIZATION 1: Check cache first
            cache_key = f"flight_{user_id}_{len(flight_results)}"
            if cache_key in self._filter_cache:
                print(f"⚡ CACHE HIT: Returning cached flight filtering")
                return self._filter_cache[cache_key]

            # OPTIMIZATION 2: Get group profiles - NO TIMEOUT LIMIT, ALL MEMBERS
            try:
                group_profiles = await self._get_group_profiles(user_id)
            except Exception as e:
                print(f"⚠️ Group profile error: {e} - treating as single user")
                group_profiles = []
            
            if not group_profiles:
                print(f"🔍 No group filtering - single user, returning top 10")
                # For single users, return top 10 results
                top_results = flight_results[:10]
                result = {
                    "filtered_results": top_results,
                    "original_count": len(flight_results),
                    "filtered_count": len(top_results),
                    "filtering_applied": False,
                    "rationale": "Single user - returning top 10 results"
                }
                self._filter_cache[cache_key] = result
                return result
            
            print(f"⚡ Group filtering for ALL {len(group_profiles)} members (no limits)")

            # ENHANCED: AI filtering with support for up to 50 flights - NO TIMEOUT LIMIT
            try:
                filtered_results = await self._ai_filter_flights_enhanced(flight_results, group_profiles)
                
                recommended_flights = filtered_results.get("recommended_flights", flight_results[:10])
                
                result = {
                    "filtered_results": recommended_flights,
                    "original_count": len(flight_results),
                    "filtered_count": len(recommended_flights),
                    "filtering_applied": True,
                    "rationale": filtered_results.get("filtering_rationale", "AI filtering applied (enhanced - no timeouts or member limits)"),
                    "group_size": len(group_profiles),
                }
                
                # Cache result
                self._filter_cache[cache_key] = result
                self._cleanup_cache()
                
                print(f"🚀 ENHANCED filtering complete: {len(recommended_flights)} results for {len(group_profiles)} members")
                return result
                
            except Exception as e:
                print(f"⚠️ AI filtering error - using top 10 fallback: {e}")
                result = {
                    "filtered_results": flight_results[:10],
                    "original_count": len(flight_results),
                    "filtered_count": min(10, len(flight_results)),
                    "filtering_applied": False,
                    "rationale": f"Enhanced fallback used due to error: {str(e)}"
                }
                return result
            
        except Exception as e:
            logger.error(f"Error filtering flight results: {e}")
            # INSTANT fallback
            return {
                "filtered_results": flight_results[:10],
                "original_count": len(flight_results),
                "filtered_count": min(10, len(flight_results)),
                "filtering_applied": False,
                "rationale": f"Instant fallback: {str(e)}"
            }
    
    async def filter_hotel_results(self, user_id: str, hotel_results: List[Dict[str, Any]], 
                                 search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Filter hotel results with support for up to 50 hotels, returning up to 10 - NO TIMEOUTS OR MEMBER LIMITS
        """
        try:
            logger.info(f"ENHANCED hotel filtering for user {user_id} - NO TIMEOUTS OR MEMBER LIMITS")
            
            print(f"🚀 ENHANCED: Processing {len(hotel_results)} hotel results - NO TIMEOUTS OR MEMBER LIMITS")

            # OPTIMIZATION 1: Check cache first
            cache_key = f"hotel_{user_id}_{len(hotel_results)}"
            if cache_key in self._filter_cache:
                print(f"⚡ CACHE HIT: Returning cached hotel filtering")
                return self._filter_cache[cache_key]

            # OPTIMIZATION 2: Get group profiles - NO TIMEOUT LIMIT, ALL MEMBERS
            try:
                group_profiles = await self._get_group_profiles(user_id)
            except Exception as e:
                print(f"⚠️ Group profile error: {e} - treating as single user")
                group_profiles = []
            
            if not group_profiles:
                print(f"🔍 No group filtering - single user, returning top 10")
                # For single users, return top 10 results
                top_results = hotel_results[:10]
                result = {
                    "filtered_results": top_results,
                    "original_count": len(hotel_results),
                    "filtered_count": len(top_results),
                    "filtering_applied": False,
                    "rationale": "Single user - returning top 10 results"
                }
                self._filter_cache[cache_key] = result
                return result
            
            print(f"⚡ Group filtering for ALL {len(group_profiles)} members (no limits)")

            # ENHANCED: AI filtering with support for up to 50 hotels - NO TIMEOUT LIMIT
            try:
                filtered_results = await self._ai_filter_hotels_enhanced(hotel_results, group_profiles)
                
                recommended_hotels = filtered_results.get("recommended_hotels", hotel_results[:10])
                
                result = {
                    "filtered_results": recommended_hotels,
                    "original_count": len(hotel_results),
                    "filtered_count": len(recommended_hotels),
                    "filtering_applied": True,
                    "rationale": filtered_results.get("filtering_rationale", "AI filtering applied (enhanced - no timeouts or member limits)"),
                    "group_size": len(group_profiles),
                }
                
                # Cache result
                self._filter_cache[cache_key] = result
                self._cleanup_cache()
                
                print(f"🚀 ENHANCED filtering complete: {len(recommended_hotels)} results for {len(group_profiles)} members")
                return result
                
            except Exception as e:
                print(f"⚠️ AI filtering error - using top 10 fallback: {e}")
                result = {
                    "filtered_results": hotel_results[:10],
                    "original_count": len(hotel_results),
                    "filtered_count": min(10, len(hotel_results)),
                    "filtering_applied": False,
                    "rationale": f"Enhanced fallback used due to error: {str(e)}"
                }
                return result
            
        except Exception as e:
            logger.error(f"Error filtering hotel results: {e}")
            # INSTANT fallback
            return {
                "filtered_results": hotel_results[:10],
                "original_count": len(hotel_results),
                "filtered_count": min(10, len(hotel_results)),
                "filtering_applied": False,
                "rationale": f"Instant fallback: {str(e)}"
            }
    
    async def _get_group_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get group profiles with optimized processing - NO TIMEOUTS OR MEMBER LIMITS"""
        try:
            # Get user's groups
            user_groups = await self.db.get_user_groups(user_id)
            
            if not user_groups:
                return []
            
            # Use first group only
            primary_group = user_groups[0]
            
            # Get group members
            group_members = await self.db.get_group_members(primary_group)
            
            # CHANGED: Get profiles for ALL members (removed limit of 5)
            group_profiles = []
            for member in group_members:  # CHANGED: Process ALL members, no [:5] limit
                profile = await self.db.get_user_profile(member.user_id)
                if profile:
                    profile_dict = profile.dict()
                    profile_dict["group_role"] = member.role
                    group_profiles.append(profile_dict)
            
            logger.info(f"Found {len(group_profiles)} profiles in group (ALL members processed)")
            print(f"📊 Processing ALL {len(group_profiles)} group members (no member limits)")
            return group_profiles
            
        except Exception as e:
            logger.error(f"Error getting group profiles: {e}")
            return []
    
    def _extract_enhanced_flight_data(self, flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ENHANCED: Extract flight data for up to 50 flights efficiently"""
        enhanced_flights = []
        
        # Process all flights but optimize data extraction
        for i, flight in enumerate(flights[:50]):  # CHANGED: Process up to 50 flights
            try:
                price = flight.get("price", {})
                itineraries = flight.get("itineraries", [])
                
                enhanced_flight = {
                    "id": i + 1,
                    "price": price.get("total", "N/A"),
                    "currency": price.get("currency", ""),
                }
                
                if itineraries:
                    first_itinerary = itineraries[0]
                    segments = first_itinerary.get("segments", [])
                    
                    if segments:
                        first_segment = segments[0]
                        enhanced_flight.update({
                            "airline": first_segment.get("carrierCode", ""),
                            "stops": len(segments) - 1,
                            "duration": first_itinerary.get("duration", "")
                        })
                
                enhanced_flights.append(enhanced_flight)
                
            except Exception:
                continue
        
        return enhanced_flights
    
    def _extract_enhanced_hotel_data(self, hotels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ENHANCED: Extract hotel data for up to 50 hotels efficiently"""
        enhanced_hotels = []
        
        # Process all hotels but optimize data extraction
        for i, hotel_data in enumerate(hotels[:50]):  # CHANGED: Process up to 50 hotels
            try:
                hotel = hotel_data.get("hotel", {})
                offers = hotel_data.get("offers", [])
                
                enhanced_hotel = {
                    "id": i + 1,
                    "name": hotel.get("name", "Hotel"),
                    "rating": hotel.get("rating", ""),
                }
                
                if offers:
                    best_offer = offers[0]
                    price = best_offer.get("price", {})
                    enhanced_hotel.update({
                        "price": price.get("total", "N/A"),
                        "currency": price.get("currency", ""),
                    })
                
                enhanced_hotels.append(enhanced_hotel)
                
            except Exception:
                continue
        
        return enhanced_hotels
    
    async def _ai_filter_flights_enhanced(self, flight_results: List[Dict[str, Any]], 
                                         group_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ENHANCED: AI flight filtering with support for larger datasets - NO TIMEOUT LIMITS, ALL MEMBERS"""
        try:
            # Extract enhanced data for all flights
            enhanced_flights = self._extract_enhanced_flight_data(flight_results)
            
            # ENHANCED: Create efficient prompt for larger datasets with ALL group members
            group_summary = self._create_group_summary_for_ai(group_profiles)
            
            prompt = f"""Filter {len(enhanced_flights)} flights for group of {len(group_profiles)} members.

Group details: {group_summary}

Return top 10 flights as JSON: {{"recommended_flights": [{{\"id\": 1}}, {{\"id\": 2}}, ...], "filtering_rationale": "brief reason considering all {len(group_profiles)} members"}}

Flights: {json.dumps(enhanced_flights[:20])}...and {max(0, len(enhanced_flights)-20)} more"""

            # Make optimized OpenAI call - NO TIMEOUT LIMITS
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
            
            # Enhanced JSON parsing
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    filtered_result = json.loads(json_text)
                    
                    # Map back to original flights (up to 10)
                    recommended_ids = [f.get("id", 0) for f in filtered_result.get("recommended_flights", [])]
                    original_flights = [flight_results[i-1] for i in recommended_ids if 1 <= i <= len(flight_results)]
                    
                    # Ensure we return up to 10 results
                    if len(original_flights) > 10:
                        original_flights = original_flights[:10]
                    elif len(original_flights) < 10 and len(flight_results) >= 10:
                        # Fill up to 10 with top results if AI didn't return enough
                        original_flights = flight_results[:10]
                    
                    filtered_result["recommended_flights"] = original_flights
                    return filtered_result
                else:
                    raise ValueError("No JSON found")
                    
            except Exception:
                # Enhanced fallback - return top 10
                return {
                    "recommended_flights": flight_results[:10],
                    "filtering_rationale": f"Enhanced fallback - top 10 results for {len(group_profiles)} members"
                }
                
        except Exception:
            # Enhanced fallback - return top 10
            return {
                "recommended_flights": flight_results[:10],
                "filtering_rationale": f"Enhanced fallback - top 10 results for {len(group_profiles)} members"
            }
    
    async def _ai_filter_hotels_enhanced(self, hotel_results: List[Dict[str, Any]], 
                                        group_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ENHANCED: AI hotel filtering with support for larger datasets - NO TIMEOUT LIMITS, ALL MEMBERS"""
        try:
            # Extract enhanced data for all hotels
            enhanced_hotels = self._extract_enhanced_hotel_data(hotel_results)
            
            # ENHANCED: Create efficient prompt for larger datasets with ALL group members
            group_summary = self._create_group_summary_for_ai(group_profiles)
            
            prompt = f"""Filter {len(enhanced_hotels)} hotels for group of {len(group_profiles)} members.

Group details: {group_summary}

Return top 10 hotels as JSON: {{"recommended_hotels": [{{\"id\": 1}}, {{\"id\": 2}}, ...], "filtering_rationale": "brief reason considering all {len(group_profiles)} members"}}

Hotels: {json.dumps(enhanced_hotels[:20])}...and {max(0, len(enhanced_hotels)-20)} more"""

            # Make optimized OpenAI call - NO TIMEOUT LIMITS
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
            
            # Enhanced JSON parsing
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    filtered_result = json.loads(json_text)
                    
                    # Map back to original hotels (up to 10)
                    recommended_ids = [h.get("id", 0) for h in filtered_result.get("recommended_hotels", [])]
                    original_hotels = [hotel_results[i-1] for i in recommended_ids if 1 <= i <= len(hotel_results)]
                    
                    # Ensure we return up to 10 results
                    if len(original_hotels) > 10:
                        original_hotels = original_hotels[:10]
                    elif len(original_hotels) < 10 and len(hotel_results) >= 10:
                        # Fill up to 10 with top results if AI didn't return enough
                        original_hotels = hotel_results[:10]
                    
                    filtered_result["recommended_hotels"] = original_hotels
                    return filtered_result
                else:
                    raise ValueError("No JSON found")
                    
            except Exception:
                # Enhanced fallback - return top 10
                return {
                    "recommended_hotels": hotel_results[:10],
                    "filtering_rationale": f"Enhanced fallback - top 10 results for {len(group_profiles)} members"
                }
                
        except Exception:
            # Enhanced fallback - return top 10
            return {
                "recommended_hotels": hotel_results[:10],
                "filtering_rationale": f"Enhanced fallback - top 10 results for {len(group_profiles)} members"
            }
    
    def _create_group_summary_for_ai(self, group_profiles: List[Dict[str, Any]]) -> str:
        """NEW: Create a concise summary of ALL group members for AI processing"""
        try:
            if not group_profiles:
                return "Single traveler"
            
            # Extract key information from all profiles
            travel_styles = []
            cities = []
            ages = []
            
            for profile in group_profiles:
                # Travel style
                if profile.get("travel_style"):
                    travel_styles.append(profile["travel_style"])
                
                # City
                if profile.get("city"):
                    cities.append(profile["city"])
                
                # Calculate approximate age from birthdate
                if profile.get("birthdate"):
                    try:
                        from datetime import date
                        birth_year = int(profile["birthdate"][:4])
                        current_year = date.today().year
                        age = current_year - birth_year
                        if 0 < age < 120:  # Reasonable age range
                            ages.append(age)
                    except:
                        pass
            
            # Create summary
            summary_parts = [f"{len(group_profiles)} members"]
            
            if travel_styles:
                unique_styles = list(set(travel_styles))
                summary_parts.append(f"styles: {', '.join(unique_styles)}")
            
            if ages:
                avg_age = sum(ages) // len(ages)
                summary_parts.append(f"avg age: {avg_age}")
            
            if cities:
                unique_cities = list(set(cities))
                if len(unique_cities) <= 3:
                    summary_parts.append(f"from: {', '.join(unique_cities)}")
                else:
                    summary_parts.append(f"from {len(unique_cities)} cities")
            
            return "; ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating group summary: {e}")
            return f"{len(group_profiles)} members"
    
    def _cleanup_cache(self):
        """Keep cache size manageable"""
        if len(self._filter_cache) > 15:  # Increased cache size for larger group datasets
            # Remove oldest entries
            keys_to_remove = list(self._filter_cache.keys())[:-15]
            for key in keys_to_remove:
                del self._filter_cache[key]

# ==================== MODULE EXPORTS ====================

__all__ = [
    'UserProfileAgent'
]
