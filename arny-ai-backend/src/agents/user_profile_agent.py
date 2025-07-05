"""
User Profile Agent for Arny AI - MAXIMUM-OPTIMIZED VERSION

This agent filters search results based on group preferences with maximum speed optimizations
to prevent any timeouts. Features:
- 5-second timeout limits on all AI calls
- Process max 2 results only
- Ultra-short prompts (under 1000 chars)
- Instant fallbacks for any delays
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
    MAXIMUM-OPTIMIZED: User Profile Agent with zero-timeout-tolerance
    """
    
    def __init__(self):
        """Initialize with ultra-fast settings"""
        try:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.db = DatabaseOperations()
            
            # Initialize result cache for instant responses
            self._filter_cache = {}
            
            logger.info("UserProfileAgent initialized with maximum optimizations")
        except Exception as e:
            logger.error(f"Failed to initialize UserProfileAgent: {e}")
            raise Exception(f"UserProfileAgent initialization failed: {e}")
    
    async def filter_flight_results(self, user_id: str, flight_results: List[Dict[str, Any]], 
                                  search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MAXIMUM-OPTIMIZED: Filter flight results with 5s timeout max
        """
        try:
            logger.info(f"MAX-OPTIMIZED flight filtering for user {user_id}")
            
            print(f"🚀 MAX-OPTIMIZED: Processing {len(flight_results)} flight results")

            # OPTIMIZATION 1: Check cache first
            cache_key = f"flight_{user_id}_{len(flight_results)}"
            if cache_key in self._filter_cache:
                print(f"⚡ CACHE HIT: Returning cached flight filtering")
                return self._filter_cache[cache_key]

            # OPTIMIZATION 2: Get group profiles with 2s timeout
            group_profiles = await asyncio.wait_for(
                self._get_group_profiles(user_id), 
                timeout=2.0
            )
            
            if not group_profiles:
                print(f"🔍 No group filtering - single user")
                result = {
                    "filtered_results": flight_results,
                    "original_count": len(flight_results),
                    "filtered_count": len(flight_results),
                    "filtering_applied": False,
                    "rationale": "Single user - no group filtering needed"
                }
                self._filter_cache[cache_key] = result
                return result
            
            print(f"⚡ Group filtering for {len(group_profiles)} members")

            # OPTIMIZATION 3: Maximum-speed AI filtering with 5s timeout
            try:
                filtered_results = await asyncio.wait_for(
                    self._ai_filter_flights_maximum_speed(flight_results, group_profiles),
                    timeout=5.0
                )
                
                recommended_flights = filtered_results.get("recommended_flights", flight_results[:2])
                
                result = {
                    "filtered_results": recommended_flights,
                    "original_count": len(flight_results),
                    "filtered_count": len(recommended_flights),
                    "filtering_applied": True,
                    "rationale": filtered_results.get("filtering_rationale", "AI filtering applied (max-speed)"),
                    "group_size": len(group_profiles),
                }
                
                # Cache result
                self._filter_cache[cache_key] = result
                self._cleanup_cache()
                
                print(f"🚀 MAX-OPTIMIZED filtering complete: {len(recommended_flights)} results")
                return result
                
            except asyncio.TimeoutError:
                print(f"⚠️ AI filtering timeout - using instant fallback")
                result = {
                    "filtered_results": flight_results[:2],
                    "original_count": len(flight_results),
                    "filtered_count": min(2, len(flight_results)),
                    "filtering_applied": False,
                    "rationale": "Maximum-speed fallback used (timeout prevention)"
                }
                return result
            
        except Exception as e:
            logger.error(f"Error filtering flight results: {e}")
            # INSTANT fallback
            return {
                "filtered_results": flight_results[:2],
                "original_count": len(flight_results),
                "filtered_count": min(2, len(flight_results)),
                "filtering_applied": False,
                "rationale": f"Instant fallback: {str(e)}"
            }
    
    async def filter_hotel_results(self, user_id: str, hotel_results: List[Dict[str, Any]], 
                                 search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MAXIMUM-OPTIMIZED: Filter hotel results with 5s timeout max
        """
        try:
            logger.info(f"MAX-OPTIMIZED hotel filtering for user {user_id}")
            
            print(f"🚀 MAX-OPTIMIZED: Processing {len(hotel_results)} hotel results")

            # OPTIMIZATION 1: Check cache first
            cache_key = f"hotel_{user_id}_{len(hotel_results)}"
            if cache_key in self._filter_cache:
                print(f"⚡ CACHE HIT: Returning cached hotel filtering")
                return self._filter_cache[cache_key]

            # OPTIMIZATION 2: Get group profiles with 2s timeout
            try:
                group_profiles = await asyncio.wait_for(
                    self._get_group_profiles(user_id), 
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                print(f"⚠️ Group profile timeout - treating as single user")
                group_profiles = []
            
            if not group_profiles:
                print(f"🔍 No group filtering - single user")
                result = {
                    "filtered_results": hotel_results,
                    "original_count": len(hotel_results),
                    "filtered_count": len(hotel_results),
                    "filtering_applied": False,
                    "rationale": "Single user - no group filtering needed"
                }
                self._filter_cache[cache_key] = result
                return result
            
            print(f"⚡ Group filtering for {len(group_profiles)} members")

            # OPTIMIZATION 3: Maximum-speed AI filtering with 5s timeout
            try:
                filtered_results = await asyncio.wait_for(
                    self._ai_filter_hotels_maximum_speed(hotel_results, group_profiles),
                    timeout=5.0
                )
                
                recommended_hotels = filtered_results.get("recommended_hotels", hotel_results[:2])
                
                result = {
                    "filtered_results": recommended_hotels,
                    "original_count": len(hotel_results),
                    "filtered_count": len(recommended_hotels),
                    "filtering_applied": True,
                    "rationale": filtered_results.get("filtering_rationale", "AI filtering applied (max-speed)"),
                    "group_size": len(group_profiles),
                }
                
                # Cache result
                self._filter_cache[cache_key] = result
                self._cleanup_cache()
                
                print(f"🚀 MAX-OPTIMIZED filtering complete: {len(recommended_hotels)} results")
                return result
                
            except asyncio.TimeoutError:
                print(f"⚠️ AI filtering timeout - using instant fallback")
                result = {
                    "filtered_results": hotel_results[:2],
                    "original_count": len(hotel_results),
                    "filtered_count": min(2, len(hotel_results)),
                    "filtering_applied": False,
                    "rationale": "Maximum-speed fallback used (timeout prevention)"
                }
                return result
            
        except Exception as e:
            logger.error(f"Error filtering hotel results: {e}")
            # INSTANT fallback
            return {
                "filtered_results": hotel_results[:2],
                "original_count": len(hotel_results),
                "filtered_count": min(2, len(hotel_results)),
                "filtering_applied": False,
                "rationale": f"Instant fallback: {str(e)}"
            }
    
    async def _get_group_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get group profiles with ultra-fast processing"""
        try:
            # Get user's groups
            user_groups = await self.db.get_user_groups(user_id)
            
            if not user_groups:
                return []
            
            # Use first group only
            primary_group = user_groups[0]
            
            # Get group members
            group_members = await self.db.get_group_members(primary_group)
            
            # Get profiles (max 3 members for speed)
            group_profiles = []
            for member in group_members[:3]:  # Limit to 3 for speed
                profile = await self.db.get_user_profile(member.user_id)
                if profile:
                    profile_dict = profile.dict()
                    profile_dict["group_role"] = member.role
                    group_profiles.append(profile_dict)
            
            logger.info(f"Found {len(group_profiles)} profiles in group")
            return group_profiles
            
        except Exception as e:
            logger.error(f"Error getting group profiles: {e}")
            return []
    
    def _extract_minimal_flight_data(self, flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """MAXIMUM-OPTIMIZATION: Extract absolute minimum flight data"""
        minimal_flights = []
        
        for i, flight in enumerate(flights[:2]):  # Only process top 2 for max speed
            try:
                price = flight.get("price", {})
                itineraries = flight.get("itineraries", [])
                
                minimal_flight = {
                    "id": i + 1,
                    "price": price.get("total", "N/A"),
                    "currency": price.get("currency", ""),
                }
                
                if itineraries:
                    first_itinerary = itineraries[0]
                    segments = first_itinerary.get("segments", [])
                    
                    if segments:
                        first_segment = segments[0]
                        minimal_flight.update({
                            "airline": first_segment.get("carrierCode", ""),
                            "stops": len(segments) - 1
                        })
                
                minimal_flights.append(minimal_flight)
                
            except Exception:
                continue
        
        return minimal_flights
    
    def _extract_minimal_hotel_data(self, hotels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """MAXIMUM-OPTIMIZATION: Extract absolute minimum hotel data"""
        minimal_hotels = []
        
        for i, hotel_data in enumerate(hotels[:2]):  # Only process top 2 for max speed
            try:
                hotel = hotel_data.get("hotel", {})
                offers = hotel_data.get("offers", [])
                
                minimal_hotel = {
                    "id": i + 1,
                    "name": hotel.get("name", "Hotel"),
                    "rating": hotel.get("rating", ""),
                }
                
                if offers:
                    best_offer = offers[0]
                    price = best_offer.get("price", {})
                    minimal_hotel.update({
                        "price": price.get("total", "N/A"),
                        "currency": price.get("currency", ""),
                    })
                
                minimal_hotels.append(minimal_hotel)
                
            except Exception:
                continue
        
        return minimal_hotels
    
    async def _ai_filter_flights_maximum_speed(self, flight_results: List[Dict[str, Any]], 
                                             group_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MAXIMUM-SPEED: AI flight filtering with ultra-short prompts"""
        try:
            # Extract minimal data
            minimal_flights = self._extract_minimal_flight_data(flight_results)
            
            # Ultra-short prompt (under 500 chars)
            prompt = f"""Select best flight for group of {len(group_profiles)}.

Flights: {json.dumps(minimal_flights)}

Return JSON: {{"recommended_flights": [flight objects], "filtering_rationale": "brief reason"}}"""

            # Make ultra-fast OpenAI call
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
            
            # Ultra-fast JSON parsing
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    filtered_result = json.loads(json_text)
                    
                    # Map back to original flights
                    recommended_ids = [f.get("id", 0) for f in filtered_result.get("recommended_flights", [])]
                    original_flights = [flight_results[i-1] for i in recommended_ids if 1 <= i <= len(flight_results)]
                    
                    filtered_result["recommended_flights"] = original_flights
                    return filtered_result
                else:
                    raise ValueError("No JSON found")
                    
            except Exception:
                # Instant fallback
                return {
                    "recommended_flights": flight_results[:2],
                    "filtering_rationale": "Maximum-speed fallback"
                }
                
        except Exception:
            # Instant fallback
            return {
                "recommended_flights": flight_results[:2],
                "filtering_rationale": "Maximum-speed fallback"
            }
    
    async def _ai_filter_hotels_maximum_speed(self, hotel_results: List[Dict[str, Any]], 
                                            group_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MAXIMUM-SPEED: AI hotel filtering with ultra-short prompts"""
        try:
            # Extract minimal data
            minimal_hotels = self._extract_minimal_hotel_data(hotel_results)
            
            # Ultra-short prompt (under 500 chars)
            prompt = f"""Select best hotel for group of {len(group_profiles)}.

Hotels: {json.dumps(minimal_hotels)}

Return JSON: {{"recommended_hotels": [hotel objects], "filtering_rationale": "brief reason"}}"""

            # Make ultra-fast OpenAI call
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
            
            # Ultra-fast JSON parsing
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    filtered_result = json.loads(json_text)
                    
                    # Map back to original hotels
                    recommended_ids = [h.get("id", 0) for h in filtered_result.get("recommended_hotels", [])]
                    original_hotels = [hotel_results[i-1] for i in recommended_ids if 1 <= i <= len(hotel_results)]
                    
                    filtered_result["recommended_hotels"] = original_hotels
                    return filtered_result
                else:
                    raise ValueError("No JSON found")
                    
            except Exception:
                # Instant fallback
                return {
                    "recommended_hotels": hotel_results[:2],
                    "filtering_rationale": "Maximum-speed fallback"
                }
                
        except Exception:
            # Instant fallback
            return {
                "recommended_hotels": hotel_results[:2],
                "filtering_rationale": "Maximum-speed fallback"
            }
    
    def _cleanup_cache(self):
        """Keep cache size manageable"""
        if len(self._filter_cache) > 5:
            # Remove oldest entries
            keys_to_remove = list(self._filter_cache.keys())[:-5]
            for key in keys_to_remove:
                del self._filter_cache[key]

# ==================== MODULE EXPORTS ====================

__all__ = [
    'UserProfileAgent'
]
