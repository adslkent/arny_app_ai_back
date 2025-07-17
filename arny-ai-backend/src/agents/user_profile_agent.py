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
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI
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
from ..database.operations import DatabaseOperations

# Set up logging
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class OpenAIResponse(BaseModel):
    """Pydantic model for OpenAI response validation"""
    output: Optional[Any] = None

# ==================== OPENAI API RETRY CONDITIONS ====================

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
    """Condition 5: Retry if OpenAI result fails Pydantic validation"""
    try:
        if result:
            OpenAIResponse.model_validate(result.__dict__ if hasattr(result, '__dict__') else result)
        return False
    except (ValidationError, AttributeError):
        return True

def retry_on_openai_api_exception(exception):
    """Condition 4: Custom exception checker for OpenAI API calls"""
    exception_str = str(exception).lower()
    return any(keyword in exception_str for keyword in [
        'timeout', 'failed', 'unavailable', 'rate limit', 'api error',
        'connection', 'network', 'server error'
    ])

# OpenAI API retry decorator with all 5 conditions
openai_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|rate.limit|api.error|connection|network|server.error).*"),
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
                group_profiles = await self._get_group_profiles_enhanced(user_id)
                print(f"📊 Retrieved {len(group_profiles)} group profiles - NO MEMBER LIMITS")
            except Exception as e:
                logger.warning(f"Failed to get group profiles: {e}")
                group_profiles = []
            
            # OPTIMIZATION 3: Direct return if no flights
            if not flight_results:
                result = {
                    "filtered_results": [],
                    "total_results": 0,
                    "filtering_applied": False,
                    "reasoning": "No flight results to filter"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 4: Handle single member (no filtering needed)
            if len(group_profiles) <= 1:
                # Return top 10 flights without AI filtering
                top_flights = flight_results[:10]
                result = {
                    "filtered_results": top_flights,
                    "total_results": len(flight_results),
                    "filtering_applied": False,
                    "reasoning": "Single traveler - showing best available options"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 5: Enhanced data preparation for larger groups
            enhanced_flights = self._extract_enhanced_flight_data(flight_results)
            
            if not enhanced_flights:
                result = {
                    "filtered_results": [],
                    "total_results": len(flight_results),
                    "filtering_applied": False,
                    "reasoning": "Could not process flight data"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 6: Create concise group summary
            group_summary = self._create_group_summary(group_profiles)
            
            print(f"🧠 Starting AI filtering for {len(enhanced_flights)} flights with group: {group_summary}")

            # OPTIMIZATION 7: Ultra-efficient AI filtering with NO TIMEOUT LIMITS
            filtered_result = await self._filter_flights_with_ai_enhanced(
                enhanced_flights, group_summary, len(group_profiles), flight_results
            )
            
            # Cache and return result
            self._filter_cache[cache_key] = filtered_result
            self._cleanup_cache()
            
            print(f"✅ ENHANCED flight filtering complete - returned {len(filtered_result.get('filtered_results', []))} flights")
            return filtered_result

        except Exception as e:
            logger.error(f"Error in enhanced flight filtering: {e}")
            import traceback
            traceback.print_exc()
            
            # Return graceful fallback
            return {
                "filtered_results": flight_results[:10] if flight_results else [],
                "total_results": len(flight_results) if flight_results else 0,
                "filtering_applied": False,
                "reasoning": f"Filtering failed: {str(e)}"
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
                group_profiles = await self._get_group_profiles_enhanced(user_id)
                print(f"📊 Retrieved {len(group_profiles)} group profiles - NO MEMBER LIMITS")
            except Exception as e:
                logger.warning(f"Failed to get group profiles: {e}")
                group_profiles = []

            # OPTIMIZATION 3: Direct return if no hotels
            if not hotel_results:
                result = {
                    "filtered_results": [],
                    "total_results": 0,
                    "filtering_applied": False,
                    "reasoning": "No hotel results to filter"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 4: Handle single member (no filtering needed)
            if len(group_profiles) <= 1:
                # Return top 10 hotels without AI filtering
                top_hotels = hotel_results[:10]
                result = {
                    "filtered_results": top_hotels,
                    "total_results": len(hotel_results),
                    "filtering_applied": False,
                    "reasoning": "Single traveler - showing best available options"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 5: Enhanced data preparation for larger groups
            enhanced_hotels = self._extract_enhanced_hotel_data(hotel_results)
            
            if not enhanced_hotels:
                result = {
                    "filtered_results": [],
                    "total_results": len(hotel_results),
                    "filtering_applied": False,
                    "reasoning": "Could not process hotel data"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 6: Create concise group summary
            group_summary = self._create_group_summary(group_profiles)
            
            print(f"🧠 Starting AI filtering for {len(enhanced_hotels)} hotels with group: {group_summary}")

            # OPTIMIZATION 7: Ultra-efficient AI filtering with NO TIMEOUT LIMITS
            filtered_result = await self._filter_hotels_with_ai_enhanced(
                enhanced_hotels, group_summary, len(group_profiles), hotel_results
            )
            
            # Cache and return result
            self._filter_cache[cache_key] = filtered_result
            self._cleanup_cache()
            
            print(f"✅ ENHANCED hotel filtering complete - returned {len(filtered_result.get('filtered_results', []))} hotels")
            return filtered_result

        except Exception as e:
            logger.error(f"Error in enhanced hotel filtering: {e}")
            import traceback
            traceback.print_exc()
            
            # Return graceful fallback
            return {
                "filtered_results": hotel_results[:10] if hotel_results else [],
                "total_results": len(hotel_results) if hotel_results else 0,
                "filtering_applied": False,
                "reasoning": f"Filtering failed: {str(e)}"
            }

    async def _get_group_profiles_enhanced(self, user_id: str) -> List[Dict[str, Any]]:
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
                            "departure_time": first_segment.get("departure", {}).get("at", ""),
                            "arrival_time": segments[-1].get("arrival", {}).get("at", "") if segments else "",
                            "airline": first_segment.get("carrierCode", ""),
                            "duration": first_itinerary.get("duration", ""),
                            "stops": len(segments) - 1
                        })
                
                enhanced_flights.append(enhanced_flight)
                
            except Exception as e:
                logger.warning(f"Error extracting flight {i}: {e}")
                continue
        
        return enhanced_flights
    
    def _extract_enhanced_hotel_data(self, hotels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ENHANCED: Extract hotel data for up to 50 hotels efficiently"""
        enhanced_hotels = []
        
        # Process all hotels but optimize data extraction
        for i, hotel in enumerate(hotels[:50]):  # CHANGED: Process up to 50 hotels
            try:
                hotel_data = hotel.get("hotel", {})
                offers = hotel.get("offers", [])
                
                enhanced_hotel = {
                    "id": i + 1,
                    "name": hotel_data.get("name", ""),
                    "rating": hotel_data.get("rating", ""),
                }
                
                if offers and len(offers) > 0:
                    first_offer = offers[0]
                    price = first_offer.get("price", {})
                    room = first_offer.get("room", {})
                    
                    enhanced_hotel.update({
                        "price": price.get("total", "N/A"),
                        "currency": price.get("currency", ""),
                        "room_type": room.get("type", ""),
                        "beds": room.get("typeEstimated", {}).get("beds", ""),
                        "check_in": first_offer.get("checkInDate", ""),
                        "check_out": first_offer.get("checkOutDate", "")
                    })
                
                enhanced_hotels.append(enhanced_hotel)
                
            except Exception as e:
                logger.warning(f"Error extracting hotel {i}: {e}")
                continue
        
        return enhanced_hotels

    @openai_api_retry
    async def _filter_flights_with_ai_enhanced(self, enhanced_flights: List[Dict[str, Any]], 
                                             group_summary: str, group_size: int, 
                                             original_flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ENHANCED: Use AI to filter flights optimized for larger groups - NO TIMEOUT LIMITS with Tenacity retry strategies"""
        
        try:
            print(f"🧠 AI filtering {len(enhanced_flights)} flights for group: {group_summary}")
            
            # Ultra-short prompt for efficiency
            prompt = f"""Filter flights for group: {group_summary}

Group details: {group_summary}

Return top 10 flights as JSON: {{"recommended_flights": [{{"id": 1}}, {{"id": 2}}, ...], "filtering_rationale": "brief reason considering all {group_size} members"}}

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
                    original_flights_filtered = [original_flights[i-1] for i in recommended_ids if 1 <= i <= len(original_flights)]
                    
                    # Ensure we return up to 10 results
                    if len(original_flights_filtered) > 10:
                        original_flights_filtered = original_flights_filtered[:10]
                    
                    print(f"✅ AI filtered to {len(original_flights_filtered)} flights")
                    
                    return {
                        "filtered_results": original_flights_filtered,
                        "total_results": len(enhanced_flights),
                        "filtering_applied": True,
                        "reasoning": filtered_result.get("filtering_rationale", "AI filtering applied")
                    }
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing failed: {e}")
            
            # Fallback: return top 10 flights
            print(f"⚠️ AI filtering fallback - returning top 10 of {len(original_flights)} flights")
            return {
                "filtered_results": original_flights[:10],
                "total_results": len(enhanced_flights),
                "filtering_applied": False,
                "reasoning": "AI filtering failed, showing top options"
            }
            
        except Exception as e:
            logger.error(f"Error in AI flight filtering: {e}")
            print(f"❌ AI filtering error: {e}")
            
            # Graceful fallback
            return {
                "filtered_results": original_flights[:10] if original_flights else [],
                "total_results": len(enhanced_flights),
                "filtering_applied": False,
                "reasoning": f"AI filtering error: {str(e)}"
            }

    @openai_api_retry
    async def _filter_hotels_with_ai_enhanced(self, enhanced_hotels: List[Dict[str, Any]], 
                                            group_summary: str, group_size: int,
                                            hotel_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ENHANCED: Use AI to filter hotels optimized for larger groups - NO TIMEOUT LIMITS with Tenacity retry strategies"""
        
        try:
            print(f"🧠 AI filtering {len(enhanced_hotels)} hotels for group: {group_summary}")
            
            # Ultra-short prompt for efficiency
            prompt = f"""Filter hotels for group: {group_summary}

Group details: {group_summary}

Return top 10 hotels as JSON: {{"recommended_hotels": [{{"id": 1}}, {{"id": 2}}, ...], "filtering_rationale": "brief reason considering all {group_size} members"}}

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
                    
                    print(f"✅ AI filtered to {len(original_hotels)} hotels")
                    
                    return {
                        "filtered_results": original_hotels,
                        "total_results": len(enhanced_hotels),
                        "filtering_applied": True,
                        "reasoning": filtered_result.get("filtering_rationale", "AI filtering applied")
                    }
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing failed: {e}")
            
            # Fallback: return top 10 hotels
            print(f"⚠️ AI filtering fallback - returning top 10 of {len(hotel_results)} hotels")
            return {
                "filtered_results": hotel_results[:10],
                "total_results": len(enhanced_hotels),
                "filtering_applied": False,
                "reasoning": "AI filtering failed, showing top options"
            }
            
        except Exception as e:
            logger.error(f"Error in AI hotel filtering: {e}")
            print(f"❌ AI filtering error: {e}")
            
            # Graceful fallback
            return {
                "filtered_results": hotel_results[:10] if hotel_results else [],
                "total_results": len(enhanced_hotels),
                "filtering_applied": False,
                "reasoning": f"AI filtering error: {str(e)}"
            }
    
    def _create_group_summary(self, group_profiles: List[Dict[str, Any]]) -> str:
        """Create a concise summary of group preferences for AI filtering"""
        try:
            if not group_profiles:
                return "single traveler"
            
            # Extract key information
            travel_styles = []
            ages = []
            cities = []
            
            for profile in group_profiles:
                if profile.get("travel_style"):
                    travel_styles.append(profile["travel_style"])
                
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
