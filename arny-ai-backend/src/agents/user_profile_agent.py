"""
Enhanced User Profile Agent with Group Support and AI Filtering - ENHANCED VERSION

This module provides user profile management and intelligent filtering for travel search results.
Enhanced with support for larger datasets and always applies AI filtering for both individuals and groups.

Key Features:
- Support for up to 50 flight/hotel results
- Always applies AI filtering for both single travelers and groups  
- Enhanced group profile processing with no member limits
- Optimized caching for better performance
- Comprehensive retry strategies for API reliability

Usage example:
```python
from user_profile_agent import UserProfileAgent

# Initialize agent
profile_agent = UserProfileAgent()

# Filter flight results
result = await profile_agent.filter_flight_results(user_id, flight_results, search_params)
```
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import requests

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
from pydantic import ValidationError

from ..utils.config import config
from ..database.operations import DatabaseOperations

# Configure logging
logger = logging.getLogger(__name__)

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
    """Condition 5: Retry if validation fails on OpenAI result"""
    try:
        # Basic validation of expected structure
        if isinstance(result, dict):
            if "filtered_results" in result and "total_results" in result:
                return False  # Valid structure
            elif result.get("error"):
                return True  # Error result, should retry
        return False  # Accept other formats
    except Exception as e:
        logger.warning(f"Unexpected validation error: {e}")
        return True
    return False

def retry_on_openai_api_exception(exception):
    """Custom exception checker for OpenAI-specific exceptions"""
    return "openai" in str(type(exception)).lower() or "api" in str(exception).lower()

# ==================== COMBINED RETRY STRATEGIES ====================

# Primary retry strategy for critical OpenAI API operations
openai_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|api.?error|rate.?limit).*"),
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
    ALWAYS APPLIES AI FILTERING for both single travelers and groups
    """
    
    def __init__(self):
        """Initialize with enhanced settings for larger datasets"""
        try:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.db = DatabaseOperations()
            
            # Initialize result cache for instant responses
            self._filter_cache = {}
            
            logger.info("UserProfileAgent initialized with enhanced support for larger datasets - ALWAYS APPLIES AI FILTERING")
        except Exception as e:
            logger.error(f"Failed to initialize UserProfileAgent: {e}")
            raise Exception(f"UserProfileAgent initialization failed: {e}")
    
    async def filter_flight_results(self, user_id: str, flight_results: List[Dict[str, Any]], 
                                  search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Filter flight results with support for up to 50 flights, returning up to 10 - ALWAYS APPLIES AI FILTERING
        """
        try:
            logger.info(f"ENHANCED flight filtering for user {user_id} - ALWAYS APPLIES AI FILTERING")
            
            print(f"🚀 ENHANCED: Processing {len(flight_results)} flight results - ALWAYS APPLIES AI FILTERING")

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

            # REMOVED: Single traveler bypass condition - ALWAYS APPLY AI FILTERING NOW
            # OLD CODE WAS: if len(group_profiles) <= 1: return top_flights without filtering

            # OPTIMIZATION 4: Enhanced data preparation for all users (single and groups)
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

            # OPTIMIZATION 5: Create profile summary (works for both single travelers and groups)
            profile_summary = self._create_profile_summary(group_profiles)
            
            print(f"🧠 Starting AI filtering for {len(enhanced_flights)} flights with profile: {profile_summary}")

            # OPTIMIZATION 6: AI filtering applied for ALL users with NO TIMEOUT LIMITS
            filtered_result = await self._filter_flights_with_ai_enhanced(
                enhanced_flights, profile_summary, len(group_profiles) or 1, flight_results
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
        ENHANCED: Filter hotel results with support for up to 50 hotels, returning up to 10 - ALWAYS APPLIES AI FILTERING
        """
        try:
            logger.info(f"ENHANCED hotel filtering for user {user_id} - ALWAYS APPLIES AI FILTERING")
            
            print(f"🚀 ENHANCED: Processing {len(hotel_results)} hotel results - ALWAYS APPLIES AI FILTERING")

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

            # REMOVED: Single traveler bypass condition - ALWAYS APPLY AI FILTERING NOW
            # OLD CODE WAS: if len(group_profiles) <= 1: return top_hotels without filtering

            # OPTIMIZATION 4: Enhanced data preparation for all users (single and groups)
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

            # OPTIMIZATION 5: Create profile summary (works for both single travelers and groups)
            profile_summary = self._create_profile_summary(group_profiles)
            
            print(f"🧠 Starting AI filtering for {len(enhanced_hotels)} hotels with profile: {profile_summary}")

            # OPTIMIZATION 6: AI filtering applied for ALL users with NO TIMEOUT LIMITS
            filtered_result = await self._filter_hotels_with_ai_enhanced(
                enhanced_hotels, profile_summary, len(group_profiles) or 1, hotel_results
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
                # If no groups, get individual user profile
                user_profile = await self.db.get_user_profile(user_id)
                if user_profile:
                    profile_dict = user_profile.dict()
                    profile_dict["group_role"] = "individual"
                    return [profile_dict]
                return []
            
            # Process all group members - NO LIMITS
            all_profiles = []
            for group in user_groups:
                try:
                    group_members = await self.db.get_group_members(group.group_code)
                    print(f"📊 Processing group {group.group_code} with {len(group_members)} members - NO MEMBER LIMITS")
                    
                    for member in group_members:
                        try:
                            member_profile = await self.db.get_user_profile(member.user_id)
                            if member_profile:
                                profile_dict = member_profile.dict()
                                profile_dict["group_role"] = member.role
                                profile_dict["group_code"] = group.group_code
                                all_profiles.append(profile_dict)
                        except Exception as e:
                            logger.warning(f"Failed to get profile for member {member.user_id}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Failed to get members for group {group.group_code}: {e}")
                    continue
            
            return all_profiles
            
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
                    "duration": "",
                    "stops": 0,
                    "departure_time": "",
                    "arrival_time": "",
                    "airline": "",
                }
                
                if itineraries and len(itineraries) > 0:
                    first_itinerary = itineraries[0]
                    segments = first_itinerary.get("segments", [])
                    
                    if segments:
                        first_segment = segments[0]
                        last_segment = segments[-1]
                        
                        enhanced_flight.update({
                            "departure_time": first_segment.get("departure", {}).get("at", ""),
                            "arrival_time": last_segment.get("arrival", {}).get("at", ""),
                            "airline": first_segment.get("carrierCode", ""),
                            "duration": first_itinerary.get("duration", ""),
                            "stops": max(0, len(segments) - 1)
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
                                             profile_summary: str, profile_count: int, 
                                             original_flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ENHANCED: Use AI to filter flights optimized for all users - NO TIMEOUT LIMITS with Tenacity retry strategies"""
        
        try:
            print(f"🧠 AI filtering {len(enhanced_flights)} flights for profile: {profile_summary}")
            
            # Ultra-short prompt for efficiency - works for both individuals and groups
            prompt = f"""Filter flights for traveler profile: {profile_summary}

Profile details: {profile_summary}

Return top 10 flights as JSON: {{"recommended_flights": [{{"id": 1}}, {{"id": 2}}, ...], "filtering_rationale": "brief reason considering the traveler profile and preferences"}}

Flights: {json.dumps(enhanced_flights[:20])}...and {max(0, len(enhanced_flights)-20)} more"""

            # Make optimized OpenAI call - NO TIMEOUT LIMITS
            try:
                response = self.openai_client.responses.create(
                    model="o4-mini",
                    input=prompt
                )
            except Exception as api_error:
                print(f"⚠️ OpenAI API error: {api_error}")
                # Return fallback result
                return {
                    "filtered_results": original_flights[:10],
                    "total_results": len(enhanced_flights),
                    "filtering_applied": False,
                    "reasoning": f"AI filtering unavailable: {str(api_error)}"
                }
            
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
            
            # Check if response is empty
            if not response_text:
                print(f"⚠️ Empty response from OpenAI API")
                return {
                    "filtered_results": original_flights[:10],
                    "total_results": len(enhanced_flights),
                    "filtering_applied": False,
                    "reasoning": "AI filtering returned empty response"
                }

            # Enhanced JSON parsing
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    filtered_result = json.loads(json_text)
                    
                    # Map back to original flights (up to 10)
                    recommended_ids = [f.get("id", 0) for f in filtered_result.get("recommended_flights", [])]
                    original_flights_mapped = [original_flights[i-1] for i in recommended_ids if 1 <= i <= len(original_flights)]
                    
                    # Ensure we return up to 10 results
                    if len(original_flights_mapped) > 10:
                        original_flights_mapped = original_flights_mapped[:10]
                    
                    print(f"✅ AI filtered to {len(original_flights_mapped)} flights")
                    
                    return {
                        "filtered_results": original_flights_mapped,
                        "total_results": len(enhanced_flights),
                        "filtering_applied": True,
                        "reasoning": filtered_result.get("filtering_rationale", "AI filtering applied based on traveler profile")
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
                                            profile_summary: str, profile_count: int,
                                            hotel_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ENHANCED: Use AI to filter hotels optimized for all users - NO TIMEOUT LIMITS with Tenacity retry strategies"""
        
        try:
            print(f"🧠 AI filtering {len(enhanced_hotels)} hotels for profile: {profile_summary}")
            
            # Ultra-short prompt for efficiency - works for both individuals and groups
            prompt = f"""Filter hotels for traveler profile: {profile_summary}

Profile details: {profile_summary}

Return top 10 hotels as JSON: {{"recommended_hotels": [{{"id": 1}}, {{"id": 2}}, ...], "filtering_rationale": "brief reason considering the traveler profile and preferences"}}

Hotels: {json.dumps(enhanced_hotels[:20])}...and {max(0, len(enhanced_hotels)-20)} more"""

            # Make optimized OpenAI call - NO TIMEOUT LIMITS
            try:
                response = self.openai_client.responses.create(
                    model="o4-mini",
                    input=prompt
                )
            except Exception as api_error:
                print(f"⚠️ OpenAI API error: {api_error}")
                # Return fallback result
                return {
                    "filtered_results": hotel_results[:10],
                    "total_results": len(enhanced_hotels),
                    "filtering_applied": False,
                    "reasoning": f"AI filtering unavailable: {str(api_error)}"
                }
            
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
            
            # Check if response is empty
            if not response_text:
                print(f"⚠️ Empty response from OpenAI API")
                return {
                    "filtered_results": hotel_results[:10],
                    "total_results": len(enhanced_hotels),
                    "filtering_applied": False,
                    "reasoning": "AI filtering returned empty response"
                }

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
                        "reasoning": filtered_result.get("filtering_rationale", "AI filtering applied based on traveler profile")
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
    
    def _create_profile_summary(self, group_profiles: List[Dict[str, Any]]) -> str:
        """
        UPDATED: Create a profile summary for both individual travelers and groups
        """
        try:
            if not group_profiles:
                return "unknown traveler profile"
            
            # Check if this is an individual traveler
            if len(group_profiles) == 1:
                profile = group_profiles[0]
                summary_parts = ["1 individual traveler"]
                
                # Add individual preferences
                if profile.get("travel_style"):
                    summary_parts.append(f"style: {profile['travel_style']}")
                
                if profile.get("city"):
                    summary_parts.append(f"from: {profile['city']}")
                
                # Calculate age if available
                if profile.get("birthdate"):
                    try:
                        from datetime import date
                        birth_year = int(profile["birthdate"][:4])
                        current_year = date.today().year
                        age = current_year - birth_year
                        if 0 < age < 120:
                            summary_parts.append(f"age: {age}")
                    except:
                        pass
                
                if profile.get("annual_income"):
                    summary_parts.append(f"income: {profile['annual_income']}")
                
                if profile.get("holiday_preferences"):
                    try:
                        if isinstance(profile["holiday_preferences"], list):
                            preferences = profile["holiday_preferences"][:2]  # First 2
                        else:
                            preferences = str(profile["holiday_preferences"])[:30]  # First 30 chars
                        summary_parts.append(f"prefers: {preferences}")
                    except:
                        pass
                
                return ", ".join(summary_parts)
            
            else:
                # Group travel summary
                summary_parts = [f"{len(group_profiles)} group travelers"]
                
                # Collect group preferences
                styles = [p.get("travel_style") for p in group_profiles if p.get("travel_style")]
                cities = [p.get("city") for p in group_profiles if p.get("city")]
                
                if styles:
                    unique_styles = list(set(styles[:3]))  # Up to 3 unique styles
                    summary_parts.append(f"styles: {unique_styles}")
                
                if cities:
                    unique_cities = list(set(cities[:2]))  # Up to 2 unique cities
                    summary_parts.append(f"from: {unique_cities}")
                
                # Group age range
                ages = []
                for profile in group_profiles:
                    if profile.get("birthdate"):
                        try:
                            from datetime import date
                            birth_year = int(profile["birthdate"][:4])
                            current_year = date.today().year
                            age = current_year - birth_year
                            if 0 < age < 120:
                                ages.append(age)
                        except:
                            pass
                
                if ages:
                    summary_parts.append(f"ages: {min(ages)}-{max(ages)}")
                
                return ", ".join(summary_parts)
                
        except Exception as e:
            logger.warning(f"Error creating profile summary: {e}")
            return f"{len(group_profiles)} travelers" if group_profiles else "unknown travelers"
    
    def _cleanup_cache(self):
        """Clean up old cache entries to prevent memory bloat"""
        try:
            if len(self._filter_cache) > 100:  # Keep cache under 100 entries
                # Remove oldest entries (simple FIFO cleanup)
                keys_to_remove = list(self._filter_cache.keys())[:-50]  # Keep last 50
                for key in keys_to_remove:
                    del self._filter_cache[key]
                print(f"🧹 Cache cleanup: removed {len(keys_to_remove)} old entries")
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")

    # ==================== LEGACY METHODS FOR BACKWARD COMPATIBILITY ====================

    async def filter_flights_enhanced(self, flights: List[Dict[str, Any]], 
                                    user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - redirects to new enhanced filtering"""
        user_id = user_profile.get("user_id", "unknown")
        return await self.filter_flight_results(user_id, flights, {})

    async def filter_hotels_enhanced(self, hotels: List[Dict[str, Any]], 
                                   user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - redirects to new enhanced filtering"""
        user_id = user_profile.get("user_id", "unknown")
        return await self.filter_hotel_results(user_id, hotels, {})
