"""
Enhanced Amadeus Service with Comprehensive Tenacity Retry Strategies

This module provides enhanced service for interacting with Amadeus APIs for flights and hotels
with comprehensive retry strategies using Tenacity library following 5 key conditions:

1. Check success flag (response.ok or equivalent) to retry on any non-2xx/3xx response
2. Inspect payload for "error"/"warning" fields, retrying when they're present  
3. Match on exception messages via retry_if_exception_message (timeout, failed, unavailable)
4. Combine predicates with retry_any to catch both exceptions and bad results
5. Validate against schema/model (Pydantic) and retry when validation fails
"""

from typing import List, Dict, Any, Optional
from amadeus import Client, ResponseError
import logging
import requests
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

# Set up logging
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class AmadeusFlightOffer(BaseModel):
    """Pydantic model for flight offer validation"""
    id: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    price: Optional[Dict[str, Any]] = None
    itineraries: Optional[List[Dict[str, Any]]] = None
    validatingAirlineCodes: Optional[List[str]] = None

class AmadeusHotelOffer(BaseModel):
    """Pydantic model for hotel offer validation"""
    type: Optional[str] = None
    hotel: Optional[Dict[str, Any]] = None
    available: Optional[bool] = None
    offers: Optional[List[Dict[str, Any]]] = None

class AmadeusSearchResponse(BaseModel):
    """Pydantic model for search response validation"""
    success: bool
    results: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

# ==================== CUSTOM RETRY CONDITIONS ====================

def retry_on_amadeus_error_result(result):
    """
    Condition 2: Inspect the payload for "error"/"warning" fields, retrying when they're present
    """
    if isinstance(result, dict):
        return (
            result.get("success") is False or
            result.get("error") is not None or
            result.get("error_code") is not None or
            "warning" in result or
            "error" in str(result).lower()
        )
    return False

def retry_on_amadeus_response_validation_failure(result):
    """
    Condition 5: Validate against schema/model (Pydantic) and retry when validation fails
    """
    if isinstance(result, dict) and result.get("success"):
        try:
            # Validate the response structure
            AmadeusSearchResponse(**result)
            
            # If it has results, validate individual items
            if result.get("results"):
                for item in result["results"][:1]:  # Validate first item as sample
                    if "itineraries" in item:  # Flight offer
                        AmadeusFlightOffer(**item)
                    elif "hotel" in item:  # Hotel offer
                        AmadeusHotelOffer(**item)
            return False  # Validation passed
        except ValidationError as e:
            logger.warning(f"Amadeus response validation failed: {e}")
            return True  # Validation failed, retry
        except Exception as e:
            logger.warning(f"Unexpected validation error: {e}")
            return True
    return False

def retry_on_amadeus_http_status(result):
    """
    Condition 1: Check success flag equivalent to retry on non-2xx/3xx responses
    """
    if isinstance(result, dict):
        error_code = result.get("error_code")
        if error_code:
            try:
                status_code = int(error_code)
                # Retry on 4xx and 5xx status codes, but not on 2xx/3xx
                return status_code >= 400
            except (ValueError, TypeError):
                # If error_code is not numeric, check if it indicates HTTP error
                return any(code in str(error_code).lower() for code in ['400', '401', '403', '404', '429', '500', '502', '503', '504'])
    return False

def retry_on_amadeus_exception_type(exception):
    """
    Custom exception checker for Amadeus-specific exceptions
    """
    return isinstance(exception, (ResponseError, requests.exceptions.RequestException, ConnectionError, TimeoutError))

# ==================== COMBINED RETRY STRATEGIES ====================

# Primary retry strategy for critical Amadeus operations (flights/hotels)
amadeus_critical_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|429|502|503|504|rate.?limit).*"),
        # Condition 4: Exception types (combining ResponseError and connection issues)
        retry_if_exception_type((ResponseError, requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        # Custom exception checker
        retry_if_exception(retry_on_amadeus_exception_type),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_amadeus_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_amadeus_http_status),
        # Condition 5: Validation failure
        retry_if_result(retry_on_amadeus_response_validation_failure)
    ),
    stop=stop_after_attempt(4),  # More attempts for critical operations
    wait=wait_exponential(multiplier=1.5, min=1, max=15),  # Longer max wait
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Secondary retry strategy for less critical operations (airport search, check-in links)
amadeus_secondary_retry = retry(
    retry=retry_any(
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|429|502|503|504).*"),
        retry_if_exception_type((ResponseError, requests.exceptions.RequestException, ConnectionError, TimeoutError)),
        retry_if_result(retry_on_amadeus_error_result),
        retry_if_result(retry_on_amadeus_http_status)
    ),
    stop=stop_after_attempt(2),  # Fewer attempts for secondary operations
    wait=wait_exponential(multiplier=1, min=0.5, max=5),  # Shorter wait times
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# ==================== AMADEUS SERVICE CLASS ====================

class AmadeusService:
    """
    Enhanced Amadeus API service with comprehensive retry strategies
    
    Implements robust error handling and retry mechanisms for all Amadeus API operations
    following comprehensive retry strategy principles.
    """
    
    def __init__(self):
        """Initialize Amadeus client with credentials"""
        self.client = Client(
            client_id=config.AMADEUS_CLIENT_ID,
            client_secret=config.AMADEUS_CLIENT_SECRET,
            hostname='test'  # Use 'production' when ready for production
        )
        logger.info("🔧 AmadeusService initialized with retry strategies")
    
    # ==================== HELPER METHODS ====================
    
    def _format_flight_offer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """Format flight offer for consistent response structure"""
        try:
            formatted_offer = {
                "id": offer.get("id"),
                "type": offer.get("type"),
                "source": offer.get("source"),
                "instantTicketingRequired": offer.get("instantTicketingRequired"),
                "nonHomogeneous": offer.get("nonHomogeneous"),
                "oneWay": offer.get("oneWay"),
                "lastTicketingDate": offer.get("lastTicketingDate"),
                "numberOfBookableSeats": offer.get("numberOfBookableSeats"),
                "price": offer.get("price", {}),
                "pricingOptions": offer.get("pricingOptions", {}),
                "validatingAirlineCodes": offer.get("validatingAirlineCodes", []),
                "travelerPricings": offer.get("travelerPricings", []),
                "itineraries": []
            }
            
            # Format itineraries
            for itinerary in offer.get("itineraries", []):
                formatted_itinerary = {
                    "duration": itinerary.get("duration"),
                    "segments": []
                }
                
                for segment in itinerary.get("segments", []):
                    formatted_segment = {
                        "departure": segment.get("departure", {}),
                        "arrival": segment.get("arrival", {}),
                        "carrierCode": segment.get("carrierCode"),
                        "number": segment.get("number"),
                        "aircraft": segment.get("aircraft", {}),
                        "operating": segment.get("operating", {}),
                        "duration": segment.get("duration"),
                        "id": segment.get("id"),
                        "numberOfStops": segment.get("numberOfStops", 0),
                        "blacklistedInEU": segment.get("blacklistedInEU", False)
                    }
                    formatted_itinerary["segments"].append(formatted_segment)
                
                formatted_offer["itineraries"].append(formatted_itinerary)
            
            return formatted_offer
            
        except Exception as e:
            logger.warning(f"Error formatting flight offer: {e}")
            return offer  # Return original if formatting fails
    
    def _format_hotel_offer(self, hotel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format hotel offer for consistent response structure"""
        try:
            formatted_hotel = {
                "type": hotel_data.get("type"),
                "hotel": hotel_data.get("hotel", {}),
                "available": hotel_data.get("available"),
                "offers": [],
                "self": hotel_data.get("self")
            }
            
            # Format offers
            for offer in hotel_data.get("offers", []):
                formatted_offer = {
                    "id": offer.get("id"),
                    "checkInDate": offer.get("checkInDate"),
                    "checkOutDate": offer.get("checkOutDate"),
                    "rateCode": offer.get("rateCode"),
                    "rateFamilyEstimated": offer.get("rateFamilyEstimated", {}),
                    "room": offer.get("room", {}),
                    "guests": offer.get("guests", {}),
                    "price": offer.get("price", {}),
                    "policies": offer.get("policies", {}),
                    "self": offer.get("self")
                }
                formatted_hotel["offers"].append(formatted_offer)
            
            return formatted_hotel
            
        except Exception as e:
            logger.warning(f"Error formatting hotel offer: {e}")
            return hotel_data  # Return original if formatting fails
    
    # ==================== FLIGHT OPERATIONS WITH RETRY STRATEGIES ====================
    
    @amadeus_critical_retry
    async def search_flights(self, origin: str, destination: str, departure_date: str,
                           return_date: Optional[str] = None, adults: int = 1, 
                           cabin_class: str = "ECONOMY", max_results: int = 50) -> Dict[str, Any]:
        """
        Search for flights using Amadeus Flight Offers Search API with comprehensive retry strategies
        
        Implements all 5 retry strategy conditions:
        1. HTTP status checking
        2. Error/warning field inspection  
        3. Exception message matching
        4. Combined exception and result predicates
        5. Pydantic validation with retry on failure
        """
        try:
            logger.info(f"Searching flights: {origin} -> {destination} on {departure_date} (with retry strategies)")
            
            # Prepare search parameters
            search_params = {
                'originLocationCode': origin,
                'destinationLocationCode': destination,
                'departureDate': departure_date,
                'adults': adults,
                'max': max_results,
                'travelClass': cabin_class
            }
            
            # Add return date if provided (for round-trip)
            if return_date:
                search_params['returnDate'] = return_date
            
            # Make API call with automatic retry on failures
            response = self.client.shopping.flight_offers_search.get(**search_params)
            
            # Process and format results
            flight_offers = []
            if hasattr(response, 'data') and response.data:
                for offer in response.data:
                    formatted_offer = self._format_flight_offer(offer)
                    flight_offers.append(formatted_offer)
            
            # Create response that will be validated by retry strategy
            result = {
                "success": True,
                "results": flight_offers,
                "meta": {
                    "count": len(flight_offers),
                    "search_params": search_params
                }
            }
            
            # Validate response structure (triggers retry if validation fails)
            try:
                AmadeusSearchResponse(**result)
                logger.info(f"Flight search successful: {len(flight_offers)} results found")
            except ValidationError as ve:
                logger.warning(f"Flight search response validation failed: {ve}")
                # Return error result to trigger retry
                return {
                    "success": False,
                    "error": f"Response validation failed: {str(ve)}",
                    "results": []
                }
            
            return result
            
        except ResponseError as e:
            error_msg = f"Amadeus flight search API error: {str(e)}"
            logger.error(error_msg)
            
            # Extract status code for retry decision
            status_code = getattr(e, 'response', {}).get('status', 'unknown')
            
            result = {
                "success": False,
                "error": error_msg,
                "error_code": str(status_code),
                "results": []
            }
            
            # Let retry strategy handle this error
            return result
            
        except Exception as e:
            error_msg = f"Unexpected flight search error: {str(e)}"
            logger.error(error_msg)
            
            result = {
                "success": False,
                "error": error_msg,
                "results": []
            }
            
            return result

    @amadeus_critical_retry
    async def get_flight_price(self, flight_offer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get accurate pricing for a specific flight offer with retry strategies
        """
        try:
            logger.info("Getting flight pricing with retry strategies")
            
            # The flight_offer should be the complete offer object from search
            response = self.client.shopping.flight_offers.pricing.post(flight_offer)
            
            if hasattr(response, 'data') and response.data:
                # Format the pricing response
                priced_offer = response.data.get('flightOffers', [{}])[0]
                
                result = {
                    "success": True,
                    "priced_offer": self._format_flight_offer(priced_offer),
                    "booking_requirements": response.data.get('bookingRequirements', {}),
                    "pricing_valid_until": response.data.get('expirationDateTime')
                }
                
                # Validate response
                try:
                    if result["priced_offer"]:
                        AmadeusFlightOffer(**result["priced_offer"])
                    logger.info("Flight pricing successful")
                except ValidationError as ve:
                    logger.warning(f"Flight pricing validation failed: {ve}")
                    return {
                        "success": False,
                        "error": f"Pricing validation failed: {str(ve)}"
                    }
                
                return result
            else:
                return {
                    "success": False,
                    "error": "No pricing data returned"
                }
                
        except ResponseError as e:
            error_msg = f"Amadeus flight pricing API error: {str(e)}"
            logger.error(error_msg)
            
            # Properly extract status code from Response object
            status_code = 'unknown'
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = str(e.response.status_code)
            
            return {
                "success": False,
                "error": error_msg,
                "error_code": status_code
            }
            
        except Exception as e:
            logger.error(f"Unexpected flight pricing error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    # ==================== HOTEL OPERATIONS WITH RETRY STRATEGIES ====================

    @amadeus_critical_retry
    async def search_hotels(self, city_code: str, check_in_date: str, check_out_date: str,
                          adults: int = 1, rooms: int = 1, max_results: int = 50) -> Dict[str, Any]:
        """
        Search for hotels using Amadeus Hotel Search API with comprehensive retry strategies
        """
        try:
            logger.info(f"Searching hotels in {city_code} with retry strategies")
            
            # First, get hotel list by city
            hotels_response = self.client.reference_data.locations.hotels.by_city.get(
                cityCode=city_code
            )
            
            if not hasattr(hotels_response, 'data') or not hotels_response.data:
                return {
                    "success": False,
                    "error": f"No hotels found for city code: {city_code}",
                    "results": []
                }
            
            # Extract hotel IDs (limit to max_results for offers search)
            hotel_ids = [hotel['hotelId'] for hotel in hotels_response.data[:max_results]]
            
            # Search for hotel offers
            search_params = {
                'hotelIds': ','.join(hotel_ids),
                'checkInDate': check_in_date,
                'checkOutDate': check_out_date,
                'adults': adults,
                'roomQuantity': rooms
            }
            
            offers_response = self.client.shopping.hotel_offers_search.get(**search_params)
            
            # Process and format results
            hotel_offers = []
            if hasattr(offers_response, 'data') and offers_response.data:
                for hotel_data in offers_response.data:
                    formatted_offer = self._format_hotel_offer(hotel_data)
                    hotel_offers.append(formatted_offer)
            
            result = {
                "success": True,
                "results": hotel_offers,
                "meta": {
                    "count": len(hotel_offers),
                    "search_params": search_params
                }
            }
            
            # Validate response structure
            try:
                AmadeusSearchResponse(**result)
                # Validate sample hotel offers
                if hotel_offers:
                    AmadeusHotelOffer(**hotel_offers[0])
                logger.info(f"Hotel search successful: {len(hotel_offers)} results found")
            except ValidationError as ve:
                logger.warning(f"Hotel search response validation failed: {ve}")
                return {
                    "success": False,
                    "error": f"Response validation failed: {str(ve)}",
                    "results": []
                }
            
            return result
            
        except ResponseError as e:
            error_msg = f"Amadeus hotel search API error: {str(e)}"
            logger.error(error_msg)
            
            # Properly extract status code from Response object
            status_code = 'unknown'
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = str(e.response.status_code)
            
            return {
                "success": False,
                "error": error_msg,
                "error_code": status_code,
                "results": []
            }
            
        except Exception as e:
            error_msg = f"Unexpected hotel search error: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "results": []
            }

    @amadeus_critical_retry
    async def get_hotel_offers(self, hotel_id: str, check_in_date: str, check_out_date: str,
                             adults: int = 1, rooms: int = 1) -> Dict[str, Any]:
        """
        Get specific hotel offers for a hotel with retry strategies
        """
        try:
            logger.info(f"Getting hotel offers for {hotel_id} with retry strategies")
            
            response = self.client.shopping.hotel_offers_search.get(
                hotelIds=hotel_id,
                checkInDate=check_in_date,
                checkOutDate=check_out_date,
                adults=adults,
                roomQuantity=rooms
            )
            
            if hasattr(response, 'data') and response.data:
                hotel_offers = [self._format_hotel_offer(hotel) for hotel in response.data]
                
                result = {
                    "success": True,
                    "offers": hotel_offers
                }
                
                # Validate response
                try:
                    if hotel_offers:
                        AmadeusHotelOffer(**hotel_offers[0])
                    logger.info(f"Hotel offers retrieved successfully: {len(hotel_offers)} offers")
                except ValidationError as ve:
                    logger.warning(f"Hotel offers validation failed: {ve}")
                    return {
                        "success": False,
                        "error": f"Offers validation failed: {str(ve)}"
                    }
                
                return result
            else:
                return {
                    "success": False,
                    "error": "No offers found for this hotel"
                }
                
        except ResponseError as e:
            error_msg = f"Amadeus hotel offers API error: {str(e)}"
            logger.error(error_msg)
            
            # Properly extract status code from Response object
            status_code = 'unknown'
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = str(e.response.status_code)
            
            return {
                "success": False,
                "error": error_msg,
                "error_code": status_code
            }
            
        except Exception as e:
            logger.error(f"Unexpected hotel offers error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    # ==================== SECONDARY OPERATIONS WITH RETRY STRATEGIES ====================

    @amadeus_secondary_retry
    async def search_airports(self, keyword: str, subtype: str = "AIRPORT") -> List[Dict[str, Any]]:
        """
        Search for airports using Amadeus Airport & City Search API with retry strategies
        """
        try:
            logger.info(f"Searching airports for '{keyword}' with retry strategies")
            
            response = self.client.reference_data.locations.get(
                keyword=keyword,
                subType=subtype
            )
            
            airports = []
            if hasattr(response, 'data') and response.data:
                for location in response.data:
                    airport_info = {
                        "iataCode": location.get("iataCode"),
                        "name": location.get("name"),
                        "address": location.get("address", {}),
                        "geoCode": location.get("geoCode", {}),
                        "timeZoneOffset": location.get("timeZoneOffset"),
                        "subType": location.get("subType")
                    }
                    airports.append(airport_info)
                
                logger.info(f"Found {len(airports)} airports for '{keyword}'")
            
            return airports
            
        except ResponseError as e:
            logger.error(f"Amadeus airport search API error: {str(e)}")
            return []
            
        except Exception as e:
            logger.error(f"Unexpected airport search error: {str(e)}")
            return []

    @amadeus_secondary_retry
    async def get_flight_status(self, flight_number: str, scheduled_departure_date: str) -> Dict[str, Any]:
        """
        Get real-time flight status with retry strategies
        """
        try:
            logger.info(f"Getting flight status for {flight_number} on {scheduled_departure_date}")
            
            response = self.client.schedule.flights.get(
                carrierCode=flight_number[:2],  # First 2 characters are airline code
                flightNumber=flight_number[2:],  # Rest is flight number
                scheduledDepartureDate=scheduled_departure_date
            )
            
            if hasattr(response, 'data') and response.data:
                flight_status = response.data[0]  # Get first result
                
                status_info = {
                    "flight_number": flight_number,
                    "scheduled_departure": flight_status.get("flightDesignator", {}).get("scheduledDepartureDate"),
                    "actual_departure": flight_status.get("flightDesignator", {}).get("actualDepartureDate"),
                    "scheduled_arrival": flight_status.get("flightDesignator", {}).get("scheduledArrivalDate"),
                    "actual_arrival": flight_status.get("flightDesignator", {}).get("actualArrivalDate"),
                    "status": flight_status.get("flightStatusType"),
                    "departure_terminal": flight_status.get("flightPoints", [{}])[0].get("departure", {}).get("terminal"),
                    "arrival_terminal": flight_status.get("flightPoints", [{}])[0].get("arrival", {}).get("terminal"),
                    "gate": flight_status.get("flightPoints", [{}])[0].get("departure", {}).get("gate")
                }
                
                logger.info(f"Flight status retrieved: {status_info.get('status')}")
                return {
                    "success": True,
                    "flight_status": status_info
                }
            else:
                return {
                    "success": False,
                    "error": "No flight status found"
                }
                
        except ResponseError as e:
            logger.error(f"Amadeus flight status API error: {str(e)}")
            return {
                "success": False,
                "error": f"Flight status API error: {str(e)}"
            }
            
        except Exception as e:
            logger.error(f"Unexpected flight status error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    @amadeus_secondary_retry
    async def get_airline_check_in_links(self, airline_code: str) -> Dict[str, Any]:
        """
        Get airline check-in links with retry strategies
        """
        try:
            logger.info(f"Getting check-in links for airline: {airline_code}")
            
            response = self.client.reference_data.urls.checkin_links.get(
                airlineCode=airline_code
            )
            
            if hasattr(response, 'data') and response.data:
                checkin_info = response.data[0]
                
                links_info = {
                    "airline_code": airline_code,
                    "airline_name": checkin_info.get("type"),
                    "checkin_url": checkin_info.get("href"),
                    "channels": checkin_info.get("channel", [])
                }
                
                logger.info(f"Check-in links retrieved for {airline_code}")
                return {
                    "success": True,
                    "checkin_links": links_info
                }
            else:
                return {
                    "success": False,
                    "error": "No check-in links found"
                }
                
        except ResponseError as e:
            logger.error(f"Amadeus check-in links API error: {str(e)}")
            return {
                "success": False,
                "error": f"Check-in links API error: {str(e)}"
            }
            
        except Exception as e:
            logger.error(f"Unexpected check-in links error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
