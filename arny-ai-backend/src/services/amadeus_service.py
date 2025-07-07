from typing import List, Dict, Any, Optional
from amadeus import Client, ResponseError
import logging

from ..utils.config import config

class AmadeusService:
    """Service for interacting with Amadeus APIs for flights and hotels"""
    
    def __init__(self):
        self.client = Client(
            client_id=config.AMADEUS_API_KEY,
            client_secret=config.AMADEUS_API_SECRET,
            hostname=config.AMADEUS_BASE_URL
        )
        self.logger = logging.getLogger(__name__)
    
    # ==================== FLIGHT OPERATIONS ====================
    
    async def search_flights(self, origin: str, destination: str, departure_date: str, 
                           return_date: Optional[str] = None, adults: int = 1, 
                           cabin_class: str = "ECONOMY", max_results: int = 10) -> Dict[str, Any]:
        """
        Search for flights using Amadeus Flight Offers Search API
        
        Args:
            origin: Origin airport/city code (e.g., 'SYD')
            destination: Destination airport/city code (e.g., 'LAX')
            departure_date: Departure date in YYYY-MM-DD format
            return_date: Return date in YYYY-MM-DD format (optional for one-way)
            adults: Number of adult passengers
            cabin_class: Cabin class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing flight search results or error
        """
        try:
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
            
            # Make API call
            response = self.client.shopping.flight_offers_search.get(**search_params)
            
            # Process and format results
            flight_offers = []
            if hasattr(response, 'data') and response.data:
                for offer in response.data:
                    flight_offers.append(self._format_flight_offer(offer))
            
            return {
                "success": True,
                "results": flight_offers,
                "meta": {
                    "count": len(flight_offers),
                    "search_params": search_params
                }
            }
            
        except ResponseError as e:
            self.logger.error(f"Amadeus API error in flight search: {e}")
            error_code = self._extract_error_code(e)
            
            return {
                "success": False,
                "error": f"Flight search failed: {str(e)}",
                "error_code": error_code
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in flight search: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    async def get_flight_price(self, flight_offer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get accurate pricing for a specific flight offer using Flight Offers Price API
        
        Args:
            flight_offer: Flight offer object from search results
            
        Returns:
            Dictionary containing accurate pricing or error
        """
        try:
            # The flight_offer should be the complete offer object from search
            response = self.client.shopping.flight_offers.pricing.post(flight_offer)
            
            if hasattr(response, 'data') and response.data:
                # Format the pricing response
                priced_offer = response.data.get('flightOffers', [{}])[0]
                return {
                    "success": True,
                    "priced_offer": self._format_flight_offer(priced_offer),
                    "booking_requirements": response.data.get('bookingRequirements', {}),
                    "pricing_valid_until": response.data.get('expirationDateTime')
                }
            else:
                return {
                    "success": False,
                    "error": "No pricing data returned"
                }
                
        except ResponseError as e:
            self.logger.error(f"Amadeus API error in flight pricing: {e}")
            error_code = self._extract_error_code(e)
            
            return {
                "success": False,
                "error": f"Flight pricing failed: {str(e)}",
                "error_code": error_code
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in flight pricing: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _format_flight_offer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """Format flight offer data for consistent response structure"""
        try:
            # Extract key information from the offer
            price = offer.get('price', {})
            itineraries = offer.get('itineraries', [])
            
            formatted_offer = {
                "id": offer.get('id'),
                "type": offer.get('type'),
                "source": offer.get('source'),
                "instantTicketingRequired": offer.get('instantTicketingRequired', False),
                "nonHomogeneous": offer.get('nonHomogeneous', False),
                "oneWay": offer.get('oneWay', False),
                "lastTicketingDate": offer.get('lastTicketingDate'),
                "numberOfBookableSeats": offer.get('numberOfBookableSeats'),
                "price": {
                    "currency": price.get('currency'),
                    "total": price.get('total'),
                    "base": price.get('base'),
                    "fees": price.get('fees', []),
                    "grandTotal": price.get('grandTotal')
                },
                "pricingOptions": offer.get('pricingOptions', {}),
                "validatingAirlineCodes": offer.get('validatingAirlineCodes', []),
                "travelerPricings": offer.get('travelerPricings', []),
                "itineraries": []
            }
            
            # Format itineraries
            for itinerary in itineraries:
                formatted_itinerary = {
                    "duration": itinerary.get('duration'),
                    "segments": []
                }
                
                for segment in itinerary.get('segments', []):
                    formatted_segment = {
                        "departure": segment.get('departure', {}),
                        "arrival": segment.get('arrival', {}),
                        "carrierCode": segment.get('carrierCode'),
                        "number": segment.get('number'),
                        "aircraft": segment.get('aircraft', {}),
                        "operating": segment.get('operating', {}),
                        "duration": segment.get('duration'),
                        "id": segment.get('id'),
                        "numberOfStops": segment.get('numberOfStops', 0),
                        "blacklistedInEU": segment.get('blacklistedInEU', False)
                    }
                    formatted_itinerary["segments"].append(formatted_segment)
                
                formatted_offer["itineraries"].append(formatted_itinerary)
            
            return formatted_offer
            
        except Exception as e:
            self.logger.error(f"Error formatting flight offer: {e}")
            return offer  # Return original if formatting fails
    
    # ==================== HOTEL OPERATIONS - COMPLETELY FIXED ====================
    
    async def search_hotels(self, city_code: str, check_in_date: str, check_out_date: str,
                          adults: int = 1, rooms: int = 1, max_results: int = 20) -> Dict[str, Any]:
        """
        COMPLETELY FIXED: Search for hotels using two-step approach
        
        Args:
            city_code: City code (e.g., 'NYC' for New York, 'LON' for London)
            check_in_date: Check-in date in YYYY-MM-DD format
            check_out_date: Check-out date in YYYY-MM-DD format
            adults: Number of adults
            rooms: Number of rooms
            max_results: Maximum number of results
            
        Returns:
            Dictionary containing hotel search results or error
        """
        try:
            self.logger.info(f"FIXED: Starting hotel search for city: {city_code}")
            
            # STEP 1: Get hotels by city using reference data API
            try:
                self.logger.info(f"STEP 1: Getting hotel list for city {city_code}")
                hotels_response = self.client.reference_data.locations.hotels.by_city.get(cityCode=city_code)
                
                if not hasattr(hotels_response, 'data') or not hotels_response.data:
                    return {
                        "success": False,
                        "error": f"No hotels found for city code '{city_code}'. Please try a major city like 'NYC', 'LON', 'PAR', or 'SYD'."
                    }
                
                # Get hotel IDs (limit to first 20 for performance)
                hotel_ids = []
                for hotel in hotels_response.data[:20]:
                    hotel_id = hotel.get('hotelId')
                    if hotel_id:
                        hotel_ids.append(hotel_id)
                
                self.logger.info(f"Found {len(hotel_ids)} hotels in {city_code}")
                
                if not hotel_ids:
                    return {
                        "success": False,
                        "error": f"No valid hotels found for '{city_code}'. Please try a different destination."
                    }
                
            except ResponseError as hotels_error:
                self.logger.error(f"Error getting hotel list: {hotels_error}")
                # Fallback: try direct hotel offers search
                return await self._fallback_direct_hotel_search(city_code, check_in_date, check_out_date, adults, rooms, max_results)
            
            # STEP 2: Search for offers using hotel IDs
            try:
                self.logger.info(f"STEP 2: Searching offers for {len(hotel_ids)} hotels")
                
                # Convert hotel IDs to comma-separated string
                hotel_ids_str = ','.join(hotel_ids[:10])  # Limit to 10 hotels for performance
                
                search_params = {
                    'hotelIds': hotel_ids_str,
                    'checkInDate': check_in_date,
                    'checkOutDate': check_out_date,
                    'adults': adults,
                    'roomQuantity': rooms
                }
                
                offers_response = self.client.shopping.hotel_offers_search.get(**search_params)
                
                # Process and format results
                hotel_offers = []
                if hasattr(offers_response, 'data') and offers_response.data:
                    # Limit results to max_results
                    limited_data = offers_response.data[:max_results]
                    for hotel_data in limited_data:
                        hotel_offers.append(self._format_hotel_offer(hotel_data))
                
                self.logger.info(f"Found {len(hotel_offers)} hotel offers for {city_code}")
                
                return {
                    "success": True,
                    "results": hotel_offers,
                    "meta": {
                        "count": len(hotel_offers),
                        "city_code": city_code,
                        "search_params": search_params
                    }
                }
                
            except ResponseError as offers_error:
                self.logger.error(f"Error getting hotel offers: {offers_error}")
                # Try fallback method
                return await self._fallback_direct_hotel_search(city_code, check_in_date, check_out_date, adults, rooms, max_results)
                
        except Exception as e:
            self.logger.error(f"Unexpected error in hotel search: {e}")
            return {
                "success": False,
                "error": f"Unexpected error searching for hotels: {str(e)}"
            }
    
    async def _fallback_direct_hotel_search(self, city_code: str, check_in_date: str, check_out_date: str,
                                          adults: int, rooms: int, max_results: int) -> Dict[str, Any]:
        """
        Fallback method: Try direct hotel offers search by city
        """
        try:
            self.logger.info(f"FALLBACK: Direct hotel search for {city_code}")
            
            search_params = {
                'cityCode': city_code,
                'checkInDate': check_in_date,
                'checkOutDate': check_out_date,
                'adults': adults,
                'roomQuantity': rooms
            }
            
            response = self.client.shopping.hotel_offers_search.get(**search_params)
            
            hotel_offers = []
            if hasattr(response, 'data') and response.data:
                limited_data = response.data[:max_results]
                for hotel_data in limited_data:
                    hotel_offers.append(self._format_hotel_offer(hotel_data))
            
            if hotel_offers:
                self.logger.info(f"FALLBACK SUCCESS: Found {len(hotel_offers)} hotels")
                return {
                    "success": True,
                    "results": hotel_offers,
                    "meta": {
                        "count": len(hotel_offers),
                        "search_params": search_params,
                        "method": "fallback_direct"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Could not find hotels for city '{city_code}'. Please try a major city name or different destination."
                }
                
        except ResponseError as e:
            self.logger.error(f"Fallback hotel search also failed: {e}")
            error_code = self._extract_error_code(e)
            
            # Provide user-friendly error messages
            if "400" in error_code or "invalid" in str(e).lower():
                return {
                    "success": False,
                    "error": f"Invalid destination '{city_code}'. Please try a major city like 'New York', 'London', 'Paris', or 'Sydney'."
                }
            elif "404" in error_code or "not found" in str(e).lower():
                return {
                    "success": False,
                    "error": f"No hotels found for '{city_code}'. Please try a major city like 'New York', 'London', 'Paris', or 'Sydney'."
                }
            else:
                return {
                    "success": False,
                    "error": f"Could not find hotels for city '{city_code}'. Please try a different destination."
                }
    
    async def get_hotel_offers(self, hotel_id: str, check_in_date: str, check_out_date: str,
                             adults: int = 1, rooms: int = 1) -> Dict[str, Any]:
        """
        Get specific hotel offers for a hotel using Hotel Offers API
        
        Args:
            hotel_id: Amadeus hotel ID
            check_in_date: Check-in date in YYYY-MM-DD format
            check_out_date: Check-out date in YYYY-MM-DD format
            adults: Number of adults
            rooms: Number of rooms
            
        Returns:
            Dictionary containing hotel offers or error
        """
        try:
            response = self.client.shopping.hotel_offers_search.get(
                hotelIds=hotel_id,
                checkInDate=check_in_date,
                checkOutDate=check_out_date,
                adults=adults,
                roomQuantity=rooms
            )
            
            if hasattr(response, 'data') and response.data:
                hotel_offers = [self._format_hotel_offer(hotel) for hotel in response.data]
                return {
                    "success": True,
                    "offers": hotel_offers
                }
            else:
                return {
                    "success": False,
                    "error": "No offers found for this hotel"
                }
                
        except ResponseError as e:
            self.logger.error(f"Amadeus API error in hotel offers: {e}")
            error_code = self._extract_error_code(e)
            
            return {
                "success": False,
                "error": f"Hotel offers search failed: {str(e)}",
                "error_code": error_code
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in hotel offers: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _format_hotel_offer(self, hotel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format hotel offer data for consistent response structure"""
        try:
            hotel = hotel_data.get('hotel', {})
            offers = hotel_data.get('offers', [])
            
            formatted_hotel = {
                "type": hotel_data.get('type'),
                "hotel": {
                    "chainCode": hotel.get('chainCode'),
                    "iataCode": hotel.get('iataCode'),
                    "dupeId": hotel.get('dupeId'),
                    "name": hotel.get('name'),
                    "hotelId": hotel.get('hotelId'),
                    "geoCode": hotel.get('geoCode', {}),
                    "address": hotel.get('address', {}),
                    "contact": hotel.get('contact', {}),
                    "amenities": hotel.get('amenities', []),
                    "rating": hotel.get('rating'),
                    "description": hotel.get('description', {})
                },
                "available": hotel_data.get('available', True),
                "offers": []
            }
            
            # Format offers
            for offer in offers:
                formatted_offer = {
                    "id": offer.get('id'),
                    "checkInDate": offer.get('checkInDate'),
                    "checkOutDate": offer.get('checkOutDate'),
                    "rateCode": offer.get('rateCode'),
                    "rateFamilyEstimated": offer.get('rateFamilyEstimated', {}),
                    "category": offer.get('category'),
                    "description": offer.get('description', {}),
                    "commission": offer.get('commission', {}),
                    "boardType": offer.get('boardType'),
                    "room": offer.get('room', {}),
                    "guests": offer.get('guests', {}),
                    "price": offer.get('price', {}),
                    "policies": offer.get('policies', {}),
                    "self": offer.get('self')
                }
                formatted_hotel["offers"].append(formatted_offer)
            
            return formatted_hotel
            
        except Exception as e:
            self.logger.error(f"Error formatting hotel offer: {e}")
            return hotel_data  # Return original if formatting fails
    
    def _extract_error_code(self, error: ResponseError) -> str:
        """Extract error code from Amadeus ResponseError"""
        error_code = "unknown"
        try:
            if hasattr(error, 'response') and error.response:
                if hasattr(error.response, 'status_code'):
                    error_code = str(error.response.status_code)
                elif hasattr(error.response, 'status'):
                    error_code = str(error.response.status)
                elif isinstance(error.response, dict) and 'status' in error.response:
                    error_code = str(error.response['status'])
        except Exception:
            error_code = "unknown"
        return error_code
    
    # ==================== AIRPORT/LOCATION SEARCH ====================
    
    async def search_airports(self, keyword: str, subtype: str = "AIRPORT") -> List[Dict[str, Any]]:
        """
        Search for airports/locations by keyword
        
        Args:
            keyword: Search keyword (city name, airport code, etc.)
            subtype: Type of location (AIRPORT, CITY, etc.)
            
        Returns:
            List of matching airports/locations
        """
        try:
            response = self.client.reference_data.locations.get(
                keyword=keyword,
                subType=subtype
            )
            
            if hasattr(response, 'data') and response.data:
                return [location for location in response.data]
            else:
                return []
                
        except ResponseError as e:
            self.logger.error(f"Error searching airports: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error searching airports: {e}")
            return []
    
    async def get_flight_checkin_links(self, airline_code: str) -> List[Dict[str, Any]]:
        """
        Get check-in links for an airline
        
        Args:
            airline_code: IATA airline code (e.g., 'CA' for Air China)
            
        Returns:
            List of check-in links for the airline
        """
        try:
            response = self.client.reference_data.urls.checkin_links.get(airlineCode=airline_code)
            
            if hasattr(response, 'data') and response.data:
                return [link for link in response.data]
            else:
                return []
                
        except ResponseError as e:
            self.logger.error(f"Error getting check-in links: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting check-in links: {e}")
            return []
