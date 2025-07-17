import re
import json
import base64
import requests
import email.mime.text
import email.mime.multipart
import os
import logging
from typing import List, Dict, Any, Optional
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import msal
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

# Set up logging
logger = logging.getLogger(__name__)

def is_lambda_environment() -> bool:
    """Detect if running in AWS Lambda environment"""
    return (
        os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None or
        os.environ.get('LAMBDA_RUNTIME_DIR') is not None or
        os.environ.get('AWS_EXECUTION_ENV') is not None
    )

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class GooglePersonName(BaseModel):
    """Pydantic model for Google People API name validation"""
    displayName: Optional[str] = None
    familyName: Optional[str] = None
    givenName: Optional[str] = None

class GooglePersonBirthday(BaseModel):
    """Pydantic model for Google People API birthday validation"""
    date: Optional[Dict[str, int]] = None

class GooglePersonGender(BaseModel):
    """Pydantic model for Google People API gender validation"""
    value: Optional[str] = None

class GooglePersonAddress(BaseModel):
    """Pydantic model for Google People API address validation"""
    value: Optional[str] = None
    type: Optional[str] = None

class GooglePersonResponse(BaseModel):
    """Pydantic model for Google People API response validation"""
    names: Optional[List[GooglePersonName]] = None
    birthdays: Optional[List[GooglePersonBirthday]] = None
    genders: Optional[List[GooglePersonGender]] = None
    addresses: Optional[List[GooglePersonAddress]] = None
    locations: Optional[List[Dict[str, Any]]] = None
    residences: Optional[List[Dict[str, Any]]] = None

class MicrosoftUserResponse(BaseModel):
    """Pydantic model for Microsoft Graph API user response validation"""
    displayName: Optional[str] = None
    mail: Optional[str] = None
    userPrincipalName: Optional[str] = None
    officeLocation: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    id: Optional[str] = None

class EmailScanResult(BaseModel):
    """Pydantic model for email scan result validation"""
    name: Optional[str] = None
    gender: Optional[str] = None
    birthdate: Optional[str] = None
    city: Optional[str] = None
    success: bool
    error: Optional[str] = None

# ==================== CUSTOM RETRY CONDITIONS ====================

def retry_on_google_api_error_result(result):
    """
    Condition 2: Inspect the payload for "error"/"warning" fields, retrying when they're present
    """
    if isinstance(result, dict):
        return (
            result.get("success") is False or
            result.get("error") is not None or
            "error" in result or
            "warning" in result or
            "error" in str(result).lower()
        )
    return False

def retry_on_http_status_error(result):
    """
    Condition 1: Check success flag equivalent to retry on non-2xx/3xx responses
    """
    if isinstance(result, dict):
        # Check for HTTP status codes in error messages or response data
        error_msg = str(result.get("error", "")).lower()
        return any(code in error_msg for code in ['400', '401', '403', '404', '429', '500', '502', '503', '504'])
    return False

def retry_on_api_validation_failure(result):
    """
    Condition 5: Validate against schema/model (Pydantic) and retry when validation fails
    """
    if isinstance(result, dict) and result.get("success"):
        try:
            # Validate email scan result structure
            EmailScanResult(**result)
            return False  # Validation passed
        except ValidationError as e:
            logger.warning(f"Email API response validation failed: {e}")
            return True  # Validation failed, retry
        except Exception as e:
            logger.warning(f"Unexpected validation error: {e}")
            return True
    return False

def retry_on_google_api_exception(exception):
    """Custom exception checker for Google API exceptions"""
    return isinstance(exception, (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
        Exception
    )) and any(keyword in str(exception).lower() for keyword in [
        'google', 'oauth', 'api', 'timeout', 'connection', 'network'
    ])

def retry_on_microsoft_api_exception(exception):
    """Custom exception checker for Microsoft API exceptions"""
    return isinstance(exception, (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
        Exception
    )) and any(keyword in str(exception).lower() for keyword in [
        'microsoft', 'graph', 'oauth', 'api', 'timeout', 'connection', 'network'
    ])

# ==================== COMBINED RETRY STRATEGIES ====================

# Retry strategy for Google API operations
google_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|429|502|503|504|rate.?limit|quota|oauth|google).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_google_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_google_api_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_api_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Retry strategy for Microsoft API operations
microsoft_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|429|502|503|504|rate.?limit|quota|oauth|microsoft|graph).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_microsoft_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_google_api_error_result),  # Same logic applies
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_api_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

class EmailService:
    """Enhanced service for email operations with Lambda-compatible OAuth and Tenacity retry strategies"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.stored_credentials = {}  # Store user credentials by user_id
        self.is_lambda = is_lambda_environment()
        
        print(f"🔧 EmailService initialized - Lambda environment: {self.is_lambda}")
    
    def extract_city_with_ai(self, address_text: str) -> Optional[str]:
        """Use OpenAI to extract city from address text"""
        if not address_text or len(address_text) < 2:
            return None
        
        try:
            print(f"🧠 AI extracting city from: '{address_text}'")
            
            input_prompt = f"""Extract ONLY the city name from this address: "{address_text}"

Rules:
- Return ONLY the city name, nothing else
- Do not include state, country, postal codes, or street details
- For international addresses, return the city in English if possible
- If multiple cities are mentioned, return the main/primary city
- If no clear city can be identified, return "UNKNOWN"

Examples:
- "803/27 King Street, Sydney NSW 2000" → "Sydney"
- "123 Main St, New York, NY 10001" → "New York"
- "45 Avenue des Champs-Élysées, 75008 Paris, France" → "Paris"
- "1-1-1 Shibuya, Shibuya City, Tokyo 150-0002, Japan" → "Tokyo"
- "Unter den Linden 1, 10117 Berlin, Germany" → "Berlin"
- "Microsoft Corporation, Redmond, WA" → "Redmond"

Address: "{address_text}"
City:"""

            response = self.openai_client.responses.create(
                model="o4-mini",
                input=input_prompt
            )
            
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                city = content_item.text.strip()
                                
                                if city and city != "UNKNOWN" and len(city) <= 50 and not any(char.isdigit() for char in city):
                                    city = city.strip('"\'')
                                    print(f"✅ AI extracted city: '{city}' from '{address_text}'")
                                    return city
                                else:
                                    print(f"❌ AI returned invalid city: '{city}' from '{address_text}'")
            
            print(f"❌ AI could not extract city from: '{address_text}'")
            return None
            
        except Exception as e:
            print(f"❌ City extraction failed for '{address_text}': {str(e)}")
            return None
    
    @google_api_retry
    def scan_gmail_profile_server_to_server(self, email: str, user_id: str) -> Dict[str, Any]:
        """
        Server-to-server Gmail profile scanning using service account with Tenacity retry strategies
        NOTE: This requires domain-wide delegation setup for business emails
        """
        try:
            print(f"📧 Attempting server-to-server Gmail profile scan for: {email}")
            
            # Check if we have service account credentials
            service_account_file = os.environ.get('GOOGLE_SERVICE_ACCOUNT_FILE')
            if not service_account_file or not os.path.exists(service_account_file):
                print("❌ No Google service account file found")
                result = {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Service account credentials not configured"
                }
                
                # Validate result before returning
                try:
                    EmailScanResult(**result)
                except ValidationError as ve:
                    logger.warning(f"Gmail scan result validation failed: {ve}")
                    return {
                        "name": None,
                        "gender": None,
                        "birthdate": None,
                        "city": None,
                        "success": False,
                        "error": f"Response validation failed: {str(ve)}"
                    }
                
                return result
            
            # Load service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=[
                    'https://www.googleapis.com/auth/userinfo.profile',
                    'https://www.googleapis.com/auth/userinfo.email',
                    'https://www.googleapis.com/auth/user.birthday.read',
                    'https://www.googleapis.com/auth/user.gender.read',
                    'https://www.googleapis.com/auth/user.addresses.read'
                ]
            )
            
            # Delegate to the user's email
            delegated_credentials = credentials.with_subject(email)
            
            # Build People API client
            people = build("people", "v1", credentials=delegated_credentials)
            
            # Get profile data
            person = people.people().get(
                resourceName="people/me",
                personFields="names,birthdays,genders,addresses,locations,biographies,interests,organizations,residences"
            ).execute()
            
            print(f"👤 Retrieved Google People API data for {email}")
            
            # Validate Google API response
            try:
                GooglePersonResponse(**person)
            except ValidationError as ve:
                logger.warning(f"Google People API response validation failed: {ve}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": f"Google API response validation failed: {str(ve)}"
                }
            
            # Extract profile information
            name = person.get("names", [{}])[0].get("displayName")
            gender = person.get("genders", [{}])[0].get("value")
            b = person.get("birthdays", [{}])[0].get("date", {})
            birthdate = f"{b.get('year','')}-{b.get('month',0):02d}-{b.get('day',0):02d}" if b.get('year') else None
            
            # Extract city using AI
            city = self.extract_city_from_google_data(person)
            
            result = {
                "name": name,
                "gender": gender,
                "birthdate": birthdate,
                "city": city,
                "success": True
            }
            
            # Validate result before returning
            try:
                EmailScanResult(**result)
            except ValidationError as ve:
                logger.warning(f"Gmail scan result validation failed: {ve}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": f"Result validation failed: {str(ve)}"
                }
            
            print(f"✅ Server-to-server Gmail profile extraction result: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Error in server-to-server Gmail scanning: {e}")
            result = {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": str(e)
            }
            
            # Let retry strategy handle this error by returning error result
            return result
    
    @microsoft_api_retry
    def scan_outlook_profile_server_to_server(self, email: str, user_id: str) -> Dict[str, Any]:
        """
        Server-to-server Outlook profile scanning using client credentials with Tenacity retry strategies
        NOTE: This works for organizational emails but has limitations for personal emails
        """
        try:
            print(f"📧 Attempting server-to-server Outlook profile scan for: {email}")
            
            if not config.OUTLOOK_CLIENT_ID or not config.OUTLOOK_CLIENT_SECRET:
                print("❌ Outlook client credentials not configured")
                result = {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Outlook client credentials not configured"
                }
                
                # Validate result before returning
                try:
                    EmailScanResult(**result)
                except ValidationError as ve:
                    logger.warning(f"Outlook scan result validation failed: {ve}")
                    return {
                        "name": None,
                        "gender": None,
                        "birthdate": None,
                        "city": None,
                        "success": False,
                        "error": f"Response validation failed: {str(ve)}"
                    }
                
                return result
            
            # Use client credentials flow for app-only access
            app = msal.ConfidentialClientApplication(
                config.OUTLOOK_CLIENT_ID,
                authority="https://login.microsoftonline.com/common",
                client_credential=config.OUTLOOK_CLIENT_SECRET
            )
            
            # Get app-only token
            token_result = app.acquire_token_for_client(
                scopes=["https://graph.microsoft.com/.default"]
            )
            
            if "access_token" not in token_result:
                error_msg = token_result.get("error_description", token_result.get("error", "Unknown error"))
                print(f"❌ Failed to get app-only token: {error_msg}")
                result = {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": f"App-only authentication failed: {error_msg}"
                }
                
                # This will trigger retry if it's a retryable error
                return result
            
            print("✅ Got app-only token for Microsoft Graph")
            
            # Extract profile data using app-only permissions
            profile_data = self.extract_outlook_profile_data_app_only(token_result['access_token'], email)
            
            result_data = {
                "name": profile_data["name"],
                "gender": profile_data["gender"],
                "birthdate": profile_data["birthdate"],
                "city": profile_data["city"],
                "success": True
            }
            
            # Validate result before returning
            try:
                EmailScanResult(**result_data)
            except ValidationError as ve:
                logger.warning(f"Outlook scan result validation failed: {ve}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": f"Result validation failed: {str(ve)}"
                }
            
            print(f"✅ Server-to-server Outlook profile extraction result: {result_data}")
            return result_data
            
        except Exception as e:
            print(f"❌ Error in server-to-server Outlook scanning: {e}")
            result = {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": str(e)
            }
            
            # Let retry strategy handle this error by returning error result
            return result
    
    @microsoft_api_retry
    def extract_outlook_profile_data_app_only(self, access_token: str, email: str) -> Dict[str, Any]:
        """Extract profile data using app-only Microsoft Graph permissions with Tenacity retry strategies"""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        profile_data = {
            "name": None,
            "gender": None,
            "birthdate": None,
            "city": None
        }
        
        try:
            print(f"🔍 Extracting Outlook profile for {email} using app-only permissions...")
            
            # Try to get user by email (requires User.Read.All permission)
            user_response = requests.get(
                f"https://graph.microsoft.com/v1.0/users/{email}",
                headers=headers,
                timeout=10
            )
            
            # Condition 1: Check HTTP status code
            if not user_response.ok:
                error_msg = f"HTTP {user_response.status_code}: {user_response.text}"
                print(f"❌ Failed to get user data: {error_msg}")
                
                # Return error result to trigger retry on non-2xx status
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "error": error_msg
                }
            
            if user_response.status_code == 200:
                user_data = user_response.json()
                print(f"👤 Found user data for {email}")
                
                # Validate Microsoft API response
                try:
                    MicrosoftUserResponse(**user_data)
                except ValidationError as ve:
                    logger.warning(f"Microsoft Graph API response validation failed: {ve}")
                    return {
                        "name": None,
                        "gender": None,
                        "birthdate": None,
                        "city": None,
                        "error": f"Microsoft API response validation failed: {str(ve)}"
                    }
                
                profile_data["name"] = user_data.get("displayName")
                
                # Try to get city from office location
                office_location = user_data.get("officeLocation")
                if office_location:
                    print(f"🏢 Found office location: '{office_location}'")
                    city = self.extract_city_with_ai(office_location)
                    if city:
                        profile_data["city"] = city
                
                # Check other location fields
                city_field = user_data.get("city")
                if city_field and not profile_data["city"]:
                    print(f"🏙️ Found city field: '{city_field}'")
                    profile_data["city"] = city_field
                    
            else:
                error_msg = f"Failed to get user data: {user_response.status_code} - {user_response.text}"
                print(f"❌ {error_msg}")
                
                # Return error to trigger retry
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "error": error_msg
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error extracting Outlook profile: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Return error to trigger retry
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Error extracting Outlook profile: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Return error to trigger retry
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "error": error_msg
            }
        
        print(f"✅ Final Outlook profile data: {profile_data}")
        return profile_data
    
    def extract_city_from_google_data(self, person: Dict[str, Any]) -> Optional[str]:
        """Extract city from Google People API data using AI"""
        print("🔍 Extracting city from Google People API data...")
        priority_fields = ['residences', 'locations', 'addresses']
        
        for field_name in priority_fields:
            field_data = person.get(field_name, [])
            
            if not field_data:
                continue
                
            print(f"📍 Checking {field_name} field with {len(field_data)} items")
            
            for item in field_data:
                if isinstance(item, dict):
                    if 'value' in item and isinstance(item['value'], str):
                        address_value = item['value']
                        print(f"📍 Found address value: '{address_value}'")
                        city = self.extract_city_with_ai(address_value)
                        if city:
                            return city
                    
                    for key, value in item.items():
                        if key != 'value' and isinstance(value, str) and value:
                            print(f"📍 Found {key}: '{value}'")
                            city = self.extract_city_with_ai(value)
                            if city:
                                return city
        
        print("❌ No city found in Google People API data")
        return None
    
    def scan_gmail_profile(self, email: str, user_id: str) -> Dict[str, Any]:
        """
        Gmail profile scanning with Lambda environment detection
        """
        if self.is_lambda:
            print(f"🚫 Lambda environment detected - attempting server-to-server Gmail scan")
            return self.scan_gmail_profile_server_to_server(email, user_id)
        else:
            print(f"🌐 Local environment detected - server-to-server not recommended for personal emails")
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": "Email scanning not available in current environment. Please provide information manually."
            }
    
    def scan_outlook_profile(self, email: str, user_id: str) -> Dict[str, Any]:
        """
        Outlook profile scanning with Lambda environment detection
        """
        if self.is_lambda:
            print(f"🚫 Lambda environment detected - attempting server-to-server Outlook scan")
            return self.scan_outlook_profile_server_to_server(email, user_id)
        else:
            print(f"🌐 Local environment detected - server-to-server not recommended for personal emails")
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": "Email scanning not available in current environment. Please provide information manually."
            }
    
    def scan_email_for_profile(self, email: str, user_id: str) -> Dict[str, Any]:
        """Main method to scan email for profile information with environment detection"""
        print(f"📧 Starting email profile scan for: {email} (Lambda: {self.is_lambda})")
        
        if email.lower().endswith("@gmail.com"):
            return self.scan_gmail_profile(email, user_id)
        else:
            return self.scan_outlook_profile(email, user_id)
    
    def validate_email_addresses(self, email_string: str) -> List[str]:
        """Extract and validate email addresses from a string"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, email_string)
        return list(set(emails))
    
    def create_invite_email_content(self, sender_name: str, sender_email: str, group_code: str) -> tuple[str, str]:
        """Create the content for the group invite email"""
        subject = f"You're invited to join {sender_name}'s travel group on Arny!"
        
        body = f"""Hi there!

{sender_name} ({sender_email}) has invited you to join their travel group on Arny, the personal travel planning assistant.

Arny helps you and your group plan amazing trips together by:
- Creating personalized travel recommendations
- Coordinating group schedules and preferences  
- Finding the best deals for group bookings
- Managing group itineraries and activities

To join {sender_name}'s travel group:
1. Download the Arny app
2. Create your account 
3. Enter this Group Code during signup: {group_code}

We're excited to help you plan your next adventure together!

Best regards,
The Arny Team

---
This invitation was sent by {sender_name} through the Arny app.
Group Code: {group_code}
"""
        
        return subject, body
    
    async def send_group_invites(self, user_id: str, email_addresses: str, group_code: str, 
                               sender_name: str = "Arny User") -> Dict[str, Any]:
        """Send group invitation emails - currently limited in Lambda environment"""
        
        if self.is_lambda:
            return {
                "success": False,
                "error": "Email sending not available in serverless environment. Please share the group code manually.",
                "group_code": group_code,
                "message": f"Please share this group code with your contacts: {group_code}"
            }
        
        # For local testing or when email service is properly configured
        valid_emails = self.validate_email_addresses(email_addresses)
        
        if not valid_emails:
            return {
                "success": False,
                "error": "No valid email addresses found in the input."
            }
        
        return {
            "success": True,
            "message": f"Email sending would work in production environment. Group code: {group_code}",
            "sent_to": valid_emails,
            "group_code": group_code
        }
