import re
import json
import base64
import requests
import email.mime.text
import email.mime.multipart
import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any, Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import msal
from openai import OpenAI

from ..utils.config import config

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callbacks"""
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/google/callback":
            self.send_response(404)
            self.end_headers()
            return

        qs = urllib.parse.parse_qs(parsed.query)
        code = qs.get("code", [None])[0]
        if not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing code")
            return

        # Exchange code for tokens
        token_res = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": config.GOOGLE_CLIENT_ID,
                "client_secret": config.GOOGLE_CLIENT_SECRET,
                "redirect_uri": config.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        tok = token_res.json()
        if "access_token" not in tok:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Token exchange failed: {tok}".encode())
            return

        # Decode ID token for email
        id_token = tok.get("id_token", "")
        parts = id_token.split(".")
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))

        # Store tokens and claims for retrieval
        self.server.tok = tok
        self.server.claims = claims

        # Return success response
        message = {"message": f"Email {claims.get('email')} linked successfully! You can close this window."}
        body = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Suppress HTTP server logs"""
        pass

class EmailService:
    """Enhanced service for email operations including OAuth and profile extraction"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.stored_credentials = {}  # Store user credentials by user_id
    
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
        
        # Check other fields that might contain location info
        other_fields = ['biographies', 'interests', 'organizations']
        for field_name in other_fields:
            field_data = person.get(field_name, [])
            if field_data:
                print(f"📍 Checking {field_name} field with {len(field_data)} items")
                for item in field_data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str) and value and len(value) > 5:
                                if any(word in value.lower() for word in ['street', 'avenue', 'road', 'city', 'town', 'address']):
                                    print(f"📍 Found potential address in {key}: '{value}'")
                                    city = self.extract_city_with_ai(value)
                                    if city:
                                        return city
        
        print("❌ No city found in Google People API data")
        return None
    
    def scan_gmail_profile(self, email: str, user_id: str) -> Dict[str, Any]:
        """Scan Gmail for profile information using OAuth"""
        try:
            print(f"📧 Starting Gmail profile scan for: {email}")
            
            # Define comprehensive scopes
            scopes = [
                "openid", 
                "email", 
                "profile",
                "https://www.googleapis.com/auth/user.birthday.read",
                "https://www.googleapis.com/auth/user.gender.read",
                "https://www.googleapis.com/auth/user.addresses.read",
                "https://www.googleapis.com/auth/userinfo.profile",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/user.organization.read",
                "https://www.googleapis.com/auth/gmail.send"
            ]
            
            # Build OAuth URL
            auth_url = (
                "https://accounts.google.com/o/oauth2/v2/auth"
                f"?client_id={config.GOOGLE_CLIENT_ID}"
                f"&redirect_uri={urllib.parse.quote(config.GOOGLE_REDIRECT_URI, safe='')}"
                "&response_type=code"
                "&access_type=offline"
                "&prompt=consent"
                f"&scope={'%20'.join(urllib.parse.quote(s) for s in scopes)}"
            )
            
            print(f"🌐 Opening OAuth URL: {auth_url}")
            webbrowser.open(auth_url)

            # Wait for callback
            server = HTTPServer(("localhost", 8000), OAuthCallbackHandler)
            server.handle_request()
            
            tok = server.tok
            claims = server.claims

            # Build credentials
            creds = Credentials(
                tok["access_token"],
                refresh_token=tok.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=config.GOOGLE_CLIENT_ID,
                client_secret=config.GOOGLE_CLIENT_SECRET,
                scopes=scopes
            )
            
            # Store credentials for later use
            self.stored_credentials[user_id] = {
                "email": email,
                "credentials": creds,
                "type": "gmail"
            }
            
            # Get profile data from People API
            people = build("people", "v1", credentials=creds)
            person = people.people().get(
                resourceName="people/me",
                personFields="names,birthdays,genders,addresses,locations,biographies,interests,organizations,residences"
            ).execute()

            print(f"👤 Retrieved Google People API data: {json.dumps(person, indent=2)[:500]}...")

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
            
            print(f"✅ Gmail profile extraction result: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Error scanning Gmail profile: {e}")
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": str(e)
            }
    
    def extract_outlook_profile_data(self, access_token: str) -> Dict[str, Any]:
        """Extract profile data from Microsoft Graph API"""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        profile_data = {
            "name": None,
            "gender": None,
            "birthdate": None,
            "city": None
        }
        
        try:
            print("🔍 Starting Outlook profile data extraction...")
            
            # Get basic user profile
            print("📞 Calling Microsoft Graph /v1.0/me API...")
            user_response = requests.get(
                "https://graph.microsoft.com/v1.0/me",
                headers=headers
            )
            
            if user_response.status_code == 200:
                user_data = user_response.json()
                print(f"👤 Basic user data: {json.dumps(user_data, indent=2)}")
                
                profile_data["name"] = user_data.get("displayName")
                print(f"📝 Found name: {profile_data['name']}")
                
                # Try to get city from office location
                office_location = user_data.get("officeLocation")
                if office_location:
                    print(f"🏢 Found office location: '{office_location}'")
                    city = self.extract_city_with_ai(office_location)
                    if city:
                        profile_data["city"] = city
                
                # Check other location fields
                street_address = user_data.get("streetAddress")
                if street_address and not profile_data["city"]:
                    print(f"🏠 Found street address: '{street_address}'")
                    city = self.extract_city_with_ai(street_address)
                    if city:
                        profile_data["city"] = city
                
                city_field = user_data.get("city")
                if city_field and not profile_data["city"]:
                    print(f"🏙️ Found city field: '{city_field}'")
                    profile_data["city"] = city_field
                    
                country = user_data.get("country")
                if country and not profile_data["city"]:
                    print(f"🌍 Found country: '{country}'")
                    city = self.extract_city_with_ai(country)
                    if city:
                        profile_data["city"] = city
            else:
                print(f"❌ Failed to get basic user profile: {user_response.status_code} - {user_response.text}")

            # Try extended profile information
            try:
                print("📞 Calling Microsoft Graph /v1.0/me/profile API...")
                profile_response = requests.get(
                    "https://graph.microsoft.com/v1.0/me/profile",
                    headers=headers
                )
                
                if profile_response.status_code == 200:
                    profile_info = profile_response.json()
                    print(f"👤 Extended profile data: {json.dumps(profile_info, indent=2)}")
                    
                    if isinstance(profile_info, dict):
                        for date_field in ['birthdate', 'birthday', 'dateOfBirth']:
                            if profile_info.get(date_field) and not profile_data["birthdate"]:
                                profile_data["birthdate"] = profile_info.get(date_field)
                                print(f"📅 Found birthdate: {profile_data['birthdate']}")
                        
                        if profile_info.get('gender') and not profile_data["gender"]:
                            profile_data["gender"] = profile_info.get('gender')
                            print(f"👤 Found gender: {profile_data['gender']}")
                            
                        for location_field in ['location', 'address', 'homeAddress', 'city']:
                            location_data = profile_info.get(location_field)
                            if location_data and not profile_data["city"]:
                                if isinstance(location_data, str):
                                    print(f"📍 Found location in {location_field}: '{location_data}'")
                                    city = self.extract_city_with_ai(location_data)
                                    if city:
                                        profile_data["city"] = city
                                elif isinstance(location_data, dict):
                                    for key, value in location_data.items():
                                        if isinstance(value, str) and value:
                                            print(f"📍 Found nested location {key}: '{value}'")
                                            city = self.extract_city_with_ai(value)
                                            if city:
                                                profile_data["city"] = city
                                                break
                else:
                    print(f"❌ Extended profile API failed: {profile_response.status_code}")
            except Exception as e:
                print(f"⚠️ Extended profile API error: {str(e)}")

            # Try Microsoft Graph beta endpoint for more personal data
            try:
                print("📞 Calling Microsoft Graph /beta/me API...")
                beta_response = requests.get(
                    "https://graph.microsoft.com/beta/me",
                    headers=headers
                )
                
                if beta_response.status_code == 200:
                    beta_data = beta_response.json()
                    print(f"👤 Beta profile data: {json.dumps(beta_data, indent=2)[:500]}...")
                    
                    if beta_data.get("birthday") and not profile_data["birthdate"]:
                        profile_data["birthdate"] = beta_data.get("birthday")
                        print(f"📅 Found beta birthdate: {profile_data['birthdate']}")
                    if beta_data.get("gender") and not profile_data["gender"]:
                        profile_data["gender"] = beta_data.get("gender")
                        print(f"👤 Found beta gender: {profile_data['gender']}")
                    if beta_data.get("city") and not profile_data["city"]:
                        profile_data["city"] = beta_data.get("city")
                        print(f"🏙️ Found beta city: {profile_data['city']}")
                else:
                    print(f"❌ Beta API failed: {beta_response.status_code}")
            except Exception as e:
                print(f"⚠️ Beta API error: {str(e)}")

            # Try to get calendar events to extract location information
            try:
                print("📞 Calling Microsoft Graph /v1.0/me/events API...")
                calendar_response = requests.get(
                    "https://graph.microsoft.com/v1.0/me/events?$top=20&$select=subject,location",
                    headers=headers
                )
                
                if calendar_response.status_code == 200:
                    events_data = calendar_response.json()
                    events = events_data.get("value", [])
                    print(f"📅 Found {len(events)} calendar events")
                    
                    for event in events:
                        location = event.get("location", {})
                        if isinstance(location, dict):
                            display_name = location.get("displayName", "")
                            if display_name and not profile_data["city"]:
                                print(f"📍 Found calendar location: '{display_name}'")
                                city = self.extract_city_with_ai(display_name)
                                if city:
                                    profile_data["city"] = city
                                    break
                else:
                    print(f"❌ Calendar API failed: {calendar_response.status_code}")
            except Exception as e:
                print(f"⚠️ Calendar API error: {str(e)}")

        except Exception as e:
            print(f"❌ Error extracting Outlook profile: {str(e)}")
        
        print(f"✅ Final Outlook profile data: {profile_data}")
        return profile_data
    
    def scan_outlook_profile(self, email: str, user_id: str) -> Dict[str, Any]:
        """Scan Outlook for profile information using OAuth"""
        try:
            print(f"📧 Starting Outlook profile scan for: {email}")
            
            if not config.OUTLOOK_CLIENT_ID:
                print("❌ OUTLOOK_CLIENT_ID not configured")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Outlook Client ID not configured"
                }
            
            # Create MSAL app
            app = msal.PublicClientApplication(
                config.OUTLOOK_CLIENT_ID,
                authority="https://login.microsoftonline.com/common"
            )
            
            scopes = [
                "User.Read",
                "Mail.Send",
                "Calendars.Read"
            ]
            
            print(f"🔐 Starting MSAL interactive authentication...")
            
            # Interactive authentication
            try:
                result = app.acquire_token_interactive(
                    scopes=scopes,
                    redirect_uri="http://localhost"
                )
            except Exception:
                result = app.acquire_token_interactive(scopes=scopes)
            
            if "access_token" not in result:
                error_msg = result.get("error_description", result.get("error", "Unknown error"))
                print(f"❌ Outlook authentication failed: {error_msg}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": f"Outlook authentication failed: {error_msg}"
                }

            print("✅ Outlook authentication successful")

            # Store credentials for later use
            self.stored_credentials[user_id] = {
                "email": email,
                "credentials": result,
                "type": "outlook"
            }

            # Extract profile data
            profile_data = self.extract_outlook_profile_data(result['access_token'])
            
            result_data = {
                "name": profile_data["name"],
                "gender": profile_data["gender"],
                "birthdate": profile_data["birthdate"],
                "city": profile_data["city"],
                "success": True
            }
            
            print(f"✅ Outlook profile extraction result: {result_data}")
            return result_data
                
        except Exception as e:
            print(f"❌ Error scanning Outlook profile: {str(e)}")
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": str(e)
            }
    
    def scan_email_for_profile(self, email: str, user_id: str) -> Dict[str, Any]:
        """Main method to scan email for profile information"""
        print(f"📧 Starting email profile scan for: {email}")
        
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
    
    async def send_gmail_invites(self, user_id: str, recipient_emails: List[str], 
                               group_code: str, sender_name: str = "Arny User") -> Dict[str, Any]:
        """Send group invitations via Gmail API"""
        try:
            if user_id not in self.stored_credentials:
                return {
                    "success": False,
                    "error": "Gmail credentials not available. Please link your email first."
                }
            
            creds_data = self.stored_credentials[user_id]
            if creds_data["type"] != "gmail":
                return {
                    "success": False,
                    "error": "Gmail credentials not found for this user."
                }
            
            gmail_service = build("gmail", "v1", credentials=creds_data["credentials"])
            
            # Create email content
            subject, body = self.create_invite_email_content(sender_name, creds_data["email"], group_code)
            
            successful_sends = []
            failed_sends = []
            
            for recipient_email in recipient_emails:
                try:
                    # Create message
                    message = email.mime.text.MIMEText(body)
                    message['to'] = recipient_email
                    message['from'] = creds_data["email"]
                    message['subject'] = subject
                    
                    # Encode message
                    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
                    
                    # Send message
                    gmail_service.users().messages().send(
                        userId='me',
                        body={'raw': raw_message}
                    ).execute()
                    
                    successful_sends.append(recipient_email)
                    
                except Exception as e:
                    failed_sends.append({"email": recipient_email, "error": str(e)})
            
            return {
                "success": True,
                "sent_to": successful_sends,
                "failed": failed_sends
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Gmail service error: {str(e)}"
            }
    
    async def send_outlook_invites(self, user_id: str, recipient_emails: List[str], 
                                 group_code: str, sender_name: str = "Arny User") -> Dict[str, Any]:
        """Send group invitations via Microsoft Graph API"""
        try:
            if user_id not in self.stored_credentials:
                return {
                    "success": False,
                    "error": "Outlook credentials not available. Please link your email first."
                }
            
            creds_data = self.stored_credentials[user_id]
            if creds_data["type"] != "outlook":
                return {
                    "success": False,
                    "error": "Outlook credentials not found for this user."
                }
            
            headers = {"Authorization": f"Bearer {creds_data['credentials']['access_token']}"}
            
            # Create email content
            subject, body = self.create_invite_email_content(sender_name, creds_data["email"], group_code)
            
            successful_sends = []
            failed_sends = []
            
            for recipient_email in recipient_emails:
                try:
                    # Create message payload
                    message_payload = {
                        "message": {
                            "subject": subject,
                            "body": {
                                "contentType": "Text",
                                "content": body
                            },
                            "toRecipients": [
                                {
                                    "emailAddress": {
                                        "address": recipient_email
                                    }
                                }
                            ]
                        }
                    }
                    
                    # Send message
                    response = requests.post(
                        "https://graph.microsoft.com/v1.0/me/sendMail",
                        headers={**headers, "Content-Type": "application/json"},
                        json=message_payload
                    )
                    
                    if response.status_code == 202:  # Success for sendMail
                        successful_sends.append(recipient_email)
                    else:
                        failed_sends.append({
                            "email": recipient_email, 
                            "error": f"HTTP {response.status_code}: {response.text}"
                        })
                        
                except Exception as e:
                    failed_sends.append({"email": recipient_email, "error": str(e)})
            
            return {
                "success": True,
                "sent_to": successful_sends,
                "failed": failed_sends
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Outlook service error: {str(e)}"
            }
    
    async def send_group_invites(self, user_id: str, email_addresses: str, group_code: str, 
                               sender_name: str = "Arny User") -> Dict[str, Any]:
        """Main method to send group invitation emails"""
        # Validate and extract email addresses
        valid_emails = self.validate_email_addresses(email_addresses)
        
        if not valid_emails:
            return {
                "success": False,
                "error": "No valid email addresses found in the input."
            }
        
        if user_id not in self.stored_credentials:
            return {
                "success": False,
                "error": "Email credentials not available. Please link your email first."
            }
        
        creds_data = self.stored_credentials[user_id]
        
        # Send invites based on email type
        if creds_data["type"] == "gmail":
            result = await self.send_gmail_invites(user_id, valid_emails, group_code, sender_name)
        elif creds_data["type"] == "outlook":
            result = await self.send_outlook_invites(user_id, valid_emails, group_code, sender_name)
        else:
            return {
                "success": False,
                "error": f"Unsupported email type: {creds_data['type']}"
            }
        
        # Format response
        if result["success"]:
            sent_count = len(result.get("sent_to", []))
            failed_count = len(result.get("failed", []))
            
            if sent_count > 0 and failed_count == 0:
                return {
                    "success": True,
                    "message": f"Successfully sent invites to: {', '.join(result['sent_to'])}",
                    "sent_to": result["sent_to"],
                    "group_code": group_code
                }
            elif sent_count > 0 and failed_count > 0:
                return {
                    "success": True,
                    "message": f"Sent invites to: {', '.join(result['sent_to'])}. Failed to send to: {', '.join([f['email'] for f in result['failed']])}",
                    "sent_to": result["sent_to"],
                    "failed": result["failed"],
                    "group_code": group_code
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to send all invites. Errors: {result['failed']}"
                }
        else:
            return result
