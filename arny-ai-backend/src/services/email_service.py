import re
import json
import base64
import requests
import email.mime.text
import email.mime.multipart
from typing import List, Dict, Any, Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import msal

from ..utils.config import config

class EmailService:
    """Service for sending group invitation emails via Gmail or Outlook"""
    
    def __init__(self):
        pass
    
    def validate_email_addresses(self, email_string: str) -> List[str]:
        """
        Extract and validate email addresses from a string
        Returns list of valid email addresses
        """
        # Regular expression for email validation
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Find all email addresses in the string
        emails = re.findall(email_pattern, email_string)
        
        # Remove duplicates and return
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
    
    async def send_gmail_invites(self, credentials: Credentials, sender_name: str, 
                               sender_email: str, recipient_emails: List[str], 
                               group_code: str) -> Dict[str, Any]:
        """Send group invitations via Gmail API"""
        try:
            gmail_service = build("gmail", "v1", credentials=credentials)
            
            # Create email content
            subject, body = self.create_invite_email_content(sender_name, sender_email, group_code)
            
            successful_sends = []
            failed_sends = []
            
            for recipient_email in recipient_emails:
                try:
                    # Create message
                    message = email.mime.text.MIMEText(body)
                    message['to'] = recipient_email
                    message['from'] = sender_email
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
    
    async def send_outlook_invites(self, access_token: str, sender_name: str, 
                                 sender_email: str, recipient_emails: List[str], 
                                 group_code: str) -> Dict[str, Any]:
        """Send group invitations via Microsoft Graph API"""
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            
            # Create email content
            subject, body = self.create_invite_email_content(sender_name, sender_email, group_code)
            
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
    
    async def send_group_invites(self, email_type: str, credentials: Any, sender_name: str,
                               sender_email: str, email_addresses: str, group_code: str) -> Dict[str, Any]:
        """
        Main method to send group invitation emails
        
        Args:
            email_type: "gmail" or "outlook"
            credentials: Email service credentials
            sender_name: Name of the person sending invites
            sender_email: Email of the person sending invites
            email_addresses: String containing email addresses
            group_code: Group code to include in invite
            
        Returns:
            Result dictionary with success status and details
        """
        # Validate and extract email addresses
        valid_emails = self.validate_email_addresses(email_addresses)
        
        if not valid_emails:
            return {
                "success": False,
                "error": "No valid email addresses found in the input."
            }
        
        # Send invites based on email type
        if email_type == "gmail":
            result = await self.send_gmail_invites(
                credentials, sender_name, sender_email, valid_emails, group_code
            )
        elif email_type == "outlook":
            result = await self.send_outlook_invites(
                credentials, sender_name, sender_email, valid_emails, group_code
            )
        else:
            return {
                "success": False,
                "error": f"Unsupported email type: {email_type}"
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