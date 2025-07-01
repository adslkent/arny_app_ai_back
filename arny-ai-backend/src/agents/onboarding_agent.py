import json
import base64
import urllib.parse
import requests
import email.mime.text
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from openai import OpenAI
from agents import Agent, Runner, function_tool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import msal

from ..utils.config import config
from ..utils.group_codes import GroupCodeGenerator
from ..services.email_service import EmailService
from ..database.operations import DatabaseOperations
from ..database.models import OnboardingStep, OnboardingProgress

# Global variable to store the current agent instance
_current_onboarding_agent = None

def _get_onboarding_agent():
    """Get the current onboarding agent instance"""
    global _current_onboarding_agent
    return _current_onboarding_agent

# ==================== STANDALONE TOOL FUNCTIONS ====================

@function_tool
def scan_email_for_profile_tool(email: str) -> dict:
    """
    Scan email for profile information
    Returns a dict with keys: name, gender, birthdate, city
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        # Store email in collected data
        agent.current_collected_data["email"] = email
        
        # For now, return empty profile (email scanning would require OAuth setup)
        # In production, this would implement the full OAuth flow
        profile_data = {
            "name": None,
            "gender": None,
            "birthdate": None,
            "city": None
        }
        
        return profile_data
        
    except Exception as e:
        return {
            "name": None,
            "gender": None,
            "birthdate": None,
            "city": None,
            "error": str(e)
        }

@function_tool
def send_group_invites_tool(email_addresses: str) -> dict:
    """
    Send group invitation emails
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        group_code = agent.current_collected_data.get("group_code")
        sender_name = agent.current_collected_data.get("name", "Arny User")
        sender_email = agent.current_collected_data.get("email", "")
        
        if not group_code:
            return {"success": False, "error": "No group code available"}
        
        # Validate email addresses
        valid_emails = agent.email_service.validate_email_addresses(email_addresses)
        
        if not valid_emails:
            return {"success": False, "error": "No valid email addresses found"}
        
        # For now, simulate successful sending
        # In production, this would use the full email service
        return {
            "success": True,
            "message": f"Invitations would be sent to: {', '.join(valid_emails)}",
            "sent_to": valid_emails,
            "group_code": group_code
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@function_tool
def validate_group_code_tool(group_code: str) -> dict:
    """
    Validate a group code
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"valid": False, "error": "Agent not available"}
        
        # Check if user wants to skip
        if group_code.lower() in ["skip", "no", "none", "later"]:
            return {"valid": False, "skip": True, "message": "User wants to skip group setup"}
        
        # Validate format
        formatted_code = agent.group_generator.format_group_code(group_code)
        is_valid = agent.group_generator.validate_group_code(formatted_code)
        
        if not is_valid:
            return {"valid": False, "exists": False, "message": "Invalid group code format"}
        
        # Check if group exists (would need database call)
        # For now, simulate group existence check
        group_exists = True  # This would be: await agent.db.check_group_exists(formatted_code)
        
        if group_exists:
            # Store group info
            agent.current_collected_data["group_code"] = formatted_code
            agent.current_collected_data["group_role"] = "member"
            
            return {
                "valid": True,
                "exists": True,
                "group_code": formatted_code,
                "message": f"Successfully joined group {formatted_code}"
            }
        else:
            return {
                "valid": True,
                "exists": False,
                "message": "Group code format is valid but group does not exist"
            }
            
    except Exception as e:
        return {"valid": False, "error": str(e)}

@function_tool
def create_new_group_tool() -> dict:
    """
    Create a new group for the user
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        # Generate unique group code
        new_group_code = agent.group_generator.generate_group_code()
        
        # Store group info
        agent.current_collected_data["group_code"] = new_group_code
        agent.current_collected_data["group_role"] = "admin"
        
        return {
            "success": True,
            "group_code": new_group_code,
            "role": "admin",
            "message": f"Created new group with code: {new_group_code}"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================== ONBOARDING AGENT CLASS ====================

class OnboardingAgent:
    """
    LLM-driven onboarding agent using OpenAI Agents SDK with tools
    """
    
    def __init__(self):
        global _current_onboarding_agent
        
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.email_service = EmailService()
        self.db = DatabaseOperations()
        self.group_generator = GroupCodeGenerator()
        
        # Store this instance globally for tool access
        _current_onboarding_agent = self
        
        # Create the agent with tools using Agents SDK
        self.agent = Agent(
            name="Arny Onboarding Assistant",
            instructions=(
                "You are Arny AI, a helpful onboarding assistant for a travel planner "
                "personal assistant app. Your task is to obtain personal information from the "
                "app user as part of the onboarding process. Follow these steps:\n"
                "1. When the user provides a Group Code, use the validate_group_code_tool. If they say 'skip' or similar, use the create_new_group_tool.\n"
                "2. Ask about the user's Gmail or Outlook address, then use the scan_email_for_profile_tool "
                "to fetch name, gender, birthdate, and city. If you successfully extract ANY information (even partial), "
                "present what was found and ask the user to confirm or provide the missing details. "
                "For example: 'I found your name is John Smith, but I couldn't find your gender, birthdate, or city. "
                "Could you please provide these missing details?' Only if NO information is extracted at all should you "
                "ask for all details manually.\n"
                "3. Ask about the user's job details (specifically the user's employer, working schedule and holiday frequency). If any part is "
                "missing, prompt until complete, then confirm.\n"
                "4. Ask about the user's annual income and average monthly spending amount; repeat missing-item prompts until complete, then confirm.\n"
                "5. Ask about the user's holiday preferences (specifically the user's preferences for holiday activities the user likes to do); repeat missing-item prompts until complete, then confirm.\n"
                "6. Ask: 'Would you like to setup a group with people you know? This can always be done later.' If the user says yes, respond with 'Please invite users to your new group by providing their email addresses.' When they provide email addresses, respond with 'To confirm, I will be sending invites to {list all provided email addresses}. Are they correct?' If they confirm yes, use the send_group_invites_tool to send the invitation emails.\n"
                "Finally ONLY when all the above onboarding process is completed, then respond: 'Thank you, this completes your onboarding to Arny!'\n"
                "\n"
                "Always be friendly, conversational, and helpful. Keep track of what information you've already collected to avoid asking the same questions twice. "
                "The first message from the user will be their group code response."
            ),
            model="o4-mini",
            tools=[
                scan_email_for_profile_tool,
                send_group_invites_tool,
                validate_group_code_tool,
                create_new_group_tool
            ]
        )
    
    async def process_message(self, user_id: str, message: str, session_id: str, 
                            current_progress: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process message using OpenAI Agents SDK
        """
        
        try:
            # Store user context for tool calls
            self.current_user_id = user_id
            self.current_collected_data = current_progress.get('collected_data', {})
            
            # Get conversation history
            conversation_history = current_progress.get('conversation_history', [])
            
            # Create conversation context for the agent
            context_messages = []
            for msg in conversation_history:
                context_messages.append(msg)
            
            # Process with agent
            
            if not context_messages:
                # First message - FIXED: use await Runner.run instead of Runner.run_sync
                result = await Runner.run(self.agent, message)
            else:
                # Continue conversation with context - FIXED: use await Runner.run instead of Runner.run_sync
                result = await Runner.run(self.agent, context_messages + [{"role": "user", "content": message}])
            
            # Extract response
            assistant_message = result.final_output
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep conversation history manageable
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
            # Check if onboarding is complete
            onboarding_complete = "thank you, this completes your onboarding to arny" in assistant_message.lower()
            
            # Update progress
            updated_progress = {
                'collected_data': self.current_collected_data,
                'conversation_history': conversation_history
            }
            
            # Save progress to database
            if not onboarding_complete:
                current_step = self._determine_current_step(self.current_collected_data)
                await self.db.update_onboarding_progress(user_id, current_step, updated_progress)
            else:
                # Complete onboarding
                await self.db.complete_onboarding(user_id, self.current_collected_data)
            
            return {
                "message": assistant_message,
                "onboarding_complete": onboarding_complete,
                "collected_data": self.current_collected_data,
                "progress_data": updated_progress
            }
            
        except Exception as e:
            print(f"Error in onboarding agent: {e}")
            return {
                "message": "I apologize, but I encountered an error. Could you please try again?",
                "onboarding_complete": False,
                "collected_data": self.current_collected_data,
                "error": str(e)
            }
    
    def _determine_current_step(self, collected_data: Dict[str, Any]) -> OnboardingStep:
        """Determine current onboarding step based on collected data"""
        
        if not collected_data.get("group_code"):
            return OnboardingStep.GROUP_CODE
        elif not collected_data.get("email"):
            return OnboardingStep.EMAIL_SCAN
        elif not all([collected_data.get("name"), collected_data.get("gender"), 
                     collected_data.get("birthdate"), collected_data.get("city")]):
            return OnboardingStep.PERSONAL_INFO
        elif not all([collected_data.get("employer"), collected_data.get("working_schedule"), 
                     collected_data.get("holiday_frequency")]):
            return OnboardingStep.JOB_DETAILS
        elif not all([collected_data.get("annual_income"), collected_data.get("monthly_spending")]):
            return OnboardingStep.FINANCIAL_INFO
        elif not collected_data.get("holiday_preferences"):
            return OnboardingStep.HOLIDAY_PREFERENCES
        else:
            return OnboardingStep.GROUP_SETUP
