import json
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from openai import OpenAI
from agents import Agent, Runner, function_tool

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

def _run_async_safely(coro):
    """Run async coroutine safely by using the current event loop or creating a new one"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_in_new_loop, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

def _run_in_new_loop(coro):
    """Run coroutine in a completely new event loop"""
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
        asyncio.set_event_loop(None)

# ==================== ENHANCED TOOL FUNCTIONS ====================

@function_tool
def scan_email_for_profile_tool(email: str) -> dict:
    """
    Scan email for profile information using environment-appropriate method
    Returns a dict with keys: name, gender, birthdate, city
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"📧 Scanning email for profile: {email}")
        
        # Store email in collected data
        agent.current_collected_data["email"] = email
        
        # Use the enhanced email service to scan for profile
        profile_data = agent.email_service.scan_email_for_profile(email, agent.current_user_id)
        
        print(f"📊 Profile scan results: {profile_data}")
        
        # Handle email scanning failure gracefully
        if not profile_data.get("success"):
            print(f"📧 Email scanning failed: {profile_data.get('error', 'Unknown error')}")
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": profile_data.get('error', 'Email scanning not available'),
                "message": "I couldn't scan your email automatically. Let me ask for your information manually instead."
            }
        
        # Store any found profile data in collected data
        if profile_data.get("success"):
            if profile_data.get("name"):
                agent.current_collected_data["name"] = profile_data["name"]
            if profile_data.get("gender"):
                agent.current_collected_data["gender"] = profile_data["gender"]
            if profile_data.get("birthdate"):
                agent.current_collected_data["birthdate"] = profile_data["birthdate"]
            if profile_data.get("city"):
                agent.current_collected_data["city"] = profile_data["city"]
        
        return {
            "name": profile_data.get("name"),
            "gender": profile_data.get("gender"),
            "birthdate": profile_data.get("birthdate"),
            "city": profile_data.get("city"),
            "success": profile_data.get("success", False),
            "error": profile_data.get("error")
        }
        
    except Exception as e:
        print(f"Error in scan_email_for_profile_tool: {str(e)}")
        return {
            "name": None,
            "gender": None,
            "birthdate": None,
            "city": None,
            "success": False,
            "error": str(e),
            "message": "I encountered an error scanning your email. Let me ask for your information manually instead."
        }

@function_tool
def store_personal_info_tool(name: str = None, gender: str = None, birthdate: str = None, city: str = None) -> dict:
    """
    Store personal information from user input
    
    Args:
        name: User's full name
        gender: User's gender 
        birthdate: User's birthdate in YYYY-MM-DD format
        city: User's city
        
    Returns:
        Dict with success status and stored information
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        print(f"💾 Storing personal info: name={name}, gender={gender}, birthdate={birthdate}, city={city}")
        
        # Store non-empty values in collected data
        if name:
            agent.current_collected_data["name"] = name
        if gender:
            agent.current_collected_data["gender"] = gender
        if birthdate:
            agent.current_collected_data["birthdate"] = birthdate
        if city:
            agent.current_collected_data["city"] = city
        
        print(f"📈 Updated collected data: {agent.current_collected_data}")
        
        # Check what we have collected so far
        collected_fields = []
        if agent.current_collected_data.get("name"):
            collected_fields.append(f"name: {agent.current_collected_data['name']}")
        if agent.current_collected_data.get("gender"):
            collected_fields.append(f"gender: {agent.current_collected_data['gender']}")
        if agent.current_collected_data.get("birthdate"):
            collected_fields.append(f"birthdate: {agent.current_collected_data['birthdate']}")
        if agent.current_collected_data.get("city"):
            collected_fields.append(f"city: {agent.current_collected_data['city']}")
        
        return {
            "success": True,
            "message": f"Stored personal information: {', '.join(collected_fields)}",
            "collected_fields": collected_fields
        }
        
    except Exception as e:
        print(f"Error in store_personal_info_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def store_job_details_tool(employer: str = None, working_schedule: str = None, holiday_frequency: str = None) -> dict:
    """
    Store job details from user input
    
    Args:
        employer: User's employer/company name
        working_schedule: User's working schedule
        holiday_frequency: How often user takes holidays per year
        
    Returns:
        Dict with success status and stored information
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        print(f"💼 Storing job details: employer={employer}, schedule={working_schedule}, holidays={holiday_frequency}")
        
        # Store non-empty values in collected data
        if employer:
            agent.current_collected_data["employer"] = employer
        if working_schedule:
            agent.current_collected_data["working_schedule"] = working_schedule
        if holiday_frequency:
            agent.current_collected_data["holiday_frequency"] = holiday_frequency
        
        print(f"📈 Updated collected data: {agent.current_collected_data}")
        
        # Check what we have collected so far
        collected_fields = []
        if agent.current_collected_data.get("employer"):
            collected_fields.append(f"employer: {agent.current_collected_data['employer']}")
        if agent.current_collected_data.get("working_schedule"):
            collected_fields.append(f"schedule: {agent.current_collected_data['working_schedule']}")
        if agent.current_collected_data.get("holiday_frequency"):
            collected_fields.append(f"holidays: {agent.current_collected_data['holiday_frequency']}")
        
        return {
            "success": True,
            "message": f"Stored job details: {', '.join(collected_fields)}",
            "collected_fields": collected_fields
        }
        
    except Exception as e:
        print(f"Error in store_job_details_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def store_financial_info_tool(annual_income: str = None, monthly_spending: str = None) -> dict:
    """
    Store financial information from user input
    
    Args:
        annual_income: User's annual income range
        monthly_spending: User's average monthly spending
        
    Returns:
        Dict with success status and stored information
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        print(f"💰 Storing financial info: income={annual_income}, spending={monthly_spending}")
        
        # Store non-empty values in collected data
        if annual_income:
            agent.current_collected_data["annual_income"] = annual_income
        if monthly_spending:
            agent.current_collected_data["monthly_spending"] = monthly_spending
        
        print(f"📈 Updated collected data: {agent.current_collected_data}")
        
        # Check what we have collected so far
        collected_fields = []
        if agent.current_collected_data.get("annual_income"):
            collected_fields.append(f"income: {agent.current_collected_data['annual_income']}")
        if agent.current_collected_data.get("monthly_spending"):
            collected_fields.append(f"spending: {agent.current_collected_data['monthly_spending']}")
        
        return {
            "success": True,
            "message": f"Stored financial information: {', '.join(collected_fields)}",
            "collected_fields": collected_fields
        }
        
    except Exception as e:
        print(f"Error in store_financial_info_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def store_holiday_preferences_tool(holiday_preferences: str) -> dict:
    """
    Store holiday preferences from user input
    
    Args:
        holiday_preferences: User's holiday/travel activity preferences
        
    Returns:
        Dict with success status and stored information
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        print(f"🏖️ Storing holiday preferences: {holiday_preferences}")
        
        # Store holiday preferences in collected data
        agent.current_collected_data["holiday_preferences"] = holiday_preferences
        
        print(f"📈 Updated collected data: {agent.current_collected_data}")
        
        return {
            "success": True,
            "message": f"Stored holiday preferences: {holiday_preferences}",
            "preferences": holiday_preferences
        }
        
    except Exception as e:
        print(f"Error in store_holiday_preferences_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def send_group_invites_tool(email_addresses: str) -> dict:
    """
    Send group invitation emails using the enhanced email service
    
    Args:
        email_addresses: String containing email addresses to send invites to
        
    Returns:
        Dict with success status and details
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        group_code = agent.current_collected_data.get("group_code")
        sender_name = agent.current_collected_data.get("name", "Arny User")
        
        if not group_code:
            return {"success": False, "error": "No group code available"}
        
        print(f"📧 Sending group invites to: {email_addresses}")
        print(f"📋 Group code: {group_code}, Sender: {sender_name}")
        
        # Use the enhanced email service to send invites
        result = _run_async_safely(
            agent.email_service.send_group_invites(
                user_id=agent.current_user_id,
                email_addresses=email_addresses,
                group_code=group_code,
                sender_name=sender_name
            )
        )
        
        print(f"📬 Invite sending result: {result}")
        
        if result.get("success"):
            # Mark invites as sent
            agent.current_collected_data["group_invites_sent"] = True
            agent.current_collected_data["invited_emails"] = result.get("sent_to", [])
        else:
            # Handle email sending limitation in Lambda
            agent.current_collected_data["group_invites_sent"] = True
            agent.current_collected_data["group_code_shared"] = group_code
        
        return result
        
    except Exception as e:
        print(f"Error in send_group_invites_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def validate_group_code_tool(group_code: str) -> dict:
    """
    Validate a group code and check if it exists in the database
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"valid": False, "error": "Agent not available"}
        
        print(f"🔍 Validating group code: {group_code}")
        
        # Check if user wants to skip
        if group_code.lower() in ["skip", "no", "none", "later"]:
            print("👤 User wants to skip group setup")
            return {"valid": False, "skip": True, "message": "User wants to skip group setup"}
        
        # Validate format
        formatted_code = agent.group_generator.format_group_code(group_code)
        is_valid = agent.group_generator.validate_group_code(formatted_code)
        
        print(f"📝 Formatted code: {formatted_code}, Valid format: {is_valid}")
        
        if not is_valid:
            return {"valid": False, "exists": False, "message": "Invalid group code format. Group codes should be 4-10 alphanumeric characters. Please check the group code and try again, or type 'skip' to skip group setup for now."}
        
        # Check if group exists in database using safe async handling
        try:
            print("🔍 Checking if group exists in database...")
            group_exists = _run_async_safely(agent.db.check_group_exists(formatted_code))
            print(f"📊 Group exists: {group_exists}")
        except Exception as db_error:
            print(f"❌ Database error checking group existence: {str(db_error)}")
            import traceback
            traceback.print_exc()
            return {"valid": False, "error": f"Database error: {str(db_error)}"}
        
        if group_exists:
            # Add user to existing group as member
            try:
                print(f"➕ Adding user {agent.current_user_id} to existing group {formatted_code}")
                success = _run_async_safely(agent.db.add_group_member(formatted_code, agent.current_user_id, "member"))
                print(f"✅ Successfully added to group: {success}")
            except Exception as db_error:
                print(f"❌ Database error joining group: {str(db_error)}")
                return {"valid": True, "exists": True, "error": f"Failed to join group: {str(db_error)}"}
            
            if success:
                # Store group info
                agent.current_collected_data["group_code"] = formatted_code
                agent.current_collected_data["group_role"] = "member"
                
                print(f"📈 Updated collected data: {agent.current_collected_data}")
                
                return {
                    "valid": True,
                    "exists": True,
                    "group_code": formatted_code,
                    "message": f"Successfully joined group {formatted_code}"
                }
            else:
                return {
                    "valid": True,
                    "exists": True,
                    "error": f"Failed to join group {formatted_code}. You may already be a member."
                }
        else:
            print(f"❌ Group {formatted_code} does not exist")
            return {
                "valid": True,
                "exists": False,
                "group_code": formatted_code,
                "message": f"Group {formatted_code} does not exist. Please check the group code and try again, or type 'skip' to skip group setup for now."
            }
            
    except Exception as e:
        print(f"❌ Error in validate_group_code_tool: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"valid": False, "error": str(e)}

@function_tool
def decline_group_invites_tool() -> dict:
    """
    Mark that the user declined to send group invites (for admin users)
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        print("⏭️ User declined to send group invites")
        
        # Mark that user declined group invites
        agent.current_collected_data["group_invites_declined"] = True
        
        print(f"📈 Updated collected data: {agent.current_collected_data}")
        
        return {
            "success": True,
            "message": "Group invites declined. You can always invite people later through the app."
        }
        
    except Exception as e:
        print(f"❌ Error in decline_group_invites_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def skip_group_setup_tool() -> dict:
    """
    Skip group setup - automatically generate a random group code for the user (hidden from user)
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}
        
        print("⏭️ Skipping group setup - generating new group code")
        
        # Check if group setup was already skipped to avoid duplicate processing
        if agent.current_collected_data.get("group_skipped"):
            print("⚠️ Group setup already skipped, using existing data")
            return {
                "success": True,
                "message": "Group setup skipped. You can always invite family or friends later."
            }
        
        # Generate a unique random group code for this user
        try:
            print("🔍 Getting existing group codes...")
            existing_codes = _run_async_safely(agent.db.get_existing_group_codes())
            print(f"📊 Found {len(existing_codes)} existing codes")
        except Exception as db_error:
            print(f"❌ Database error getting existing codes: {str(db_error)}")
            return {"success": False, "error": f"Database error getting existing codes: {str(db_error)}"}
        
        new_group_code = agent.group_generator.generate_unique_group_code(existing_codes)
        print(f"🎲 Generated new group code: {new_group_code}")
        
        # Create group in database with user as admin
        try:
            print(f"➕ Creating new group {new_group_code} with user {agent.current_user_id} as admin")
            success = _run_async_safely(agent.db.add_group_member(new_group_code, agent.current_user_id, "admin"))
            print(f"✅ Group creation success: {success}")
        except Exception as db_error:
            print(f"❌ Database error creating group: {str(db_error)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Database error creating group: {str(db_error)}"}
        
        if success:
            # Store that user skipped group setup but has a group code (hidden from user)
            agent.current_collected_data["group_code"] = new_group_code
            agent.current_collected_data["group_role"] = "admin"
            agent.current_collected_data["group_skipped"] = True
            
            print(f"📈 Updated collected data: {agent.current_collected_data}")
            print(f"🎉 Successfully created group {new_group_code} for user {agent.current_user_id}")
            
            # FIXED: Don't mention the specific group code to the user
            return {
                "success": True,
                "message": "Group setup skipped. You can always invite family or friends later."
            }
        else:
            return {
                "success": False,
                "error": "Failed to create personal group in database"
            }
        
    except Exception as e:
        print(f"❌ Error in skip_group_setup_tool: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# ==================== ENHANCED ONBOARDING AGENT CLASS ====================

class OnboardingAgent:
    """
    Enhanced LLM-driven onboarding agent with improved completion detection
    """
    
    def __init__(self):
        global _current_onboarding_agent
        
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.email_service = EmailService()  # Use the enhanced email service
        self.db = DatabaseOperations()
        self.group_generator = GroupCodeGenerator()
        
        # Store this instance globally for tool access
        _current_onboarding_agent = self
        
        # Create the agent with enhanced tools using Agents SDK
        self.agent = Agent(
            name="Arny Onboarding Assistant",
            instructions=(
                "You are Arny AI, a helpful onboarding assistant for a travel planner "
                "personal assistant app. Your task is to obtain personal information from the "
                "app user as part of the onboarding process. Follow these steps:\n\n"
                "IMPORTANT: Continue conversations from where they left off based on collected data.\n\n"
                "1. GROUP CODE SETUP:\n"
                "   - When the user provides a Group Code, use validate_group_code_tool to check if it exists.\n"
                "   - If the group EXISTS: They will automatically join it as a member.\n"
                "   - If the group DOES NOT EXIST: Tell them the group doesn't exist and ask them to check the group code and try again, or type 'skip' to skip group setup for now.\n"
                "   - If they say 'skip', 'no', 'none', or 'later', use skip_group_setup_tool ONLY ONCE.\n"
                "   - IMPORTANT: When a user skips group setup, NEVER mention any specific group code to them. Just say that group setup has been skipped and they can invite people later.\n\n"
                "2. EMAIL SCANNING:\n"
                "   Ask about the user's Gmail or Outlook address, then use scan_email_for_profile_tool "
                "to fetch name, gender, birthdate, and city. IMPORTANT: If email scanning fails or returns an error, "
                "gracefully proceed to manual data collection without making the user feel like something went wrong. "
                "For example, if scanning fails, say: 'I'll help you fill out your profile manually instead.' "
                "If you successfully extract ANY information (even partial), "
                "present what was found and ask the user to confirm or provide the missing details. "
                "For example: 'I found your name is John Smith, but I couldn't find your gender, birthdate, or city. "
                "Could you please provide these missing details?' Only if NO information is extracted at all should you "
                "ask for all details manually.\n\n"
                "3. PERSONAL INFO:\n"
                "   Collect: name, gender, birthdate, city. When the user provides this information, use store_personal_info_tool to save it. "
                "Ask for any missing details and confirm when complete.\n\n"
                "4. JOB DETAILS:\n"
                "   Ask about: employer, working schedule, holiday frequency. When the user provides this information, use store_job_details_tool to save it. "
                "Prompt for missing items until complete, then confirm.\n\n"
                "5. FINANCIAL INFO:\n"
                "   Ask about: annual income range, average monthly spending amount. When the user provides this information, use store_financial_info_tool to save it. "
                "Prompt until complete, then confirm.\n\n"
                "6. HOLIDAY PREFERENCES:\n"
                "   Ask about: holiday activities the user likes to do. When the user provides this information, use store_holiday_preferences_tool to save it. "
                "Prompt until complete, then confirm.\n\n"
                "7. GROUP INVITATIONS (ONLY if user has group_role = 'admin'):\n"
                "   - If the user joined an existing group (group_role = 'member'), SKIP this step entirely.\n"
                "   - If the user skipped group setup (group_role = 'admin'), ask: 'Would you like to setup a group with people you know? This can always be done later.' "
                "If the user says yes, respond with 'Please invite users to your new group by providing their email addresses.' "
                "When they provide email addresses, respond with 'To confirm, I will be sending invites to {list all provided email addresses}. Are they correct?' "
                "If they confirm yes, use send_group_invites_tool to send the invitation emails. "
                "If the user says no to setting up a group or declines to send invites, use decline_group_invites_tool to mark this step as complete. "
                "If email sending fails, gracefully explain that the group code can be shared manually.\n\n"
                "Finally, ONLY when all the above onboarding process is completed, respond: "
                "'Thank you, this completes your onboarding to Arny!'\n\n"
                "Always be friendly, conversational, and helpful. Keep track of what information you've already collected to avoid asking the same questions twice. "
                "DO NOT call the same tool multiple times in one response. "
                "CONTINUE FROM WHERE THE CONVERSATION LEFT OFF - check collected data to see what step to proceed with. "
                "NEVER reveal specific group codes to users when they skip group setup. "
                "ALWAYS use the appropriate store_*_tool when users provide information to ensure it gets saved. "
                "Handle email scanning failures gracefully without making the user feel like something is broken."
            ),
            model="o4-mini",
            tools=[
                scan_email_for_profile_tool,
                store_personal_info_tool,
                store_job_details_tool,
                store_financial_info_tool,
                store_holiday_preferences_tool,
                send_group_invites_tool,
                decline_group_invites_tool,
                validate_group_code_tool,
                skip_group_setup_tool
            ]
        )
    
    async def process_message(self, user_id: str, message: str, session_id: str, 
                            current_progress: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process message using OpenAI Agents SDK with enhanced completion detection
        """
        
        try:
            # Store user context for tool calls
            self.current_user_id = user_id
            self.current_collected_data = current_progress.get('collected_data', {})
            
            print(f"🔧 Processing message for user {user_id}: {message}")
            print(f"📊 Current collected data: {self.current_collected_data}")
            
            # Get conversation history
            conversation_history = current_progress.get('conversation_history', [])
            
            # Determine current step based on collected data
            current_step = self._determine_current_step_from_data(self.current_collected_data)
            print(f"🎯 Determined current step: {current_step}")
            
            # Create conversation context for the agent
            context_messages = []
            
            # Add system context about current progress
            if self.current_collected_data:
                progress_summary = self._create_progress_summary(self.current_collected_data, current_step)
                context_messages.append({
                    "role": "system", 
                    "content": f"CURRENT PROGRESS:\n{progress_summary}\n\nContinue from this point based on what's missing."
                })
            
            # Add conversation history
            for msg in conversation_history:
                context_messages.append(msg)
            
            # Process with agent
            if not conversation_history:
                # First message in conversation
                print("🚀 Starting new conversation")
                result = await Runner.run(self.agent, message)
            else:
                # Continue conversation with context
                print(f"🔄 Continuing conversation with {len(context_messages)} previous messages")
                result = await Runner.run(self.agent, context_messages + [{"role": "user", "content": message}])
            
            # Extract response
            assistant_message = result.final_output
            print(f"🤖 Agent response: {assistant_message}")
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep conversation history manageable
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
            # ENHANCED: Check if onboarding is complete using both data-based and LLM-based detection
            onboarding_complete = await self._detect_onboarding_completion_enhanced(assistant_message)
            
            # Update progress
            updated_progress = {
                'collected_data': self.current_collected_data,
                'conversation_history': conversation_history
            }
            
            print(f"📈 Updated collected data: {self.current_collected_data}")
            print(f"🏁 Onboarding complete: {onboarding_complete}")
            
            # Save progress to database
            if not onboarding_complete:
                current_step_enum = self._determine_current_step(self.current_collected_data)
                print(f"💾 Saving progress - Current step: {current_step_enum.value}")
                await self.db.update_onboarding_progress(user_id, current_step_enum, updated_progress)
            else:
                # Complete onboarding
                print("🎉 Completing onboarding...")
                completion_success = await self.db.complete_onboarding(user_id, self.current_collected_data)
                print(f"💾 Onboarding completion in database: {completion_success}")
                
                if not completion_success:
                    print("⚠️ Database completion failed, but marking as complete anyway")
            
            return {
                "message": assistant_message,
                "onboarding_complete": onboarding_complete,
                "collected_data": self.current_collected_data,
                "progress_data": updated_progress
            }
            
        except Exception as e:
            print(f"❌ Error in onboarding agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": "I apologize, but I encountered an error. Could you please try again?",
                "onboarding_complete": False,
                "collected_data": self.current_collected_data,
                "error": str(e)
            }
    
    async def _detect_onboarding_completion_enhanced(self, message: str) -> bool:
        """Enhanced onboarding completion detection using both data completeness and LLM analysis"""
        try:
            print(f"🔍 Enhanced completion detection for message: '{message[:100]}...'")
            
            # Method 1: Data-based completion check (primary method)
            data_complete = self._check_data_completeness()
            print(f"📊 Data completeness check: {data_complete}")
            
            # Method 2: LLM-based completion check (secondary method)
            llm_complete = await self._detect_onboarding_completion_llm(message)
            print(f"🤖 LLM completion check: {llm_complete}")
            
            # Method 3: Phrase-based completion check (fallback method)
            phrase_complete = self._fallback_phrase_detection(message)
            print(f"📝 Phrase completion check: {phrase_complete}")
            
            # Completion logic: Data complete AND (LLM complete OR phrase complete)
            # This ensures all data is collected but also respects the agent's intention
            final_complete = data_complete and (llm_complete or phrase_complete)
            
            print(f"🏁 Final completion decision: {final_complete} (data: {data_complete}, llm: {llm_complete}, phrase: {phrase_complete})")
            
            return final_complete
            
        except Exception as e:
            print(f"❌ Error in enhanced completion detection: {e}")
            # Fallback to data completeness only
            return self._check_data_completeness()
    
    def _check_data_completeness(self) -> bool:
        """Check if all required onboarding data has been collected"""
        try:
            required_fields = [
                "group_code",           # Group setup completed
                "email",               # Email provided
                "name", "gender", "birthdate", "city",  # Personal info
                "employer", "working_schedule", "holiday_frequency",  # Job details
                "annual_income", "monthly_spending",  # Financial info
                "holiday_preferences"  # Holiday preferences
            ]
            
            missing_fields = []
            for field in required_fields:
                if not self.current_collected_data.get(field):
                    missing_fields.append(field)
            
            is_complete = len(missing_fields) == 0
            
            if missing_fields:
                print(f"📋 Missing required fields: {missing_fields}")
            else:
                print(f"✅ All required data collected!")
            
            # Additional check for group invites (if applicable)
            if is_complete and self.current_collected_data.get("group_role") == "admin":
                # For admin users, check if they've been asked about group invites
                if not self.current_collected_data.get("group_invites_sent") and not self.current_collected_data.get("group_invites_declined"):
                    print(f"⏳ Admin user hasn't completed group invites step yet")
                    is_complete = False
            
            return is_complete
            
        except Exception as e:
            print(f"❌ Error checking data completeness: {e}")
            return False
    
    async def _detect_onboarding_completion_llm(self, message: str) -> bool:
        """LLM-based onboarding completion detection"""
        try:
            print(f"🔍 Using LLM to detect onboarding completion for message: '{message[:100]}...'")
            
            # Use OpenAI's o4-mini model to detect onboarding completion
            prompt = f"""You are analyzing a message from an AI onboarding assistant to determine if it indicates that the onboarding process has been completed.

Message to analyze: "{message}"

Determine if this message indicates that the onboarding process has been completed. Look for indicators such as:
- Explicit statements about completing onboarding (e.g., "this completes your onboarding", "onboarding is complete")
- Thank you messages that suggest finalization
- Statements that the user can now proceed to use the main features
- Congratulations or completion confirmations
- Messages indicating the user is ready to start using the app

Respond with only "YES" if the message clearly indicates onboarding completion, or "NO" if it does not."""

            response = self.openai_client.responses.create(
                model="o4-mini",
                input=prompt
            )
            
            # Extract response
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                response_text = content_item.text.strip().upper()
                                
                                print(f"🤖 LLM completion detection result: '{response_text}'")
                                
                                is_complete = response_text == "YES"
                                if is_complete:
                                    print("🎉 LLM detected onboarding completion!")
                                else:
                                    print("🔄 LLM determined onboarding is not complete yet")
                                
                                return is_complete
            
            # If we can't parse the response, fall back to phrase detection
            print("⚠️ Could not parse LLM response, using fallback detection")
            return self._fallback_phrase_detection(message)
            
        except Exception as e:
            print(f"❌ Error in LLM-based completion detection: {e}")
            # Fallback to phrase-based detection on error
            return self._fallback_phrase_detection(message)

    def _fallback_phrase_detection(self, message: str) -> bool:
        """Fallback phrase-based detection for onboarding completion"""
        print("🔄 Using fallback phrase-based detection")
        
        message_lower = message.lower()

        # Key phrases that indicate completion
        completion_phrases = [
            "this completes your onboarding to arny",
            "completes your onboarding to arny", 
            "your onboarding to arny is complete",
            "onboarding to arny is now complete",
            "thank you, this completes your onboarding",
            "that completes your onboarding",
            "onboarding complete",
            "welcome to arny"
        ]

        # Check if any completion phrase is found
        for phrase in completion_phrases:
            if phrase in message_lower:
                print(f"✅ Fallback detection found completion phrase: '{phrase}'")
                return True

        print("❌ Fallback detection found no completion phrases")
        return False

    def _determine_current_step_from_data(self, collected_data: Dict[str, Any]) -> str:
        """Determine current step based on collected data for progress summary"""
        
        if not collected_data.get("group_code"):
            return "waiting_for_group_code"
        elif not collected_data.get("email"):
            return "waiting_for_email"
        elif not all([collected_data.get("name"), collected_data.get("gender"), 
                     collected_data.get("birthdate"), collected_data.get("city")]):
            return "collecting_personal_info"
        elif not all([collected_data.get("employer"), collected_data.get("working_schedule"), 
                     collected_data.get("holiday_frequency")]):
            return "collecting_job_details"
        elif not all([collected_data.get("annual_income"), collected_data.get("monthly_spending")]):
            return "collecting_financial_info"
        elif not collected_data.get("holiday_preferences"):
            return "collecting_holiday_preferences"
        elif collected_data.get("group_role") == "admin" and not collected_data.get("group_invites_sent"):
            return "offering_group_invites"
        else:
            return "ready_to_complete"
    
    def _create_progress_summary(self, collected_data: Dict[str, Any], current_step: str) -> str:
        """Create a summary of current progress for the agent"""
        
        summary = []
        summary.append("ALREADY COMPLETED:")
        
        if collected_data.get("group_code"):
            summary.append(f"- Group Setup: Completed (Role: {collected_data.get('group_role', 'unknown')})")
        
        if collected_data.get("email"):
            summary.append(f"- Email: {collected_data['email']}")
        
        personal_info = []
        for field in ["name", "gender", "birthdate", "city"]:
            if collected_data.get(field):
                personal_info.append(f"{field}: {collected_data[field]}")
        if personal_info:
            summary.append(f"- Personal Info: {', '.join(personal_info)}")
        
        job_info = []
        for field in ["employer", "working_schedule", "holiday_frequency"]:
            if collected_data.get(field):
                job_info.append(f"{field}: {collected_data[field]}")
        if job_info:
            summary.append(f"- Job Details: {', '.join(job_info)}")
        
        financial_info = []
        for field in ["annual_income", "monthly_spending"]:
            if collected_data.get(field):
                financial_info.append(f"{field}: {collected_data[field]}")
        if financial_info:
            summary.append(f"- Financial Info: {', '.join(financial_info)}")
        
        if collected_data.get("holiday_preferences"):
            summary.append(f"- Holiday Preferences: {collected_data['holiday_preferences']}")
        
        summary.append(f"\nCURRENT STEP: {current_step}")
        
        return "\n".join(summary)
    
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
