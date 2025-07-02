"""
Database operations for Arny AI - FIXED VERSION

Fixed Issues:
1. Consistent UUID validation across all methods
2. Better error handling for UUID validation
3. Improved progress loading and saving
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, date, timedelta
import json
import uuid
import logging
from supabase import create_client, Client
from postgrest.exceptions import APIError

from ..utils.config import config
from .models import (
    UserProfile, OnboardingProgress, GroupMember, ChatMessage,
    FlightSearch, HotelSearch, UserPreferences, BookingRequest,
    TravelItinerary, OnboardingStep, UserRole, MessageType,
    BookingStatus, PaginationInfo
)

# Set up logging
logger = logging.getLogger(__name__)

class DatabaseOperations:
    """
    Database operations using Supabase - FIXED VERSION
    
    This class provides all database operations for the Arny AI application.
    It uses Supabase as the backend and handles all CRUD operations with
    proper error handling and data validation.
    """
    
    def __init__(self):
        """Initialize the database connection"""
        try:
            self.client: Client = create_client(
                config.SUPABASE_URL, 
                config.SUPABASE_SERVICE_ROLE_KEY
            )
            logger.info("Database client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database client: {e}")
            raise ConnectionError(f"Database connection failed: {e}")
    
    def _validate_and_format_uuid(self, user_id: str) -> tuple[bool, str]:
        """
        Consistent UUID validation and formatting across all methods
        
        Args:
            user_id: User ID to validate
            
        Returns:
            Tuple of (is_valid, cleaned_user_id or error_message)
        """
        if not user_id:
            return False, "User ID cannot be empty"
        
        try:
            # Convert to string and strip whitespace
            user_id_str = str(user_id).strip()
            
            # Remove any quotes that might be present
            user_id_str = user_id_str.strip('"\'')
            
            # Validate UUID format
            uuid_obj = uuid.UUID(user_id_str)
            
            # Return the string representation to ensure consistency
            return True, str(uuid_obj)
            
        except ValueError as e:
            return False, f"Invalid UUID format: {e}"
        except Exception as e:
            return False, f"Unexpected validation error: {e}"
    
    # ==================== USER PROFILE OPERATIONS ====================
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by user_id - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_user_profile: {validated_user_id}")
                return None
            
            logger.info(f"Getting user profile for user_id: {validated_user_id}")
            
            response = self.client.table("user_profiles").select("*").eq("user_id", validated_user_id).execute()
            
            if response.data and len(response.data) > 0:
                profile_data = response.data[0]
                logger.info(f"User profile found for user_id: {validated_user_id}")
                return UserProfile(**profile_data)
            
            logger.info(f"No user profile found for user_id: {validated_user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting user profile for {user_id}: {e}")
            return None  # Return None instead of raising exception for better error handling
    
    async def create_user_profile(self, profile: UserProfile) -> bool:
        """
        Create a new user profile - FIXED VERSION
        """
        try:
            # Validate the user_id in the profile
            is_valid, validated_user_id = self._validate_and_format_uuid(profile.user_id)
            if not is_valid:
                logger.error(f"Invalid user_id in profile: {validated_user_id}")
                return False
            
            logger.info(f"Creating user profile for user_id: {validated_user_id}")
            
            # Convert profile to dict and handle datetime fields
            profile_dict = profile.dict()
            profile_dict["user_id"] = validated_user_id  # Use validated user_id
            profile_dict["created_at"] = datetime.utcnow().isoformat()
            profile_dict["updated_at"] = datetime.utcnow().isoformat()
            
            # Convert date fields to string if present
            if profile_dict.get("birthdate"):
                profile_dict["birthdate"] = profile_dict["birthdate"].isoformat()
            
            # Convert list fields to JSON if present
            if "holiday_preferences" in profile_dict and profile_dict["holiday_preferences"]:
                profile_dict["holiday_preferences"] = json.dumps(profile_dict["holiday_preferences"])
            
            response = self.client.table("user_profiles").insert(profile_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User profile created successfully for user_id: {validated_user_id}")
            else:
                logger.warning(f"Failed to create user profile for user_id: {validated_user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error creating user profile for {profile.user_id}: {e}")
            return False  # Return False instead of raising exception
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update user profile - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.error(f"Invalid user_id format in update_user_profile: {validated_user_id}")
                return False
            
            logger.info(f"Updating user profile for user_id: {validated_user_id}")
            
            # Add updated_at timestamp
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            # Handle special fields
            if "birthdate" in updates and updates["birthdate"]:
                if isinstance(updates["birthdate"], date):
                    updates["birthdate"] = updates["birthdate"].isoformat()
            
            if "holiday_preferences" in updates and updates["holiday_preferences"]:
                if isinstance(updates["holiday_preferences"], list):
                    updates["holiday_preferences"] = json.dumps(updates["holiday_preferences"])
            
            response = self.client.table("user_profiles").update(updates).eq("user_id", validated_user_id).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User profile updated successfully for user_id: {validated_user_id}")
            else:
                logger.warning(f"No user profile found to update for user_id: {validated_user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating user profile for {user_id}: {e}")
            return False  # Return False instead of raising exception
    
    async def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete user profile (soft delete by setting is_active to False) - FIXED VERSION
        """
        try:
            logger.info(f"Soft deleting user profile for user_id: {user_id}")
            
            updates = {
                "is_active": False,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            return await self.update_user_profile(user_id, updates)
            
        except Exception as e:
            logger.error(f"Error deleting user profile for {user_id}: {e}")
            return False
    
    async def complete_onboarding(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Mark onboarding as complete and update profile - FIXED VERSION
        """
        try:
            logger.info(f"Completing onboarding for user_id: {user_id}")
            
            # Mark onboarding as completed
            profile_data["onboarding_completed"] = True
            profile_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Update user profile
            success = await self.update_user_profile(user_id, profile_data)
            
            if success:
                # Update onboarding progress to completed
                await self.update_onboarding_progress(user_id, OnboardingStep.COMPLETED, profile_data)
                logger.info(f"Onboarding completed successfully for user_id: {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error completing onboarding for {user_id}: {e}")
            return False
    
    # ==================== ONBOARDING OPERATIONS - FIXED ====================
    
    async def get_onboarding_progress(self, user_id: str) -> Optional[OnboardingProgress]:
        """
        Get onboarding progress for user - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_onboarding_progress: {validated_user_id}")
                return None
            
            logger.info(f"Getting onboarding progress for user_id: {validated_user_id}")
            
            response = self.client.table("onboarding_progress").select("*").eq("user_id", validated_user_id).execute()
            
            if response.data and len(response.data) > 0:
                progress_data = response.data[0]
                logger.info(f"Onboarding progress found for user_id: {validated_user_id}")
                return OnboardingProgress(**progress_data)
            
            logger.info(f"No onboarding progress found for user_id: {validated_user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting onboarding progress for {user_id}: {e}")
            return None  # Return None instead of raising exception
    
    async def update_onboarding_progress(self, user_id: str, step: OnboardingStep, data: Dict[str, Any]) -> bool:
        """
        Update onboarding progress - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.error(f"Invalid user_id format in update_onboarding_progress: {validated_user_id}")
                return False
            
            logger.info(f"Updating onboarding progress for user_id: {validated_user_id}, step: {step.value}")
            
            progress_data = {
                "user_id": validated_user_id,  # Use validated user_id
                "current_step": step.value,
                "collected_data": json.dumps(data),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Try to update first, if no rows affected, insert
            response = self.client.table("onboarding_progress").update(progress_data).eq("user_id", validated_user_id).execute()
            
            if not response.data:
                # No existing record, create new one
                progress_data["created_at"] = datetime.utcnow().isoformat()
                response = self.client.table("onboarding_progress").insert(progress_data).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Onboarding progress updated successfully for user_id: {validated_user_id}")
            else:
                logger.error(f"Failed to update onboarding progress for user_id: {validated_user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating onboarding progress for {user_id}: {e}")
            return False  # Return False instead of raising exception
    
    # ==================== GROUP OPERATIONS - FIXED ====================
    
    async def get_group_members(self, group_code: str) -> List[GroupMember]:
        """
        Get all members of a group
        """
        try:
            logger.info(f"Getting group members for group_code: {group_code}")
            
            response = self.client.table("group_members").select("*").eq("group_code", group_code).eq("is_active", True).execute()
            
            members = [GroupMember(**member) for member in response.data]
            logger.info(f"Found {len(members)} members for group_code: {group_code}")
            
            return members
            
        except Exception as e:
            logger.error(f"Error getting group members for {group_code}: {e}")
            return []  # Return empty list instead of raising exception
    
    async def add_group_member(self, group_code: str, user_id: str, role: str = "member") -> bool:
        """
        Add user to a group - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.error(f"Invalid user_id format in add_group_member: {validated_user_id}")
                return False
            
            logger.info(f"Adding user {validated_user_id} to group {group_code} with role {role}")
            
            member_data = {
                "id": str(uuid.uuid4()),
                "group_code": group_code.upper(),
                "user_id": validated_user_id,  # Use validated user_id
                "role": role,
                "joined_at": datetime.utcnow().isoformat(),
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("group_members").insert(member_data).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User {validated_user_id} added to group {group_code} successfully")
            else:
                logger.warning(f"Failed to add user {validated_user_id} to group {group_code}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding group member {user_id} to {group_code}: {e}")
            # Check if it's a duplicate entry error
            if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                logger.info(f"User {user_id} is already a member of group {group_code}")
                return True  # Consider it successful if already a member
            return False
    
    async def remove_group_member(self, group_code: str, user_id: str) -> bool:
        """
        Remove user from a group (soft delete) - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.error(f"Invalid user_id format in remove_group_member: {validated_user_id}")
                return False
            
            logger.info(f"Removing user {validated_user_id} from group {group_code}")
            
            updates = {
                "is_active": False,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("group_members").update(updates).eq("group_code", group_code).eq("user_id", validated_user_id).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User {validated_user_id} removed from group {group_code} successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error removing group member {user_id} from {group_code}: {e}")
            return False
    
    async def check_group_exists(self, group_code: str) -> bool:
        """
        Check if group code exists
        """
        try:
            logger.info(f"Checking if group exists: {group_code}")
            
            response = self.client.table("group_members").select("group_code").eq("group_code", group_code.upper()).eq("is_active", True).limit(1).execute()
            
            exists = len(response.data) > 0
            logger.info(f"Group {group_code} exists: {exists}")
            
            return exists
            
        except Exception as e:
            logger.error(f"Error checking group existence for {group_code}: {e}")
            return False
    
    async def get_existing_group_codes(self) -> set:
        """
        Get all existing group codes
        """
        try:
            logger.info("Getting all existing group codes")
            
            response = self.client.table("group_members").select("group_code").eq("is_active", True).execute()
            
            codes = {member["group_code"] for member in response.data}
            logger.info(f"Found {len(codes)} existing group codes")
            
            return codes
            
        except Exception as e:
            logger.error(f"Error getting existing group codes: {e}")
            return set()
    
    async def get_user_groups(self, user_id: str) -> List[str]:
        """
        Get all groups a user belongs to - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_user_groups: {validated_user_id}")
                return []
            
            logger.info(f"Getting groups for user: {validated_user_id}")
            
            response = self.client.table("group_members").select("group_code").eq("user_id", validated_user_id).eq("is_active", True).execute()
            
            groups = [member["group_code"] for member in response.data]
            logger.info(f"User {validated_user_id} belongs to {len(groups)} groups")
            
            return groups
            
        except Exception as e:
            logger.error(f"Error getting user groups for {user_id}: {e}")
            return []  # Return empty list instead of raising exception
    
    # ==================== USER STATUS CHECK - FIXED ====================
    
    async def get_user_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user status including onboarding completion - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_user_status: {validated_user_id}")
                return None
            
            logger.info(f"Getting user status for user: {validated_user_id}")
            
            profile = await self.get_user_profile(validated_user_id)
            if not profile:
                logger.warning(f"No profile found for user: {validated_user_id}")
                return None
            
            # Get onboarding progress
            onboarding = await self.get_onboarding_progress(validated_user_id)
            
            # Get user groups
            groups = await self.get_user_groups(validated_user_id)
            
            status = {
                "user_id": validated_user_id,
                "email": profile.email,
                "onboarding_completed": profile.onboarding_completed,
                "is_active": profile.is_active,
                "current_step": onboarding.current_step.value if onboarding else "group_code",
                "completion_percentage": onboarding.completion_percentage if onboarding else 0.0,
                "groups": groups,
                "profile": profile.dict()
            }
            
            logger.info(f"User status retrieved for user: {validated_user_id}")
            return status
            
        except Exception as e:
            logger.error(f"Error getting user status for {user_id}: {e}")
            return None  # Return None instead of raising exception
    
    # ==================== CHAT OPERATIONS ====================
    
    async def save_chat_message(self, message: ChatMessage) -> bool:
        """
        Save a chat message
        """
        try:
            logger.info(f"Saving chat message for user: {message.user_id}")
            
            message_dict = message.dict()
            message_dict["created_at"] = datetime.utcnow().isoformat()
            
            # Convert metadata to JSON if present
            if message_dict.get("metadata"):
                message_dict["metadata"] = json.dumps(message_dict["metadata"])
            
            response = self.client.table("chat_messages").insert(message_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Chat message saved successfully for user: {message.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving chat message for {message.user_id}: {e}")
            return False
    
    async def get_conversation_history(self, user_id: str, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """
        Get conversation history for a session - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid_user, validated_user_id = self._validate_and_format_uuid(user_id)
            is_valid_session, validated_session_id = self._validate_and_format_uuid(session_id)
            
            if not is_valid_user or not is_valid_session:
                logger.warning(f"Invalid UUID format in get_conversation_history")
                return []
            
            logger.info(f"Getting conversation history for user: {validated_user_id}, session: {validated_session_id}")
            
            response = (self.client.table("chat_messages")
                       .select("*")
                       .eq("user_id", validated_user_id)
                       .eq("session_id", validated_session_id)
                       .order("created_at", desc=False)
                       .limit(limit)
                       .execute())
            
            messages = []
            for msg_data in response.data:
                # Parse metadata JSON if present
                if msg_data.get("metadata") and isinstance(msg_data["metadata"], str):
                    try:
                        msg_data["metadata"] = json.loads(msg_data["metadata"])
                    except json.JSONDecodeError:
                        msg_data["metadata"] = {}
                
                messages.append(ChatMessage(**msg_data))
            
            logger.info(f"Retrieved {len(messages)} messages for session: {validated_session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history for {user_id}: {e}")
            return []  # Return empty list instead of raising exception
    
    async def save_conversation_turn(self, user_id: str, session_id: str, user_message: str, 
                                   assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a complete conversation turn (user message + assistant response)
        """
        try:
            # Save user message
            user_msg = ChatMessage(
                id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                message_type=MessageType.USER,
                content=user_message,
                metadata=metadata or {}
            )
            
            # Save assistant message
            assistant_msg = ChatMessage(
                id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                message_type=MessageType.ASSISTANT,
                content=assistant_response,
                metadata=metadata or {}
            )
            
            # Save both messages
            user_saved = await self.save_chat_message(user_msg)
            assistant_saved = await self.save_chat_message(assistant_msg)
            
            success = user_saved and assistant_saved
            if success:
                logger.info(f"Conversation turn saved successfully for user: {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving conversation turn for {user_id}: {e}")
            return False
    
    # ==================== SEARCH OPERATIONS ====================
    
    async def save_flight_search(self, search: FlightSearch) -> bool:
        """
        Save flight search results
        """
        try:
            logger.info(f"Saving flight search for user: {search.user_id}")
            
            search_dict = search.dict()
            search_dict["created_at"] = datetime.utcnow().isoformat()
            
            # Convert date fields to strings
            if search_dict.get("departure_date"):
                search_dict["departure_date"] = search_dict["departure_date"].isoformat()
            if search_dict.get("return_date"):
                search_dict["return_date"] = search_dict["return_date"].isoformat()
            
            # Convert search results to JSON
            search_dict["search_results"] = json.dumps(search_dict["search_results"])
            
            response = self.client.table("flight_searches").insert(search_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Flight search saved successfully for user: {search.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving flight search for {search.user_id}: {e}")
            return False
    
    async def save_hotel_search(self, search: HotelSearch) -> bool:
        """
        Save hotel search results
        """
        try:
            logger.info(f"Saving hotel search for user: {search.user_id}")
            
            search_dict = search.dict()
            search_dict["created_at"] = datetime.utcnow().isoformat()
            
            # Convert date fields to strings
            if search_dict.get("check_in_date"):
                search_dict["check_in_date"] = search_dict["check_in_date"].isoformat()
            if search_dict.get("check_out_date"):
                search_dict["check_out_date"] = search_dict["check_out_date"].isoformat()
            
            # Convert search results to JSON
            search_dict["search_results"] = json.dumps(search_dict["search_results"])
            
            response = self.client.table("hotel_searches").insert(search_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Hotel search saved successfully for user: {search.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving hotel search for {search.user_id}: {e}")
            return False
    
    # ==================== USER PREFERENCES OPERATIONS ====================
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """
        Get user preferences - FIXED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id)
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_user_preferences: {validated_user_id}")
                return None
            
            logger.info(f"Getting user preferences for user: {validated_user_id}")
            
            response = self.client.table("user_preferences").select("*").eq("user_id", validated_user_id).execute()
            
            if response.data and len(response.data) > 0:
                prefs_data = response.data[0]
                
                # Parse JSON fields
                for field in ["preferred_airlines", "preferred_hotels", "dietary_restrictions", "accessibility_needs", "trip_types"]:
                    if prefs_data.get(field) and isinstance(prefs_data[field], str):
                        try:
                            prefs_data[field] = json.loads(prefs_data[field])
                        except json.JSONDecodeError:
                            prefs_data[field] = []
                
                if prefs_data.get("budget_range") and isinstance(prefs_data["budget_range"], str):
                    try:
                        prefs_data["budget_range"] = json.loads(prefs_data["budget_range"])
                    except json.JSONDecodeError:
                        prefs_data["budget_range"] = {}
                
                return UserPreferences(**prefs_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user preferences for {user_id}: {e}")
            return None
    
    async def save_user_preferences(self, preferences: UserPreferences) -> bool:
        """
        Save or update user preferences
        """
        try:
            logger.info(f"Saving user preferences for user: {preferences.user_id}")
            
            prefs_dict = preferences.dict()
            prefs_dict["updated_at"] = datetime.utcnow().isoformat()
            
            # Convert list and dict fields to JSON
            for field in ["preferred_airlines", "preferred_hotels", "dietary_restrictions", "accessibility_needs", "trip_types"]:
                if prefs_dict.get(field):
                    prefs_dict[field] = json.dumps(prefs_dict[field])
            
            if prefs_dict.get("budget_range"):
                prefs_dict["budget_range"] = json.dumps(prefs_dict["budget_range"])
            
            # Try to update first, if no rows affected, insert
            response = self.client.table("user_preferences").update(prefs_dict).eq("user_id", preferences.user_id).execute()
            
            if not response.data:
                # No existing record, create new one
                prefs_dict["created_at"] = datetime.utcnow().isoformat()
                response = self.client.table("user_preferences").insert(prefs_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User preferences saved successfully for user: {preferences.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving user preferences for {preferences.user_id}: {e}")
            return False
    
    # ==================== DATABASE HEALTH AND MAINTENANCE ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection
        """
        try:
            logger.info("Performing database health check")
            
            # Test basic connectivity
            response = self.client.table("user_profiles").select("user_id").limit(1).execute()
            
            # Get table counts (approximate)
            tables_status = {}
            table_configs = {
                "user_profiles": "user_id",
                "chat_messages": "id",
                "group_members": "id",
                "flight_searches": "search_id",
                "hotel_searches": "id",
                "onboarding_progress": "user_id",
                "user_preferences": "user_id"
            }

            for table, primary_key in table_configs.items():
                try:
                    count_response = self.client.table(table).select(primary_key, count="exact").limit(1).execute()
                    tables_status[table] = "healthy"
                except Exception as e:
                    tables_status[table] = f"error: {str(e)}"
            
            health = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "connected",
                "tables": tables_status
            }
            
            logger.info("Database health check completed successfully")
            return health
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "disconnected",
                "error": str(e)
            }

# ==================== MODULE EXPORTS ====================

__all__ = [
    'DatabaseOperations'
]
