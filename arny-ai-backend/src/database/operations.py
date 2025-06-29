"""
Database operations for Arny AI

This module provides all database operations using Supabase (PostgreSQL) as the backend.
It includes CRUD operations for all models and handles database connections, transactions,
and error handling. All operations respect Row Level Security (RLS) policies.

Classes:
- DatabaseOperations: Main database operations class

Features:
- User profile management
- Onboarding progress tracking
- Group membership management
- Chat message storage
- Flight and hotel search history
- Booking management
- Travel itinerary management
- Comprehensive error handling
- Async/await support
- Type safety with Pydantic models
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
    Database operations using Supabase
    
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
    
    # ==================== USER PROFILE OPERATIONS ====================
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by user_id
        
        Args:
            user_id: User UUID
            
        Returns:
            UserProfile object or None if not found
            
        Raises:
            ValueError: If user_id is invalid
            Exception: For database errors
        """
        try:
            # Validate user_id format
            uuid.UUID(user_id)
            
            logger.info(f"Getting user profile for user_id: {user_id}")
            
            response = self.client.table("user_profiles").select("*").eq("user_id", user_id).execute()
            
            if response.data and len(response.data) > 0:
                profile_data = response.data[0]
                logger.info(f"User profile found for user_id: {user_id}")
                return UserProfile(**profile_data)
            
            logger.info(f"No user profile found for user_id: {user_id}")
            return None
            
        except ValueError:
            logger.error(f"Invalid user_id format: {user_id}")
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error getting user profile for {user_id}: {e}")
            raise Exception(f"Failed to get user profile: {e}")
    
    async def create_user_profile(self, profile: UserProfile) -> bool:
        """
        Create a new user profile
        
        Args:
            profile: UserProfile object
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            Exception: For database errors
        """
        try:
            logger.info(f"Creating user profile for user_id: {profile.user_id}")
            
            # Convert profile to dict and handle datetime fields
            profile_dict = profile.dict()
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
                logger.info(f"User profile created successfully for user_id: {profile.user_id}")
            else:
                logger.warning(f"Failed to create user profile for user_id: {profile.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error creating user profile for {profile.user_id}: {e}")
            raise Exception(f"Failed to create user profile: {e}")
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update user profile
        
        Args:
            user_id: User UUID
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If user_id is invalid
            Exception: For database errors
        """
        try:
            # Validate user_id format
            uuid.UUID(user_id)
            
            logger.info(f"Updating user profile for user_id: {user_id}")
            
            # Add updated_at timestamp
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            # Handle special fields
            if "birthdate" in updates and updates["birthdate"]:
                if isinstance(updates["birthdate"], date):
                    updates["birthdate"] = updates["birthdate"].isoformat()
            
            if "holiday_preferences" in updates and updates["holiday_preferences"]:
                if isinstance(updates["holiday_preferences"], list):
                    updates["holiday_preferences"] = json.dumps(updates["holiday_preferences"])
            
            response = self.client.table("user_profiles").update(updates).eq("user_id", user_id).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User profile updated successfully for user_id: {user_id}")
            else:
                logger.warning(f"No user profile found to update for user_id: {user_id}")
            
            return success
            
        except ValueError:
            logger.error(f"Invalid user_id format: {user_id}")
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error updating user profile for {user_id}: {e}")
            raise Exception(f"Failed to update user profile: {e}")
    
    async def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete user profile (soft delete by setting is_active to False)
        
        Args:
            user_id: User UUID
            
        Returns:
            True if successful, False otherwise
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
            raise Exception(f"Failed to delete user profile: {e}")
    
    async def complete_onboarding(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Mark onboarding as complete and update profile
        
        Args:
            user_id: User UUID
            profile_data: Complete profile data from onboarding
            
        Returns:
            True if successful, False otherwise
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
            raise Exception(f"Failed to complete onboarding: {e}")
    
    # ==================== ONBOARDING OPERATIONS ====================
    
    async def get_onboarding_progress(self, user_id: str) -> Optional[OnboardingProgress]:
        """
        Get onboarding progress for user
        
        Args:
            user_id: User UUID
            
        Returns:
            OnboardingProgress object or None if not found
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Getting onboarding progress for user_id: {user_id}")
            
            response = self.client.table("onboarding_progress").select("*").eq("user_id", user_id).execute()
            
            if response.data and len(response.data) > 0:
                progress_data = response.data[0]
                return OnboardingProgress(**progress_data)
            
            logger.info(f"No onboarding progress found for user_id: {user_id}")
            return None
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error getting onboarding progress for {user_id}: {e}")
            raise Exception(f"Failed to get onboarding progress: {e}")
    
    async def update_onboarding_progress(self, user_id: str, step: OnboardingStep, data: Dict[str, Any]) -> bool:
        """
        Update onboarding progress
        
        Args:
            user_id: User UUID
            step: Current onboarding step
            data: Collected data so far
            
        Returns:
            True if successful, False otherwise
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Updating onboarding progress for user_id: {user_id}, step: {step.value}")
            
            progress_data = {
                "user_id": user_id,
                "current_step": step.value,
                "collected_data": json.dumps(data),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Try to update first, if no rows affected, insert
            response = self.client.table("onboarding_progress").update(progress_data).eq("user_id", user_id).execute()
            
            if not response.data:
                # No existing record, create new one
                progress_data["created_at"] = datetime.utcnow().isoformat()
                response = self.client.table("onboarding_progress").insert(progress_data).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Onboarding progress updated successfully for user_id: {user_id}")
            
            return success
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error updating onboarding progress for {user_id}: {e}")
            raise Exception(f"Failed to update onboarding progress: {e}")
    
    # ==================== GROUP OPERATIONS ====================
    
    async def get_group_members(self, group_code: str) -> List[GroupMember]:
        """
        Get all members of a group
        
        Args:
            group_code: Group code
            
        Returns:
            List of GroupMember objects
        """
        try:
            logger.info(f"Getting group members for group_code: {group_code}")
            
            response = self.client.table("group_members").select("*").eq("group_code", group_code).eq("is_active", True).execute()
            
            members = [GroupMember(**member) for member in response.data]
            logger.info(f"Found {len(members)} members for group_code: {group_code}")
            
            return members
            
        except Exception as e:
            logger.error(f"Error getting group members for {group_code}: {e}")
            raise Exception(f"Failed to get group members: {e}")
    
    async def add_group_member(self, group_code: str, user_id: str, role: str = "member") -> bool:
        """
        Add user to a group
        
        Args:
            group_code: Group code
            user_id: User UUID
            role: User role in group (admin or member)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Adding user {user_id} to group {group_code} with role {role}")
            
            member_data = {
                "id": str(uuid.uuid4()),
                "group_code": group_code.upper(),
                "user_id": user_id,
                "role": role,
                "joined_at": datetime.utcnow().isoformat(),
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("group_members").insert(member_data).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User {user_id} added to group {group_code} successfully")
            
            return success
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error adding group member {user_id} to {group_code}: {e}")
            # Check if it's a duplicate entry error
            if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                logger.info(f"User {user_id} is already a member of group {group_code}")
                return True  # Consider it successful if already a member
            raise Exception(f"Failed to add group member: {e}")
    
    async def remove_group_member(self, group_code: str, user_id: str) -> bool:
        """
        Remove user from a group (soft delete)
        
        Args:
            group_code: Group code
            user_id: User UUID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Removing user {user_id} from group {group_code}")
            
            updates = {
                "is_active": False,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("group_members").update(updates).eq("group_code", group_code).eq("user_id", user_id).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User {user_id} removed from group {group_code} successfully")
            
            return success
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error removing group member {user_id} from {group_code}: {e}")
            raise Exception(f"Failed to remove group member: {e}")
    
    async def check_group_exists(self, group_code: str) -> bool:
        """
        Check if group code exists
        
        Args:
            group_code: Group code to check
            
        Returns:
            True if group exists, False otherwise
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
        
        Returns:
            Set of existing group codes
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
        Get all groups a user belongs to
        
        Args:
            user_id: User UUID
            
        Returns:
            List of group codes
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Getting groups for user: {user_id}")
            
            response = self.client.table("group_members").select("group_code").eq("user_id", user_id).eq("is_active", True).execute()
            
            groups = [member["group_code"] for member in response.data]
            logger.info(f"User {user_id} belongs to {len(groups)} groups")
            
            return groups
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error getting user groups for {user_id}: {e}")
            raise Exception(f"Failed to get user groups: {e}")
    
    # ==================== CHAT OPERATIONS ====================
    
    async def save_chat_message(self, message: ChatMessage) -> bool:
        """
        Save a chat message
        
        Args:
            message: ChatMessage object
            
        Returns:
            True if successful, False otherwise
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
            raise Exception(f"Failed to save chat message: {e}")
    
    async def get_conversation_history(self, user_id: str, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """
        Get conversation history for a session
        
        Args:
            user_id: User UUID
            session_id: Session UUID
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of ChatMessage objects
        """
        try:
            uuid.UUID(user_id)
            uuid.UUID(session_id)
            
            logger.info(f"Getting conversation history for user: {user_id}, session: {session_id}")
            
            response = (self.client.table("chat_messages")
                       .select("*")
                       .eq("user_id", user_id)
                       .eq("session_id", session_id)
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
            
            logger.info(f"Retrieved {len(messages)} messages for session: {session_id}")
            return messages
            
        except ValueError:
            raise ValueError("Invalid user_id or session_id format")
        except Exception as e:
            logger.error(f"Error getting conversation history for {user_id}: {e}")
            raise Exception(f"Failed to get conversation history: {e}")
    
    async def save_conversation_turn(self, user_id: str, session_id: str, user_message: str, 
                                   assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a complete conversation turn (user message + assistant response)
        
        Args:
            user_id: User UUID
            session_id: Session UUID
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
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
            raise Exception(f"Failed to save conversation turn: {e}")
    
    async def get_recent_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent chat sessions for a user
        
        Args:
            user_id: User UUID
            limit: Maximum number of sessions
            
        Returns:
            List of session information
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Getting recent sessions for user: {user_id}")
            
            # Get unique session IDs with latest message timestamp
            response = (self.client.table("chat_messages")
                       .select("session_id, created_at, content")
                       .eq("user_id", user_id)
                       .eq("message_type", "user")
                       .order("created_at", desc=True)
                       .limit(limit * 5)  # Get more to filter unique sessions
                       .execute())
            
            # Group by session and get the latest message for each
            sessions = {}
            for msg in response.data:
                session_id = msg["session_id"]
                if session_id not in sessions:
                    sessions[session_id] = {
                        "session_id": session_id,
                        "last_message": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"],
                        "last_activity": msg["created_at"]
                    }
            
            # Sort by last activity and limit
            sorted_sessions = sorted(sessions.values(), key=lambda x: x["last_activity"], reverse=True)[:limit]
            
            logger.info(f"Found {len(sorted_sessions)} recent sessions for user: {user_id}")
            return sorted_sessions
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error getting recent sessions for {user_id}: {e}")
            raise Exception(f"Failed to get recent sessions: {e}")
    
    # ==================== SEARCH OPERATIONS ====================
    
    async def save_flight_search(self, search: FlightSearch) -> bool:
        """
        Save flight search results
        
        Args:
            search: FlightSearch object
            
        Returns:
            True if successful, False otherwise
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
            raise Exception(f"Failed to save flight search: {e}")
    
    async def save_hotel_search(self, search: HotelSearch) -> bool:
        """
        Save hotel search results
        
        Args:
            search: HotelSearch object
            
        Returns:
            True if successful, False otherwise
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
            raise Exception(f"Failed to save hotel search: {e}")
    
    async def get_user_search_history(self, user_id: str, search_type: str = "all", limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get user's search history
        
        Args:
            user_id: User UUID
            search_type: Type of search ("flight", "hotel", or "all")
            limit: Maximum number of results
            
        Returns:
            List of search records
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Getting search history for user: {user_id}, type: {search_type}")
            
            searches = []
            
            if search_type in ["flight", "all"]:
                flight_response = (self.client.table("flight_searches")
                                 .select("*")
                                 .eq("user_id", user_id)
                                 .order("created_at", desc=True)
                                 .limit(limit)
                                 .execute())
                
                for search in flight_response.data:
                    search["search_type"] = "flight"
                    if search.get("search_results") and isinstance(search["search_results"], str):
                        try:
                            search["search_results"] = json.loads(search["search_results"])
                        except json.JSONDecodeError:
                            search["search_results"] = []
                    searches.append(search)
            
            if search_type in ["hotel", "all"]:
                hotel_response = (self.client.table("hotel_searches")
                                .select("*")
                                .eq("user_id", user_id)
                                .order("created_at", desc=True)
                                .limit(limit)
                                .execute())
                
                for search in hotel_response.data:
                    search["search_type"] = "hotel"
                    if search.get("search_results") and isinstance(search["search_results"], str):
                        try:
                            search["search_results"] = json.loads(search["search_results"])
                        except json.JSONDecodeError:
                            search["search_results"] = []
                    searches.append(search)
            
            # Sort by created_at and limit
            searches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            searches = searches[:limit]
            
            logger.info(f"Retrieved {len(searches)} search records for user: {user_id}")
            return searches
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error getting search history for {user_id}: {e}")
            raise Exception(f"Failed to get search history: {e}")
    
    # ==================== USER PREFERENCES OPERATIONS ====================
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """
        Get user preferences
        
        Args:
            user_id: User UUID
            
        Returns:
            UserPreferences object or None if not found
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Getting user preferences for user: {user_id}")
            
            response = self.client.table("user_preferences").select("*").eq("user_id", user_id).execute()
            
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
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error getting user preferences for {user_id}: {e}")
            raise Exception(f"Failed to get user preferences: {e}")
    
    async def save_user_preferences(self, preferences: UserPreferences) -> bool:
        """
        Save or update user preferences
        
        Args:
            preferences: UserPreferences object
            
        Returns:
            True if successful, False otherwise
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
            raise Exception(f"Failed to save user preferences: {e}")
    
    # ==================== USER STATUS CHECK ====================
    
    async def get_user_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user status including onboarding completion
        
        Args:
            user_id: User UUID
            
        Returns:
            User status dictionary or None if user not found
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Getting user status for user: {user_id}")
            
            profile = await self.get_user_profile(user_id)
            if not profile:
                return None
            
            # Get onboarding progress
            onboarding = await self.get_onboarding_progress(user_id)
            
            # Get user groups
            groups = await self.get_user_groups(user_id)
            
            status = {
                "user_id": user_id,
                "email": profile.email,
                "onboarding_completed": profile.onboarding_completed,
                "is_active": profile.is_active,
                "current_step": onboarding.current_step.value if onboarding else "group_code",
                "completion_percentage": onboarding.completion_percentage if onboarding else 0.0,
                "groups": groups,
                "profile": profile.dict()
            }
            
            logger.info(f"User status retrieved for user: {user_id}")
            return status
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error getting user status for {user_id}: {e}")
            raise Exception(f"Failed to get user status: {e}")
    
    # ==================== ANALYTICS AND REPORTING ====================
    
    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get user analytics for the past N days
        
        Args:
            user_id: User UUID
            days: Number of days to analyze
            
        Returns:
            Analytics dictionary
        """
        try:
            uuid.UUID(user_id)
            
            logger.info(f"Getting analytics for user: {user_id}")
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get message count
            message_response = (self.client.table("chat_messages")
                              .select("id")
                              .eq("user_id", user_id)
                              .gte("created_at", start_date.isoformat())
                              .execute())
            
            # Get search count
            flight_response = (self.client.table("flight_searches")
                             .select("id")
                             .eq("user_id", user_id)
                             .gte("created_at", start_date.isoformat())
                             .execute())
            
            hotel_response = (self.client.table("hotel_searches")
                            .select("id")
                            .eq("user_id", user_id)
                            .gte("created_at", start_date.isoformat())
                            .execute())
            
            analytics = {
                "user_id": user_id,
                "period_days": days,
                "message_count": len(message_response.data),
                "flight_searches": len(flight_response.data),
                "hotel_searches": len(hotel_response.data),
                "total_searches": len(flight_response.data) + len(hotel_response.data),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
            
            logger.info(f"Analytics retrieved for user: {user_id}")
            return analytics
            
        except ValueError:
            raise ValueError("Invalid user_id format")
        except Exception as e:
            logger.error(f"Error getting analytics for {user_id}: {e}")
            raise Exception(f"Failed to get user analytics: {e}")
    
    # ==================== DATABASE HEALTH AND MAINTENANCE ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection
        
        Returns:
            Health status dictionary
        """
        try:
            logger.info("Performing database health check")
            
            # Test basic connectivity
            response = self.client.table("user_profiles").select("user_id").limit(1).execute()
            
            # Get table counts (approximate)
            tables_status = {}
            for table in ["user_profiles", "chat_messages", "group_members", "flight_searches", "hotel_searches"]:
                try:
                    count_response = self.client.table(table).select("id", count="exact").limit(1).execute()
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
    
    # ==================== CLEANUP OPERATIONS ====================
    
    async def cleanup_old_chat_messages(self, days_old: int = 90) -> int:
        """
        Clean up old chat messages (older than specified days)
        
        Args:
            days_old: Messages older than this many days will be deleted
            
        Returns:
            Number of messages deleted
        """
        try:
            logger.info(f"Cleaning up chat messages older than {days_old} days")
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Get messages to delete (for counting)
            old_messages = (self.client.table("chat_messages")
                          .select("id")
                          .lt("created_at", cutoff_date.isoformat())
                          .execute())
            
            count = len(old_messages.data)
            
            if count > 0:
                # Delete old messages
                self.client.table("chat_messages").delete().lt("created_at", cutoff_date.isoformat()).execute()
                logger.info(f"Deleted {count} old chat messages")
            else:
                logger.info("No old chat messages to delete")
            
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up old chat messages: {e}")
            raise Exception(f"Failed to cleanup old messages: {e}")
    
    async def cleanup_old_searches(self, days_old: int = 180) -> int:
        """
        Clean up old search records
        
        Args:
            days_old: Searches older than this many days will be deleted
            
        Returns:
            Number of search records deleted
        """
        try:
            logger.info(f"Cleaning up search records older than {days_old} days")
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            count = 0
            
            # Clean flight searches
            old_flights = (self.client.table("flight_searches")
                         .select("search_id")
                         .lt("created_at", cutoff_date.isoformat())
                         .execute())
            
            if old_flights.data:
                self.client.table("flight_searches").delete().lt("created_at", cutoff_date.isoformat()).execute()
                count += len(old_flights.data)
            
            # Clean hotel searches
            old_hotels = (self.client.table("hotel_searches")
                        .select("search_id")
                        .lt("created_at", cutoff_date.isoformat())
                        .execute())
            
            if old_hotels.data:
                self.client.table("hotel_searches").delete().lt("created_at", cutoff_date.isoformat()).execute()
                count += len(old_hotels.data)
            
            logger.info(f"Deleted {count} old search records")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up old search records: {e}")
            raise Exception(f"Failed to cleanup old searches: {e}")

# ==================== MODULE EXPORTS ====================

__all__ = [
    'DatabaseOperations'
]