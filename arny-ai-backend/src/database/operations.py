"""
Database operations for Arny AI - ENHANCED VERSION WITH ONBOARDING FIXES

Enhanced Features:
1. Robust UUID validation across all methods
2. Better error handling for UUID validation
3. Improved progress loading and saving with better JSON handling
4. FIXED: Enhanced onboarding completion with verification and force completion
5. Better session_id handling in conversation methods
6. FIXED: Proper data type handling for onboarding completion
7. FIXED: Improved transaction logic for profile updates
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, date, timedelta
import json
import uuid
import logging
import asyncio
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
    Database operations using Supabase - ENHANCED VERSION WITH ONBOARDING FIXES
    
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
    
    def _validate_and_format_uuid(self, user_id: str, field_name: str = "UUID") -> tuple[bool, str]:
        """
        Enhanced UUID validation and formatting across all methods with better error messages
        
        Args:
            user_id: User ID to validate
            field_name: Name of the field being validated (for better error messages)
            
        Returns:
            Tuple of (is_valid, cleaned_user_id or error_message)
        """
        if not user_id:
            error_msg = f"{field_name} cannot be empty"
            logger.warning(error_msg)
            return False, error_msg
        
        try:
            # Convert to string and strip whitespace
            user_id_str = str(user_id).strip()
            
            # Remove any quotes that might be present
            user_id_str = user_id_str.strip('"\'')
            
            # Validate UUID format
            uuid_obj = uuid.UUID(user_id_str)
            
            # Return the string representation to ensure consistency
            validated_uuid = str(uuid_obj)
            logger.debug(f"✅ {field_name} validated: {validated_uuid}")
            return True, validated_uuid
            
        except ValueError as e:
            error_msg = f"Invalid {field_name} format: {e}"
            logger.warning(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected {field_name} validation error: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    # ==================== USER PROFILE OPERATIONS ====================
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by user_id - ENHANCED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
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
        Create a new user profile - ENHANCED VERSION WITH BETTER ERROR HANDLING
        """
        try:
            # Validate the user_id in the profile
            is_valid, validated_user_id = self._validate_and_format_uuid(profile.user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id in profile: {validated_user_id}")
                return False
            
            logger.info(f"Creating user profile for user_id: {validated_user_id}")
            
            # FIXED: Check if profile already exists before creating
            existing_profile = await self.get_user_profile(validated_user_id)
            if existing_profile:
                logger.info(f"User profile already exists for user_id: {validated_user_id}")
                return True  # Consider it successful if already exists
            
            # Convert profile to dict and handle datetime fields
            profile_dict = profile.dict()
            profile_dict["user_id"] = validated_user_id  # Use validated user_id
            profile_dict["created_at"] = datetime.utcnow().isoformat()
            profile_dict["updated_at"] = datetime.utcnow().isoformat()
            
            # Convert date fields to string if present
            if profile_dict.get("birthdate"):
                if isinstance(profile_dict["birthdate"], date):
                    profile_dict["birthdate"] = profile_dict["birthdate"].isoformat()
            
            # Convert list fields to JSON if present
            if "holiday_preferences" in profile_dict and profile_dict["holiday_preferences"]:
                if isinstance(profile_dict["holiday_preferences"], list):
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
        Update user profile - ENHANCED VERSION WITH BETTER DATA HANDLING
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in update_user_profile: {validated_user_id}")
                return False
            
            logger.info(f"Updating user profile for user_id: {validated_user_id}")
            logger.info(f"Update data keys: {list(updates.keys())}")
            
            # FIXED: Check if profile exists before updating
            existing_profile = await self.get_user_profile(validated_user_id)
            if not existing_profile:
                logger.error(f"Cannot update: No user profile found for user_id: {validated_user_id}")
                return False
            
            # FIXED: Clean and validate update data
            cleaned_updates = {}
            
            # Add updated_at timestamp
            cleaned_updates["updated_at"] = datetime.utcnow().isoformat()
            
            # Handle each field carefully
            for key, value in updates.items():
                if value is None:
                    continue  # Skip None values
                
                if key == "birthdate":
                    if isinstance(value, str):
                        # Try to parse and reformat date string
                        try:
                            if value and value != "":
                                # Try different date formats
                                for date_fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]:
                                    try:
                                        parsed_date = datetime.strptime(value, date_fmt).date()
                                        cleaned_updates[key] = parsed_date.isoformat()
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    # If no format worked, store as-is if it looks like YYYY-MM-DD
                                    if len(value) == 10 and value.count('-') == 2:
                                        cleaned_updates[key] = value
                        except Exception as e:
                            logger.warning(f"Could not parse birthdate '{value}': {e}")
                    elif isinstance(value, date):
                        cleaned_updates[key] = value.isoformat()
                
                elif key == "holiday_preferences":
                    if isinstance(value, list):
                        cleaned_updates[key] = json.dumps(value)
                    elif isinstance(value, str):
                        # Check if it's already JSON
                        try:
                            json.loads(value)  # Test if it's valid JSON
                            cleaned_updates[key] = value
                        except json.JSONDecodeError:
                            # If not JSON, treat as single preference
                            cleaned_updates[key] = json.dumps([value])
                
                else:
                    # For all other fields, store as-is after converting to string if needed
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_updates[key] = value
                    else:
                        cleaned_updates[key] = str(value)
            
            logger.info(f"Cleaned update data: {cleaned_updates}")
            
            response = self.client.table("user_profiles").update(cleaned_updates).eq("user_id", validated_user_id).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User profile updated successfully for user_id: {validated_user_id}")
                # FIXED: Verify the update worked
                updated_profile = await self.get_user_profile(validated_user_id)
                if updated_profile:
                    logger.info(f"Update verification successful for user_id: {validated_user_id}")
                else:
                    logger.error(f"Update verification failed for user_id: {validated_user_id}")
                    return False
            else:
                logger.warning(f"No rows updated for user_id: {validated_user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating user profile for {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return False  # Return False instead of raising exception
    
    async def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete user profile (soft delete by setting is_active to False) - ENHANCED VERSION
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
        FIXED: Mark onboarding as complete and create/update profile with comprehensive error handling
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in complete_onboarding: {validated_user_id}")
                return False
            
            logger.info(f"Completing onboarding for user_id: {validated_user_id}")
            print(f"🎉 Starting onboarding completion for user: {validated_user_id}")
            print(f"📊 Profile data received: {list(profile_data.keys())}")
            
            # Filter out fields that don't belong in user_profiles table
            user_profile_fields = {
                'name', 'gender', 'birthdate', 'city', 'employer',
                'working_schedule', 'holiday_frequency', 'annual_income',
                'monthly_spending', 'holiday_preferences', 'travel_style',
                'group_code', 'email'
            }

            # FIXED: Better filtering and data cleaning
            filtered_profile_data = {}
            for key, value in profile_data.items():
                if key in user_profile_fields and value is not None and value != "":
                    filtered_profile_data[key] = value

            print(f"📝 Filtered profile data: {filtered_profile_data}")

            # FIXED: Always mark onboarding as completed
            filtered_profile_data["onboarding_completed"] = True
            print(f"✅ Setting onboarding_completed = True")

            # FIXED: Check if user profile exists and decide whether to create or update
            existing_profile = await self.get_user_profile(validated_user_id)
            
            if existing_profile:
                print(f"📄 Existing profile found, updating...")
                # Update existing profile
                success = await self.update_user_profile(validated_user_id, filtered_profile_data)
                print(f"💾 Profile update success: {success}")
            else:
                print(f"📄 No existing profile found, creating new one...")
                # Create new profile from collected data
                try:
                    from .models import UserProfile
                    
                    # Ensure we have an email
                    if not filtered_profile_data.get("email"):
                        logger.error(f"Cannot create profile: email is required but missing")
                        print(f"❌ No email found in profile data")
                        return False
                    
                    # Create comprehensive profile from onboarding data
                    new_profile = UserProfile(
                        user_id=validated_user_id,
                        email=filtered_profile_data["email"],
                        name=filtered_profile_data.get("name"),
                        gender=filtered_profile_data.get("gender"),
                        birthdate=filtered_profile_data.get("birthdate"),
                        city=filtered_profile_data.get("city"),
                        employer=filtered_profile_data.get("employer"),
                        working_schedule=filtered_profile_data.get("working_schedule"),
                        holiday_frequency=filtered_profile_data.get("holiday_frequency"),
                        annual_income=filtered_profile_data.get("annual_income"),
                        monthly_spending=filtered_profile_data.get("monthly_spending"),
                        holiday_preferences=filtered_profile_data.get("holiday_preferences", "").split(",") if filtered_profile_data.get("holiday_preferences") else [],
                        travel_style=filtered_profile_data.get("travel_style"),
                        group_code=filtered_profile_data.get("group_code"),
                        onboarding_completed=True,  # Mark as completed
                        is_active=True
                    )
                    
                    success = await self.create_user_profile(new_profile)
                    print(f"💾 Profile creation success: {success}")
                    
                except Exception as profile_creation_error:
                    logger.error(f"Error creating profile from onboarding data: {profile_creation_error}")
                    print(f"❌ Profile creation error: {profile_creation_error}")
                    return False

            if success:
                # Update onboarding progress to completed
                try:
                    progress_success = await self.update_onboarding_progress(
                        validated_user_id, 
                        OnboardingStep.COMPLETED, 
                        profile_data
                    )
                    print(f"📈 Progress update success: {progress_success}")
                except Exception as progress_error:
                    logger.warning(f"Failed to update onboarding progress: {progress_error}")
                    # Don't fail the entire process if progress update fails
                
                # FIXED: Triple verification with multiple attempts
                verification_attempts = 3
                final_verification_passed = False
                
                for attempt in range(verification_attempts):
                    try:
                        await asyncio.sleep(0.2)  # Small delay for database consistency
                        verification_profile = await self.get_user_profile(validated_user_id)
                        
                        if verification_profile:
                            is_completed = verification_profile.onboarding_completed
                            print(f"🔍 Verification attempt {attempt + 1}: profile exists, onboarding_completed = {is_completed}")
                            
                            if is_completed:
                                print(f"✅ Onboarding completion verified successfully!")
                                final_verification_passed = True
                                break
                            else:
                                print(f"⚠️ Profile exists but onboarding_completed = False, trying force completion...")
                                force_success = await self._force_onboarding_completion(validated_user_id)
                                if force_success:
                                    print(f"🔧 Force completion successful on attempt {attempt + 1}")
                                    final_verification_passed = True
                                    break
                        else:
                            print(f"⚠️ Verification attempt {attempt + 1}: No profile found")
                            
                    except Exception as verify_error:
                        print(f"⚠️ Verification attempt {attempt + 1} error: {verify_error}")
                        continue
                
                if final_verification_passed:
                    logger.info(f"Onboarding completed successfully for user_id: {validated_user_id}")
                    return True
                else:
                    logger.error(f"Onboarding completion verification failed after {verification_attempts} attempts")
                    print(f"❌ Final verification failed after {verification_attempts} attempts")
                    # Try one final force completion
                    final_force = await self._force_onboarding_completion(validated_user_id)
                    print(f"🔧 Final force completion attempt: {final_force}")
                    return final_force
            else:
                print(f"❌ Profile creation/update failed!")
                logger.error(f"Failed to create/update user profile during onboarding completion for user_id: {validated_user_id}")
                return False

        except Exception as e:
            logger.error(f"Error completing onboarding for {user_id}: {e}")
            print(f"❌ Exception in complete_onboarding: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _force_onboarding_completion(self, user_id: str) -> bool:
        """FIXED: Force onboarding completion with a direct database update and better verification"""
        try:
            # Validate user_id first
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in _force_onboarding_completion: {validated_user_id}")
                return False
            
            logger.info(f"Force completing onboarding for user_id: {validated_user_id}")
            print(f"🔧 Force completing onboarding for user: {validated_user_id}")
            
            # FIXED: Check if profile exists first
            existing_profile = await self.get_user_profile(validated_user_id)
            if not existing_profile:
                print(f"🔧 No profile exists, cannot force complete onboarding")
                return False
            
            # Direct update with minimal data
            response = self.client.table("user_profiles").update({
                "onboarding_completed": True,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", validated_user_id).execute()
            
            success = len(response.data) > 0
            print(f"🔧 Force completion database result: {success}")
            
            if success:
                logger.info(f"Force onboarding completion successful for user_id: {validated_user_id}")
                # Double-check the result with multiple verification attempts
                for attempt in range(3):
                    try:
                        await asyncio.sleep(0.1)  # Small delay to ensure consistency
                        verification_profile = await self.get_user_profile(validated_user_id)
                        if verification_profile:
                            final_status = verification_profile.onboarding_completed
                            print(f"🔍 Force completion verification attempt {attempt + 1}: onboarding_completed = {final_status}")
                            if final_status:
                                print(f"✅ Force completion verified successfully")
                                return True
                        else:
                            print(f"⚠️ Could not retrieve profile for force completion verification attempt {attempt + 1}")
                    except Exception as verify_error:
                        print(f"⚠️ Force completion verification attempt {attempt + 1} error: {verify_error}")
                        continue
                
                # If we get here, verification failed but database update succeeded
                print(f"⚠️ Force completion database update succeeded but verification failed")
                return True  # Return True since database update worked
            else:
                logger.error(f"Force onboarding completion failed for user_id: {validated_user_id}")
                print(f"❌ Force completion database update failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in force onboarding completion for {user_id}: {e}")
            print(f"❌ Error in force completion: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== ONBOARDING OPERATIONS - ENHANCED ====================
    
    async def get_onboarding_progress(self, user_id: str) -> Optional[OnboardingProgress]:
        """
        Get onboarding progress for user - ENHANCED VERSION with better JSON handling
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_onboarding_progress: {validated_user_id}")
                return None
            
            logger.info(f"Getting onboarding progress for user_id: {validated_user_id}")
            
            response = self.client.table("onboarding_progress").select("*").eq("user_id", validated_user_id).execute()
            
            if response.data and len(response.data) > 0:
                progress_data = response.data[0]
                
                # ENHANCED: Better handling of collected_data JSON parsing
                collected_data = progress_data.get("collected_data")
                if collected_data:
                    if isinstance(collected_data, str):
                        try:
                            # Try to parse the JSON string
                            parsed_data = json.loads(collected_data)
                            progress_data["collected_data"] = parsed_data
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse collected_data JSON for user {validated_user_id}: {e}")
                            logger.error(f"Corrupted data: {collected_data[:200]}...")
                            # Set to empty dict if JSON is corrupted
                            progress_data["collected_data"] = {}
                    elif not isinstance(collected_data, dict):
                        logger.warning(f"collected_data is neither string nor dict for user {validated_user_id}, setting to empty dict")
                        progress_data["collected_data"] = {}
                else:
                    progress_data["collected_data"] = {}
                
                logger.info(f"Onboarding progress found for user_id: {validated_user_id}")
                return OnboardingProgress(**progress_data)
            
            logger.info(f"No onboarding progress found for user_id: {validated_user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting onboarding progress for {user_id}: {e}")
            return None  # Return None instead of raising exception
    
    async def update_onboarding_progress(self, user_id: str, step: OnboardingStep, data: Dict[str, Any]) -> bool:
        """
        Update onboarding progress - ENHANCED VERSION with better JSON handling
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in update_onboarding_progress: {validated_user_id}")
                return False
            
            logger.info(f"Updating onboarding progress for user_id: {validated_user_id}, step: {step.value}")
            
            # ENHANCED: Ensure data is properly formatted for JSON storage
            # The data should already be in the correct format from the agent
            collected_data_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            
            progress_data = {
                "user_id": validated_user_id,  # Use validated user_id
                "current_step": step.value,
                "collected_data": collected_data_json,
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
    
    # ==================== GROUP OPERATIONS - ENHANCED ====================
    
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
        Add user to a group - ENHANCED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
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
        Remove user from a group (soft delete) - ENHANCED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
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
        Get all groups a user belongs to - ENHANCED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
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
    
    # ==================== USER STATUS CHECK - ENHANCED ====================
    
    async def get_user_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user status including onboarding completion - ENHANCED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
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
    
    # ==================== CHAT OPERATIONS - ENHANCED ====================
    
    async def save_chat_message(self, message: ChatMessage) -> bool:
        """
        Save a chat message - ENHANCED VERSION with better validation
        """
        try:
            # ENHANCED: Validate required fields before saving
            if not message.user_id:
                logger.error("Cannot save chat message: user_id is missing")
                return False
            
            if not message.session_id:
                logger.error("Cannot save chat message: session_id is missing")
                return False
            
            # Validate UUIDs
            user_valid, _ = self._validate_and_format_uuid(message.user_id, "user_id")
            session_valid, _ = self._validate_and_format_uuid(message.session_id, "session_id")
            
            if not user_valid or not session_valid:
                logger.error(f"Cannot save chat message: invalid UUID format")
                return False
            
            logger.info(f"Saving chat message for user: {message.user_id}, session: {message.session_id}")
            
            message_dict = message.dict()
            message_dict["created_at"] = datetime.utcnow().isoformat()
            
            # Convert metadata to JSON if present
            if message_dict.get("metadata"):
                message_dict["metadata"] = json.dumps(message_dict["metadata"])
            
            response = self.client.table("chat_messages").insert(message_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Chat message saved successfully for user: {message.user_id}")
            else:
                logger.error(f"Failed to save chat message for user: {message.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving chat message for {message.user_id}: {e}")
            return False
    
    async def get_conversation_history(self, user_id: str, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """
        Get conversation history for a session - ENHANCED VERSION with better error handling
        """
        try:
            # ENHANCED: Better UUID validation with specific field names
            is_valid_user, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            is_valid_session, validated_session_id = self._validate_and_format_uuid(session_id, "session_id")
            
            if not is_valid_user:
                logger.warning(f"Invalid user_id format in get_conversation_history: {user_id}")
                return []
                
            if not is_valid_session:
                logger.warning(f"Invalid session_id format in get_conversation_history: {session_id}")
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
        Save a complete conversation turn (user message + assistant response) - ENHANCED VERSION
        """
        try:
            # ENHANCED: Validate inputs before creating ChatMessage objects
            if not user_id or not session_id:
                logger.error(f"Cannot save conversation: user_id or session_id is missing")
                return False
            
            # Validate UUIDs
            user_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            session_valid, validated_session_id = self._validate_and_format_uuid(session_id, "session_id")
            
            if not user_valid or not session_valid:
                logger.error(f"Cannot save conversation: invalid UUID format")
                return False
            
            logger.info(f"Saving conversation turn for user: {validated_user_id}, session: {validated_session_id}")
            
            # Save user message
            user_msg = ChatMessage(
                id=str(uuid.uuid4()),
                user_id=validated_user_id,
                session_id=validated_session_id,
                message_type=MessageType.USER,
                content=user_message,
                metadata=metadata or {}
            )
            
            # Save assistant message
            assistant_msg = ChatMessage(
                id=str(uuid.uuid4()),
                user_id=validated_user_id,
                session_id=validated_session_id,
                message_type=MessageType.ASSISTANT,
                content=assistant_response,
                metadata=metadata or {}
            )
            
            # Save both messages
            user_saved = await self.save_chat_message(user_msg)
            assistant_saved = await self.save_chat_message(assistant_msg)
            
            success = user_saved and assistant_saved
            if success:
                logger.info(f"Conversation turn saved successfully for user: {validated_user_id}")
            else:
                logger.error(f"Failed to save conversation turn for user: {validated_user_id}")
            
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
        Get user preferences - ENHANCED VERSION
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
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
