import json
import uuid
from typing import Dict, Any
from datetime import datetime

from ..agents.onboarding_agent import OnboardingAgent
from ..database.operations import DatabaseOperations
from ..auth.supabase_auth import SupabaseAuth

class OnboardingHandler:
    """
    Handler for LLM-driven onboarding conversation flow - FIXED VERSION
    
    Fixed Issues:
    1. Better error handling for database operations
    2. Proper loading of existing onboarding progress
    3. Improved session continuity
    """
    
    def __init__(self):
        self.db = DatabaseOperations()
        self.auth = SupabaseAuth()
        self.onboarding_agent = OnboardingAgent()
    
    def _validate_user_id(self, user_id: str) -> tuple[bool, str]:
        """
        Enhanced user ID validation with better error handling
        
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
    
    async def handle_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle onboarding conversation requests - FIXED VERSION
        
        Args:
            event: Lambda event containing request data
            context: Lambda context
            
        Returns:
            Response dictionary
        """
        
        try:
            # Parse the incoming request
            body = json.loads(event.get('body', '{}'))
            
            # Extract key information
            raw_user_id = body.get('user_id')
            message = body.get('message')
            session_id = body.get('session_id', str(uuid.uuid4()))
            access_token = body.get('access_token')
            
            if not raw_user_id or not message:
                return self._error_response(400, "Missing user_id or message")
            
            if not access_token:
                return self._error_response(401, "Missing access_token")
            
            # Enhanced user_id validation
            is_valid, user_id_or_error = self._validate_user_id(raw_user_id)
            if not is_valid:
                print(f"UUID validation failed for '{raw_user_id}': {user_id_or_error}")
                return self._error_response(400, f"Invalid user_id: {user_id_or_error}")
            
            user_id = user_id_or_error  # Now contains the cleaned user_id
            print(f"✅ User ID validated: {user_id}")
            
            # Verify user authentication
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                return self._error_response(401, "Invalid or expired token")
            
            # Check if onboarding is already complete - with better error handling
            try:
                user_status = await self.db.get_user_status(user_id)
                if user_status and user_status.get('onboarding_completed', False):
                    print(f"✅ User {user_id} onboarding already complete")
                    return self._success_response({
                        'response': 'Your onboarding is already complete! You can now use Arny to plan your travels.',
                        'onboarding_complete': True,
                        'redirect_to_main': True
                    })
            except Exception as status_error:
                print(f"Note: Could not check user status (user may be new): {status_error}")
                # Continue with onboarding even if status check fails
            
            # Get user's onboarding progress - with better error handling and proper loading
            progress_data = {
                'collected_data': {},
                'conversation_history': []
            }
            
            try:
                onboarding_progress = await self.db.get_onboarding_progress(user_id)
                
                if onboarding_progress:
                    print(f"📥 Found existing onboarding progress for user {user_id}")
                    
                    # Extract conversation history and collected data from stored progress
                    try:
                        stored_data = onboarding_progress.collected_data or {}
                        if isinstance(stored_data, str):
                            stored_data = json.loads(stored_data)
                        
                        # Properly extract the nested structure
                        if isinstance(stored_data, dict):
                            progress_data['collected_data'] = stored_data.get('collected_data', {})
                            progress_data['conversation_history'] = stored_data.get('conversation_history', [])
                            progress_data['current_step'] = onboarding_progress.current_step.value
                            
                            print(f"📊 Loaded progress: {len(progress_data['conversation_history'])} messages, {len(progress_data['collected_data'])} data fields")
                            print(f"🎯 Current step: {progress_data.get('current_step', 'unknown')}")
                            print(f"📋 Collected data keys: {list(progress_data['collected_data'].keys())}")
                        else:
                            print("⚠️  Stored data is not in expected format, starting fresh")
                            
                    except (json.JSONDecodeError, AttributeError) as parse_error:
                        print(f"Warning: Could not parse progress data: {parse_error}")
                        # Continue with empty progress data
                else:
                    print(f"📄 No existing progress found for user {user_id}, starting fresh")
                    
            except Exception as progress_error:
                print(f"Note: Could not get onboarding progress (starting fresh): {progress_error}")
                # Continue with empty progress data
            
            # Process message with LLM-driven onboarding agent
            try:
                print(f"🤖 Processing message with agent: '{message}'")
                print(f"📊 Passing collected data: {progress_data['collected_data']}")
                
                agent_response = await self.onboarding_agent.process_message(
                    user_id=user_id,
                    message=message,
                    session_id=session_id,
                    current_progress=progress_data
                )
                
                print(f"✅ Agent response received: {agent_response.get('message', '')[:100]}...")
                
            except Exception as agent_error:
                print(f"Error in onboarding agent: {agent_error}")
                import traceback
                traceback.print_exc()
                return self._error_response(500, f"Onboarding agent error: {str(agent_error)}")
            
            # Check if onboarding is complete
            if agent_response.get('onboarding_complete', False):
                print(f"🎉 Onboarding completed for user {user_id}")
                return self._success_response({
                    'response': agent_response.get('message'),
                    'onboarding_complete': True,
                    'collected_data': agent_response.get('collected_data', {}),
                    'redirect_to_main': True,
                    'session_id': session_id
                })
            else:
                print(f"🔄 Onboarding continuing for user {user_id}")
                collected_data = agent_response.get('collected_data', {})
                conversation_context = agent_response.get('progress_data', {}).get('conversation_history', [])
                
                print(f"📈 Updated collected data: {list(collected_data.keys())}")
                print(f"💬 Conversation length: {len(conversation_context)}")
                
                return self._success_response({
                    'response': agent_response.get('message'),
                    'onboarding_complete': False,
                    'collected_data': collected_data,
                    'session_id': session_id,
                    'agent_status': 'continuing',
                    'conversation_context': {
                        'messages_count': len(conversation_context),
                        'data_collected': list(collected_data.keys())
                    }
                })
            
        except Exception as e:
            print(f"Error in onboarding handler: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._error_response(500, "Internal server error")
    
    async def handle_group_code_check(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle group code validation requests
        
        Args:
            event: Lambda event containing group code check request
            context: Lambda context
            
        Returns:
            Group code validation response
        """
        
        try:
            body = json.loads(event.get('body', '{}'))
            group_code = body.get('group_code')
            access_token = body.get('access_token')
            
            if not group_code:
                return self._error_response(400, "Missing group_code")
            
            if not access_token:
                return self._error_response(401, "Missing access_token")
            
            # Verify authentication
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                return self._error_response(401, "Invalid or expired token")
            
            # Validate group code format
            from ..utils.group_codes import GroupCodeGenerator
            
            formatted_code = GroupCodeGenerator.format_group_code(group_code)
            is_valid_format = GroupCodeGenerator.validate_group_code(formatted_code)
            
            if not is_valid_format:
                return self._success_response({
                    'valid': False,
                    'exists': False,
                    'message': 'Invalid group code format'
                })
            
            # Check if group exists
            group_exists = await self.db.check_group_exists(formatted_code)
            
            if group_exists:
                # Get group members for additional info
                members = await self.db.get_group_members(formatted_code)
                return self._success_response({
                    'valid': True,
                    'exists': True,
                    'group_code': formatted_code,
                    'member_count': len(members),
                    'message': f'Group found with {len(members)} members'
                })
            else:
                return self._success_response({
                    'valid': True,
                    'exists': False,
                    'group_code': formatted_code,
                    'message': 'Group code format is valid but group does not exist'
                })
                
        except Exception as e:
            print(f"Error in group code check: {str(e)}")
            return self._error_response(500, "Internal server error")
    
    async def handle_create_group(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle creating a new group for a user
        
        Args:
            event: Lambda event containing create group request
            context: Lambda context
            
        Returns:
            New group creation response
        """
        
        try:
            body = json.loads(event.get('body', '{}'))
            raw_user_id = body.get('user_id')
            access_token = body.get('access_token')
            
            if not raw_user_id:
                return self._error_response(400, "Missing user_id")
            
            if not access_token:
                return self._error_response(401, "Missing access_token")
            
            # Enhanced user_id validation
            is_valid, user_id_or_error = self._validate_user_id(raw_user_id)
            if not is_valid:
                return self._error_response(400, f"Invalid user_id: {user_id_or_error}")
            
            user_id = user_id_or_error
            
            # Verify authentication
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                return self._error_response(401, "Invalid or expired token")
            
            # Generate new group code
            from ..utils.group_codes import GroupCodeGenerator
            
            existing_codes = await self.db.get_existing_group_codes()
            new_group_code = GroupCodeGenerator.generate_unique_group_code(existing_codes)
            
            # Add user as admin of new group
            success = await self.db.add_group_member(new_group_code, user_id, "admin")
            
            if success:
                return self._success_response({
                    'group_code': new_group_code,
                    'role': 'admin',
                    'message': f'New group created with code: {new_group_code}'
                })
            else:
                return self._error_response(500, "Failed to create group")
                
        except Exception as e:
            print(f"Error creating group: {str(e)}")
            return self._error_response(500, "Internal server error")
    
    async def handle_join_group(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle joining an existing group
        
        Args:
            event: Lambda event containing join group request
            context: Lambda context
            
        Returns:
            Group join response
        """
        
        try:
            body = json.loads(event.get('body', '{}'))
            raw_user_id = body.get('user_id')
            group_code = body.get('group_code')
            access_token = body.get('access_token')
            
            if not raw_user_id or not group_code:
                return self._error_response(400, "Missing user_id or group_code")
            
            if not access_token:
                return self._error_response(401, "Missing access_token")
            
            # Enhanced user_id validation
            is_valid, user_id_or_error = self._validate_user_id(raw_user_id)
            if not is_valid:
                return self._error_response(400, f"Invalid user_id: {user_id_or_error}")
            
            user_id = user_id_or_error
            
            # Verify authentication
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                return self._error_response(401, "Invalid or expired token")
            
            # Format and validate group code
            from ..utils.group_codes import GroupCodeGenerator
            
            formatted_code = GroupCodeGenerator.format_group_code(group_code)
            
            # Check if group exists
            group_exists = await self.db.check_group_exists(formatted_code)
            
            if not group_exists:
                return self._error_response(404, "Group not found")
            
            # Add user to group
            success = await self.db.add_group_member(formatted_code, user_id, "member")
            
            if success:
                # Get group info
                members = await self.db.get_group_members(formatted_code)
                
                return self._success_response({
                    'group_code': formatted_code,
                    'role': 'member',
                    'member_count': len(members),
                    'message': f'Successfully joined group {formatted_code}'
                })
            else:
                return self._error_response(500, "Failed to join group")
                
        except Exception as e:
            print(f"Error joining group: {str(e)}")
            return self._error_response(500, "Internal server error")
    
    async def handle_onboarding_status(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle requests for onboarding status and progress
        
        Args:
            event: Lambda event containing status request
            context: Lambda context
            
        Returns:
            Onboarding status response
        """
        
        try:
            body = json.loads(event.get('body', '{}'))
            raw_user_id = body.get('user_id')
            access_token = body.get('access_token')
            
            if not raw_user_id:
                return self._error_response(400, "Missing user_id")
            
            if not access_token:
                return self._error_response(401, "Missing access_token")
            
            # Enhanced user_id validation
            is_valid, user_id_or_error = self._validate_user_id(raw_user_id)
            if not is_valid:
                return self._error_response(400, f"Invalid user_id: {user_id_or_error}")
            
            user_id = user_id_or_error
            
            # Verify authentication
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                return self._error_response(401, "Invalid or expired token")
            
            # Get user status and onboarding progress
            user_status = await self.db.get_user_status(user_id)
            onboarding_progress = await self.db.get_onboarding_progress(user_id)
            
            if user_status:
                progress_info = {}
                if onboarding_progress:
                    stored_data = onboarding_progress.collected_data or {}
                    if isinstance(stored_data, str):
                        try:
                            stored_data = json.loads(stored_data)
                        except json.JSONDecodeError:
                            stored_data = {}
                    
                    progress_info = {
                        'current_step': onboarding_progress.current_step.value,
                        'collected_fields': list(stored_data.get('collected_data', {}).keys()),
                        'conversation_length': len(stored_data.get('conversation_history', [])),
                        'last_updated': onboarding_progress.updated_at.isoformat() if onboarding_progress.updated_at else None
                    }
                
                return self._success_response({
                    'user_id': user_id,
                    'onboarding_completed': user_status.get('onboarding_completed', False),
                    'profile_exists': user_status.get('profile') is not None,
                    'progress': progress_info
                })
            else:
                return self._error_response(404, "User not found")
                
        except Exception as e:
            print(f"Error getting onboarding status: {str(e)}")
            return self._error_response(500, "Internal server error")
    
    def _success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return successful API response"""
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({
                'success': True,
                'data': data,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
    
    def _error_response(self, status_code: int, error_message: str) -> Dict[str, Any]:
        """Return error API response"""
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': error_message,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
