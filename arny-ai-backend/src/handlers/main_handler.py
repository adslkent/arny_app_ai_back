import json
import uuid
from typing import Dict, Any
from datetime import datetime

from ..agents.supervisor_agent import SupervisorAgent
from ..agents.flight_agent import FlightAgent
from ..agents.hotel_agent import HotelAgent
from ..database.operations import DatabaseOperations
from ..database.models import ChatMessage
from ..auth.supabase_auth import SupabaseAuth

class MainHandler:
    """
    Main handler that routes requests to appropriate agents for the main travel conversation flow
    """
    
    def __init__(self):
        self.db = DatabaseOperations()
        self.auth = SupabaseAuth()
        self.supervisor_agent = SupervisorAgent()
        self.flight_agent = FlightAgent()
        self.hotel_agent = HotelAgent()
    
    async def handle_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Main handler for travel agent requests
        
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
            user_id = body.get('user_id')
            message = body.get('message')
            session_id = body.get('session_id', str(uuid.uuid4()))
            access_token = body.get('access_token')
            
            if not user_id or not message:
                return self._error_response(400, "Missing user_id or message")
            
            if not access_token:
                return self._error_response(401, "Missing access_token")
            
            # Verify user authentication
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                return self._error_response(401, "Invalid or expired token")
            
            # Get user's profile and conversation history
            user_profile = await self.db.get_user_profile(user_id)
            if not user_profile:
                return self._error_response(404, "User profile not found")
            
            conversation_history = await self.db.get_conversation_history(user_id, session_id)
            
            # Process message through supervisor agent first
            supervisor_response = await self.supervisor_agent.process_message(
                user_id=user_id,
                message=message,
                session_id=session_id,
                user_profile=user_profile.model_dump(),
                conversation_history=conversation_history
            )
            
            # Check if we need to route to a specific agent
            if supervisor_response.get("requires_routing"):
                route_to = supervisor_response.get("route_to")
                
                if route_to == "flight_agent":
                    agent_response = await self.flight_agent.process_message(
                        user_id=user_id,
                        message=message,
                        session_id=session_id,
                        user_profile=user_profile.model_dump(),
                        conversation_history=conversation_history
                    )
                elif route_to == "hotel_agent":
                    agent_response = await self.hotel_agent.process_message(
                        user_id=user_id,
                        message=message,
                        session_id=session_id,
                        user_profile=user_profile.model_dump(),
                        conversation_history=conversation_history
                    )
                else:
                    # Fallback to supervisor for general conversation
                    agent_response = await self.supervisor_agent.handle_general_conversation(
                        user_id=user_id,
                        message=message,
                        session_id=session_id,
                        user_profile=user_profile.model_dump(),
                        conversation_history=conversation_history
                    )
            else:
                # Use supervisor response directly
                agent_response = supervisor_response
            
            # Save conversation to database
            await self.db.save_conversation_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=message,
                assistant_response=agent_response.get('message', ''),
                metadata=agent_response.get('metadata', {})
            )
            
            # Format response
            response_data = {
                'response': agent_response.get('message'),
                'agent_type': agent_response.get('agent_type', 'main_travel'),
                'session_id': session_id,
                'requires_action': agent_response.get('requires_action', False)
            }
            
            # Add additional data based on agent type
            if agent_response.get('agent_type') == 'flight':
                response_data.update({
                    'search_results': agent_response.get('search_results', []),
                    'search_id': agent_response.get('search_id'),
                    'flight_selected': agent_response.get('flight_selected')
                })
            elif agent_response.get('agent_type') == 'hotel':
                response_data.update({
                    'search_results': agent_response.get('search_results', []),
                    'search_id': agent_response.get('search_id'),
                    'hotel_selected': agent_response.get('hotel_selected')
                })
            
            # Add metadata if present
            if agent_response.get('metadata'):
                response_data['metadata'] = agent_response['metadata']
            
            return self._success_response(response_data)
            
        except Exception as e:
            print(f"Error in main handler: {str(e)}")
            return self._error_response(500, "Internal server error")
    
    async def handle_auth_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle authentication requests (sign up, sign in, etc.)
        
        Args:
            event: Lambda event containing auth request
            context: Lambda context
            
        Returns:
            Authentication response
        """
        
        try:
            body = json.loads(event.get('body', '{}'))
            action = body.get('action')  # 'signup', 'signin', 'refresh', 'signout'
            
            if action == 'signup':
                email = body.get('email')
                password = body.get('password')
                metadata = body.get('metadata', {})
                
                if not email or not password:
                    return self._error_response(400, "Missing email or password")
                
                auth_result = await self.auth.sign_up(email, password, metadata)
                
                if auth_result.get("success"):
                    # Create initial user profile
                    from ..database.models import UserProfile
                    
                    user_profile = UserProfile(
                        user_id=auth_result["user"]["id"],
                        email=email,
                        onboarding_completed=False
                    )
                    
                    await self.db.create_user_profile(user_profile)
                    
                    return self._success_response({
                        'user': auth_result["user"],
                        'session': auth_result["session"],
                        'message': 'User created successfully'
                    })
                else:
                    return self._error_response(400, auth_result.get("error", "Signup failed"))
            
            elif action == 'signin':
                email = body.get('email')
                password = body.get('password')
                
                if not email or not password:
                    return self._error_response(400, "Missing email or password")
                
                auth_result = await self.auth.sign_in(email, password)
                
                if auth_result.get("success"):
                    return self._success_response({
                        'user': auth_result["user"],
                        'session': auth_result["session"],
                        'message': 'Signed in successfully'
                    })
                else:
                    return self._error_response(401, auth_result.get("error", "Sign in failed"))
            
            elif action == 'refresh':
                refresh_token = body.get('refresh_token')
                
                if not refresh_token:
                    return self._error_response(400, "Missing refresh_token")
                
                auth_result = await self.auth.refresh_session(refresh_token)
                
                if auth_result.get("success"):
                    return self._success_response({
                        'session': auth_result["session"],
                        'message': 'Session refreshed successfully'
                    })
                else:
                    return self._error_response(401, auth_result.get("error", "Token refresh failed"))
            
            elif action == 'signout':
                auth_result = await self.auth.sign_out()
                return self._success_response({
                    'message': 'Signed out successfully'
                })
            
            else:
                return self._error_response(400, f"Unknown action: {action}")
                
        except Exception as e:
            print(f"Error in auth handler: {str(e)}")
            return self._error_response(500, "Internal server error")
    
    async def handle_user_status(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle user status requests (check onboarding completion, etc.)
        
        Args:
            event: Lambda event containing user status request
            context: Lambda context
            
        Returns:
            User status response
        """
        
        try:
            body = json.loads(event.get('body', '{}'))
            user_id = body.get('user_id')
            access_token = body.get('access_token')
            
            if not user_id or not access_token:
                return self._error_response(400, "Missing user_id or access_token")
            
            # Verify authentication
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                return self._error_response(401, "Invalid or expired token")
            
            # Get user status
            user_status = await self.db.get_user_status(user_id)
            
            if user_status:
                return self._success_response(user_status)
            else:
                return self._error_response(404, "User not found")
                
        except Exception as e:
            print(f"Error in user status handler: {str(e)}")
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