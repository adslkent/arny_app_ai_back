import json
import os
import asyncio
from typing import Dict, Any

# Import handlers
from .handlers.main_handler import MainHandler
from .handlers.onboarding_handler import OnboardingHandler
from .utils.config import config

# Initialize handlers
main_handler = MainHandler()
onboarding_handler = OnboardingHandler()

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main AWS Lambda handler - routes requests to appropriate handlers
    
    FIXED: This is now synchronous as required by AWS Lambda runtime.
    All async operations are handled via asyncio.run()
    
    Args:
        event: Lambda event containing request data
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    
    # Run the async handler using asyncio.run()
    return asyncio.run(_async_lambda_handler(event, context))

async def _async_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Internal async handler that contains all the original logic
    
    Args:
        event: Lambda event containing request data
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    
    try:
        # Validate configuration
        config.validate()
        
        # Extract path from event
        path = event.get('path', '')
        http_method = event.get('httpMethod', 'POST')
        
        print(f"🔍 Processing request: {http_method} {path}")
        
        # Handle CORS preflight requests
        if http_method == 'OPTIONS':
            return _cors_response()
        
        # Route based on path
        if path.startswith('/auth'):
            return await _handle_auth_routes(event, context)
        elif path.startswith('/onboarding'):
            return await _handle_onboarding_routes(event, context)
        elif path.startswith('/chat') or path.startswith('/travel'):
            return await _handle_main_routes(event, context)
        elif path.startswith('/user'):
            return await _handle_user_routes(event, context)
        elif path == '/health':
            return await _handle_health_route(event, context)
        else:
            return _error_response(404, f"Route not found: {path}")
            
    except ValueError as e:
        # Configuration error
        print(f"❌ Configuration error: {str(e)}")
        return _error_response(500, f"Configuration error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error in lambda_handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, "Internal server error")

async def _handle_auth_routes(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle authentication routes"""
    
    path = event.get('path', '')
    
    if path == '/auth/signup' or path == '/auth/signin' or path == '/auth/refresh' or path == '/auth/signout':
        return await main_handler.handle_auth_request(event, context)
    else:
        return _error_response(404, f"Auth route not found: {path}")

async def _handle_onboarding_routes(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle onboarding routes"""
    
    path = event.get('path', '')
    
    if path == '/onboarding/chat':
        return await onboarding_handler.handle_request(event, context)
    elif path == '/onboarding/group/check':
        return await onboarding_handler.handle_group_code_check(event, context)
    elif path == '/onboarding/group/create':
        return await onboarding_handler.handle_create_group(event, context)
    elif path == '/onboarding/group/join':
        return await onboarding_handler.handle_join_group(event, context)
    else:
        return _error_response(404, f"Onboarding route not found: {path}")

async def _handle_main_routes(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle main travel conversation routes"""
    
    path = event.get('path', '')
    
    if path == '/chat' or path == '/travel/chat':
        return await main_handler.handle_request(event, context)
    else:
        return _error_response(404, f"Main route not found: {path}")

async def _handle_user_routes(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle user-related routes"""
    
    path = event.get('path', '')
    
    if path == '/user/status':
        return await main_handler.handle_user_status(event, context)
    else:
        return _error_response(404, f"User route not found: {path}")

async def _handle_health_route(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle health check route"""
    
    try:
        # Basic health check
        from .database.operations import DatabaseOperations
        db = DatabaseOperations()
        health_status = await db.health_check()
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'healthy',
                'service': 'arny-ai-backend',
                'database': health_status,
                'version': '1.0.0'
            })
        }
        
    except Exception as e:
        return _error_response(500, f"Health check failed: {str(e)}")

def _cors_response() -> Dict[str, Any]:
    """Return CORS preflight response"""
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET,PUT,DELETE'
        },
        'body': ''
    }

def _error_response(status_code: int, error_message: str) -> Dict[str, Any]:
    """Return error response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'success': False,
            'error': error_message
        })
    }

# For local testing
if __name__ == "__main__":
    import asyncio
    
    # Example test event
    test_event = {
        'path': '/health',
        'httpMethod': 'GET',
        'body': None
    }
    
    class MockContext:
        def __init__(self):
            self.function_name = "arny-ai-backend"
            self.function_version = "$LATEST"
            self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:arny-ai-backend"
            self.memory_limit_in_mb = "128"
            self.remaining_time_in_millis = lambda: 30000
    
    context = MockContext()
    response = lambda_handler(test_event, context)
    print(json.dumps(response, indent=2))
