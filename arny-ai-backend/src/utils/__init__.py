"""
Utilities module for Arny AI Backend

This module provides utility functions and configurations including:
- Configuration management for environment variables
- Group code generation and validation for family/group travel
- Common utility functions for data validation and formatting
- Helper functions for date/time operations and string processing

Usage:
    from utils import config, GroupCodeGenerator
    from utils import validate_email, format_date, generate_session_id
    
    # Configuration
    api_key = config.OPENAI_API_KEY
    
    # Group codes
    generator = GroupCodeGenerator()
    code = generator.generate_group_code()
    
    # Utilities
    is_valid = validate_email("user@example.com")
    session_id = generate_session_id()
"""

import re
import uuid
import hashlib
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union
from email_validator import validate_email as email_validate, EmailNotValidError

from .config import Config, config
from .group_codes import GroupCodeGenerator

# Export main classes and functions
__all__ = [
    # Configuration
    'Config',
    'config',
    'get_config',
    'validate_config',
    
    # Group codes
    'GroupCodeGenerator',
    'generate_group_code',
    'validate_group_code',
    'format_group_code',
    
    # Validation utilities
    'validate_email',
    'validate_phone',
    'validate_date',
    'validate_uuid',
    'validate_session_id',
    
    # Formatting utilities
    'format_date',
    'format_currency',
    'format_phone',
    'sanitize_string',
    'truncate_string',
    
    # Generation utilities
    'generate_session_id',
    'generate_unique_id',
    'generate_hash',
    'generate_timestamp',
    
    # Data utilities
    'safe_get',
    'merge_dicts',
    'flatten_dict',
    'clean_dict',
    'normalize_text',
    
    # Constants
    'DEFAULT_DATE_FORMAT',
    'DEFAULT_DATETIME_FORMAT',
    'SUPPORTED_CURRENCIES',
    'PHONE_REGEX_PATTERNS'
]

# Version information
__version__ = '1.0.0'

# ==================== CONSTANTS ====================

DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIMEZONE = "UTC"

SUPPORTED_CURRENCIES = [
    'USD', 'EUR', 'GBP', 'AUD', 'CAD', 'JPY', 'CNY', 'INR', 'SGD', 'HKD'
]

PHONE_REGEX_PATTERNS = {
    'international': r'^\+[1-9]\d{1,14}$',
    'us': r'^(\+1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$',
    'au': r'^(\+61[-.\s]?)?(\(0\)|0)?[2-9]\d{8}$',
    'uk': r'^(\+44[-.\s]?)?(\(0\)|0)?[1-9]\d{8,9}$'
}

# ==================== CONFIGURATION UTILITIES ====================

def get_config() -> Config:
    """
    Get the global configuration instance
    
    Returns:
        Config: Global configuration object
    """
    return config

def validate_config() -> Dict[str, Any]:
    """
    Validate the current configuration
    
    Returns:
        dict: Configuration validation results
    """
    try:
        config.validate()
        return {
            'valid': True,
            'message': 'Configuration is valid',
            'missing_fields': []
        }
    except ValueError as e:
        missing_fields = str(e).replace('Required environment variable ', '').replace(' is not set', '').split(', ')
        return {
            'valid': False,
            'message': str(e),
            'missing_fields': missing_fields
        }

# ==================== GROUP CODE UTILITIES ====================

def generate_group_code(length: int = 6) -> str:
    """
    Generate a random group code
    
    Args:
        length: Length of the group code
        
    Returns:
        Random group code
    """
    return GroupCodeGenerator.generate_group_code(length)

def validate_group_code(code: str) -> bool:
    """
    Validate group code format
    
    Args:
        code: Group code to validate
        
    Returns:
        True if valid, False otherwise
    """
    return GroupCodeGenerator.validate_group_code(code)

def format_group_code(code: str) -> str:
    """
    Format group code to standard format
    
    Args:
        code: Raw group code
        
    Returns:
        Formatted group code
    """
    return GroupCodeGenerator.format_group_code(code)

# ==================== VALIDATION UTILITIES ====================

def validate_email(email: str) -> bool:
    """
    Validate email address format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        email_validate(email)
        return True
    except EmailNotValidError:
        return False

def validate_phone(phone: str, country_code: str = 'international') -> bool:
    """
    Validate phone number format
    
    Args:
        phone: Phone number to validate
        country_code: Country code pattern to use ('international', 'us', 'au', 'uk')
        
    Returns:
        True if valid, False otherwise
    """
    if country_code not in PHONE_REGEX_PATTERNS:
        country_code = 'international'
    
    pattern = PHONE_REGEX_PATTERNS[country_code]
    return bool(re.match(pattern, phone.strip()))

def validate_date(date_string: str, date_format: str = DEFAULT_DATE_FORMAT) -> bool:
    """
    Validate date string format
    
    Args:
        date_string: Date string to validate
        date_format: Expected date format
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False

def validate_uuid(uuid_string: str) -> bool:
    """
    Validate UUID string format
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format (should be a UUID)
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    return validate_uuid(session_id)

# ==================== FORMATTING UTILITIES ====================

def format_date(date_obj: Union[datetime, date, str], output_format: str = DEFAULT_DATE_FORMAT) -> str:
    """
    Format date object to string
    
    Args:
        date_obj: Date object, datetime object, or date string
        output_format: Output format string
        
    Returns:
        Formatted date string
    """
    if isinstance(date_obj, str):
        # Try to parse string date
        for fmt in [DEFAULT_DATE_FORMAT, DEFAULT_DATETIME_FORMAT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"]:
            try:
                date_obj = datetime.strptime(date_obj, fmt)
                break
            except ValueError:
                continue
        else:
            return date_obj  # Return original if can't parse
    
    if isinstance(date_obj, datetime):
        return date_obj.strftime(output_format)
    elif isinstance(date_obj, date):
        return date_obj.strftime(output_format)
    else:
        return str(date_obj)

def format_currency(amount: Union[float, int, str], currency: str = 'USD', include_symbol: bool = True) -> str:
    """
    Format currency amount
    
    Args:
        amount: Amount to format
        currency: Currency code
        include_symbol: Whether to include currency symbol
        
    Returns:
        Formatted currency string
    """
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return str(amount)
    
    # Currency symbols
    symbols = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 'AUD': 'A$', 'CAD': 'C$',
        'JPY': '¥', 'CNY': '¥', 'INR': '₹', 'SGD': 'S$', 'HKD': 'HK$'
    }
    
    formatted_amount = f"{amount:,.2f}"
    
    if include_symbol and currency in symbols:
        return f"{symbols[currency]}{formatted_amount}"
    else:
        return f"{formatted_amount} {currency}"

def format_phone(phone: str, country_code: str = 'international') -> str:
    """
    Format phone number to standard format
    
    Args:
        phone: Phone number to format
        country_code: Country code for formatting
        
    Returns:
        Formatted phone number
    """
    # Remove all non-digit characters except +
    cleaned = re.sub(r'[^\d+]', '', phone)
    
    if country_code == 'us' and not cleaned.startswith('+'):
        if len(cleaned) == 10:
            return f"+1-{cleaned[:3]}-{cleaned[3:6]}-{cleaned[6:]}"
        elif len(cleaned) == 11 and cleaned.startswith('1'):
            return f"+{cleaned[:1]}-{cleaned[1:4]}-{cleaned[4:7]}-{cleaned[7:]}"
    
    return cleaned

def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string by removing dangerous characters
    
    Args:
        text: Text to sanitize
        max_length: Maximum length to truncate to
        
    Returns:
        Sanitized string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\';\\]', '', text)
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    if max_length:
        sanitized = truncate_string(sanitized, max_length)
    
    return sanitized

def truncate_string(text: str, max_length: int, suffix: str = '...') -> str:
    """
    Truncate string to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

# ==================== GENERATION UTILITIES ====================

def generate_session_id() -> str:
    """
    Generate a unique session ID
    
    Returns:
        UUID string for session ID
    """
    return str(uuid.uuid4())

def generate_unique_id(prefix: str = '') -> str:
    """
    Generate a unique identifier
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}{unique_id}" if prefix else unique_id

def generate_hash(data: str, algorithm: str = 'sha256') -> str:
    """
    Generate hash of data
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm ('sha256', 'md5', 'sha1')
        
    Returns:
        Hash string
    """
    if algorithm == 'sha256':
        return hashlib.sha256(data.encode()).hexdigest()
    elif algorithm == 'md5':
        return hashlib.md5(data.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(data.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def generate_timestamp() -> str:
    """
    Generate current timestamp in ISO format
    
    Returns:
        ISO format timestamp string
    """
    return datetime.utcnow().isoformat()

# ==================== DATA UTILITIES ====================

def safe_get(data: Dict[str, Any], key: str, default: Any = None, transform: Optional[callable] = None) -> Any:
    """
    Safely get value from dictionary with optional transformation
    
    Args:
        data: Dictionary to get value from
        key: Key to look for (supports dot notation like 'user.profile.name')
        default: Default value if key not found
        transform: Optional transformation function
        
    Returns:
        Value from dictionary or default
    """
    try:
        value = data
        for k in key.split('.'):
            value = value[k]
        
        if transform and value is not None:
            value = transform(value)
        
        return value
    except (KeyError, TypeError, IndexError):
        return default

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def flatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        data: Dictionary to flatten
        separator: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(data)

def clean_dict(data: Dict[str, Any], remove_none: bool = True, remove_empty: bool = False) -> Dict[str, Any]:
    """
    Clean dictionary by removing None/empty values
    
    Args:
        data: Dictionary to clean
        remove_none: Remove None values
        remove_empty: Remove empty strings/lists/dicts
        
    Returns:
        Cleaned dictionary
    """
    cleaned = {}
    for k, v in data.items():
        if remove_none and v is None:
            continue
        if remove_empty and v == '' or v == [] or v == {}:
            continue
        cleaned[k] = v
    return cleaned

def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-.,!?]', '', text)
    
    return text

# ==================== FACTORY FUNCTIONS ====================

def create_group_code_generator() -> GroupCodeGenerator:
    """
    Factory function to create a GroupCodeGenerator instance
    
    Returns:
        GroupCodeGenerator: Configured group code generator
    """
    return GroupCodeGenerator()

# ==================== HEALTH CHECK ====================

def health_check() -> Dict[str, Any]:
    """
    Perform health check on utilities module
    
    Returns:
        dict: Health status of utilities
    """
    status = {
        'module': 'utils',
        'version': __version__,
        'status': 'healthy',
        'components': {}
    }
    
    try:
        # Test configuration
        config_status = validate_config()
        status['components']['config'] = {
            'status': 'healthy' if config_status['valid'] else 'warning',
            'details': config_status
        }
        
        # Test group code generator
        generator = GroupCodeGenerator()
        test_code = generator.generate_group_code()
        is_valid = generator.validate_group_code(test_code)
        
        status['components']['group_codes'] = {
            'status': 'healthy' if is_valid else 'error',
            'test_code_generated': test_code,
            'test_validation_passed': is_valid
        }
        
        # Test utilities
        test_email = validate_email('test@example.com')
        test_uuid = validate_uuid(str(uuid.uuid4()))
        
        status['components']['validators'] = {
            'status': 'healthy' if test_email and test_uuid else 'error',
            'email_validator': test_email,
            'uuid_validator': test_uuid
        }
        
    except Exception as e:
        status['status'] = 'error'
        status['error'] = str(e)
    
    return status

# ==================== MODULE INITIALIZATION ====================

# Validate configuration on import (optional - can be disabled for testing)
try:
    config.validate()
except ValueError as e:
    # Don't fail on import, just log warning
    import warnings
    warnings.warn(f"Configuration validation failed: {e}", UserWarning)

# Export commonly used instances
group_code_generator = GroupCodeGenerator()