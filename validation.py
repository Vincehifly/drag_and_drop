"""Field validation functions."""

import re
from typing import Dict, Any, List

# Email validation regex
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def validate_field(config: Dict[str, Any], key: str, value: str) -> bool:
    """
    Validate a field value based on configuration rules.
    
    Args:
        config: Configuration containing field validation rules
        key: Field key to validate
        value: Value to validate
        
    Returns:
        True if valid, False otherwise
    """
    spec = next((f for f in config.get("required_fields", []) if f.get("key", f.get("name", "")) == key), None)
    if not spec:
        return True
        
    validation_type = spec.get("validate")
    if validation_type == "email":
        return bool(EMAIL_REGEX.match(value.strip()))
    elif validation_type == "required":
        return bool(value.strip())
    
    return True

def validate_all_fields(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate all fields in the data against configuration rules.
    
    Args:
        config: Configuration containing validation rules
        data: Data to validate
        
    Returns:
        Dictionary with validation errors (empty if all valid)
    """
    errors = {}
    required_fields = config.get("required_fields", [])
    
    for field in required_fields:
        key = field.get("key", field.get("name", ""))
        value = data.get(key, "")
        
        field_errors = []
        
        # Check if required field is present
        if not value.strip():
            field_errors.append(f"{key} is required")
        else:
            # Check specific validation rules
            if not validate_field(config, key, value):
                validation_type = field.get("validate", "")
                if validation_type == "email":
                    field_errors.append(f"{key} must be a valid email address")
                else:
                    field_errors.append(f"{key} validation failed")
        
        if field_errors:
            errors[key] = field_errors
    
    return errors

def get_missing_fields(config: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    """
    Get list of required fields that are missing from the data.
    
    Args:
        config: Configuration containing required fields
        data: Current data
        
    Returns:
        List of missing field keys
    """
    required_fields = config.get("required_fields", [])
    missing = []
    
    for field in required_fields:
        key = field.get("key", field.get("name", ""))
        if not data.get(key, "").strip():
            missing.append(key)
    
    return missing

def is_data_complete(config: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """
    Check if all required fields are present and valid.
    
    Args:
        config: Configuration containing required fields
        data: Data to check
        
    Returns:
        True if all required data is complete and valid
    """
    required_fields = config.get("required_fields", [])
    
    for field in required_fields:
        key = field.get("key", field.get("name", ""))
        value = data.get(key, "")
        
        # Check if field is present and not empty
        if not value.strip():
            return False
            
        # Check if field passes validation
        if not validate_field(config, key, value):
            return False
    
    return True


def validate_against_schema(data: Dict[str, Any], schema: List[Dict[str, Any]]) -> (Dict[str, Any], List[str]):
    """
    Validate and coerce `data` according to `schema`.

    Returns a tuple (validated_data, errors).
    """
    validated = {}
    errors: List[str] = []

    for field in schema:
        name = field.get("name")
        required = bool(field.get("required", False))
        ftype = field.get("type", "string")
        fmt = field.get("format")

        val = data.get(name)
        if val is None or (isinstance(val, str) and not val.strip()):
            if required:
                errors.append(f"{name} is required")
            continue

        # Coerce types
        try:
            if ftype == "string":
                v = str(val).strip()
            elif ftype == "int":
                v = int(val)
            elif ftype == "float":
                v = float(val)
            else:
                v = val
        except Exception:
            errors.append(f"{name} failed to coerce to {ftype}")
            continue

        # Format checks
        if fmt == "email":
            if not EMAIL_REGEX.match(str(v)):
                errors.append(f"{name} must be a valid email")
                continue

        validated[name] = v

    return validated, errors