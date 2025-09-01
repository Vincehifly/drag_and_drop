"""
Google Sheets integration for the LangGraph agent.
Provides functions to load the Google Sheets client and append data to sheets.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Optional gspread dependency
try:
    import gspread
    from gspread import Client as GspreadClient
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GspreadClient = None
    Credentials = None
    GSPREAD_AVAILABLE = False

def load_gspread_client(credentials_path: Optional[str] = None) -> GspreadClient:
    """
    Load and authenticate Google Sheets client.
    
    Args:
        credentials_path: Path to Google service account credentials JSON file.
                         If None, will try to use environment variables.
    
    Returns:
        Authenticated gspread client
    
    Raises:
        ImportError: If gspread is not available
        ValueError: If credentials are invalid
        Exception: For other authentication errors
    """
    if not GSPREAD_AVAILABLE:
        raise ImportError(
            "gspread library not available. Install with: pip install gspread google-auth"
        )
    
    # Try to find credentials
    if not credentials_path:
        # Look for credentials in common locations
        possible_paths = [
            "smart-mark-411015-e7b745e2077e.json",  # Default credentials file
            os.path.expanduser("~/.config/gspread/credentials.json"),
            os.path.expanduser("~/google_sheets_credentials.json"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                credentials_path = path
                break
        
        if not credentials_path:
            raise ValueError(
                "No credentials path provided and no default credentials found. "
                "Please set credentials_path in tool config or place credentials file in a standard location."
            )
    
    # Validate credentials file exists
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
    
    try:
        # Load credentials from JSON file
        with open(credentials_path, 'r') as f:
            creds_data = json.load(f)
        
        # Create credentials object
        credentials = Credentials.from_service_account_info(
            creds_data,
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
        )
        
        # Create and return client
        client = gspread.authorize(credentials)
        return client
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in credentials file: {e}")
    except Exception as e:
        raise Exception(f"Failed to authenticate with Google Sheets: {e}")

def append_to_sheet(
    client: GspreadClient, 
    config: Dict[str, Any], 
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Append data to a Google Sheet.
    
    Args:
        client: Authenticated gspread client
        config: Tool configuration containing sheet info
        data: Data to append to the sheet
    
    Returns:
        Dict with operation result and metadata
    
    Raises:
        ValueError: If sheet configuration is invalid
        Exception: For other sheet operation errors
    """
    try:
        # Extract sheet configuration
        sheet_config = config.get("sheet", {})
        spreadsheet_title = sheet_config.get("spreadsheet_title")
        worksheet_name = sheet_config.get("worksheet_name")
        
        if not spreadsheet_title:
            raise ValueError("Missing spreadsheet_title in sheet configuration")
        if not worksheet_name:
            raise ValueError("Missing worksheet_name in sheet configuration")
        
        # Open or create spreadsheet
        try:
            spreadsheet = client.open(spreadsheet_title)
        except gspread.SpreadsheetNotFound:
            # Try to create the spreadsheet if it doesn't exist
            try:
                spreadsheet = client.create(spreadsheet_title)
            except Exception as e:
                raise Exception(f"Failed to create spreadsheet '{spreadsheet_title}': {e}")
        
        # Get or create worksheet
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            # Create worksheet if it doesn't exist
            try:
                worksheet = spreadsheet.add_worksheet(
                    title=worksheet_name,
                    rows=1000,
                    cols=26
                )
            except Exception as e:
                raise Exception(f"Failed to create worksheet '{worksheet_name}': {e}")
        
        # Prepare data for insertion
        # Get existing headers to determine column order
        try:
            headers = worksheet.row_values(1)
        except Exception:
            headers = []
        
        # If no headers exist, create them from the data
        if not headers:
            headers = list(data.keys())
            worksheet.append_row(headers)
        
        # Create row data in the correct order
        row_data = []
        for header in headers:
            if header in data:
                value = data[header]
                # Convert None to empty string for Google Sheets
                if value is None:
                    value = ""
                row_data.append(str(value))
            else:
                row_data.append("")  # Empty cell for missing fields
        
        # Append the row
        worksheet.append_row(row_data)
        
        # Return success response
        return {
            "success": True,
            "spreadsheet": spreadsheet_title,
            "worksheet": worksheet_name,
            "rows_added": 1,
            "headers": headers,
            "data_inserted": data
        }
        
    except Exception as e:
        raise Exception(f"Failed to append data to sheet: {e}")

def verify_sheet_access(client: GspreadClient, config: Dict[str, Any]) -> bool:
    """
    Verify that the client can access the specified sheet.
    
    Args:
        client: Authenticated gspread client
        config: Tool configuration containing sheet info
    
    Returns:
        True if access is verified, False otherwise
    """
    try:
        sheet_config = config.get("sheet", {})
        spreadsheet_title = sheet_config.get("spreadsheet_title")
        worksheet_name = sheet_config.get("worksheet_name")
        
        if not spreadsheet_title or not worksheet_name:
            return False
        
        # Try to open the spreadsheet
        spreadsheet = client.open(spreadsheet_title)
        
        # Try to access the worksheet
        worksheet = spreadsheet.worksheet(worksheet_name)
        
        # If we get here, access is verified
        return True
        
    except Exception:
        return False

def get_sheet_info(client: GspreadClient, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about the specified sheet.
    
    Args:
        client: Authenticated gspread client
        config: Tool configuration containing sheet info
    
    Returns:
        Dict with sheet information
    """
    try:
        sheet_config = config.get("sheet", {})
        spreadsheet_title = sheet_config.get("spreadsheet_title")
        worksheet_name = sheet_config.get("worksheet_name")
        
        if not spreadsheet_title or not worksheet_name:
            return {"error": "Missing sheet configuration"}
        
        # Open spreadsheet
        spreadsheet = client.open(spreadsheet_title)
        
        # Get worksheet
        worksheet = spreadsheet.worksheet(worksheet_name)
        
        # Get basic info
        info = {
            "spreadsheet_title": spreadsheet_title,
            "worksheet_name": worksheet_name,
            "spreadsheet_id": spreadsheet.id,
            "worksheet_id": worksheet.id,
            "row_count": worksheet.row_count,
            "col_count": worksheet.col_count,
            "url": spreadsheet.url
        }
        
        # Try to get headers
        try:
            headers = worksheet.row_values(1)
            info["headers"] = headers
            info["column_count"] = len(headers)
        except Exception:
            info["headers"] = []
            info["column_count"] = 0
        
        return info
        
    except Exception as e:
        return {"error": f"Failed to get sheet info: {e}"}





