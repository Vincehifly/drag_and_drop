"""Tool implementations for the LangGraph agent."""
# CONTEXT for GitHub Copilot:
# - Write minimal, clean code without excessive prints or debug output
# - Focus on core functionality, avoid verbose instructions
# - Use modern Python patterns and best practices
# - For LangGraph: use interrupt()/Command(resume=...) pattern for human-in-the-loop
# - Prefer concise, working solutions over detailed explanations in code

import os
import re
import json
from typing import Dict, Any, Optional
from urllib.parse import quote_plus
from google_sheets import load_gspread_client, append_to_sheet
from validation import is_data_complete


# Optional Tavily client
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception:
    TavilyClient = None
    TAVILY_AVAILABLE = False

def api_retrieval_tool(extracted_data: Dict[str, Any], config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Generic API retrieval tool.

    Config keys expected under config["api_retrieval"]:
      - base_url: str (required)
      - method: "GET"|"POST" (default "GET")
      - query_param_key: str (optional; if present use extracted_data["query"]) 
      - param_map: dict mapping extracted field -> api param name (optional)
      - headers: dict (optional)
      - auth: { type: "header"|"query", env_key: str, name: str } (optional)
      - timeout: int seconds (default 15)
    """
    if verbose:
        print("--- TOOL USE NODE (api_retrieval) ---")

    # Dependency guard
    try:
        import requests  # noqa: F401
    except Exception:
        return {
            "success": False,
            "data": extracted_data,
            "type": "api_retrieval",
            "message": "Requests library not available. Install: pip install requests",
            "error": "Missing dependency: requests",
        }

    api_cfg = config.get("api_retrieval", {}) or {}
    base_url = api_cfg.get("base_url")
    if not base_url:
        return {
            "success": False,
            "data": extracted_data,
            "type": "api_retrieval",
            "message": "Missing required config: base_url",
            "error": "Invalid configuration",
        }

    method = str(api_cfg.get("method", "GET")).upper()
    timeout = int(api_cfg.get("timeout", 15))
    headers = dict(api_cfg.get("headers", {}) or {})

    # Build params/body
    params: Dict[str, Any] = {}
    body: Optional[Dict[str, Any]] = None

    # Single-string query support
    qkey = api_cfg.get("query_param_key")
    query_value = extracted_data.get("query") or extracted_data.get("search_query")
    if qkey and query_value:
        params[qkey] = query_value

    # Structured param mapping support
    param_map = api_cfg.get("param_map") or {}
    if isinstance(param_map, dict):
        for in_key, api_key in param_map.items():
            if in_key in extracted_data and extracted_data[in_key] not in (None, ""):
                params[str(api_key)] = extracted_data[in_key]

    # Auth injection
    try:
        auth = api_cfg.get("auth") or {}
        auth_type = (auth.get("type") or "").lower()
        env_key = auth.get("env_key")
        name = auth.get("name")
        token = os.environ.get(env_key) if env_key else None
        if token and name:
            if auth_type == "header":
                headers[name] = token
            elif auth_type == "query":
                params[name] = token
    except Exception:
        pass

    url = base_url
    try:
        if method == "GET":
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        else:
            # Default to JSON body for POST-like methods
            body = params if not qkey else {qkey: query_value, **{k: v for k, v in params.items() if k != qkey}}
            params = {}  # keep querystring clean for POST
            resp = requests.request(method, url, params=params, json=body, headers=headers, timeout=timeout)
        status = resp.status_code
        text = resp.text
        try:
            payload = resp.json()
        except Exception:
            payload = None

        ok = 200 <= status < 300
        # Short human summary string
        preview = ""
        try:
            if isinstance(payload, dict):
                preview = json.dumps({k: payload[k] for k in list(payload.keys())[:3]}, ensure_ascii=False)
            elif isinstance(payload, list):
                preview = f"list[{len(payload)}]"
            else:
                preview = (text or "")[:200]
        except Exception:
            preview = (text or "")[:200]

        summary = f"API {method} {status} for {url}"
        if query_value:
            summary += f" | query='{str(query_value)[:50]}'"

        return {
            "success": ok,
            "type": "api_retrieval",
            "message": summary,
            "summary": summary,
            "request": {
                "url": url,
                "method": method,
                "params": params,
                "body": body,
                "headers": {k: ("***" if k.lower().startswith("authorization") else v) for k, v in headers.items()}
            },
            "status_code": status,
            "json": payload,
            "text": text,
            "data": {"query": query_value}
        }
    except Exception as e:
        return {
            "success": False,
            "type": "api_retrieval",
            "message": f"API request failed: {e}",
            "error": str(e),
            "request": {"url": url, "method": method, "params": params, "body": body, "headers": headers},
            "data": {"query": query_value}
        }

def sheets_tool(extracted_data: Dict[str, Any], config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Google Sheets tool implementation with comprehensive error handling.
    
    Args:
        extracted_data: Data to write to the sheet
        config: Configuration containing sheet details
        verbose: Enable verbose logging
        
    Returns:
        Tool execution result
    """
    if verbose:
        print("--- TOOL USE NODE (sheets) ---")
        print("TOOL PROMPT INJECTION: sheets tool execution")
        print(f"Using extracted data: {extracted_data}")
    
    try:
        # Validate that we have all required data
        if not is_data_complete(config, extracted_data):
            summary = f"Failed: missing required data for Google Sheets write."
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": "Cannot execute tool: missing required data",
                "error": "Incomplete data",
                "summary": summary
            }
        
        # Comprehensive configuration validation
        if verbose:
            print(f"🔍 Debug: Config keys: {list(config.keys())}")
            print(f"🔍 Debug: Config content: {config}")
        
        # Check if we have sheet configuration
        if "sheet" not in config:
            error_msg = "Google Sheets configuration missing in agent config"
            if verbose:
                print(f"❌ {error_msg}")
                print(f"   Expected: config['sheet'] with spreadsheet_title and worksheet_name")
                print(f"   Found: {list(config.keys())}")
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": "Missing sheet configuration",
                "summary": error_msg
            }
        
        # Validate sheet configuration details
        sheet_config = config.get("sheet", {})
        spreadsheet_title = sheet_config.get("spreadsheet_title")
        worksheet_name = sheet_config.get("worksheet_name")
        
        if not spreadsheet_title:
            error_msg = "Missing spreadsheet_title in sheet configuration"
            if verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": "Missing spreadsheet_title",
                "summary": error_msg
            }
        
        if not worksheet_name:
            error_msg = "Missing worksheet_name in sheet configuration"
            if verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": "Missing worksheet_name",
                "summary": error_msg
            }
        
        # Check credentials path
        credentials_path = config.get("credentials_path")
        if not credentials_path:
            error_msg = "Missing credentials_path in agent configuration"
            if verbose:
                print(f"❌ {error_msg}")
                print(f"   Expected: config['credentials_path'] pointing to Google service account JSON")
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": "Missing credentials_path",
                "summary": error_msg
            }
        
        if verbose:
            print(f"🔍 Configuration validation passed:")
            print(f"   📊 Spreadsheet: {spreadsheet_title}")
            print(f"   📋 Worksheet: {worksheet_name}")
            print(f"   🔑 Credentials: {credentials_path}")
            print(f"   📁 Data fields: {list(extracted_data.keys())}")
        
        # Load Google Sheets client with comprehensive error handling
        try:
            if verbose:
                print(f"🔐 Attempting to authenticate with Google Sheets...")
            gc = load_gspread_client(credentials_path)
            if verbose:
                print(f"✅ Google Sheets authentication successful")
        except FileNotFoundError as e:
            error_msg = f"Credentials file not found: {credentials_path}"
            if verbose:
                print(f"❌ {error_msg}")
                print(f"   Please check if the file exists and the path is correct")
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": "Credentials file not found",
                "summary": error_msg
            }
        except ImportError as e:
            error_msg = f"Missing Google Sheets dependencies: {e}"
            if verbose:
                print(f"❌ {error_msg}")
                print(f"   Install with: pip install gspread google-auth")
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": "Missing dependencies",
                "summary": error_msg
            }
        except Exception as client_error:
            error_msg = f"Failed to authenticate with Google Sheets: {client_error}"
            if verbose:
                print(f"❌ {error_msg}")
                print(f"   Please check your credentials file and Google service account permissions")
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": str(client_error),
                "summary": error_msg
            }
        
        # Test sheet access before writing
        try:
            if verbose:
                print(f"🔍 Testing access to spreadsheet '{spreadsheet_title}' and worksheet '{worksheet_name}'...")
            
            # Try to open the spreadsheet first
            try:
                spreadsheet = gc.open(spreadsheet_title)
                if verbose:
                    print(f"✅ Spreadsheet '{spreadsheet_title}' found")
            except Exception as e:
                error_msg = f"Spreadsheet '{spreadsheet_title}' not found or not accessible"
                if verbose:
                    print(f"❌ {error_msg}")
                    print(f"   Error: {e}")
                    print(f"   Please check if the spreadsheet exists and your service account has access")
                return {
                    "success": False,
                    "data": extracted_data,
                    "type": "sheets",
                    "message": error_msg,
                    "error": str(e),
                    "summary": error_msg
                }
            
            # Try to access the worksheet
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
                if verbose:
                    print(f"✅ Worksheet '{worksheet_name}' found")
            except Exception as e:
                error_msg = f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet_title}'"
                if verbose:
                    print(f"❌ {error_msg}")
                    print(f"   Error: {e}")
                    print(f"   Please check if the worksheet exists or create it first")
                return {
                    "success": False,
                    "data": extracted_data,
                    "type": "sheets",
                    "message": error_msg,
                    "error": str(e),
                    "summary": error_msg
                }
            
            if verbose:
                print(f"✅ Sheet access verified, proceeding to write data...")
        
        except Exception as access_error:
            error_msg = f"Failed to verify sheet access: {access_error}"
            if verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": str(access_error),
                "summary": error_msg
            }
        
        # Write to sheet with error handling
        try:
            sheet_resp = append_to_sheet(gc, config, extracted_data)

            success_message = f"Successfully saved your information to Google Sheets"

            # Build a human-readable summary: number of non-empty fields and which sheet
            non_empty_fields = [k for k, v in extracted_data.items() if v not in (None, "", [], {})]
            num_fields = len(non_empty_fields)
            sample_fields = ", ".join(non_empty_fields[:5]) if non_empty_fields else "(no fields)"
            if len(non_empty_fields) > 5:
                sample_fields += ", ..."

            spreadsheet_title = config['sheet'].get('spreadsheet_title')
            worksheet_name = config['sheet'].get('worksheet_name')
            summary = f"Saved {num_fields} field(s) ({sample_fields}) to spreadsheet '{spreadsheet_title}' (worksheet: '{worksheet_name}')."

            tool_result = {
                "success": True,
                "data": extracted_data,
                "type": "sheets",
                "message": success_message,
                "sheet_info": {
                    "spreadsheet": spreadsheet_title,
                    "worksheet": worksheet_name
                },
                "summary": summary,
                # attach raw append response if available for debugging (may be True or metadata)
                "append_response": sheet_resp
            }

            if verbose:
                print(f"✅ Data successfully written to Google Sheets!")

            return tool_result
            
        except Exception as sheet_error:
            error_msg = f"Failed to write to Google Sheets: {sheet_error}"
            if verbose:
                print(f"❌ {error_msg}")

            summary = f"Failed to write to spreadsheet '{config.get('sheet',{}).get('spreadsheet_title','<unknown>')}': {str(sheet_error)}"

            return {
                "success": False,
                "data": extracted_data,
                "type": "sheets",
                "message": error_msg,
                "error": str(sheet_error),
                "summary": summary
            }
            
    except Exception as e:
        error_msg = f"Sheets tool error: {e}"
        if verbose:
            print(f"❌ {error_msg}")
        
        summary = f"Sheets tool error: {str(e)}"
        return {
            "success": False,
            "data": extracted_data,
            "type": "sheets",
            "message": error_msg,
            "error": str(e),
            "summary": summary
        }

def web_search_tool(extracted_data: Dict[str, Any], config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Web search tool implementation using Tavily search API.
    
    Configuration options in config["web_search"]:
    - max_results: int (default: 5)
    - include_snippets: bool (default: True)
    - timeout: int (default: 10)
    
    Args:
        extracted_data: Search query data (expects "search_query" or "query" field)
        config: Tool configuration
        verbose: Enable verbose logging
        
    Returns:
        Search results with URLs, titles, and snippets from Tavily
    """
    if verbose:
        print("--- TOOL USE NODE (web_search) ---")
    
    # Extract search query from various possible field names
    query = (
        extracted_data.get("search_query") or 
        extracted_data.get("query") or 
        extracted_data.get("question") or
        "general search"
    )
    
    if verbose:
        print(f"Search query: {query}")
    
    # Check if Tavily is available and configured
    tavily_key = os.environ.get("tavily_api_key")
    if TAVILY_AVAILABLE and tavily_key:
        try:
            client = TavilyClient(api_key=tavily_key)
            top_k = config.get("max_results", 5)
            resp = client.search(query)
            results = []
            for r in resp.get("results", []):
                results.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": r.get("content") or "",
                    "score": r.get("score")
                })
            summary = f"Found {len(results)} results for: {query}"
            return {
                "success": True,
                "data": {
                    "query": query,
                    "engine": "tavily",
                    "results": results,
                    "result_count": len(results)
                },
                "type": "tavily",
                "message": summary,
                "summary": summary
            }
        except Exception as e:
            if verbose:
                print(f"WARNING: Tavily search failed: {e}")
            summary = f"Tavily search failed: {e}"
            return {
                "success": False,
                "data": {"query": query},
                "type": "web_search",
                "message": f"Tavily search failed: {e}",
                "error": str(e),
                "summary": summary
            }
    
    # If Tavily is not available, return error
    if verbose:
        print("WARNING: Tavily search not available. Set tavily_api_key environment variable")
    
    summary = "Web search unavailable: Tavily API key not configured"
    return {
        "success": False,
        "data": {"query": query},
        "type": "web_search",
        "message": "Web search unavailable: Tavily API key not configured",
        "error": "Missing Tavily API key",
        "summary": summary
    }


def email_tool(extracted_data: Dict[str, Any], config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Email tool implementation with configurable SMTP settings.
    
    Configuration options in config["email"]:
    - smtp_server: str (e.g., "smtp.gmail.com")
    - smtp_port: int (default: 587)
    - username: str (sender email)
    - password: str (app password or regular password)
    - use_tls: bool (default: True)
    - default_subject: str (default subject if not provided)
    
    Args:
        extracted_data: Email data (expects "email", "subject", "message" fields)
        config: Email configuration
        verbose: Enable verbose logging
        
    Returns:
        Email sending result
    """
    if verbose:
        print("--- TOOL USE NODE (email) ---")
    
    # Check for email dependencies
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
    except ImportError as e:
        return {
            "success": False,
            "data": extracted_data,
            "type": "email",
            "message": f"Email dependencies not available: {e}",
            "error": "Missing dependencies"
        }
    
    # Extract email data
    recipient = extracted_data.get("email") or extracted_data.get("recipient")
    subject = extracted_data.get("subject") or config.get("email", {}).get("default_subject", "Message from Assistant")
    message = extracted_data.get("message") or extracted_data.get("body") or "No message content provided"
    
    if not recipient:
        return {
            "success": False,
            "data": extracted_data,
            "type": "email",
            "message": "No recipient email address provided",
            "error": "Missing recipient"
        }
    
    # Get email configuration
    email_config = config.get("email", {})
    smtp_server = email_config.get("smtp_server")
    smtp_port = email_config.get("smtp_port", 587)
    username = email_config.get("username")
    password = email_config.get("password")
    use_tls = email_config.get("use_tls", True)
    
    if not all([smtp_server, username, password]):
        if verbose:
            print("⚠️ Email configuration incomplete. Required: smtp_server, username, password")
        
        return {
            "success": False,
            "data": extracted_data,
            "type": "email",
            "message": "Email configuration incomplete (missing smtp_server, username, or password)",
            "error": "Incomplete configuration"
        }
    
    if verbose:
        print(f"Sending email to: {recipient}")
        print(f"Subject: {subject}")
        print(f"SMTP Server: {smtp_server}:{smtp_port}")
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if use_tls:
                server.starttls()
            server.login(username, password)
            server.send_message(msg)
        
        if verbose:
            print("✅ Email sent successfully")
        
        return {
            "success": True,
            "data": {
                "recipient": recipient,
                "subject": subject,
                "message": message,
                "smtp_server": smtp_server
            },
            "type": "email",
            "message": f"Email sent successfully to {recipient}"
        }
        
    except Exception as e:
        error_msg = f"Email sending failed: {e}"
        if verbose:
            print(f"❌ {error_msg}")
        
        return {
            "success": False,
            "data": extracted_data,
            "type": "email", 
            "message": error_msg,
            "error": str(e)
        }

# Tool registry for easy lookup
AVAILABLE_TOOLS = {
    "sheets": sheets_tool,
    "web_search": web_search_tool,
    "tavily": web_search_tool,
    "api_retrieval": api_retrieval_tool,
}

def execute_tool(tool_type: str, extracted_data: Dict[str, Any], config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Execute a tool by type with error handling.
    
    Args:
        tool_type: Type of tool to execute
        extracted_data: Data for the tool
        config: Tool configuration
        verbose: Enable verbose logging
        
    Returns:
        Tool execution result
    """
    try:
        # Debug: show which tool is requested and what tools are available
        # concise logging removed (no DEBUG prints)
        tool_func = AVAILABLE_TOOLS.get(tool_type, AVAILABLE_TOOLS["generic"])
        # concise logging removed
        return tool_func(extracted_data, config, verbose)
    except Exception as e:
        return {
            "success": False,
            "data": extracted_data,
            "type": tool_type,
            "message": f"Tool execution failed: {e}",
            "error": str(e)
        }


