import os
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Scopes required for Google Sheets and Google Drive
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

CREDENTIALS_FILE = "google_credentials.json"

def get_gspread_client():
    if not os.path.exists(CREDENTIALS_FILE):
        logger.error(f"Credentials file not found: {CREDENTIALS_FILE}")
        return None
    
    try:
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Sheets: {e}")
        return None

def sync_good_fit_channels(channels):
    """
    Syncs a list of channel dictionaries to the specified Google Sheet.
    Only adds new entries based on Channel URL.
    """
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    tab_name = os.getenv("GOOGLE_SHEET_TAB", "Sheet1")
    
    if not sheet_id:
        return False, "GOOGLE_SHEET_ID not found in environment."
    
    client = get_gspread_client()
    if not client:
        return False, "Failed to connect to Google Sheets."
    
    try:
        spreadsheet = client.open_by_key(sheet_id)
        
        # Try to open the tab, create it if it doesn't exist
        try:
            worksheet = spreadsheet.worksheet(tab_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=tab_name, rows="100", cols="20")
            logger.info(f"Created new tab: {tab_name}")
            
        # Get all existing data
        all_values = worksheet.get_all_values()
        
        headers = ["Channel Name", "Channel URL", "Email", "Contact Status"]
        
        if not all_values:
            # Sheet is empty, add headers
            worksheet.append_row(headers)
            existing_urls = set()
            logger.info("Added headers to empty sheet.")
        else:
            # Find column index for "Channel URL" (usually index 1, which is column B)
            try:
                url_col_idx = all_values[0].index("Channel URL")
            except ValueError:
                # If header is missing or different, assume column B (index 1)
                url_col_idx = 1
                logger.warning("Could not find 'Channel URL' header. Assuming column B.")
            
            existing_urls = {row[url_col_idx] for row in all_values if len(row) > url_col_idx}
            
        new_rows = []
        for ch in channels:
            # Ensure the URL is in the correct format (youtube.com/channel/ID)
            # ch['ID'] is the channel ID (e.g. UC...)
            channel_url = f"https://www.youtube.com/channel/{ch['ID']}"
            
            if channel_url not in existing_urls:
                # Add new row: [Name, URL, "", ""]
                new_rows.append([ch['Channel'], channel_url, "", ""])
        
        if new_rows:
            worksheet.append_rows(new_rows)
            logger.info(f"Appended {len(new_rows)} new channels to Google Sheets.")
            return True, f"Successfully synced {len(new_rows)} new channels!"
        else:
            logger.info("No new channels to sync.")
            return True, "All channels are already in the sheet."
            
    except Exception as e:
        logger.error(f"Error syncing to Google Sheets: {e}")
        return False, f"Error: {str(e)}"
