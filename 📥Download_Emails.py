import streamlit as st
import requests
from datetime import datetime, time
import pandas as pd
import shutil
import os
import time as time_module
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from collections import deque
from datetime import datetime, timedelta

# Constants
API_BASE_URL = "https://api.kit.com/v4"
DEFAULT_PER_PAGE = 500
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 120  # for API key auth

class RateLimiter:
    """Handles rate limiting for API requests"""
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.retry_count = 0
        self.max_retries = 3

    def wait_if_needed(self):
        """Calculate wait time and sleep if necessary"""
        now = datetime.now()
        
        # Remove requests outside the current window
        while self.requests and self.requests[0] < now - timedelta(seconds=self.window_seconds):
            self.requests.popleft()

        # If we're at the limit, wait until we can make another request
        if len(self.requests) >= self.max_requests:
            oldest_request = self.requests[0]
            wait_time = (oldest_request + timedelta(seconds=self.window_seconds) - now).total_seconds()
            if wait_time > 0:
                # Add exponential backoff if we've retried multiple times
                if self.retry_count > 0:
                    wait_time *= (2 ** self.retry_count)
                    wait_time = min(wait_time, 60)  # Cap at 60 seconds
                
                st.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds before retrying...")
                time_module.sleep(wait_time)
                self.retry_count += 1
                return True

        # Add current request to the window
        self.requests.append(now)
        self.retry_count = 0
        return False

class KitAPI:
    """Kit API client handling authentication and API requests"""
    def __init__(self, api_secret: str):
        self.api_secret = api_secret
        self.headers = {
            'Accept': 'application/json',
            'X-Kit-Api-Key': api_secret
        }
        self.rate_limiter = RateLimiter(MAX_REQUESTS_PER_WINDOW, RATE_LIMIT_WINDOW)

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, json: Optional[Dict] = None) -> requests.Response:
        """Make an API request with rate limiting and error handling"""
        url = f"{API_BASE_URL}/{endpoint}"
        
        while True:
            # Check rate limit and wait if necessary
            self.rate_limiter.wait_if_needed()
            
            try:
                response = requests.request(method, url, headers=self.headers, params=params, json=json)
                
                if response.status_code == 429:
                    if self.rate_limiter.retry_count >= self.rate_limiter.max_retries:
                        st.error("Maximum retry attempts reached. Please try again later.")
                        return None
                    continue  # Try again after waiting
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                st.error(f"API request error: {str(e)}")
                return None

    def get_broadcasts(self, cursor: Optional[str] = None, per_page: int = DEFAULT_PER_PAGE) -> Dict:
        """Get broadcasts with cursor-based pagination"""
        @st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
        def _cached_get_broadcasts(api_secret: str, cursor: Optional[str], per_page: int) -> Dict:
            headers = {
                'Accept': 'application/json',
                'X-Kit-Api-Key': api_secret
            }
            params = {'per_page': per_page}
            if cursor:
                params['after'] = cursor

            url = f"{API_BASE_URL}/broadcasts"
            
            # Use the rate limiter directly in cached function
            while True:
                self.rate_limiter.wait_if_needed()
                
                try:
                    response = requests.get(url, headers=headers, params=params)
                    if response.status_code == 429:
                        continue
                    response.raise_for_status()
                    return response.json() if response.ok else None
                except requests.exceptions.RequestException:
                    return None

        return _cached_get_broadcasts(self.api_secret, cursor, per_page)

    def get_broadcast_details(self, broadcast_id: int) -> Dict:
        """Get detailed information for a specific broadcast"""
        @st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
        def _cached_get_broadcast_details(api_secret: str, broadcast_id: int) -> Dict:
            headers = {
                'Accept': 'application/json',
                'X-Kit-Api-Key': api_secret
            }
            url = f"{API_BASE_URL}/broadcasts/{broadcast_id}"
            
            while True:
                self.rate_limiter.wait_if_needed()
                
                try:
                    response = requests.get(url, headers=headers)
                    if response.status_code == 429:
                        continue
                    response.raise_for_status()
                    return response.json().get('broadcast') if response.ok else None
                except requests.exceptions.RequestException:
                    return None

        return _cached_get_broadcast_details(self.api_secret, broadcast_id)

    def get_broadcast_stats(self, broadcast_id: int) -> Dict:
        """Get statistics for a specific broadcast"""
        @st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
        def _cached_get_broadcast_stats(api_secret: str, broadcast_id: int) -> Dict:
            headers = {
                'Accept': 'application/json',
                'X-Kit-Api-Key': api_secret
            }
            url = f"{API_BASE_URL}/broadcasts/{broadcast_id}/stats"
            
            while True:
                self.rate_limiter.wait_if_needed()
                
                try:
                    response = requests.get(url, headers=headers)
                    if response.status_code == 429:
                        continue
                    response.raise_for_status()
                    return response.json().get('broadcast', {}).get('stats', {}) if response.ok else None
                except requests.exceptions.RequestException:
                    return None

        return _cached_get_broadcast_stats(self.api_secret, broadcast_id)

def download_broadcasts(api_client: KitAPI, date_range: List[str], published_filter: str, public_filter: str) -> pd.DataFrame:
    """Download broadcasts with filtering and pagination"""
    all_broadcasts = []
    cursor = None
    start_datetime = datetime.fromisoformat(date_range[0].replace('Z', '+00:00'))
    end_datetime = datetime.fromisoformat(date_range[1].replace('Z', '+00:00'))

    progress_container = st.empty()
    progress_bar = progress_container.progress(0)
    processed_count = 0
    total_processed = 0
    batch_size = DEFAULT_PER_PAGE

    try:
        # First, get total count for better progress tracking
        initial_response = api_client.get_broadcasts(per_page=1)
        if initial_response and 'pagination' in initial_response:
            # Estimate total based on what we can see
            total_processed = len(initial_response.get('broadcasts', []))

        while True:
            response_data = api_client.get_broadcasts(cursor=cursor)
            if not response_data:
                break

            broadcasts = response_data.get('broadcasts', [])
            if not broadcasts:
                break

            for broadcast in broadcasts:
                details = api_client.get_broadcast_details(broadcast["id"])
                if not details or "created_at" not in details:
                    continue

                created_at = datetime.fromisoformat(details["created_at"].replace('Z', '+00:00'))
                
                if start_datetime <= created_at <= end_datetime:
                    if _matches_filters(details, published_filter, public_filter):
                        stats = api_client.get_broadcast_stats(broadcast["id"])
                        details.update(stats or {})
                        broadcast.update(details)
                        all_broadcasts.append(broadcast)

                processed_count += 1
                progress = min(processed_count / max(total_processed, batch_size), 1.0)
                progress_bar.progress(progress, f"Downloaded {len(all_broadcasts)} broadcasts...")

            pagination = response_data.get('pagination', {})
            if not pagination.get('has_next_page'):
                break
            cursor = pagination.get('end_cursor')

        progress_container.empty()  # Clean up progress bar
        st.success(f"Successfully downloaded {len(all_broadcasts)} broadcasts")
        
        return pd.DataFrame(all_broadcasts) if all_broadcasts else pd.DataFrame()
        
    except Exception as e:
        progress_container.empty()  # Clean up progress bar
        st.error(f"Error downloading broadcasts: {str(e)}")
        return pd.DataFrame()


def _matches_filters(broadcast: Dict, published_filter: str, public_filter: str) -> bool:
    """
    Check if broadcast matches the selected filters
    """
    if published_filter != "All":
        is_published = broadcast.get("send_at") is not None
        if (published_filter == "Published") != is_published:
            return False

    if public_filter != "All":
        is_public = broadcast.get("public", False)
        if (public_filter == "Public") != is_public:
            return False

    return True

def save_to_markdown(df: pd.DataFrame, output_dir: str) -> str:
    """
    Save broadcasts to individual markdown files and create a ZIP
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for _, row in df.iterrows():
        filename = f"{row.iloc[0]}.md"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            row = row.str.strip()
            f.write(row.to_markdown())

    # Create ZIP archive
    shutil.make_archive(output_dir, 'zip', output_dir)
    return f"{output_dir}.zip"

def main():
    st.title("Kit Email Utility")
    st.markdown("Download and analyze your email broadcasts from Kit (formerly ConvertKit).")
    
    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None

    api_secret = st.text_input("API Secret", type="password", 
                              help="Find this in your Kit Account Settings")

    if api_secret:
        api_client = KitAPI(api_secret)

        # Create layout columns
        col1, col2, col3 = st.columns(3)

        # Date range selector
        date_range = col1.date_input(
            "Date range",
            [datetime(2000, 1, 1).date(), datetime.now().date()]
        )

        # Convert dates to ISO format with timezone
        start_datetime = datetime.combine(date_range[0], time.min).isoformat() + "Z"
        end_datetime = datetime.combine(date_range[1], time.min).isoformat() + "Z"
        date_range = [start_datetime, end_datetime]

        # Filters
        published_filter = col2.selectbox("Published status", ["All", "Published", "Draft"])
        public_filter = col3.selectbox("Public status", ["All", "Public", "Private"])

        if st.button("Download Broadcasts"):
            try:
                df = download_broadcasts(api_client, date_range, published_filter, public_filter)
                st.session_state.df = df
                
                if not df.empty:
                    st.download_button(
                        label="Save to CSV",
                        data=df.to_csv(index=False),
                        file_name="kit_broadcasts.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No broadcasts found matching your criteria.")
            except Exception as e:
                st.error(f"Error downloading broadcasts: {str(e)}")

        if st.button("Save to Markdown") and st.session_state.df is not None:
            try:
                output_dir = 'output_markdown'
                zip_file = save_to_markdown(st.session_state.df, output_dir)
                
                with open(zip_file, "rb") as f:
                    st.download_button(
                        label="Download Markdown ZIP",
                        data=f,
                        file_name="broadcasts_markdown.zip",
                        mime="application/zip",
                    )
            except Exception as e:
                st.error(f"Error creating Markdown files: {str(e)}")

        # Display the DataFrame
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df, use_container_width=True)

if __name__ == "__main__":
    main()
