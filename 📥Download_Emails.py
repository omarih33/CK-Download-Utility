import streamlit as st
import requests
from datetime import datetime, time
import pandas as pd
import shutil
import os
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

# Constants
API_BASE_URL = "https://api.kit.com/v4"
DEFAULT_PER_PAGE = 500

class KitAPI:
    """
    Kit API client handling authentication and API requests
    """
    def __init__(self, api_secret: str):
        self.api_secret = api_secret
        self.headers = {
            'Accept': 'application/json',
            'X-Kit-Api-Key': api_secret
        }

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, json: Optional[Dict] = None) -> requests.Response:
        """
        Make an API request with error handling
        """
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.request(method, url, headers=self.headers, params=params, json=json)
        
        if response.status_code == 429:
            st.error("Rate limit exceeded. Please wait a moment and try again.")
            return None
        
        response.raise_for_status()
        return response

    @st.cache_data(show_spinner=False)
    def get_broadcasts(self, cursor: Optional[str] = None, per_page: int = DEFAULT_PER_PAGE) -> Dict:
        """
        Get broadcasts with cursor-based pagination
        """
        params = {'per_page': per_page}
        if cursor:
            params['after'] = cursor
            
        response = self._make_request('GET', 'broadcasts', params=params)
        return response.json() if response else None

    @st.cache_data(show_spinner=False)
    def get_broadcast_details(self, broadcast_id: int) -> Dict:
        """
        Get detailed information for a specific broadcast
        """
        response = self._make_request('GET', f'broadcasts/{broadcast_id}')
        return response.json().get('broadcast') if response else None

    @st.cache_data(show_spinner=False)
    def get_broadcast_stats(self, broadcast_id: int) -> Dict:
        """
        Get statistics for a specific broadcast
        """
        response = self._make_request('GET', f'broadcasts/{broadcast_id}/stats')
        return response.json().get('broadcast', {}).get('stats', {}) if response else None

def remove_html_tags(html_content: str) -> str:
    """
    Clean HTML content while preserving link URLs
    """
    if not html_content:
        return ""
        
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Replace links with their URLs
    for a_tag in soup.find_all('a'):
        a_tag.replace_with(f" {a_tag.get('href')} ")
    
    # Get text content without HTML tags
    text_content = soup.get_text(separator=' ', strip=True)
    return text_content

def download_broadcasts(api_client: KitAPI, date_range: List[str], published_filter: str, public_filter: str) -> pd.DataFrame:
    """
    Download broadcasts with filtering and pagination
    """
    all_broadcasts = []
    cursor = None
    start_datetime = datetime.fromisoformat(date_range[0].replace('Z', '+00:00'))
    end_datetime = datetime.fromisoformat(date_range[1].replace('Z', '+00:00'))

    progress_bar = st.progress(0, text="Downloading broadcasts...")
    processed_count = 0

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
            progress_bar.progress(min(processed_count / 100, 1.0))  # Assume max 100 broadcasts for progress

        pagination = response_data.get('pagination', {})
        if not pagination.get('has_next_page'):
            break
        cursor = pagination.get('end_cursor')

    return pd.DataFrame(all_broadcasts) if all_broadcasts else pd.DataFrame()

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
