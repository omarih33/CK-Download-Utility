import streamlit as st
import requests
from datetime import datetime, time
import pandas as pd
import shutil
import os
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

class KitAPI:
    """Handles interaction with the Kit API"""
    def __init__(self, api_secret: str):
        self.api_secret = api_secret
        self.headers = {
            'Accept': 'application/json',
            'X-Kit-Api-Key': api_secret
        }

    def get_broadcasts(self, cursor: Optional[str] = None, per_page: int = 1000) -> Dict:
        """Fetches broadcasts from Kit API with cursor-based pagination"""
        @st.cache_data(show_spinner=False, ttl=300)
        def _cached_get_broadcasts(api_secret: str, cursor: Optional[str], per_page: int) -> Dict:
            url = "https://api.kit.com/v4/broadcasts"
            params = {'per_page': per_page}
            if cursor:
                params['after'] = cursor
                
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 429:
                st.warning("We need to slow down a bit. Taking a short break...")
                return None
                
            response.raise_for_status()
            return response.json() if response.ok else None
            
        return _cached_get_broadcasts(self.api_secret, cursor, per_page)

def clean_content(html_content: str) -> str:
    """Cleans HTML content while preserving readable text and links"""
    if not isinstance(html_content, str):
        return ""
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Keep links readable
    for a_tag in soup.find_all('a'):
        href = a_tag.get('href', '')
        text = a_tag.get_text(strip=True)
        a_tag.replace_with(f"{text} ({href})")
    
    return soup.get_text(separator=' ', strip=True)

def main():
    st.title("Kit Email Archive Tool")
    
    st.markdown("""
    Welcome! This tool helps you download and save your Kit (formerly ConvertKit) email broadcasts. 
    Whether you want to create a backup, analyze your content, or just keep a record of your emails, 
    I'll guide you through the process step by step.
    """)
    
    with st.expander("🔑 Where to Find Your API Key", expanded=True):
        st.markdown("""
        To get started, you'll need your Kit API key. Here's how to find it:
        
        1. Log into your Kit account
        2. Go to Settings (click your avatar in the top right)
        3. Look for "API & Webhooks"
        4. Your API key will be listed there. If you don't see one, you can create a new one
        
        Your API key is like a secure password for accessing your data - keep it safe and don't share it publicly.
        """)
    
    api_secret = st.text_input(
        "Enter your Kit API Key",
        type="password",
        help="This is used to securely access your email data"
    )

    if api_secret:
        st.markdown("### Download Options")
        st.markdown("""
        Let's configure your download. You can choose what emails to include based on their date 
        and status. This helps you get exactly the data you need.
        """)
        
        # Create layout columns for filters
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("📅 **Date Range**")
            date_range = st.date_input(
                "Select period",
                [datetime(2000, 1, 1).date(), datetime.now().date()],
                help="Choose the time period for emails you want to download"
            )

        with col2:
            st.markdown("📬 **Email Status**")
            published_filter = st.selectbox(
                "Published status",
                ["All", "Published", "Draft"],
                help="Filter emails by their publication status"
            )

        with col3:
            st.markdown("🌐 **Visibility**")
            public_filter = st.selectbox(
                "Public status",
                ["All", "Public", "Private"],
                help="Filter emails by their visibility setting"
            )

        if st.button("Start Download", help="Click to begin downloading your emails"):
            try:
                with st.status("Downloading your emails...") as status:
                    api_client = KitAPI(api_secret)
                    
                    # Convert dates to ISO format
                    start_datetime = datetime.combine(date_range[0], time.min).isoformat() + "Z"
                    end_datetime = datetime.combine(date_range[1], time.min).isoformat() + "Z"
                    
                    # Initialize download tracking
                    all_broadcasts = []
                    cursor = None
                    processed = 0
                    
                    while True:
                        status.write(f"Downloaded {processed} emails so far...")
                        response = api_client.get_broadcasts(cursor=cursor)
                        
                        if not response:
                            break
                            
                        broadcasts = response.get('broadcasts', [])
                        if not broadcasts:
                            break
                            
                        # Process the batch
                        for broadcast in broadcasts:
                            processed += 1
                            if _matches_filters(broadcast, published_filter, public_filter):
                                all_broadcasts.append(broadcast)
                        
                        # Check for more pages
                        pagination = response.get('pagination', {})
                        if not pagination.get('has_next_page'):
                            break
                        cursor = pagination.get('end_cursor')
                    
                    # Create DataFrame
                    df = pd.DataFrame(all_broadcasts)
                    if not df.empty:
                        st.session_state.df = df
                        
                        # Create download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                label="📥 Download as CSV",
                                data=df.to_csv(index=False),
                                file_name="kit_broadcasts.csv",
                                mime="text/csv",
                                help="Save your emails in CSV format - great for spreadsheets"
                            )
                        
                        with col2:
                            st.download_button(
                                label="📝 Download as Markdown",
                                data=_create_markdown_archive(df),
                                file_name="kit_broadcasts.md",
                                mime="text/markdown",
                                help="Save your emails in Markdown format - perfect for reading"
                            )
                            
                        st.success(f"Successfully downloaded {len(df)} emails! You can now save them using the buttons above.")
                        
                        st.markdown("""
                        ### What's Next?
                        
                        Now that you have your emails downloaded, you can:
                        - Keep them as a backup of your content
                        - Import them into other tools for analysis
                        - Use them as reference for future emails
                        - Create an organized archive of your work
                        """)
                    else:
                        st.info("No emails found matching your criteria. Try adjusting the filters.")
                        
            except Exception as e:
                st.error(f"Something went wrong during the download: {str(e)}")
                st.markdown("""
                Common issues:
                - Invalid API key
                - Internet connection problems
                - Kit API service interruption
                
                Please try again in a few moments. If the problem persists, double-check your API key.
                """)

def _matches_filters(broadcast: Dict, published_filter: str, public_filter: str) -> bool:
    """Checks if a broadcast matches the selected filters"""
    if published_filter != "All":
        is_published = broadcast.get("send_at") is not None
        if (published_filter == "Published") != is_published:
            return False

    if public_filter != "All":
        is_public = broadcast.get("public", False)
        if (public_filter == "Public") != is_public:
            return False

    return True

def _create_markdown_archive(df: pd.DataFrame) -> str:
    """Creates a readable Markdown archive of emails"""
    markdown_content = "# Your Kit Email Archive\n\n"
    
    for _, row in df.iterrows():
        markdown_content += f"## {row.get('email_name', 'Untitled Email')}\n\n"
        markdown_content += f"**Sent**: {row.get('created_at', 'Unknown date')}\n\n"
        markdown_content += f"**Open Rate**: {row.get('open_rate', 0)*100:.1f}%\n\n"
        markdown_content += f"**Click Rate**: {row.get('click_rate', 0)*100:.1f}%\n\n"
        markdown_content += "### Content\n\n"
        markdown_content += f"{clean_content(row.get('content', ''))}\n\n"
        markdown_content += "---\n\n"
    
    return markdown_content

if __name__ == "__main__":
    main()
