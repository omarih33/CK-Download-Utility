import streamlit as st
import requests
import time as tm 
import shutil
from typing import List 
import os

import pandas as pd
from datetime import datetime
from datetime import time

from pathlib import Path

from io import StringIO




# Prepare the data
from bs4 import BeautifulSoup




@st.cache_data(show_spinner=False)
def get_broadcasts(api_secret, page=1):
    url = f"https://api.convertkit.com/v3/broadcasts?page={page}&api_secret={api_secret}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["broadcasts"]
    return []

@st.cache_data(show_spinner=False)
def get_broadcast_details(api_secret, broadcast_id):
    url = f"https://api.convertkit.com/v3/broadcasts/{broadcast_id}?api_secret={api_secret}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "broadcast" in data:
            return data["broadcast"]
    return {}

@st.cache_data(show_spinner=False)
def get_broadcast_stats(api_secret, broadcast_id):
    url = f"https://api.convertkit.com/v3/broadcasts/{broadcast_id}/stats?api_secret={api_secret}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "broadcast" in data:
            return data["broadcast"]["stats"]
    return {}

@st.cache_data(show_spinner=False)
def download_broadcasts(api_secret, date_range, published_filter, public_filter):
    all_broadcasts = []
    page = 1

    # Convert the strings in date_range to datetime objects
    start_datetime = datetime.fromisoformat(date_range[0].replace('Z', '+00:00'))
    end_datetime = datetime.fromisoformat(date_range[1].replace('Z', '+00:00'))

    # Estimate the total number of broadcasts
    estimated_total_broadcasts = estimate_total_broadcasts(api_secret)

    # Create a progress bar
    progress_text = "Downloading broadcasts. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    processed_broadcasts = 0


    while True:
        broadcasts = get_broadcasts(api_secret, page)
        if not broadcasts:
            break
        for broadcast in broadcasts:
            broadcast_details = get_broadcast_details(api_secret, broadcast["id"])
            if "created_at" in broadcast_details:
                created_at = datetime.fromisoformat(broadcast_details["created_at"].replace('Z', '+00:00'))
                # Compare the datetime objects directly
                if start_datetime <= created_at <= end_datetime:
                    if (published_filter == "All" or
                        (published_filter == "Published" and broadcast_details["send_at"] is not None) or
                        (published_filter == "Draft" and broadcast_details["send_at"] is None)):
                        if (public_filter == "All" or
                            (public_filter == "Public" and broadcast_details["public"]) or
                            (public_filter == "Private" and not broadcast_details["public"])):
                            broadcast_stats = get_broadcast_stats(api_secret, broadcast["id"])
                            broadcast_details.update(broadcast_stats)
                            broadcast.update(broadcast_details)
                            # Parse the HTML in the 'content' column using BeautifulSoup and extract only the text
                            soup = BeautifulSoup(broadcast['content'], 'html.parser')
                            broadcast['content'] = soup.get_text()
                            all_broadcasts.append(broadcast)
            processed_broadcasts += 1
            # Update the progress bar
            progress = processed_broadcasts / estimated_total_broadcasts
            progress_bar.progress(progress, text=progress_text)
            tm.sleep(0.33)
        page += 1

    df = pd.DataFrame(all_broadcasts)
    return df


@st.cache_data(show_spinner=False)
def estimate_total_broadcasts(api_secret):
    page = 1
    total_pages = 0
    while True:
        url = f"https://api.convertkit.com/v3/broadcasts?page={page}&api_secret={api_secret}"
        response = requests.get(url)
        if response.status_code == 200:
            broadcasts = response.json()["broadcasts"]
            if not broadcasts:
                break
            total_pages += 1
            page += 1
        else:
            break
    # Estimate the total number of broadcasts by multiplying the number of pages by 50
    estimated_total_broadcasts = total_pages * 50
    return estimated_total_broadcasts







def save_to_markdown_and_create_zip(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for index, row in df.iterrows():
        filename = f"{row.iloc[0]}.md"  # Use the value in the first column as the filename
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w') as f:
            # Remove leading and trailing white space from each row
            row = row.str.strip()
            f.write(row.to_markdown())
    # Create a ZIP file containing all the Markdown files
    shutil.make_archive(output_dir, 'zip', output_dir)
    return f"{output_dir}.zip"


# Initialize session state with an empty "df" key
if "df" not in st.session_state:
    st.session_state.df = None

st.title("ConvertKit Email Utility")
st.markdown("Use this to download your email broadcasts.")
st.markdown("Query them in the next tab.")
st.markdown("Start by pasting your API key below. You can find it in your CK Account Settings.")

api_secret = st.text_input("API Secret", type="password")
if api_secret:
    # Create three columns for the date input, published filter, and public filter
    col1, col2, col3 = st.columns(3)

    # Get the date range from the user input and place it in the first column
    date_range_input = col1.date_input("Date range", [datetime(2000, 1, 1).date(), datetime.now().date()])

    # Combine the start and end dates with the time set to 0:00 (midnight)
    start_datetime = datetime.combine(date_range_input[0], time.min).isoformat() + "Z"
    end_datetime = datetime.combine(date_range_input[1], time.min).isoformat() + "Z"

    # Update the date range with the new datetime objects
    date_range = [start_datetime, end_datetime]

    # Place the published filter in the second column
    published_filter = col2.selectbox("Published status", ["All", "Published", "Draft"])

    # Place the public filter in the third column
    public_filter = col3.selectbox("Public status", ["All", "Public", "Private"])

    
    if st.button("Download Broadcasts"):
        df = download_broadcasts(api_secret, date_range, published_filter, public_filter)
        st.session_state.df = df  # Store the DataFrame in the session state
        st.download_button(
            label="Save to CSV",
            data=df.to_csv(index=False),
            file_name="broadcasts_and_stats_with_content_filtered.csv",
            mime="text/csv",
        )
    # Function to run the GPTPandasIndex QnA

    # Create a button to save each row to a Markdown file and download the ZIP file
    if st.button("Save to Markdown"):
        output_dir = 'output_markdown'  # Define the output directory
        zip_file = save_to_markdown_and_create_zip(st.session_state.df, output_dir)
        st.download_button(
            label="Download Markdown ZIP",
            data=open(zip_file, "rb"),
            file_name="broadcasts_markdown.zip",
            mime="application/zip",
        )


# Display the DataFrame using st.dataframe
if st.session_state.df is not None:
    st.dataframe(st.session_state.df, use_container_width=True)

    


