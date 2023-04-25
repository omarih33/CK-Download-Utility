import os
import logging
import sys
import re
import streamlit as st
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Union
from langchain import SQLDatabase, SQLDatabaseChain, SerpAPIWrapper, LLMChain, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.agents import AgentType, tool, load_tools, initialize_agent, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate, BaseChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, HumanMessage

from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, DateTime, Float, Boolean

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# App title
st.title("Query Your ConvertKit Emails")


# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


# Create a SQL table schema for the data
engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData(bind=engine)

cast_table = Table(
    "emails",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime),
    Column("email_name", String),
    Column("description", String),
    Column("content", String),
    Column("public", Boolean),
    Column("published_at", DateTime),
    Column("send_at", DateTime),
    Column("thumbnail_alt", String),
    Column("thumbnail_url", String),
    Column("sent_from", String),
    Column("email_layout_template", String),
    Column("recipients", Integer),
    Column("open_rate", Float),
    Column("click_rate", Float),
    Column("unsubscribes", Integer),
    Column("total_clicks", Integer),
    Column("show_total_clicks", Boolean),
    Column("status", String),
    Column("progress", Float),
    Column("open_tracking_disabled", Boolean),
    Column("click_tracking_disabled", Boolean),
)

metadata_obj.create_all()



# Function to remove HTML tags and keep link URLs intact
def remove_html_tags(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find all 'a' tags and replace them with their URLs
    for a_tag in soup.find_all('a'):
        a_tag.replace_with(a_tag.get('href'))
    
    # Get the text content without HTML tags
    text_content = soup.get_text()
    
    return text_content
if uploaded_file is not None:
    # Load data into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Rename the 'subject' column to 'email_name'
    df = df.rename(columns={"subject": "email_name"})
    df = df.rename(columns={"email_address": "sent_from"})

    # Apply the function to the content column
    
    df['content'] = df['content'].apply(lambda x: remove_html_tags(str(x)))
    df = df.where(pd.notnull(df), None)


    # Ensure date columns are datetime objects in the original DataFrame
    date_columns = ["created_at", "published_at", "send_at"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Create a separate DataFrame for working with Chroma
    df_chroma = df.copy()

    # Convert date columns to string format in the df_chroma DataFrame
    for col in date_columns:
        df_chroma[col] = df_chroma[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    st.write(df_chroma)

    # Insert the data from the DataFrame into the SQL table
    with engine.connect() as connection:
        for index, row in df.iterrows():
            stmt = cast_table.insert().values(**row.to_dict())
            connection.execute(stmt)

    # Apply the function to the content column
    df_chroma['content'] = (
        + df_chroma['email_name']
        + "\n\n"
        + df_chroma['content'].apply(remove_html_tags)
        + "\n\n"
    )

    # Create an instance of DataFrameLoader with the DataFrame and the new 'combined_content' column
    loader = DataFrameLoader(df_chroma, page_content_column="content")

    # Load the documents from the DataFrameLoader
    documents = loader.load()

    # Create the chroma
    embeddings = OpenAIEmbeddings()
    content_index = Chroma.from_documents(documents, embeddings)

    



# Build the SQL index using the GPTSQLStructStoreIndex
sql_database = SQLDatabase(engine, include_tables=["emails"])



_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct query to run, then look at the results of the query and return the answer.
When using using exact-match email_names, be case-insensitive and use the LIKE function instead. 

Use the following format:

Question: "Question here"
SQLQuery: SQL Query to run
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Question: {input}
"""
PROMPT = PromptTemplate(
    input_variables=["input"], template=_DEFAULT_TEMPLATE
)



llm1 = OpenAI(temperature=0)
db_chain = SQLDatabaseChain(llm=llm1, database=sql_database, prompt=PROMPT, verbose=True)



@tool("Email Analytics")
def sql_index_tool(query: str) -> str:
    """Use this for email analytics. This table is a list of emails where columns are id, email_name, description, content, open_rate, click_rate, unsubscribes, total_clicks, recipients, sent_from, published at, send at, public, and thumbnail. Query structured data using SQL syntax."""
    query = query.replace('"', '')
    sql_response = db_chain.run(query)
    return f"Result of the SQL Query:\n{sql_response}\nThis is the final response. You do not need to parse text."







# Function to remove HTML tags and keep link URLs intact
def remove_html_tags(html_content):
    if isinstance(html_content, float):
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text().strip()


@tool("Email Retrieval and Display")
def print_email(query: str) -> str:
    """The Email Retrieval and Display tool will search for similar emails and display their content. Use this tool when you need to show the user an email."""
    
    # Find similar previous emails
    similar_emails = content_index.similarity_search(query, k=1)

    # Extract email content
    context = "\n\n".join([email.page_content for email in similar_emails])

    return f"\nThe email you requested is: {context}. \nUse the content above as your final Response."


@tool("Email Summarizer")
def summarize_email(query: str) -> str:
    """DO NOT USE Unless the query asks for a summary."""
    # Find similar previous emails
    similar_emails = content_index.similarity_search(query, k=1)

    # Extract email content
    context = "\n\n".join([email.page_content for email in similar_emails])

    # Custom prompt for generating emails
    summary_prompt_template = """Write a summary about the email below:
    Email: {context}
    Topic: {topic}
    Summary:"""

    SUMMARY_PROMPT = PromptTemplate(
        template=summary_prompt_template, input_variables=["context", "topic"]
    )

    # Create a new LLMChain with the custom email prompt
    summary_chain = LLMChain(llm=llm, prompt=SUMMARY_PROMPT)

    # Generate the new email
    summarized_email = summary_chain({"context": context, "topic": query})

    return f"Summarized email based on previous emails:\n{summarized_email}"


@tool("Email writer")
def generate_email(query: str) -> str:
    """This tool writes emails based on previous emails. Input is email subject."""
    # Find similar previous emails
    similar_emails = content_index.similarity_search(query, k=1)

    # Extract email content
    context = "\n\n".join([email.page_content for email in similar_emails])

    # Custom prompt for generating emails
    email_prompt_template = """You are an email copywriter: {topic}\n\n
    Email Context: {context}
    
    New email: """

    EMAIL_PROMPT = PromptTemplate(
        template=email_prompt_template, input_variables=["context", "topic"]
    )

    # Create a new LLMChain with the custom email prompt
    email_chain = LLMChain(llm=llm, prompt=EMAIL_PROMPT)

    # Generate the new email
    new_email = email_chain({"context": context, "topic": query})

    return f"{new_email}\n Parse your final Response from the text"

tools = [generate_email, sql_index_tool, summarize_email, print_email]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


llm = ChatOpenAI(verbose=True, temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)



# User input


user_input = st.text_input("Please ask a question or make a request(or 'q' to quit): ")

# Check if input is not empty and not 'q'

if user_input and user_input.lower() != 'q':
    with st.spinner('Processing your request...'):
        response = agent_chain.run(user_input)
        st.write(response)

