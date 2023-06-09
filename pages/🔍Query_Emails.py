import os
import logging
import sys
import re

from typing import Any, List, Optional, Sequence, Tuple
import streamlit as st
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Union


from langchain.agents.conversational_chat.base import ConversationalChatAgent

from langchain import SQLDatabase, SQLDatabaseChain, SerpAPIWrapper, LLMChain, OpenAI
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

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.agents.conversational_chat.prompt import (
    PREFIX,
    SUFFIX,
    TEMPLATE_TOOL_RESPONSE,
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AgentAction,
    AIMessage,
    BaseLanguageModel,
    BaseMessage,
    BaseOutputParser,
    HumanMessage,
)
from langchain.tools.base import BaseTool


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
    
    df['content'] = df['content'].astype(str).apply(remove_html_tags)
    df = df.where(pd.notnull(df), None)
    df = df.fillna(0)





    # Ensure date columns are datetime objects in the original DataFrame
    date_columns = ["created_at", "published_at", "send_at"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Create a separate DataFrame for working with Chroma
    df_chroma = df.copy()

    # Convert date columns to string format in the df_chroma DataFrame
    for col in date_columns:
        df_chroma[col] = df_chroma[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))


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
When using exact-match email_names, be case-insensitive and use the LIKE function instead.

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
db_chain = SQLDatabaseChain(llm=llm1, database=sql_database, prompt=PROMPT)



@tool("Email Analytics")
def sql_index_tool(query: str) -> str:
    """Use this for email analytics. It will query a table of emails where columns are id, email_name, description, content, open_rate, click_rate, unsubscribes, total_clicks, recipients, sent_from, published_at, send_at, public, thumbnail_url, and status (draft/completed). Query structured data using SQL syntax.
    Some examples:
    Question: "How many emails were sent in 2022?"
    SQLQuery: SELECT COUNT(*) FROM emails WHERE strftime('%Y', send_at) = '2022';

    Question: "What email had the most engagement in January 2023?"
    SQLQuery: SELECT email_name FROM emails WHERE published_at BETWEEN '2023-01-01' AND '2023-01-31' ORDER BY total_clicks DESC LIMIT 1

    Question: "top 5 emails by click rate"
    SQLQuery: SELECT email_name, click_rate FROM emails WHERE status = 'completed' ORDER BY click_rate DESC LIMIT 5;

    Question: "What is the average open rate for emails sent on weekends?"
    SQLQuery: SELECT AVG(open_rate) FROM emails WHERE status = 'completed' AND (strftime('%w', send_at) = '0' OR strftime('%w', send_at) = '6');

    Question: "Which published day of the week has the highest email engagement?"
    SQLQuery: SELECT strftime('%w', published_at) AS day_of_week, AVG(open_rate) AS avg_open_rate FROM emails WHERE status = 'completed' GROUP BY day_of_week ORDER BY avg_open_rate DESC LIMIT 1;

    """
    query = query.replace('"', '')
    sql_response = db_chain.run(query)
    return f"\nThe SQL Result is: {sql_response}\n"










@tool("Simple Email Retrieval", return_direct=True)
def print_email(query: str) -> str:
    """Use this tool when the user wants to see an email."""
    
    # Find similar previous emails
    similar_emails = content_index.similarity_search(query, k=1)

    # Extract email content
    context = "\n\n".join([email.page_content for email in similar_emails])

    return context


@tool("Email Summarizer", return_direct=True)
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


@tool("Email writer", return_direct=True)
def generate_email(query: str) -> str:
    """This tool writes new emails and other content based on existing emails. Provide a FULL query describing the task."""
    # Find similar previous emails
    similar_emails = content_index.similarity_search(query, k=1)

    # Extract email content
    context = "\n\n".join([email.page_content for email in similar_emails])

    # Custom prompt for generating emails
    email_prompt_template = """This is a laidback email newsletter about Squarespace resources. Using the following email context, {query}\n\n
    
    Email Context: {context}
    
    """

    EMAIL_PROMPT = PromptTemplate(
        template=email_prompt_template, input_variables=["context", "query"]
    )

    # Create a new LLMChain with the custom email prompt
    email_chain = LLMChain(llm=llm, prompt=EMAIL_PROMPT)

    # Generate the new email
    new_email = email_chain({"query": query, "context": context})
    email_text = new_email['text']

    return email_text

tools = [generate_email, sql_index_tool, summarize_email, print_email]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



def initialize_custom_agent_executor(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    custom_prefix: str,
    custom_suffix: str,
    custom_template_tool_response: str,
    callback_manager=None,
    agent_kwargs=None,
    **kwargs: Any,
) -> AgentExecutor:
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=custom_prefix,
        human_message=custom_suffix,
        # Pass any other required arguments or optional keyword arguments here
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        **kwargs,
    )

# Define your custom strings

MY_PREFIX = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of email analysis and email related tasks.

"""

MY_SUFFIX = """TOOLS
------
You have access to the following tools.

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{{{{input}}}}"""

FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me, please output a response in one of two formats:

**Option 1:**
Use this if you want to use a tool.
Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": string \\ The action to take. Must be one of {tool_names}
    "action_input": string \\ The input to the action
}}}}
```

**Option #2:**
Use this to provide a final answer. Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": "Final Answer",
    "action_input": string \\ 
}}}}
```"""

MY_TEMPLATE_TOOL_RESPONSE = """TOOL RESPONSE: 
---------------------
{observation}

"""

# Initialize your custom agent executor
llm = ChatOpenAI(temperature=0)
agent_chain = initialize_custom_agent_executor(tools, llm, MY_PREFIX, MY_SUFFIX, MY_TEMPLATE_TOOL_RESPONSE, verbose=True, memory=memory)






# User input

st.markdown("Examples:")
st.markdown("What email had the most engagement in March 2023?")
st.markdown("Rewrite the email about...")
st.markdown("Show me the email where...")

st.markdown("")

user_input = st.text_input("Please ask a question or make a request(or 'q' to quit): ")

if user_input and user_input.lower() != 'q':
    with st.spinner('Processing your request...'):
        response = agent_chain.run(user_input)
        st.write(response)
# Check if df_chroma is defined before writing it to the Streamlit app
if 'df_chroma' in locals():
    st.write(df_chroma)
# Check if input is not empty and not 'q'
