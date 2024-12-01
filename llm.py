import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Create the LLM
llm = ChatOpenAI(
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"],
    model="gpt-3.5-turbo"
)

# Create the Embedding model
embeddings = OpenAIEmbeddings(
    api_key=st.secrets["OPENAI_API_KEY"]
)
