from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
#Langsmith Tracing
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

##LLAMA3 model
llm=Ollama(model="LLAMA3.2")
#Prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked."),
        ("user","Question:{question}")
    ]
)
##output parser
output_parser=StrOutputParser()
## Document chain
document_chain = prompt|llm|output_parser

##Streamlit Framework
st.title("Langchain demo with LLAMA3")
input_text=st.text_input("What question you have in mind?")

if input_text:
    st.write(document_chain.invoke({"question":input_text}))