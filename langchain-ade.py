# %% [markdown]
# #### IMPORT PACKAGES

# %%
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

# %% [markdown]
# #### SET OPENAI API KEY

# %%
# set API keys to authenticate requests to the API
API_KEY = '<API Key>'

os.environ["OPENAI_API_KEY"] = API_KEY

# %% [markdown]
# #### USE LANGCHAIN

# %%
# load text document
loaders = TextLoader('<path to text document>')

# create vector representation of the loaded document
index = VectorstoreIndexCreator().from_loaders([loaders])

# %% [markdown]
# #### SET UP STREAMLIT APP

# %%
# display page title and text box for the user to ask questions
# to run streamlit, input into terminal: streamlit run the .py version of this file
st.title('🦜 LangChain: Chat with Adverse Events Report')
prompt = st.text_input("Enter your question")

# %% [markdown]
# #### GENERATE RESPONSE

# %%
# get the resonse from LLM
if prompt:
    response = index.query(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2), 
        question = prompt, 
        chain_type = 'stuff')

    # write the results from the LLM to the UI
    st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )

# %% [markdown]
# #### CONVERT TO .PY

# %%
# if all ok, convert to .py notebook

# bash
# ! python3 -m nbconvert --to script langchain-ade.ipynb


