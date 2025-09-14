import os
import requests
from bs4 import BeautifulSoup
import logging
import streamlit as st
from langchain.schema import Document
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec

import os

# ------------------ Logging Setup ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------ Helper functions ------------------
def write_file(content: str, filename: str = 'spring_boot_rest_service_guide.txt') -> None:
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(content + "\n")
    logger.info(f"Written content to {filename}")


def fetch_spring_guides_json(url: str) -> list:
    logger.info(f"Fetching Spring guides JSON from {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    nodes = data.get('result', {}).get('data', {}).get('guides', {}).get('nodes', [])
    guide_urls = []
    for node in nodes:
        url_path = 'https://spring.io' + node.get('path', '')
        #write_file(url_path, 'href.txt')
        guide_urls.append(url_path)
        logger.info(f"Found guide URL: {url_path}")
    return guide_urls


def fetch_spring_guide_text(url: str) -> str:
    logger.info(f"Fetching guide text from {url}")
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    main_content = soup.find('div', class_='ascii-doc')
    if not main_content:
        logger.warning(f"No content found at {url}")
        return "No content found at the specified URL."
    text_only = main_content.get_text(separator="\n")
    clean_text = "\n".join([line.strip() for line in text_only.splitlines() if line.strip()])
    logger.info(f"Fetched content length: {len(clean_text)} characters")
    return clean_text


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if 'store' not in st.session_state:
        st.session_state.store = {}
        logger.info("Initialized session_state store")
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
        logger.info(f"Created new chat history for session_id: {session_id}")
    return st.session_state.store[session_id]


# ------------------ Constants ------------------
DOC_URL = "https://spring.io/page-data/guides/page-data.json"
API_KEY = "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"
HF_TOKEN = "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"
PINECONE_API_KEY = "pcsk_7HbFDU_KnB38ny9zWM8MFpJxu8bk6szDnjAaYhWwFBEpRV4TkD5hwe3wKjrJSjtxS6QcV6"
INDEX_NAME = "spring-guides"



pc = PineconeClient(api_key=PINECONE_API_KEY)

# Create index if it does not exist
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )
    print(f"Created Pinecone index: {INDEX_NAME}")
else:
    print(f"Index {INDEX_NAME} already exists")

index = pc.Index(INDEX_NAME)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    logger.info("HuggingFace token set in environment")
else:
    raise ValueError("HF_TOKEN is not set. Please set it as an environment variable.")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------ Streamlit UI ------------------
st.title("Conversational RAG for Spring")
llm = ChatGroq(
    groq_api_key=API_KEY, 
    model_name="Gemma2-9b-It",
    max_tokens=2048  # allow longer answers
    )
session_id = "default_session"
session_history = get_session_history(session_id)

# ------------------ Load documents ------------------
logger.info("Fetching all guide URLs...")
all_guide_urls = fetch_spring_guides_json(DOC_URL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)

all_docs = []
for url in all_guide_urls:
    logger.info(f"Processing guide: {url}")
    guide_text = fetch_spring_guide_text(url)
    if guide_text.strip():
        docs = [Document(page_content=guide_text, metadata={"source": url})]
        splits = text_splitter.split_documents(docs)
        all_docs.extend(splits)

# Create ONE vectorstore across all docs
#vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings)

# Create Pinecone vectorstore
vectorstore = Pinecone.from_documents(documents=all_docs,embedding=embeddings,index_name=INDEX_NAME)

retriever = vectorstore.as_retriever()

# ------------------ RAG Prompts ------------------
answer_system_promt = (
    "You are given the chat history, the user’s latest question, and an uploaded text is source of truth. "
    "Use the chat history to resolve context or references, but always base your answer strictly on the content. "
    "If the information is not present in the text context, respond with: 'The document does not contain this information.' "
    "Do not guess or add external knowledge."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_system_promt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know."
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# ------------------ Streamlit Interaction ------------------
user_input = st.text_input("Your question:")

if user_input:
    logger.info(f"User question: {user_input}")
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    answer = response['answer']
    logger.info(f"Assistant answer length: {len(response['answer'])} characters")
    #st.write("Assistant:", response['answer'])
    st.subheader("Assistant Response:")
    st.markdown(answer) 

