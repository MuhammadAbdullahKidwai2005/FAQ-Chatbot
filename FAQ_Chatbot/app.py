import os
import requests
import streamlit as st
from streamlit_lottie import st_lottie

# 1. Suppress TensorFlow Warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

# 2. Logic Imports (Unchanged)
from preprocessing import chunks
from model import local_llm, build_vectorstore
from chatbot import build_qa_chain
from dotenv import load_dotenv

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

# --- CUSTOM CSS (Sleek monochrome chat) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Hide the 'made with streamlit' and other headers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Adjusting chat message styling */
    .stChatMessage {
        background-color: transparent !important;
        padding-top: 1rem;
    }
    
    /* Make the assistant bubble slightly different if desired */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #161b22 !important;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ANIMATION LOADER ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_bot = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_m6cu9sc0.json")

# --- SETUP LOGIC (Cached) ---
@st.cache_resource
def setup():
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore, local_llm, all_chunks=chunks)
    return qa_chain

qa_chain = setup()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/78/Dropbox_Icon.svg", width=40)
    st.title("Secure FAQ")
    st.markdown("---")
    st.write("● **Engine:** GPT-4o-Mini")
    st.write("● **Source:** Dropbox Docs")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- CHAT INTERFACE ---

# 1. Header with animation
col1, col2 = st.columns([1, 6])
with col1:
    if lottie_bot:
        st_lottie(lottie_bot, height=60, key="bot")
with col2:
    st.subheader("How can I help you?")

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input (The GPT-like bar at the bottom)
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        # We don't use st.success here, just plain text to look like a chat
        placeholder = st.empty()
        with st.spinner(""):
            try:
                response = qa_chain.invoke(prompt)
                placeholder.markdown(response.strip())
                st.session_state.messages.append({"role": "assistant", "content": response.strip()})
            except Exception as e:
                placeholder.error(f"Something went wrong. {str(e)}")