<<<<<<< HEAD
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
import time
import datetime
import random
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain
from streamlit_modal import Modal

# Streamlit UI
st.set_page_config(page_title="ChatBot", layout="wide")

# Load BERT Model & Tokenizer
MODEL_PATH = "Trained Model/chatbot_model"  # Path where your model & tokenizer are saved

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

intent_mapping = {i: intent['tag'] for i, intent in enumerate(intents)}

# Chat history file
CHAT_HISTORY_FILE = "chat_history.json"

# Load Chat History (Ensuring Persistence)
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# Save Chat History
def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(st.session_state.chat_history, file)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    return intent_mapping.get(predicted_index, "unknown")

def get_response(intent_label):
    for intent in intents:
        if intent_label == intent['tag']:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Sidebar menu
with st.sidebar:
    st.image("Images/Bot Image.jpeg", use_container_width=True)
    st.title("💬 Chatbot Menu")
    add_vertical_space(1)

    if st.button("🆕 New Chat"):
        st.session_state.chat_input = ""
        save_chat_history()

    if st.button("📜 Chat History"):
        st.session_state.show_history = True

    if st.button("ℹ️ About"):
        st.session_state.show_about = True
    st.caption("🚀 Built with ❤️ by AI Enthusiast")

# Modal Windows (Chat History & About)
history_modal = Modal("📜Chat History", key="history_modal")
about_modal = Modal("ℹ️About", key="about_modal")

# Display chat history without affecting the chat window
if st.session_state.show_history:
    with history_modal.container():
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history[-10:]:  # Show last 10 messages
                st.write(chat)
        else:
            st.info("No chat history available.")

        # Clear History Button
        if st.button("🗑 Clear History", key="clear_history"):
            st.session_state.chat_history = []  # Clear session history
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)  # Delete saved file
            st.rerun()  # Refresh UI to reflect changes

        # Close Modal Button
        if st.button("Close", key="close_history"):
            st.session_state.show_history = False
            st.rerun()

if st.session_state.show_about:
    with about_modal.container():
        st.info("""This chatbot is powered by a deep learning approach using Hugging Face Transformers and a BERT model for intent recognition. 
It understands user queries by predicting intent and responding with predefined answers based on a trained dataset. 

✅ Natural Language Processing (NLP)​
                
🔹 Transformers (Hugging Face) 🤖 – BERT-based intent classification​

✅ Deep Learning​
                
🔹 PyTorch 🔥 – Model training & fine-tuning
""")
        
        if st.button("Close", key="close_about"):
            st.session_state.show_about = False  # Close modal without resetting UI
            st.rerun()

# Chat UI
st.title("🤖 AI Chatbot")
st.markdown("### Talk to me, I'm listening...")

rain(emoji="💬", font_size=10, falling_speed=5, animation_length="infinite")
rain(emoji="❄️", font_size=10, falling_speed=5, animation_length="infinite")

# Chat Input with Enter Button
with st.form("chat_form", clear_on_submit=True):
    chat_input = st.text_input("You:", key="chat_input")
    submit_button = st.form_submit_button("Enter")

if submit_button and chat_input:
    intent_label = predict_intent(chat_input)
    response = get_response(intent_label)
    chat_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append messages to chat history (User & Bot)
    st.session_state.chat_history.append(f"[{chat_time}] You: {chat_input}")
    st.session_state.chat_history.append(f"[{chat_time}] Bot: {response}")

    # Save chat history persistently
    save_chat_history()

    # Display conversation immediately
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat_input}")
        time.sleep(0.5)
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {response}")

# Style customization
st.markdown("""
<style>

    /* Fix Modal Height & Enable Scroll */
    div[data-modal-container="true"] {
        position: fixed !important;
        top: 5% !important;  /* Adjust this value to position it higher */
        color: #000000;
        max-height: 1000px !important;  /* Set max height */
    }

    /* Ensure the close button stays visible */
    div[data-modal-container="true"] button {
        position: sticky;
        bottom: 10px;
        background-color: #000000;
        color: white;
    }
            
    .stChatMessage {padding: 10px; border-radius: 10px; background-color: #900C3F;}
    .stChatMessageUser {background-color: #900C3F;}
    .stChatMessageAssistant {background-color: #900C3F;}
</style>
=======
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
import time
import datetime
import random
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain
from streamlit_modal import Modal

# Streamlit UI
st.set_page_config(page_title="ChatBot", layout="wide")

# Load BERT Model & Tokenizer
MODEL_PATH = "Trained Model/chatbot_model"  # Path where your model & tokenizer are saved

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

intent_mapping = {i: intent['tag'] for i, intent in enumerate(intents)}

# Chat history file
CHAT_HISTORY_FILE = "chat_history.json"

# Load Chat History (Ensuring Persistence)
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# Save Chat History
def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(st.session_state.chat_history, file)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    return intent_mapping.get(predicted_index, "unknown")

def get_response(intent_label):
    for intent in intents:
        if intent_label == intent['tag']:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Sidebar menu
with st.sidebar:
    st.image("Images/Bot Image.jpeg", use_container_width=True)
    st.title("💬 Chatbot Menu")
    add_vertical_space(1)

    if st.button("🆕 New Chat"):
        st.session_state.chat_input = ""
        save_chat_history()

    if st.button("📜 Chat History"):
        st.session_state.show_history = True

    if st.button("ℹ️ About"):
        st.session_state.show_about = True
    st.caption("🚀 Built with ❤️ by AI Enthusiast")

# Modal Windows (Chat History & About)
history_modal = Modal("📜Chat History", key="history_modal")
about_modal = Modal("ℹ️About", key="about_modal")

# Display chat history without affecting the chat window
if st.session_state.show_history:
    with history_modal.container():
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history[-10:]:  # Show last 10 messages
                st.write(chat)
        else:
            st.info("No chat history available.")

        # Clear History Button
        if st.button("🗑 Clear History", key="clear_history"):
            st.session_state.chat_history = []  # Clear session history
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)  # Delete saved file
            st.rerun()  # Refresh UI to reflect changes

        # Close Modal Button
        if st.button("Close", key="close_history"):
            st.session_state.show_history = False
            st.rerun()

if st.session_state.show_about:
    with about_modal.container():
        st.info("""This chatbot is powered by a deep learning approach using Hugging Face Transformers and a BERT model for intent recognition. 
It understands user queries by predicting intent and responding with predefined answers based on a trained dataset. 

✅ Natural Language Processing (NLP)​
                
🔹 Transformers (Hugging Face) 🤖 – BERT-based intent classification​

✅ Deep Learning​
                
🔹 PyTorch 🔥 – Model training & fine-tuning
""")
        
        if st.button("Close", key="close_about"):
            st.session_state.show_about = False  # Close modal without resetting UI
            st.rerun()

# Chat UI
st.title("🤖 AI Chatbot")
st.markdown("### Talk to me, I'm listening...")

rain(emoji="💬", font_size=10, falling_speed=5, animation_length="infinite")
rain(emoji="❄️", font_size=10, falling_speed=5, animation_length="infinite")

# Chat Input with Enter Button
with st.form("chat_form", clear_on_submit=True):
    chat_input = st.text_input("You:", key="chat_input")
    submit_button = st.form_submit_button("Enter")

if submit_button and chat_input:
    intent_label = predict_intent(chat_input)
    response = get_response(intent_label)
    chat_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append messages to chat history (User & Bot)
    st.session_state.chat_history.append(f"[{chat_time}] You: {chat_input}")
    st.session_state.chat_history.append(f"[{chat_time}] Bot: {response}")

    # Save chat history persistently
    save_chat_history()

    # Display conversation immediately
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat_input}")
        time.sleep(0.5)
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {response}")

# Style customization
st.markdown("""
<style>

    /* Fix Modal Height & Enable Scroll */
    div[data-modal-container="true"] {
        position: fixed !important;
        top: 5% !important;  /* Adjust this value to position it higher */
        color: #000000;
        max-height: 1000px !important;  /* Set max height */
    }

    /* Ensure the close button stays visible */
    div[data-modal-container="true"] button {
        position: sticky;
        bottom: 10px;
        background-color: #000000;
        color: white;
    }
            
    .stChatMessage {padding: 10px; border-radius: 10px; background-color: #900C3F;}
    .stChatMessageUser {background-color: #900C3F;}
    .stChatMessageAssistant {background-color: #900C3F;}
</style>
>>>>>>> hf/main
""", unsafe_allow_html=True)