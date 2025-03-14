import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import time
import datetime
import random
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain
from streamlit_modal import Modal

# Load BERT Model & Tokenizer
MODEL_PATH = "Trained Model/chatbot_model"  # Path where your model & tokenizer are saved

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Move model to device
model.to(device)
model.eval()

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Create a mapping between index and intent tag
intent_mapping = {i: intent['tag'] for i, intent in enumerate(intents)}

DEBUG = False  # Set to True only when debugging

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    predicted_tag = intent_mapping.get(predicted_index, "unknown")  # Convert index to tag

    if DEBUG:  # Only print if debugging is enabled
        print(f"Predicted Index: {predicted_index}, Mapped Intent: {predicted_tag}")

    return predicted_tag

def get_response(intent_label):
    if DEBUG:
        print(f"Intent Label: {intent_label}")

    for intent in intents:
        if DEBUG:
            print(f"Checking: {intent['tag']}")
        if intent_label == intent['tag']:
            return random.choice(intent['responses'])

    return "I'm not sure how to respond to that."

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# Sidebar menu
with st.sidebar:
    st.title("💬 Chatbot Menu")
    add_vertical_space(1)
    if st.button("🆕 New Chat"):
        st.session_state.chat_history = []
        st.session_state.chat_input = ""
        st.rerun()
    if st.button("📜 Chat History"):
        st.session_state.show_history = True
    if st.button("ℹ️ About"):
        st.session_state.show_about = True

# Modals for Chat History and About
history_modal = Modal("Chat History", key="history_modal")
about_modal = Modal("About", key="about_modal")

if "show_history" in st.session_state and st.session_state.show_history:
    with history_modal.container():
        st.subheader("🕰 Chat History")
        for chat in st.session_state.chat_history:
            st.write(chat)
        if st.button("Close", key="close_history"):
            st.session_state.show_history = False
            st.rerun()

if "show_about" in st.session_state and st.session_state.show_about:
    with about_modal.container():
        st.subheader("ℹ️ About")
        st.info("This is an intent-based chatbot powered by BERT, built using Streamlit with an elegant UI.")
        if st.button("Close", key="close_about"):
            st.session_state.show_about = False
            st.rerun()

# Chat UI
st.title("🤖 AI Chatbot")
st.markdown("### Talk to me, I'm listening...")

rain(
    emoji="💬",
    font_size=10,
    falling_speed=5,
    animation_length="infinite",
)

# User input
st.session_state.chat_input = st.text_input("You:", st.session_state.chat_input, key="input")
if st.session_state.chat_input:
    intent_label = predict_intent(st.session_state.chat_input)
    response = get_response(intent_label)
    chat_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append(f"[{chat_time}] You: {st.session_state.chat_input}")
    st.session_state.chat_history.append(f"[{chat_time}] Bot: {response}")

    # Display conversation
    with st.chat_message("user"):
        st.markdown(f"**You:** {st.session_state.chat_input}")
        time.sleep(0.5)
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {response}")

    # Clear input box after sending
    st.session_state.chat_input = ""

# Style customization
st.markdown("""
<style>
    .stChatMessage {padding: 10px; border-radius: 10px; background-color: #000000;}
    .stChatMessageUser {background-color: #000000;}
    .stChatMessageAssistant {background-color: #000000;}
</style>
""", unsafe_allow_html=True)