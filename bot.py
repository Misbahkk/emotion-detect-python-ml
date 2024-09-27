
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline
import os


load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


# Initialize the LLM model (adjust with your specific model details)
model = genai.GenerativeModel("gemini-1.5-flash")

# # Initialize the emotion detection model (from Hugging Face)
emotion_model = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")


st.set_page_config(
    page_title="Chatbot",
    page_icon=":brain",
    layout="centered",
)

st.title("Mental Health Support Chatbot 🤖")
st.write("Hi! How can I help you today?")


def translate_rol_for_stremlit(user_role):
    return 'assistant' if user_role == 'model' else user_role
    


# # Intilize mood traking
mood_tracking = []


def dedect_mood(text):
    result = emotion_model(text)
    emotion = result[0]['label']
    return emotion






# Store the conversation history in a session state
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Send initial prompt to assistant when the session starts, but don't show it in the chat
if 'session_started' not in st.session_state:
    initial_prompt = "Assume you are a Mental health assistant. Please start a therapy session according to the user's prompt."
    # Send the prompt to the assistant but don't display it
    st.session_state.chat_session.send_message(initial_prompt)
    st.session_state.session_started = True

# Display the previous conversation
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_rol_for_stremlit(message.role)):
        st.markdown(message.parts[0].text)
                         

# Initialize conversation history in session state if not already initialized
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


if 'mood_tracking' not in st.session_state:
    st.session_state.mood_tracking = []

# Sidebar to display conversation history
st.sidebar.title("Mood Tracking")
if st.session_state.conversation_history:
    for i,(user_input, mood,assistant_response) in enumerate(st.session_state.conversation_history):
        st.sidebar.markdown(f"**You:** {user_input}")
        st.sidebar.markdown(f"**Mood:** {mood}")
        st.sidebar.markdown("---")
else:
    st.sidebar.markdown("**NO Modd Dedect yet..**")


# User input box
user_input = st.chat_input("Ask gemini pro...")
# Process the user input and get a response from the LLM
if user_input:

    # dectct the mood from user input
    mood = dedect_mood(user_input)
    mood_tracking.append((user_input,mood))
    # st.write(f"Detected Mood: {mood}")
    # Append user input to the conversation
    # user_input_clear = "Asume you are a Mental health assistent."+user_input+" write in short and also provide tharepy session. "
    st.chat_message("user").markdown(user_input)
    st.sidebar.markdown(f"**You:** {user_input}")
    st.sidebar.markdown(f"**Detected Mod:** {mood}")
    st.sidebar.markdown("---")

    # Call the LLM API to generate a response
    response = st.session_state.chat_session.send_message(user_input)

    with st.chat_message('assistant'):
        st.markdown(response.text)

    # Save conversation in session state for sidebar
    st.session_state.conversation_history.append((user_input, mood,response.text))





# Reset conversion when user click the clear button
if st.button("Clear Conversation"):
    st.session_state.chat_session = model.start_chat(history=[])
    st.session_state.input_text = ''
    mood_tracking.clear()


   



