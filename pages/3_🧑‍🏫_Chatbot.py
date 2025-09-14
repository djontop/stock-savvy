import streamlit as st
from google import genai
import os
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Initialize session state
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "cnt" not in st.session_state:
    st.session_state["cnt"] = 0
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

def get_text():
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Enter 'Hi' to start conversation with Bot. ", 
                            label_visibility='hidden')
    return input_text

def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state["conversation_history"] = []

def generate_response_with_gemini(user_input, conversation_history):
    """
    Generate response using Gemini API with conversation context
    """
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Create system prompt
        system_prompt = """As a finance advisor, your task is to familiarize clients with stock and investment-related terms in a concise and engaging manner. Through this role play, aim to simplify complex concepts and encourage active participation to ensure comprehension and confidence in navigating financial discussions. Your expertise lies in assessing risk levels and tailoring investment strategies accordingly."""
        
        # Build conversation context
        context = system_prompt + "\n\nConversation History:\n"
        for entry in conversation_history[-10:]:  # Keep last 10 exchanges
            context += f"{entry}\n"
        
        context += f"\nHuman: {user_input}\nYou:"
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=context
        )
        
        return response.text
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again."

st.subheader("Chat with AI Advisor to get a brief about stock related terms.")

# Load Gemini API key
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# New Chat button
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

st.write(st.session_state["cnt"])

# Initial greeting
if st.session_state["cnt"] == 0:
    initial_greeting = "Hi! I'm your AI Finance Advisor. I'm here to help you understand stock and investment-related terms in a simple and engaging way. I can explain complex financial concepts, assess risk levels, and help you navigate investment strategies. How can I assist you today?"
    st.session_state.past.append("")  
    st.session_state.generated.append(initial_greeting)
    st.session_state.conversation_history.append(f"Bot: {initial_greeting}")
    st.session_state["cnt"] += 1

# Handle user input
if st.session_state["cnt"] >= 1:
    user_input = get_text()
    submit_button = st.button("Generate")
    
    if submit_button and user_input:
        # Generate response using Gemini
        output = generate_response_with_gemini(user_input, st.session_state.conversation_history)
        
        # Update session state
        st.session_state.past.append(user_input)  
        st.session_state.generated.append(output)
        st.session_state.conversation_history.append(f"Human: {user_input}")
        st.session_state.conversation_history.append(f"Bot: {output}")

# Display conversation
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        if i < len(st.session_state["past"]) and st.session_state["past"][i]:
            st.info(st.session_state["past"][i], icon="😊")
        st.success(st.session_state["generated"][i], icon="🤖")

# Display stored sessions in sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Clear all sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
