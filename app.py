import streamlit as st
import logging
from kcc_chat import run_response_app
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log', filemode='a')
logger = logging.getLogger(__name__)

def main():

    # Page configuration
    st.set_page_config(page_title="KCC-GPT", page_icon="🌱", layout="wide")
    st.title("KCC Chat Assistant")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages =[]

    # Display history of chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt:=st.chat_input("what would you like to know?"):
        # Display message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({'role':'user', 'content':prompt})
        logger.info(f"User query: {prompt}")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Process the user input and generate a response
                response = run_response_app(prompt)
            st.markdown(response)
            logger.info("Response generated")
        
        # Add assistant response to chat history
        st.session_state.messages.append({'role':'assistant', 'content':response})
        logger.info(f"Assistant response: {response}")

    with st.sidebar:
        logo = Image.open("logo.jpg")
        st.image(logo, width=200)

        st.title("Options")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            logger.info("Chat cleared")
        
        st.subheader("About")
        st.write("Simple KCC chatbot")
        
if __name__ == "__main__":
    logger.info("Starting the app...")
    main()