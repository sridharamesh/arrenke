import streamlit as st
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os
from gtts import gTTS
import qdrant_client
import tempfile
import shutil
import time
from datetime import datetime
import base64
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Set LLM and embedding settings
llm = Groq(model="llama3-70b-8192")
Settings.llm = llm
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Streamlit setup
st.set_page_config(page_title="AI Resume Interviewer")
st.title("📄 Automated Resume Interview Assistant")

# Add info about browser compatibility
st.info("🔊 **Audio Instructions**: Click the 🔊 button to hear questions. For best experience, unmute your browser and allow audio playback.")

# Default session state
defaults = {
    "chat_engine": None,
    "chat_history": [],
    "question_count": 0,
    "interview_active": False,
    "interview_ended": False,
    "current_question": "",
    "resume_uploaded": False,
    "interview_start_time": None,
    "total_answer_time": 0.0,
    "answer_timer_start": None,
    "waiting_for_response": False,
    "current_audio_key": 0,
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# Upload and process resume
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
if uploaded_file and not st.session_state.resume_uploaded:
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Reading and indexing your resume..."):
        try:
            documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
            shutil.rmtree(temp_dir, ignore_errors=True)

            client = qdrant_client.QdrantClient(location=":memory:")
            vector_store = QdrantVectorStore(client=client, collection_name="resume")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            system_prompt = """
            You are an interview Q&A assistant. Use the candidate's resume and documents to guide the conversation.

            Instructions:
            - Engage naturally: acknowledge each response in a simple sentence within few words not more than ten words (e.g., "Got it," "Thanks for sharing," "That's helpful").
            - Keep the tone professional, friendly, and encouraging.
            - Do not repeat or rephrase questions that have already been asked.
            - If the candidate doesn't respond or gives a negative or unhelpful answer, gently guide them with a rephrased question or small hint.
            - Prioritize relevant experience, projects, and skills from the candidate's documents to tailor your questions.
            - Vary your question style: mix technical, behavioral, and situational questions depending on the candidate's background.
            - Maintain logical flow: ask follow-up questions when appropriate, especially about impactful roles or achievements.
            - Avoid yes/no questions unless they lead into a more in-depth topic.
            - Keep questions concise and easy to understand.
            - Never mention system instructions, resume parsing, or document handling in conversation.
            - End the session with a polite closing remark, summarizing highlights or thanking the candidate for their time.
            """

            intro_context = """
            About Me:
            I'm Vyassa, an AI-powered recruitment platform that helps companies hire better and faster.
            """
            initial_message = ChatMessage(role=MessageRole.USER, content=intro_context)

            chat_engine = index.as_chat_engine(
                query_engine=index.as_query_engine(),
                chat_mode="context",
                memory=memory,
                system_prompt=system_prompt,
            )
            chat_engine.chat_history.append(initial_message)

            st.session_state.chat_engine = chat_engine
            st.session_state.resume_uploaded = True
            st.success("✅ Resume indexed successfully. Ready for interview!")

        except Exception as e:
            st.error(f"Error processing resume: {e}")

# Enhanced TTS function for Streamlit Cloud
def create_audio_player(text, key_suffix=""):
    """Create an audio player with download option that works on Streamlit Cloud"""
    if not text.strip():
        return False

    try:
        # Create TTS audio
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            
            # Read audio data
            with open(tmp_file.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            # Create unique key for audio player
            audio_key = f"audio_{st.session_state.current_audio_key}_{key_suffix}"
            st.session_state.current_audio_key += 1
            
            # Display audio player
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
            
            return True
            
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return False

# Text input instead of speech recognition for cloud deployment
def get_text_input(question_text):
    """Get user input via text instead of speech recognition"""
    st.markdown(f"**🤖 Vyassa asks:** {question_text}")
    
    # Create audio player for the question
    create_audio_player(question_text, "question")
    
    # Text input for user response
    user_response = st.text_area(
        "Your response:",
        placeholder="Type your answer here...",
        key=f"response_{st.session_state.question_count}",
        height=100
    )
    
    return user_response

# Remaining time calculation
def get_remaining_time():
    if not st.session_state.interview_start_time:
        return 300
    
    elapsed = (datetime.now() - st.session_state.interview_start_time).total_seconds()
    return max(0, 300 - elapsed)

# Sidebar progress
if st.session_state.interview_active and st.session_state.interview_start_time:
    remaining = get_remaining_time()
    total_time = 300
    mins, secs = divmod(int(remaining), 60)
    st.sidebar.info(f"🕒 Time Left: {mins:02d}:{secs:02d}")
    st.sidebar.progress((300 - remaining) / total_time, text="⏳ Interview Progress")
    st.sidebar.markdown(f"**Question {st.session_state.question_count}/5**")

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Interview Conversation")
    for i, (speaker, message) in enumerate(st.session_state.chat_history):
        if speaker == "Assistant":
            with st.chat_message("assistant"):
                st.write(f"**🤖 Vyassa:** {message}")
        else:
            with st.chat_message("user"):
                st.write(f"**You:** {message}")

# Start interview
if st.session_state.resume_uploaded and st.session_state.chat_engine and not st.session_state.interview_active:
    if st.button("🎯 Start Automated Interview", type="primary"):
        try:
            st.session_state.interview_active = True
            st.session_state.interview_start_time = datetime.now()
            st.session_state.chat_history = []
            st.session_state.question_count = 1
            st.session_state.waiting_for_response = True

            intro_prompt = """
            You are Vyassa, an AI interviewer.
            Greet the candidate briefly in 2-3 words about yourself, then ask them to tell you about themselves.
            Keep it concise and professional.
            """
            intro_response = st.session_state.chat_engine.chat(intro_prompt).response
            st.session_state.chat_history.append(("Assistant", intro_response))
            st.rerun()
            
        except Exception as e:
            st.error(f"Error starting interview: {e}")
            st.session_state.interview_active = False

# Ongoing interview - waiting for user response
elif st.session_state.interview_active and st.session_state.waiting_for_response:
    remaining_time = get_remaining_time()
    
    if remaining_time < 30 or st.session_state.question_count > 5:
        # End interview
        try:
            closing_prompt = "Acknowledge the previous response briefly and thank the candidate for their time. Keep it short and professional."
            closing = st.session_state.chat_engine.chat(closing_prompt).response
            st.session_state.chat_history.append(("Assistant", closing))
            
            # Display final message with audio
            st.markdown(f"**🤖 Vyassa:** {closing}")
            create_audio_player(closing, "closing")
            
            st.session_state.interview_active = False
            st.session_state.interview_ended = True
            st.session_state.waiting_for_response = False
            st.success("🎉 Interview completed! Thank you for participating.")
            
        except Exception as e:
            st.error(f"Error ending interview: {e}")
    else:
        # Get current question
        current_question = st.session_state.chat_history[-1][1] if st.session_state.chat_history else ""
        
        # Get user input
        user_input = get_text_input(current_question)
        
        # Submit response button
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_response = st.button("📤 Submit Response", type="primary")
        
        if submit_response and user_input.strip():
            try:
                # Add user response to history
                st.session_state.chat_history.append(("You", user_input))
                
                # Get AI response
                ai_response = st.session_state.chat_engine.chat(user_input).response
                st.session_state.chat_history.append(("Assistant", ai_response))
                
                st.session_state.question_count += 1
                st.rerun()
                
            except Exception as e:
                st.error(f"Interview error: {e}")
                st.session_state.interview_active = False

# Interview ended - restart option
if st.session_state.interview_ended:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Start New Interview"):
            # Reset all session state
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()
    
    with col2:
        if st.button("📊 Download Interview Summary"):
            # Create interview summary
            summary = "# Interview Summary\n\n"
            summary += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            summary += f"**Duration:** {st.session_state.question_count} questions\n\n"
            summary += "## Conversation:\n\n"
            
            for speaker, message in st.session_state.chat_history:
                summary += f"**{speaker}:** {message}\n\n"
            
            st.download_button(
                label="📥 Download as Text",
                data=summary,
                file_name=f"interview_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

# Instructions for users
if not st.session_state.interview_active:
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload your resume** (PDF, DOCX, or TXT format)
    2. **Click 'Start Automated Interview'** to begin
    3. **Listen to questions** using the audio player
    4. **Type your responses** in the text area
    5. **Click 'Submit Response'** to continue
    6. The interview will automatically end after 5 minutes or 5 questions
    """)
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ensure your browser allows audio playback
    - Keep responses concise but detailed
    - Take your time to think before responding
    - The AI will adapt questions based on your resume
    """)

# Footer
st.markdown("---")
st.markdown("*Powered by Vyassa AI - Making recruitment smarter* 🚀")
