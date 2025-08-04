import streamlit as st
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os
from gtts import gTTS
import qdrant_client
import tempfile
import shutil
import speech_recognition as sr
import time
from datetime import datetime
import base64
import pygame
import threading
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

# Initialize pygame for audio playback
pygame.mixer.init()

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
st.info("🎤 **Local Version**: Automatic audio playback and speech recognition enabled!")

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
    "is_listening": False,
    "audio_playing": False,
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

# Enhanced TTS with automatic playback using pygame
def play_audio_automatically(text):
    """Play audio automatically using pygame"""
    if not text.strip():
        return False

    st.session_state.current_question = text
    status_placeholder = st.empty()
    
    try:
        status_placeholder.markdown(f"**🤖 Vyassa is speaking...**  \n{text}")
        st.session_state.audio_playing = True
        
        # Generate TTS
        tts = gTTS(text=text, lang='en', slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            
            # Play audio using pygame
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            pygame.mixer.music.unload()
            os.unlink(tmp_file.name)
        
        st.session_state.audio_playing = False
        status_placeholder.empty()
        return True
        
    except Exception as e:
        st.error(f"Audio playback error: {e}")
        st.session_state.audio_playing = False
        status_placeholder.empty()
        return False

# Enhanced speech recognition with better error handling
def listen_for_speech_automatically():
    """Automatically listen for speech input"""
    r = sr.Recognizer()
    r.pause_threshold = 2.0
    r.dynamic_energy_threshold = True
    r.energy_threshold = 300
    
    status_placeholder = st.empty()
    
    try:
        with sr.Microphone() as source:
            st.session_state.is_listening = True
            status_placeholder.info("🎙️ Adjusting for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source, duration=1)
            
            status_placeholder.success("🎤 **Listening for your response... Speak now!**")
            
            # Listen with timeout
            audio = r.listen(source, timeout=30, phrase_time_limit=60)
            
            status_placeholder.info("🔄 Processing your response...")
            
            # Try different recognition services
            try:
                # Try Google first
                text = r.recognize_google(audio)
            except:
                try:
                    # Fallback to Whisper if available
                    text = r.recognize_whisper(audio)
                except:
                    text = "I couldn't understand your response clearly."
            
            st.session_state.is_listening = False
            status_placeholder.empty()
            return text.strip()
            
    except sr.WaitTimeoutError:
        status_placeholder.warning("⏰ No speech detected. Moving to next question...")
        time.sleep(2)
        st.session_state.is_listening = False
        status_placeholder.empty()
        return "No response provided - timeout"
        
    except sr.RequestError as e:
        status_placeholder.error(f"❌ Speech recognition error: {e}")
        st.session_state.is_listening = False
        status_placeholder.empty()
        return "Speech recognition service error"
        
    except Exception as e:
        status_placeholder.error(f"❌ Unexpected error: {e}")
        st.session_state.is_listening = False
        status_placeholder.empty()
        return "Unexpected error occurred"

# Complete interview step with automatic audio and speech
def conduct_automatic_interview_step(question_text):
    """Conduct a complete interview step with automatic audio and speech"""
    if not question_text.strip():
        return "No question provided"
    
    # Play question automatically
    if not play_audio_automatically(question_text):
        return "Audio playback failed"
    
    # Small pause before listening
    time.sleep(1)
    
    # Start answer timer
    st.session_state.answer_timer_start = datetime.now()
    
    # Listen for response automatically
    user_response = listen_for_speech_automatically()
    
    # Calculate answer time
    if st.session_state.answer_timer_start:
        answer_duration = (datetime.now() - st.session_state.answer_timer_start).total_seconds()
        st.session_state.total_answer_time += answer_duration
        st.session_state.answer_timer_start = None
    
    return user_response

# Remaining time calculation
def get_remaining_time():
    if not st.session_state.interview_start_time:
        return 300
    
    elapsed = (datetime.now() - st.session_state.interview_start_time).total_seconds()
    # Add current answer time if actively answering
    if st.session_state.answer_timer_start:
        current_answer_time = (datetime.now() - st.session_state.answer_timer_start).total_seconds()
        elapsed += current_answer_time
    
    return max(0, 300 - elapsed)

# Sidebar with real-time updates
if st.session_state.interview_active and st.session_state.interview_start_time:
    remaining = get_remaining_time()
    mins, secs = divmod(int(remaining), 60)
    
    st.sidebar.info(f"🕒 Time Left: {mins:02d}:{secs:02d}")
    st.sidebar.progress((300 - remaining) / 300, text="⏳ Interview Progress")
    st.sidebar.markdown(f"**Question {st.session_state.question_count}/5**")
    
    # Status indicators
    if st.session_state.audio_playing:
        st.sidebar.success("🔊 Audio Playing")
    elif st.session_state.is_listening:
        st.sidebar.success("🎤 Listening...")
    else:
        st.sidebar.info("⏸️ Ready")

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Interview Conversation")
    for speaker, message in st.session_state.chat_history:
        if speaker == "Assistant":
            with st.chat_message("assistant"):
                st.write(f"**🤖 Vyassa:** {message}")
        else:
            with st.chat_message("user"):
                st.write(f"**You:** {message}")

# Start interview
if st.session_state.resume_uploaded and st.session_state.chat_engine and not st.session_state.interview_active:
    st.markdown("### 🎯 Ready to Start")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Automatic Interview", type="primary"):
            try:
                st.session_state.interview_active = True
                st.session_state.interview_start_time = datetime.now()
                st.session_state.chat_history = []
                st.session_state.question_count = 1

                intro_prompt = """
                You are Vyassa, an AI interviewer.
                Greet the candidate briefly, introduce yourself in 2-3 words, then ask them to tell you about themselves.
                Keep it concise and conversational.
                """
                intro_response = st.session_state.chat_engine.chat(intro_prompt).response
                st.session_state.chat_history.append(("Assistant", intro_response))
                
                # Automatically conduct first step
                user_input = conduct_automatic_interview_step(intro_response)
                st.session_state.chat_history.append(("You", user_input))
                st.rerun()
                
            except Exception as e:
                st.error(f"Error starting interview: {e}")
                st.session_state.interview_active = False

# Ongoing interview with automatic flow
elif st.session_state.interview_active and st.session_state.chat_engine:
    remaining_time = get_remaining_time()
    
    # Check if interview should end
    if remaining_time < 30 or st.session_state.question_count > 5:
        try:
            closing_prompt = "Acknowledge the previous response briefly and thank the candidate professionally. Keep it concise."
            closing = st.session_state.chat_engine.chat(closing_prompt).response
            st.session_state.chat_history.append(("Assistant", closing))
            
            # Play closing message
            play_audio_automatically(closing)
            
            st.session_state.interview_active = False
            st.session_state.interview_ended = True
            st.success("🎉 Interview completed automatically! Thank you for participating.")
            
        except Exception as e:
            st.error(f"Error ending interview: {e}")
            st.session_state.interview_active = False
    
    else:
        # Continue interview automatically
        if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "You":
            last_user_input = st.session_state.chat_history[-1][1]
            
            try:
                # Get AI response
                ai_response = st.session_state.chat_engine.chat(last_user_input).response
                st.session_state.chat_history.append(("Assistant", ai_response))
                
                # Automatically conduct next step
                user_input = conduct_automatic_interview_step(ai_response)
                st.session_state.chat_history.append(("You", user_input))
                
                st.session_state.question_count += 1
                st.rerun()
                
            except Exception as e:
                st.error(f"Interview error: {e}")
                st.session_state.interview_active = False

# Emergency controls during interview
if st.session_state.interview_active:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⏸️ Pause Interview"):
            st.session_state.interview_active = False
            st.info("Interview paused. You can resume or restart.")
    
    with col2:
        if st.button("🔄 Skip Question"):
            if st.session_state.chat_history:
                st.session_state.chat_history.append(("You", "Skipped question"))
                st.rerun()
    
    with col3:
        if st.button("🛑 End Interview"):
            st.session_state.interview_active = False
            st.session_state.interview_ended = True

# Restart interview
if st.session_state.interview_ended:
    st.markdown("---")
    st.markdown("### 📊 Interview Summary")
    
    if st.session_state.chat_history:
        total_questions = len([msg for msg in st.session_state.chat_history if msg[0] == "Assistant"])
        st.info(f"✅ Completed {total_questions} questions in {st.session_state.question_count} rounds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Start New Interview"):
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()
    
    


st.markdown("---")
st.markdown("*🤖 Powered by Vyassa AI - Fully Automated Interview Experience*")
