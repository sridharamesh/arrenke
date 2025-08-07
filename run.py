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
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
import speech_recognition as sr
import io

import base64
from audio_recorder_streamlit import audio_recorder
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
    "audio_playing": False,
    "current_audio_key": 0,
    "processing_response": False,
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# Utility Functions
def autoplay_audio(file_path: str):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay hidden>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
        return True
    except Exception as e:
        st.error(f"Audio playback error: {e}")
        return False
    finally:
        try:
            os.unlink(file_path)
        except:
            pass


def play_tts_with_display(text):
    if not text.strip(): 
        return False
    
    st.session_state.current_question = text
    status = st.empty()
    
    try:
        tts = gTTS(text, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_file_path = fp.name
        
        st.session_state.audio_playing = True
        success = autoplay_audio(temp_file_path)
        
        if success:
            word_count = len(text.split())
            estimated_duration = max(3, word_count * 0.4 + 2)
            for remaining in range(int(estimated_duration), 0, -1):
                status.markdown(f" 🔊 Vyassa is speaking...  ")
                time.sleep(1)
        
        st.session_state.audio_playing = False
        
    except Exception as e:
        st.error(f"TTS Error: {e}")
        st.session_state.audio_playing = False
        return False
    
    status.empty()
    return True


def recognize_speech_enhanced():
    """Enhanced speech recognition with persistent recorder button"""
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("🎤 **Record your answer:**")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x",
            key=f"audio_recorder_{st.session_state.current_audio_key}"
        )
    
    with col2:
        if st.button("Skip Question", key=f"skip_{st.session_state.current_audio_key}"):
            return "Question skipped by user"
    
    if audio_bytes:
        # Show processing status
        with st.spinner("🔄 Moving on to the next question..."):
            try:
                # Save the audio file
                audio_path = os.path.join(os.getcwd(), f"audio_{st.session_state.current_audio_key}.wav")
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                
                # Transcribe the audio
                # result = model.transcribe(audio_path)
                # text = result['text'].strip()
                r = sr.Recognizer()
                with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                    audio_data = r.record(source)

                text = r.recognize_groq(audio_data)
                
                # Clean up the audio file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                
                if text:
                    st.session_state.current_audio_key += 1
                    return text
                else:
                    st.warning("⚠️ No speech detected. Please try again.")
                    return None
                    
            except Exception as e:
                st.error(f"⚠️ No speech detected. Please try again.")
                return None
    
    # If no audio yet, show waiting message
    if not st.session_state.processing_response:
        st.info("⏳ Waiting for your response...")
    
    return None


def get_remaining_time():
    if not st.session_state.interview_start_time:
        return 300
    elapsed = (datetime.now() - st.session_state.interview_start_time).total_seconds()
    return max(0, 300 - elapsed)


def safe_chat(prompt):
    """Defensive against blank or invalid input"""
    if prompt and isinstance(prompt, str) and prompt.strip():
        return st.session_state.chat_engine.chat(prompt).response
    return "Could you please clarify or rephrase your answer?"


# File upload section
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

if uploaded_file and not st.session_state.resume_uploaded:
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("🔍 Reading and indexing your resume..."):
            documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            client = qdrant_client.QdrantClient(location=":memory:")
            vector_store = QdrantVectorStore(client=client, collection_name="resume")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            
            system_prompt = """
            You are an interview Q&A assistant. Use the candidate's resume and documents to guide the conversation.

            Instructions:
            - Engage naturally: acknowledge each response in a simple sentence within a few words (e.g., "Got it," "Thanks for sharing," "That's helpful").
            - Keep the tone professional, friendly, and encouraging.
            - Do not repeat or rephrase questions that have already been asked.
            - If the candidate doesn’t respond, gently instruct them once, then continue to the next relevant question without waiting indefinitely.
            - Prioritize relevant experience, projects, and skills from the candidate’s documents to tailor your questions.
            - Vary your question style: mix technical, behavioral, and situational questions depending on the candidate’s background.
            - Maintain logical flow: ask follow-up questions when appropriate, especially about impactful roles or achievements.
            - Avoid yes/no questions unless they lead into a more in-depth topic.
            - Keep questions concise and easy to understand.
            - If the candidate mentions an experience, degree, or skill not listed in their resume, acknowledge it politely and ask them to elaborate with details (e.g., where, when, how they applied it).
            - Do not question or highlight inconsistencies—focus on letting the candidate explain their background.
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


# Sidebar progress - shows only if interview started
if st.session_state.interview_active and st.session_state.interview_start_time:
    remaining = get_remaining_time()
    total_time = 300
    mins, secs = divmod(int(remaining), 60)
    
    st.sidebar.info(f"🕒 Time Left: {mins:02d}:{secs:02d}")
    st.sidebar.progress(remaining / total_time, text="⏳ Interview Progress")
    st.sidebar.info(f"📊 Questions: {st.session_state.question_count}/5")
    



# Start interview button
if (st.session_state.resume_uploaded and
    st.session_state.chat_engine and
    not st.session_state.interview_active and
    not st.session_state.audio_playing):
    
    if st.button("🎯 Start Automated Interview", type="primary"):
        try:
            st.session_state.interview_active = True
            st.session_state.interview_start_time = datetime.now()
            st.session_state.chat_history = []
            st.session_state.question_count = 1
            st.session_state.current_audio_key = 0
            st.session_state.processing_response = False
            
            intro_prompt = """
            You are Vyassa, an AI interviewer.
            Greet the candidate briefly and ask them to tell you about themselves.
            """
            
            with st.spinner("🤖 Vyassa is preparing..."):
                intro_response = safe_chat(intro_prompt)
            
            st.session_state.chat_history.append(("Assistant", intro_response))
            
            # Play the first question immediately
            if not play_tts_with_display(intro_response):
                st.error("Failed to play initial question")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error starting interview: {e}")
            st.session_state.interview_active = False


# Interview in progress
elif (st.session_state.interview_active and
      st.session_state.chat_engine and
      not st.session_state.audio_playing):

    remaining_time = get_remaining_time()
    
    # Check if interview should end
    if remaining_time < 30 or st.session_state.question_count > 5:
        if not st.session_state.interview_ended:
            with st.spinner("🤖 Vyassa is concluding..."):
                closing = safe_chat(
                    "Acknowledge the previous response briefly, then end the interview politely and thank the candidate."
                )
            
            st.session_state.chat_history.append(("Assistant", closing))
            play_tts_with_display(closing)
            st.session_state.interview_active = False
            st.session_state.interview_ended = True
            
            st.success("🎉 Interview completed. Thank you for participating!")
            
            # Show final summary
            st.markdown("### 📊 Interview Summary")
            st.write(f"**Duration:** {5 - int(remaining_time/60)} minutes")
            st.write(f"**Questions Asked:** {st.session_state.question_count}")
            st.write(f"**Total Answer Time:** {int(st.session_state.total_answer_time)} seconds")
            
        st.rerun()

    else:
        # Main interview logic
        if st.session_state.chat_history:
            last_entry = st.session_state.chat_history[-1]
            
            # If last entry was from Assistant, we need user response
            if last_entry[0] == "Assistant":
                question = last_entry[1]
                
                # Display the current question
                st.markdown("### 🤖 Vyassa asks:")
                st.markdown(f"*{question}*")
                st.markdown("---")
                
                # Start answer timer if not already started
                if not st.session_state.answer_timer_start:
                    st.session_state.answer_timer_start = datetime.now()
                
                # Get user response
                user_input = recognize_speech_enhanced()
                
                if user_input:
                    # Calculate response time
                    if st.session_state.answer_timer_start:
                        answer_time = (datetime.now() - st.session_state.answer_timer_start).total_seconds()
                        st.session_state.total_answer_time += answer_time
                        st.session_state.answer_timer_start = None
                    
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.processing_response = True
                    st.rerun()
            
            # If last entry was from user, generate next question
            elif last_entry[0] == "You":
                last_user_input = last_entry[1]
                
                try:
                    st.session_state.processing_response = True
                    
                    with st.spinner("🤖 Vyassa is thinking..."):
                        response = safe_chat(last_user_input)
                    
                    st.session_state.chat_history.append(("Assistant", response))
                    st.session_state.question_count += 1
                    st.session_state.processing_response = False
                    
                    # Play the response
                    if not play_tts_with_display(response):
                        st.error("Failed to play audio response")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Interview error: {e}")
                    st.session_state.interview_active = False
                    st.session_state.processing_response = False
                    st.rerun()


# Display current status when not in active interview
elif not st.session_state.interview_active and not st.session_state.audio_playing:
    if not st.session_state.resume_uploaded:
        st.info("📄 Please upload your resume to get started.")
    elif st.session_state.interview_ended:
        st.success("✅ Interview completed!")
    else:
        st.info("🎯 Click 'Start Automated Interview' to begin.")


# Restart section
if st.session_state.interview_ended or (
    not st.session_state.interview_active and st.session_state.question_count > 0):
    
    st.markdown("---")
    st.markdown("### 🔄 Start Over")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Start New Interview", type="secondary"):
            # Reset all session state except resume data
            for key in defaults:
                if key not in ["chat_engine", "resume_uploaded"]:
                    st.session_state[key] = defaults[key]
            st.rerun()
    
    with col2:
        if st.button("📄 Upload New Resume", type="secondary"):
            # Reset everything
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
    Powered by Vyassa AI • Resume Interview Assistant
    </div>
    """, 
    unsafe_allow_html=True
)
