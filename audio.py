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
            - If the candidate doesn’t respond /
                gives a negative or unhelpful answer /
                says “I can’t understand your question” —
                gently guide them (e.g., rephrase the question / give a small hint).
                If they still hesitate, say:
                “Try to answer the question — right or wrong —
                it will help you understand what you’ve gone through.”
                Then continue to the next relevant question without waiting too long.
            - Prioritize relevant experience, projects, and skills from the candidate's documents to tailor your questions.
            - Vary your question style: mix technical, behavioral, and situational questions depending on the candidate’s background.
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

# ✅ Utility: Text-to-speech using embedded base64 autoplay
def play_tts_with_display(text):
    if not text.strip():
        return False
    st.session_state.current_question = text
    status = st.empty()
    status.markdown(f"**🤖 Vyassa is speaking...**  \n{text}", unsafe_allow_html=True)

    try:
        tts = gTTS(text, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            with open(fp.name, "rb") as f:
                audio_bytes = f.read()
            b64_audio = base64.b64encode(audio_bytes).decode()

            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            os.unlink(fp.name)
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return False

    status.empty()
    return True

# Speech recognition
def recognize_speech_enhanced():
    r = sr.Recognizer()
    r.pause_threshold = 1.75
    r.dynamic_energy_threshold = False
    status = st.empty()
    try:
        with sr.Microphone() as source:
            status.info("🎙️ Adjusting for ambient noise...")
            status.info("🎤 Vyassa is Listening...")
            audio = r.listen(source)
            status.info("⏭️ Moving on to next question...")
            text = r.recognize_groq(audio)
            status.empty()
            return text
    except Exception:
        status.warning("⚠️ Could not process speech.")
    status.empty()
    return "No response provided"

# Interview step
def conduct_interview_step(text_to_speak):
    if not text_to_speak.strip():
        return "No question provided"
    if not play_tts_with_display(text_to_speak):
        return "TTS failed"
    st.session_state.answer_timer_start = datetime.now()
    user_input = recognize_speech_enhanced()
    st.session_state.total_answer_time += (datetime.now() - st.session_state.answer_timer_start).total_seconds()
    st.session_state.answer_timer_start = None
    return user_input

# Remaining time
def get_remaining_time():
    used_time = st.session_state.total_answer_time
    if st.session_state.answer_timer_start:
        used_time += (datetime.now() - st.session_state.answer_timer_start).total_seconds()
    return max(0, 300 - used_time)

# Sidebar progress
if st.session_state.interview_active and st.session_state.interview_start_time:
    remaining = get_remaining_time()
    total_time = 300
    mins, secs = divmod(int(remaining), 60)
    st.sidebar.info(f"🕒 Time Left: {mins:02d}:{secs:02d}")
    st.sidebar.progress(remaining / total_time, text="⏳ Interview Progress")

# Start interview
if st.session_state.resume_uploaded and st.session_state.chat_engine and not st.session_state.interview_active:
    if st.button("🎯 Start Automated Interview"):
        try:
            st.session_state.interview_active = True
            st.session_state.interview_start_time = datetime.now()
            st.session_state.chat_history = []
            st.session_state.question_count = 1

            intro_prompt = """
            You are Vyassa, an AI interviewer.
            Greet the candidate, tell about you in two to three words, then ask them to tell you about themselves.
            """
            intro_response = st.session_state.chat_engine.chat(intro_prompt).response
            st.session_state.chat_history.append(("Assistant", intro_response))

            user_input = conduct_interview_step(intro_response)
            st.session_state.chat_history.append(("You", user_input))
            st.rerun()
        except Exception as e:
            st.error(f"Error starting interview: {e}")
            st.session_state.interview_active = False

# Ongoing interview
elif st.session_state.interview_active and st.session_state.chat_engine:
    remaining_time = get_remaining_time()

    if remaining_time < 40 or st.session_state.question_count > 4:
        closing = st.session_state.chat_engine.chat(
            "Acknowledge with the previous response, end the interview and thank the candidate."
        ).response
        st.session_state.chat_history.append(("Assistant", closing))
        play_tts_with_display(closing)
        st.session_state.interview_active = False
        st.session_state.interview_ended = True
        st.success("🎉 Interview completed. Thank you for participating.")
    else:
        if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "You":
            last_user_input = st.session_state.chat_history[-1][1]
            try:
                response = st.session_state.chat_engine.chat(last_user_input).response
                st.session_state.chat_history.append(("Assistant", response))

                user_input = conduct_interview_step(response)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.question_count += 1
                st.rerun()
            except Exception as e:
                st.error(f"Interview error: {e}")
                st.session_state.interview_active = False

# Restart interview
if st.session_state.interview_ended or (not st.session_state.interview_active and st.session_state.question_count > 0):
    st.markdown("---")
    if st.button("🔄 Start New Interview"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()
