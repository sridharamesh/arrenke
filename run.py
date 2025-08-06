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
    "waiting_for_audio": False,
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
    status.markdown(f"**🤖 Vyassa is speaking...**\n\n{text}")
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
                status.markdown("**🔊 Vyassa is speaking... ")
                time.sleep(1)
        st.session_state.audio_playing = False
    except Exception as e:
        st.error(f"TTS Error: {e}")
        st.session_state.audio_playing = False
        return False
    status.empty()
    return True

def recognize_speech_enhanced():
    status = st.empty()
    try:
        status.info("🎤 Click the microphone to record your answer!")
        audio_bytes = audio_recorder()
        if audio_bytes:
            status.info("🔄 Moving on to the next question...")
            audio_path = os.path.join(os.getcwd(), "sample_audio.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
        
            try:
                api_key = os.getenv("GROQ_API_KEY")
                client = Groq(api_key=api_key)  # <-- Replace this with your actual API key
                with open(audio_path, "rb") as file:
                        transcription = client.audio.transcriptions.create(
                      file=file, # Required audio file
                      model="whisper-large-v3-turbo", # Required model to use for transcription
                      prompt="Specify context or spelling",  # Optional
                      response_format="verbose_json",  # Optional
                      timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
                      language="en",  # Optional
                      temperature=0.0  # Optional
                    )
                text = transcription.text
                status.empty()
                return text if text else "No response provided"

            except Exception as transcription_error:
                status.error(f"⚠️ Transcription error: {str(transcription_error)}")
                return "Transcription error"
        else:
            status.warning("⏳ Waiting for your voice input...")
            return None

    except Exception as e:
        status.error(f"⚠️ Speech recognition error: {str(e)}")
        return "Speech recognition error"


def conduct_interview_step(text_to_speak):
    if not text_to_speak.strip():
        return "No question provided"
    # Play the question
    if not play_tts_with_display(text_to_speak):
        return "Speech synthesis failed."
    st.session_state.waiting_for_audio = True
    st.session_state.answer_timer_start = datetime.now()
    user_input = recognize_speech_enhanced()
    if user_input: # valid response
        st.session_state.waiting_for_audio = False
        if st.session_state.answer_timer_start:
            answer_time = (datetime.now() - st.session_state.answer_timer_start).total_seconds()
            st.session_state.total_answer_time += answer_time
            st.session_state.answer_timer_start = None
        return user_input
    return None

def get_remaining_time():
    if not st.session_state.interview_start_time:
        return 300
    elapsed = (datetime.now() - st.session_state.interview_start_time).total_seconds()
    return max(0, 300 - elapsed)

def safe_chat(prompt):
    # Defensive against blank or invalid input
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
            - Engage naturally: acknowledge each response in a simple sentence within few words not more than ten words(e.g., "Got it," "Thanks for sharing," "That's helpful").
            - Keep the tone professional, friendly, and encouraging.
            - Do not repeat or rephrase questions that have already been asked.
            - If the candidate doesn’t respond, gently instruct them once, then continue to the next relevant question without waiting indefinitely.
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
    not st.session_state.audio_playing and
    not st.session_state.waiting_for_audio):
    if st.button("🎯 Start Automated Interview"):
        try:
            st.session_state.interview_active = True
            st.session_state.interview_start_time = datetime.now()
            st.session_state.chat_history = []
            st.session_state.question_count = 1
            intro_prompt = """
            You are Vyassa, an AI interviewer.
            Greet the candidate briefly and ask them to tell you about themselves.
            """
            with st.spinner("🤖 Vyassa is preparing..."):
                intro_response = safe_chat(intro_prompt)
            st.session_state.chat_history.append(("Assistant", intro_response))
            st.rerun()
        except Exception as e:
            st.error(f"Error starting interview: {e}")
            st.session_state.interview_active = False

# Interview in progress
elif (st.session_state.interview_active and
      st.session_state.chat_engine and
      not st.session_state.audio_playing):

    remaining_time = get_remaining_time()
    # End the interview if overtime or question limit exceeded
    if remaining_time < 30 or st.session_state.question_count > 5:
        with st.spinner("🤖 Vyassa is concluding..."):
            closing = safe_chat(
                "Acknowledge the previous response briefly, then end the interview politely and thank the candidate."
            )
        st.session_state.chat_history.append(("Assistant", closing))
        play_tts_with_display(closing)
        st.session_state.interview_active = False
        st.session_state.interview_ended = True
        st.session_state.waiting_for_audio = False
        st.success("🎉 Interview completed. Thank you for participating!")

    else:
        if st.session_state.waiting_for_audio:
            result = recognize_speech_enhanced()
            if result:
                st.session_state.waiting_for_audio = False
                st.session_state.chat_history.append(("You", result))
                st.rerun()
        else:
            # If it's Assistant's turn to ask, speak the question
            if (st.session_state.chat_history and
                st.session_state.chat_history[-1][0] == "Assistant" and
                len([msg for msg in st.session_state.chat_history if msg[0] == "You"]) < st.session_state.question_count):
                question = st.session_state.chat_history[-1][1]
                user_input = conduct_interview_step(question)
                if user_input:
                    st.session_state.chat_history.append(("You", user_input))
                    st.rerun()
            elif (st.session_state.chat_history and
                  st.session_state.chat_history[-1][0] == "You"):
                last_user_input = st.session_state.chat_history[-1][1]
                try:
                    with st.spinner("🤖 Vyassa is thinking..."):
                        response = safe_chat(last_user_input)
                    st.session_state.chat_history.append(("Assistant", response))
                    st.session_state.question_count += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Interview error: {e}")
                    st.session_state.interview_active = False

# Restar
if st.session_state.interview_ended or (
    not st.session_state.interview_active and st.session_state.question_count > 0):
    st.markdown("### 🔄 Start Over")
    if st.button("🔄 Start New Interview"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()
