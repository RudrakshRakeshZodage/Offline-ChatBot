import os
import io
import tempfile
import requests
import streamlit as st
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import docx
import speech_recognition as sr
from pydub import AudioSegment

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "llama3"
PAGE_TITLE  = "🧠 Offline AI Suite"
ICON        = "🤖"

st.set_page_config(page_title=PAGE_TITLE, page_icon=ICON, layout="wide")

# ✅ Set tesseract path (Update if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def chat_with_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.RequestException as e:
        return f"❌ API Error: {e}"

def extract_text_from_pdf(uploaded) -> str:
    try:
        with fitz.open(stream=uploaded.read(), filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

def extract_text_from_docx(uploaded) -> str:
    try:
        doc = docx.Document(uploaded)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"❌ Error reading DOCX: {e}")
        return ""

def extract_text_from_image(uploaded_img) -> str:
    try:
        image = Image.open(uploaded_img)
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"❌ Error reading image: {e}")
        return ""

def convert_to_wav(uploaded_file) -> str:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in_path = tmp_in.name
    
    wav_path = tmp_in_path + ".wav"
    audio = AudioSegment.from_file(tmp_in_path)  # pydub auto detects format
    audio.export(wav_path, format="wav")
    
    os.remove(tmp_in_path)  # cleanup original temp file
    return wav_path

def transcribe_audio(uploaded_file) -> str:
    wav_path = convert_to_wav(uploaded_file)
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as src:
            audio = recognizer.record(src)
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "❌ Could not understand audio."
    except sr.RequestError as e:
        text = f"❌ Speech recognition error: {e}"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    return text

# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────
st.title(f"{ICON} {PAGE_TITLE}")
tab_chat, tab_docs = st.tabs(["💬 Chatbot", "📄 Docs + OCR Upload"])

# ──────────────────────────────────────────────────────────────────────────────
# CHAT TAB
# ──────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.header("💬 Chat with Local LLM (Ollama)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me something...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_ollama(user_input)
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

# ──────────────────────────────────────────────────────────────────────────────
# DOCS TAB
# ──────────────────────────────────────────────────────────────────────────────
with tab_docs:
    st.header("📄 Upload PDF, Word, or Image for RAG")

    uploaded_file = st.file_uploader("Upload PDF, DOCX, or Image", type=["pdf", "docx", "png", "jpg", "jpeg"])
    extracted_text = ""

    if uploaded_file:
        file_ext = uploaded_file.name.lower()
        if file_ext.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif file_ext.endswith(".docx"):
            extracted_text = extract_text_from_docx(uploaded_file)
        elif file_ext.endswith((".png", ".jpg", ".jpeg")):
            extracted_text = extract_text_from_image(uploaded_file)

        if extracted_text.strip():
            st.success("✅ Text Extracted Successfully")
            st.text_area("📄 Extracted Text", value=extracted_text, height=300)
        else:
            st.warning("⚠️ No readable text found in the uploaded file.")

    st.subheader("🎤 Optional: Voice Input (wav, mp3, mpeg, mp4 supported)")
    voice_up = st.file_uploader("Upload Audio/Video file", type=["wav", "mp3", "mpeg", "mp4"])
    voice_txt = ""

    if voice_up:
        try:
            voice_txt = transcribe_audio(voice_up)
            st.info(f"🎧 Recognized Text: **{voice_txt}**")
        except Exception as e:
            st.error(f"Speech Recognition error: {e}")

    user_question = st.text_input("💬 Ask a question about the uploaded content", value=voice_txt)
    if user_question and extracted_text.strip():
        final_prompt = f"Use the extracted content below to answer:\n\n{extracted_text}\n\nQuestion: {user_question}"
        with st.spinner("Thinking..."):
            answer = chat_with_ollama(final_prompt)
        st.success("🤖 Answer")
        st.write(answer)
