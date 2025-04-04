
import streamlit as st
from llama_cpp import Llama
import os
import requests
from tqdm import tqdm
import datetime

MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"
LOG_FILE = "teacher_logs.txt"

# Download model if not already present with progress bar
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (~4GB)... this may take a few minutes..."):
            response = requests.get(MODEL_URL, stream=True)
            total = int(response.headers.get('content-length', 0))
            with open(MODEL_PATH, "wb") as f, st.progress(0) as progress_bar:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_bar.progress(min(downloaded / total, 1.0))
            st.success("Model downloaded successfully!")

# Load the model (ensure GGUF is downloaded)
@st.cache_resource
def load_model():
    download_model()
    return Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=2048,
        chat_format="chatml"
    )

llm = load_model()

st.title("🎓 Educational AI Assistant")
st.markdown("Generate teaching materials, summaries, questions, and more!")

# Simple login system
authenticated = False
with st.sidebar:
    st.header("🔐 Login")
    user_type = st.radio("Are you a:", ["Teacher", "Student"])
    password = st.text_input("Enter access code:", type="password")
    if user_type == "Teacher" and password == "teach123":
        authenticated = True
        st.success("Logged in as Teacher")
    elif user_type == "Student" and password == "learn123":
        authenticated = True
        st.success("Logged in as Student")
    elif password:
        st.error("Incorrect access code")

if authenticated:
    # Preset prompts
    st.markdown("### 🎯 Choose a preset subject prompt or write your own:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📐 Math Quiz"):
            st.session_state["preset"] = "Create a math quiz for Grade 9 students on algebra."
    with col2:
        if st.button("📖 English Summary"):
            st.session_state["preset"] = "Summarize Act 2 of Macbeth for a high school audience."
    with col3:
        if st.button("🔬 Science Questions"):
            st.session_state["preset"] = "Generate 5 multiple choice questions on cell biology for Grade 10."

    col4, col5 = st.columns(2)
    with col4:
        if st.button("🏛️ History Summary"):
            st.session_state["preset"] = "Summarize the causes of the American Revolution for Grade 11."
    with col5:
        if st.button("💻 CS Concepts"):
            st.session_state["preset"] = "Explain how recursion works in Python for high school students."

    # Prompt input
    default_prompt = st.session_state.get("preset", "")
    user_prompt = st.text_area("Enter your prompt:", value=default_prompt)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Generate"): 
        if user_prompt.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating response..."):
                result = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that helps teachers and students with educational tasks."},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                response = result["choices"][0]["message"]["content"]
                st.session_state.chat_history.append((user_prompt, response))
                st.success("Done!")

                if user_type == "Teacher":
                    with open(LOG_FILE, "a") as log:
                        log.write(f"[{datetime.datetime.now()}]\nPrompt: {user_prompt}\nResponse: {response}\n\n")

    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")
else:
    st.warning("Please login using the sidebar to access the assistant.")
