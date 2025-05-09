import streamlit as st
from llama_cpp import Llama
from pathlib import Path
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# App Config
st.set_page_config(page_title="🧠 GenAI ChatBot", layout="wide")
st.title("💬 DeepSeek 1.3B - Local LLM ChatBot (Async GGUF)")

# Path Setup
user_base = Path(os.environ.get("USERPROFILE", "C:/Users/Default"))
model_path = user_base / "code" / "modelslist" / "deepseek-1_3b-gguf" / "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"

if not model_path.exists():
    st.error(f"❌ Model not found at:\n{model_path}")
    st.stop()

# Load model once in session
if "llm_model" not in st.session_state:
    st.session_state.llm_model = Llama(
        model_path=str(model_path),
        n_ctx=1024,
        n_threads=6,
        temperature=0.7,
        top_p=0.95,
        repeat_penalty=1.1
    )

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Async generation function using ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)

async def generate_async_response(prompt_messages):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: st.session_state.llm_model.create_chat_completion(
        messages=prompt_messages,
        max_tokens=512
    )["choices"][0]["message"]["content"])

# Chat History UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.spinner("Thinking..."):
        prompt_messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + st.session_state.messages[-5:]
        response = asyncio.run(generate_async_response(prompt_messages))

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
