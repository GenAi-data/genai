import streamlit as st
from llama_cpp import Llama
from pathlib import Path
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import atexit

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_path = Path(os.path.expanduser("~")) / "code" / "modelslist" / "deepseek-1_3b-gguf" / "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
# Streamlit UI setup
st.set_page_config(page_title="🧠 GenAI ChatBot", layout="wide")
st.title("💬 DeepSeek 1.3B - Local LLM ChatBot (Async GGUF)")

with st.expander("📘 How the Model Works"):
    resolved_path = model_path.resolve()
    st.markdown("""
| Parameter        | Description |
|------------------|-------------|
| `model_path`     | `C:\\Users\\%USERPROFILE%\\code\\modelslist\\deepseek-1_3b-gguf\\deepseek-coder-1.3b-instruct.Q4_K_M.gguf`  |
| `n_ctx=2048`     | Max tokens the model can process per input |
| `n_threads=6`    | Number of CPU threads to use |
| `n_gpu_layers=0` | 0 = use CPU only. Increase if using GPU |
| `temperature=0.7`| Controls randomness. Lower = safer |
| `top_p=0.95`     | Nucleus sampling: picks top 95% likely words |
| `repeat_penalty=1.1` | Penalizes repetitive output |
| `verbose=False`  | Disables detailed logs for cleaner output |
    """)

    st.markdown("""
> The `Llama()` model is locally loaded using the `llama-cpp-python` backend.  
It generates smart replies based on recent messages. You can control how creative or deterministic the model is using parameters like `temperature`, `top_p`, and `repeat_penalty`.
""")

with st.expander("💡 Sample Prompts to Try"):
    st.markdown("""
#### 💻 Programming
- `Write a Python function to validate email addresses.`
- `Convert this SQL to PySpark: SELECT * FROM orders WHERE status = 'shipped'`
- `How do I merge two DataFrames in pandas?`

#### 🧪 Data Quality
- `What is a data quality rule?`
- `Suggest 3 checks for null values in an ETL pipeline.`

#### ☁️ Cloud & GenAI
- `What is IAM in AWS and why is it important?`
- `Explain what RAG (Retrieval-Augmented Generation) means.`

#### 🧠 Knowledge
- `What’s the difference between AI, ML, and DL?`
- `Who was the first President of India?`
    """)



# Thread-safe globals
executor = ThreadPoolExecutor(max_workers=1)
llm = None

# Load model
def load_model():
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = Llama(
        model_path=str(model_path),
        n_ctx=3072,             # ⬆️ Increases context window (default: 2048)
        n_threads=6,            # 🧠 Maximize thread usage for your CPU (Ryzen 6-core)
        n_gpu_layers=0,         # ⚙️ Keep 0 for CPU-only
        temperature=0.65,       # 🎯 Lower = more focused/accurate answers
        top_p=0.9,              # 📉 Slightly restricts randomness for coherence
        repeat_penalty=1.2,     # 🔁 Penalizes repetition more aggressively
        verbose=False           # 🔇 Quiet logs unless debugging
    )

    try:
        test = model.create_chat_completion(
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5
        )
        logger.info(f"Model test response: {test['choices'][0]['message']['content']}")
    except Exception as e:
        logger.warning(f"Model test skipped: {e}")
    
    return model

# Cleanup on exit
def cleanup():
    executor.shutdown(wait=False)
    logger.info("Cleaned up executor.")

atexit.register(cleanup)

# Ensure messages are in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a highly knowledgeable, friendly AI assistant. Respond clearly, use examples where helpful, format code properly, and always explain your reasoning in simple terms."}]

# Load model once
if "model_loaded" not in st.session_state:
    with st.spinner("🔄 Loading DeepSeek model..."):
        try:
            llm = load_model()
            st.session_state.llm = llm  # store for visibility
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()
else:
    llm = st.session_state.llm

# Async generation
async def generate_response(prompt, llm_ref, msg_history):
    loop = asyncio.get_event_loop()
    chat_messages = msg_history[-4:] + [{"role": "user", "content": prompt}]
    try:
        response = await loop.run_in_executor(
            executor,
            lambda: llm_ref.create_chat_completion(
                messages=chat_messages,
                max_tokens=512
            )["choices"][0]["message"]["content"]
        )
        return response
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return f"⚠️ Error generating response: {e}"

# Show chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(generate_response(prompt, llm, st.session_state.messages))
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.subheader("🧪 Controls")
    if st.button("🔁 Reset Chat"):
        
        st.session_state.messages = [{"role": "system", "content": "You are a highly knowledgeable, friendly AI assistant. Respond clearly, use examples where helpful, format code properly, and always explain your reasoning in simple terms."}]
        st.rerun()
    st.write(f"💬 Messages: {len([m for m in st.session_state.messages if m['role'] != 'system'])}")
    st.write(f"📦 Model Status: {'✅ Loaded' if st.session_state.get('model_loaded') else '❌ Not Loaded'}")
