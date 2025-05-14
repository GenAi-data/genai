import streamlit as st
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv
from pathlib import Path
from llama_cpp import Llama
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
LOCAL_MODE = True  # Configured for local model only
MODEL_PATH = Path(os.path.expanduser("~")) / "code" / "modelslist" / "deepseek-1_3b-gguf" / "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"

# Predefined LLM mappings
LLM_MAPPINGS = {
    "programming": [
        {"name": "CodeLlama", "description": "Fine-tuned for code generation and debugging across languages.", "url": "https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf"},
        {"name": "DeepSeek-Coder", "description": "Optimized for complex coding tasks and completions.", "url": "https://huggingface.co/deepseek/DeepSeek-Coder-33B-instruct"}
    ],
    "healthcare": [
        {"name": "BioMedLM", "description": "Specialized for medical texts and clinical data analysis.", "url": "https://huggingface.co/stanford-crfm/BioMedLM"},
        {"name": "ClinicalBERT", "description": "Fine-tuned for clinical NLP tasks.", "url": "https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT"}
    ],
    "finance": [
        {"name": "FinBERT", "description": "Trained for financial sentiment analysis.", "url": "https://huggingface.co/ProsusAI/finbert"},
        {"name": "FinancialBERT", "description": "Fine-tuned for financial text analysis.", "url": "https://huggingface.co/ahmedrachid/FinancialBERT"}
    ],
    "education": [
        {"name": "BERT-Edu", "description": "Fine-tuned for educational text processing.", "url": "https://huggingface.co/textattack/bert-base-uncased"},  # Placeholder
        {"name": "EduRoBERTa", "description": "Optimized for educational content.", "url": "https://huggingface.co/roberta-base"}  # Placeholder
    ],
    "legal": [
        {"name": "LegalBERT", "description": "Specialized for legal document analysis.", "url": "https://huggingface.co/nlpaueb/legal-bert"},
        {"name": "InLegalBERT", "description": "Trained for Indian legal texts.", "url": "https://huggingface.co/law-ai/InLegalBERT"}
    ]
}

# Initialize components
@st.cache_resource
def load_local_llm():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        logger.info(f"Loading local LLM from {MODEL_PATH}")
        return Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            temperature=0.7,
            verbose=False
        )
    except Exception as e:
        logger.exception(f"Failed to load local LLM: {e}")
        return None

# Thread executor
executor = ThreadPoolExecutor(max_workers=1)

# Cleanup on exit
def cleanup():
    executor.shutdown(wait=False)
    logger.info("Cleaned up executor.")

import atexit
atexit.register(cleanup)

# Streamlit UI
st.set_page_config(page_title="🧠 LLM Recommender", layout="wide")
st.title("💬 Domain-Specific LLM Recommendation ChatBot")

with st.expander("📘 How the ChatBot Works"):
    st.markdown("""
| Parameter        | Description |
|------------------|-------------|
| `mode`           | Local (`deepseek-coder-1.3b-instruct.Q4_K_M.gguf`) |
| `max_tokens`     | 512 tokens for response length |
| `temperature`    | 0.7 for balanced randomness |
| `output`         | Streamlit DataFrame with LLM name, description, and download URL |

> This app recommends LLMs for specific domains (e.g., programming, healthcare, finance) using a local LLM. Results are displayed in a Streamlit DataFrame.
""")

with st.expander("💡 Sample Prompts to Try"):
    st.markdown("""
#### 🎯 Domain-Specific Questions
- `Which LLM is best for programming tasks?`
- `What’s the best LLM for healthcare applications?`
- `Which model should I use for financial analysis?`
- `What LLM is suitable for educational content creation?`
- `Can you recommend an LLM for legal document processing?`
""")

# Initialize local LLM
local_llm = load_local_llm()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "Follow the instructions provided in the user query formatting."
            )
        }
    ]

# LangChain PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template="""
You are a knowledgeable AI assistant specializing in recommending large language models (LLMs) for specific domains such as programming, healthcare, finance, education, and legal.
When a user asks about LLMs for a domain, first clearly state the domain you identified (e.g., 'The domain is programming'), then provide a brief explanation of why certain LLMs are suitable for that domain, focusing on their strengths.
Do not mention specific model names or URLs. If the user does not specify a domain, ask them to clarify.

User Query: {user_query}
"""
)

# LangChain Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,  # Buffer below 2048 for response tokens
    chunk_overlap=100,  # Maintain context
    length_function=lambda text: len(tiktoken.get_encoding("cl100k_base").encode(text))
)

# Async response generation
async def get_llm_response(prompt, message_history):
    try:
        # Format the user's query with PromptTemplate
        formatted_prompt = prompt_template.format(user_query=prompt)

        # Combine all messages into a single string for splitting
        full_conversation = ""
        for msg in message_history[-4:] + [{"role": "user", "content": formatted_prompt}]:
            full_conversation += f"{msg['role'].capitalize()}: {msg['content']}\n"

        # Check token count
        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_count = len(tokenizer.encode(full_conversation))

        if token_count > 1800:
            # Split the conversation if it exceeds the limit
            chunks = text_splitter.split_text(full_conversation)
            # Use the most recent chunk that fits
            full_conversation = chunks[-1]

        # Reconstruct the message list
        valid_messages = [{"role": "system", "content": "Follow the instructions provided in the user query formatting."}]
        for line in full_conversation.split("\n"):
            if line.startswith("User:"):
                valid_messages.append({"role": "user", "content": line[6:].strip()})
            elif line.startswith("Assistant:"):
                valid_messages.append({"role": "assistant", "content": line[11:].strip()})

        if LOCAL_MODE and local_llm:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: local_llm.create_chat_completion(
                    messages=valid_messages,
                    max_tokens=512,
                    temperature=0.7
                )["choices"][0]["message"]["content"]
            )
            # Detect domain from prompt for LLM recommendations
            domain = next((k for k in LLM_MAPPINGS if k in prompt.lower()), None)
            llms = LLM_MAPPINGS.get(domain, []) if domain else []
            if not domain:
                response += "\n\nPlease specify a domain (e.g., programming, healthcare, finance) to receive LLM recommendations."
            return response, llms
        else:
            return "⚠️ No LLM backend available", []
    except Exception as e:
        logger.exception(f"Generation error: {e}")
        return f"⚠️ Error: {str(e)}", []

# Async wrapper for handling input
async def handle_input(prompt):
    response, llms = await get_llm_response(prompt, st.session_state.messages)
    return response, llms

# Display chat
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and isinstance(msg["content"], tuple):
                response, llms = msg["content"]
                st.markdown(response)
                if llms:
                    df = pd.DataFrame(llms)
                    st.dataframe(df, use_container_width=True)
            else:
                st.markdown(msg["content"])

# Handle input
if prompt := st.chat_input("Ask about LLMs for a domain..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, llms = asyncio.run(handle_input(prompt))
                st.markdown(response)
                if llms:
                    df = pd.DataFrame(llms)
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                logger.exception(f"App crashed during response generation: {e}")
                st.error(f"❌ Error: {e}")
                response, llms = "⚠️ An error occurred. Please try again.", []

    st.session_state.messages.append({"role": "assistant", "content": (response, llms)})

# Sidebar controls
with st.sidebar:
    st.subheader("🧪 Controls")
    if st.button("🔁 Reset Chat"):
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "Follow the instructions provided in the user query formatting."
                )
            }
        ]
        st.rerun()
    st.write(f"💬 Messages: {len([m for m in st.session_state.messages if m['role'] != 'system'])}")
    st.write(f"🔧 Mode: {'Local' if LOCAL_MODE else 'Hugging Face API'}")
    st.write(f"📦 Local Model: {MODEL_PATH.name}")