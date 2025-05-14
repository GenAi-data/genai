import streamlit as st
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv
from pathlib import Path
from llama_cpp import Llama
import pandas as pd
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from huggingface_hub import InferenceClient
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Follow the instructions provided in the user query formatting.")
    ]
if "last_llms" not in st.session_state:
    st.session_state.last_llms = []

# Configuration
LOCAL_MODE = False  # Must use Hugging Face API for demo
DEFAULT_HF_API_TOKEN = os.getenv("HF_API_TOKEN", "qwertyuiopasdfghjklzxcvbnm")  # Replace with your token
MODEL_PATH = Path(os.path.expanduser("~")) / "code" / "modelslist" / "deepseek-1_3b-gguf" / "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"

# Supported Hugging Face models
HF_MODELS = [
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Versatile model for instruction-following and chat tasks."
    },
    {
        "name": "Granite-3.3-8B-Instruct",
        "id": "ibm-granite/granite-3.3-8b-instruct",
        "description": "IBM’s 8B model fine-tuned for instruction-following with structured reasoning via <think> and <response> tags."
    },
    {
        "name": "Grok-3-gemma3-12B-distilled",
        "id": "reedmayhew/Grok-3-gemma3-12B-distilled",
        "description": "Optimized for reasoning and domain-specific queries, developed by xAI."
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "High-performance model for complex reasoning tasks."
    },
    {
        "name": "Gemma-3-27B-IT",
        "id": "google/gemma-3-27b-it",
        "description": "Google’s 27B multimodal model for long context text/image understanding with 128K context."
    }
]


DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

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
        {"name": "BERT-Edu", "description": "Fine-tuned for educational text processing.", "url": "https://huggingface.co/textattack/bert-base-uncased"},
        {"name": "EduRoBERTa", "description": "Optimized for educational content.", "url": "https://huggingface.co/roberta-base"}
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

def initialize_hf_client(model_id, api_token):
    try:
        logger.info(f"Initializing Hugging Face Inference Client for {model_id}")
        return InferenceClient(model=model_id, token=api_token)
    except Exception as e:
        logger.exception(f"Failed to initialize HF client: {e}")
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
| `mode`           | Local (`deepseek-coder-1.3b-instruct.Q4_K_M.gguf`) or Hugging Face API (configurable model) |
| `max_tokens`     | 512 tokens for response length |
| `temperature`    | 0.7 for balanced randomness |
| `output`         | Streamlit DataFrame with LLM name, description, and download URL |

> This app recommends LLMs for specific domains (e.g., programming, healthcare, finance) using a local LLM or Hugging Face API. Results are displayed in a Streamlit DataFrame.
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

# Sidebar controls
with st.sidebar:
    st.subheader("🧪 Controls")
    
    mode = st.selectbox("Select Mode", ["Hugging Face API"], index=0)  # Only Hugging Face API for demo
    LOCAL_MODE = False  # Enforce Hugging Face API
    
    model_id = st.selectbox(
        "Select Hugging Face Model",
        options=[model["id"] for model in HF_MODELS],
        format_func=lambda x: next(model["name"] for model in HF_MODELS if model["id"] == x),
        index=[model["id"] for model in HF_MODELS].index(DEFAULT_MODEL_ID)
    )
    api_token = st.text_input("Hugging Face API Token", value=DEFAULT_HF_API_TOKEN, type="password")
    
    with st.expander("ℹ️ Help - Popular LLMs"):
        st.markdown("Explore popular LLMs for Hugging Face API mode:")
        df_models = pd.DataFrame(HF_MODELS)
        st.dataframe(df_models[["name", "description"]], use_container_width=True)
        selected_model = st.selectbox(
            "Choose a Model",
            options=[model["id"] for model in HF_MODELS],
            format_func=lambda x: next(model["name"] for model in HF_MODELS if model["id"] == x),
            index=[model["id"] for model in HF_MODELS].index(model_id)
        )
        if selected_model != model_id:
            model_id = selected_model
            st.rerun()
    
    if st.button("🔁 Reset Chat"):
        st.session_state.messages = [SystemMessage(content="Follow the instructions provided in the user query formatting.")]
        st.session_state.last_llms = []
        st.rerun()
    
    message_count = len([m for m in st.session_state.messages if m.type != 'system'])
    st.write(f"💬 Messages: {message_count}")
    st.write(f"🔧 Mode: {mode}")
    st.write(f"🌐 API Model: {next(model['name'] for model in HF_MODELS if model['id'] == model_id)}")

# Initialize clients
hf_client = initialize_hf_client(model_id, api_token)

# LangChain PromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
You are a knowledgeable AI assistant specializing in recommending large language models (LLMs), which are artificial intelligence models designed for natural language processing tasks. Your task is to recommend LLMs for specific domains such as programming, healthcare, finance, education, and legal.

When a user asks about LLMs for a domain, follow these steps:
1. Identify the domain from the query (e.g., finance, programming). State the identified domain clearly (e.g., 'The domain is finance').
2. Explain why certain LLMs are suitable for that domain, focusing on their technical strengths (e.g., ability to process financial texts, perform sentiment analysis, or handle numerical data). Do not mention specific model names or URLs.
3. If the domain is unclear, ask the user to specify a domain (e.g., programming, healthcare, finance).

For example, if the query is 'Which model should I use for financial analysis?', respond with:
'The domain is finance. LLMs for financial analysis should excel in processing financial texts, performing sentiment analysis on market data, and understanding economic terminology. Models trained on financial corpora are ideal for tasks like risk assessment and market trend prediction.'

Do not interpret 'LLM' as an academic degree (e.g., Master of Laws). Focus solely on AI language models.
"""),
    HumanMessagePromptTemplate.from_template("{user_query}")
])

# LangChain Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=100,
    length_function=lambda text: len(tiktoken.get_encoding("cl100k_base").encode(text))
)

# Ensure valid alternating roles for Hugging Face API
def ensure_valid_hf_sequence(messages, new_user_input):
    fixed_sequence = []
    last_role = None

    # Convert LangChain messages to dictionary format
    dict_messages = [
        {"role": "system" if isinstance(msg, SystemMessage) else "user" if isinstance(msg, HumanMessage) else "assistant",
         "content": msg.content}
        for msg in messages
    ]

    # Always start with system message if present
    for msg in dict_messages:
        if msg["role"] == "system":
            fixed_sequence.append(msg)
            last_role = "system"
            break

    # Process remaining messages ensuring alternation
    for msg in dict_messages:
        if msg["role"] != "system":
            if msg["role"] == last_role:
                # Insert dummy assistant response to fix consecutive user messages
                if last_role == "user":
                    fixed_sequence.append({"role": "assistant", "content": "Understood."})
                    last_role = "assistant"
                elif last_role == "assistant":
                    continue  # Skip duplicate assistant messages
            fixed_sequence.append(msg)
            last_role = msg["role"]

    # Append current user input
    if last_role != "user":
        fixed_sequence.append({"role": "user", "content": new_user_input})
    else:
        # Fix consecutive user message case
        fixed_sequence.append({"role": "assistant", "content": "Got it."})
        fixed_sequence.append({"role": "user", "content": new_user_input})

    # Convert back to LangChain messages
    langchain_messages = []
    for msg in fixed_sequence:
        if msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))
    
    return langchain_messages

# Async response generation
async def get_llm_response(prompt, message_history):
    try:
        # Format the user's query with ChatPromptTemplate
        formatted_messages = prompt_template.format_messages(user_query=prompt)
        
        # Ensure valid sequence for Hugging Face API
        valid_messages = ensure_valid_hf_sequence(message_history, prompt)
        
        # Validate role alternation
        roles = [msg.type for msg in valid_messages]
        for i in range(1, len(roles)):
            if roles[i] == roles[i-1] and roles[i] != "system":
                raise ValueError(f"Invalid sequence: consecutive {roles[i]} messages detected")
        
        # Convert to string for token counting
        full_conversation = ""
        for msg in valid_messages:
            role = "system" if isinstance(msg, SystemMessage) else "user" if isinstance(msg, HumanMessage) else "assistant"
            full_conversation += f"{role.capitalize()}: {msg.content}\n"
        
        # Check token count and truncate if necessary
        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_count = len(tokenizer.encode(full_conversation))
        if token_count > 1800:
            chunks = text_splitter.split_text(full_conversation)
            full_conversation = chunks[-1]
            valid_messages = [SystemMessage(content=formatted_messages[0].content)]
            last_role = "system"
            for line in full_conversation.split("\n"):
                if line.startswith("User:") and (last_role == "system" or last_role == "assistant"):
                    valid_messages.append(HumanMessage(content=line[6:].strip()))
                    last_role = "user"
                elif line.startswith("Assistant:") and last_role == "user":
                    valid_messages.append(AIMessage(content=line[11:].strip()))
                    last_role = "assistant"
        
        # Convert to JSON format for Hugging Face API
        json_messages = [
            {"role": "system" if msg.type == "system" else "user" if msg.type == "human" else "assistant",
             "content": msg.content}
            for msg in valid_messages
        ]
        
        # Log the messages for debugging
        logger.info(f"Messages sent to API: {json_messages}")
        
        if hf_client:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: hf_client.chat_completion(
                    messages=json_messages,
                    max_tokens=512,
                    temperature=0.7
                ).choices[0].message.content
            )
        else:
            return "⚠️ No Hugging Face client available. Check API token.", []
        
        # Detect domain for LLM recommendations
        domain = next((k for k in LLM_MAPPINGS if k in prompt.lower()), None)
        llms = LLM_MAPPINGS.get(domain, []) if domain else []
        if not domain:
            response += "\n\nPlease specify a domain (e.g., programming, healthcare, finance) to receive LLM recommendations."
        return response, llms
    except Exception as e:
        logger.exception(f"Generation error: {e}")
        return f"⚠️ Error: {str(e)}. Verify your API token or select a different model.", []

# Async wrapper for handling input
async def handle_input(prompt):
    response, llms = await get_llm_response(prompt, st.session_state.messages)
    return response, llms

# Display chat
for msg in st.session_state.messages:
    if msg.type != "system":
        with st.chat_message("user" if msg.type == "human" else "assistant"):
            st.markdown(msg.content)
            if msg.type == "ai" and st.session_state.last_llms:
                df = pd.DataFrame(st.session_state.last_llms)
                st.dataframe(df, use_container_width=True)

# Handle input
if prompt := st.chat_input("Ask about LLMs for a domain..."):
    # Prevent appending user message if the last message is also user
    if not st.session_state.messages or not isinstance(st.session_state.messages[-1], HumanMessage):
        st.session_state.messages.append(HumanMessage(content=prompt))
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
                    st.session_state.last_llms = llms
                    st.session_state.messages.append(AIMessage(content=response))
                except Exception as e:
                    logger.exception(f"App crashed during response generation: {e}")
                    st.error(f"❌ Error: {e}. Verify your API token or select a different model.")
                    response, llms = "⚠️ An error occurred. Please try again.", []
                    st.session_state.last_llms = llms
                    st.session_state.messages.append(AIMessage(content=response))
    else:
        st.error("Please wait for the assistant to respond before sending another message.")
