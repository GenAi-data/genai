import os
from pathlib import Path
import platform
from typing import Final, Optional, List, Union, Tuple
import uuid
import logging
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import json

import streamlit as st
import torch
import pandas as pd
from docx import Document
import chromadb
from llama_cpp import Llama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain.schema import Document as LangchainDocument
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredFileLoader, CSVLoader, UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings.base import Embeddings

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Single ThreadPoolExecutor for reuse
executor = ThreadPoolExecutor(max_workers=4)

# --------- 📁 Path Configuration ----------


class AppPaths:
    if platform.system() == "Windows":
        BASE_DIR: Final[Path] = Path(os.path.expanduser("~")) / "code" / "rag"
        MODELS_DIR: Final[Path] = Path(os.path.expanduser("~")) / "code" / "modelslist"
    else:
        BASE_DIR: Final[Path] = Path("/home/appadmin/code/Rag")
        MODELS_DIR: Final[Path] = Path("/home/appadmin/code/modelslist")

    TEMP_DIR = BASE_DIR / "temp"
    VECTORDB_DIR = BASE_DIR / "vectordb"
    FAISS_DIR = BASE_DIR / "faiss_index_store"
    STATE_DIR = BASE_DIR / "state"
    PROCESSED_DOCS_FILE = STATE_DIR / "processed_docs.json"
    FILE_HASHES_FILE = STATE_DIR / "processed_file_hashes.json"
    METADATA_FILE = STATE_DIR / "document_metadata.json"

    @classmethod
    def init_paths(cls):
        for d in [cls.TEMP_DIR, cls.VECTORDB_DIR, cls.FAISS_DIR, cls.STATE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

# Initialize paths and export MODELS_DIR
AppPaths.init_paths()
MODELS_DIR = AppPaths.MODELS_DIR
TEMP_DIR = AppPaths.TEMP_DIR
VECTORDB_DIR = AppPaths.VECTORDB_DIR
FAISS_DIR = AppPaths.FAISS_DIR

# Model definitions
MODELS = {
    "DeepSeek-Coder-1.3B": MODELS_DIR / "deepseek-1_3b-gguf/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
}

# Filter available models
AVAILABLE_MODELS = {name: path for name, path in MODELS.items() if os.path.exists(path)}
if not AVAILABLE_MODELS:
    st.error("No models found. Please verify model paths.")
    logger.error("No models found in MODELS_DIR")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------- 🛠️ Persistent State Management ----------
def save_session_state():
    """Save session state variables to disk."""
    try:
        with open(AppPaths.PROCESSED_DOCS_FILE, 'w') as f:
            json.dump(st.session_state.get("processed_docs", []), f)
        with open(AppPaths.FILE_HASHES_FILE, 'w') as f:
            json.dump(list(st.session_state.get("processed_file_hashes", set())), f)
        with open(AppPaths.METADATA_FILE, 'w') as f:
            json.dump(st.session_state.get("document_metadata", {}), f)
        logger.debug("Session state saved to disk")
    except Exception as e:
        logger.error(f"Failed to save session state: {e}")
        st.error(f"Failed to save session state: {e}")

def load_session_state():
    """Load session state variables from disk."""
    state = {}
    try:
        if AppPaths.PROCESSED_DOCS_FILE.exists():
            with open(AppPaths.PROCESSED_DOCS_FILE, 'r') as f:
                state["processed_docs"] = json.load(f)
        if AppPaths.FILE_HASHES_FILE.exists():
            with open(AppPaths.FILE_HASHES_FILE, 'r') as f:
                state["processed_file_hashes"] = set(json.load(f))
        if AppPaths.METADATA_FILE.exists():
            with open(AppPaths.METADATA_FILE, 'r') as f:
                state["document_metadata"] = json.load(f)
        logger.debug("Session state loaded from disk")
    except Exception as e:
        logger.error(f"Failed to load session state: {e}")
        st.error(f"Failed to load session state: {e}")
    return state

# --------- 🛠️ Initialize Session State ----------
def initialize_session_state():
    """Initialize critical session state variables for RAG, loading from disk if available."""
    persisted_state = load_session_state()
    defaults = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vector_store_manager": None,
        "processed_docs": persisted_state.get("processed_docs", []),
        "processed_file_hashes": persisted_state.get("processed_file_hashes", set()),
        "document_metadata": persisted_state.get("document_metadata", {}),
        "chunk_size": 512,
        "vector_store_type": "Chroma",
        "faiss_index_type": "Flat",
        "embedding_model": "all-MiniLM-L6-v2",
        "selected_model": "DeepSeek-Coder-1.3B",
        "max_tokens": 512,
        "temp": 0.7,
        "top_p": 0.95,
        "llm_model": None,
        "loaded_model_id": None,
        "uploaded_files": [],
        "retriever_k": 3,
        "sentence_transformer": None,
        "embedding_model_name": None,
        "selected_doc_ids": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            logger.debug(f"Initialized st.session_state.{key} = {value}")

# Initialize session state
initialize_session_state()

# --------- 🌟 Streamlit UI ----------
st.set_page_config(page_title="📚 RAG System", layout="wide")
st.title("📚 RAG Demo")

# Validate model paths
for model_name, model_path in MODELS.items():
    if not os.path.exists(model_path):
        st.warning(f"Model '{model_name}' not found at {model_path}. Excluded from selection.")
        logger.warning(f"Model '{model_name}' not found at {model_path}")

# --------- 🛠️ Custom Embedding Function ----------
class CustomSentenceTransformerEmbedding(Embeddings):
    def __init__(self, model_name: str):
        try:
            if ("sentence_transformer" not in st.session_state or 
                st.session_state.get("embedding_model_name") != model_name):
                logger.info(f"Initializing SentenceTransformer with model: {model_name} on CPU")
                self.model = SentenceTransformer(model_name, device="cpu")
                st.session_state.sentence_transformer = self.model
                st.session_state.embedding_model_name = model_name
            else:
                self.model = st.session_state.sentence_transformer
            self.set_device(st.session_state.get("device", "cpu"))
            logger.info(f"Model device: {next(self.model.parameters()).device}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            st.error(f"Failed to initialize SentenceTransformer: {e}")
            raise

    def set_device(self, device: str):
        if device in ["cpu", "cuda"]:
            logger.info(f"Moving model to {device}")
            self.model = self.model.to(device, non_blocking=True)
            st.session_state.sentence_transformer = self.model

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        try:
            logger.info(f"Computing embeddings for {len(texts)} documents")
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embeddings.extend(batch_embeddings.tolist())
            return embeddings
        except Exception as e:
            logger.error(f"Embedding computation error for documents: {e}")
            st.error(f"Embedding computation error for documents: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        try:
            logger.info(f"Computing embedding for query")
            embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Embedding computation error for query: {e}")
            st.error(f"Embedding computation error for query: {e}")
            return []

# --------- 🛠️ File Loader ----------
class AdvancedFileLoader:
    def __init__(self):
        self.temp_dir = str(TEMP_DIR)
        os.makedirs(self.temp_dir, exist_ok=True)

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def _load_excel(self, file_path: str) -> List[LangchainDocument]:
        try:
            df = await asyncio.get_event_loop().run_in_executor(executor, pd.read_excel, file_path)
            text = "\n".join([df.to_string(), f"\n\nMetadata: {str(df.dtypes)}"])
            doc_id = str(uuid.uuid4())
            doc = LangchainDocument(page_content=text, metadata={"source": file_path, "id": doc_id})
            st.session_state.document_metadata[doc_id] = {"source": file_path, "type": "excel"}
            return [doc]
        except Exception as e:
            st.error(f"Excel processing error {file_path}: {e}")
            logger.error(f"Excel processing error {file_path}: {e}")
            return []

    async def _load_powerpoint(self, file_path: str) -> List[LangchainDocument]:
        try:
            loader = UnstructuredPowerPointLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(executor, loader.load)
            for doc in documents:
                doc_id = str(uuid.uuid4())
                doc.metadata['id'] = doc_id
                st.session_state.document_metadata[doc_id] = {"source": file_path, "type": "powerpoint"}
            return documents
        except Exception as e:
            st.error(f"PowerPoint processing error {file_path}: {e}")
            logger.error(f"PowerPoint processing error {file_path}: {e}")
            return []

    async def load_file(self, file_path: str) -> List[LangchainDocument]:
        try:
            if not os.path.exists(file_path):
                st.error(f"File not found: {file_path}")
                logger.error(f"File not found: {file_path}")
                return []
            file_hash = self.compute_file_hash(file_path)
            if file_hash in st.session_state.get("processed_file_hashes", set()):
                logger.info(f"Skipping duplicate file: {file_path}")
                return []
            st.session_state.setdefault("processed_file_hashes", set()).add(file_hash)
            ext = Path(file_path).suffix.lower()
            loader_map = {
                '.pdf': lambda fp: PyPDFLoader(fp).load(),
                '.txt': lambda fp: TextLoader(fp).load(),
                '.doc': lambda fp: UnstructuredWordDocumentLoader(fp).load(),
                '.docx': lambda fp: UnstructuredWordDocumentLoader(fp).load(),
                '.md': lambda fp: UnstructuredMarkdownLoader(fp).load(),
                '.csv': lambda fp: CSVLoader(fp).load(),
                '.xlsx': self._load_excel,
                '.xls': self._load_excel,
                '.pptx': self._load_powerpoint,
                '.ppt': self._load_powerpoint,
                'default': lambda fp: UnstructuredFileLoader(fp).load()
            }
            loader = loader_map.get(ext, loader_map['default'])
            if ext in ['.xlsx', '.xls', '.pptx', '.ppt']:
                documents = await loader(file_path)
            else:
                documents = await asyncio.get_event_loop().run_in_executor(executor, loader, file_path)
            for doc in documents:
                doc_id = str(uuid.uuid4())
                doc.metadata['id'] = doc_id
                st.session_state.document_metadata[doc_id] = {
                    "source": file_path,
                    "type": ext.lstrip('.')
                }
            return documents
        except Exception as e:
            st.error(f"Failed to load {file_path}: {e}")
            logger.error(f"Failed to load {file_path}: {e}")
            return []

# --------- 📚 Vector Store Manager ----------
class VectorStoreManager:
    def __init__(self, embedding_model_name: str, faiss_index_type: str, chunk_size: int = 512):
        logger.info(f"Creating VectorStoreManager with embedding model: {embedding_model_name}, FAISS index: {faiss_index_type}")
        if chunk_size < 256 or chunk_size > 1024:
            st.warning(f"Chunk size {chunk_size} is extreme. Recommended range: 256–1024 characters.")
        self.embedding_fn = CustomSentenceTransformerEmbedding(model_name=embedding_model_name)
        self.faiss_index_type = faiss_index_type
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            length_function=len
        )
        self.chroma_client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
        self.faiss_store = None
        self.chroma_store = None
        self.retriever = None

    async def create_vector_stores(self, paperwork: List[LangchainDocument]):
        try:
            if not paperwork:
                st.error("No valid documents to process")
                logger.error("No valid documents to process")
                return False

            logger.info(f"Splitting {len(paperwork)} documents into chunks")
            chunk_hashes = set()
            chunks = []
            chunk_ids = []
            for doc in paperwork:
                if doc.page_content.strip():
                    split_docs = self.text_splitter.split_documents([doc])
                    for split_doc in split_docs:
                        content_hash = hashlib.sha256(split_doc.page_content.encode()).hexdigest()
                        if content_hash not in chunk_hashes:
                            chunk_hashes.add(content_hash)
                            chunk_id = str(uuid.uuid4())
                            split_doc.metadata['id'] = chunk_id
                            split_doc.metadata['parent_doc_id'] = doc.metadata['id']
                            chunks.append(split_doc)
                            chunk_ids.append(chunk_id)
            if not chunks:
                st.error("No valid document chunks after splitting")
                logger.error("No valid document chunks after splitting")
                return False
            logger.info(f"Created {len(chunks)} unique document chunks")

            # Embed documents
            texts = [chunk.page_content for chunk in chunks]
            embeddings = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: self.embedding_fn.embed_documents(texts)
            )
            embeddings = np.array(embeddings, dtype=np.float32)

            # Initialize FAISS index
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)

            # Add embeddings to FAISS index
            faiss_index.add(embeddings)

            # Initialize FAISS vector store manually
            self.faiss_store = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: FAISS(
                    embedding_function=self.embedding_fn,
                    index=faiss_index,
                    docstore={chunk.metadata['id']: chunk for chunk in chunks},
                    index_to_docstore_id={i: chunk.metadata['id'] for i, chunk in enumerate(chunks)}
                )
            )
            await asyncio.get_event_loop().run_in_executor(executor, self.faiss_store.save_local, str(FAISS_DIR))

            # Initialize Chroma store
            self.chroma_store = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: Chroma(
                    collection_name="document_collection",
                    embedding_function=self.embedding_fn,
                    client=self.chroma_client,
                    persist_directory=str(VECTORDB_DIR)
                )
            )
            existing_ids = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: self.chroma_store.get()['ids']
            )
            chunks_to_add = [(chunk, chunk_id) for chunk, chunk_id in zip(chunks, chunk_ids) if chunk_id not in existing_ids]
            if chunks_to_add:
                new_chunks, new_chunk_ids = zip(*chunks_to_add)
                await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: self.chroma_store.add_documents(
                        documents=new_chunks,
                        ids=new_chunk_ids
                    )
                )

            self.retriever = self.chroma_store.as_retriever(search_kwargs={"k": st.session_state.retriever_k})

            st.toast(f"Vector stores created with Flat FAISS index and all-MiniLM-L6-v2 embeddings")
            logger.info(f"Vector stores created with Flat FAISS index")
            return True
        except Exception as e:
            st.error(f"Error creating vector stores: {e}")
            logger.error(f"Error creating vector stores: {e}")
            return False

    async def query(self, query_text: str, doc_ids: Optional[List[str]] = None) -> List[LangchainDocument]:
        try:
            if not self.retriever:
                st.error("Retriever not initialized")
                logger.error("Retriever not initialized")
                return []
            logger.info(f"Retrieving documents for query: {query_text} using Chroma")
            if doc_ids:
                search_kwargs = {
                    "k": st.session_state.retriever_k,
                    "filter": {"parent_doc_id": {"$in": doc_ids}}
                }
                retriever = self.chroma_store.as_retriever(search_kwargs=search_kwargs)
            else:
                retriever = self.retriever
            docs = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: retriever.get_relevant_documents(query_text)
            )
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            st.error(f"Error querying vector store: {e}")
            logger.error(f"Error querying vector store: {e}")
            return []

# --------- 🤖 Model Loading & Generation ----------
async def load_model():
    model_name = "DeepSeek-Coder-1.3B"
    logger.debug(f"Loading model: {model_name}, Device: {st.session_state.get('device', 'cpu')}")
    
    if st.session_state.llm_model is not None and st.session_state.loaded_model_id == model_name:
        logger.debug(f"Model {model_name} already loaded")
        return st.session_state.llm_model
    
    with st.spinner(f"Loading '{model_name}' (this may take a while)..."):
        try:
            model_path = str(AVAILABLE_MODELS[model_name])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            model = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=40 if st.session_state.get("device", "cpu") == "cuda" else 0
                )
            )
            st.session_state.llm_model = model
            st.session_state.loaded_model_id = model_name
            st.toast(f"Loaded: {model_name}")
            logger.info(f"Loaded {model_name}")
            return model
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            logger.error(f"Failed to load model: {str(e)}")
            return None

async def generate_response(prompt: str, context: str = "") -> str:
    model = await load_model()
    if model is None:
        return "Error: Model not loaded"

    system_prompt = """You are a highly knowledgeable and helpful AI assistant. Your task is to provide accurate and concise answers to the user's question based on the provided context from documents. Follow these guidelines:
    - Use the provided context as the primary source to answer the question.
    - If the context does not fully address the question, indicate that the context was insufficient and provide a general answer based on your knowledge.
    - Keep the response clear, relevant, and focused on the question.
    - Avoid unnecessary elaboration unless requested.

    Context:
    {context}

    Question: {question}
    Answer:"""

    formatted_system_prompt = system_prompt.format(
        context=context,
        question=prompt
    )

    messages = [{"role": "system", "content": formatted_system_prompt}, {"role": "user", "content": prompt}]

    try:
        max_tokens = st.session_state.get("max_tokens", 512)
        temp = st.session_state.get("temp", 0.7)
        top_p = st.session_state.get("top_p", 0.95)
        logger.debug(f"Generating response with max_tokens={max_tokens}, temp={temp}, top_p={top_p}")

        response = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p
            )["choices"][0]["message"]["content"]
        )
        return response.strip()
    except Exception as e:
        st.error(f"Generation error: {e}")
        logger.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)}"

async def rag_query(question: str, doc_ids: Optional[List[str]] = None) -> Tuple[str, List[LangchainDocument]]:
    if st.session_state.vector_store_manager is None:
        return "Vector store not initialized. Please initialize and process documents first.", []
    docs = await st.session_state.vector_store_manager.query(
        question,
        doc_ids=doc_ids
    )
    if not docs:
        return "No relevant documents found.", []
    context = "\n\n".join([
        f"Document {idx+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
        for idx, doc in enumerate(docs)
    ])
    return await generate_response(question, context), docs

# --------- 🖥️ Main UI Components ----------
def run_async(coro):
    """
    Run an async coroutine in a thread-safe manner, compatible with Streamlit.
    """
    def run_in_thread(coro):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    if threading.current_thread() is threading.main_thread() or not hasattr(asyncio, '_get_running_loop'):
        return run_in_thread(coro)
    else:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return run_in_thread(coro)

# Initialize vector store automatically
if st.session_state.vector_store_manager is None:
    try:
        st.session_state.vector_store_manager = VectorStoreManager(
            embedding_model_name=EMBEDDING_MODEL,
            faiss_index_type="Flat",
            chunk_size=st.session_state.chunk_size
        )
        st.toast("Vector store initialized!")
        logger.info("Vector store initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize vector store: {e}")
        logger.error(f"Failed to initialize vector store: {e}")

# Automatically load the model
if st.session_state.llm_model is None:
    try:
        run_async(load_model())
    except Exception as e:
        st.error(f"Failed to auto-load model: {str(e)}")
        logger.error(f"Failed to auto-load model: {str(e)}")

# Main area: Document Upload
st.header("Document Upload")
st.markdown("Select one or more files to upload. Supported formats: PDF, DOCX, CSV, XLSX, TXT, PPTX.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "csv", "xlsx", "txt", "pptx"],
    key="file_uploader",
    accept_multiple_files=True,
    help="Hold Ctrl (Windows) or Cmd (Mac) to select multiple files."
)
st.session_state.uploaded_files = uploaded_files or []

# Process button
if st.button("Process Documents") and st.session_state.uploaded_files:
    async def process_uploaded():
        loader = AdvancedFileLoader()
        documents = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(st.session_state.uploaded_files)

        for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
            status_text.text(f"Processing file {idx + 1}/{total_files}: {uploaded_file.name}")
            temp_path = TEMP_DIR / uploaded_file.name
            try:
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(uploaded_file.getbuffer())
                docs = await loader.load_file(str(temp_path))
                if docs:
                    documents.extend(docs)
                    if uploaded_file.name not in st.session_state.processed_docs:
                        st.session_state.processed_docs.append(uploaded_file.name)
                else:
                    st.warning(f"Skipped {uploaded_file.name}: No new content or duplicate.")
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                logger.error(f"Failed to process {uploaded_file.name}: {str(e)}")
            progress_bar.progress((idx + 1) / total_files)

        status_text.text("Finalizing vector store...")
        if documents:
            if await st.session_state.vector_store_manager.create_vector_stores(documents):
                st.toast(f"Processed {len(documents)} documents into vector store.")
                logger.info(f"Processed {len(documents)} documents into chunks")
                save_session_state()  # Save session state after processing
            else:
                st.error("Failed to create vector stores")
                logger.error("Failed to create vector stores")
        else:
            st.warning("No new content extracted from uploaded files.")
            logger.warning("No new content extracted")
        status_text.empty()
        progress_bar.empty()

    run_async(process_uploaded())

# Main RAG interface
st.header("Query Documents")
if st.session_state.vector_store_manager is None:
    st.warning("Vector store initialization failed")
elif not st.session_state.processed_docs:
    st.warning("Please upload and process a document")
else:
    st.toast(f"RAG system ready! Using chunk size: {st.session_state.chunk_size} characters")
    # Dropdown for selecting multiple documents
    doc_options = st.session_state.processed_docs
    doc_id_map = {}
    for doc_name in st.session_state.processed_docs:
        for doc_id, meta in st.session_state.document_metadata.items():
            if meta['source'].endswith(doc_name):
                doc_id_map[doc_name] = doc_id
                break
    selected_docs = st.multiselect(
        "Select Documents to Query",
        options=doc_options,
        key="doc_selector",
        help="Select one or more documents to query. Leave empty to query all documents."
    )
    st.session_state.selected_doc_ids = [doc_id_map[doc] for doc in selected_docs] if selected_docs else []
    question = st.text_input("Ask about your documents:")
    if st.button("Get Answer") and question:
        with st.spinner("Searching documents..."):
            response, docs = run_async(rag_query(question, st.session_state.selected_doc_ids))
        st.subheader("Answer")
        st.write(response)
        if docs:
            with st.expander("View document sources"):
                st.write("### Retrieved Documents")
                for idx, doc in enumerate(docs):
                    st.write(f"#### Document {idx+1}")
                    st.json(doc.metadata)
                    st.write("Content preview:")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                    st.divider()