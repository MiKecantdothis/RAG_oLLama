import os
import PyPDF2
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import pickle
from typing import List, Dict, Tuple
import tempfile
from datetime import datetime
from huggingface_hub import login

hf_token = os.getenv('HF_TOKEN')
login(token = hf_token)

class EnhancedRAGAgent:
    def __init__(self, 
                 llm_model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the enhanced RAG agent with sentence transformers and FAISS.
        
        Args:
            llm_model: HuggingFace model identifier for Llama 3.2
            embedding_model: Sentence transformer model for embeddings
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        
        # Initialize models (will be loaded lazily)
        self.embedding_model = None
        self.tokenizer = None
        self.llm_model = None
        
        # Document storage
        self.documents = []
        self.document_chunks = []
        self.chunk_metadata = []
        
        # FAISS index
        self.faiss_index = None
        self.embedding_dim = None
        
    @st.cache_resource
    def load_embedding_model(_self):
        """Load sentence transformer model (cached)."""
        return SentenceTransformer(_self.embedding_model_name)
    
    @st.cache_resource
    def load_llm_model(_self):
        """Load Llama model and tokenizer (cached)."""
        tokenizer = AutoTokenizer.from_pretrained(_self.llm_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            _self.llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer, model
    
    def _ensure_models_loaded(self):
        """Ensure all models are loaded."""
        if self.embedding_model is None:
            self.embedding_model = self.load_embedding_model()
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
        if self.tokenizer is None or self.llm_model is None:
            self.tokenizer, self.llm_model = self.load_llm_model()
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_file: File object or path to PDF
            
        Returns:
            Extracted text as string
        """
        try:
            if hasattr(pdf_file, 'read'):
                # It's a file object
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            else:
                # It's a file path
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size - 100:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + chunk_size - overlap, end)
        
        return chunks
    
    def add_pdf_document(self, pdf_file, filename: str = None) -> bool:
        """
        Add a PDF document to the knowledge base.
        
        Args:
            pdf_file: PDF file object or path
            filename: Optional filename for display
            
        Returns:
            Success status
        """
        try:
            self._ensure_models_loaded()
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            if not chunks:
                return False
            
            # Store document info
            doc_info = {
                'filename': filename or 'uploaded_document.pdf',
                'text': text,
                'chunk_count': len(chunks),
                'added_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.documents.append(doc_info)
            
            # Add chunks with metadata
            start_idx = len(self.document_chunks)
            self.document_chunks.extend(chunks)
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                metadata = {
                    'doc_index': len(self.documents) - 1,
                    'chunk_index': i,
                    'filename': doc_info['filename'],
                    'chunk_start_idx': start_idx + i
                }
                self.chunk_metadata.append(metadata)
            
            # Update FAISS index
            self._update_faiss_index(chunks)
            
            return True
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return False
    
    def _update_faiss_index(self, new_chunks: List[str]):
        """Update FAISS index with new chunks."""
        self._ensure_models_loaded()
        
        # Generate embeddings for new chunks
        embeddings = self.embedding_model.encode(new_chunks)
        embeddings = embeddings.astype('float32')
        
        if self.faiss_index is None:
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.faiss_index.add(embeddings)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve the most relevant document chunks for a query using FAISS.
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
        if not self.document_chunks or self.faiss_index is None:
            return []
        
        self._ensure_models_loaded()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, top_k)
        
        relevant_chunks = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.document_chunks) and similarity > 0.1:  # Similarity threshold
                chunk_text = self.document_chunks[idx]
                metadata = self.chunk_metadata[idx]
                relevant_chunks.append((chunk_text, float(similarity), metadata))
        
        return relevant_chunks
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using Llama 3.2 with the given context.
        
        Args:
            query: User query
            context: Retrieved context from documents
            
        Returns:
            Generated response
        """
        self._ensure_models_loaded()
        
        # Create prompt with context
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context. Use the context to answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def chat(self, query: str) -> Tuple[str, List[Tuple[str, float, Dict]]]:
        """
        Main chat function that retrieves context and generates response.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (AI response, relevant chunks with metadata)
        """
        if not self.document_chunks:
            return "No documents have been added to the knowledge base. Please upload PDF documents first.", []
        
        # Retrieve relevant context
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=3)
        
        if not relevant_chunks:
            return "I couldn't find relevant information in the loaded documents to answer your question.", []
        
        # Combine context from retrieved chunks
        context = "\n\n".join([chunk for chunk, _, _ in relevant_chunks])
        
        # Generate response
        response = self.generate_response(query, context)
        
        return response, relevant_chunks


# Streamlit UI
def main():
    st.set_page_config(
        page_title="RAG Agent with Llama 3.2",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Agent with Llama 3.2")
    st.markdown("Upload PDF documents and chat with them using advanced AI!")
    
    # Initialize session state
    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = EnhancedRAGAgent()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to add to the knowledge base"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success = st.session_state.rag_agent.add_pdf_document(
                            uploaded_file, uploaded_file.name
                        )
                        if success:
                            st.success(f"✅ Added {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
        
        # Document info
        st.subheader("📄 Loaded Documents")
        docs = st.session_state.rag_agent.documents
        if docs:
            for i, doc in enumerate(docs):
                with st.expander(f"{doc['filename']}"):
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    st.write(f"**Added:** {doc['added_at']}")
                    st.write(f"**Text length:** {len(doc['text'])} characters")
        else:
            st.info("No documents loaded yet.")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, ai_msg, sources) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(user_msg)
                
                with st.chat_message("assistant"):
                    st.write(ai_msg)
                    
                    if sources:
                        with st.expander("📖 Sources"):
                            for j, (chunk, score, metadata) in enumerate(sources):
                                st.write(f"**Source {j+1}** (Score: {score:.3f}) - {metadata['filename']}")
                                st.write(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                                st.divider()
        
        # Chat input
        if query := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.rag_agent.documents:
                st.error("Please upload and process PDF documents first!")
            else:
                with st.chat_message("user"):
                    st.write(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response, sources = st.session_state.rag_agent.chat(query)
                    
                    st.write(response)
                    
                    if sources:
                        with st.expander("📖 Sources"):
                            for j, (chunk, score, metadata) in enumerate(sources):
                                st.write(f"**Source {j+1}** (Score: {score:.3f}) - {metadata['filename']}")
                                st.write(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                                st.divider()
                
                # Add to chat history
                st.session_state.chat_history.append((query, response, sources))
    
    with col2:
        st.header("⚙️ Settings")
        
        # Model info
        with st.expander("🔧 Model Information"):
            st.write("**LLM Model:** meta-llama/Llama-3.2-3B-Instruct")
            st.write("**Embedding Model:** all-MiniLM-L6-v2")
            st.write("**Vector Database:** FAISS")
            st.write("**Similarity:** Cosine Similarity")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Statistics
        if st.session_state.rag_agent.documents:
            st.subheader("📊 Statistics")
            total_chunks = sum(doc['chunk_count'] for doc in st.session_state.rag_agent.documents)
            st.metric("Total Documents", len(st.session_state.rag_agent.documents))
            st.metric("Total Chunks", total_chunks)
            st.metric("Chat Messages", len(st.session_state.chat_history))


if __name__ == "__main__":
    st.markdown("""
    ### 🚀 Setup Instructions
    
    Make sure to install the required packages:
    ```bash
    pip install streamlit torch transformers sentence-transformers faiss-cpu PyPDF2 numpy
    ```
    
    For GPU support, install `faiss-gpu` instead of `faiss-cpu`.
    
    You may need to login to HuggingFace Hub:
    ```bash
    huggingface-cli login
    ```
    
    Run the app with:
    ```bash
    streamlit run app.py
    ```
    """)
    
    main()
