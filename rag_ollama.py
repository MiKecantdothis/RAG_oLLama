import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


class PDFChatbot:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = None
        self.document_chunks = []

        # Load TinyLlama
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.model.eval()

    def generate_response(self, query: str, context: list[str]) -> str:
        context_text = "\n".join(context)
        prompt = f"""<|system|>You are a helpful assistant.<|end|>
          <|user|>Based on the following document context, answer the question.
           Context: {context_text} 
           Question: {query}
           <|end|> 
           <|assistant|>"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|assistant|>" in decoded:
            decoded = decoded.split("<|assistant|>")[-1].strip()
        return decoded

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())

        return chunks

    def create_vector_store(self, chunks: List[str]):
        """Create FAISS vector store from text chunks"""
        embeddings = self.embedding_model.encode(chunks)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings.astype('float32'))
        self.document_chunks = chunks

    def search_similar_chunks(self, query: str, k: int = 3) -> List[str]:
        """Search for similar chunks in the vector store"""
        if self.vector_store is None:
            return []

        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)

        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.document_chunks):
                relevant_chunks.append(self.document_chunks[idx])

        return relevant_chunks

    
def main():
    st.title("📚 RAG PDF Chatbot with Llama")
    st.markdown("Upload a PDF document and chat with it using local Llama model!")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )

        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    text = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)

                    # Chunk text
                    chunks = st.session_state.chatbot.chunk_text(text)

                    # Create vector store
                    st.session_state.chatbot.create_vector_store(chunks)

                    # Store in session state
                    st.session_state.vector_store = st.session_state.chatbot.vector_store
                    st.session_state.document_chunks = st.session_state.chatbot.document_chunks

                    st.success(f"✅ Document processed! Created {len(chunks)} chunks.")

        # Display document info
        if st.session_state.vector_store is not None:
            st.success("📊 Document Ready")
            st.info(f"Chunks: {len(st.session_state.document_chunks)}")

        st.markdown("---")
        st.markdown("### 🛠️ Requirements")
        st.markdown("""
        Make sure you have:
        1. **Ollama** installed and running
        2. **Llama model** downloaded (e.g., `ollama pull llama3.2`)
        3. All required Python packages installed
        """)

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    if st.session_state.vector_store is None:
        st.info("👆 Please upload and process a PDF document to start chatting!")
        return

    # Display chat history
    st.subheader("💬 Chat History")
    chat_container = st.container()

    with chat_container:
        for i, (query, response) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                st.write(response)

    # Chat input
    query = st.chat_input("Ask a question about your document...")

    if query:
        # Add user message to chat history
        with st.chat_message("user"):
            st.write(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Update chatbot's vector store and chunks
                st.session_state.chatbot.vector_store = st.session_state.vector_store
                st.session_state.chatbot.document_chunks = st.session_state.document_chunks

                # Search for relevant chunks
                relevant_chunks = st.session_state.chatbot.search_similar_chunks(query)

                # Generate response
                response = st.session_state.chatbot.generate_response(query, relevant_chunks)

                st.write(response)

                # Add to chat history
                st.session_state.chat_history.append((query, response))

        st.rerun()

if __name__ == "__main__":
    # Check for required packages
    try:
        main()
    except ImportError as e:
        st.error(f"""
        Missing required package: {e}

        Please install the required packages:
        ```bash
        pip install streamlit PyPDF2 faiss-cpu sentence-transformers ollama numpy
        ```
        """)

