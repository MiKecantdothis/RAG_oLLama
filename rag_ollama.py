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
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_store = None
            self.document_chunks = []

            # Load TinyLlama with error handling
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
                
            self.model.eval()
            self.device = device
            
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            raise e

    def generate_response(self, query: str, context: List[str]) -> str:
        try:
            context_text = "\n".join(context[:3])  # Limit context to avoid token limit
            
            # Create a more structured prompt
            prompt = f"""<|system|>
You are a helpful assistant. Answer questions based only on the provided context. If the answer cannot be found in the context, say "I cannot find this information in the provided document."
<|end|>
<|user|>
Context: {context_text}

Question: {query}
<|end|>
<|assistant|>"""

            # Tokenize with proper padding and truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024,
                truncation=True,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced for better performance
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
                
            # Decode response
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract assistant response
            if "<|assistant|>" in decoded:
                response = decoded.split("<|assistant|>")[-1].strip()
            else:
                response = decoded.strip()
                
            # Clean up the response
            response = re.sub(r'^[\s\n]*', '', response)  # Remove leading whitespace
            response = re.sub(r'[\s\n]*$', '', response)  # Remove trailing whitespace
            
            return response if response else "I couldn't generate a proper response."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
                except Exception as e:
                    st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                    continue
                    
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
                
            return text
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            raise e

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        try:
            # Clean the text first
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
            text = text.strip()
            
            # Split by sentences first for better chunking
            sentences = re.split(r'[.!?]+', text)
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence would exceed chunk size, save current chunk
                if len(current_chunk.split()) + len(sentence.split()) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Keep some overlap
                        words = current_chunk.split()
                        if len(words) > overlap:
                            current_chunk = ' '.join(words[-overlap:]) + ' ' + sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += ' ' + sentence if current_chunk else sentence
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Filter out very short chunks
            chunks = [chunk for chunk in chunks if len(chunk.split()) >= 10]
            
            return chunks
            
        except Exception as e:
            st.error(f"Error chunking text: {str(e)}")
            return []

    def create_vector_store(self, chunks: List[str]):
        """Create FAISS vector store from text chunks"""
        try:
            if not chunks:
                raise ValueError("No chunks provided for vector store creation")
                
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                chunks, 
                show_progress_bar=True,
                batch_size=32
            )

            # Create FAISS index - using cosine similarity
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            # Add to index
            self.vector_store.add(embeddings_normalized.astype('float32'))
            self.document_chunks = chunks
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            raise e

    def search_similar_chunks(self, query: str, k: int = 3) -> List[str]:
        """Search for similar chunks in the vector store"""
        try:
            if self.vector_store is None or not self.document_chunks:
                return []

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.vector_store.search(
                query_embedding.astype('float32'), 
                min(k, len(self.document_chunks))
            )

            relevant_chunks = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(self.document_chunks) and score > 0.1:  # Threshold for relevance
                    relevant_chunks.append(self.document_chunks[idx])

            return relevant_chunks
            
        except Exception as e:
            st.error(f"Error searching chunks: {str(e)}")
            return []

    
def main():
    st.set_page_config(
        page_title="RAG PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG PDF Chatbot with TinyLlama")
    st.markdown("Upload a PDF document and chat with it using a local language model!")

    # Initialize chatbot with error handling
    try:
        if 'chatbot' not in st.session_state:
            with st.spinner("Loading language model... This may take a few minutes on first run."):
                st.session_state.chatbot = PDFChatbot()
            st.success("✅ Language model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load language model: {str(e)}")
        st.stop()

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )

        if uploaded_file is not None:
            st.info(f"📎 File: {uploaded_file.name}")
            
            if st.button("Process Document", type="primary"):
                try:
                    with st.spinner("Processing PDF..."):
                        # Extract text
                        text = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)
                        
                        if not text.strip():
                            st.error("No text found in the PDF. Please upload a different file.")
                            return

                        # Chunk text
                        chunks = st.session_state.chatbot.chunk_text(text)
                        
                        if not chunks:
                            st.error("Could not create text chunks. Please try a different PDF.")
                            return

                        # Create vector store
                        st.session_state.chatbot.create_vector_store(chunks)

                        # Store in session state
                        st.session_state.vector_store = st.session_state.chatbot.vector_store
                        st.session_state.document_chunks = st.session_state.chatbot.document_chunks

                        st.success(f"✅ Document processed! Created {len(chunks)} chunks.")
                        
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

        # Display document info
        if st.session_state.vector_store is not None:
            st.success("📊 Document Ready")
            st.info(f"Chunks: {len(st.session_state.document_chunks)}")

        st.markdown("---")
        st.markdown("### 🛠️ System Info")
        st.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        st.markdown("### 📋 Requirements")
        st.markdown("""
        Install required packages:
        ```bash
        pip install streamlit PyPDF2 faiss-cpu sentence-transformers transformers torch numpy
        ```
        """)

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    if st.session_state.vector_store is None:
        st.info("👆 Please upload and process a PDF document to start chatting!")
        return

    # Display chat history
    st.subheader("💬 Chat with your document")
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        for i, (query, response) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                st.write(response)

        # Chat input
        query = st.chat_input("Ask a question about your document...")

        if query:
            # Add user message to chat history immediately
            with st.chat_message("user"):
                st.write(query)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document and generating response..."):
                    try:
                        # Update chatbot's vector store and chunks
                        st.session_state.chatbot.vector_store = st.session_state.vector_store
                        st.session_state.chatbot.document_chunks = st.session_state.document_chunks

                        # Search for relevant chunks
                        relevant_chunks = st.session_state.chatbot.search_similar_chunks(query, k=3)
                        
                        if not relevant_chunks:
                            response = "I couldn't find relevant information in the document to answer your question."
                        else:
                            # Generate response
                            response = st.session_state.chatbot.generate_response(query, relevant_chunks)

                        st.write(response)

                        # Add to chat history
                        st.session_state.chat_history.append((query, response))
                        
                    except Exception as e:
                        error_response = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_response)
                        st.session_state.chat_history.append((query, error_response))

            st.rerun()
    
    with col2:
        if st.session_state.document_chunks:
            st.subheader("📊 Document Stats")
            st.metric("Total Chunks", len(st.session_state.document_chunks))
            st.metric("Chat Messages", len(st.session_state.chat_history))


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        st.error(f"""
        Missing required package: {e}

        Please install the required packages:
        ```bash
        pip install streamlit PyPDF2 faiss-cpu sentence-transformers transformers torch numpy
        ```
        """)
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()
