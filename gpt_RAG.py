import re
import faiss
import torch
import streamlit as st
import PyPDF2
from datetime import datetime
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
hf_token = st.secrets['HF_TOKEN']
login(token = hf_token)


class EnhancedRAGAgent:
    def __init__(self, 
                 llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model

        self.embedding_model = None
        self.tokenizer = None
        self.llm_model = None

        self.documents = []
        self.document_chunks = []
        self.chunk_metadata = []

        self.faiss_index = None
        self.embedding_dim = None

    def load_embedding_model(self):
        if not hasattr(self, '_embedding_model_cache'):
            self._embedding_model_cache = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model_cache

    def load_llm_model(self):
        if not hasattr(self, '_llm_model_cache'):
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._llm_model_cache = (tokenizer, model)
        return self._llm_model_cache

    def _ensure_models_loaded(self):
        if self.embedding_model is None:
            self.embedding_model = self.load_embedding_model()
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        if self.tokenizer is None or self.llm_model is None:
            self.tokenizer, self.llm_model = self.load_llm_model()

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            if hasattr(pdf_file, 'read'):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            else:
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
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
        try:
            self._ensure_models_loaded()
            text = self.extract_text_from_pdf(pdf_file)
            chunks = self.chunk_text(text)

            if not chunks:
                return False

            doc_info = {
                'filename': filename or 'uploaded_document.pdf',
                'text': text,
                'chunk_count': len(chunks),
                'added_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.documents.append(doc_info)
            start_idx = len(self.document_chunks)
            self.document_chunks.extend(chunks)

            for i, chunk in enumerate(chunks):
                metadata = {
                    'doc_index': len(self.documents) - 1,
                    'chunk_index': i,
                    'filename': doc_info['filename'],
                    'chunk_start_idx': start_idx + i
                }
                self.chunk_metadata.append(metadata)

            self._update_faiss_index(chunks)
            return True

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return False

    def _update_faiss_index(self, new_chunks: List[str]):
        self._ensure_models_loaded()
        embeddings = self.embedding_model.encode(new_chunks).astype('float32')
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        if not self.document_chunks or self.faiss_index is None:
            return []
        self._ensure_models_loaded()
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.faiss_index.search(query_embedding, top_k)

        relevant_chunks = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.document_chunks) and similarity > 0.1:
                chunk_text = self.document_chunks[idx]
                metadata = self.chunk_metadata[idx]
                relevant_chunks.append((chunk_text, float(similarity), metadata))
        return relevant_chunks

    def generate_response(self, query: str, context: str) -> str:
        self._ensure_models_loaded()
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context. Use the context to answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def chat(self, query: str) -> Tuple[str, List[Tuple[str, float, Dict]]]:
        if not self.document_chunks:
            return "No documents have been added to the knowledge base. Please upload PDF documents first.", []
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=3)
        if not relevant_chunks:
            return "I couldn't find relevant information in the loaded documents to answer your question.", []
        context = "\n\n".join([chunk for chunk, _, _ in relevant_chunks])
        response = self.generate_response(query, context)
        return response, relevant_chunks


def main():
    st.set_page_config(page_title="RAG Agent with Tiny Llama", page_icon="🐒", layout="wide")
    st.title("🐒 RAG Agent with Tiny Llama")
    st.markdown("Upload PDF documents and chat with them using advanced AI!")

    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = EnhancedRAGAgent()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("📚 Document Management")
        uploaded_files = st.file_uploader("Upload PDF documents", type=['pdf'], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success = st.session_state.rag_agent.add_pdf_document(uploaded_file, uploaded_file.name)
                        if success:
                            st.success(f"✅ Added {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")

        st.subheader("📄 Loaded Documents")
        for doc in st.session_state.rag_agent.documents:
            with st.expander(f"{doc['filename']}"):
                st.write(f"**Chunks:** {doc['chunk_count']}")
                st.write(f"**Added:** {doc['added_at']}")
                st.write(f"**Text length:** {len(doc['text'])} characters")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")
        chat_container = st.container()
        with chat_container:
            for user_msg, ai_msg, sources in st.session_state.chat_history:
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

        if query := st.chat_input("Ask a question about your documents..."):
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
            st.session_state.chat_history.append((query, response, sources))

    with col2:
        st.header("⚙️ Settings")
        with st.expander("🔧 Model Information"):
            st.write("**LLM Model:** meta-llama/Llama-3.2-3B-Instruct")
            st.write("**Embedding Model:** all-MiniLM-L6-v2")
            st.write("**Vector Database:** FAISS")
            st.write("**Similarity:** Cosine Similarity")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        if st.session_state.rag_agent.documents:
            st.subheader("📊 Statistics")
            total_chunks = sum(doc['chunk_count'] for doc in st.session_state.rag_agent.documents)
            st.metric("Total Documents", len(st.session_state.rag_agent.documents))
            st.metric("Total Chunks", total_chunks)
            st.metric("Chat Messages", len(st.session_state.chat_history))


if __name__ == "__main__":
    main()
