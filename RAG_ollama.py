{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0cda0bdc-5b2b-4799-a971-a09ee94a2324",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting PyPDF2\n",
      "  Downloading pypdf2-3.0.1-py3-none-any.whl.metadata (6.8 kB)\n",
      "Downloading pypdf2-3.0.1-py3-none-any.whl (232 kB)\n",
      "Installing collected packages: PyPDF2\n",
      "Successfully installed PyPDF2-3.0.1\n"
     ]
    }
   ],
   "source": [
    "!pip install PyPDF2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6ca416b4-19f1-487b-a1f6-22029c6b18cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting ollama\n",
      "  Downloading ollama-0.5.1-py3-none-any.whl.metadata (4.3 kB)\n",
      "Requirement already satisfied: httpx>=0.27 in c:\\users\\mike\\anaconda3\\lib\\site-packages (from ollama) (0.28.1)\n",
      "Requirement already satisfied: pydantic>=2.9 in c:\\users\\mike\\anaconda3\\lib\\site-packages (from ollama) (2.11.7)\n",
      "Requirement already satisfied: anyio in c:\\users\\mike\\anaconda3\\lib\\site-packages (from httpx>=0.27->ollama) (4.9.0)\n",
      "Requirement already satisfied: certifi in c:\\users\\mike\\anaconda3\\lib\\site-packages (from httpx>=0.27->ollama) (2025.6.15)\n",
      "Requirement already satisfied: httpcore==1.* in c:\\users\\mike\\anaconda3\\lib\\site-packages (from httpx>=0.27->ollama) (1.0.2)\n",
      "Requirement already satisfied: idna in c:\\users\\mike\\anaconda3\\lib\\site-packages (from httpx>=0.27->ollama) (3.7)\n",
      "Requirement already satisfied: h11<0.15,>=0.13 in c:\\users\\mike\\anaconda3\\lib\\site-packages (from httpcore==1.*->httpx>=0.27->ollama) (0.14.0)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in c:\\users\\mike\\anaconda3\\lib\\site-packages (from pydantic>=2.9->ollama) (0.6.0)\n",
      "Requirement already satisfied: pydantic-core==2.33.2 in c:\\users\\mike\\anaconda3\\lib\\site-packages (from pydantic>=2.9->ollama) (2.33.2)\n",
      "Requirement already satisfied: typing-extensions>=4.12.2 in c:\\users\\mike\\anaconda3\\lib\\site-packages (from pydantic>=2.9->ollama) (4.14.1)\n",
      "Requirement already satisfied: typing-inspection>=0.4.0 in c:\\users\\mike\\anaconda3\\lib\\site-packages (from pydantic>=2.9->ollama) (0.4.1)\n",
      "Requirement already satisfied: sniffio>=1.1 in c:\\users\\mike\\anaconda3\\lib\\site-packages (from anyio->httpx>=0.27->ollama) (1.3.0)\n",
      "Downloading ollama-0.5.1-py3-none-any.whl (13 kB)\n",
      "Installing collected packages: ollama\n",
      "Successfully installed ollama-0.5.1\n"
     ]
    }
   ],
   "source": [
    "!pip install ollama"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5591d525-6848-4937-9037-25ee724650ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-01 19:38:18.217 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-01 19:38:18.217 WARNING streamlit.runtime.state.session_state_proxy: Session state does not function when running a script without `streamlit run`\n",
      "2025-08-01 19:38:18.220 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-01 19:38:18.220 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-01 19:38:18.220 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-01 19:38:18.220 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-01 19:38:18.224 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-01 19:38:18.224 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-01 19:38:18.226 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-01 19:38:18.228 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import PyPDF2\n",
    "import faiss\n",
    "import numpy as np\n",
    "from sentence_transformers import SentenceTransformer\n",
    "import ollama\n",
    "import pickle\n",
    "import os\n",
    "from typing import List, Dict\n",
    "import re\n",
    "\n",
    "# Initialize session state\n",
    "if 'vector_store' not in st.session_state:\n",
    "    st.session_state.vector_store = None\n",
    "if 'document_chunks' not in st.session_state:\n",
    "    st.session_state.document_chunks = []\n",
    "if 'chat_history' not in st.session_state:\n",
    "    st.session_state.chat_history = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a629232-b6c4-457c-ad27-4d1b436db62b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
