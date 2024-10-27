import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import fitz  # PyMuPDF for PDF text extraction

# Load your data from PDF
@st.cache_resource
def load_data(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf_file:
        for page in pdf_file:
            text += page.get_text()
    return pd.DataFrame({'combined_text': [text]})

# Initialize the embedding model and FAISS index
@st.cache_resource
def initialize_embeddings(data):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(data['combined_text'].tolist(), convert_to_tensor=False)
    embedding_dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))
    return embedder, index

# OpenAI API setup
openai_api_key = "RAG_OPENAI_API_KEY"  # Replace with your actual OpenAI API key

# Function to retrieve top-k similar documents from FAISS index
def retrieve(query, embedder, index, data, top_k=5):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    return data.iloc[indices[0]]

# Function for RAG using OpenAI API
def rag_query(query, embedder, index, data, top_k=5):
    retrieved_docs = retrieve(query, embedder, index, data, top_k)
    context = "\n".join(retrieved_docs['combined_text'].tolist())
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

    # Call the OpenAI API
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",  # Change to your preferred model
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code == 200:
        answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")
    else:
        answer = f"Error: {response.json().get('error', 'Unknown error')}"

    return answer

# Streamlit UI
st.title("RAG Application with OpenAI API")
st.write("Ask a question, and I'll find the answer for you!")

# File uploader for PDF data
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Load data
    data = load_data(uploaded_file)
    
    # Initialize embeddings and FAISS index
    embedder, index = initialize_embeddings(data)
    
    # User input for query
    query = st.text_input("Your question:")
    
    if st.button("Get Answer"):
        if query:
            answer = rag_query(query, embedder, index, data)
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a PDF file to start.")
