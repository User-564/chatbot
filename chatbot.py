import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Custom CSS for styling
st.markdown("""
<style>     
    .stApp {         
        background: #ff00ff; /* Magenta background */
        background-image: radial-gradient(#d4d0c4 1px, transparent 1px);         
        background-size: 20px 20px;     
    }     

    .chat-font {         
        font-family: 'Times New Roman', serif;         
        color: #000000 !important; /* Black font color */
        font-size: smaller; /* Reduced font size */
    }     

    .user-msg {         
        background: #ffffff !important;         
        border-radius: 15px !important;         
        border: 2px solid #2c5f2d !important;     
        color: #000000 !important; /* Ensure user message text is black */
    }     

    .bot-msg {         
        background: #fff9e6 !important;         
        border-radius: 15px !important;         
        border: 2px solid #ffd700 !important;     
        color: #000000 !important; /* Ensure bot message text is black */
    }     

    .stChatInput {         
        background: #ffffff !important; /* White input background */
        color: #000000 !important; /* Black text inside input box */
        border: 2px solid #000000 !important; /* Black border for input box */
    } 

    .stChatInput textarea {
        color: #000000 !important; /* Ensures typed text inside the input box is black */
    }
</style>


""", unsafe_allow_html=True)

# Configure Google Gemini
genai.configure(api_key="AIzaSyBxiO38S5ZXAi-kCG3B71Q8oBFJiGCFRiY")  # Replace with your Gemini API key
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model

# Load data and create FAISS index
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data.csv')  # Replace with your dataset file name
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()
        df['context'] = df.apply(
            lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", 
            axis=1
        )
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS index for similarity search
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

# Load dataset and FAISS index
df, faiss_index = load_data()

# App Header
st.markdown('<h1 class="chat-font">🤖 SANJNA LANKADI Clone Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond as SANJANA LANAKDI!</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find the closest matching question using FAISS
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)  # Top 1 match
    if I.size > 0:
        return df.iloc[I[0][0]]['answer']  # Return the closest answer
    return None

# Function to generate a refined answer using Gemini
def generate_refined_answer(query, retrieved_answer):
    prompt = f"""You are SANJANA LANAKDI. Respond to the following question in a friendly and conversational tone:
    Question: {query}
    Retrieved Answer: {retrieved_answer}
    - Provide a detailed and romantic response.
    - Ensure the response is grammatically correct and engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            # Find the closest answer
            retrieved_answer = find_closest_question(prompt, faiss_index, df)
            if retrieved_answer:
                # Generate a refined answer using Gemini
                refined_answer = generate_refined_answer(prompt, retrieved_answer)
                response = f"**SANJANA LANKADI**:\n{refined_answer}"
            else:
                response = "**SANJANA LANKADI**:\nI'm sorry, I cannot answer that question."
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
