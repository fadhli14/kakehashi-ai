import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# --- IMPORTS THAT WORK ON YOUR MACHINE ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kakehashi-AI 🇯🇵🇺🇸 (Gemini Edition)", layout="wide")

st.title("🌉 Kakehashi-AI: Bilingual Knowledge Base")
st.markdown("""
**Concept:** Bridging the language gap in Japanese offices using **Google Gemini**.
Upload a Japanese PDF manual, and ask questions in English.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    default_key = os.getenv("GOOGLE_API_KEY")
    api_key = st.text_input("Google Gemini API Key", value=default_key, type="password")
    uploaded_file = st.file_uploader("Upload PDF Document (Japanese)", type="pdf")

# --- HELPER FUNCTION FOR LCEL ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- MAIN LOGIC ---
if uploaded_file and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with st.spinner("Processing document... (Reading & Chunking)"):
            # 1. Load & Split
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            # 2. Embed & Store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever()
            
            # 3. Define the LLM
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            
            # 4. The "LCEL" Chain (Bypassing the broken library)
            # This manual approach is actually the 'Pro' way to do it in 2025
            template = """You are a helpful bilingual assistant.
            Answer the user's question based strictly on the provided context.
            If the context is in Japanese and the question is in English,
            answer in English but quote the original Japanese text for reference.

            Context:
            {context}

            Question:
            {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            
            # The Chain: Retrieve -> Format -> Prompt -> LLM -> String Output
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
        st.success("✅ Document Processed! Ready for questions.")

        # --- CHAT INTERFACE ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking..."):
                    # Invoke the LCEL chain
                    response = rag_chain.invoke(user_input)
                    st.markdown(response)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

else:
    st.info("👈 Please enter your Google Gemini API Key and upload a PDF to start.")