# === Setup ===
import streamlit as st
import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# === Configuration ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Data Sharing Assistant", layout="wide")
st.title("🛡️ Data Sharing Policy Assistant (SDAIA)")

# === Sidebar: Developer Credits ===
with st.sidebar:
    st.divider()
    st.markdown("### 👨‍💻 Developers Team")
    st.caption("Proudly developed by:")
    
    # Replace these with your actual names
    st.markdown("• **Abdulrahman Alenizy**") 
    st.markdown("• **Abdulaiziz Alzuaiber**")
    st.markdown("• **Ayoub Alzahim**")
    st.markdown("• **Hamad Dahash**")
    st.markdown("• **Khalid Alotaibi**")
    
    st.markdown("---")
    st.markdown("© 2025 Data Sharing Policy")

# === 1. Custom Cleaning & Extraction Strategy ===
def clean_page_content(text):
    """
    Removes headers, footers, and recurring noise found in the SDAIA document
    to improve embedding quality.
    """
    # Remove header/footer noise patterns observed in the file
    patterns = [
        r"SDAIA",
        r"Saudi Data & AI Authority",
        r"الهيئة السعودية للبيانات",
        r"والذكاء الاصطناعي",
        r"Document Classification: Public",
        r"Version \d+\.\d+",
        r"^\d+$" # Standalone page numbers
    ]
    
    for p in patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Compress multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    return text

def process_policy_document(file_path: str, source_name: str):
    """
    Loads PDF and splits based on the specific 'First:', 'Second:' structure 
    of the Data Sharing Policy.
    """
    loader = PyPDFLoader(file_path)
    raw_pages = loader.load()
    
    # 1. Merge all pages into one text block for coherent splitting
    full_text = "\n".join([clean_page_content(p.page_content) for p in raw_pages])
    
    # 2. Define High-Level Separators based on document structure
    # The document uses "First:", "Second:", etc. and "Definitions"
    separators = [
        "\nFirst:", "\nSecond:", "\nThird:", "\nFourth:", 
        "\nFifth:", "\nSixth:", "\nSeventh:", "\nEighth:", "\nDefinitions"
    ]
    
    # 3. Use Recursive Splitter with specific separators to keep sections together
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators + ["\n\n", "\n", "."], # Fallback separators
        chunk_size=2000, # Large chunk size to keep full clauses together
        chunk_overlap=200,
        keep_separator=True # Keep the "First:", "Second:" in the text
    )
    
    chunks = text_splitter.split_text(full_text)
    
    # 4. Convert back to Documents
    documents = []
    for chunk in chunks:
        # Attempt to extract the section title for metadata
        first_line = chunk.strip().split('\n')[0]
        section_title = first_line[:50] + "..." if len(first_line) > 50 else first_line
        
        doc = Document(
            page_content=chunk,
            metadata={"source": source_name, "section": section_title}
        )
        documents.append(doc)
        
    return documents

# === 3. Initialization (Updated with Re-ranking) ===
@st.cache_resource
def initialize():
    doc_dir = "./documents"
    all_docs = []
    
    # Ensure directory exists
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        st.error(f"Please place 'DataSharingPolicy.pdf' in the '{doc_dir}' folder.")
        return None, None

    # Load Documents
    for filename in os.listdir(doc_dir):
        if filename.endswith(".pdf"):
            full_path = os.path.join(doc_dir, filename)
            source_name = os.path.splitext(filename)[0]
            docs = process_policy_document(full_path, source_name)
            all_docs.extend(docs)

    if not all_docs:
        return None, None

    # 1. Embeddings & Vector Store
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(all_docs, embedding_model)
    
    # 2. Base Retriever (High Recall)
    # We fetch 20 documents here to ensure we cast a wide net initially.
    base_retriever = vector_store.as_retriever(
        search_type="mmr", # Still use MMR for initial diversity
        search_kwargs={"k": 20, "lambda_mult": 0.7} 
    )

    # 3. Re-ranking (High Precision)
    # We use a CrossEncoder (BGE-Reranker is excellent for this).
    # This runs locally on CPU (or GPU if available).
    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    
    # Configure the compressor to pick the top 5 most relevant from the 20
    compressor = CrossEncoderReranker(model=rerank_model, top_n=5)
    
    # Combine them into a Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 4. LLM & Memory
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # 5. Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
                    You are a Data Governance Consultant specialized in SDAIA's Data Sharing Policy.
                    Answer the user's question strictly based on the context provided.

                    Guidelines:
                    1. If the user asks for a procedure, list the steps clearly.
                    2. If not in context, say so.

                    Context:
                    {context}

                    Chat History:
                    {chat_history}

                    Question: {question}
                    Answer:
                    """
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever, # <--- Use the Re-ranking Retriever here
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        output_key="answer"
    )

    return qa_chain, memory

# === Main Execution ===
qa_chain, memory = initialize()

if qa_chain:
    # Chat UI
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_display = []
        memory.clear()
        st.rerun()

    for user_msg, bot_msg, sources in st.session_state.chat_display:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)
            # with st.expander("📚 Source Sections"):
            #     for doc in sources:
            #         st.markdown(f"- **{doc.metadata.get('source')}** | *{doc.metadata.get('section')}*")

    query = st.chat_input("Ask about Data Sharing Controls, Principles, or Procedures...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Analyzing Policy..."):
            result = qa_chain({"question": query})
            answer = result["answer"]
            sources = result.get("source_documents", [])

        with st.chat_message("assistant"):
            st.markdown(answer)
            # with st.expander("📚 Source Sections"):
            #     for doc in sources:
            #         st.markdown(f"- **{doc.metadata.get('source')}** | *{doc.metadata.get('section')}*")

        st.session_state.chat_display.append((query, answer, sources))
    
else:
    st.info("Please upload the 'DataSharingPolicy.pdf' to the documents folder to begin.")

