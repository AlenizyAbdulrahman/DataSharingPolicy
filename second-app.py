# === Setup ===
import streamlit as st
import os
import re
import shutil
# ⚠️ CHANGED: Using PyMuPDFLoader (Fitz) - Best for Arabic
from langchain_community.document_loaders import PyMuPDFLoader
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
st.set_page_config(page_title="SDAIA Data Sharing Assistant", layout="wide")
st.title("🛡️ Data Sharing Policy Assistant (SDAIA)")

# === Sidebar: Developer Credits ===
with st.sidebar:
    st.divider()
    st.markdown("### 👨‍💻 Developers Team")
    st.caption("Proudly developed by:")
    st.markdown("• **Abdulrahman Alenizy**") 
    st.markdown("• **Abdulaiziz Alzuaiber**")
    st.markdown("• **Ayoub Alzahim**")
    st.markdown("• **Hamad Dahash**")
    st.markdown("• **Khalid Alotaibi**")
    st.markdown("---")
    st.markdown("© 2025 Data Sharing Policy")

# === 1. Custom Cleaning Strategy ===
def clean_page_content(text):
    """
    Standard cleaning for Fitz extracted text.
    Fitz usually fixes the order, so we just remove noise.
    """
    if not text: return ""
    
    # Patterns to match Arabic headers/footers in your specific PDF
    patterns = [
        r"SDAIA",
        r"الهيئة السعودية للبيانات",
        r"والذكاء الاصطناعي",
        r"Saudi Data & AI Authority",
        r"Data & AI Authority",
        r"تصنيف الوثيقة: عام",
        r"رقم الإصدار 2.0",
        r"^\d+$"  # Standalone page numbers
    ]
    
    for p in patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove extra newlines and spaces
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    return text

# === 2. Document Processing (Using PyMuPDF) ===
def process_policy_document(file_path: str, source_name: str):
    """
    Loads PDF using PyMuPDF (Fitz) and splits based on structure.
    """
    # ⚠️ CHANGED: Using PyMuPDFLoader
    loader = PyMuPDFLoader(file_path)
    raw_pages = loader.load()
    
    # 1. Clean and Merge Content
    full_text = "\n".join([clean_page_content(p.page_content) for p in raw_pages])
    
    # 2. Define High-Level Separators based on ARABIC document structure
    separators = [
        "\nأولاً", "\nثانياً", "\nثالثاً", "\nرابعاً", 
        "\nخامساً", "\nسادساً", "\nسابعاً", "\nثامناً", 
        "\nالتعريفات", "\nالمبادئ الرئيسية"
    ]
    
    # 3. Use Recursive Splitter to keep legal clauses together
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators + ["\n\n", "\n", "."], 
        chunk_size=2000, 
        chunk_overlap=300,
        keep_separator=True 
    )
    
    chunks = text_splitter.split_text(full_text)
    
    # 4. Convert back to Documents
    documents = []
    for chunk in chunks:
        # Extract first line as section title
        first_line = chunk.strip().split('\n')[0]
        section_title = first_line[:50] + "..." if len(first_line) > 50 else first_line
        
        doc = Document(
            page_content=chunk,
            metadata={"source": source_name, "section": section_title}
        )
        documents.append(doc)
        
    return documents

# === 3. Initialization (Advanced RAG Pipeline) ===
@st.cache_resource
def initialize():
    doc_dir = "./documents"
    all_docs = []
    
    # Ensure directory exists
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        st.error(f"Please place 'DataSharingPolicyAR.pdf' in the '{doc_dir}' folder.")
        return None, None

    # Load Documents
    for filename in os.listdir(doc_dir):
        if filename.endswith("DataSharingPolicyAR.pdf"):
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
    base_retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 20, "lambda_mult": 0.7} 
    )

    # 3. Re-ranking (High Precision)
    try:
        rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=rerank_model, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        final_retriever = compression_retriever
    except Exception as e:
        st.warning(f"Re-ranker model failed: {e}. Falling back to simple retrieval.")
        final_retriever = base_retriever

    # 4. LLM & Memory
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # 5. System Prompt (Arabic)
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
        You are an expert Data Governance Consultant specialized in SDAIA's Data Sharing Policy.
        
        Guidelines:
        1. Answer strictly based on the provided Context.
        2. Answer in Arabic (unless asked otherwise).
        3. If the answer is not in the context, say "I cannot find this information in the policy."

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
        retriever=final_retriever, 
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

    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.chat_display = []
        memory.clear()
        st.rerun()

    for user_msg, bot_msg in st.session_state.chat_display:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)
            # with st.expander("📚 مصادر الإجابة"):
            #     for doc in sources:
            #          st.markdown(f"- **{doc.metadata.get('source')}** | *{doc.metadata.get('section')}*")

    query = st.chat_input("اسأل عن سياسة مشاركة البيانات...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("جاري تحليل السياسة..."):
            result = qa_chain({"question": query})
            answer = result["answer"]
            # sources = result.get("source_documents", [])

        with st.chat_message("assistant"):
            st.markdown(answer)
            # with st.expander("📚 مصادر الإجابة"):
            #     for doc in sources:
            #         st.markdown(f"- **{doc.metadata.get('source')}** | *{doc.metadata.get('section')}*")

        st.session_state.chat_display.append((query, answer))
    
else:
    st.info("⚠️ Please ensure `documents/DataSharingPolicyAR.pdf` exists.")
