# 🛡️ Data Sharing Policy Assistant – SDAIA 🇸🇦

The **Data Sharing Policy Assistant** is an AI-powered tool designed to streamline the understanding of the **Data Sharing Policy** issued by the Saudi Data & Artificial Intelligence Authority (SDAIA).

Built using an advanced **RAG (Retrieval-Augmented Generation)** pipeline with **Cross-Encoder Re-ranking**, this assistant allows government entities and private sector organizations to instantly query regulations regarding data sharing requests, classification protocols, response timeframes, and dispute resolution mechanisms.

---

## 🚀 Live Demo

🔗 [Click here to interact with the Data Sharing Assistant](https://datasharingpolicy-vccud9xjrv2dsdlrbm9sbf.streamlit.app/)

---

## 🧠 Features

- [cite_start]💬 **Context-Aware Q&A**: Answers queries based strictly on the official Data Sharing Policy document[cite: 1, 4].
- 🎯 **High-Precision Retrieval**: Utilizes a two-stage retrieval process (Vector Search + Cross-Encoder Re-ranking) to ensure the most relevant clauses are cited.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (LangChain Framework)
- **LLM**: OpenAI (GPT-4o-mini)
- **Vector Store**: FAISS (using MMR for diverse retrieval)
- **Embeddings**: OpenAIEmbeddings (`text-embedding-3-small`)
- **Re-Ranking**: HuggingFace Cross-Encoders (`BAAI/bge-reranker-base`)
- **Document Processing**: `pypdf` with custom noise reduction for SDAIA headers

---

## 📋 How It Works

1. **Document Ingestion**: The policy PDF is cleaned (removing headers/footers) and split into semantic chunks based on "Clauses" and "Principles".
2. **Vector Retrieval**: The system first retrieves a broad set of candidate chunks (Top-20) using FAISS.
3. **Semantic Re-Ranking**: A **Cross-Encoder model** scores these candidates against the user's specific question to filter out irrelevant matches.
4. **Answer Generation**: The Top-5 highest-scored chunks are passed to GPT-4o-mini to generate a legally grounded answer.



---

## ✅ Use Cases

- **Government Entities**: Verifying the correct procedure for submitting data requests via the Government Service Bus.
- **Compliance Officers**: Checking response deadlines (10 days) and rejection protocols.
- **Data Stewards**: Understanding responsibilities regarding data quality and "Single Source of Truth" principles.
- **Legal Advisors**: Quickly finding dispute resolution mechanisms and liability clauses.

---

## 👨‍💻 Development Team

This project was proudly developed by:

* **Abdulrahman Alenizy**
* **Abdulaiziz Alzuaiber**
* **Ayoub Alzahim**
* **Hamad Dahash**
* **Khalid Alotaibi**

---

## ⚠️ Disclaimer

This chatbot is an educational and assistive tool. It **does not constitute legal advice**.  
For official interpretation of regulations, always refer to the original policy documents issued by the **Saudi Data & Artificial Intelligence Authority (SDAIA)**.

---

## 📦 Installation (For Local Development)

```bash
# Clone the repository
git clone https://github.com/AlenizyAbdulrahman/DataSharingPolicy.git
cd DataSharingPolicy

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run second-app.py
