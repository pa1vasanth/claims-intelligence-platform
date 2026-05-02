# Real-time Claims Intelligence Platform

An AI-powered Medicare insurance claims analysis system built with RAG (Retrieval-Augmented Generation) and Groq LLaMA 3.3 70B. Ingests live CMS public data, indexes insurance policy documents into a vector database, and runs intelligent fraud/risk analysis on real claims.

---

## What It Does

- Fetches **real Medicare claims data** from CMS public APIs (no signup required)
- Builds a **RAG knowledge base** from insurance policy documents using ChromaDB
- Analyzes claims with **Groq LLaMA 3.3 70B** — grounded by retrieved policy context
- Scores each claim **0–100 risk** with findings, policy concerns, and recommended actions
- Interactive **Streamlit dashboard** with charts, claim explorer, and conversational Q&A

---

## Architecture

```
CMS Live APIs (Inpatient / Physician / Outpatient)
         ↓
   Data Ingestion → SQLite Database
   
Insurance Policy Docs (.txt)
         ↓ chunk → embed (sentence-transformers)
   ChromaDB Vector Store
   
Incoming Claim
         ↓ similarity search → Retrieved Policy Chunks  (RAG)
         ↓ prompt + context  → Groq LLaMA 3.3 70B      (LLM)
         ↓
   Risk Score + Findings + Recommended Action
         ↓
   Streamlit Dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — LLaMA 3.3 70B |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (local persistent) |
| Data Source | CMS Medicare Public APIs |
| Database | SQLite |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.9+ |

---

## Project Structure

```
├── app/
│   ├── config.py            # API keys, paths, CMS endpoints
│   ├── data_ingestion.py    # CMS API fetch + SQLite storage
│   ├── rag_pipeline.py      # Chunking, embeddings, ChromaDB indexing & retrieval
│   └── llm_analysis.py      # Groq LLM calls with RAG-augmented prompts
├── dashboard/
│   └── app.py               # Streamlit UI (5 tabs)
├── data/
│   └── policies/            # Insurance policy knowledge base documents
│       ├── medicare_drg_payment_rules.txt
│       ├── medicare_physician_billing_guidelines.txt
│       ├── fraud_detection_patterns.txt
│       └── coverage_medical_necessity.txt
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/pa1vasanth/claims-intelligence-platform.git
cd claims-intelligence-platform
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key (free at [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501**

---

## Usage

In the dashboard sidebar, click the setup buttons in order:

1. **Initialize Database** — creates SQLite tables
2. **Load CMS Claims Data** — fetches 300 real Medicare inpatient + physician claims
3. **Index Policy Knowledge Base** — embeds policy docs into ChromaDB

Then use the tabs:

| Tab | What you can do |
|---|---|
| Overview | KPI metrics, charge ratio charts, risk distribution |
| Claims Explorer | Browse real CMS claims, flag high-risk ones |
| AI Analysis | Run RAG + LLM analysis on any claim, ask follow-up questions |
| History | View all past analyses with risk scores |
| Knowledge Base | Explore indexed policies, test retrieval queries |

---

## CMS Data Sources

All data is fetched from [data.cms.gov](https://data.cms.gov) public APIs — no API key required.

| Dataset | Fields |
|---|---|
| Medicare Inpatient Hospitals by Provider & Service | DRG code, hospital, discharges, charges, Medicare payment |
| Medicare Physician & Other Practitioners by Provider & Service | NPI, specialty, HCPCS code, charges, Medicare payment |
| Medicare Outpatient Hospitals by Provider & Service | APC code, beneficiaries, charges, payment |
| Medicare Fee-for-Service Error Rate Testing | Claim #, procedure, review decision, error code |

---

## RAG Pipeline — How It Works

```python
# INDEXING (one-time)
chunks = chunk_document(policy_text)          # split into 400-word pieces
vectors = model.encode(chunks)                # embed with sentence-transformers
collection.add(documents=chunks, embeddings=vectors)  # store in ChromaDB

# RETRIEVAL (every claim analysis)
query_vector = model.encode([claim_details])
results = collection.query(query_vector, n_results=3)  # cosine similarity search
context = format_context(results)             # top-k relevant policy chunks

# GENERATION
prompt = f"Policy context: {context}\nClaim: {claim}\nAnalyze for fraud risk."
response = groq.chat.completions.create(model="llama-3.3-70b-versatile", ...)
```

---

## What You Learn From This Project

- **RAG pipeline** — chunking, embeddings, vector search, augmented generation
- **Prompt engineering** — structured JSON output, context injection
- **Vector databases** — ChromaDB indexing and cosine similarity retrieval
- **LLM APIs** — Groq/OpenAI-compatible API pattern
- **Real data ingestion** — CMS REST APIs, data normalization, SQLite
- **Healthcare domain** — DRG codes, HCPCS codes, Medicare fraud patterns, risk scoring

---

## License

MIT
