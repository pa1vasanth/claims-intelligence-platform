import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "claims.db")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
POLICIES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "policies")

# CMS Live API endpoints
CMS_APIS = {
    "inpatient": "https://data.cms.gov/data-api/v1/dataset/690ddc6c-2767-4618-b277-420ffb2bf27c/data",
    "physician": "https://data.cms.gov/data-api/v1/dataset/92396110-2aed-4d63-a6a2-5d6207d46a29/data",
    "outpatient": "https://data.cms.gov/data-api/v1/dataset/ccbc9a44-40d4-46b4-a709-5caa59212e50/data",
    "error_rate": "https://data.cms.gov/data-api/v1/dataset/6395b458-2f89-4828-8c1a-e1e16b723d48/data",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
