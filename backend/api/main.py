from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
import pdfplumber
import requests
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
RESUME_CSV = "data/resumes.csv"

client = MongoClient(MONGO_URI)
db = client["skillelevatr"]
collection = db["jobs"]

weights = {
    "education": 0.05,
    "experience": 0.25,
    "projects": 0.25,
    "skills": 0.40,
    "other": 0.05
}

# ========== Utils ==========
def get_embedding(text):
    try:
        res = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
        return res.json()["embedding"]
    except:
        return []

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if not a.any() or not b.any():
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ========== Upload Resume ==========
def extract_sections(text):
    sections = {k: "" for k in weights}
    patterns = {
        "education": r"education|academic",
        "experience": r"experience|work",
        "projects": r"projects",
        "skills": r"skills|technologies",
        "other": r"certifications|awards|languages"
    }
    text = text.lower()
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            sections[key] = text[match.start():match.end()+500]
    return sections

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes supported")

    content = await file.read()
    with open("temp_resume.pdf", "wb") as f:
        f.write(content)

    with pdfplumber.open("temp_resume.pdf") as pdf:
        text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())

    os.remove("temp_resume.pdf")

    sections = extract_sections(text)
    embeddings = {k: get_embedding(v) for k, v in sections.items()}

    name = file.filename.split(".")[0]
    new_row = {
        "file": file.filename,
        "name": name,
        **sections,
        **{f"emb_{k}": json.dumps(v) for k, v in embeddings.items()}
    }

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([new_row])
    if not os.path.exists(RESUME_CSV):
        df.to_csv(RESUME_CSV, index=False)
    else:
        df.to_csv(RESUME_CSV, mode="a", header=False, index=False)

    return {"message": "Resume uploaded and embedded successfully", "embedding": list(embeddings.values())[0][:5]}

# ========== Match Jobs ==========
class MatchRequest(BaseModel):
    title: str
    location: str
    job_type: str
    min_salary: float
    max_salary: float

@app.post("/match_jobs")
def match_jobs(data: MatchRequest):
    resumes = pd.read_csv(RESUME_CSV)
    if resumes.empty:
        return {"matches": []}

    # Load last uploaded resume
    row = resumes.iloc[-1]
    resume_embeds = {
        k: json.loads(row[f"emb_{k}"]) for k in weights
    }

    filters = {
        "title": {"$regex": data.title, "$options": "i"},
        "location": {"$regex": data.location, "$options": "i"},
        "job_type": {"$regex": data.job_type, "$options": "i"},
        "embedding": {"$exists": True},
        "min_amount": {"$lte": data.max_salary},
        "max_amount": {"$gte": data.min_salary}
    }

    jobs = list(collection.find(filters))
    matches = []
    for job in jobs:
        job_emb = job["embedding"]
        score = sum(weights[k] * cosine_similarity(job_emb, resume_embeds[k]) for k in weights)
        matches.append({
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "score": round(score * 100, 2)
        })
    return {"matches": matches}

# ========== Get Top Matches ==========
@app.get("/get_matches")
def get_matches(top_n: int = 5):
    resumes = pd.read_csv(RESUME_CSV)
    if resumes.empty:
        return {"matches": []}

    row = resumes.iloc[-1]
    resume_embeds = {
        k: json.loads(row[f"emb_{k}"]) for k in weights
    }

    jobs = list(collection.find({"embedding": {"$exists": True}}))
    matches = []
    for job in jobs:
        job_emb = job["embedding"]
        score = sum(weights[k] * cosine_similarity(job_emb, resume_embeds[k]) for k in weights)
        matches.append({
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "score": round(score * 100, 2)
        })
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)
    return {"matches": matches[:top_n]}
