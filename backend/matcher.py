import os
import json
import requests
from decimal import Decimal
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient

app = FastAPI()

# === Environment Setup ===
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENAI_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "nomic-embed-text"
GENAI_MODEL = "llama3"

client = MongoClient(MONGO_URI)
collection = client["skillelevatr"]["jobs"]

class MatchRequest(BaseModel):
    job_title: str
    location: str

# === Vector Functions ===
def sanitize_vector(vec):
    try:
        arr = np.array(vec, dtype=np.float32)
        if not arr.size or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return None
        return arr
    except Exception:
        return None

def cosine_similarity(vec1, vec2) -> Optional[float]:
    v1 = sanitize_vector(vec1)
    v2 = sanitize_vector(vec2)
    if v1 is None or v2 is None:
        return None
    try:
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return round(sim, 6) if np.isfinite(sim) and sim > 0 else None
    except Exception:
        return None

# === Embedding & GenAI ===
def get_embedding(text: str) -> Optional[List[float]]:
    try:
        res = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
        print(f"[DEBUG] Embedding status: {res.status_code}")
        if res.status_code == 200:
            vec = res.json().get("embedding", [])
            cleaned = sanitize_vector(vec)
            if cleaned is None:
                print("[WARN] Invalid embedding vector")
                return None
            return cleaned.tolist()
        else:
            print(f"[ERROR] Embedding failed: {res.text}")
    except Exception as e:
        print(f"[ERROR] Embedding exception: {e}")
    return None

def get_required_skills(role: str) -> str:
    prompt = f"List only the top 10 technical skills required for a '{role}' role. Comma-separated, no extra text."
    try:
        res = requests.post(OLLAMA_GENAI_URL, json={"model": GENAI_MODEL, "prompt": prompt, "stream": False})
        if res.status_code == 200:
            response = res.json().get("response", "").strip()
            print(f"[DEBUG] GenAI skills: {response}")
            return ", ".join(response.split(",")[:10])
    except Exception as e:
        print(f"[ERROR] GenAI call failed: {e}")
    return ""

# === Resume Loader ===
def load_resumes(csv_path="data/resumes.csv"):
    if not os.path.exists(csv_path):
        print("[ERROR] Resume file not found.")
        return []

    df = pd.read_csv(csv_path)
    resumes = []
    for _, row in df.iterrows():
        try:
            embeddings = {
                "education": json.loads(row["emb_education"]),
                "experience": json.loads(row["emb_experience"]),
                "projects": json.loads(row["emb_projects"]),
                "skills": json.loads(row["emb_skills"]),
                "other": json.loads(row["emb_other"])
            }
            resumes.append({
                "name": row["name"],
                "file": row["file"],
                "embeddings": embeddings
            })
        except Exception as e:
            print(f"[WARN] Skipping resume row: {e}")
    return resumes

# === Weighted Score ===
weights = {
    "education": 0.05,
    "experience": 0.25,
    "projects": 0.25,
    "skills": 0.40,
    "other": 0.05
}

def match_score(job_vec, resume_embeds):
    score = 0
    for section, weight in weights.items():
        score += weight * cosine_similarity(job_vec, resume_embeds.get(section, []))
    return round(score * 100, 2)

# === Main Matcher Route ===
@app.post("/")
def match_jobs(request: MatchRequest):
    job_title = request.job_title.strip().lower()
    location = request.location.strip().lower()
    print(f"[INFO] Matcher triggered with job_title='{job_title}', location='{location}'")

    resumes = load_resumes()
    print(f"[INFO] Loaded {len(resumes)} resumes.")
    if not resumes:
        return {"matches": []}

    skill_text = get_required_skills(job_title)
    skill_vec = get_embedding(skill_text)
    if skill_vec is None:
        print("[ERROR] GenAI skill embedding failed")
        return {"matches": []}

    job_query = {
        "embedding": {"$exists": True},
        "role": {"$regex": f"^{job_title}$", "$options": "i"},
        "location_searched": {"$regex": f"^{location}$", "$options": "i"},
    }
    jobs = list(collection.find(job_query))
    print(f"[INFO] Found {len(jobs)} jobs in MongoDB")

    results = []
    for job in jobs:
        job_vec = sanitize_vector(job.get("embedding", []))
        if job_vec is None:
            print(f"[WARN] Skipping job {job.get('title')} - invalid embedding")
            continue

        for resume in resumes:
            score = match_score(job_vec, resume["embeddings"])
            if score > 0:
                results.append({
                    "resume": resume["name"],
                    "job_title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "score": score
                })

    print(f"[INFO] Total matches found: {len(results)}")
    return {"matches": results}
