import os
import json
import requests
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# FastAPI app
app = FastAPI()

# Logging setup (like print)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Load environment
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENAI_URL = "http://localhost:11434/api/generate"
MODEL = "nomic-embed-text"
GENAI_MODEL = "llama3"

# MongoDB
client = MongoClient(MONGO_URI)
db = client["skillelevatr"]
collection = db["jobs"]

# Resume section weights
weights = {
    "genaiskills": 0.225,
    "education": 0.05,
    "experience": 0.225,
    "projects": 0.225,
    "skills": 0.225,
    "other": 0.05
}

class MatchRequest(BaseModel):
    job_title: str
    location: str

def cosine_similarity(a, b):
    if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
        return 0.7
    a, b = np.array(a), np.array(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.7
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.7
    return float(np.dot(a, b) / (norm_a * norm_b))

def load_resume_embeddings(csv_file="data/resumes.csv"):
    if not os.path.exists(csv_file):
        log.error("[ERROR] resumes.csv not found.")
        return []
    df = pd.read_csv(csv_file)
    resumes = []
    for _, row in df.iterrows():
        resumes.append({
            "name": row["name"],
            "file": row["file"],
            "embeddings": {
                "education": json.loads(row["emb_education"]),
                "experience": json.loads(row["emb_experience"]),
                "projects": json.loads(row["emb_projects"]),
                "skills": json.loads(row["emb_skills"]),
                "other": json.loads(row["emb_other"]),
            }
        })
    return resumes

def get_required_skills_from_genai(role):
    prompt = f"List the most important skills required by top companies for the role of a '{role}'. Only return a comma-separated list of skills."
    try:
        response = requests.post(
            OLLAMA_GENAI_URL,
            json={"model": GENAI_MODEL, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            log.error("[ERROR] GenAI response failed.")
            return ""
    except Exception as e:
        log.error(f"[ERROR] GenAI call failed: {e}")
        return ""

def get_embedding(text):
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
        if response.status_code == 200:
            return response.json().get("embedding", [])
        else:
            log.error("[ERROR] Failed to get embedding.")
            return []
    except Exception as e:
        log.error(f"[ERROR] Embedding call failed: {e}")
        return []

def match_score(job_embedding, resume_embeds, genai_skill_embedding, resume_skill_embedding):
    score = 0
    score += weights["genaiskills"] * cosine_similarity(genai_skill_embedding, resume_skill_embedding)
    for section, weight in weights.items():
        if section == "genaiskills":
            continue
        score += weight * cosine_similarity(job_embedding, resume_embeds.get(section, []))
    return round(score * 100, 2)

@app.post("/match_jobs")
async def match_jobs(request: MatchRequest):
    title_query = request.job_title.strip().lower()
    location_query = request.location.strip().lower()

    resumes = load_resume_embeddings()
    if not resumes:
        raise HTTPException(status_code=400, detail="No resumes found.")

    print("[INFO] Loaded resumes from CSV.")

    print("[INFO] Generating required skills using GenAI...")
    genai_skill_list = get_required_skills_from_genai(title_query)
    print(f"[INFO] GenAI inferred skills: {genai_skill_list}")
    genai_skill_embedding = get_embedding(genai_skill_list)

    print("[INFO] Fetching jobs from MongoDB...")
    all_jobs = list(collection.find({"embedding": {"$exists": True}}))
    print(f"[DEBUG] MongoDB contains {len(all_jobs)} job(s) total.")

    matching_jobs = []
    for job in all_jobs:
        title = (job.get("title") or "").lower()
        location = (job.get("location") or "").lower()
        if title_query in title and location_query in location:
            matching_jobs.append(job)

    print(f"[DEBUG] Found {len(matching_jobs)} matching job(s).")
    if not matching_jobs:
        print("[WARN] No jobs found for given inputs.")
        return {"matches": []}

    matches = []
    for job in matching_jobs:
        job_title = job.get("title")
        company = job.get("company")
        location = job.get("location")
        job_embedding = job.get("embedding")

        for resume in resumes:
            resume_skill_embedding = resume["embeddings"].get("skills", [])
            score = match_score(job_embedding, resume["embeddings"], genai_skill_embedding, resume_skill_embedding)
            print(f"[MATCH] {resume['name']} → {job_title} at {company} in {location} = {score}%")
            matches.append({
                "resume": resume["name"],
                "job": job_title,
                "company": company,
                "location": location,
                "score": score
            })

    matches.sort(key=lambda x: -x["score"])
    return {"matches": matches}
