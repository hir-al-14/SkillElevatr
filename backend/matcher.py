import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["skillelevatr"]
collection = db["jobs"]

# Cosine similarity function
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if not a.any() or not b.any():
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Resume loading from CSV
def load_resume_embeddings(csv_file="data/resumes.csv"):
    if not os.path.exists(csv_file):
        print("[ERROR] resumes.csv not found.")
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
                "other": json.loads(row["emb_other"])
            }
        })
    return resumes

# Weighted similarity
weights = {
    "education": 0.05,
    "experience": 0.25,
    "projects": 0.25,
    "skills": 0.40,
    "other": 0.05
}

def match_score(job_embedding, resume_embeds):
    score = 0
    for section, weight in weights.items():
        score += weight * cosine_similarity(job_embedding, resume_embeds.get(section, []))
    return round(score * 100, 2)

def main():
    title_query = input("Enter job title to match: ").strip().lower()
    location_query = input("Enter preferred location: ").strip().lower()

    print("[INFO] Loaded resumes from CSV.")
    resumes = load_resume_embeddings()
    if not resumes:
        return

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
        return

    for job in matching_jobs:
        job_title = job.get("title")
        company = job.get("company")
        location = job.get("location")
        job_embedding = job.get("embedding")

        for resume in resumes:
            score = match_score(job_embedding, resume["embeddings"])
            print(f"[MATCH] {resume['name']} → {job_title} at {company} in {location} = {score}%")

if __name__ == "__main__":
    main()
