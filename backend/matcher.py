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
OLLAMA_GENAI_URL = "http://localhost:11434/api/generate"
MODEL = "nomic-embed-text"
GENAI_MODEL = "llama3"

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["skillelevatr"]
collection = db["jobs"]

# Weights for different resume sections
weights = {
    "genaiskills": 0.10,
    "education": 0.05,
    "experience": 0.25,
    "projects": 0.25,
    "skills": 0.40,
    "other": 0.05
}

# Cosine similarity
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if not a.any() or not b.any():
        return 0.5
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Load resumes with embeddings from CSV
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

# Get GenAI-inferred skills for a role
def get_required_skills_from_genai(role):
    prompt = f"List the most important skills required by top companies for the role of a '{role}'. Only return a comma-separated list of skills."
    try:
        response = requests.post(
            OLLAMA_GENAI_URL,
            json={"model": GENAI_MODEL, "prompt": prompt}
        )
        if response.status_code == 200:
            result = response.json()
            return result["response"].strip()
        else:
            print("[ERROR] GenAI response failed.")
            return ""
    except Exception as e:
        print(f"[ERROR] GenAI call failed: {e}")
        return ""

# Get embedding for a text using Ollama
def get_embedding(text):
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
        if response.status_code == 200:
            return response.json().get("embedding", [])
        else:
            print("[ERROR] Failed to get embedding.")
            return []
    except Exception as e:
        print(f"[ERROR] Embedding call failed: {e}")
        return []

# Calculate overall match score
def match_score(job_embedding, resume_embeds, genai_skill_embedding, resume_skill_embedding):
    score = 0
    score += weights["genaiskills"] * cosine_similarity(genai_skill_embedding, resume_skill_embedding)
    for section, weight in weights.items():
        if section == "genaiskills":
            continue
        score += weight * cosine_similarity(job_embedding, resume_embeds.get(section, []))
    return round(score * 100, 2)

# Main function
def main():
    title_query = input("Enter job title to match: ").strip().lower()
    location_query = input("Enter preferred location: ").strip().lower()

    # Load resumes
    resumes = load_resume_embeddings()
    if not resumes:
        return
    print("[INFO] Loaded resumes from CSV.")

    # GenAI skills for role
    print("[INFO] Generating required skills using GenAI...")
    genai_skill_list = get_required_skills_from_genai(title_query)
    print(f"[INFO] GenAI inferred skills: {genai_skill_list}")
    genai_skill_embedding = get_embedding(genai_skill_list)

    # Fetch jobs from MongoDB
    print("[INFO] Fetching jobs from MongoDB...")
    all_jobs = list(collection.find({"embedding": {"$exists": True}}))
    print(f"[DEBUG] MongoDB contains {len(all_jobs)} job(s) total.")

    # Filter matching jobs
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

    # Score matches
    for job in matching_jobs:
        job_title = job.get("title")
        company = job.get("company")
        location = job.get("location")
        job_embedding = job.get("embedding")

        for resume in resumes:
            resume_skill_embedding = resume["embeddings"].get("skills", [])
            score = match_score(job_embedding, resume["embeddings"], genai_skill_embedding, resume_skill_embedding)
            print(f"[MATCH] {resume['name']} → {job_title} at {company} in {location} = {score}%")

if __name__ == "__main__":
    main()
