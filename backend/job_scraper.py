import sys
import time
import uuid
import requests
import pandas as pd
from jobspy import scrape_jobs
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import json

# Load MongoDB connection
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

client = MongoClient(MONGO_URI)
db = client["skillelevatr"]
collection = db["jobs"]

def get_embedding(text: str) -> list:
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"[ERROR] Failed to embed description: {e}")
        return []

def generate_fallback_id(job_doc: dict) -> str:
    base = f"{job_doc.get('title', '')}-{job_doc.get('company', '')}-{job_doc.get('location', '')}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

def scrape_and_store_jobs(role: str, location: str, job_type: str = None, limit: int = 30):
    print(f"\n[INFO] Scraping: '{role}' in '{location}' | Job Type: {job_type or 'Any'}")

    collection.delete_many({})
    print("[INFO] Cleared old job data from MongoDB.")

    filters = {}
    if job_type and job_type.lower() != "all":
        filters["employment_types"] = [job_type.lower()]

    try:
        jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "google"],
            search_term=role,
            google_search_term=f"{role} jobs in {location}",
            location=location,
            results_wanted=limit,
            hours_old=72,
            filters=filters,
            verbose=True
        )
        df = jobs
        df.dropna(subset=["description"], inplace=True)

        if df.empty:
            print(f"[WARN] No jobs with description found.")
            return

        print(f"[INFO] {len(df)} jobs found.")
    except Exception as e:
        print(f"[ERROR] Scraping failed: {e}")
        return

    for _, row in df.iterrows():
        job_doc = {
            "role": role,
            "location_searched": location,
            "job_type_filter": job_type,
            "id": row.get("job_id") or generate_fallback_id(row),
            "site": row.get("site"),
            "job_url": row.get("job_url"),
            "title": row.get("title"),
            "company": row.get("company"),
            "location": row.get("location"),
            "date_posted": str(row.get("date_posted")),
            "job_type": row.get("job_type"),
            "min_amount": row.get("min_amount"),
            "max_amount": row.get("max_amount"),
            "description": row.get("description"),
            "embedding": get_embedding(row.get("description"))
        }

        collection.update_one(
            {"id": job_doc["id"]},
            {"$set": job_doc},
            upsert=True
        )

    print(f"[SUCCESS] Stored {len(df)} jobs for role: '{role}' in '{location}'")

def main():
    if len(sys.argv) < 3:
        print("Usage: python job_scraper.py <job_role> <location> [job_type]")
        print("Example: python job_scraper.py 'data scientist' 'India' fulltime")
        sys.exit(1)

    role = sys.argv[1]
    location = sys.argv[2]
    job_type = sys.argv[3] if len(sys.argv) > 3 else None

    scrape_and_store_jobs(role, location, job_type)

if __name__ == "__main__":
    main()
