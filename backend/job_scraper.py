import os
import uuid
import json
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from jobspy import scrape_jobs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# === Configuration ===
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# === MongoDB Connection ===
client = MongoClient(MONGO_URI)
db = client["skillelevatr"]
collection = db["jobs"]

# === FastAPI App ===
app = FastAPI()

# === Helper Functions ===
def get_embedding(text: str) -> list:
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        logger.error(f"Failed to embed description: {e}")
        return []

def generate_fallback_id(job_doc: dict) -> str:
    base = f"{job_doc.get('title', '')}-{job_doc.get('company', '')}-{job_doc.get('location', '')}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

def check_existing_jobs(role: str, location: str, job_type: str, limit: int) -> bool:
    query = {
        "role": role,
        "location_searched": location,
        "job_type_filter": job_type,
        "embedding": {"$exists": True}
    }
    existing_count = collection.count_documents(query)
    logger.info(f"Found {existing_count} existing jobs for '{role}' in '{location}' with type '{job_type}'.")
    return existing_count >= limit

def scrape_and_store_jobs(role: str, location: str, job_type: str = None, limit: int = 100):
    if check_existing_jobs(role, location, job_type, limit):
        logger.info(f"[SKIP] Already have {limit} or more jobs. Skipping scrape.")
        return

    logger.info(f"Scraping: '{role}' in '{location}' | Job Type: {job_type or 'Any'}")

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
            logger.warning("No jobs with description found.")
            return

        logger.info(f"{len(df)} jobs found.")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
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

        collection.update_one({"id": job_doc["id"]}, {"$set": job_doc}, upsert=True)

    logger.info(f"[SUCCESS] Stored {len(df)} jobs for role: '{role}' in '{location}'")

# === Pydantic Model ===
class JobScrapeRequest(BaseModel):
    role: str
    location: str
    job_type: str = ""
    limit: int = 100

# === FastAPI Endpoint ===
@app.post("/scrape_jobs")
def trigger_scrape(request: JobScrapeRequest):
    try:
        scrape_and_store_jobs(request.role, request.location, request.job_type, request.limit)
        return {"message": "Scraping complete or skipped if already present."}
    except Exception as e:
        logger.exception("Job scraping failed.")
        raise HTTPException(status_code=500, detail="Failed to scrape jobs.")
