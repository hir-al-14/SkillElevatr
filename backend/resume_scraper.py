import os
import io
import csv
import re
import json
import pdfplumber
import requests
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# === Configuration ===
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
OUTPUT_CSV = "data/resumes.csv"

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# === FastAPI App ===
app = FastAPI()

# === Embedding ===
def get_embedding_ollama(text):
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []

# === Section Extraction ===
def extract_sections(text):
    section_patterns = {
        "education": r"(education|academic background|educational qualifications)",
        "experience": r"(experience|work experience|employment history|professional experience)",
        "projects": r"(projects|personal projects|academic projects)",
        "skills": r"(skills|technical skills|core competencies)",
        "other": r"(certifications|awards|honors|activities|interests|languages|volunteering)"
    }

    text_lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    text = '\n'.join(text_lines)

    section_indices = {}
    for key, pattern in section_patterns.items():
        match = re.search(rf"\b{pattern}\b", text, re.IGNORECASE)
        if match:
            section_indices[key] = match.start()

    sorted_sections = sorted(section_indices.items(), key=lambda x: x[1])
    sections = {}
    for i, (section, start_idx) in enumerate(sorted_sections):
        end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        sections[section] = text[start_idx:end_idx].strip()

    for key in section_patterns:
        if key not in sections:
            sections[key] = ""

    name = text_lines[0].title() if text_lines and len(text_lines[0]) < 100 else "Unknown"
    logger.debug(f"Extracted sections: {list(sections.keys())}")

    return {
        "name": name,
        "education": sections["education"],
        "experience": sections["experience"],
        "projects": sections["projects"],
        "skills": sections["skills"],
        "other": sections["other"]
    }

# === Save to CSV ===
def save_to_csv(data_row):
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        # Always write the header
        writer.writerow([
            "file", "name", "education", "experience", "projects", "skills", "other",
            "emb_education", "emb_experience", "emb_projects", "emb_skills", "emb_other"
        ])
        # Write only the new resume data
        writer.writerow(data_row)
    logger.info(f"Resume CSV cleared and updated with new entry.")

# === Upload Resume Endpoint ===
@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        content = await file.read()
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        sections = extract_sections(text)
        embeddings = {k: get_embedding_ollama(v) for k, v in sections.items() if k != "name"}

        row = [
            file.filename,
            sections["name"],
            sections["education"],
            sections["experience"],
            sections["projects"],
            sections["skills"],
            sections["other"],
            json.dumps(embeddings["education"]),
            json.dumps(embeddings["experience"]),
            json.dumps(embeddings["projects"]),
            json.dumps(embeddings["skills"]),
            json.dumps(embeddings["other"]),
        ]

        save_to_csv(row)

        logger.info(f"Resume processed successfully: {file.filename}")
        return JSONResponse({
            "message": "Resume uploaded and embedded successfully",
            "name": sections["name"],
            "sections": list(sections.keys())
        })

    except Exception as e:
        logger.exception("Failed to process resume")
        raise HTTPException(status_code=500, detail="Something went wrong while processing the resume.")
