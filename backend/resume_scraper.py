import os
import sys
import pdfplumber
import requests
import csv
import re
import json

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
OUTPUT_CSV = "data/resumes.csv"

def get_embedding_ollama(text):
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return []

def extract_sections(text):
    section_patterns = {
        "education": r"(education|academic background|educational qualifications)",
        "experience": r"(experience|work experience|employment history|professional experience)",
        "projects": r"(projects|personal projects|academic projects)",
        "skills": r"(skills|technical skills|core competencies)",
        "other": r"(certifications|awards|honors|activities|interests|languages|volunteering)"
    }

    # Normalize and clean text
    text_lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    text = '\n'.join(text_lines)

    # Find section starts
    section_indices = {}
    for key, pattern in section_patterns.items():
        match = re.search(rf"\b{pattern}\b", text, re.IGNORECASE)
        if match:
            section_indices[key] = match.start()

    sorted_sections = sorted(section_indices.items(), key=lambda x: x[1])

    # Slice sections from full text
    sections = {}
    for i, (section, start_idx) in enumerate(sorted_sections):
        end_idx = sorted_sections[i+1][1] if i + 1 < len(sorted_sections) else len(text)
        content = text[start_idx:end_idx].strip()
        sections[section] = content

    for key in section_patterns:
        if key not in sections:
            sections[key] = ""

    # Assume first line is name if short
    name = text_lines[0].title() if text_lines and len(text_lines[0]) < 100 else "Unknown"

    return {
        "name": name,
        "education": sections["education"],
        "experience": sections["experience"],
        "projects": sections["projects"],
        "skills": sections["skills"],
        "other": sections["other"]
    }

def process_resume(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())

    sections = extract_sections(text)
    embeddings = {
        k: get_embedding_ollama(v) for k, v in sections.items() if k != "name"
    }

    return [
        os.path.basename(pdf_path),
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python resume_scraper.py <resume.pdf>")
        sys.exit(1)

    resume_path = sys.argv[1]
    if not os.path.isfile(resume_path) or not resume_path.endswith(".pdf"):
        print("Error: Provide a valid PDF file.")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    print(f"[INFO] Processing {resume_path}")
    row = process_resume(resume_path)

    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "file", "name", "education", "experience", "projects", "skills", "other",
                "emb_education", "emb_experience", "emb_projects", "emb_skills", "emb_other"
            ])
        writer.writerow(row)

if __name__ == "__main__":
    main()
