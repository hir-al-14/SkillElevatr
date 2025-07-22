from fastapi import FastAPI
from resume_scraper import app as resume_app
from job_scraper import app as job_app
from matcher import app as matcher_app

app = FastAPI(title="SkillElevatr Main API")

# Mount sub-apps
app.mount("/resumes", resume_app)
app.mount("/jobs", job_app)
app.mount("/matcher", matcher_app)
