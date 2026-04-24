from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# ✅ Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Replace with your API key
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request model
class UserInput(BaseModel):
    skills: str
    interests: str
    time: str


@app.get("/")
def home():
    return {"message": "Skill2Income API Running"}


@app.post("/generate-income-plan")
def generate_plan(user: UserInput):

    prompt = f"""
Return ONLY valid JSON. No explanation.

Return EXACTLY 3 ideas.

Format:
[
  {{
    "title": "string",
    "description": "string",
    "steps": ["step1", "step2", "step3"],
    "earnings": "string",
    "monthly_estimate": "string",
    "difficulty": "Beginner / Intermediate / Advanced",
    "recommended": true/false
  }}
]

Rules:
- Only ONE idea should have "recommended": true

User:
Skills: {user.skills}
Interests: {user.interests}
Time: {user.time}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You return ONLY clean JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    result = response.choices[0].message.content.strip()

    # Clean markdown if AI adds it
    if result.startswith("```"):
        result = result.replace("```json", "").replace("```", "").strip()

    return {"result": result}