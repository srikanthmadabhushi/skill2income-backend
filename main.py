from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import os

app = FastAPI()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class UserInput(BaseModel):
    skills: str
    interests: str
    time: str

@app.get("/")
def home():
    return {"message": "Skill2Income API Running 🚀"}

@app.post("/generate-income-plan")
def generate_income_plan(data: UserInput):
    try:
        prompt = f"""
        Generate 3 income ideas based on:
        Skills: {data.skills}
        Interests: {data.interests}
        Time Available: {data.time}

        Return ONLY valid JSON in this format:

        [
          {{
            "title": "...",
            "description": "...",
            "steps": ["...", "..."],
            "earnings": "...",
            "monthly_estimate": "...",
            "difficulty": "...",
            "recommended": true
          }}
        ]
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        result_text = response.choices[0].message.content.strip()

        # ✅ Convert string → JSON
        result_json = json.loads(result_text)

        return {"result": result_json}

    except Exception as e:
        return {"error": str(e)}
