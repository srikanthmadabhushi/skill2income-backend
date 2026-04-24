from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (safe for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
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
        Generate 3 UNIQUE income ideas.

        Skills: {data.skills}
        Interests: {data.interests}
        Time Available: {data.time}

        Rules:
        - Ideas must be specific to the skill
        - Do NOT repeat generic ideas
        - Make them practical and realistic

        Return ONLY JSON:

        [
          {{
            "title": "Idea title",
            "description": "Short description",
            "steps": ["Step 1", "Step 2"],
            "earnings": "$20-$100 per hour",
            "monthly_estimate": "$500-$2000"
          }}
        ]
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        result_text = response.choices[0].message.content.strip()

        # Clean response
        result_text = result_text.replace("```json", "").replace("```", "").strip()

        result_json = json.loads(result_text)

        return {"result": result_json}

    except Exception as e:
        return {"error": str(e)}