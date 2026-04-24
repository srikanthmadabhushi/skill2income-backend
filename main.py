from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Client (uses environment variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request model
class UserInput(BaseModel):
    skills: str
    interests: str
    time: str

# Health check API
@app.get("/")
def home():
    return {"message": "Skill2Income API Running"}

# Main API
@app.post("/generate-income-plan")
def generate_income_plan(user_input: UserInput):
    try:
        prompt = f"""
        You are an expert career and business advisor.

        Based on the following user details:
        Skills: {user_input.skills}
        Interests: {user_input.interests}
        Time Available: {user_input.time}

        Generate exactly 3 income ideas.

        Return ONLY JSON (no explanation), in this format:

        [
            {{
                "title": "Idea Title",
                "description": "Short explanation",
                "steps": ["Step 1", "Step 2", "Step 3"],
                "earnings": "$500-$2000/month",
                "difficulty": "Beginner/Intermediate/Advanced",
                "recommended": true
            }}
        ]
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()

        # 🔥 IMPORTANT FIX: Convert string → JSON
        try:
            parsed_result = json.loads(response_text)
            return {"result": parsed_result}
        except Exception:
            return {
                "error": "Failed to parse AI response",
                "raw": response_text
            }

    except Exception as e:
        return {"error": str(e)}