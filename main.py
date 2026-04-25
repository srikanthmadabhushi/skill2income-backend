from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ensure_fields(idea: dict) -> dict:
    """Guarantee all fields exist with non-empty values."""
    defaults = {
        "title": "Income Idea",
        "who_is_this_for": "Beginners looking to start earning online",
        "description": "A practical way to generate income using your skills.",
        "why_it_works": "Strong demand and scalable execution.",
        "tools_needed": ["Fiverr", "Upwork"],
        "steps": ["Identify a niche", "Create a simple offering", "Start outreach"],
        "time_to_start": "1–3 days",
        "earnings": "$20–$50 per hour",
        "monthly_estimate": "$500–$2000",
        "difficulty": "Beginner",
        "pro_tip": "Start with a niche and validate quickly."
    }

    for k, v in defaults.items():
        if not idea.get(k) or idea.get(k) in ["-", ""]:
            idea[k] = v
    return idea


def make_unique(ideas: list) -> list:
    """Ensure ideas are not duplicates by tweaking titles if needed."""
    seen = set()
    for i, idea in enumerate(ideas):
        title = idea.get("title", "").lower()
        if title in seen:
            idea["title"] += f" ({i+1})"
        seen.add(title)
    return ideas


@app.post("/generate-income-plan")
async def generate_income_plan(data: dict):
    try:
        history = data.get("history", [])

        messages = []

        # 🧠 memory
        for h in history:
            messages.append({
                "role": h["role"],
                "content": h["text"]
            })

        # 🔥 STRONG PROMPT (forces uniqueness + specifics)
        messages.append({
            "role": "user",
            "content": f"""
You are a senior business strategist.

Generate EXACTLY 3 completely DIFFERENT income ideas.

User:
Skills: {data.get("skills")}
Interests: {data.get("interests")}
Time Available: {data.get("time")}

STRICT RULES:
- Each idea MUST target a DIFFERENT audience
- Each idea MUST use DIFFERENT tools/platforms
- Each idea MUST be a DIFFERENT type of income (freelance / content / product / etc.)
- NO generic phrases like "general audience"
- NO repetition
- ALL fields must be filled

Return ONLY valid JSON:

[
  {{
    "title": "...",
    "who_is_this_for": "... specific audience",
    "description": "...",
    "why_it_works": "... specific reason",
    "tools_needed": ["tool1","tool2"],
    "steps": ["step1","step2","step3"],
    "time_to_start": "...",
    "earnings": "...",
    "monthly_estimate": "...",
    "difficulty": "...",
    "pro_tip": "..."
  }}
]
"""
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=1.1,  # 🔥 encourages diversity
        )

        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()

        try:
            ideas = json.loads(result_text)
        except:
            return {"error": "Invalid JSON from AI", "raw": result_text}

        # 🔥 CLEAN + FIX
        cleaned = []
        for idea in ideas:
            idea = ensure_fields(idea)
            cleaned.append(idea)

        cleaned = make_unique(cleaned)

        return {"result": cleaned}

    except Exception as e:
        return {"error": str(e)}