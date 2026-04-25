from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json

app = FastAPI()

# ✅ CORS (no change)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# 🔧 HELPER FUNCTIONS
# =========================

def ensure_fields(idea: dict) -> dict:
    """Ensure all required fields exist (safe fallback)."""
    defaults = {
        "title": "Income Idea",
        "who_is_this_for": "Beginners looking to start earning online",
        "description": "A practical way to generate income.",
        "why_it_works": "Growing demand.",
        "tools_needed": ["Fiverr", "Upwork"],
        "steps": ["Start small", "Validate idea"],
        "time_to_start": "1–3 days",
        "earnings": "$20–$50",
        "monthly_estimate": "$500–$2000",
        "difficulty": "Beginner",
        "pro_tip": "Stay consistent",

        # 🔥 NEW DECISION ENGINE FIELDS
        "score": 70,
        "risk_level": "Medium",
        "best_option": False,
        "reason_for_best": "Balanced option",
        "timeline": "Start → Learn → Earn within 30 days",
        "what_if": "If you invest more time, results grow faster"
    }

    for k, v in defaults.items():
        if not idea.get(k) or idea.get(k) in ["-", ""]:
            idea[k] = v

    return idea


def make_unique(ideas: list) -> list:
    """Avoid duplicate titles."""
    seen = set()
    for i, idea in enumerate(ideas):
        title = idea.get("title", "").lower()
        if title in seen:
            idea["title"] += f" ({i+1})"
        seen.add(title)
    return ideas


def normalize_score(score):
    """Ensure score is between 0–100."""
    try:
        score = int(score)
        return max(0, min(score, 100))
    except:
        return 70


def fix_best_option(ideas):
    """Ensure ONLY ONE best_option = True."""
    best_found = False

    for idea in ideas:
        if idea.get("best_option") and not best_found:
            best_found = True
        else:
            idea["best_option"] = False

    # If none marked → pick highest score
    if not best_found and ideas:
        best = max(ideas, key=lambda x: x.get("score", 0))
        best["best_option"] = True

    return ideas


# =========================
# 🚀 MAIN API
# =========================

@app.post("/generate-income-plan")
async def generate_income_plan(data: dict):
    try:
        skills = data.get("skills")
        interests = data.get("interests")
        time = data.get("time")

        # ✅ FIXED PROMPT (CRITICAL)
        messages = [
            {
                "role": "system",
                "content": "You are an AI income decision engine that generates realistic, personalized, and actionable income strategies."
            },
            {
                "role": "user",
                "content": f"""
User Profile:
Skills: {skills}
Interests: {interests}
Time available: {time}

Generate EXACTLY 3 DIFFERENT income ideas.

For EACH idea return JSON with:

- title
- who_is_this_for
- description
- why_it_works
- tools_needed (array)
- steps (array)
- time_to_start
- earnings
- monthly_estimate
- difficulty

NEW FIELDS:
- score (0–100)
- risk_level (Low / Medium / High)
- best_option (true/false)
- reason_for_best
- timeline
- what_if

Rules:
- Only ONE idea must have best_option = true
- All ideas must be different
- Make results realistic and practical
- Return ONLY JSON array (no text, no explanation)
"""
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )

        result_text = response.choices[0].message.content.strip()

        # Clean markdown if any
        result_text = result_text.replace("```json", "").replace("```", "").strip()

        try:
            ideas = json.loads(result_text)
        except:
            return {"error": "Invalid JSON from AI", "raw": result_text}

        # ✅ CLEAN + VALIDATE
        cleaned = []
        for idea in ideas:
            idea = ensure_fields(idea)
            idea["score"] = normalize_score(idea.get("score", 70))
            cleaned.append(idea)

        cleaned = make_unique(cleaned)
        cleaned = fix_best_option(cleaned)

        return {"result": cleaned}

    except Exception as e:
        return {"error": str(e)}