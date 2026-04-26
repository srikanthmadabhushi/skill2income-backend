from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# ENFORCEMENT RULES
# =========================

BANNED = ["freelance", "course", "blog", "tutorial"]

ALLOWED_CATEGORIES = ["saas", "automation", "product", "marketplace", "consulting"]

def hard_filter(ideas):
    clean = []
    for idea in ideas:
        title = idea.get("title", "").lower()

        # ❌ remove bad patterns
        if any(b in title for b in BANNED):
            continue

        clean.append(idea)

    return clean


def enforce_categories(ideas):
    seen = set()
    final = []

    for idea in ideas:
        model = idea.get("business_model", "").lower()

        # map loosely into categories
        if "saas" in model:
            cat = "saas"
        elif "automation" in model:
            cat = "automation"
        elif "market" in model:
            cat = "marketplace"
        elif "product" in model or "template" in model:
            cat = "product"
        else:
            cat = "consulting"

        if cat not in seen:
            seen.add(cat)
            final.append(idea)

    return final


def is_valid(ideas):
    if len(ideas) < 3:
        return False

    titles = [i.get("title","").lower() for i in ideas]

    # ❌ if any banned slipped through
    if any(any(b in t for b in BANNED) for t in titles):
        return False

    # ❌ if all same type
    models = set(i.get("business_model","") for i in ideas)
    if len(models) < 2:
        return False

    return True


# =========================
# AI CALL
# =========================

def generate_ideas(skills, history_text):
    prompt = f"""
You are a startup strategist.

Generate EXACTLY 5 income ideas.

Skills: {skills}
Context: {history_text}

STRICT:
- Avoid freelancing, courses, blogs
- Focus on SaaS, automation, tools, marketplaces
- Each idea must be different

Return JSON ARRAY.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1.3,
        messages=[
            {"role": "system", "content": "Generate high-quality startup ideas."},
            {"role": "user", "content": prompt}
        ]
    )

    txt = response.choices[0].message.content.strip()
    txt = txt.replace("```json","").replace("```","")

    try:
        return json.loads(txt)
    except:
        return []


# =========================
# MAIN API
# =========================

@app.post("/generate-income-plan")
async def generate_income_plan(data: dict):

    skills = data.get("skills","")
    history = data.get("history",[])
    history_text = "\n".join([h.get("text","") for h in history])

    max_attempts = 5
    attempt = 0
    ideas = []

    while attempt < max_attempts:

        raw = generate_ideas(skills, history_text)

        # 🔥 ENFORCEMENT PIPELINE
        filtered = hard_filter(raw)
        categorized = enforce_categories(filtered)

        ideas = categorized[:3]

        if is_valid(ideas):
            break

        attempt += 1

    # 🚨 FINAL FALLBACK (guaranteed output)
    if len(ideas) < 3:
        ideas = [
            {
                "title": f"{skills} SaaS Analytics Tool",
                "description": "Build a subscription tool solving a niche problem",
                "business_model": "SaaS",
                "score": 80,
                "risk_level": "Medium",
                "monthly_projection": {"month1":"$0","month2":"$500","month3":"$2000"}
            },
            {
                "title": f"{skills} Automation System",
                "description": "Automate repetitive business workflows",
                "business_model": "Automation",
                "score": 75,
                "risk_level": "Low",
                "monthly_projection": {"month1":"$200","month2":"$800","month3":"$2500"}
            },
            {
                "title": f"{skills} Template Marketplace",
                "description": "Sell reusable templates/assets",
                "business_model": "Product",
                "score": 78,
                "risk_level": "Low",
                "monthly_projection": {"month1":"$100","month2":"$600","month3":"$1800"}
            }
        ]

    # best option
    best = max(ideas, key=lambda x: x.get("score", 0))
    for idea in ideas:
        idea["best_option"] = (idea == best)

    return {"result": ideas}