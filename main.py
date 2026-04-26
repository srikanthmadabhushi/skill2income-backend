from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os, json, re, random

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
FALLBACK_PATTERNS = {
    "SaaS": [
        ("Maintenance Cost Estimator", "Build a subscription app for teams that estimate maintenance, hosting, and upgrade costs before touching aging systems."),
        ("Compliance Change Tracker", "Create a niche tracking tool that alerts teams when internal rules, audit needs, or policy changes require updates."),
        ("Client Portal Builder", "Offer a self-serve platform that gives small agencies and IT firms branded client portals powered by your stack."),
    ],
    "Automation": [
        ("Lead Routing Engine", "Automate lead scoring, enrichment, and routing for small sales teams that waste time triaging inbound requests."),
        ("Invoice Exception Resolver", "Automate back-office workflows that flag broken invoices, missing fields, and payment mismatches before finance reviews them."),
        ("Support Triage Workflow", "Build workflow automations that classify support tickets, suggest fixes, and route issues to the right queue."),
    ],
    "Product": [
        ("Admin Panel Starter Kit", "Sell production-ready starter kits for operations dashboards, reporting portals, and internal CRUD systems."),
        ("Industry Template Pack", "Package reusable components, templates, and deployment blueprints for a specific niche with repeated workflow needs."),
        ("Migration Audit Toolkit", "Create a paid toolkit that helps teams assess legacy codebases, map risks, and plan staged rewrites."),
    ],
}


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def extract_previous_ideas(history):
    previous_titles = set()
    previous_models = set()

    for item in history or []:
        text = item.get("text", "")

        try:
            parsed = json.loads(text) if isinstance(text, str) else text
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            parsed = parsed.get("result", parsed.get("ideas", []))

        if isinstance(parsed, list):
            for idea in parsed:
                if not isinstance(idea, dict):
                    continue
                previous_titles.add(normalize_text(idea.get("title", "")))
                previous_models.add(normalize_text(idea.get("business_model", "")))

    return previous_titles, previous_models

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


def remove_repeats(ideas, previous_titles):
    unique = []
    seen_titles = set(previous_titles)

    for idea in ideas:
        title_key = normalize_text(idea.get("title", ""))
        if not title_key or title_key in seen_titles:
            continue

        seen_titles.add(title_key)
        unique.append(idea)

    return unique


def build_fallback_ideas(skills, previous_titles):
    ideas = []
    category_order = [("SaaS", 80, "Medium"), ("Automation", 75, "Low"), ("Product", 78, "Low")]

    for business_model, score, risk_level in category_order:
        options = FALLBACK_PATTERNS[business_model][:]
        random.shuffle(options)

        for suffix, description in options:
            title = f"{skills} {suffix}"
            if normalize_text(title) in previous_titles:
                continue

            ideas.append(
                {
                    "title": title,
                    "description": description,
                    "business_model": business_model,
                    "score": score,
                    "risk_level": risk_level,
                    "monthly_projection": {"month1":"$100","month2":"$700","month3":"$2200"}
                }
            )
            break

    return ideas


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

def generate_ideas(skills, history_text, banned_titles, banned_models):
    banned_titles_text = ", ".join(sorted(banned_titles)) or "none"
    banned_models_text = ", ".join(sorted(banned_models)) or "none"

    prompt = f"""
You are a startup strategist.

Generate EXACTLY 5 income ideas.

Skills: {skills}
Context: {history_text}
Previously returned titles to avoid: {banned_titles_text}
Previously used business models to avoid reusing too heavily: {banned_models_text}

STRICT:
- Avoid freelancing, courses, blogs
- Focus on SaaS, automation, tools, marketplaces
- Every idea must target a specific niche, audience, and pain point
- Do not repeat or lightly reword any previous title
- Avoid generic titles like "<skill> SaaS Tool", "<skill> Automation", or "<skill> Marketplace"
- Make the concepts materially different from one another in customer, workflow, or monetization
- Use concrete names and descriptions, not generic placeholders

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
    previous_titles, previous_models = extract_previous_ideas(history)

    max_attempts = 5
    attempt = 0
    ideas = []

    while attempt < max_attempts:

        raw = generate_ideas(skills, history_text, previous_titles, previous_models)

        # 🔥 ENFORCEMENT PIPELINE
        filtered = hard_filter(raw)
        filtered = remove_repeats(filtered, previous_titles)
        categorized = enforce_categories(filtered)

        ideas = categorized[:3]

        if is_valid(ideas):
            break

        attempt += 1

    # 🚨 FINAL FALLBACK (guaranteed output)
    if len(ideas) < 3:
        ideas = build_fallback_ideas(skills, previous_titles)

    # best option
    best = max(ideas, key=lambda x: x.get("score", 0))
    for idea in ideas:
        idea["best_option"] = (idea == best)

    return {"result": ideas}
