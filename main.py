from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os, json, re, random, base64
from urllib import error, request
from urllib.parse import quote

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SERVICENOW_INSTANCE = os.getenv("SERVICENOW_INSTANCE", "").rstrip("/")
SERVICENOW_USERNAME = os.getenv("SERVICENOW_USERNAME", "")
SERVICENOW_PASSWORD = os.getenv("SERVICENOW_PASSWORD", "")
SERVICENOW_INCIDENT_STATE_MAP = {
    "1": "New",
    "2": "In Progress",
    "3": "On Hold",
    "6": "Resolved",
    "7": "Closed",
    "8": "Canceled",
}

# =========================
# ENFORCEMENT RULES
# =========================

BANNED = ["freelance", "course", "blog", "tutorial"]
PLAN_KEYWORDS = ["30 day", "30 days", "plan", "roadmap", "week by week", "daily plan", "day by day"]
GENERIC_TITLE_PATTERNS = [
    "saas tool",
    "automation system",
    "template marketplace",
    "business platform",
    "service app",
]

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


def is_plan_request(text):
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in PLAN_KEYWORDS)


def get_user_messages(history):
    return [item.get("text", "") for item in history or [] if item.get("role") == "user" and item.get("text")]


def build_history_summary(history, limit=6):
    trimmed = []

    for item in history or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue

        role = item.get("role", "user")
        compact_text = " ".join(text.split())
        if len(compact_text) > 220:
            compact_text = compact_text[:220] + "..."
        trimmed.append(f"{role}: {compact_text}")

    return "\n".join(trimmed[-limit:])


def compact_json_snippet(value, limit=1800):
    if value is None:
        return ""
    if isinstance(value, str):
        compact = " ".join(value.split())
    else:
        try:
            compact = json.dumps(value, ensure_ascii=True)
        except Exception:
            compact = str(value)
        compact = " ".join(compact.split())
    if len(compact) > limit:
        compact = compact[:limit] + "..."
    return compact


def build_rerun_instruction(workflow_name, rerun_count=0, previous_result=None, freshness_goal=""):
    rerun_count = int(rerun_count or 0)
    if rerun_count <= 0 and not previous_result:
        return ""

    previous_summary = compact_json_snippet(previous_result) or "No prior result summary was provided."
    freshness_line = freshness_goal.strip() or "Bring a fresh angle, different emphasis, and updated practical steps."
    return f"""
RERUN CONTEXT:
- This {workflow_name} has already been run for the same project or idea.
- Keep the same core facts, but do NOT repeat the same phrasing, same ordered bullets, or same recommendation framing.
- {freshness_line}
- Prior result summary to avoid repeating:
{previous_summary}
"""


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "have", "will",
    "about", "what", "when", "where", "which", "their", "there", "then", "them", "been",
    "want", "need", "like", "than", "just", "only", "also", "into", "over", "under", "more",
    "less", "very", "once", "each", "such", "able", "could", "would", "should", "using",
    "used", "user", "users", "idea", "project"
}


def keyword_tokens(value):
    return [token for token in normalize_text(value).split() if len(token) > 2 and token not in STOPWORDS]


def chunk_text(text, chunk_size=110, overlap=20):
    words = str(text or "").split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def retrieve_knowledge_context(query, knowledge_items, limit=4):
    query_tokens = set(keyword_tokens(query))
    if not query_tokens:
        return {"context_text": "", "citations": []}

    scored = []
    for item in knowledge_items or []:
        title = item.get("title", "")
        source = item.get("source", "")
        content = item.get("content", "")
        for idx, chunk in enumerate(chunk_text(content)):
            chunk_tokens = set(keyword_tokens(f"{title} {source} {chunk}"))
            overlap = len(query_tokens & chunk_tokens)
            if overlap <= 0:
                continue
            score = overlap * 3
            if any(token in normalize_text(title) for token in query_tokens):
                score += 2
            scored.append({
                "score": score,
                "title": title or f"Note {idx + 1}",
                "source": source,
                "chunk": chunk
            })

    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]
    if not ranked:
        return {"context_text": "", "citations": []}

    context_lines = []
    citations = []
    for idx, item in enumerate(ranked, start=1):
        label = item["title"]
        if item["source"]:
            label = f"{label} ({item['source']})"
        citations.append(label)
        context_lines.append(f"[{idx}] {label}: {item['chunk']}")

    return {
        "context_text": "\n".join(context_lines),
        "citations": citations
    }


def strip_code_fences(text):
    return str(text or "").replace("```json", "").replace("```", "").strip()


def extract_json_payload(text):
    cleaned = strip_code_fences(text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    for opener, closer in [("{", "}"), ("[", "]")]:
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start:end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                continue

    return None


def structured_json_completion(system_prompt, user_prompt, fallback, expected_type="dict", temperature=0.8, max_tokens=1000, attempts=2):
    for _ in range(max(1, attempts)):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        parsed = extract_json_payload(response.choices[0].message.content.strip())

        if expected_type == "list" and isinstance(parsed, list):
            return parsed
        if expected_type == "dict" and isinstance(parsed, dict):
            return parsed

    return fallback


def infer_focus(skills, history):
    cleaned_skills = (skills or "").strip()
    if cleaned_skills and not is_plan_request(cleaned_skills):
        return cleaned_skills

    user_messages = get_user_messages(history)

    for message in reversed(user_messages):
        if not is_plan_request(message):
            return message.strip()

    previous_titles, _ = extract_previous_ideas(history)
    if previous_titles:
        return next(iter(previous_titles))

    return cleaned_skills or "the selected skill"


def is_generic_title(title):
    normalized = normalize_text(title)
    return any(pattern in normalized for pattern in GENERIC_TITLE_PATTERNS)


def extract_previous_ideas(history, extra_ideas=None):
    previous_titles = set()
    previous_models = set()

    def add_idea(idea):
        if not isinstance(idea, dict):
            return
        title = normalize_text(idea.get("title", "") or idea.get("idea_title", ""))
        model = normalize_text(idea.get("business_model", "") or idea.get("businessModel", ""))
        if title:
            previous_titles.add(title)
        if model:
            previous_models.add(model)

    def read_payload(payload):
        if isinstance(payload, list):
            for idea in payload:
                add_idea(idea)
            return

        if not isinstance(payload, dict):
            return

        result_payload = payload.get("result")
        ideas_payload = payload.get("ideas")

        if isinstance(result_payload, list):
            read_payload(result_payload)
            return
        if isinstance(ideas_payload, list):
            read_payload(ideas_payload)
            return
        if isinstance(result_payload, dict):
            add_idea(result_payload)
            return

        add_idea(payload)

    for item in history or []:
        text = item.get("text", "")

        try:
            parsed = json.loads(text) if isinstance(text, str) else text
        except Exception:
            parsed = None

        read_payload(parsed)

    read_payload(extra_ideas or [])

    return previous_titles, previous_models

def hard_filter(ideas):
    clean = []
    for idea in ideas:
        title = idea.get("title", "").lower()

        # ❌ remove bad patterns
        if any(b in title for b in BANNED):
            continue

        if is_generic_title(title):
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


def build_fallback_ideas(skills, previous_titles, batch_offset=0):
    ideas = []
    category_order = [("SaaS", 80, "Medium"), ("Automation", 75, "Low"), ("Product", 78, "Low")]

    for category_index, (business_model, score, risk_level) in enumerate(category_order):
        options = FALLBACK_PATTERNS[business_model][:]
        if not options:
            continue

        start_index = (batch_offset + category_index) % len(options)
        ordered_options = options[start_index:] + options[:start_index]

        for suffix, description in ordered_options:
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

    if any(is_generic_title(t) for t in titles):
        return False

    # ❌ if all same type
    models = set(i.get("business_model","") for i in ideas)
    if len(models) < 2:
        return False

    return True


# =========================
# AI CALL
# =========================

def generate_ideas(skills, history_text, banned_titles, banned_models, repeat_count=0):
    banned_titles_text = ", ".join(sorted(banned_titles)) or "none"
    banned_models_text = ", ".join(sorted(banned_models)) or "none"
    variation_instruction = (
        f"This is batch #{repeat_count + 1} for the same topic. Make this batch clearly different from earlier ones."
        if repeat_count > 0
        else "Generate the first batch for this topic."
    )

    prompt = f"""
You are a startup strategist.

Generate EXACTLY 3 income ideas.

Skills: {skills}
Context: {history_text}
Previously returned titles to avoid: {banned_titles_text}
Previously used business models to avoid reusing too heavily: {banned_models_text}
Variation instruction: {variation_instruction}

STRICT:
- Avoid freelancing, courses, blogs
- Focus on SaaS, automation, tools, marketplaces
- Ideas must belong to 3 different archetypes: one workflow SaaS, one automation service/product, one reusable product or toolkit
- Every idea must target a specific niche, audience, and pain point
- Do not repeat or lightly reword any previous title
- Avoid generic titles like "<skill> SaaS Tool", "<skill> Automation", or "<skill> Marketplace"
- Avoid broad audiences like "businesses", "everyone", or "developers" without a niche qualifier
- Make the concepts materially different from one another in customer, workflow, or monetization
- Use concrete names and descriptions, not generic placeholders
- Include these fields for each object: title, description, business_model, who_is_this_for, why_it_works, tools_needed, steps, time_to_start, earnings, monthly_estimate, difficulty, timeline, what_if, pro_tip, reason_for_best, personal_reason, score, risk_level, monthly_projection

Return JSON ARRAY.
"""
    return structured_json_completion(
        system_prompt="Generate high-quality startup ideas in valid JSON.",
        user_prompt=prompt,
        fallback=[],
        expected_type="list",
        temperature=1.0,
        max_tokens=1400,
        attempts=2
    )


def generate_plan(focus, history_text):
    prompt = f"""
You are a startup execution coach.

Create a practical 30-day execution plan for: {focus}

Conversation context:
{history_text}

STRICT:
- Return JSON ARRAY with EXACTLY 4 objects
- Each object must contain: week, goal, tasks
- week must be "Week 1", "Week 2", "Week 3", "Week 4"
- tasks must be an array of 6 to 8 concrete action items
- Build the plan as a real 30-day roadmap, not more income ideas
- Keep the plan specific to the focus area and previous chat context
- Do not use generic advice like "do research" without saying what to research
"""
    parsed = structured_json_completion(
        system_prompt="Generate a practical 30-day action plan in valid JSON.",
        user_prompt=prompt,
        fallback=None,
        expected_type="list",
        temperature=0.8,
        max_tokens=1200,
        attempts=2
    )
    if parsed is not None:
        return parsed

    return [
        {"week": "Week 1", "goal": f"Define the {focus} offer", "tasks": ["Pick one niche audience.", "List the main problem you will solve.", "Write the core offer promise.", "Choose a pricing hypothesis.", "Collect 5 examples of similar products.", "Draft a simple landing page outline."]},
        {"week": "Week 2", "goal": f"Build the first {focus} version", "tasks": ["Define the minimum feature set.", "Set up the project structure.", "Build the core workflow.", "Create one usable demo path.", "Test the main user flow end to end.", "Fix the most obvious friction points."]},
        {"week": "Week 3", "goal": "Validate with real users", "tasks": ["Share the demo with 5 target users.", "Collect objections and repeated questions.", "Improve the positioning message.", "Tighten the onboarding flow.", "Add one proof point or sample result.", "Prepare a short sales message."]},
        {"week": "Week 4", "goal": "Launch and iterate", "tasks": ["Publish the landing page.", "Reach out to 20 target users.", "Track replies and demo requests.", "Adjust pricing based on feedback.", "Document the onboarding steps.", "Plan the next month based on traction."]},
    ]


def generate_execution_plan(idea_title, idea_description, history_text):
    prompt = f"""
You are a startup execution coach.

Create a highly practical execution plan for this income idea:
Title: {idea_title}
Description: {idea_description}

Conversation context:
{history_text}

STRICT:
- Return JSON OBJECT
- Include these keys: title, summary, steps, quick_wins, first_offer
- title should be a short label for the plan
- summary should explain how to start in 2 sentences max
- steps must be an array of EXACTLY 5 concrete steps
- quick_wins must be an array of EXACTLY 3 fast actions
- first_offer must be one specific first offer the user can pitch or launch
- Do not return generic advice
- Keep actions specific to the idea title and description
"""
    parsed = structured_json_completion(
        system_prompt="Generate a practical execution plan in valid JSON.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.8,
        max_tokens=900,
        attempts=2
    )
    if parsed is not None:
        return parsed

    return {
        "title": f"Launch Plan for {idea_title}",
        "summary": f"Start with one narrow customer segment for {idea_title} and build only the smallest usable version. Validate demand with real outreach before expanding features.",
        "steps": [
            "Pick one niche customer profile with a painful, repeated workflow.",
            "Outline the smallest version of the idea that solves one concrete problem.",
            "Build a clickable demo or simple working prototype for that core use case.",
            "Show the offer to 10 target users and collect objections and missing needs.",
            "Turn the best feedback into a paid pilot or early-access offer."
        ],
        "quick_wins": [
            "Write a one-sentence offer for the idea.",
            "Create a simple landing page or demo screen.",
            "Send outreach to 5 target prospects this week."
        ],
        "first_offer": f"Offer a paid pilot for {idea_title} to one niche customer group with setup support included."
    }


def generate_ai_coach_reply(idea_title, idea_description, user_message, history_text):
    prompt = f"""
You are an AI Income Coach.

Selected idea:
Title: {idea_title}
Description: {idea_description}

Conversation history:
{history_text}

Latest user message:
{user_message}

STRICT:
- Return JSON OBJECT
- Include these keys: reply, suggested_replies, next_steps
- reply should sound like a practical mentor, not a generic chatbot
- suggested_replies must be an array of EXACTLY 3 short follow-up options
- next_steps must be an array of EXACTLY 3 concrete actions
- Help the user refine, simplify, validate, or scale the selected idea
- Ask at most one direct question in the reply
- Keep the tone encouraging and specific
"""
    parsed = structured_json_completion(
        system_prompt="You are a practical AI income coach who helps users refine and execute business ideas. Return valid JSON only.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.9,
        max_tokens=900,
        attempts=2
    )
    if parsed is not None:
        return parsed

    return {
        "reply": f"{idea_title} is a solid direction if you keep the first version narrow. Start with one niche customer, one painful workflow, and one offer that can be validated quickly. Do you want a beginner path or a faster-income path?",
        "suggested_replies": [
            "Give me a beginner version",
            "Give me a faster income version",
            "How do I find first customers?"
        ],
        "next_steps": [
            "Pick one niche audience for the idea.",
            "Define the smallest outcome your offer will deliver.",
            "Write a short pitch and show it to 5 target users."
        ]
    }


def generate_launch_agent_plan(idea_title, idea_description, history_text, rerun_count=0, previous_result=None):
    rerun_instruction = build_rerun_instruction(
        "Launch Agent",
        rerun_count,
        previous_result,
        "Keep the same project direction, but shift the target angle, positioning emphasis, and daily launch wording."
    )
    prompt = f"""
You are a launch agent for early-stage income ideas.

Selected idea:
Title: {idea_title}
Description: {idea_description}

Conversation context:
{history_text}

{rerun_instruction}

STRICT:
- Return JSON OBJECT
- Include these keys: niche, first_offer, landing_page_copy, outreach_message, seven_day_plan
- niche must be a short, specific target audience and pain point
- first_offer must be a concrete starter offer
- landing_page_copy must be an object with keys: headline, subheadline, cta
- outreach_message must be one ready-to-send outreach message
- seven_day_plan must be an array of EXACTLY 7 short actions, one per day
- Avoid generic startup advice
- Make the output specific to this idea
"""
    parsed = structured_json_completion(
        system_prompt="You are a practical launch agent that turns ideas into immediate launch assets. Return valid JSON only.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.8,
        max_tokens=1000,
        attempts=2
    )
    if parsed is not None:
        return parsed

    return {
        "niche": f"Small teams that need {idea_title} to solve one repeated workflow problem faster.",
        "first_offer": f"Offer a focused pilot of {idea_title} for one niche customer with setup and feedback support included.",
        "landing_page_copy": {
            "headline": f"Launch {idea_title} without the usual delay",
            "subheadline": "A focused solution for teams that need a simpler way to handle this workflow and get results quickly.",
            "cta": "Book an early access demo"
        },
        "outreach_message": f"Hi [Name], I am building {idea_title} for teams dealing with this workflow pain every week. I can show a simple pilot version and would love 15 minutes of feedback if this is relevant to your team.",
        "seven_day_plan": [
            "Day 1: Pick one narrow niche and define the one painful workflow you will solve.",
            "Day 2: Write the offer promise, target outcome, and starter pricing.",
            "Day 3: Create a basic landing page and short demo outline.",
            "Day 4: Build the smallest usable version or mockup.",
            "Day 5: Send outreach to 10 target users in the niche.",
            "Day 6: Collect objections, questions, and buying signals from replies.",
            "Day 7: Improve the offer and ask one strong prospect for a pilot."
        ]
    }


def generate_validation_toolkit(idea_title, idea_description, history_text, rerun_count=0, previous_result=None):
    rerun_instruction = build_rerun_instruction(
        "Validation Toolkit",
        rerun_count,
        previous_result,
        "Keep the same project context, but surface a different validation angle, different questions, and different risk emphasis where possible."
    )
    prompt = f"""
You are a startup validation advisor.

Selected idea:
Title: {idea_title}
Description: {idea_description}

Conversation context:
{history_text}

{rerun_instruction}

STRICT:
- Return JSON OBJECT
- Include these keys: validation_summary, validation_score, target_customer, pains_to_confirm, competitor_snapshot, outreach_questions, green_flags, red_flags, next_actions
- validation_summary must be 2 sentences max
- validation_score must be an object with keys: label, score, reason
- label must be one of: Green, Yellow, Red
- score must be an integer from 1 to 10
- reason must explain the score in one short sentence
- target_customer must describe one clear niche customer
- pains_to_confirm must be an array of EXACTLY 3 pain points to verify
- competitor_snapshot must be an array of EXACTLY 3 short observations about alternatives or competitors
- outreach_questions must be an array of EXACTLY 5 short customer validation questions
- green_flags must be an array of EXACTLY 3 positive signals to watch for
- red_flags must be an array of EXACTLY 3 warning signs to watch for
- next_actions must be an array of EXACTLY 3 practical next validation steps
- Make the output specific to this exact idea, not generic startup advice
"""
    parsed = structured_json_completion(
        system_prompt="You are a practical startup validation advisor who helps users test demand before building too much. Return valid JSON only.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.8,
        max_tokens=1100,
        attempts=2
    )
    if parsed is not None:
        return parsed

    return {
        "validation_summary": f"{idea_title} looks strongest when positioned for one narrow customer with a repeated pain that already costs time or money. Validate urgency first before expanding the offer.",
        "validation_score": {
            "label": "Yellow",
            "score": 7,
            "reason": "The direction is promising, but demand needs proof from real customer conversations."
        },
        "target_customer": f"A niche buyer who repeatedly faces the problem solved by {idea_title} and wants a faster, simpler outcome.",
        "pains_to_confirm": [
            "How often the target user faces this problem each week.",
            "What the current workaround costs in time, money, or frustration.",
            "Whether the user would pay for a simpler result instead of continuing with the current process."
        ],
        "competitor_snapshot": [
            "Some users may already patch this problem with manual spreadsheets or general-purpose tools.",
            "The main competition may be a service provider or in-house workaround rather than a direct software product.",
            "The best positioning angle is usually speed, simplicity, or niche-specific results."
        ],
        "outreach_questions": [
            "How are you solving this problem today?",
            "What is the most frustrating part of the current workflow?",
            "How often does this issue happen in a normal week?",
            "What would a good solution need to do to feel worth paying for?",
            "Would you try a pilot if it solved this in a simpler way?"
        ],
        "green_flags": [
            "People describe the same pain point in similar words.",
            "Users already spend time or money on a workaround.",
            "Prospects ask when they can try it or see a demo."
        ],
        "red_flags": [
            "The problem sounds nice to solve but not urgent.",
            "Users say they rarely face the issue.",
            "People like the idea but avoid committing to a pilot or next conversation."
        ],
        "next_actions": [
            "Talk to 5 target users in one narrow niche.",
            "Test one simple offer statement and see which pain point gets the strongest reaction.",
            "Use the feedback to tighten the niche before building more features."
        ]
    }


def generate_first_customer_pack(idea_title, idea_description, history_text, rerun_count=0, previous_result=None):
    rerun_instruction = build_rerun_instruction(
        "First Customer Pack",
        rerun_count,
        previous_result,
        "Keep the same project context, but vary the buyer hook, offer framing, and outreach wording so it does not read like the same pack again."
    )
    prompt = f"""
You are a practical go-to-market advisor.

Selected idea:
Title: {idea_title}
Description: {idea_description}

Conversation context:
{history_text}

{rerun_instruction}

STRICT:
- Return JSON OBJECT
- Include these keys: positioning, cold_email, linkedin_dm, whatsapp_pitch, pricing_offer, landing_page_copy, call_to_action
- positioning must be one short paragraph explaining the offer and ideal buyer
- cold_email must be a ready-to-send short email
- linkedin_dm must be a short direct message
- whatsapp_pitch must be a short informal pitch
- pricing_offer must be an object with keys: package_name, price_range, what_is_included
- landing_page_copy must be an object with keys: headline, subheadline, proof_points
- proof_points must be an array of EXACTLY 3 bullets
- call_to_action must be one specific next action the user should take this week
- Make the outputs specific, concrete, and ready to use
"""
    parsed = structured_json_completion(
        system_prompt="You turn startup ideas into practical first-customer assets. Return valid JSON only.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.8,
        max_tokens=1200,
        attempts=2
    )
    if parsed is not None:
        return parsed

    return {
        "positioning": f"{idea_title} should be positioned as a focused solution for a narrow buyer who already feels this pain each week and wants faster results without extra complexity.",
        "cold_email": f"Subject: Quick idea for improving [pain point]\n\nHi [Name], I am building {idea_title} for teams that deal with this workflow repeatedly. I put together a simple approach that could remove a lot of manual effort around [specific pain point]. Would you be open to a short 15-minute call this week so I can show you the concept and get your feedback?",
        "linkedin_dm": f"Hi [Name], I am testing {idea_title} for teams dealing with [specific pain point]. I would love to show you a quick concept and hear whether this would be useful for your workflow.",
        "whatsapp_pitch": f"Hi [Name], I am building {idea_title} to help teams handle this workflow faster. I have a simple early version and would love to get your feedback if this problem comes up often for you.",
        "pricing_offer": {
            "package_name": f"{idea_title} Starter Pilot",
            "price_range": "$250-$750 for the first setup or pilot",
            "what_is_included": [
                "One focused workflow setup for a narrow use case",
                "Basic onboarding and feedback loop",
                "A short review call after the first usage period"
            ]
        },
        "landing_page_copy": {
            "headline": f"Get results faster with {idea_title}",
            "subheadline": "A focused offer for niche buyers who want a simpler way to handle this workflow and start seeing value quickly.",
            "proof_points": [
                "Focused on one painful workflow instead of a bloated platform",
                "Fast pilot setup for an early real-world use case",
                "Built around real buyer feedback before scaling features"
            ]
        },
        "call_to_action": "Send the cold email or LinkedIn DM to 5 target prospects and book 2 short discovery conversations this week."
    }


def generate_idea_comparison(ideas, history_text):
    compact_ideas = []
    for idea in ideas[:3]:
        compact_ideas.append({
            "title": idea.get("title", ""),
            "description": idea.get("description", ""),
            "who_is_this_for": idea.get("who_is_this_for", ""),
            "why_it_works": idea.get("why_it_works", ""),
            "difficulty": idea.get("difficulty", ""),
            "timeline": idea.get("timeline", ""),
            "monthly_estimate": idea.get("monthly_estimate", "")
        })

    prompt = f"""
You are a startup decision advisor.

Ideas to compare:
{json.dumps(compact_ideas, ensure_ascii=True)}

Conversation context:
{history_text}

STRICT:
- Return JSON OBJECT
- Include these keys: comparison_summary, winner_title, rankings, recommendation
- comparison_summary must be 2 sentences max
- winner_title must match one of the input titles exactly
- rankings must be an array with one object per idea
- Each ranking object must include: title, effort, speed_to_income, scalability, validation_difficulty, best_for, caution
- effort, speed_to_income, scalability, validation_difficulty must each be one of: Low, Medium, High
- recommendation must explain which idea to start with and why
- Keep the comparison practical and specific
"""
    parsed = structured_json_completion(
        system_prompt="You compare startup ideas and help users choose the best one to start with. Return valid JSON only.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.7,
        max_tokens=1200,
        attempts=2
    )
    if parsed is not None:
        return parsed

    rankings = []
    for idx, idea in enumerate(compact_ideas):
        rankings.append({
            "title": idea.get("title", f"Idea {idx + 1}"),
            "effort": "Medium" if idx == 0 else ("Low" if idx == 1 else "High"),
            "speed_to_income": "High" if idx == 0 else ("Medium" if idx == 1 else "Low"),
            "scalability": "Medium" if idx == 0 else ("Low" if idx == 1 else "High"),
            "validation_difficulty": "Medium" if idx == 0 else ("Low" if idx == 1 else "High"),
            "best_for": "Starting with a focused niche and validating demand quickly.",
            "caution": "Keep the first version narrow so you do not overbuild too early."
        })

    winner_title = compact_ideas[0].get("title", "Idea 1") if compact_ideas else "Idea 1"
    return {
        "comparison_summary": "These ideas differ most in how fast you can validate them and how much complexity they require up front. Start with the one that reaches real customer conversations fastest without too much build effort.",
        "winner_title": winner_title,
        "rankings": rankings,
        "recommendation": f"Start with {winner_title} first because it gives the best balance of speed, clarity, and validation potential for an early launch."
    }


def generate_project_strategy_agent(project, history_text, rerun_count=0, previous_result=None):
    compact_project = {
        "title": project.get("title", ""),
        "description": project.get("description", ""),
        "who_is_this_for": project.get("who_is_this_for", ""),
        "why_it_works": project.get("why_it_works", ""),
        "current_goal": project.get("current_goal", ""),
        "progress": project.get("progress", {}),
        "validation_score": (project.get("validation_toolkit") or {}).get("validation_score", {}),
        "execution_summary": (project.get("execution_plan") or {}).get("summary", ""),
        "first_offer": (project.get("launch_agent") or {}).get("first_offer", ""),
        "pricing_offer": ((project.get("first_customer_pack") or {}).get("pricing_offer") or {}).get("price_range", "")
    }

    prompt = f"""
You are a multi-step Project Strategy Agent for an AI startup workspace.

Project context:
{json.dumps(compact_project, ensure_ascii=True)}

Conversation context:
{history_text}

{build_rerun_instruction("Project Strategy Agent", rerun_count, previous_result, "Keep the diagnosis aligned to the same facts, but refresh the bottleneck framing, action queue emphasis, and follow-up questions.")}

STRICT:
- Return JSON OBJECT
- Include these keys: diagnosis, next_best_action, action_queue, agent_workflow, memory_updates, follow_up_questions
- diagnosis must be an object with keys: stage, bottleneck, confidence, reason
- stage must be one of: Idea, Validation, Offer, Outreach, Delivery
- bottleneck must describe the main blocker in one short phrase
- confidence must be an integer from 1 to 10
- reason must explain the diagnosis briefly
- next_best_action must be one specific next action to take in the next 48 hours
- action_queue must be an array of EXACTLY 3 objects with keys: step, owner, success_signal
- owner must be one of: User, Coach, Agent
- agent_workflow must be an array of EXACTLY 3 objects with keys: agent, purpose, input_needed, output_expected
- memory_updates must be an array of EXACTLY 3 short memory items the app should remember
- follow_up_questions must be an array of EXACTLY 3 short questions
- Make the workflow practical and sequential, not generic
- Base the diagnosis on the actual project state, not a blank template
"""

    parsed = structured_json_completion(
        system_prompt="You are a practical multi-step project strategy agent. Return valid JSON only.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.7,
        max_tokens=1200,
        attempts=2
    )
    if parsed is not None:
        return parsed

    return {
        "diagnosis": {
            "stage": "Validation",
            "bottleneck": "No confirmed buyer signal yet",
            "confidence": 7,
            "reason": "The project has direction, but the next highest-value move is still proving that a specific buyer wants this outcome."
        },
        "next_best_action": "Send one focused offer message to 5 target buyers and ask for a short feedback call this week.",
        "action_queue": [
            {
                "step": "Narrow the buyer to one exact niche and rewrite the offer in one sentence.",
                "owner": "User",
                "success_signal": "You can explain the offer in one line without broad terms like everyone or businesses."
            },
            {
                "step": "Turn the current project into one pilot offer with a price range and outcome promise.",
                "owner": "Agent",
                "success_signal": "A simple pilot offer exists with scope, price, and target result."
            },
            {
                "step": "Reach out to 5 prospects and collect objections, buying signals, or demo requests.",
                "owner": "User",
                "success_signal": "At least 2 real responses or 1 live conversation happens."
            }
        ],
        "agent_workflow": [
            {
                "agent": "Validation Agent",
                "purpose": "Tighten the niche and confirm the real pain point.",
                "input_needed": "Current buyer guess and pain point wording.",
                "output_expected": "A clearer target customer and 5 outreach questions."
            },
            {
                "agent": "Offer Agent",
                "purpose": "Convert the idea into a paid pilot or starter package.",
                "input_needed": "Problem statement, target niche, and likely outcome.",
                "output_expected": "A first offer, pricing range, and one-sentence positioning."
            },
            {
                "agent": "Outreach Agent",
                "purpose": "Create the messages needed to get first customer conversations.",
                "input_needed": "Offer summary and target buyer.",
                "output_expected": "Cold email, LinkedIn DM, and follow-up message."
            }
        ],
        "memory_updates": [
            f"Primary project is {compact_project.get('title') or 'the selected idea'}.",
            "The next milestone is buyer validation before feature expansion.",
            "The app should remember the most responsive niche and strongest objection."
        ],
        "follow_up_questions": [
            "Which exact buyer do you want to target first?",
            "What painful workflow happens repeatedly for that buyer?",
            "Would you rather optimize for faster income or stronger long-term scale first?"
        ]
    }


def generate_rag_advisor(project, knowledge_items, history_text, question, rerun_count=0, previous_result=None):
    compact_project = {
        "title": project.get("title", ""),
        "description": project.get("description", ""),
        "who_is_this_for": project.get("who_is_this_for", ""),
        "why_it_works": project.get("why_it_works", ""),
        "current_goal": project.get("current_goal", ""),
        "validation_summary": (project.get("validation_toolkit") or {}).get("validation_summary", ""),
        "execution_summary": (project.get("execution_plan") or {}).get("summary", ""),
        "first_offer": (project.get("launch_agent") or {}).get("first_offer", ""),
        "pricing_offer": ((project.get("first_customer_pack") or {}).get("pricing_offer") or {}).get("price_range", "")
    }
    retrieval_query = " ".join(
        part for part in [
            question,
            compact_project.get("title"),
            compact_project.get("current_goal"),
            compact_project.get("who_is_this_for")
        ]
        if part
    ).strip()
    retrieval = retrieve_knowledge_context(retrieval_query, knowledge_items or [], limit=4)
    context_text = retrieval.get("context_text", "").strip()
    citations = retrieval.get("citations", [])

    if not context_text:
        return {
            "answer": "I could not find relevant knowledge notes for this project yet. Add notes about the customer, workflow, constraints, or internal process, then run the Knowledge Advisor again.",
            "retrieval_summary": "No relevant knowledge chunks matched the current project question.",
            "cited_notes": [],
            "recommended_actions": [
                "Add a note that describes the target customer and their pain point.",
                "Add a note with any internal process, technical constraint, or prior learning.",
                "Run the Knowledge Advisor again with a more specific question."
            ],
            "knowledge_gaps": [
                "Target customer detail",
                "Workflow or process detail",
                "Constraint or assumption detail"
            ]
        }

    prompt = f"""
You are a retrieval-grounded startup advisor.

Project context:
{json.dumps(compact_project, ensure_ascii=True)}

Conversation context:
{history_text}

{build_rerun_instruction("Knowledge Advisor", rerun_count, previous_result, "Stay grounded in the same cited notes, but answer from a fresh perspective and avoid repeating the same recommended actions in the same order.")}

User question:
{question}

Retrieved knowledge context:
{context_text}

STRICT:
- Return JSON OBJECT only
- Include these keys: answer, retrieval_summary, cited_notes, recommended_actions, knowledge_gaps
- answer must be 2 to 4 sentences, practical, and clearly grounded in the retrieved notes
- retrieval_summary must explain what the notes mainly say in one short sentence
- cited_notes must be an array with 1 to 4 note labels taken from the retrieved context
- recommended_actions must be an array of EXACTLY 3 next steps
- knowledge_gaps must be an array of EXACTLY 3 missing facts or assumptions the user should clarify
- Do not invent facts that are not supported by the retrieved notes
- If the notes are incomplete, say so briefly in the answer
"""
    parsed = structured_json_completion(
        system_prompt="You answer using provided retrieval context and return valid JSON only.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.4,
        max_tokens=900,
        attempts=2
    )
    fallback = {
        "answer": "Based on the saved notes, the project should move forward with a narrow scope and a clearer validation step before expanding. The current knowledge suggests the best next move is to translate those notes into one concrete customer conversation or pilot.",
        "retrieval_summary": "The saved notes point toward a focused niche, a small first deliverable, and a need for validation before expansion.",
        "cited_notes": citations[:4],
        "recommended_actions": [
            "Turn the strongest retrieved note into one validation conversation this week.",
            "Use the saved constraints to define the smallest first deliverable.",
            "Capture one more note after the next customer or stakeholder interaction."
        ],
        "knowledge_gaps": [
            "Proof that the customer will pay",
            "Evidence that the stated pain is urgent",
            "Specific constraints that could block delivery"
        ]
    }
    if not parsed:
        return fallback

    parsed["answer"] = parsed.get("answer") or fallback["answer"]
    parsed["retrieval_summary"] = parsed.get("retrieval_summary") or fallback["retrieval_summary"]
    parsed["cited_notes"] = [item for item in parsed.get("cited_notes", []) if item][:4] or citations[:4]
    parsed["recommended_actions"] = [item for item in (parsed.get("recommended_actions") or []) if item][:3] or fallback["recommended_actions"]
    parsed["knowledge_gaps"] = [item for item in (parsed.get("knowledge_gaps") or []) if item][:3] or fallback["knowledge_gaps"]
    return parsed


def generate_recommendation_engine(project, analytics_summary, history_text, rerun_count=0, previous_result=None):
    progress = project.get("progress", {}) or {}
    linked_tickets = project.get("support_tickets", []) or []
    validation_score = (((project.get("validation_toolkit") or {}).get("validation_score") or {}).get("score")) or 0
    progress_done = sum(1 for value in progress.values() if value)
    total_progress = max(len(progress), 1)
    knowledge_notes = int((analytics_summary or {}).get("knowledge_notes", 0) or 0)
    timeline_events = int((analytics_summary or {}).get("timeline_events", 0) or 0)
    coach_turns = int((analytics_summary or {}).get("coach_turns", 0) or 0)
    usage_signals = int((analytics_summary or {}).get("usage_signals", 0) or 0)
    open_ticket_count = int((analytics_summary or {}).get("open_ticket_count", 0) or 0)

    readiness_score = min(
        100,
        progress_done * 11
        + min(validation_score * 4, 32)
        + (12 if project.get("execution_plan") else 0)
        + (10 if project.get("launch_agent") else 0)
        + (10 if project.get("first_customer_pack") else 0)
        + min(knowledge_notes * 4, 12)
    )
    momentum_score = min(
        100,
        progress_done * 9
        + min(timeline_events * 3, 24)
        + min(coach_turns * 4, 16)
        + min(usage_signals * 2, 14)
        - min(open_ticket_count * 5, 15)
    )

    if not progress.get("knowledge_advisor_completed") and knowledge_notes > 0:
        recommended_workflow = "Knowledge Advisor"
        next_best_action = "Run the Knowledge Advisor to turn your saved notes into a grounded next-step recommendation."
    elif not progress.get("validation_completed"):
        recommended_workflow = "Validation Toolkit"
        next_best_action = "Run the Validation Toolkit to confirm buyer pain, red flags, and validation questions before expanding."
    elif not progress.get("execution_plan_opened"):
        recommended_workflow = "Execution Plan"
        next_best_action = "Open the execution plan so the project has a concrete step-by-step path instead of only a concept."
    elif not project.get("launch_agent"):
        recommended_workflow = "Launch Agent"
        next_best_action = "Run the Launch Agent to define the niche, offer, landing copy, and first 7-day launch path."
    elif not progress.get("first_customer_pack_ready"):
        recommended_workflow = "First Customer Pack"
        next_best_action = "Generate the First Customer Pack so you have real messages and an offer ready to send."
    else:
        recommended_workflow = "AI Coach"
        next_best_action = "Use AI Coach to refine objections, outreach, and the next experiment based on everything completed so far."

    blockers = []
    if validation_score and validation_score < 6:
        blockers.append("Validation score is still weak, so demand risk remains high.")
    if not knowledge_notes:
        blockers.append("No saved knowledge notes yet, so answers are less grounded in your own context.")
    if not project.get("first_customer_pack"):
        blockers.append("Outreach assets are missing, which slows down first-customer conversations.")
    if open_ticket_count:
        blockers.append("There are unresolved support or workflow tickets that may block execution.")
    if not blockers:
        blockers.append("Main blocker is consistency: the project needs repeated customer-facing action this week.")

    focus_rotation = [
        {
            "focus": "customer evidence",
            "workflow": "Validation Toolkit",
            "reason": "Pressure-test the buyer pain and proof signals so the next move is backed by real evidence.",
            "blocker": "Customer evidence is still too thin to confidently prioritize the next move."
        },
        {
            "focus": "knowledge depth",
            "workflow": "Knowledge Advisor",
            "reason": "Use your saved notes to tighten the next step around internal constraints and real user observations.",
            "blocker": "The project still needs more grounded internal context before the next workflow is fully reliable."
        },
        {
            "focus": "offer clarity",
            "workflow": "Launch Agent",
            "reason": "Sharpen the offer and target angle so the next customer-facing step feels more concrete.",
            "blocker": "Offer positioning still needs more clarity before outreach can feel compelling."
        },
        {
            "focus": "outreach speed",
            "workflow": "First Customer Pack",
            "reason": "Convert the current project state into ready-to-send outreach assets so momentum turns into conversations.",
            "blocker": "Execution is slowing because the project still lacks fast customer-facing outreach motion."
        }
    ]
    rerun_focus = focus_rotation[rerun_count % len(focus_rotation)] if rerun_count > 0 else None
    if rerun_focus:
        next_best_action = f"{next_best_action} This pass, focus especially on {rerun_focus['focus']}."
        blockers = [rerun_focus["blocker"]] + [item for item in blockers if item != rerun_focus["blocker"]]

    top_recommendations = [
        {
            "workflow": recommended_workflow,
            "reason": next_best_action,
            "confidence": 9 if readiness_score >= 45 else 8
        },
        {
            "workflow": "Knowledge Base" if knowledge_notes < 3 else "My Projects",
            "reason": "Strengthen reusable context so later agents and advisors can work with more specific evidence.",
            "confidence": 7
        },
        {
            "workflow": "ServiceNow Ticket Review" if open_ticket_count else "Project Timeline Review",
            "reason": "Review blockers or recent activity so the next experiment reflects the current project state.",
            "confidence": 6
        }
    ]
    if rerun_focus:
        top_recommendations[1] = {
            "workflow": rerun_focus["workflow"],
            "reason": rerun_focus["reason"],
            "confidence": 7
        }
        top_recommendations[2] = {
            "workflow": "Project Timeline Review" if rerun_focus["workflow"] != "Project Timeline Review" else "AI Coach",
            "reason": f"Review the latest project signals with a fresh lens on {rerun_focus['focus']} before choosing the next experiment.",
            "confidence": 6
        }

    return {
        "readiness_score": readiness_score,
        "momentum_score": max(momentum_score, 0),
        "project_stage": "Ready to sell" if readiness_score >= 75 else "Launch prep" if readiness_score >= 55 else "Validation" if readiness_score >= 35 else "Early discovery",
        "recommended_workflow": recommended_workflow,
        "next_best_action": next_best_action,
        "top_recommendations": top_recommendations,
        "blockers": blockers[:3],
        "feature_snapshot": {
            "progress_completed": progress_done,
            "progress_total": total_progress,
            "validation_score": validation_score,
            "knowledge_notes": knowledge_notes,
            "timeline_events": timeline_events,
            "linked_ticket_count": len(linked_tickets),
            "open_ticket_count": open_ticket_count
        }
    }


def generate_project_evaluation(project, analytics_summary, history_text, rerun_count=0, previous_result=None):
    compact_project = {
        "title": project.get("title", ""),
        "description": project.get("description", ""),
        "who_is_this_for": project.get("who_is_this_for", ""),
        "why_it_works": project.get("why_it_works", ""),
        "current_goal": project.get("current_goal", ""),
        "progress": project.get("progress", {}),
        "execution_plan_summary": (project.get("execution_plan") or {}).get("summary", ""),
        "validation_summary": (project.get("validation_toolkit") or {}).get("validation_summary", ""),
        "launch_offer": (project.get("launch_agent") or {}).get("first_offer", ""),
        "knowledge_summary": (project.get("rag_advisor") or {}).get("retrieval_summary", ""),
        "recommendation_workflow": (project.get("recommendation_engine") or {}).get("recommended_workflow", ""),
        "timeline_events": len(project.get("timeline", []) or [])
    }
    compact_analytics = analytics_summary or {}
    prompt = f"""
You are an AI project evaluator and observability reviewer.

Project:
{json.dumps(compact_project, ensure_ascii=True)}

Analytics summary:
{json.dumps(compact_analytics, ensure_ascii=True)}

Conversation context:
{history_text}

{build_rerun_instruction("Project Evaluation", rerun_count, previous_result, "Keep the same evidence base, but vary the evaluation lens, blind-spot emphasis, and monitoring recommendations.")}

STRICT:
- Return JSON OBJECT only
- Include these keys: evaluation_summary, output_quality_score, evidence_coverage, consistency_checks, risk_flags, blind_spots, monitoring_recommendations
- evaluation_summary must be 2 sentences max
- output_quality_score must be an object with keys: score, label, reason
- score must be an integer from 1 to 10
- label must be one of: Strong, Moderate, Weak
- evidence_coverage must be an array of EXACTLY 3 short bullets about what evidence is present or missing
- consistency_checks must be an array of EXACTLY 3 short checks across project signals, with each item written as pass or caution style text
- risk_flags must be an array of EXACTLY 3 short risks
- blind_spots must be an array of EXACTLY 3 missing perspectives or data points
- monitoring_recommendations must be an array of EXACTLY 3 concrete things to track next
- Base the evaluation on the provided project data and analytics only
"""
    parsed = structured_json_completion(
        system_prompt="You evaluate project quality and observability. Return valid JSON only.",
        user_prompt=prompt,
        fallback=None,
        expected_type="dict",
        temperature=0.35,
        max_tokens=1100,
        attempts=2
    )
    fallback = {
        "evaluation_summary": "The project has useful execution structure, but it still needs stronger proof that users will engage and pay. Observability is improving, though customer evidence is still the weakest area.",
        "output_quality_score": {
            "score": 7,
            "label": "Moderate",
            "reason": "The workspace has multiple useful outputs, but customer evidence and monitoring still need to mature."
        },
        "evidence_coverage": [
            "Project structure and workflow outputs are present.",
            "Customer validation evidence is only partially covered.",
            "Operational notes and timeline data exist but are still light."
        ],
        "consistency_checks": [
            "Pass: Project has a clear title, goal, and reusable workspace state.",
            "Caution: Validation and launch artifacts may not yet align with actual buyer feedback.",
            "Caution: Monitoring signals are present, but they are not yet rich enough for confident prioritization."
        ],
        "risk_flags": [
            "The project may be overbuilt before demand is proven.",
            "Important buyer objections may still be missing from the knowledge base.",
            "Execution momentum could drop if the next step is not customer-facing."
        ],
        "blind_spots": [
            "Direct proof that a buyer will pay",
            "Measured response rate from outreach",
            "Evidence about which niche responds fastest"
        ],
        "monitoring_recommendations": [
            "Track one validation or outreach outcome each week.",
            "Track which workflow is used most before the next milestone.",
            "Track the strongest buyer objection and whether it changes over time."
        ]
    }
    return parsed or fallback


def build_servicenow_payload(project):
    title = project.get("title", "Income idea")
    description = project.get("description", "")
    audience = project.get("who_is_this_for", "")
    why_it_works = project.get("why_it_works", "")
    current_goal = project.get("current_goal", "")
    niche = project.get("niche", "")
    first_offer = project.get("first_offer", "")
    launch_plan = project.get("launch_plan", [])

    launch_plan_text = ""
    if isinstance(launch_plan, list) and launch_plan:
        launch_plan_text = "\n".join([f"- {item}" for item in launch_plan])

    full_description = "\n\n".join(
        part for part in [
            f"Idea: {title}",
            description,
            f"Audience: {audience}" if audience else "",
            f"Why it works: {why_it_works}" if why_it_works else "",
            f"Current goal: {current_goal}" if current_goal else "",
            f"Niche: {niche}" if niche else "",
            f"First offer: {first_offer}" if first_offer else "",
            f"Launch plan:\n{launch_plan_text}" if launch_plan_text else "",
            "Created from Skill2Income Active Project"
        ] if part
    )

    return {
        "short_description": f"Skill2Income: {title}",
        "description": full_description,
        "comments": "Imported from Skill2Income Active Project",
        "category": "inquiry"
    }


def create_servicenow_record(table_name, payload):
    if not SERVICENOW_INSTANCE or not SERVICENOW_USERNAME or not SERVICENOW_PASSWORD:
        raise ValueError("ServiceNow credentials are not configured on the backend.")

    table = table_name or "incident"
    url = f"{SERVICENOW_INSTANCE}/api/now/table/{table}"

    credentials = f"{SERVICENOW_USERNAME}:{SERVICENOW_PASSWORD}".encode("utf-8")
    auth_header = base64.b64encode(credentials).decode("utf-8")

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            response_body = response.read().decode("utf-8")
            parsed = json.loads(response_body)
            return parsed.get("result", {})
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"ServiceNow returned {exc.code}: {detail or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach ServiceNow: {exc.reason}") from exc


def fetch_servicenow_record_by_number(table_name, record_number):
    if not SERVICENOW_INSTANCE or not SERVICENOW_USERNAME or not SERVICENOW_PASSWORD:
        raise ValueError("ServiceNow credentials are not configured on the backend.")

    table = table_name or "incident"
    encoded_number = quote(f"number={record_number}", safe="=&")
    url = f"{SERVICENOW_INSTANCE}/api/now/table/{table}?sysparm_limit=1&sysparm_query={encoded_number}&sysparm_display_value=all"

    credentials = f"{SERVICENOW_USERNAME}:{SERVICENOW_PASSWORD}".encode("utf-8")
    auth_header = base64.b64encode(credentials).decode("utf-8")

    req = request.Request(
        url,
        headers={
            "Authorization": f"Basic {auth_header}",
            "Accept": "application/json"
        },
        method="GET"
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            response_body = response.read().decode("utf-8")
            parsed = json.loads(response_body)
            results = parsed.get("result", []) or []
            return results[0] if results else None
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"ServiceNow returned {exc.code}: {detail or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach ServiceNow: {exc.reason}") from exc


def build_servicenow_record_url(table_name, sys_id):
    if not SERVICENOW_INSTANCE or not table_name or not sys_id:
        return ""

    target = quote(f"{table_name}.do?sys_id={sys_id}", safe="")
    return f"{SERVICENOW_INSTANCE}/nav_to.do?uri={target}"


def get_servicenow_field_value(record, field_name):
    value = record.get(field_name, "")
    if isinstance(value, dict):
        return value.get("value", "") or value.get("display_value", "")
    return value


def get_servicenow_field_display(record, field_name):
    value = record.get(field_name, "")
    if isinstance(value, dict):
        return value.get("display_value", "") or value.get("value", "")
    return value


def get_servicenow_status_label(record, table_name="incident"):
    display = str(get_servicenow_field_display(record, "state") or "").strip()
    if display and not display.isdigit():
        return display

    raw_value = str(get_servicenow_field_value(record, "state") or "").strip()
    if table_name == "incident":
        return SERVICENOW_INCIDENT_STATE_MAP.get(raw_value, raw_value or "Unknown")

    return raw_value or "Unknown"


def build_issue_payload(issue):
    title = issue.get("title", "Skill2Income issue")
    description = issue.get("description", "")
    severity = issue.get("severity", "Medium")
    feature = issue.get("feature", "")
    user_email = issue.get("user_email", "")
    chat_id = issue.get("chat_id", "")
    active_project = issue.get("active_project", "")

    full_description = "\n\n".join(
        part for part in [
            f"Issue: {title}",
            f"Severity: {severity}",
            f"Feature: {feature}" if feature else "",
            f"User Email: {user_email}" if user_email else "",
            f"Chat ID: {chat_id}" if chat_id else "",
            f"Active Project: {active_project}" if active_project else "",
            description,
            "Raised from Skill2Income support form"
        ] if part
    )

    impact_map = {"Low": "3", "Medium": "2", "High": "1"}
    urgency_map = {"Low": "3", "Medium": "2", "High": "1"}

    return {
        "short_description": f"Skill2Income Issue: {title}",
        "description": full_description,
        "comments": "User raised an issue from the Skill2Income app.",
        "category": "inquiry",
        "impact": impact_map.get(severity, "2"),
        "urgency": urgency_map.get(severity, "2")
    }


# =========================
# MAIN API
# =========================

@app.post("/generate-income-plan")
async def generate_income_plan(data: dict):

    skills = data.get("skills","")
    interests = data.get("interests", "")
    request_type = data.get("request_type", "").strip().lower()
    plan_focus = data.get("plan_focus", "").strip()
    idea_title = data.get("idea_title", "").strip()
    idea_description = data.get("idea_description", "").strip()
    history = data.get("history",[])
    previous_ideas = data.get("previous_ideas", []) or []
    history_text = build_history_summary(history)
    previous_titles, previous_models = extract_previous_ideas(history, previous_ideas)
    focus = plan_focus or infer_focus(interests or skills, history)
    normalized_request = normalize_text(interests or skills)
    repeat_count = sum(
        1
        for message in get_user_messages(history)
        if normalize_text(message) == normalized_request
    )

    rerun_count = int(data.get("rerun_count", 0) or 0)
    previous_result = data.get("previous_result")

    if request_type == "execution_plan":
        return {"result": generate_execution_plan(idea_title or focus, idea_description, history_text)}

    if request_type == "launch_agent":
        return {"result": generate_launch_agent_plan(idea_title or focus, idea_description, history_text, rerun_count=rerun_count, previous_result=previous_result)}

    if request_type == "validation_toolkit":
        return {"result": generate_validation_toolkit(idea_title or focus, idea_description, history_text, rerun_count=rerun_count, previous_result=previous_result)}

    if request_type == "first_customer_pack":
        return {"result": generate_first_customer_pack(idea_title or focus, idea_description, history_text, rerun_count=rerun_count, previous_result=previous_result)}

    if request_type == "idea_comparison":
        ideas_to_compare = data.get("ideas", []) or []
        return {"result": generate_idea_comparison(ideas_to_compare, history_text)}

    if request_type == "plan" or is_plan_request(interests or skills):
        return {"result": generate_plan(focus, history_text)}

    max_attempts = 4
    attempt = 0
    ideas = []
    attempt_banned_titles = set(previous_titles)
    attempt_banned_models = set(previous_models)

    while attempt < max_attempts:

        raw = generate_ideas(
            focus,
            history_text,
            attempt_banned_titles,
            attempt_banned_models,
            repeat_count=repeat_count + attempt
        )

        for idea in raw:
            if not isinstance(idea, dict):
                continue
            title_key = normalize_text(idea.get("title", ""))
            model_key = normalize_text(idea.get("business_model", ""))
            if title_key:
                attempt_banned_titles.add(title_key)
            if model_key:
                attempt_banned_models.add(model_key)

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
        ideas = build_fallback_ideas(focus, attempt_banned_titles, batch_offset=repeat_count)

    # best option
    best = max(ideas, key=lambda x: x.get("score", 0))
    for idea in ideas:
        idea["best_option"] = (idea == best)

    return {"result": ideas}


@app.post("/ai-coach")
async def ai_coach(data: dict):
    idea_title = data.get("idea_title", "").strip()
    idea_description = data.get("idea_description", "").strip()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)

    if not idea_title:
        return {
            "result": {
                "reply": "Choose an idea first, then I can help refine it into a practical path.",
                "suggested_replies": [
                    "Help me pick the best idea",
                    "Give me a beginner version",
                    "How do I validate demand?"
                ],
                "next_steps": [
                    "Pick one idea card to coach.",
                    "Share the version you like most.",
                    "Ask for refinement, validation, or growth."
                ]
            }
        }

    return {
        "result": generate_ai_coach_reply(
            idea_title=idea_title,
            idea_description=idea_description,
            user_message=user_message or "Help me move forward with this idea.",
            history_text=history_text
        )
    }


@app.post("/launch-agent")
async def launch_agent(data: dict):
    idea_title = data.get("idea_title", "").strip()
    idea_description = data.get("idea_description", "").strip()
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)
    rerun_count = int(data.get("rerun_count", 0) or 0)
    previous_result = data.get("previous_result")

    return {
        "result": generate_launch_agent_plan(
            idea_title=idea_title or "the selected idea",
            idea_description=idea_description,
            history_text=history_text,
            rerun_count=rerun_count,
            previous_result=previous_result
        )
    }


@app.post("/validation-toolkit")
async def validation_toolkit(data: dict):
    idea_title = data.get("idea_title", "").strip()
    idea_description = data.get("idea_description", "").strip()
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)
    rerun_count = int(data.get("rerun_count", 0) or 0)
    previous_result = data.get("previous_result")

    return {
        "result": generate_validation_toolkit(
            idea_title=idea_title or "the selected idea",
            idea_description=idea_description,
            history_text=history_text,
            rerun_count=rerun_count,
            previous_result=previous_result
        )
    }


@app.post("/first-customer-pack")
async def first_customer_pack(data: dict):
    idea_title = data.get("idea_title", "").strip()
    idea_description = data.get("idea_description", "").strip()
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)
    rerun_count = int(data.get("rerun_count", 0) or 0)
    previous_result = data.get("previous_result")

    return {
        "result": generate_first_customer_pack(
            idea_title=idea_title or "the selected idea",
            idea_description=idea_description,
            history_text=history_text,
            rerun_count=rerun_count,
            previous_result=previous_result
        )
    }


@app.post("/idea-comparison")
async def idea_comparison(data: dict):
    ideas = data.get("ideas", []) or []
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)

    return {
        "result": generate_idea_comparison(ideas, history_text)
    }


@app.post("/project-strategy-agent")
async def project_strategy_agent(data: dict):
    project = data.get("project", {}) or {}
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)
    rerun_count = int(data.get("rerun_count", 0) or 0)
    previous_result = data.get("previous_result")

    if not isinstance(project, dict) or not project.get("title"):
        return {"error": "Project data is required."}

    return {
        "result": generate_project_strategy_agent(project, history_text, rerun_count=rerun_count, previous_result=previous_result)
    }


@app.post("/rag-advisor")
async def rag_advisor(data: dict):
    project = data.get("project", {}) or {}
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)
    question = (data.get("question") or "").strip() or "What should I do next based on my saved knowledge notes?"
    knowledge_items = [item for item in (data.get("knowledge_items", []) or []) if isinstance(item, dict)]
    rerun_count = int(data.get("rerun_count", 0) or 0)
    previous_result = data.get("previous_result")

    if not isinstance(project, dict) or not project.get("title"):
        return {"error": "Project data is required."}

    return {
        "result": generate_rag_advisor(project, knowledge_items, history_text, question, rerun_count=rerun_count, previous_result=previous_result)
    }


@app.post("/recommendation-engine")
async def recommendation_engine(data: dict):
    project = data.get("project", {}) or {}
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)
    analytics_summary = data.get("analytics_summary", {}) or {}
    rerun_count = int(data.get("rerun_count", 0) or 0)
    previous_result = data.get("previous_result")

    if not isinstance(project, dict) or not project.get("title"):
        return {"error": "Project data is required."}

    return {
        "result": generate_recommendation_engine(project, analytics_summary, history_text, rerun_count=rerun_count, previous_result=previous_result)
    }


@app.post("/project-evaluation")
async def project_evaluation(data: dict):
    project = data.get("project", {}) or {}
    history = data.get("history", [])
    history_text = build_history_summary(history, limit=10)
    analytics_summary = data.get("analytics_summary", {}) or {}
    rerun_count = int(data.get("rerun_count", 0) or 0)
    previous_result = data.get("previous_result")

    if not isinstance(project, dict) or not project.get("title"):
        return {"error": "Project data is required."}

    return {
        "result": generate_project_evaluation(project, analytics_summary, history_text, rerun_count=rerun_count, previous_result=previous_result)
    }


@app.post("/servicenow/export")
async def servicenow_export(data: dict):
    project = data.get("project", {}) or {}
    table_name = (data.get("table_name") or "incident").strip()

    if not isinstance(project, dict) or not project.get("title"):
        return {"error": "Project data is required."}

    try:
        payload = build_servicenow_payload(project)
        record = create_servicenow_record(table_name, payload)
        sys_id = record.get("sys_id")

        return {
            "result": {
                "table": table_name,
                "sys_id": sys_id,
                "number": get_servicenow_field_value(record, "number"),
                "display": get_servicenow_field_display(record, "short_description") or project.get("title", ""),
                "url": build_servicenow_record_url(table_name, sys_id),
                "status": get_servicenow_status_label(record, table_name) or "Submitted"
            }
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/servicenow/report-issue")
async def servicenow_report_issue(data: dict):
    issue = data.get("issue", {}) or {}
    table_name = (data.get("table_name") or "incident").strip()

    if not isinstance(issue, dict) or not issue.get("title"):
        return {"error": "Issue title is required."}

    try:
        payload = build_issue_payload(issue)
        record = create_servicenow_record(table_name, payload)
        sys_id = record.get("sys_id")

        return {
            "result": {
                "table": table_name,
                "sys_id": sys_id,
                "number": get_servicenow_field_value(record, "number"),
                "display": get_servicenow_field_display(record, "short_description") or issue.get("title", ""),
                "url": build_servicenow_record_url(table_name, sys_id),
                "status": get_servicenow_status_label(record, table_name) or "Submitted"
            }
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/servicenow/ticket-status")
async def servicenow_ticket_status(data: dict):
    ticket_number = (data.get("ticket_number") or "").strip().upper()
    table_name = (data.get("table_name") or "incident").strip()

    if not ticket_number:
        return {"error": "Ticket number is required."}

    try:
        record = fetch_servicenow_record_by_number(table_name, ticket_number)
        if not record:
            return {"error": f"Ticket {ticket_number} was not found."}

        return {
            "result": {
                "table": table_name,
                "number": get_servicenow_field_value(record, "number") or ticket_number,
                "status": get_servicenow_status_label(record, table_name),
                "display": get_servicenow_field_display(record, "short_description"),
                "sys_id": get_servicenow_field_value(record, "sys_id"),
                "updated_at": get_servicenow_field_display(record, "sys_updated_on") or get_servicenow_field_value(record, "sys_updated_on"),
                "created_at": get_servicenow_field_display(record, "sys_created_on") or get_servicenow_field_value(record, "sys_created_on"),
                "priority": get_servicenow_field_display(record, "priority") or get_servicenow_field_value(record, "priority")
            }
        }
    except Exception as exc:
        return {"error": str(exc)}
