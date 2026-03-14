"""
NCA Food Access — Community Needs Triage & Response Assistant
GBA 479 — Food Access Department
Northbridge Community Alliance

Architecture:
  - Intake loader     (Python/deterministic): reads partner intake reports CSV
  - Policy retriever  (Python/deterministic): keyword-scored FAQ chunks
  - Classifier        (LLM/Claude Haiku):     triage — category + urgency + escalation flag
  - Response generator(LLM/Claude Opus):      policy-grounded partner-facing reply
  - Validator         (Python/deterministic): checks policy compliance in response

Triage categories:
  ELIGIBILITY   — who can access, documentation, frequency
  SCHEDULING    — hours, proxy pickup, delivery
  DIETARY       — accommodations, substitutions
  BULK_EVENT    — bulk/event requests
  INVENTORY     — stock levels, high-demand items
  DONATION      — surplus donation intake
  SNAP_BENEFITS — benefits gap, SNAP interruption
  CAPACITY      — volume/demand forecasting
  OUT_OF_SCOPE  — not food access related

Urgency levels:
  ROUTINE    — no immediate need, informational
  MODERATE   — need likely within 1–2 weeks
  URGENT     — immediate or imminent food gap
"""

import os
import json
import re
import csv
import pandas as pd
from typing import Optional
from anthropic import Anthropic

# ─────────────────────────────────────────────
# POLICY KNOWLEDGE BASE  (from FAQ PDF)
# ─────────────────────────────────────────────

POLICY_CHUNKS = [
    {
        "section": "Eligibility",
        "id": "eligibility",
        "text": (
            "Households residing in Northbridge and surrounding zip codes (14621, 14609, 14613) are eligible. "
            "Proof of residency is requested but not required on first visit. Self-attestation is permitted for initial intake. "
            "Standard frequency: once per calendar month. "
            "Exception: biweekly visits permitted for households with 5+ members or documented SNAP interruption. "
            "Emergency override: staff discretion, requires supervisor approval."
        ),
        "keywords": ["eligible", "eligibility", "who can", "qualify", "documentation", "residency", "proof",
                     "frequency", "how often", "visit limit", "monthly", "household"],
    },
    {
        "section": "Hours & Pickup",
        "id": "hours",
        "text": (
            "Operating hours: Tuesday 9:00am–1:00pm, Thursday 12:00pm–6:00pm, Saturday 10:00am–2:00pm. "
            "Evening pickup: Thursday extends to 6:00pm. "
            "Proxy pickup: allowed with written note or text confirmation, household name and address, and proxy ID."
        ),
        "keywords": ["hours", "open", "schedule", "pickup", "proxy", "evening", "tuesday", "thursday",
                     "saturday", "when", "timing", "extended hours", "work schedule"],
    },
    {
        "section": "Delivery",
        "id": "delivery",
        "text": (
            "Delivery is limited and available only for households with verified mobility constraints, "
            "medically homebound individuals, or temporary housing placements without transportation. "
            "Delivery requests must be submitted at least 48 hours in advance. Same-day delivery is not available."
        ),
        "keywords": ["delivery", "transport", "mobility", "homebound", "cannot travel", "transportation",
                     "pick up", "come in", "bring", "drop off"],
    },
    {
        "section": "Dietary Accommodations",
        "id": "dietary",
        "text": (
            "The pantry cannot guarantee strict compliance with religious or medical dietary standards. "
            "Staff may provide substitutions when available. "
            "Nut-free requests are prioritized when documented. "
            "Lower-sodium and higher-protein substitutions may be accommodated depending on inventory. "
            "All dietary accommodations are subject to current stock levels."
        ),
        "keywords": ["diet", "dietary", "halal", "kosher", "religious", "allergy", "nut-free", "sodium",
                     "protein", "medical", "nutrition", "food restriction", "culturally", "accommodate"],
    },
    {
        "section": "Inventory & Substitution",
        "id": "inventory",
        "text": (
            "When inventory falls below 25% of par level, item limits may be temporarily reduced, "
            "substitutions may be provided, and partner organizations may be notified. "
            "Current high-demand categories (February 2026): infant formula, peanut butter, "
            "diapers (sizes 4–6), shelf-stable protein. "
            "Inventory status should be confirmed with warehouse dashboard before communicating guarantees."
        ),
        "keywords": ["inventory", "stock", "available", "shortage", "formula", "protein", "diapers",
                     "peanut butter", "item", "supply", "levels", "high-demand", "limited"],
    },
    {
        "section": "Bulk & Event Requests",
        "id": "bulk",
        "text": (
            "Bulk requests must be submitted at least 7 days in advance. "
            "Maximum bulk allocation: 50 households equivalent per month per organization. "
            "Shelf-stable items only. Dependent on surplus inventory. "
            "Bulk support may be denied during periods of low stock."
        ),
        "keywords": ["bulk", "event", "dinner", "large", "group", "community", "quantity", "donation",
                     "how much", "allocation", "50 household", "advance"],
    },
    {
        "section": "Temporary Housing & Limited Kitchens",
        "id": "temp_housing",
        "text": (
            "Microwave only: provide ready-to-heat meals, rice pouches, canned meals with pop-tops. "
            "Mini-fridge only: provide smaller volume, high-rotation items. "
            "No can opener: avoid standard cans unless opener provided. "
            "Package customization must be noted in intake record."
        ),
        "keywords": ["temporary housing", "transitional", "mini-fridge", "microwave", "no kitchen",
                     "limited", "housing", "shelter", "can opener", "small", "package"],
    },
    {
        "section": "Donation Intake Standards",
        "id": "donation",
        "text": (
            "Non-expired, sealed items only. At least 3 months before expiration (preferred). "
            "No damaged packaging. No home-canned goods. "
            "Large donation drop-offs must be scheduled in advance."
        ),
        "keywords": ["donate", "donation", "surplus", "extra", "canned", "drop off", "give",
                     "contribute", "expiration", "expired", "sealed"],
    },
    {
        "section": "SNAP & Benefits Gaps",
        "id": "snap",
        "text": (
            "SNAP interruption qualifies for biweekly pantry visits. "
            "Documentation not required for first override. "
            "Staff should provide SNAP recertification assistance referral information."
        ),
        "keywords": ["snap", "benefits", "recertification", "gap", "interruption", "delayed",
                     "ebt", "food stamps", "government assistance", "interim"],
    },
    {
        "section": "Communication Standards",
        "id": "communication",
        "text": (
            "Avoid guaranteeing specific items unless inventory is confirmed. "
            "Use conditional language when appropriate. "
            "Clearly state hours and documentation requirements. "
            "Clarify if request requires supervisor approval."
        ),
        "keywords": ["communicate", "response", "guarantee", "confirm", "conditional", "language",
                     "policy", "standard", "approval", "supervisor"],
    },
]

STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
              "for", "of", "and", "or", "but", "we", "our", "it", "be", "have",
              "that", "this", "with", "as", "by", "from", "they", "their", "can",
              "would", "will", "not", "if", "has", "had", "do", "does", "i", "you"}


def retrieve_policy(query: str, top_k: int = 3) -> list[dict]:
    """Keyword-scored policy chunk retrieval — deterministic."""
    words = set(re.findall(r'\b\w+\b', query.lower())) - STOP_WORDS
    scored = []
    for chunk in POLICY_CHUNKS:
        kw_hits = sum(1 for kw in chunk["keywords"] if kw in query.lower())
        word_hits = sum(1 for w in words if any(w in kw for kw in chunk["keywords"]))
        score = kw_hits * 2 + word_hits
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


# ─────────────────────────────────────────────
# INTAKE REPORT LOADER  (deterministic)
# ─────────────────────────────────────────────

def load_intake_reports(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_report(df: pd.DataFrame, report_id: str) -> Optional[dict]:
    row = df[df["report_id"] == report_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def list_reports(df: pd.DataFrame) -> list[dict]:
    return df[["report_id", "source_organization", "timestamp"]].to_dict("records")


# ─────────────────────────────────────────────
# URGENCY CLASSIFIER  (deterministic pre-check)
# ─────────────────────────────────────────────

URGENT_SIGNALS = [
    "out of food", "no food", "immediate", "today", "tomorrow", "emergency",
    "crisis", "running out", "acute", "right now", "this week", "no longer have",
    "next week", "food gap", "nothing to eat",
]

ESCALATION_SIGNALS = [
    "supervisor", "override", "emergency override", "acute", "crisis",
    "medical emergency", "homebound", "imminent",
]


def pre_classify_urgency(narrative: str) -> dict:
    """Deterministic pre-check before LLM routing."""
    text = narrative.lower()
    urgent = any(sig in text for sig in URGENT_SIGNALS)
    escalate = any(sig in text for sig in ESCALATION_SIGNALS)
    return {
        "pre_urgent": urgent,
        "pre_escalate": escalate,
    }


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

TRIAGE_SYSTEM = """You are a triage classifier for the NCA Food Access program.
Classify partner intake reports into a structured triage result.

Categories: ELIGIBILITY, SCHEDULING, DIETARY, BULK_EVENT, INVENTORY,
            DONATION, SNAP_BENEFITS, CAPACITY, MULTI, OUT_OF_SCOPE

Urgency levels:
  ROUTINE  — informational, no immediate need
  MODERATE — need anticipated within 1–2 weeks
  URGENT   — immediate or imminent food gap this week

Return ONLY a JSON object (no markdown):
{
  "category": "<category>",
  "urgency": "<ROUTINE|MODERATE|URGENT>",
  "escalate": <true|false>,
  "households_affected": <number or null>,
  "key_needs": ["<need1>", "<need2>"],
  "reasoning": "<one sentence>"
}

Escalate=true if: acute food gap, emergency override needed, medical situation,
                  supervisor approval required by policy, or households imminently at risk.
"""

RESPONSE_SYSTEM = """You are the NCA Food Access Response Assistant.
You draft professional, policy-grounded replies to partner organization intake reports.

Rules:
- Address the partner's specific question(s) directly.
- Ground every policy statement in the retrieved FAQ sections provided.
- Use conditional language where policy requires it (e.g., "subject to availability").
- Use person-first language (e.g., "households experiencing food insecurity" not "food insecure households").
- Never guarantee specific items unless inventory is confirmed.
- If escalation is needed, include a clear escalation note.
- Format: brief acknowledgment → policy answers → next steps → closing.
- Keep under 220 words.
- End with: [NCA Food Access Team | Policy ref: <section IDs used>]
"""

OUT_OF_SCOPE_MSG = """This assistant is scoped to Food Access program operations — intake triage,
policy guidance, and partner-facing response drafting.

I can help with:
• Triaging and classifying incoming partner reports
• Drafting policy-grounded responses to partner questions
• Checking eligibility, hours, dietary accommodation, bulk request, and SNAP policies
• Flagging reports that need staff escalation

Please provide a partner intake report or a food access policy question.
"""


# ─────────────────────────────────────────────
# RESPONSE VALIDATOR  (deterministic)
# ─────────────────────────────────────────────

REQUIRED_POLICY_PHRASES = [
    "subject to", "depending on", "available", "staff", "confirm",
    "policy ref", "NCA Food Access",
]

PROHIBITED_GUARANTEES = [
    "we guarantee", "we will definitely", "always available", "always have",
    "will definitely provide",
]


def validate_response(response: str) -> tuple[bool, str]:
    """
    Deterministic checks:
    1. Contains at least one conditional/policy phrase (no false guarantees).
    2. Contains policy reference footer.
    3. Does not contain prohibited guarantee language.
    """
    text = response.lower()
    has_conditional = any(p in text for p in [p.lower() for p in REQUIRED_POLICY_PHRASES])
    has_footer = "policy ref" in text or "nca food access" in text
    has_prohibited = any(p in text for p in PROHIBITED_GUARANTEES)

    if has_prohibited:
        return False, "Response contains prohibited guarantee language."
    if not has_conditional:
        return False, "Response missing conditional/policy language."
    if not has_footer:
        return False, "Response missing policy reference footer."
    return True, "ok"


# ─────────────────────────────────────────────
# MAIN ASSISTANT
# ─────────────────────────────────────────────

def process_report(client: Anthropic, report: dict, df: pd.DataFrame) -> dict:
    """Full pipeline: triage → retrieve → generate → validate."""
    narrative = report["narrative"]
    org = report["source_organization"]
    report_id = report["report_id"]

    # ── STEP 1: Pre-classify urgency (Python — deterministic) ──
    pre = pre_classify_urgency(narrative)

    # ── STEP 2: Retrieve relevant policy chunks (Python — deterministic) ──
    policy_chunks = retrieve_policy(narrative, top_k=3)
    policy_context = "\n\n".join(
        f"[{c['section']}]\n{c['text']}" for c in policy_chunks
    )

    # ── STEP 3: Triage (LLM — Haiku) ──
    triage_resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system=TRIAGE_SYSTEM,
        messages=[{"role": "user", "content": f"Report from {org}:\n{narrative}"}],
    )
    triage = json.loads(triage_resp.content[0].text.strip())

    # Override urgency if deterministic pre-check caught something stronger
    if pre["pre_urgent"] and triage["urgency"] == "ROUTINE":
        triage["urgency"] = "MODERATE"
        triage["reasoning"] += " [Urgency upgraded by deterministic pre-check.]"
    if pre["pre_escalate"]:
        triage["escalate"] = True

    # ── STEP 4: Generate response (LLM — Opus) ──
    gen_prompt = (
        f"Intake report — {report_id} from {org}:\n{narrative}\n\n"
        f"Triage result: {json.dumps(triage)}\n\n"
        f"Relevant policy sections:\n{policy_context}\n\n"
        f"Draft a professional partner-facing response."
    )
    gen_resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        system=RESPONSE_SYSTEM,
        messages=[{"role": "user", "content": gen_prompt}],
    )
    response = gen_resp.content[0].text.strip()

    # ── STEP 5: Validate (Python — deterministic) ──
    passed, reason = validate_response(response)
    if not passed:
        # Regenerate once with explicit correction
        gen_resp2 = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            system=RESPONSE_SYSTEM,
            messages=[
                {"role": "user", "content": gen_prompt},
                {"role": "assistant", "content": response},
                {"role": "user", "content": (
                    f"Validation failed: {reason}. "
                    "Please revise: include conditional language (e.g., 'subject to availability'), "
                    "and end with [NCA Food Access Team | Policy ref: <sections>]."
                )},
            ],
        )
        response = gen_resp2.content[0].text.strip()
        passed, reason = validate_response(response)

    return {
        "report_id": report_id,
        "org": org,
        "triage": triage,
        "policy_sections_used": [c["id"] for c in policy_chunks],
        "response": response,
        "validation": {"passed": passed, "reason": reason},
    }


def display_result(result: dict):
    """Pretty-print triage + response."""
    t = result["triage"]
    urgency_icon = {"URGENT": "🔴", "MODERATE": "🟡", "ROUTINE": "🟢"}.get(t["urgency"], "⚪")
    escalate_tag = " ⚠️  ESCALATE" if t.get("escalate") else ""
    validation_tag = "✅" if result["validation"]["passed"] else f"⚠️  {result['validation']['reason']}"

    print(f"\n{'═'*62}")
    print(f"  {result['report_id']} — {result['org']}")
    print(f"{'═'*62}")
    print(f"  Category : {t['category']}")
    print(f"  Urgency  : {urgency_icon} {t['urgency']}{escalate_tag}")
    if t.get("households_affected"):
        print(f"  Affected : ~{t['households_affected']} households")
    print(f"  Needs    : {', '.join(t.get('key_needs', []))}")
    print(f"  Policy   : {', '.join(result['policy_sections_used'])}")
    print(f"  Valid    : {validation_tag}")
    print(f"\n  DRAFT RESPONSE:\n")
    for line in result["response"].split("\n"):
        print(f"  {line}")
    print()


def chat(df: pd.DataFrame):
    client = Anthropic()

    print("\n" + "═" * 62)
    print("  NCA FOOD ACCESS — TRIAGE & RESPONSE ASSISTANT")
    print("  Food Access Department")
    print(f"  {len(df)} intake reports loaded")
    print("═" * 62)
    print("  Commands:")
    print("  list              — show all pending reports")
    print("  process <id>      — triage + draft response for a report")
    print("  process all       — process all reports")
    print("  policy <question> — ask a food access policy question")
    print("  quit              — exit\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nAssistant: Goodbye.")
            break

        cmd = user_input.lower()

        # ── LIST ──
        if cmd == "list":
            reports = list_reports(df)
            print(f"\n  {len(reports)} intake reports:\n")
            for r in reports:
                print(f"  {r['report_id']} | {r['source_organization']:<40} | {r['timestamp']}")
            print()
            continue

        # ── PROCESS ALL ──
        if cmd == "process all":
            print(f"\n  Processing {len(df)} reports...\n")
            for _, row in df.iterrows():
                report = row.to_dict()
                try:
                    result = process_report(client, report, df)
                    display_result(result)
                except Exception as e:
                    print(f"  [Error processing {report['report_id']}: {e}]\n")
            continue

        # ── PROCESS <ID> ──
        if cmd.startswith("process "):
            report_id = user_input.split(" ", 1)[1].strip().upper()
            report = get_report(df, report_id)
            if not report:
                print(f"\n  Report '{report_id}' not found. Use 'list' to see available reports.\n")
                continue
            print(f"\n  Processing {report_id}...\n")
            try:
                result = process_report(client, report, df)
                display_result(result)
            except Exception as e:
                print(f"  [Error: {e}]\n")
            continue

        # ── POLICY QUESTION ──
        if cmd.startswith("policy "):
            question = user_input[7:].strip()
            chunks = retrieve_policy(question, top_k=3)
            if not chunks:
                print(f"\n  Assistant: {OUT_OF_SCOPE_MSG}\n")
                continue
            context = "\n\n".join(f"[{c['section']}]\n{c['text']}" for c in chunks)
            resp = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=300,
                system=(
                    "You are an NCA Food Access policy reference assistant. "
                    "Answer concisely based only on the provided policy sections. "
                    "Use conditional language where appropriate. "
                    "End with [Policy ref: <section IDs>]."
                ),
                messages=[{
                    "role": "user",
                    "content": f"Policy sections:\n{context}\n\nQuestion: {question}"
                }],
            )
            print(f"\n  Assistant: {resp.content[0].text.strip()}\n")
            continue

        # ── UNRECOGNISED ──
        print(f"\n  {OUT_OF_SCOPE_MSG}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = "northbridge_partner_intake_reports.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    try:
        df = load_intake_reports(csv_path)
    except FileNotFoundError:
        try:
            df = load_intake_reports("/mnt/user-data/uploads/northbridge_partner_intake_reports.csv")
        except FileNotFoundError:
            print("Error: Could not find intake reports CSV.")
            sys.exit(1)

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    chat(df)
