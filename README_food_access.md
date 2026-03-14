# NCA Food Access — Community Needs Triage & Response Assistant

**Food Access Department — Northbridge Community Alliance**  
GBA 479 Take Home Final · Simon Business School · 2026

---

## What It Does

A triage and response assistant for NCA Food Access program staff. Incoming partner intake reports are automatically classified by need category and urgency level. The system then drafts policy-grounded, partner-facing replies — with built-in prohibited guarantee detection to prevent staff from accidentally over-promising inventory availability.

---

## Architecture

```
Partner Intake Report (text narrative)
     │
     ▼
┌─────────────────────────────────┐
│  PRE-CLASSIFIER  [Python]       │  Keyword scan for urgent signals
│  pre_classify_urgency()         │  ("out of food", "crisis", "today")
└──────────────┬──────────────────┘  Upgrades urgency if LLM misses it
               │
               ▼
┌─────────────────────────────────┐
│  POLICY RETRIEVAL  [Python]     │  Keyword-scored retrieval across
│  retrieve_policy()              │  10 FAQ policy chunks → top-3
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  TRIAGE  [LLM — Haiku]          │  Category + urgency + escalation
│                                 │  ELIGIBILITY / SCHEDULING /
└──────────────┬──────────────────┘  DIETARY / BULK_EVENT / SNAP /
               │                     DONATION / CAPACITY / INVENTORY
               ▼
┌─────────────────────────────────┐
│  GENERATOR  [LLM — Opus]        │  Policy-grounded partner reply
│                                 │  using retrieved FAQ sections
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  VALIDATOR  [Python]            │  Blocks prohibited guarantees +
│  validate_response()            │  checks conditional language +
└─────────────────────────────────┘  requires policy reference footer
```

**Task decomposition:**
- **Python (deterministic):** Urgency pre-classification, policy chunk retrieval, prohibited language validation
- **LLM — Haiku:** Semantic triage — category and urgency classification
- **LLM — Opus:** Response generation grounded in retrieved policy

---

## Triage Categories

| Category | Triggers |
|----------|---------|
| `ELIGIBILITY` | Who can access, documentation, visit frequency |
| `SCHEDULING` | Hours, proxy pickup, delivery, evening access |
| `DIETARY` | Halal, nut-free, low-sodium, cultural accommodations |
| `BULK_EVENT` | Community event support, bulk item requests |
| `INVENTORY` | Stock levels, high-demand items, availability |
| `DONATION` | Surplus donation intake standards |
| `SNAP_BENEFITS` | Benefits gaps, SNAP interruption, recertification |
| `CAPACITY` | Volume forecasting, anticipated demand increases |

**Urgency levels:** `ROUTINE` / `MODERATE` / `URGENT`  
**Escalation flag:** Set when supervisor approval or emergency override is required by policy.

---

## Key Design Decisions

**Two-layer urgency detection.** A deterministic Python pre-classifier scans for urgent signal keywords before the LLM runs. If Python flags urgency that the LLM classifies as routine, urgency is automatically upgraded — ensuring imminent food gaps are never downgraded by probabilistic routing.

**Prohibited guarantee detection.** The validator blocks any response containing language like "we guarantee" or "always available" — enforcing NCA's Communication Standards that require conditional language around inventory.

**Policy retrieval is deterministic.** The FAQ knowledge base is segmented into 10 labelled chunks. Retrieval is keyword-scored Python — no vector embeddings, no hallucinated policy citations. Every response includes a policy reference footer citing the specific sections used.

**Person-first language enforced at the prompt level.** The system prompt requires language like "households experiencing food insecurity" rather than "food insecure households" — aligned with NCA brand guidelines.

---

## Data & Knowledge Base

| Source | Content |
|--------|---------|
| `northbridge_partner_intake_reports.csv` | 14 partner intake reports (text narratives) |
| `Northbridge_Food_Access_Internal_FAQ.pdf` | 11 policy sections — eligibility, hours, dietary, bulk, SNAP, donation standards |

---

## Setup

```bash
pip install anthropic pandas
export ANTHROPIC_API_KEY=your_key_here
python nca_food_access_assistant.py
# or specify CSV path:
python nca_food_access_assistant.py path/to/northbridge_partner_intake_reports.csv
```

---

## Example Interactions

```
You: list
→ Shows all 14 intake reports with org name and timestamp

You: process RPT-110
→ Triages: SNAP_BENEFITS | MODERATE | escalate=False
→ Policy sections: snap, eligibility
→ Draft response: Explains biweekly visit exception for SNAP interruption,
   no documentation required for first override, SNAP recertification referral.
   [NCA Food Access Team | Policy ref: snap, eligibility]

You: process RPT-105
→ Triages: SCHEDULING | ROUTINE | escalate=False
→ Policy sections: temp_housing, delivery
→ Draft response: Confirms microwave-ready food package options,
   notes delivery eligibility for temporary housing placements,
   subject to 48-hour advance request.

You: policy Can families use a proxy to pick up their food package?
→ Direct policy lookup: Hours & Pickup section
→ Yes — with written note/text confirmation, household name/address, proxy ID.
```

---

## Governance Dimensions

| Dimension | Implementation |
|-----------|---------------|
| Risk Classification | Urgency tiers + escalation flag |
| Auditability | Policy section IDs cited in every response |
| Explainability | Triage reasoning field in output |
| Human Oversight | Escalation flag triggers staff review requirement |
| Transparency | Prohibited guarantee detection; conditional language enforced |
