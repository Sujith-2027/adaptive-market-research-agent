"""Playbooks tell the agent what matters for each business type. Unknown types get a proposed playbook."""
import json, pathlib
from llm import llm_json

FILE = pathlib.Path("playbooks.json")

BUILTIN = {
    "food_delivery": {
        "key_columns": ["order_date", "restaurant", "city", "order_value", "delivery_fee", "rating"],
        "questions": ["Which cities and restaurants drive the most revenue?",
                      "How do delivery fees and ratings relate to order value?",
                      "What does the monthly trend show?"],
        "external": ["typical commission rate charged to restaurants by delivery platforms",
                     "online food delivery market growth"],
    },
    "retail": {
        "key_columns": ["order_date", "product", "category", "quantity", "revenue", "region"],
        "questions": ["Which products and categories earn the most?",
                      "Which regions are strongest and weakest?",
                      "What does the sales trend over time show?"],
        "external": ["retail sector growth rate", "average retail profit margin by category"],
    },
    "saas": {
        "key_columns": ["signup_date", "plan", "monthly_price", "churned", "region"],
        "questions": ["Which plans bring the most revenue?",
                      "Where does churn concentrate?",
                      "How is growth trending?"],
        "external": ["typical SaaS churn rate benchmark", "SaaS pricing benchmarks for small teams"],
    },
    "general": {
        "key_columns": [],
        "questions": ["What are the main drivers of the key numeric measure?",
                      "Which groups perform best and worst?",
                      "What trends or anomalies stand out?"],
        "external": [],
    },
}


def _saved():
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_playbook(btype, pb):
    data = _saved()
    data[btype] = pb
    FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def propose_playbook(btype):
    d = llm_json(f"""Create an analysis playbook for a business of type "{btype}".
Return JSON with keys:
  key_columns: 5 to 7 canonical snake_case columns a dataset of this business usually has
  questions: 3 plain-language business questions worth answering from such data
  external: 2 public market facts worth fetching from the web for context (short phrases)""")
    return {"key_columns": list(d.get("key_columns", []))[:8],
            "questions": list(d.get("questions", []))[:5],
            "external": list(d.get("external", []))[:4]}


def get_playbook(btype):
    saved = _saved()
    if btype in saved:
        return saved[btype]
    if btype in BUILTIN:
        return BUILTIN[btype]
    try:
        return propose_playbook(btype)
    except Exception:
        return BUILTIN["general"]
