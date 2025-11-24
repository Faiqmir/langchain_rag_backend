import os
import json
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

logger = logging.getLogger(__name__)

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL", "deepseek-chat")


def _normalize_scope(scope: str) -> str:
    scope_key = (scope or "").strip().lower() or "local"
    if scope_key not in {"local", "international"}:
        scope_key = "local"
    return scope_key


def _normalize_currency(currency: str) -> str:
    cur = (currency or "").strip().upper() or "PKR"
    if cur not in {"PKR", "USD", "EUR"}:
        cur = "PKR"
    return cur


def _normalize_project_type(project_type: str) -> str:
    key = (project_type or "").strip().lower() or "web_app"
    allowed = {
        "web_app",
        "android_app",
        "ios_app",
        "backend_api",
        "desktop_app",
        "data_pipeline",
    }
    if key not in allowed:
        key = "web_app"
    return key


def generate_costing(
    context: str,
    development_scope: str,
    currency: str,
    project_type: str,
    developer_count: int | None = None,
    project_budget: float | None = None,
) -> Dict[str, Any]:
    """Generate a structured costing estimate using DeepSeek (LLM-driven).

    The LLM is responsible for choosing team composition, monthly rates, and
    total cost based on:
      - development_scope (local vs international)
      - currency (PKR, USD, EUR)
      - project_type (web_app, android_app, etc.)
      - developer_count and project_budget hints
      - high-level project context
      - Time-duration-months

    The model must return a *single JSON object* with a stable schema so both
    the frontend and the PDF generator can render the same "Costing Estimate"
    block.
    """

    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY is not set; returning empty costing payload.")
        return {}

    scope_key = _normalize_scope(development_scope)
    currency_key = _normalize_currency(currency)
    project_type_key = _normalize_project_type(project_type)
    effective_devs = max(1, developer_count or 1)

    # Truncate very long context to keep prompt size manageable
    max_context_chars = 6000
    trimmed_context = (context or "")[:max_context_chars]

    schema_example = {
        "development_scope": scope_key,
        "currency": currency_key,
        "project_type": project_type_key,
        "assumed_duration_months": 3,
        "developer_count": effective_devs,
        "monthly_team_cost": 123456.0,
        "total_estimated_cost": 987654.0,
        "items": [
            {
                "role": "Software Developer",
                "quantity": effective_devs,
                "monthly_rate": 12345.0,
                "duration_months": 3,
                "subtotal": 12345.0 * 3 * effective_devs,
            }
        ],
        "natural_language_summary": "Short explanation of the estimate for the client.",
    }

    system_instructions = (
        "You are a senior project manager and cost analyst. "
        "Your task is to produce a realistic software project cost not too high but realistic costs depending upon the requirements of the project estimate as a single JSON object."
    )

    user_prompt = f"""
{system_instructions}

CONTEXT (client requirements, partially truncated if very long):
------------------------------
{trimmed_context}
------------------------------

PROJECT SETTINGS PROVIDED BY USER:
- Development scope: {scope_key} (local vs international team)
- Currency: {currency_key}
- Project type: {project_type_key}
- Developer count hint: {effective_devs}
- Budget hint (may be approximate): {project_budget}

REQUIREMENTS:
- Return ONLY one valid JSON object. No prose outside JSON. No markdown fences.
- The JSON MUST have at least these fields:
  - development_scope (string)
  - currency (string)
  - project_type (string)
  - assumed_duration_months (number)  // You must estimate this based on context
  - developer_count (number)
  - monthly_team_cost (number)  // total cost per month for the whole team in the chosen currency
  - total_estimated_cost (number)  // total project cost in the chosen currency
  - items (array of objects). Each item MUST have:
        role (string),
        quantity (number),
        monthly_rate (number),
        duration_months (number),
        subtotal (number)
  - natural_language_summary (string)  // 3-6 sentence explanation in business language

- All numeric values must be realistic for the specified scope and currency.
- Use the budget and developer_count hints as guidance but you may adjust them
  if the scope clearly requires more or fewer resources.
  - Estimate assumed_duration_months realistically (e.g., 1-12 months) based on project complexity.

OUTPUT JSON EXAMPLE (structure only, numbers are placeholders):
{json.dumps(schema_example, indent=2)}

Now return ONLY the final JSON object.
"""

    try:
        chat = ChatDeepSeek(
            model=CHAT_MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.2,
        )

        response = chat.invoke(user_prompt)
        raw_text = str(response.content).strip()

        # Remove possible markdown fences if the model added them
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        # Extract the first JSON object
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning("No JSON object found in costing LLM response.")
            return {}

        json_str = raw_text[start : end + 1]
        payload = json.loads(json_str)
        logger.info("✅ Parsed costing JSON successfully.")
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("Costing LLM generation failed: %s", exc)
        return {}
