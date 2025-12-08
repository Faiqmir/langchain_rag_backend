import os
import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL", "deepseek-chat")


# ---------------------------
# Normalizers / validators
# ---------------------------
def _normalize_scope(scope: str) -> str:
    scope_key = (scope or "").strip().lower()
    if scope_key not in {"local", "international"}:
        raise ValueError("Invalid development_scope. Must be 'local' or 'international'")
    return scope_key


def _normalize_currency(currency: str) -> str:
    cur = (currency or "").strip().upper()
    if cur not in {"PKR", "USD", "EUR"}:
        raise ValueError("Invalid currency. Must be one of: PKR, USD, EUR")
    return cur


def _normalize_project_type(project_type: str) -> str:
    key = (project_type or "").strip().lower()
    allowed = {
        "web_app",
        "android_app",
        "ios_app",
        "backend_api",
        "desktop_app",
        "data_pipeline",
    }
    if key not in allowed:
        raise ValueError(f"Invalid project_type '{project_type}'. Allowed: {', '.join(sorted(allowed))}")
    return key


def _ensure_positive_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _ensure_positive_int_or_none(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


# ---------------------------
# LLM prompt & call
# ---------------------------
def _build_user_prompt(
    scope_key: str,
    currency_key: str,
    project_type_key: str,
    context: str,
    resources_needed: Optional[int],
    timeline_weeks: Optional[int],
    technical_hourly_rate: Optional[float],
    non_technical_hourly_rate: Optional[float],
    fixed_budget: Optional[float],
) -> str:
    trimmed_context = (context or "")[:6000]
    return f"""
You are a project cost analyst. Generate a project cost estimate based on the provided constraints.

PROJECT SETTINGS:
- Development scope: {scope_key}
- Currency: {currency_key}
- Project type: {project_type_key}
- Resources: {resources_needed if resources_needed else 'Estimate based on project needs'}
- Timeline: {timeline_weeks if timeline_weeks else 'Estimate based on project complexity'} weeks
- Technical Hourly Rate: {technical_hourly_rate if technical_hourly_rate else 'Estimate based on market rates'} {currency_key}
- Non-Technical Hourly Rate: {non_technical_hourly_rate if non_technical_hourly_rate else 'Estimate based on market rates'} {currency_key}
- Fixed Budget Limit: {fixed_budget if fixed_budget else 'No limit'} {currency_key}

CONTEXT:
{trimmed_context}

RULES:
1. If resources are provided, create exactly that many team members
2. If timeline is provided, use exactly that duration
3. If rates are provided, use exactly those rates
4. If any values are not provided, estimate them reasonably based on project type and scope
5. Total cost MUST NOT exceed fixed budget if provided
6. Distribute roles appropriately for {project_type_key}
7. Each member works 40 hours/week
8. ALL monetary values must be in {currency_key}

REQUIRED JSON OUTPUT:
{{
  "development_scope": "{scope_key}",
  "currency": "{currency_key}",
  "project_type": "{project_type_key}",
  "developer_count": <number>,
  "total_estimated_cost": <CALCULATED_TOTAL>,
  "natural_language_summary": "<Brief summary>",
  "items": [
    {{
      "role": "<Role Name>",
      "quantity": 1,
      "hourly_rate": <rate>,
      "duration_weeks": <weeks>,
      "hours_per_week": 40,
      "subtotal": <quantity * hourly_rate * duration_weeks * hours_per_week>
    }}
  ]
}}

Return ONLY the JSON object. No explanations or markdown formatting.
"""


def _call_llm(prompt: str) -> str:
    """Call DeepSeek LLM and return raw text response (string)."""
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY is not set; skipping LLM call.")
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    chat = ChatDeepSeek(
        model=CHAT_MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.2,
    )
    response = chat.invoke(prompt)
    text = str(response.content).strip()
    logger.debug(f"LLM response (truncated): {text[:1000]}")
    return text


# ---------------------------
# Simple JSON extraction (no complex parsing)
# ---------------------------
def _extract_simple_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Simple JSON extraction - look for JSON object and extract it"""
    # Remove any markdown fences
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    
    # Find JSON object between { and }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    
    if start == -1 or end == -1 or end <= start:
        return None
    
    # Extract the JSON portion
    json_str = cleaned[start:end+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.debug(f"Problematic JSON: {json_str[:500]}")
        return None


# ---------------------------
# Simple constraint enforcement
# ---------------------------
def _items_total(items: List[Dict[str, Any]]) -> float:
    total = 0.0
    for it in items:
        subtotal = it.get("subtotal")
        if subtotal is None:
            # Attempt to calculate from fields
            rate = it.get("hourly_rate") or 0.0
            qty = it.get("quantity") or 1
            weeks = it.get("duration_weeks") or 0
            hours_per_week = it.get("hours_per_week") or 40
            subtotal = qty * rate * weeks * hours_per_week
        total += float(subtotal)
    return total


def _apply_constraints(
    payload: Dict[str, Any],
    timeline_weeks: Optional[int],
    resources_needed: Optional[int],
    fixed_budget: Optional[float],
    technical_hourly_rate: Optional[float],
    non_technical_hourly_rate: Optional[float],
    currency_key: str = "USD",
) -> Dict[str, Any]:
    """
    Apply user constraints to the payload with proper cost calculations.
    """
    payload = dict(payload)  # Shallow copy
    items = payload.get("items", [])
    
    # Ensure items is a list of dicts
    items = [dict(i) for i in items]
    
    # Enforce timeline: update every item duration_weeks to timeline_weeks if provided
    if timeline_weeks is not None:
        for it in items:
            it["duration_weeks"] = timeline_weeks
    
    # Enforce resource count: adjust number of items to resources_needed
    if resources_needed is not None:
        current = len(items)
        if current > resources_needed:
            items = items[:resources_needed]
            logger.info("Truncated items to meet resource constraint: %d -> %d", current, resources_needed)
        elif current < resources_needed:
            # Add reasonable roles to reach target
            roles = [
                "Full Stack Developer",
                "Frontend Developer",
                "Backend Developer",
                "Mobile Developer",
                "UI/UX Designer",
                "QA Engineer",
                "DevOps Engineer",
                "Project Manager",
                "Data Analyst",
            ]
            idx = 0
            needed = resources_needed - current
            for _ in range(needed):
                role = roles[idx % len(roles)]
                idx += 1
                # Use provided rates or reasonable defaults
                if "Developer" in role:
                    hr_rate = technical_hourly_rate if technical_hourly_rate is not None else (non_technical_hourly_rate if non_technical_hourly_rate is not None else 75.0)
                elif "Designer" in role or "QA" in role:
                    hr_rate = non_technical_hourly_rate if non_technical_hourly_rate is not None else (technical_hourly_rate if technical_hourly_rate is not None else 60.0)
                else:
                    hr_rate = technical_hourly_rate if technical_hourly_rate is not None else (non_technical_hourly_rate if non_technical_hourly_rate is not None else 65.0)

                # Use provided weeks or reasonable default
                weeks = timeline_weeks or 12

                # Calculate proper subtotal
                qty = 1
                hours_per_week = 40
                subtotal = qty * hr_rate * weeks * hours_per_week
                
                items.append({
                    "role": role,
                    "quantity": qty,
                    "hourly_rate": hr_rate,
                    "duration_weeks": weeks,
                    "hours_per_week": hours_per_week,
                    "subtotal": round(subtotal, 2),
                })
            logger.info("Padded items to meet resource constraint: %d -> %d", current, len(items))
    
    # Apply user-provided rates and recalculate all subtotals
    logger.info(f"Applying user rates - Technical: {technical_hourly_rate}, Non-technical: {non_technical_hourly_rate}")
    for it in items:
        role = it.get("role", "").lower()
        
        # Apply user rates if provided - more comprehensive role detection
        if technical_hourly_rate is not None and any(dev_role in role for dev_role in ["developer", "engineer", "programmer"]):
            it["hourly_rate"] = technical_hourly_rate
            logger.info(f"Applied technical rate {technical_hourly_rate} to {it.get('role')}")
        elif non_technical_hourly_rate is not None and any(non_dev_role in role for non_dev_role in ["designer", "qa", "test", "analyst", "manager"]):
            it["hourly_rate"] = non_technical_hourly_rate
            logger.info(f"Applied non-technical rate {non_technical_hourly_rate} to {it.get('role')}")
        else:
            # If no specific rate matches, use technical rate as default for any role
            if technical_hourly_rate is not None:
                it["hourly_rate"] = technical_hourly_rate
                logger.info(f"Applied default technical rate {technical_hourly_rate} to {it.get('role')}")
            elif non_technical_hourly_rate is not None:
                it["hourly_rate"] = non_technical_hourly_rate
                logger.info(f"Applied default non-technical rate {non_technical_hourly_rate} to {it.get('role')}")
        
        # Recalculate subtotal with proper values
        try:
            rate = float(it.get("hourly_rate", 0))
            qty = int(it.get("quantity", 1))
            weeks = int(it.get("duration_weeks", timeline_weeks or 12))
            hours_per_week = int(it.get("hours_per_week", 40))
            
            if rate > 0:
                it["subtotal"] = round(qty * rate * weeks * hours_per_week, 2)
                logger.info(f"Calculated subtotal for {it.get('role')}: {qty} * {rate} * {weeks} * {hours_per_week} = {it['subtotal']}")
            else:
                it["subtotal"] = 0.0
        except Exception as e:
            logger.warning(f"Error calculating subtotal for {it.get('role', 'Unknown')}: {e}")
            it["subtotal"] = 0.0
    
    # Calculate total cost
    total = _items_total(items)
    
    # Instead of scaling, just warn if over budget but preserve user rates
    if fixed_budget is not None and total > fixed_budget:
        logger.warning(f"Total cost {total:.2f} exceeds budget {fixed_budget:.2f}, but preserving user-specified rates")
    
    payload["items"] = items
    payload["developer_count"] = resources_needed if resources_needed is not None else payload.get("developer_count")
    payload["total_estimated_cost"] = round(total, 2)
    
    # Prepend summary note if constraints were enforced
    note_parts = []
    if timeline_weeks is not None:
        note_parts.append(f"timeline={timeline_weeks} weeks")
    if resources_needed is not None:
        note_parts.append(f"resources={resources_needed}")
    if fixed_budget is not None:
        note_parts.append(f"budget={fixed_budget}")
    if note_parts:
        prev_summary = payload.get("natural_language_summary", "")
        payload["natural_language_summary"] = f"Estimate adjusted to match strict user constraints ({', '.join(note_parts)}). {prev_summary}"

    return payload


# ---------------------------
# Public function
# ---------------------------
def generate_costing(
    context: str,
    development_scope: str,
    currency: str,
    project_type: str,
    developer_count: Optional[int] = None,
    project_budget: Optional[float] = None,
    non_technical_hourly_rate: Optional[float] = None,
    technical_hourly_rate: Optional[float] = None,
    timeline_weeks: Optional[int] = None,
    fixed_budget: Optional[float] = None,
    resources_needed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a structured costing estimate using DeepSeek (LLM-driven).

    Simplified approach:
      - Direct JSON prompt (no markdown intermediate step)
      - Simple constraint enforcement
      - No complex JSON parsing
    """
    # Basic normalization and validation
    try:
        scope_key = _normalize_scope(development_scope)
        currency_key = _normalize_currency(currency)
        project_type_key = _normalize_project_type(project_type)
    except ValueError as ve:
        logger.error("Invalid input: %s", ve)
        raise

    technical_hourly_rate = _ensure_positive_or_none(technical_hourly_rate)
    non_technical_hourly_rate = _ensure_positive_or_none(non_technical_hourly_rate)
    timeline_weeks = _ensure_positive_int_or_none(timeline_weeks)
    
    # Use project_budget as fixed_budget if provided and fixed_budget is None
    if fixed_budget is None and project_budget is not None:
        fixed_budget = _ensure_positive_or_none(project_budget)
    
    # Use developer_count as resources_needed if provided and resources_needed is None
    if resources_needed is None and developer_count is not None:
        resources_needed = _ensure_positive_int_or_none(developer_count)

    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY is not set; returning empty costing payload.")
        return {}

    prompt = _build_user_prompt(
        scope_key=scope_key,
        currency_key=currency_key,
        project_type_key=project_type_key,
        context=context,
        resources_needed=resources_needed,
        timeline_weeks=timeline_weeks,
        technical_hourly_rate=technical_hourly_rate,
        non_technical_hourly_rate=non_technical_hourly_rate,
        fixed_budget=fixed_budget,
    )

    try:
        raw = _call_llm(prompt)
    except Exception as exc:
        logger.warning("Costing LLM generation failed: %s", exc)
        return {}

    # Simple JSON extraction
    parsed = _extract_simple_json(raw)
    if not parsed:
        logger.warning("Failed to extract JSON from LLM response")
        return {}

    # Apply constraints
    final_payload = _apply_constraints(
        payload=parsed,
        timeline_weeks=timeline_weeks,
        resources_needed=resources_needed,
        fixed_budget=fixed_budget,
        technical_hourly_rate=technical_hourly_rate,
        non_technical_hourly_rate=non_technical_hourly_rate,
        currency_key=currency_key,
    )

    return final_payload