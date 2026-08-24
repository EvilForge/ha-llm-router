import os
import re
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Home Assistant LLM Router")

ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:14b-instruct")
FAST_MODEL = os.getenv("FAST_MODEL", "mistral:7b-instruct")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", FAST_MODEL)
HA_URL = os.getenv("HOME_ASSISTANT_URL", "").rstrip("/")
HA_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN", "")
MAX_ENTITIES_SENT = int(os.getenv("MAX_ENTITIES_SENT", "40"))
MAX_RELEVANT_ENTITIES = int(os.getenv("MAX_RELEVANT_ENTITIES", "8"))
ENTITY_SCORE_WINDOW = int(os.getenv("ENTITY_SCORE_WINDOW", "20"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG_LLM_PAYLOADS = os.getenv("DEBUG_LLM_PAYLOADS", "false").lower() == "true"

ROUTE_SETTINGS = {
    "control": {
        "model": FAST_MODEL,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 700,
    },
    "status": {
        "model": FAST_MODEL,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 700,
    },
    "summary": {
        "model": SUMMARY_MODEL,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 2048,
    },
    "general": {
        "model": DEFAULT_MODEL,
        "temperature": 0.4,
        "top_p": 0.95,
        "max_tokens": 2048,
    },
}

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("ha-llm-router")


def safe_json(data: Any, max_chars: int = 12000) -> str:
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = str(data)

    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>..."

    return text

CONTROL_WORDS = {
    "turn",
    "switch",
    "set",
    "dim",
    "brighten",
    "open",
    "close",
    "lock",
    "unlock",
    "start",
    "stop",
    "toggle",
    "activate",
    "deactivate",
    "dock",
    "mow",
    "pause",
    "resume",
}

SUMMARY_WORDS = {
    "summarize",
    "summary",
    "morning",
    "briefing",
    "update",
    "weather",
    "calendar",
    "what's happening",
    "whats happening",
}

STATUS_WORDS = {
    "status",
    "state",
    "is",
    "are",
    "what",
    "whether",
    "open",
    "closed",
    "locked",
    "unlocked",
    "on",
    "off",
    "running",
    "active",
    "current",
    "temperature",
    "temp",
    "humidity",
    "battery",
}

DOMAIN_HINTS = {
    "light": ["light"],
    "lights": ["light"],
    "lamp": ["light"],
    "lamps": ["light"],
    "fan": ["fan", "switch"],
    "fans": ["fan", "switch"],
    "switch": ["switch"],
    "switches": ["switch"],
    "outlet": ["switch"],
    "outlets": ["switch"],
    "plug": ["switch"],
    "plugs": ["switch"],
    "lock": ["lock"],
    "locks": ["lock"],
    "door": ["lock", "cover", "binary_sensor"],
    "doors": ["lock", "cover", "binary_sensor"],
    "window": ["binary_sensor", "cover"],
    "windows": ["binary_sensor", "cover"],
    "garage": ["cover", "binary_sensor", "switch", "lock"],
    "thermostat": ["climate"],
    "thermostats": ["climate"],
    "climate": ["climate"],
    "hvac": ["climate"],
    "heater": ["climate"],
    "furnace": ["climate"],
    "heating": ["climate"],
    "cooling": ["climate"],
    "setpoint": ["climate"],
    "air conditioner": ["climate"],
    "air conditioning": ["climate"],
    "ac": ["climate"],
    "temperature": ["sensor", "climate"],
    "temp": ["sensor", "climate"],
    "humidity": ["sensor"],
    "battery": ["sensor"],
    "motion": ["binary_sensor"],
    "occupancy": ["binary_sensor"],
    "presence": ["binary_sensor", "person", "device_tracker"],
    "blind": ["cover"],
    "blinds": ["cover"],
    "shade": ["cover"],
    "shades": ["cover"],
    "cover": ["cover"],
    "covers": ["cover"],
    "scene": ["scene"],
    "scenes": ["scene"],
    "script": ["script"],
    "scripts": ["script"],
    "vacuum": ["vacuum"],
    "mower": ["lawn_mower"],
    "mowers": ["lawn_mower"],
    "lawnmower": ["lawn_mower"],
    "lawnmowers": ["lawn_mower"],
    "lawn mower": ["lawn_mower"],
    "robot mower": ["lawn_mower"],
    "alarm": ["alarm_control_panel"],
    "weather": ["weather"],
    "calendar": ["calendar"],
}

AREA_HINTS = [
    "kitchen",
    "living room",
    "bedroom",
    "master bedroom",
    "primary bedroom",
    "office",
    "garage",
    "hallway",
    "bathroom",
    "patio",
    "porch",
    "backyard",
    "front yard",
    "laundry",
    "dining room",
    "entry",
    "entryway",
    "foyer",
    "upstairs",
    "downstairs",
    "outside",
    "shop",
    "closet",
    "pantry",
]


def check_auth(authorization: Optional[str]) -> None:
    if not ROUTER_API_KEY:
        return

    expected = f"Bearer {ROUTER_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def latest_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return " ".join(parts)

    return ""


def contains_hint(text: str, hint: str) -> bool:
    """Match whole words/phrases so short hints such as 'ac' are safe."""
    return re.search(rf"(?<![a-z0-9_]){re.escape(hint)}(?![a-z0-9_])", text) is not None


def classify_request(text: str) -> str:
    t = text.lower().strip()
    tokens = set(re.findall(r"[a-z0-9_']+", t))

    if any(w in t for w in SUMMARY_WORDS):
        return "summary"

    # Status questions:
    # "Is the garage open?"
    # "Are the doors locked?"
    # "What's the thermostat set to?"
    # "Is the porch light on?"
    # "What is the battery level on the front door lock?"
    looks_like_question = (
        "?" in t
        or t.startswith(
            (
                "is ",
                "are ",
                "was ",
                "were ",
                "what ",
                "what's ",
                "whats ",
                "which ",
                "show ",
                "tell me ",
                "give me ",
                "do i have ",
                "do we have ",
            )
        )
        or any(w in tokens for w in {"status", "state", "current"})
    )

    mentions_device_or_area = (
        any(contains_hint(t, k) for k in DOMAIN_HINTS)
        or any(area in t for area in AREA_HINTS)
    )

    if looks_like_question and mentions_device_or_area:
        return "status"

    if any(w in tokens for w in CONTROL_WORDS):
        return "control"

    # Voice shorthand:
    # "kitchen lights off"
    # "garage door open"
    # "bedroom fan on"
    if any(contains_hint(t, k) for k in DOMAIN_HINTS):
        if any(x in tokens for x in {
            "on", "off", "open", "closed", "close", "dim", "bright",
            "heat", "cool", "dock", "mow", "pause", "resume", "home",
        }):
            return "control"

    return "general"


def extract_area_hints(text: str) -> List[str]:
    t = text.lower()
    return [area for area in AREA_HINTS if area in t]


def extract_domain_hints(text: str) -> List[str]:
    t = text.lower()
    domains = set()

    for word, mapped_domains in DOMAIN_HINTS.items():
        if contains_hint(t, word):
            domains.update(mapped_domains)

    return sorted(domains)


def normalize_entity_name(entity: Dict[str, Any]) -> str:
    attrs = entity.get("attributes", {}) or {}
    friendly = attrs.get("friendly_name")
    return friendly or entity.get("entity_id", "")


def score_entity(
    entity: Dict[str, Any],
    text: str,
    areas: List[str],
    domains: List[str],
) -> int:
    entity_id = entity.get("entity_id", "")
    domain = entity_id.split(".")[0]
    name = normalize_entity_name(entity).lower()
    attrs = entity.get("attributes", {}) or {}
    device_class = str(attrs.get("device_class", "")).lower()

    haystack = f"{entity_id} {name} {device_class}".lower()
    t = text.lower()

    score = 0

    if domains and domain in domains:
        score += 50

    for area in areas:
        if area in haystack:
            score += 45

    # Token match against friendly name / entity_id / device_class.
    for token in re.findall(r"[a-z0-9_]+", t):
        if len(token) >= 3 and token in haystack:
            score += 10

    # Device-class nudges for status questions.
    if "door" in t and device_class == "door":
        score += 35
    if "garage" in t and ("garage" in haystack or device_class == "garage_door"):
        score += 35
    if "window" in t and device_class == "window":
        score += 35
    if "motion" in t and device_class == "motion":
        score += 35
    if "occupancy" in t and device_class == "occupancy":
        score += 35
    if "battery" in t and device_class == "battery":
        score += 35
    if ("temperature" in t or "temp" in t) and device_class == "temperature":
        score += 35
    if "humidity" in t and device_class == "humidity":
        score += 35
    if any(word in t for word in ["hvac", "heating", "cooling", "heater", "setpoint"]):
        if domain == "climate":
            score += 40
    if any(word in t for word in ["mower", "lawnmower", "lawn mower"]):
        if domain == "lawn_mower":
            score += 40

    # Prefer useful Home Assistant domains.
    if domain in {
        "light",
        "switch",
        "fan",
        "lock",
        "cover",
        "climate",
        "scene",
        "script",
        "sensor",
        "binary_sensor",
        "weather",
        "calendar",
        "vacuum",
        "alarm_control_panel",
        "lawn_mower",
    }:
        score += 5

    return score


async def fetch_ha_states() -> List[Dict[str, Any]]:
    if not HA_URL or not HA_TOKEN:
        return []

    headers = {"Authorization": f"Bearer {HA_TOKEN}"}

    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(f"{HA_URL}/api/states", headers=headers)
        resp.raise_for_status()
        return resp.json()


COMMON_ENTITY_ATTRIBUTES = {
    "friendly_name",
    "supported_features",
    "device_class",
    "unit_of_measurement",
    "battery",
    "battery_level",
    "available",
}

ENTITY_ATTRIBUTES_BY_DOMAIN = {
    "climate": {
        "current_temperature",
        "temperature",
        "target_temp_high",
        "target_temp_low",
        "hvac_mode",
        "hvac_action",
        "hvac_modes",
        "preset_mode",
        "preset_modes",
        "fan_mode",
        "fan_modes",
        "swing_mode",
        "swing_modes",
        "min_temp",
        "max_temp",
        "target_temp_step",
        "current_humidity",
        "target_humidity",
    },
    "lawn_mower": {
        "activity",
        "status",
        "error",
        "error_code",
    },
}


def compact_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    attrs = entity.get("attributes", {}) or {}
    keep_attrs = {}
    domain = entity.get("entity_id", "").split(".")[0]

    useful_attrs = COMMON_ENTITY_ATTRIBUTES | ENTITY_ATTRIBUTES_BY_DOMAIN.get(domain, set()) | {
        "brightness",
        "color_temp_kelvin",
    }

    for key in useful_attrs:
        if key in attrs:
            keep_attrs[key] = attrs[key]

    return {
        "entity_id": entity.get("entity_id"),
        "state": entity.get("state"),
        "last_changed": entity.get("last_changed"),
        "attributes": keep_attrs,
    }


async def select_relevant_entities(text: str, mode: str) -> List[Dict[str, Any]]:
    # General knowledge requests must not expose HA state to the large model.
    if mode == "general":
        return []

    states = await fetch_ha_states()
    if not states:
        return []

    areas = extract_area_hints(text)
    domains = extract_domain_hints(text)
    t = text.lower()

    if mode == "summary":
        preferred_domains = {
            "sensor",
            "binary_sensor",
            "weather",
            "calendar",
            "climate",
            "lock",
            "cover",
            "light",
            "switch",
            "fan",
            "alarm_control_panel",
            "lawn_mower",
        }

        candidates = [
            s for s in states
            if s.get("entity_id", "").split(".")[0] in preferred_domains
        ]

    elif mode == "status":
        preferred_domains = {
            "light",
            "switch",
            "fan",
            "lock",
            "cover",
            "climate",
            "sensor",
            "binary_sensor",
            "person",
            "device_tracker",
            "vacuum",
            "alarm_control_panel",
            "weather",
            "calendar",
            "lawn_mower",
        }

        if domains:
            domain_set = set(domains)

            # Door/window/garage status is often represented by binary_sensor,
            # cover, or lock, depending on the device/integration.
            if any(word in t for word in ["door", "window", "garage", "motion", "occupancy"]):
                domain_set.add("binary_sensor")
                domain_set.add("cover")
                domain_set.add("lock")

            # Temperature status may be from climate or sensor.
            if any(word in t for word in ["temperature", "temp", "thermostat"]):
                domain_set.add("sensor")
                domain_set.add("climate")

            # Battery usually lives under sensor.
            if "battery" in t:
                domain_set.add("sensor")

            candidates = [
                s for s in states
                if s.get("entity_id", "").split(".")[0] in domain_set
            ]

        else:
            candidates = [
                s for s in states
                if s.get("entity_id", "").split(".")[0] in preferred_domains
            ]

    elif domains:
        candidates = [
            s for s in states
            if s.get("entity_id", "").split(".")[0] in domains
        ]

    else:
        candidates = states

    ranked = sorted(
        candidates,
        key=lambda e: score_entity(e, text, areas, domains),
        reverse=True,
    )

    scored = [
        (e, score_entity(e, text, areas, domains))
        for e in ranked
    ]
    positive = [(e, score) for e, score in scored if score > 0]

    if mode == "summary":
        # Summaries intentionally cover a wider slice of the home.
        selected = [e for e, _score in positive]
        if not selected:
            selected = ranked
        limit = MAX_ENTITIES_SENT
    else:
        # A domain match gives every entity in that domain a baseline score. Keep
        # only entities close to the strongest match so a named room/device does
        # not drag dozens of unrelated sensors or lights into the prompt.
        if positive:
            best_score = positive[0][1]
            cutoff = max(1, best_score - ENTITY_SCORE_WINDOW)
            selected = [e for e, score in positive if score >= cutoff]
        else:
            selected = []
        limit = MAX_RELEVANT_ENTITIES

    return [compact_entity(e) for e in selected[:limit]]


def service_instruction() -> str:
    return """
You are assisting with Home Assistant.

General rules:
- Never invent entity IDs.
- Prefer the entity IDs and states from RELEVANT_HOME_ASSISTANT_CONTEXT.
- If the requested device is not present in context, say you do not see that device.
- Be concise.

For control requests:
- Use real Home Assistant domains such as light.turn_on, light.turn_off, switch.turn_on, switch.turn_off, fan.turn_on, fan.turn_off, cover.open_cover, cover.close_cover, lock.lock, lock.unlock, climate.set_temperature.
- Climate controls may also use climate.set_hvac_mode, climate.set_fan_mode, and climate.set_preset_mode.
- Lawn mower controls may use lawn_mower.start_mowing, lawn_mower.pause, and lawn_mower.dock.
- Return the exact tool/function call requested by Home Assistant if tools are provided.
- Do not use homeassistant.turn_on.
- Do not call a service on an entity that is not present in RELEVANT_HOME_ASSISTANT_CONTEXT.
- Ask one short clarification question if the target device is ambiguous.

For status requests:
- Do not call a service.
- Answer using the current state values in RELEVANT_HOME_ASSISTANT_CONTEXT.
- Mention the friendly name and state.
- For climate entities, distinguish the configured state/HVAC mode from hvac_action (what it is currently doing), current_temperature (measured), and temperature (target).
- If several matching entities exist, summarize the relevant ones.
- Translate common binary_sensor states naturally:
  - on can mean open, detected, occupied, wet, unsafe, or active depending on device_class.
  - off can mean closed, clear, unoccupied, dry, safe, or inactive depending on device_class.
- If device_class is available, use it to phrase the answer naturally.
- If the state is unavailable or unknown, say that clearly.

For summary requests:
- Summarize the useful current states briefly.
- Do not list every entity unless asked.
"""


def inject_context(
    messages: List[Dict[str, Any]],
    mode: str,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    context = {
        "request_type": mode,
        "relevant_home_assistant_entities": entities,
    }

    system_msg = {
        "role": "system",
        "content": (
            service_instruction()
            + "\n\nRELEVANT_HOME_ASSISTANT_CONTEXT:\n"
            + str(context)
        ),
    }

    # Put our system message first, but keep the integration's own messages too.
    return [system_msg] + messages


def filter_tools_for_request(
    tools: Optional[List[Dict[str, Any]]],
    text: str,
    mode: str,
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return tools

    # Non-control requests do not need Home Assistant service tools.
    if mode in {"status", "summary", "general"}:
        return None

    t = text.lower()

    wanted_fragments = set()

    if "light" in t or "lamp" in t:
        wanted_fragments.update(["light"])
    if "switch" in t or "outlet" in t or "plug" in t:
        wanted_fragments.update(["switch"])
    if "fan" in t:
        wanted_fragments.update(["fan"])
    if "lock" in t or "door" in t:
        wanted_fragments.update(["lock"])
    if "garage" in t or "cover" in t or "blind" in t or "shade" in t:
        wanted_fragments.update(["cover"])
    if "thermostat" in t or "temperature" in t or "temp" in t:
        wanted_fragments.update(["climate"])
    if any(contains_hint(t, word) for word in [
        "climate", "hvac", "heater", "furnace", "heating", "cooling", "setpoint",
        "air conditioner", "air conditioning", "ac",
    ]):
        wanted_fragments.update(["climate"])
    if any(word in t for word in ["mower", "lawnmower", "lawn mower"]):
        wanted_fragments.update(["lawn_mower", "mower"])

    if not wanted_fragments:
        return tools[:24]

    filtered = []

    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "").lower()
        desc = fn.get("description", "").lower()
        blob = f"{name} {desc}"

        if any(fragment in blob for fragment in wanted_fragments):
            filtered.append(tool)

    return filtered or tools[:24]


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)

    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "owned_by": "ollama",
            },
            {
                "id": FAST_MODEL,
                "object": "model",
                "owned_by": "ollama",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)

    body = await request.json()
    original_body = dict(body)

    messages = body.get("messages", [])
    user_text = latest_user_text(messages)
    mode = classify_request(user_text)

    logger.info("Incoming request: mode=%s user_text=%r", mode, user_text)

    if DEBUG_LLM_PAYLOADS:
        redacted_headers = {
            "authorization_present": bool(authorization),
        }
        logger.info("Incoming headers summary:\n%s", safe_json(redacted_headers))
        logger.info("Incoming OpenAI-compatible payload:\n%s", safe_json(original_body))

    entities = await select_relevant_entities(user_text, mode)

    logger.info(
        "Selected %s relevant entities: %s",
        len(entities),
        [e.get("entity_id") for e in entities],
    )

    # The router owns model selection and generation policy. Applying these
    # settings after reading the client payload intentionally overrides any
    # corresponding values supplied by Home Assistant.
    body.update(ROUTE_SETTINGS[mode])

    if mode == "general":
        body["messages"] = messages
    else:
        body["messages"] = inject_context(messages, mode, entities)

    filtered_tools = filter_tools_for_request(body.get("tools"), user_text, mode)
    if filtered_tools:
        body["tools"] = filtered_tools
    else:
        body.pop("tools", None)

    logger.info(
        "Sending request to Ollama: model=%s mode=%s message_count=%s tool_count=%s",
        body.get("model"),
        mode,
        len(body.get("messages", [])),
        len(body.get("tools", []) or []),
    )

    if DEBUG_LLM_PAYLOADS:
        logger.info("Outgoing payload to Ollama:\n%s", safe_json(body))

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/chat/completions",
            headers={"Authorization": "Bearer ollama"},
            json=body,
        )

    try:
        content = resp.json()
    except Exception:
        content = {
            "error": "Ollama returned a non-JSON response",
            "status_code": resp.status_code,
            "text": resp.text,
        }

    logger.info("Ollama response status=%s", resp.status_code)

    if DEBUG_LLM_PAYLOADS:
        logger.info("Ollama response body:\n%s", safe_json(content))

    return JSONResponse(status_code=resp.status_code, content=content)
