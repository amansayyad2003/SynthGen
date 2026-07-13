import json
import os
import io
import zipfile
import streamlit as st
import pandas as pd
import time
from groq import Groq
from dotenv import load_dotenv
import re as _re
import random
import datetime
import uuid as _uuid

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(
    page_title="SynthGen",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
    }

    div[data-testid="stMetric"] label {
        font-size: 12px;
        color: var(--text-color);
        opacity: 0.6;
    }

    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        margin-top: 4px;
    }

    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    .pattern-chip {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 4px 4px 0;
        border: 1px solid rgba(99, 102, 241, 0.4);
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
    }

    .pattern-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        background: var(--secondary-background-color);
    }

    /* Style each column in the fraud pattern grid as a card */
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 10px 12px;
        background: var(--secondary-background-color);
    }

    .explanation-box {
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        padding: 14px 16px;
        background: rgba(99, 102, 241, 0.06);
        font-size: 14px;
        line-height: 1.7;
        word-break: break-word;
        white-space: pre-wrap;
    }

    /* Prevent full page dim during processing — show clean spinner overlay instead */
    div[data-testid="stSpinner"] > div {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 16px;
        border-radius: 8px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        font-size: 14px;
    }

    .stApp > div:first-child {
        opacity: 1 !important;
    }

    .fraud-pattern-desc {
        font-size: 12px;
        line-height: 1.55;
        margin-top: 2px;
        margin-bottom: 8px;
        color: var(--text-color);
        opacity: 0.92;
    }

    .fraud-pattern-fields {
        font-size: 12.5px;
        line-height: 1.6;
        color: #16a34a;
        background: rgba(22, 163, 74, 0.07);
        border: 1px solid rgba(22, 163, 74, 0.3);
        border-radius: 8px;
        padding: 8px 10px;
        margin-top: 4px;
        margin-bottom: 10px;
    }

    .fraud-pattern-fields strong {
        color: #15803d;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LLM HELPER
# ─────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8192,
    usage_label: str = "llm_call"
) -> str:

    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )

    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

    if "token_usage_log" not in st.session_state:
        st.session_state.token_usage_log = []
    if "token_usage_totals" not in st.session_state:
        st.session_state.token_usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    st.session_state.token_usage_log.append({
        "label": usage_label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    })
    st.session_state.token_usage_totals["prompt_tokens"] += prompt_tokens
    st.session_state.token_usage_totals["completion_tokens"] += completion_tokens
    st.session_state.token_usage_totals["total_tokens"] += total_tokens
    st.session_state.token_usage_totals["calls"] += 1

    return response.choices[0].message.content


# ─────────────────────────────────────────────
# JSON PARSER
# ─────────────────────────────────────────────

def _close_truncated_json(s: str) -> str:
    """Append closing chars to make a truncated JSON string parseable.
    Handles: mid-string cuts, trailing commas/colons, unclosed brackets.
    """
    stack = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    result = s
    # Close any open string first
    if in_string:
        result += '"'
    # Strip trailing comma or colon — both produce invalid JSON when followed by a closer
    result = result.rstrip()
    while result and result[-1] in (',', ':'):
        result = result[:-1].rstrip()
    # Append missing closing brackets
    result += ''.join(']' if c == '[' else '}' for c in reversed(stack))
    return result


def _recover_columnar_dict(s: str) -> dict | None:
    """Last-resort: extract only fully-complete \"key\": [...] pairs from a
    truncated columnar JSON response. Any incomplete final key is dropped.
    Returns a dict (possibly partial) or None if nothing complete found.
    """
    result = {}
    i = 0
    while i < len(s):
        # Find the next quoted key
        q1 = s.find('"', i)
        if q1 == -1:
            break
        q2 = s.find('"', q1 + 1)
        if q2 == -1:
            break
        key = s[q1 + 1:q2]
        # Expect  :  then  [
        rest = s[q2 + 1:].lstrip()
        if not rest.startswith(':'):
            i = q2 + 1
            continue
        rest = rest[1:].lstrip()
        if not rest.startswith('['):
            i = q2 + 1
            continue
        arr_start = s.index('[', q2)
        # Scan for the matching ']'
        depth = 0
        j = arr_start
        in_str = False
        esc = False
        found = False
        while j < len(s):
            ch = s[j]
            if esc:
                esc = False
            elif ch == '\\' and in_str:
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            result[key] = json.loads(s[arr_start:j + 1])
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        found = True
                        break
            j += 1
        if not found:
            break  # Truncated mid-array — stop; everything before is complete
    return result if result else None


def parse_json_response(raw: str):

    clean = raw.strip()

    if clean.startswith("```json"):
        clean = clean[7:]

    if clean.startswith("```"):
        clean = clean[3:]

    if clean.endswith("```"):
        clean = clean[:-3]

    clean = clean.strip()

    # Try full parse first
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Try extracting the JSON array/object bounds
    start = clean.find("[")
    end = clean.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(clean[start:end + 1])
        except json.JSONDecodeError:
            # Response was truncated — recover complete objects from partial array
            fragment = clean[start:]
            objects = []
            depth = 0
            obj_start = None
            for i, ch in enumerate(fragment):
                if ch == "{":
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        try:
                            objects.append(json.loads(fragment[obj_start:i + 1]))
                        except json.JSONDecodeError:
                            pass
                        obj_start = None
            if objects:
                return objects

    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(clean[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Last-resort 1: close any truncated JSON (handles LLM hitting token limit mid-response)
    for search_char in ('{', '['):
        pos = clean.find(search_char)
        if pos == -1:
            continue
        closed = _close_truncated_json(clean[pos:])
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            pass

    # Last-resort 2: extract only fully-complete "key": [...] pairs, drop broken last key
    recovered = _recover_columnar_dict(clean)
    if recovered:
        return recovered

    raise Exception("Could not parse JSON response — the model output may have been truncated. Try generating fewer records at a time.")


# ─────────────────────────────────────────────
# SCHEMA / TEMPLATE HELPERS
# ─────────────────────────────────────────────

def _is_schema_spec(obj: dict) -> bool:
    """Return True if the dict looks like a JSON Schema spec document.

    Heuristic: a spec node typically has a 'name'+'description'+'type' triplet
    and/or an 'example' key, rather than holding raw data values.
    We check the top-level keys — if the majority of immediate children are
    dicts that carry 'example' or ('name' AND 'type'), it's a spec.
    """
    if not isinstance(obj, dict):
        return False
    children = [v for v in obj.values() if isinstance(v, dict)]
    if not children:
        return False
    spec_like = sum(
        1 for c in children
        if "example" in c or ("name" in c and "type" in c)
    )
    return spec_like / len(children) >= 0.5


def _spec_to_payload(spec) -> any:
    """Recursively convert a JSON Schema spec node into a sample payload value.

    Rules
    -----
    • If the node is a dict with an 'example' key → return that example value,
      but if the example value itself is a dict/list recurse into it.
    • If the node has 'properties' → build an object from its properties.
    • If the node has 'items' with 'properties' → build a 1-element array.
    • If the node is a plain dict with no spec keys → recurse key-by-key
      (handles top-level wrapper dicts like the inputschema root).
    • Primitive / non-dict nodes are returned as-is.
    """
    if not isinstance(spec, dict):
        return spec

    # Spec node that carries an example: use it (but still recurse if complex)
    if "example" in spec:
        ex = spec["example"]
        if isinstance(ex, (dict, list)) and ex:
            return _spec_to_payload(ex)
        if ex != "" and ex is not None:
            return ex
        # Empty example → fall through to type-based default

    # Has nested properties → build object from them
    if "properties" in spec and isinstance(spec["properties"], dict):
        return {k: _spec_to_payload(v) for k, v in spec["properties"].items()}

    # Has items.properties → 1-element array
    if "items" in spec and isinstance(spec["items"], dict):
        items_node = spec["items"]
        if "properties" in items_node:
            return [_spec_to_payload(items_node)]
        return [_spec_to_payload(items_node)]

    # Type-based fallbacks for leaf nodes with no example
    t = spec.get("type", "")
    fmt = spec.get("format", "")
    if t in ("string",) and fmt in ("date-time", "date"):
        return "2019-09-26T00:00:00.000+0000"
    if t in ("integer", "number"):
        return 0
    if t == "boolean":
        return False
    if t == "array":
        return []

    # Plain dict (e.g. top-level root or a nested group): recurse into values
    result = {}
    for k, v in spec.items():
        # Skip meta-keys that are not data fields
        if k in ("name", "description", "format", "type"):
            continue
        if isinstance(v, (dict, list)):
            result[k] = _spec_to_payload(v)
    return result if result else None


def _spec_root_to_payload(spec: dict) -> dict:
    """Convert the top-level schema spec dict into a single sample payload dict.

    The inputschema.json top level looks like:
      { "application": { "properties": {...} }, "companies": {...}, ... }
    We iterate each top-level key and convert it.
    """
    payload = {}
    for key, node in spec.items():
        if not isinstance(node, dict):
            payload[key] = node
            continue
        converted = _spec_to_payload(node)
        if converted is not None:
            payload[key] = converted
    return payload


def _extract_flat_fields(obj: dict, prefix: str = "") -> dict:
    """Recursively extract {dot.path[].leaf: python_type} from a sample record.
    Leaf arrays of primitives keep their key; arrays of objects recurse into first item.
    """
    result = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_extract_flat_fields(v, key))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            result.update(_extract_flat_fields(v[0], f"{key}[]"))
        elif isinstance(v, list):
            result[key] = f"array of {type(v[0]).__name__ if v else 'str'}"
        else:
            result[key] = type(v).__name__
    return result


def _make_template_structure(obj):
    """Replace all values in a sample record with descriptive type hints.
    Arrays are collapsed to one representative item so the LLM sees the shape.
    """
    if isinstance(obj, dict):
        return {k: _make_template_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_template_structure(obj[0])] if obj else []
    elif isinstance(obj, bool):
        return "<boolean>"
    elif isinstance(obj, int):
        return "<integer>"
    elif isinstance(obj, float):
        return "<float>"
    elif isinstance(obj, str):
        if _re.match(r'\d{4}-\d{2}-\d{2}T', obj):
            return "<ISO-8601 datetime>"
        if '@' in obj and '.' in obj:
            return "<email>"
        if _re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', obj):
            return "<IPv4>"
        if _re.match(r'^\+?[\d\s\-()]+$', obj) and len(obj) >= 7:
            return "<phone>"
        if _re.match(r'^[A-F0-9]{2}(-[A-F0-9]{2}){5}$', obj, _re.I):
            return "<MAC address>"
        return "<string>"
    return "<value>"


# ─────────────────────────────────────────────
# SCHEMA COMPRESSION
# ─────────────────────────────────────────────

def compress_schema(schema) -> dict:
    """Convert any schema (spec doc, flat dict, or sample payload) into a
    compact dot-path dictionary suitable for LLM token-efficient processing.

    Output format:  { "dot.path[].field": "type(format) | description | ex: value" }

    Handles three input shapes:
    1. JSON Schema spec doc  — nodes with 'type', 'description', 'example', 'properties', 'items'
    2. Sample payload        — real data values; types inferred from Python types
    3. Flat {field: type}    — already flat, returned as-is with minor normalisation

    Array fields use the '[]' suffix convention so the LLM understands cardinality.
    Meta-only or empty nodes are skipped to keep the output lean.
    """

    result: dict = {}

    def _field_label(field_name: str) -> str:
        """Convert camelCase / snake_case / PascalCase field name into a
        short readable label that acts as a description hint for the LLM.
        e.g. 'dateOfBirth' → 'Date of birth'
             'usdAmount'   → 'Usd amount'
             'isNew'       → 'Is new flag'
        Keeps the label short (≤ 4 words) to stay token-efficient.
        """
        # Insert space before uppercase letters (camelCase → words)
        s = _re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', field_name)
        # Replace underscores/hyphens with spaces
        s = _re.sub(r'[_\-]+', ' ', s)
        # Lower-case everything then capitalise first letter only
        s = s.strip().lower()
        # Append short semantic suffixes for well-known suffixes
        if s.endswith(" date time") or s.endswith(" date"):
            pass  # already clear
        if s.startswith("is ") or s.startswith("has "):
            s += " flag"
        return s.capitalize()

    def _infer_type(val) -> str:
        if isinstance(val, bool):
            return "boolean"
        if isinstance(val, int):
            return "integer"
        if isinstance(val, float):
            return "number"
        if isinstance(val, list):
            return "array"
        if isinstance(val, dict):
            return "object"
        if isinstance(val, str):
            if _re.match(r'\d{4}-\d{2}-\d{2}T', val):
                return "string(date-time)"
            if "@" in val and "." in val:
                return "string(email)"
            if _re.match(r'^\d{1,3}(\.\d{1,3}){3}$', val):
                return "string(ipv4)"
            if _re.match(r'^\+?[\d\s\-()+]{7,}$', val):
                return "string(phone)"
            if _re.match(r'^[A-F0-9]{2}(-[A-F0-9]{2}){5}$', val, _re.I):
                return "string(mac)"
        return "string"

    def _walk_spec(node, path: str):
        """Walk a JSON Schema spec node."""
        if not isinstance(node, dict):
            return

        node_type = node.get("type", "")
        # type can itself be a nested spec object — normalise to string
        if isinstance(node_type, dict):
            node_type = node_type.get("type", "") or node_type.get("name", "")

        # ── Object with properties ──────────────────────────────────────────
        if "properties" in node and isinstance(node["properties"], dict):
            for k, v in node["properties"].items():
                _walk_spec(v, f"{path}.{k}" if path else k)
            return

        # ── Array ────────────────────────────────────────────────────────────
        if node_type == "array" or "items" in node:
            arr_path = f"{path}[]"
            items = node.get("items", {})
            if isinstance(items, dict):
                if "properties" in items:
                    # Array of objects → recurse into item properties
                    for k, v in items["properties"].items():
                        _walk_spec(v, f"{arr_path}.{k}")
                    # Also walk any non-'properties' siblings inside items
                    # (e.g. 'employment' array or nested 'items' for sub-arrays)
                    for k, v in items.items():
                        if k == "properties":
                            continue
                        if isinstance(v, dict):
                            _walk_spec(v, f"{arr_path}.{k}")
                elif items.get("type") and "properties" not in items:
                    # Array of primitives
                    item_type = items.get("type", "string")
                    desc = node.get("description", "")
                    result[arr_path] = f"{item_type}" + (f" | {desc}" if desc else "")
                else:
                    desc = node.get("description", "")
                    result[arr_path] = "array" + (f" | {desc}" if desc else "")
            else:
                desc = node.get("description", "")
                result[arr_path] = "array" + (f" | {desc}" if desc else "")

            # Walk any additional fields defined directly alongside 'items' on this node
            # (non-standard but present in inputschema — e.g. applicants.firstName,
            #  applicants.locations, applicants.devices, applicants.phones, etc.)
            for k, v in node.items():
                if k in ("name", "description", "format", "type", "example", "items"):
                    continue
                if isinstance(v, dict):
                    _walk_spec(v, f"{arr_path}.{k}")
            return

        # ── Primitive leaf ────────────────────────────────────────────────────
        if node_type and node_type not in ("object",):
            parts = [node_type]
            fmt = node.get("format", "")
            if fmt:
                parts[0] = f"{node_type}({fmt})"
            desc = node.get("description", "")
            if desc:
                parts.append(desc)
            result[path] = " | ".join(parts)
            return

        # ── Nested group without explicit type (walk children) ───────────────
        for k, v in node.items():
            if k in ("name", "description", "format", "type", "example"):
                continue
            if isinstance(v, dict):
                _walk_spec(v, f"{path}.{k}" if path else k)
            elif isinstance(v, list):
                # treat as inline array spec
                _walk_spec({"type": "array", "items": {}, "description": ""}, f"{path}.{k}" if path else k)

    def _walk_payload(node, path: str):
        """Walk a raw sample-data payload and infer types from values."""
        if isinstance(node, dict):
            for k, v in node.items():
                _walk_payload(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            arr_path = f"{path}[]"
            if node and isinstance(node[0], dict):
                _walk_payload(node[0], arr_path)
            else:
                item_type = _infer_type(node[0]) if node else "string"
                result[arr_path] = item_type
        else:
            result[path] = _infer_type(node)

    # ── Dispatch based on detected input shape ────────────────────────────────

    if not isinstance(schema, dict):
        return {}

    values = list(schema.values())

    # Already flat {field: "type-string"} — return as-is
    if all(isinstance(v, str) for v in values):
        return dict(schema)

    # JSON Schema spec doc — nodes carry 'type'/'description'/'example'/'properties'
    if _is_schema_spec(schema):
        for root_key, root_node in schema.items():
            if isinstance(root_node, dict):
                _walk_spec(root_node, root_key)
            else:
                result[root_key] = str(root_node)
        return result

    # Sample payload — infer types from actual values
    _walk_payload(schema, "")
    # _walk_payload puts "" prefix for top-level; clean that up
    cleaned = {}
    for k, v in result.items():
        cleaned[k.lstrip(".")] = v
    return cleaned


# ─────────────────────────────────────────────
# FRAUD PATTERN IDENTIFICATION
# ─────────────────────────────────────────────

def _assign_fraud_patterns(num_records: int, patterns: list) -> list:
    """Pre-assign fraud patterns to each record with these guarantees:
    - Every selected pattern is used in at least one record.
    - Each record gets a random number of patterns (1 to min(3, len(patterns))).
    - Assignments are varied — no two adjacent records get the same single pattern.

    Returns a list of lists, e.g. [["velocity_abuse"], ["geo_ip_mismatch","disposable_email"], ...]
    """
    if not patterns or num_records == 0:
        return [[] for _ in range(num_records)]

    n = num_records
    p = patterns[:]
    max_combo = min(3, len(p))
    assignments = [None] * n

    # Step 1: guarantee every pattern appears at least once.
    # Shuffle patterns and assign one to each of the first len(p) slots (wrapping if n < len(p)).
    shuffled = p[:]
    random.shuffle(shuffled)
    for i, pat in enumerate(shuffled):
        slot = i % n
        if assignments[slot] is None:
            assignments[slot] = [pat]
        elif pat not in assignments[slot]:
            assignments[slot].append(pat)

    # Step 2: fill remaining None slots with random subsets.
    for i in range(n):
        if assignments[i] is None:
            k = random.randint(1, max_combo)
            assignments[i] = random.sample(p, k)

    # Step 3: optionally enrich some records with extra patterns for variety.
    for i in range(n):
        # 40% chance to add one more random pattern if not already at max_combo
        if len(assignments[i]) < max_combo and random.random() < 0.4:
            extras = [x for x in p if x not in assignments[i]]
            if extras:
                assignments[i].append(random.choice(extras))

    return assignments


def get_fraud_patterns(schema: dict, raw_spec: dict = None):
    # Use pre-computed compressed schema from session state if available,
    # otherwise compress on the fly (fallback for direct function calls)
    compressed = (
        st.session_state.get("compressed_schema")
        or compress_schema(raw_spec if raw_spec is not None else schema)
    )

    system = """You are a fraud detection expert. Respond ONLY with a valid JSON array. No markdown, no explanation.
Each item: {"id":"snake_case","label":"Display Name","description":"one line","fields":["field.path"]}"""

    field_names = list(compressed.keys())
    user = f"""Fields:{json.dumps(field_names,separators=(',',':'))}
Suggest fraud patterns embeddable in these field values (e.g. disposable_email if email field exists, unusual_amounts if amount exists, geo_ip_mismatch if ip+country exist, address_inconsistency if address fields exist).
Use exact field paths in "fields". Only suggest patterns for fields that exist."""

    raw = call_llm(system, user, usage_label="identify_patterns")

    return parse_json_response(raw)


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def _build_nested_template(schema_fields: list) -> tuple[dict, dict]:
    """Build a NESTED template and short-key mapping from dot-path schema fields.

    Each leaf in the nested template gets a short key like {k1}, {k2}, etc.
    Array segments (ending with []) produce a single-element list in the template.

    Returns:
        template: nested dict mirroring schema structure
                  e.g. {"application": {"amount": "{k1}", "channel": "{k2}"}}
        mapping:  {"k1": "application.amount", "k2": "application.channel"}
    """
    template: dict = {}
    mapping: dict = {}

    for i, field in enumerate(schema_fields):
        key = f"k{i + 1}"
        mapping[key] = field

        parts = field.split(".")
        node = template

        for part in parts[:-1]:
            is_arr = part.endswith("[]")
            clean = part[:-2] if is_arr else part
            if is_arr:
                if clean not in node:
                    node[clean] = [{}]
                elif not isinstance(node[clean], list) or not node[clean]:
                    node[clean] = [{}]
                node = node[clean][0]
            else:
                if clean not in node or not isinstance(node[clean], dict):
                    node[clean] = {}
                node = node[clean]

        leaf = parts[-1]
        is_arr_leaf = leaf.endswith("[]")
        clean_leaf = leaf[:-2] if is_arr_leaf else leaf
        node[clean_leaf] = [f"{{{key}}}"] if is_arr_leaf else f"{{{key}}}"

    return template, mapping


def _set_dot_path(record: dict, dot_path: str, value) -> None:
    """Set a value at a dot-path like 'application.amount' in a nested dict.
    Handles [] array notation by creating/navigating single-element lists.
    """
    parts = dot_path.split(".")
    node = record

    for part in parts[:-1]:
        is_arr = part.endswith("[]")
        clean = part[:-2] if is_arr else part
        if is_arr:
            if clean not in node:
                node[clean] = [{}]
            elif not isinstance(node[clean], list) or not node[clean]:
                node[clean] = [{}]
            node = node[clean][0]
        else:
            if clean not in node or not isinstance(node[clean], dict):
                node[clean] = {}
            node = node[clean]

    leaf = parts[-1]
    is_arr_leaf = leaf.endswith("[]")
    clean_leaf = leaf[:-2] if is_arr_leaf else leaf
    node[clean_leaf] = value if isinstance(value, list) else [value] if is_arr_leaf else value



# Mapping from field-path fragment → realistic example string to inject into the prompt.
# Used to override placeholder examples like "cpu1", "model1" that come from the schema.
_REALISTIC_EXAMPLES: dict = {
    # Device hardware
    "cpu": "Apple A17 Pro",
    "model": "iPhone 15 Pro",
    "modelversion": "15.0.1",
    "operatingsystem": "iOS 17.3",
    "osversion": "17.3.1",
    "browsername": "Safari",
    "browserversion": "17.2",
    "browserlanguage": "en-US",
    "screencolor": "32-bit",
    "screenheight": "2556",
    "screenwidth": "1179",
    "devicemanufactureridentifier": "APPLE_PROD_001",
    "deviceid": "A3F92B1D4E7C",
    # IP fields
    "ipaddress": "203.0.113.42",
    "ipaddressv6": "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
    "ipcity": "San Francisco",
    "ipcountrycode": "US",
    "ipdistrict": "Financial District",
    "ipdomain": "comcast.net",
    "ipipisp": "Comcast Cable",
    "ipisp": "Comcast Cable",
    "iplatitude": "37.7749",
    "iplongitude": "-122.4194",
    "ipoffsetzt": "-8.0",
    "ippostcode": "94105",
    "ipregion": "California",
    # GPS
    "gpslat": "37.7749",
    "gpslon": "-122.4194",
    "gpslatitude": "37.7749",
    "gpslongitude": "-122.4194",
    # Network
    "datanetworkname": "Verizon LTE",
    "wifinetworkname": "HomeNetwork_5G",
    "macaddress": "A4:C3:F0:85:AC:ED",
    # Bluetooth
    "bluetoothconnections": "AirPods-Pro-A1",
    "bluetoothinrange": "JBL-Flip-5",
    # JavaScript
    "javascriptind": "enabled",
    # Phone parts
    "areacode": "415",
    "e164": "+14155552671",
    "e164hash": "7B3A9F2E1D4C8B56",
    "prefix": "415",
    "extension": "101",
    "line": "5552671",
    "linetype": "mobile",
    "phoneid": "PHN-8A2F",
    # Names
    "firstname": "James",
    "lastname": "Smith",
    "middlename": "Robert",
    "fullname": "James Robert Smith",
    "canonicalname": "james.smith",
    "phoneticname": "JMSMITH",
    "previousfirstname": "Jim",
    "previouslastname": "Smyth",
    "previousname": "Jim Smyth",
    "username": "jsmith92",
    # Geography
    "city": "San Francisco",
    "town": "Marin",
    "state": "California",
    "county": "San Francisco County",
    "region": "West Coast",
    "country": "United States",
    "countrycode": "US",
    "countryofissue": "United States",
    "postcode": "94105",
    "street": "123 Market St",
    "fulladdress": "123 Market St, San Francisco, CA 94105",
    "canonicaladdress": "123 market st san francisco ca 94105",
    "buildingname": "Salesforce Tower",
    "buildingnumber": "101",
    "floor": "12",
    "postbox": "PO Box 4210",
    # Employment
    "employername": "TechCorp Inc",
    "employerindustrytype": "Technology",
    "employerfulladress": "456 Tech Blvd, San Jose, CA 95110",
    "employerfuladdress": "456 Tech Blvd, San Jose, CA 95110",
    "employeejobtitle": "Senior Software Engineer",
    "employeeidentifier": "EMP-20941",
    "incomecurrency": "USD",
    "incomefrequency": "monthly",
    "salaryfrequency": "monthly",
    "designation": "Mr",
    "educationlevel": "Bachelor's Degree",
    "maritalstatus": "married",
    "residentialstatus": "owner",
    "incomestatustype": "employed",
    # Finance
    "annualincome": "72000.00",
    "monthlyregularincome": "6000.00",
    "monthlyregularoutgoing": "2200.00",
    "outstandingdebt": "15000.00",
    "outstandingdebtcurrency": "USD",
    # IDs / hashes
    "entityid": "ENT-F3A92B",
    "identifier": "ID-7C3F9A",
    "identifierdesc": "Primary ID",
    "identifierchecksum": "A3F9B2C1D4E5",
    "identifierhash": "SHA256:7b3a9f2e1d4c8b56af",
    "identityhashstrict": "SHA256:a1b2c3d4e5f60718",
    "identityhashrelaxed": "SHA256:1a2b3c4d5e6f7081",
    "serialnumber": "SN-20419-XA",
    "serialnumberhash": "HASH-9F2E1D4C",
    # Identification docs
    "documenttype": "Passport",
    "nationality": "American",
    "issuedby": "US Department of State",
    "canonicalissuedby": "us dept of state",
    "mrz": "P<USASMITH<<JAMES<ROBERT",
    "mrzchecksum": "P<USASMITH<<JAMES<ROBERT<<<<<0",
    # Account
    "accountid": "ACC-8472910",
    "userid": "USR-29401",
    "primaryapplicantref": "APP-10294",
    # Company
    "legaltype": "LLC",
    "officephone": "+1-415-555-0100",
    "joindate": "2020-03-15T09:00:00",
    "registrationdate": "2019-01-10T00:00:00",
    # Status / misc
    "status": "active",
    "language": "en",
    "languagecode": "en-US",
    "gender": "male",
    "gmtoffset": "-8",
    "vipflg": "false",
    "domain": "gmail.com",
    "observed": "42",
    "entityind": "Y",
    "entityvariable": "VAR-001",
    "providername": "Experian",
}

def _is_placeholder(val: str) -> bool:
    """Return True if val looks like a schema placeholder (e.g. 'cpu1', 'model_version1', 'ip6 1')."""
    if not isinstance(val, str):
        return False
    # Matches patterns like: word1, word_word1, word word1, word version1
    return bool(_re.fullmatch(r'[\w\s][\w\s]*\d+', val.strip()))


def _realistic_example(field_path: str, schema_example) -> str | None:
    """Return a realistic example for the field, overriding placeholder schema examples."""
    # Normalise path to last segment, lowercase, no brackets
    segment = field_path.lower().replace("[]", "").split(".")[-1]
    # 1. If schema example is NOT a placeholder, use it directly
    if schema_example is not None and not _is_placeholder(str(schema_example)):
        s = str(schema_example)
        if len(s) <= 35:
            return s
    # 2. Look up by exact segment match
    if segment in _REALISTIC_EXAMPLES:
        return _REALISTIC_EXAMPLES[segment]
    # 3. Substring match (longest key that appears in segment wins)
    best = max(
        (k for k in _REALISTIC_EXAMPLES if k in segment),
        key=len, default=None
    )
    if best:
        return _REALISTIC_EXAMPLES[best]
    return None


def _fallback_value(field_path: str, type_hint: str = ""):
    """Return a realistic non-null fallback for a field when the LLM skipped it."""

    f = field_path.lower().replace("[]", "").replace(".", "_")
    t = (type_hint or "string").lower()

    _first_names = ["James","Maria","John","Patricia","Robert","Jennifer","Michael","Linda","David","Susan"]
    _last_names  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Wilson","Taylor"]

    # ── Boolean (by type OR field name pattern) ──────────────────────────────
    if ("boolean" in t or "bool" in t
            or f.endswith("flag") or f.endswith("flg") or f.endswith("ind")
            or f.startswith("is_") or f.startswith("has_")):
        return random.choice([True, False])

    # ── GPS / coordinates (field-name first, regardless of declared type) ────
    if any(x in f for x in ["gpslat", "iplat", "latitude"]):
        return round(random.uniform(-90, 90), 6)
    if any(x in f for x in ["gpslon", "iplon", "longitude"]):
        return round(random.uniform(-180, 180), 6)

    # ── IP addresses ──────────────────────────────────────────────────────────
    if any(x in f for x in ["ipaddressv6", "ipv6", "addressv6"]):
        seg = lambda: format(random.randint(0, 65535), "x")
        return ":".join(seg() for _ in range(8))
    if any(x in f for x in ["ipaddress", "ip_address"]) or f.endswith("_ip") or f == "ipaddress":
        return f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    # ── MAC address ───────────────────────────────────────────────────────────
    if "macaddress" in f or "mac_address" in f:
        return ":".join(format(random.randint(0, 255), "02X") for _ in range(6))

    # ── Device fields ─────────────────────────────────────────────────────────
    if "deviceid" in f or "device_id" in f:
        return _uuid.uuid4().hex[:12].upper()
    if "operatingsystem" in f or f == "os":
        return random.choice(["Android 13", "iOS 17", "Windows 11", "macOS Ventura", "Ubuntu 22.04"])
    if "osversion" in f:
        return random.choice(["13.1", "17.2", "11.0", "22.04", "14.5"])
    if "browserversion" in f:
        return random.choice(["120.0", "119.3", "18.0", "109.0", "98.0.4758"])
    if "browsername" in f:
        return random.choice(["Chrome", "Firefox", "Safari", "Edge", "Samsung Internet"])
    if "browserlanguage" in f:
        return random.choice(["en-US", "en-GB", "fr-FR", "de-DE", "es-ES", "zh-CN"])
    if "modelversion" in f:
        return f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}"
    if f.endswith("_model") or f == "model":
        return random.choice(["iPhone 15", "Samsung Galaxy S24", "Pixel 8", "OnePlus 12", "Xiaomi 14"])
    if "cpu" in f:
        return random.choice(["Apple A17", "Snapdragon 8 Gen 3", "MediaTek Dimensity 9300", "Intel Core i7"])
    if "screencolor" in f or "screen_color" in f:
        return random.choice(["24-bit", "32-bit", "16-bit"])
    if "screenheight" in f:
        return random.choice([2532, 2778, 2400, 1920, 2340])
    if "screenwidth" in f:
        return random.choice([1170, 1284, 1080, 1440, 1080])
    if "datanetworkname" in f or "data_network" in f:
        return random.choice(["Verizon LTE", "AT&T 5G", "T-Mobile 5G", "EE 4G", "Vodafone 5G"])
    if any(x in f for x in ["wifinetworksinrange", "wifinetworkname", "wifi_network"]):
        return random.choice(["HomeNetwork_5G", "OfficeWiFi", "CoffeeShop_Guest", "Airport_Free"])
    if any(x in f for x in ["bluetoothconnections", "bluetoothinrange", "bluetooth_"]):
        return random.choice(["AirPods-ABC1", "Galaxy Buds-XY9", "JBL-Speaker-01"])
    if "devicemanufacturer" in f or "manufacturer" in f:
        return random.choice(["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Huawei"])

    # ── IP geo/ISP fields ─────────────────────────────────────────────────────
    if "ipisp" in f or "ip_isp" in f:
        return random.choice(["Comcast", "Verizon", "AT&T", "BT", "Deutsche Telekom", "Airtel"])
    if "ipdistrict" in f:
        return random.choice(["Downtown", "Midtown", "Westside", "East End", "Harbor District"])
    if "ipcountrycode" in f:
        return random.choice(["US", "GB", "CA", "AU", "DE", "FR", "IN", "JP"])
    if "ipcity" in f:
        return random.choice(["New York", "London", "Toronto", "Sydney", "Berlin", "Mumbai"])
    if "ipregion" in f:
        return random.choice(["California", "New York", "Texas", "Ontario", "Bavaria"])
    if "ippostcode" in f or "ip_postcode" in f:
        return str(random.randint(10000, 99999))
    if "ipoffsetzt" in f or "ip_offset" in f:
        return random.choice([-8.0, -5.0, 0.0, 1.0, 5.5, 9.0])
    if "ipdomain" in f:
        return random.choice(["comcast.net", "verizon.net", "att.net", "bt.com", "airtel.in"])

    # ── Phone number parts ────────────────────────────────────────────────────
    if any(x in f for x in ["areacode", "area_code"]):
        return str(random.randint(200, 999))
    if any(x in f for x in ["e164hash"]):
        return _uuid.uuid4().hex[:16].upper()
    if any(x in f for x in ["e164"]):
        return f"+1{random.randint(2000000000, 9999999999)}"
    if any(x in f for x in ["prefix", "extension"]):
        return str(random.randint(100, 999))
    if "linetype" in f or "line_type" in f:
        return random.choice(["mobile", "landline", "voip", "toll-free"])
    if f.endswith("_line") or f == "line":
        return str(random.randint(1000, 9999))
    if "phoneid" in f or "phone_id" in f:
        return _uuid.uuid4().hex[:8].upper()

    # ── Integer (by type) ─────────────────────────────────────────────────────
    if "integer" in t or t == "int":
        if "age" in f: return random.randint(18, 75)
        if any(x in f for x in ["year", "birthyear"]): return random.randint(1950, 2005)
        if any(x in f for x in ["month", "birthmonth"]): return random.randint(1, 12)
        if any(x in f for x in ["day", "birthday"]): return random.randint(1, 28)
        if "dependents" in f: return random.randint(0, 4)
        if any(x in f for x in ["duration", "term", "months"]): return random.randint(6, 60)
        if any(x in f for x in ["count", "num", "quantity", "observed"]): return random.randint(1, 50)
        return random.randint(1, 100)

    # ── Number / float (by type) ──────────────────────────────────────────────
    if "number" in t or "float" in t or "decimal" in t:
        if any(x in f for x in ["amount", "income", "salary", "balance", "price", "usd"]):
            return round(random.uniform(1000, 80000), 2)
        if any(x in f for x in ["gmtoffset", "gmt_offset", "offsettz"]):
            return round(random.choice([-8, -5, 0, 1, 5.5, 9]), 2)
        if "observed" in f: return round(random.uniform(1, 100), 2)
        if any(x in f for x in ["speed", "interval", "time", "press", "release"]):
            return round(random.uniform(0.1, 200.0), 2)
        if any(x in f for x in ["x", "y"]) and len(f) <= 10:
            return round(random.uniform(0, 1920), 2)
        return round(random.uniform(0, 100), 2)

    # ── Date-time ─────────────────────────────────────────────────────────────
    if "date-time" in t or "datetime" in t:
        base = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=random.randint(0, 2000), hours=random.randint(0, 23))
        return base.strftime("%Y-%m-%dT%H:%M:%S")

    # ── Date (string) ─────────────────────────────────────────────────────────
    if t == "string(date)" or ("date" in f and "time" not in f and "datetime" not in f):
        base = datetime.date(1970, 1, 1) + datetime.timedelta(days=random.randint(0, 20000))
        return base.strftime("%Y-%m-%d")

    # ── Email ─────────────────────────────────────────────────────────────────
    if "email" in t or "email" in f or "mail" in f:
        first = random.choice(["james", "maria", "alex", "sam", "chris", "pat"])
        return f"{first}{random.randint(10,99)}@{random.choice(['gmail.com','yahoo.com','outlook.com','hotmail.com'])}"

    # ── Phone ─────────────────────────────────────────────────────────────────
    if "phone" in t or any(x in f for x in ["phone", "mobile", "tel", "cell"]):
        return f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

    # ── Names ─────────────────────────────────────────────────────────────────
    if "firstname" in f or f.endswith("_first"):
        return random.choice(_first_names)
    if "lastname" in f or "surname" in f or f.endswith("_last"):
        return random.choice(_last_names)
    if "middlename" in f or "middle_name" in f:
        return random.choice(_first_names)
    if "previousfirstname" in f or "previouslastname" in f or "previousname" in f:
        return f"{random.choice(_first_names)} {random.choice(_last_names)}"
    if any(x in f for x in ["fullname", "full_name", "canonicalname"]):
        return f"{random.choice(_first_names)} {random.choice(_last_names)}"
    if "phoneticname" in f:
        return f"{random.choice(_first_names)} {random.choice(_last_names)}"
    if f.endswith("name") and not any(x in f for x in ["employer", "building", "user", "domain", "wifi", "data", "browser"]):
        return f"{random.choice(_first_names)} {random.choice(_last_names)}"

    # ── Gender ────────────────────────────────────────────────────────────────
    if "gender" in f:
        return random.choice(["male", "female", "other"])

    # ── Geography ─────────────────────────────────────────────────────────────
    if any(x in f for x in ["countryofissue", "countryofissuecode"]):
        return random.choice(["US", "GB", "CA", "AU", "DE"])
    if any(x in f for x in ["countrycode", "country_code"]):
        return random.choice(["US", "GB", "CA", "AU", "DE", "FR", "JP", "IN"])
    if any(x in f for x in ["country", "nation"]):
        return random.choice(["United States", "United Kingdom", "Canada", "Australia", "Germany"])
    if any(x in f for x in ["statecode", "state_code"]):
        return random.choice(["CA", "TX", "NY", "FL", "IL", "PA"])
    if any(x in f for x in ["state", "region", "county"]):
        return random.choice(["California", "Texas", "New York", "Florida", "Illinois"])
    if any(x in f for x in ["city", "town"]):
        return random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
    if any(x in f for x in ["postcode", "zip", "postal"]):
        return str(random.randint(10000, 99999))
    if any(x in f for x in ["fulladdress", "full_address", "canonicaladdress", "employerfulladdress"]):
        return f"{random.randint(1, 9999)} {random.choice(['Main','Oak','Elm','Park','Lake','River'])} St"
    if "street" in f:
        return f"{random.randint(1, 9999)} {random.choice(['Main','Oak','Elm','Park','Lake','River'])} St"
    if "buildingname" in f or "building_name" in f:
        return f"Building {random.choice(['A','B','C','D'])}"
    if any(x in f for x in ["buildingnumber", "building_number", "floor"]):
        return str(random.randint(1, 50))
    if any(x in f for x in ["postbox", "post_box"]):
        return f"PO Box {random.randint(100, 9999)}"

    # ── Finance / employment ──────────────────────────────────────────────────
    if any(x in f for x in ["currencycode", "currency_code", "incomecurrency"]):
        return random.choice(["USD", "EUR", "GBP", "CAD", "AUD"])
    if "currency" in f:
        return random.choice(["US Dollar", "Euro", "British Pound"])
    if any(x in f for x in ["incomefrequency", "income_frequency", "salaryfrequency", "salary_frequency"]):
        return random.choice(["monthly", "annual", "weekly", "bi-weekly"])
    if any(x in f for x in ["employername", "employer_name"]):
        return random.choice(["Acme Corp", "TechSolutions Ltd", "Global Finance Inc", "Metro Services"])
    if any(x in f for x in ["employerindustry", "industrytype", "industry_type"]):
        return random.choice(["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"])
    if any(x in f for x in ["jobtitle", "job_title", "designation", "employeejobtitle"]):
        return random.choice(["Software Engineer", "Analyst", "Manager", "Director", "Consultant"])
    if any(x in f for x in ["educationlevel", "education_level", "education"]):
        return random.choice(["High School", "Bachelor's Degree", "Master's Degree", "Associate's Degree"])
    if any(x in f for x in ["maritalstatus", "marital_status"]):
        return random.choice(["single", "married", "divorced", "widowed"])
    if any(x in f for x in ["residentialstatus", "residential_status", "incomestatus"]):
        return random.choice(["owner", "renter", "living with family", "other"])
    if "employeeidentifier" in f or "employee_identifier" in f:
        return f"EMP{random.randint(10000, 99999)}"

    # ── Biometrics ────────────────────────────────────────────────────────────
    if "biometricreadingid" in f or "biometric_reading_id" in f:
        return _uuid.uuid4().hex[:8].upper()
    if "useridentifier" in f or "user_identifier" in f:
        return _uuid.uuid4().hex[:8].upper()
    if any(x in f for x in ["averagekeypress", "averagekeyrelease", "keypressinterval"]):
        return round(random.uniform(50, 200), 2)
    if any(x in f for x in ["averagespeed", "average_speed"]):
        return round(random.uniform(0.1, 5.0), 3)
    if "button" in f:
        return random.choice(["left", "right", "middle"])

    # ── Demographics ──────────────────────────────────────────────────────────
    if any(x in f for x in ["newdata", "new_data", "previousdata", "previous_data", "entityvariable"]):
        return f"DATA{random.randint(100, 999)}"
    if "entityind" in f:
        return random.choice(["Y", "N"])
    if "typedesc" in f or "type_desc" in f:
        return random.choice(["Standard", "Premium", "Basic", "Other"])
    if "providername" in f or "provider_name" in f:
        return random.choice(["DataProvider Inc", "InfoSource LLC", "VerifyPro", "IdentityCheck"])

    # ── Identification docs ───────────────────────────────────────────────────
    if any(x in f for x in ["documenttype", "document_type"]):
        return random.choice(["Passport", "Driver License", "National ID", "State ID"])
    if "nationality" in f:
        return random.choice(["American", "British", "Canadian", "Australian", "German"])
    if "mrzchecksumok" in f or "mrzchecksum" in f:
        return f"P<USA{random.choice(_last_names).upper()}<<{random.choice(_first_names).upper()}"
    if "mrz" in f:
        return f"P<USA{random.choice(_last_names).upper()}<<{random.choice(_first_names).upper()}"
    if "issuedby" in f or "issued_by" in f or "canonicalissuedby" in f:
        return random.choice(["US Dept of State", "DVLA", "Transport Canada", "Home Affairs"])
    if "details" in f:
        return f"REF{random.randint(10000, 99999)}"
    if "alternatenumber" in f or "alternate_number" in f:
        return f"ALT{random.randint(1000, 9999)}"

    # ── Generic identifiers / hashes ──────────────────────────────────────────
    if any(x in f for x in ["hash", "checksum", "serialnumber", "serial_number"]):
        return _uuid.uuid4().hex[:16].upper()
    if any(x in f for x in ["identifier", "entityid", "entity_id"]) and not "desc" in f:
        return _uuid.uuid4().hex[:8].upper()
    if "userid" in f or "user_id" in f:
        return f"USR{random.randint(10000, 99999)}"
    if "accountid" in f or "account_id" in f:
        return f"ACC{random.randint(100000, 999999)}"

    # ── Status / category ─────────────────────────────────────────────────────
    if "status" in f:
        return random.choice(["active", "pending", "approved", "verified"])
    if any(x in f for x in ["category", "subtype", "sub_type"]):
        return random.choice(["standard", "premium", "basic", "other"])
    if "channel" in f:
        return random.choice(["online", "mobile", "branch", "api"])
    if "type" in f and not any(x in f for x in ["date", "time"]):
        return random.choice(["standard", "premium", "basic", "other"])
    if any(x in f for x in ["stage", "stagedesc", "statusdesc"]):
        return random.choice(["Submitted", "Under Review", "Approved", "Completed"])
    if any(x in f for x in ["language", "languagecode"]):
        return random.choice(["en", "es", "fr", "de", "zh"])
    if any(x in f for x in ["description", "desc", "purpose"]):
        return "Standard entry"
    if "username" in f:
        return f"user{random.randint(1000, 9999)}"
    if "domain" in f:
        return random.choice(["gmail.com", "yahoo.com", "company.com", "outlook.com"])
    if any(x in f for x in ["legaltype", "legal_type"]):
        return random.choice(["LLC", "Corporation", "Partnership", "Sole Proprietor"])
    if any(x in f for x in ["gmtoffset", "gmt_offset", "offset"]):
        return random.choice([-8.0, -5.0, 0.0, 1.0, 5.5, 9.0])
    if "number" in f and not any(x in f for x in ["phone", "serial", "alternate", "checksum"]):
        return f"{random.randint(10000, 99999)}"
    if "observed" in f:
        return random.randint(1, 100)

    # ── Default: use a descriptive word based on field name tail ──────────────
    tail = f.split("_")[-1] if "_" in f else f
    return f"{tail[:8]}_{random.randint(100, 999)}"


def _columnar_to_nested_records(response: dict, mapping: dict, num_records: int, typed_map: dict = None) -> list:
    """Convert flat columnar LLM response into a list of nested records.

    response:  {"k1": [v0, v1, ...], "k2": [v0, v1, ...], ...}
    mapping:   {"k1": "application.amount", "k2": "application.channel", ...}
    typed_map: {"k1": "application.amount:number", ...}  (optional, used for null fallbacks)
    """
    records = []
    for i in range(num_records):
        record: dict = {}
        for key, dot_path in mapping.items():
            values = response.get(key, [])
            value = values[i] if i < len(values) else None
            if value is None:
                type_hint = ""
                if typed_map and key in typed_map:
                    raw = typed_map[key]  # e.g. "application.amount|number|USD amount|ex:1000.0"
                    parts = raw.split("|")
                    # parts[0]=path, parts[1]=type, rest=desc/example
                    type_hint = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                value = _fallback_value(dot_path, type_hint)
            _set_dot_path(record, dot_path, value)
        records.append(record)
    return records


def _schema_aware_batch_target(num_fields: int, remaining: int) -> int:
    """Choose a starting batch size from schema width and remaining count.

    Strategy:
    - For small remaining counts, try single-call first to avoid repeated prompt cost.
    - For larger counts, estimate a safe cap from schema width.
    - Runtime backoff in generate_batch still halves on parse failures.
    """
    if remaining <= 8:
        return remaining

    # Rough output-token estimate per record in columnar JSON.
    tokens_per_record = max(80, num_fields * 7)
    output_budget = 6000
    cap = max(1, output_budget // tokens_per_record)
    return max(1, min(remaining, cap, 40))


def generate_synthetic_data(
    schema: dict,
    num_records: int,
    fraud_pct: int,
    patterns: list[str]
):

    legit_count = round(num_records * (100 - fraud_pct) / 100)
    fraud_count = num_records - legit_count

    schema_fields = list(schema.keys())

    # Build nested template + short-key mapping
    nested_template, mapping = _build_nested_template(schema_fields)

    # Store for UI debug display
    st.session_state._debug_template = nested_template
    st.session_state._debug_llm_responses = []

    # Build compact field map: {"k1":"path:type:example"}
    # Uses _realistic_example() to replace placeholder schema examples with real values
    def _compact_entry(path, schema_val):
        parts = [p.strip() for p in schema_val.split("|")]
        type_str = parts[0]
        raw_ex = next((p[3:].strip() for p in parts[1:] if p.strip().startswith("ex:")), None)
        # Parse raw_ex from JSON if needed
        try:
            raw_ex_parsed = json.loads(raw_ex) if raw_ex else None
        except Exception:
            raw_ex_parsed = raw_ex
        ex = _realistic_example(path, raw_ex_parsed)
        if ex:
            return f"{path}:{type_str}:{ex}"
        return f"{path}:{type_str}"

    rich_map = {k: _compact_entry(v, schema.get(v, "string")) for k, v in mapping.items()}
    compact_map = json.dumps(rich_map, separators=(",", ":"))
    typed_map = rich_map  # used for null fallback type extraction

    system = """Output one JSON object: each placeholder key (k1,k2,...) → array of values. No markdown.
Rules: exact requested count per key; generate realistic values matching each field's type and example; never use null; _fraud_pattern is array of arrays."""

    def generate_batch(sys_prompt, base_user_prompt, count):
        """Generate records in batches using the nested template / short-key approach."""
        all_rows = []
        remaining = count
        while remaining > 0:
            target_batch = _schema_aware_batch_target(len(schema_fields), remaining)
            attempt_batch = target_batch
            while attempt_batch >= 1:
                # Cap tokens to the expected output size: ~15 tokens per field per record
                # This prevents the LLM from generating far more values than requested
                _max_tokens = min(8192, max(1024, len(schema_fields) * attempt_batch * 15 + 800))
                prompt = base_user_prompt.replace(
                    f"Generate {count} ", f"Generate {attempt_batch} "
                ).replace(
                    f"array of {count} values", f"array of {attempt_batch} values"
                ).replace(
                    f"key→array of {count} values", f"key→array of {attempt_batch} values"
                )
                raw = call_llm(sys_prompt, prompt, max_tokens=_max_tokens, usage_label="generate_batch")

                if "_debug_llm_responses" not in st.session_state:
                    st.session_state._debug_llm_responses = []
                st.session_state._debug_llm_responses.append({"batch": attempt_batch, "raw": raw})

                try:
                    result = parse_json_response(raw)
                except Exception as e:
                    if attempt_batch == 1:
                        raise Exception(f"Failed to parse LLM response (batch of {attempt_batch}): {e}\n\nRaw response (first 500 chars):\n{raw[:500]}")
                    attempt_batch = max(1, attempt_batch // 2)
                    continue

                if isinstance(result, list):
                    # LLM returned row-array — extract values back into columnar form
                    # so ALL mapped fields are filled (via fallback if LLM skipped them)
                    col = {k: [] for k in mapping}
                    for row in result[:attempt_batch]:
                        flat = {}
                        def _flatten_row(node, pfx=""):
                            if isinstance(node, dict):
                                for rk, rv in node.items():
                                    _flatten_row(rv, f"{pfx}.{rk}" if pfx else rk)
                            elif isinstance(node, list):
                                if node:
                                    _flatten_row(node[0], pfx)
                            else:
                                flat[pfx] = node
                        _flatten_row(row)
                        # Reverse-map flat field names to k-keys
                        inv = {v.split(":")[0]: k for k, v in rich_map.items()}
                        for path_key, kk in inv.items():
                            # Try exact match or last-segment match
                            val = flat.get(path_key) or flat.get(path_key.split(".")[-1].replace("[]", ""))
                            col[kk].append(val)
                    result = col
                    # Fall through to dict handling below

                if isinstance(result, dict):
                    # Strip _fraud_pattern if LLM included it — we attach pre-assigned ones after
                    result.pop("_fraud_pattern", None)

                    # Clip arrays to exact batch size — LLM sometimes returns more values
                    # than requested, which causes huge responses that get truncated
                    for _k in list(result.keys()):
                        if isinstance(result[_k], list):
                            result[_k] = result[_k][:attempt_batch]

                    # Reconstruct nested records using the key → dot-path mapping
                    rows = _columnar_to_nested_records(result, mapping, attempt_batch, typed_map=typed_map)

                    all_rows.extend(rows)
                    remaining -= attempt_batch
                    break

                # Unexpected parsed shape — retry with a smaller batch
                if attempt_batch == 1:
                    raise Exception(f"Failed to parse LLM response (batch of {attempt_batch}): Unsupported JSON shape\n\nRaw response (first 500 chars):\n{raw[:500]}")
                attempt_batch = max(1, attempt_batch // 2)
        return all_rows

    fraud_records = []
    if fraud_count > 0:
        # Pre-assign patterns to each record: guarantees all patterns used, random combos
        fraud_assignments = _assign_fraud_patterns(fraud_count, patterns)
        assignment_str = json.dumps(fraud_assignments, separators=(",", ":"))
        fraud_user = f"""Generate {fraud_count} fraud records.
Field map (key→path|type|description|example):{compact_map}
Per-record pattern assignment (index matches record position):{assignment_str}
Embed the assigned fraud pattern anomalies subtly into the field values for each record.
Output one JSON object: each key→array of {fraud_count} values. Never return null."""
        fraud_records = generate_batch(system, fraud_user, fraud_count)
        # Attach pre-assigned patterns directly — don't rely on LLM output
        for i, rec in enumerate(fraud_records):
            rec["_fraud_pattern"] = fraud_assignments[i] if i < len(fraud_assignments) else []

    legit_records = []
    if legit_count > 0:
        legit_user = f"""Generate {legit_count} legitimate records.
Field map (key→path|type|description|example):{compact_map}
Output one JSON object: each key→array of {legit_count} values. Never return null. All values realistic and internally consistent."""
        legit_records = generate_batch(system, legit_user, legit_count)

    return {"fraud": fraud_records, "legit": legit_records}


# ─────────────────────────────────────────────
# TEMPLATE-BASED GENERATION (deeply nested schemas)
# ─────────────────────────────────────────────

def generate_from_template(
    template: dict,
    num_records: int,
    fraud_pct: int,
    patterns: list[str],
    raw_spec: dict = None,
) -> dict:
    """Generate records for nested schemas using the same short-key columnar approach.

    Uses compress_schema on raw_spec (or template) to derive the canonical field list,
    builds a nested template with {k1}, {k2} ... placeholders, asks the LLM for a
    flat columnar response, then reconstructs the proper nested records.
    """
    legit_count = round(num_records * (100 - fraud_pct) / 100)
    fraud_count = num_records - legit_count

    # Use pre-computed compressed schema from session state (computed once on upload)
    compressed = (
        st.session_state.get("compressed_schema")
        or compress_schema(raw_spec if raw_spec is not None else template)
    )
    schema_fields = list(compressed.keys())

    # Build nested template with short keys + reverse mapping
    nested_template, mapping = _build_nested_template(schema_fields)

    # Store for UI debug display
    st.session_state._debug_template = nested_template
    st.session_state._debug_llm_responses = []

    # Build compact field map: {"k1":"path:type:example"}
    # Uses _realistic_example() to replace placeholder schema examples with real values
    def _compact_entry(path, schema_val):
        parts = [p.strip() for p in schema_val.split("|")]
        type_str = parts[0]
        raw_ex = next((p[3:].strip() for p in parts[1:] if p.strip().startswith("ex:")), None)
        try:
            raw_ex_parsed = json.loads(raw_ex) if raw_ex else None
        except Exception:
            raw_ex_parsed = raw_ex
        ex = _realistic_example(path, raw_ex_parsed)
        if ex:
            return f"{path}:{type_str}:{ex}"
        return f"{path}:{type_str}"

    rich_map = {k: _compact_entry(v, compressed.get(v, "string")) for k, v in mapping.items()}
    compact_map = json.dumps(rich_map, separators=(",", ":"))
    typed_map = rich_map  # used for null fallback type extraction

    system = """Output one JSON object: each placeholder key (k1,k2,...) → array of values. No markdown.
Rules: exact requested count per key; generate realistic values matching each field's type and example; never use null; _fraud_pattern is array of arrays."""

    def generate_batch(sys_prompt, base_user_prompt, count):
        all_rows = []
        remaining = count
        while remaining > 0:
            target_batch = _schema_aware_batch_target(len(schema_fields), remaining)
            attempt_batch = target_batch
            while attempt_batch >= 1:
                # Cap tokens to the expected output size: ~15 tokens per field per record
                # This prevents the LLM from generating far more values than requested
                _max_tokens = min(8192, max(1024, len(schema_fields) * attempt_batch * 15 + 800))
                prompt = base_user_prompt.replace(
                    f"Generate {count} ", f"Generate {attempt_batch} "
                ).replace(
                    f"key\u2192array of {count} values", f"key\u2192array of {attempt_batch} values"
                )
                raw = call_llm(sys_prompt, prompt, max_tokens=_max_tokens, usage_label="generate_batch")

                if "_debug_llm_responses" not in st.session_state:
                    st.session_state._debug_llm_responses = []
                st.session_state._debug_llm_responses.append({"batch": attempt_batch, "raw": raw})

                try:
                    result = parse_json_response(raw)
                except Exception as e:
                    if attempt_batch == 1:
                        raise Exception(f"Failed to parse LLM response (batch of {attempt_batch}): {e}\n\nRaw (first 500 chars):\n{raw[:500]}")
                    attempt_batch = max(1, attempt_batch // 2)
                    continue

                if isinstance(result, list):
                    # LLM returned row-array — extract values back into columnar form
                    col = {k: [] for k in mapping}
                    for row in result[:attempt_batch]:
                        flat = {}
                        def _flatten_row(node, pfx=""):
                            if isinstance(node, dict):
                                for rk, rv in node.items():
                                    _flatten_row(rv, f"{pfx}.{rk}" if pfx else rk)
                            elif isinstance(node, list):
                                if node:
                                    _flatten_row(node[0], pfx)
                            else:
                                flat[pfx] = node
                        _flatten_row(row)
                        inv = {v.split(":")[0]: k for k, v in rich_map.items()}
                        for path_key, kk in inv.items():
                            val = flat.get(path_key) or flat.get(path_key.split(".")[-1].replace("[]", ""))
                            col[kk].append(val)
                    result = col
                    # Fall through to dict handling below

                if isinstance(result, dict):
                    # Strip _fraud_pattern if LLM included it — we attach pre-assigned ones after
                    result.pop("_fraud_pattern", None)

                    # Clip arrays to exact batch size — LLM sometimes returns more values
                    # than requested, which causes huge responses that get truncated
                    for _k in list(result.keys()):
                        if isinstance(result[_k], list):
                            result[_k] = result[_k][:attempt_batch]

                    rows = _columnar_to_nested_records(result, mapping, attempt_batch, typed_map=typed_map)
                    all_rows.extend(rows)
                    remaining -= attempt_batch
                    break

                # Unexpected parsed shape — retry with a smaller batch
                if attempt_batch == 1:
                    raise Exception(f"Failed to parse LLM response (batch of {attempt_batch}): Unsupported JSON shape\n\nRaw (first 500 chars):\n{raw[:500]}")
                attempt_batch = max(1, attempt_batch // 2)
        return all_rows

    fraud_records = []
    if fraud_count > 0:
        # Pre-assign patterns to each record: guarantees all patterns used, random combos
        fraud_assignments = _assign_fraud_patterns(fraud_count, patterns)
        assignment_str = json.dumps(fraud_assignments, separators=(",", ":"))
        fraud_user = f"""Generate {fraud_count} fraud records.
Field map (key→path|type|description|example):{compact_map}
Per-record pattern assignment (index matches record position):{assignment_str}
Embed the assigned fraud pattern anomalies subtly into the field values for each record.
Output one JSON object: each key→array of {fraud_count} values. Never return null. If type of any field is 'array' generate more than one entries"""
        fraud_records = generate_batch(system, fraud_user, fraud_count)
        # Attach pre-assigned patterns directly — don't rely on LLM output
        for i, rec in enumerate(fraud_records):
            rec["_fraud_pattern"] = fraud_assignments[i] if i < len(fraud_assignments) else []

    legit_records = []
    if legit_count > 0:
        legit_user = f"""Generate {legit_count} legitimate records.
Field map (key→path|type|description|example):{compact_map}
Output one JSON object: each key→array of {legit_count} values. Never return null. All values realistic and internally consistent. If type of any field is 'array' generate more than one entries"""
        legit_records = generate_batch(system, legit_user, legit_count)

    return {"fraud": fraud_records, "legit": legit_records}



# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

defaults = {
    "schema": None,
    "schema_filename": None,
    "raw_spec": None,              # original uploaded JSON — preserves descriptions for compression
    "compressed_schema": None,     # compress_schema(raw_spec) — computed once on upload
    "template_record": None,       # full nested sample payload; None for flat schemas
    "fraud_patterns": [],
    "selected_patterns": [],
    "generated_data": None,
    "patterns_loaded": False,
    "token_usage_log": [],
    "token_usage_totals": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    },
    "fraud_search_query": "",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

@st.dialog("Logs", width="large")
def _open_logs_dialog():
    st.caption("Compressed schema, generation traces, and token usage")

    totals = st.session_state.get("token_usage_totals", {})
    calls = totals.get("calls", 0)
    prompt_t = totals.get("prompt_tokens", 0)
    completion_t = totals.get("completion_tokens", 0)
    total_t = totals.get("total_tokens", 0)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("LLM Calls", f"{calls}")
    t2.metric("Prompt Tokens", f"{prompt_t}")
    t3.metric("Completion Tokens", f"{completion_t}")
    t4.metric("Total Tokens", f"{total_t}")

    if st.session_state.get("generated_data"):
        recs = st.session_state.generated_data
        total_records = len(recs.get("fraud", [])) + len(recs.get("legit", []))
        if total_records > 0:
            st.caption(f"Token efficiency: {total_t / total_records:.2f} tokens per generated record")

    usage_log = st.session_state.get("token_usage_log", [])
    if usage_log:
        with st.expander("Token usage by call", expanded=False):
            st.dataframe(pd.DataFrame(usage_log), width="stretch", height=220)

    if st.session_state.get("compressed_schema"):
        with st.expander("Compressed Schema", expanded=False):
            st.json(st.session_state.compressed_schema)

    if st.session_state.get("_debug_template"):
        with st.expander("Schema Template", expanded=False):
            st.json(st.session_state._debug_template)

    if st.session_state.get("_debug_llm_responses"):
        with st.expander("LLM Responses", expanded=False):
            for i, entry in enumerate(st.session_state._debug_llm_responses):
                st.markdown(f"**Batch {i + 1}** — {entry['batch']} record(s)")
                st.code(entry["raw"], language="json")


h1, h2 = st.columns([9, 1])
with h1:
    st.markdown("## 🛡️ SynthGen")
    st.caption("AI-Powered Synthetic Fraud Data Generator")
with h2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("Logs", key="open_logs_button"):
        _open_logs_dialog()

st.divider()

# ─────────────────────────────────────────────
# STEP 1
# ─────────────────────────────────────────────

st.markdown(
    '<div class="section-title">Step 1 — Upload JSON Schema</div>',
    unsafe_allow_html=True
)

input_tab1, input_tab2 = st.tabs(["Upload File", "Paste JSON"])

def _load_schema_from_raw(raw_schema, source_name):
    """Returns (flat_schema: dict, template: dict | None).

    flat_schema  — {field_path: type_string} used for fraud-pattern detection display.
    template     — the original nested sample payload when the input is a complex
                   nested record; None when the input is already a flat {field: type} map.

    Detection rules
    ---------------
    • JSON Schema spec doc  → convert to payload via example values, then template mode.
    • list of dicts         → infer flat schema from first item; if it has nested
                              lists/dicts, treat first item as the template record too.
    • dict, all-string values → treat as a plain {field: "type"} schema (no template).
    • dict with any list/dict values → treat as a sample payload; extract flat paths
                              and keep the whole dict as the template.
    """
    if isinstance(raw_schema, list):
        if not raw_schema:
            raise Exception("JSON array is empty — cannot infer schema.")
        sample = raw_schema[0]
        if not isinstance(sample, dict):
            raise Exception("Expected an array of objects or a field-type dict.")
        # Detect spec list (e.g. array of spec objects)
        if _is_schema_spec(sample):
            payload = _spec_to_payload({"items": {"properties": sample}})
            payload = payload[0] if isinstance(payload, list) and payload else sample
            return _extract_flat_fields(payload), payload
        # Nested sample list → use first record as template
        if any(isinstance(v, (list, dict)) for v in sample.values()):
            return _extract_flat_fields(sample), sample
        return {k: type(v).__name__ for k, v in sample.items()}, None

    elif isinstance(raw_schema, dict):
        # ── JSON Schema spec document detection ──────────────────────────────
        if _is_schema_spec(raw_schema):
            payload = _spec_root_to_payload(raw_schema)
            return _extract_flat_fields(payload), payload
        # Plain {field: "type"} map — no template needed
        if all(isinstance(v, str) for v in raw_schema.values()):
            return raw_schema, None
        # Nested sample payload → template mode
        if any(isinstance(v, (list, dict)) for v in raw_schema.values()):
            return _extract_flat_fields(raw_schema), raw_schema
        return raw_schema, None

    else:
        raise Exception("Schema must be a JSON object or array of objects.")

with input_tab1:
    uploaded = st.file_uploader(
        "Upload schema",
        type=["json"],
        label_visibility="collapsed"
    )

    if uploaded:
        if uploaded.name != st.session_state.schema_filename:
            with st.spinner("Loading schema..."):
                try:
                    raw_schema = json.load(uploaded)
                    schema, template = _load_schema_from_raw(raw_schema, uploaded.name)
                    st.session_state.schema = schema
                    st.session_state.raw_spec = raw_schema
                    st.session_state.compressed_schema = compress_schema(raw_schema)
                    st.session_state.template_record = template
                    st.session_state.schema_filename = uploaded.name
                    st.session_state.patterns_loaded = False
                    st.session_state.fraud_patterns = []
                    st.session_state.selected_patterns = []
                    st.session_state.generated_data = None
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

        if st.session_state.schema:
            mode_badge = " 🗂 template mode" if st.session_state.template_record else ""
            with st.expander(f"✅ Schema loaded — {uploaded.name}{mode_badge}", expanded=True):
                st.json(st.session_state.raw_spec)

with input_tab2:
    json_text = st.text_area(
        "Paste your JSON schema or sample data here",
        height=180,
        placeholder='{\n  "field_name": "string",\n  "amount": "float"\n}',
        label_visibility="collapsed"
    )
    if st.button("Load JSON", key="load_json_text"):
        if not json_text.strip():
            st.warning("Paste some JSON first.")
        else:
            try:
                raw_schema = json.loads(json_text)
                schema, template = _load_schema_from_raw(raw_schema, "pasted_schema")
                st.session_state.schema = schema
                st.session_state.raw_spec = raw_schema
                st.session_state.compressed_schema = compress_schema(raw_schema)
                st.session_state.template_record = template
                st.session_state.schema_filename = "pasted_schema"
                st.session_state.patterns_loaded = False
                st.session_state.fraud_patterns = []
                st.session_state.selected_patterns = []
                st.session_state.generated_data = None
                mode = " (template mode)" if template else ""
                st.success(f"✅ Schema loaded — {len(schema)} field paths detected{mode}")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    if st.session_state.schema and st.session_state.schema_filename == "pasted_schema":
        with st.expander("✅ Schema loaded — pasted JSON", expanded=True):
            st.json(st.session_state.raw_spec)

st.divider()

# ─────────────────────────────────────────────
# STEP 2
# ─────────────────────────────────────────────

st.markdown(
    '<div class="section-title">Step 2 — Configure</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:
    num_records = st.number_input(
        "Records",
        min_value=2,
        max_value=5000,
        value=100
    )

with c2:
    fraud_pct = st.slider(
        "Fraud %",
        0,
        100,
        30
    )

legit_count = round(num_records * (100 - fraud_pct) / 100)
fraud_count = num_records - legit_count

m1, m2 = st.columns(2)

m1.metric("Legitimate", f"{legit_count}")
m2.metric("Fraud", f"{fraud_count}")

st.divider()

# ─────────────────────────────────────────────
# STEP 3
# ─────────────────────────────────────────────

st.markdown(
    '<div class="section-title">Step 3 — Detect Fraud Patterns</div>',
    unsafe_allow_html=True
)

if st.session_state.schema is None:

    st.info("Upload schema first.")

else:

    if st.button("🔍 Identify Fraud Patterns"):

        with st.spinner("Analysing schema..."):

            try:
                patterns = get_fraud_patterns(
                    st.session_state.schema,
                    raw_spec=st.session_state.get("raw_spec")
                )

                st.session_state.fraud_patterns = patterns
                st.session_state.selected_patterns = [
                    p["id"] for p in patterns
                ]

                st.session_state.patterns_loaded = True

            except Exception as e:
                st.error(str(e))

    if st.session_state.patterns_loaded:
        cols = st.columns(3)
        updated = []
        for i, pattern in enumerate(st.session_state.fraud_patterns):
            with cols[i % 3]:
                with st.container(border=True):
                    checked = st.checkbox(
                        pattern["label"],
                        value=pattern["id"] in st.session_state.selected_patterns,
                        key=f"chk_{pattern['id']}"
                    )
                    st.markdown(
                        f'<div class="fraud-pattern-desc">{pattern.get("description", "")}</div>',
                        unsafe_allow_html=True
                    )
                    fields_used = pattern.get("fields", [])
                    if fields_used:
                        fields_text = ", ".join(fields_used)
                        st.markdown(
                            f'<div class="fraud-pattern-fields"><strong>Fields:</strong> {fields_text}</div>',
                            unsafe_allow_html=True
                        )
                    if checked:
                        updated.append(pattern["id"])
        st.session_state.selected_patterns = updated

st.divider()

# ─────────────────────────────────────────────
# STEP 4
# ─────────────────────────────────────────────

st.markdown(
    '<div class="section-title">Step 4 — Generate Data</div>',
    unsafe_allow_html=True
)

can_generate = (
    st.session_state.schema is not None
    and st.session_state.patterns_loaded
    and len(st.session_state.selected_patterns) > 0
)

if st.button(
    "⚡ Generate Data",
    disabled=not can_generate,
    type="primary"
):

    with st.spinner("Generating records..."):

        try:
            template = st.session_state.get("template_record")
            if template:
                # Template mode: deeply nested payload → generate records that mirror its structure
                records = generate_from_template(
                    template=template,
                    num_records=num_records,
                    fraud_pct=fraud_pct,
                    patterns=st.session_state.selected_patterns,
                    raw_spec=st.session_state.get("raw_spec"),
                )
            else:
                # Flat schema mode: columnar LLM generation
                records = generate_synthetic_data(
                    schema=st.session_state.schema,
                    num_records=num_records,
                    fraud_pct=fraud_pct,
                    patterns=st.session_state.selected_patterns
                )

            st.session_state.generated_data = records
            total = len(records["fraud"]) + len(records["legit"])
            st.success(f"Generated {total} records")

        except Exception as e:
            st.error(str(e))

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if st.session_state.generated_data:

    data = st.session_state.generated_data
    fraud_records = data["fraud"]
    legit_records = data["legit"]
    all_records = fraud_records + legit_records

    st.divider()

    st.markdown(
        '<div class="section-title">Generated Data</div>',
        unsafe_allow_html=True
    )

    def flatten_record(r, include_pattern=False, _prefix=""):
        flat = {}
        pattern = r.get("_fraud_pattern", None) if not _prefix else None
        for k, v in r.items():
            if k == "_fraud_pattern" and not _prefix:
                continue
            full_key = f"{_prefix}.{k}" if _prefix else k
            if isinstance(v, dict):
                # Recursively flatten nested objects
                nested = flatten_record(v, _prefix=full_key)
                flat.update(nested)
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    # Arrays of objects → flatten each element with dot-path columns
                    if len(v) == 1:
                        # Single element: no index suffix (cleaner column names)
                        nested = flatten_record(v[0], _prefix=full_key)
                        flat.update(nested)
                    else:
                        # Multiple elements: append [0], [1], ... suffix
                        for idx, item in enumerate(v):
                            nested = flatten_record(item, _prefix=f"{full_key}[{idx}]")
                            flat.update(nested)
                else:
                    flat[full_key] = ", ".join(str(i) for i in v) if v else ""
            else:
                flat[full_key] = v
        if include_pattern and pattern is not None:
            if isinstance(pattern, list):
                flat["fraud_pattern"] = " | ".join(str(p) for p in pattern)
            else:
                flat["fraud_pattern"] = pattern
        return flat

    def to_df(records, include_pattern=False):
        rows = [flatten_record(dict(r), include_pattern) for r in records]
        df = pd.DataFrame(rows)
        # Ensure no column has mixed list/non-list types (safety net)
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
        return df

    # Build pattern label lookup for display
    pattern_labels = {
        p["id"]: p["label"]
        for p in st.session_state.fraud_patterns
    }

    def get_pattern_chips_html(raw_pattern):
        """Return HTML chips for one or multiple patterns."""
        if isinstance(raw_pattern, list):
            ids = raw_pattern
        elif isinstance(raw_pattern, str) and raw_pattern:
            ids = [raw_pattern]
        else:
            ids = []
        chips = "".join(
            f'<span class="pattern-chip">{pattern_labels.get(pid, pid)}</span>'
            for pid in ids
        )
        return chips or "—"

    def to_df_fraud_display(records):
        rows = []
        for r in records:
            flat = flatten_record(dict(r), include_pattern=False)
            rows.append(flat)
        return pd.DataFrame(rows)

    # Horizontal radio instead of st.tabs — persists across reruns so fraud tab doesn't reset
    view = st.radio(
        "View",
        ["All", "Fraud", "Legitimate"],
        horizontal=True,
        label_visibility="collapsed",
        key="results_view"
    )

    if view == "All":
        st.dataframe(to_df(all_records), width="stretch", height=400)

    elif view == "Fraud":
        s1, s2, s3 = st.columns([5, 1, 1])
        with s1:
            fraud_search_input = st.text_input(
                "Search by fraud pattern",
                value=st.session_state.get("fraud_search_query", ""),
                placeholder="e.g. disposable_email or Velocity Abuse"
            )
        with s2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Search", key="fraud_search_btn"):
                st.session_state.fraud_search_query = fraud_search_input.strip()
                st.session_state.selected_fraud_row = None
        with s3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Clear", key="fraud_search_clear_btn"):
                st.session_state.fraud_search_query = ""
                st.session_state.selected_fraud_row = None
        active_query = st.session_state.get("fraud_search_query", "").strip().lower()
        filtered_pairs = []
        for original_idx, record in enumerate(fraud_records):
            raw_pattern = record.get("_fraud_pattern", [])
            if isinstance(raw_pattern, str):
                raw_pattern = [raw_pattern]
            raw_pattern = [str(p) for p in raw_pattern]
            labels = [str(pattern_labels.get(pid, pid)) for pid in raw_pattern]
            haystack = " ".join([*raw_pattern, *labels]).lower()
            if (not active_query) or (active_query in haystack):
                filtered_pairs.append((original_idx, record))

        st.caption(f"Showing {len(filtered_pairs)} of {len(fraud_records)} fraud records")

        if filtered_pairs:
            filtered_records = [r for _, r in filtered_pairs]
            fraud_df = to_df_fraud_display(filtered_records)
            fraud_event = st.dataframe(
                fraud_df,
                width="stretch",
                height=400,
                on_select="rerun",
                selection_mode="single-row",
                key="fraud_grid"
            )
            # Persist selection in session state so it survives reruns from other widgets
            if fraud_event and fraud_event.selection.rows:
                st.session_state.selected_fraud_row = fraud_event.selection.rows[0]
        else:
            st.info("No records match the selected fraud pattern search.")
            st.session_state.selected_fraud_row = None

        selected_display_idx = st.session_state.get("selected_fraud_row")

        if filtered_pairs and selected_display_idx is not None and selected_display_idx < len(filtered_pairs):
            original_idx, selected_record = filtered_pairs[selected_display_idx]
            raw_pattern = selected_record.get("_fraud_pattern", [])
            if isinstance(raw_pattern, str):
                raw_pattern = [raw_pattern]
            chips_html = "".join(
                f'<span class="pattern-chip">{pattern_labels.get(pid, pid)}</span>'
                for pid in raw_pattern
            )
            st.markdown(
                f"**Row {original_idx + 1} \u2014 Fraud Patterns:** {chips_html}",
                unsafe_allow_html=True
            )
            if st.button("\U0001f50d Explain Fake Data", key="explain_btn"):
                record_clean = {k: v for k, v in selected_record.items() if k != "_fraud_pattern"}

                # Build slim view: only fields mentioned in the fraud pattern definitions
                pattern_objs = {p["id"]: p for p in st.session_state.get("fraud_patterns", [])}
                relevant_fields = []
                for pid in raw_pattern:
                    p_obj = pattern_objs.get(pid)
                    if p_obj:
                        relevant_fields.extend(p_obj.get("fields", []))
                relevant_fields = list(dict.fromkeys(relevant_fields))  # dedupe

                def _get_dot(rec, path):
                    parts = path.replace("[]", "").split(".")
                    node = rec
                    for part in parts:
                        if isinstance(node, list):
                            node = node[0] if node else None
                        if not isinstance(node, dict):
                            return None
                        node = node.get(part)
                    return node

                if relevant_fields:
                    slim = {f: _get_dot(record_clean, f) for f in relevant_fields}
                    slim = {k: v for k, v in slim.items() if v is not None}
                else:
                    slim = {k: v for k, v in record_clean.items() if not isinstance(v, (dict, list))}

                pattern_names = ", ".join(pattern_labels.get(p, p) for p in raw_pattern)
                with st.spinner("Explaining..."):
                    explanation = call_llm(
                        "Fraud analyst. In 2-3 sentences explain which field values are anomalous and why they indicate fraud. Be specific about the actual values.",
                        f"Patterns:{pattern_names}\nFields:{json.dumps(slim, separators=(',', ':'))}",
                        max_tokens=256,
                        usage_label="explain_fraud_record"
                    )
                st.markdown(
                    f'<div class="explanation-box">{explanation}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("Click a row to see its fraud pattern(s).")

    else:
        st.dataframe(to_df(legit_records), width="stretch", height=400)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)

    def strip_meta(r):
        """Recursively strip _fraud_pattern, preserving nested structure for JSON download."""
        if not isinstance(r, dict):
            return r
        return {k: strip_meta(v) for k, v in r.items() if k != "_fraud_pattern"}

    # JSON download: nested structure preserved, _fraud_pattern stripped
    clean_all_json = [strip_meta(r) for r in all_records]
    # CSV download: flat structure
    clean_all_flat = [flatten_record(dict(r)) for r in all_records]

    # ZIP of JSON files: one file per record
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, rec in enumerate(clean_all_json, start=1):
            zf.writestr(f"record_{i:04d}.json", json.dumps(rec, indent=2))
    zip_bytes = zip_buffer.getvalue()

    with dl1:
        st.download_button(
            "⬇ Download JSON ZIP",
            data=zip_bytes,
            file_name="synthgen_output_records.zip",
            mime="application/zip"
        )

    with dl2:
        st.download_button(
            "⬇ Download CSV",
            data=pd.DataFrame(clean_all_flat).to_csv(index=False),
            file_name="synthgen_output.csv",
            mime="text/csv"
        )

    st.divider()
    if st.button("🔄 Reset Everything", type="secondary"):
        for key in ["schema", "schema_filename", "compressed_schema", "template_record", "fraud_patterns", "selected_patterns", "generated_data", "patterns_loaded", "token_usage_log", "token_usage_totals", "fraud_search_query"]:
            st.session_state[key] = defaults[key]
        for key in ["_debug_template", "_debug_llm_responses"]:
            st.session_state.pop(key, None)
        st.rerun()