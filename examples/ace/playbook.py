import json
import re
from typing import Any


def make_empty_playbook() -> str:
    return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    for candidate in fenced:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_playbook_line(line: str) -> dict[str, Any] | None:
    pattern = r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
    match = re.match(pattern, line.strip())
    if not match:
        return None
    return {
        "id": match.group(1),
        "helpful": int(match.group(2)),
        "harmful": int(match.group(3)),
        "content": match.group(4),
        "raw_line": line,
    }


def format_playbook_line(
    bullet_id: str, helpful: int, harmful: int, content: str
) -> str:
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def get_section_slug(section_name: str) -> str:
    slug_map = {
        "strategies_and_insights": "str",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "mis",
        "problem_solving_heuristics": "heur",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta",
    }
    clean_name = (
        section_name.lower().strip().replace(" ", "_").replace("&", "and")
    )
    if clean_name in slug_map:
        return slug_map[clean_name]
    words = [w for w in clean_name.split("_") if w]
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[0] for w in words[:5])[:5]


def get_next_global_id(playbook_text: str) -> int:
    max_id = 0
    for line in playbook_text.strip().splitlines():
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        match = re.search(r"-(\d+)$", parsed["id"])
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def extract_playbook_bullets(playbook_text: str, bullet_ids: list[str]) -> str:
    if not bullet_ids:
        return "(No bullets used by generator)"
    found: list[str] = []
    for line in playbook_text.strip().splitlines():
        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in bullet_ids:
            found.append(
                format_playbook_line(
                    parsed["id"],
                    parsed["helpful"],
                    parsed["harmful"],
                    parsed["content"],
                )
            )
    if not found:
        return "(Generator referenced bullet IDs but none were found in playbook)"
    return "\n".join(found)


def update_bullet_counts(
    playbook_text: str, bullet_tags: list[dict[str, str]]
) -> str:
    tag_map: dict[str, str] = {}
    for tag in bullet_tags:
        bullet_id = tag.get("id") or tag.get("bullet", "")
        tag_value = tag.get("tag", "neutral")
        if bullet_id:
            tag_map[bullet_id] = tag_value
    if not tag_map:
        return playbook_text

    updated_lines: list[str] = []
    for line in playbook_text.strip().splitlines():
        if line.strip().startswith("#") or not line.strip():
            updated_lines.append(line)
            continue
        parsed = parse_playbook_line(line)
        if not parsed or parsed["id"] not in tag_map:
            updated_lines.append(line)
            continue
        tag = tag_map[parsed["id"]]
        helpful = parsed["helpful"] + (1 if tag == "helpful" else 0)
        harmful = parsed["harmful"] + (1 if tag == "harmful" else 0)
        updated_lines.append(
            format_playbook_line(parsed["id"], helpful, harmful, parsed["content"])
        )
    return "\n".join(updated_lines)


def apply_curator_operations(
    playbook_text: str, operations: list[dict[str, Any]], next_id: int
) -> tuple[str, int]:
    lines = playbook_text.strip().splitlines()
    bullets_to_add: list[tuple[str, str]] = []
    section_names = {
        line.strip()[2:].strip().lower().replace(" ", "_").replace("&", "and")
        for line in lines
        if line.strip().startswith("##")
    }

    for op in operations:
        if op.get("type") != "ADD":
            continue
        section = (
            op.get("section", "others").lower().replace(" ", "_").replace("&", "and")
        )
        if section not in section_names:
            section = "others"
        bullet_id = f"{get_section_slug(section)}-{next_id:05d}"
        next_id += 1
        bullets_to_add.append(
            (section, format_playbook_line(bullet_id, 0, 0, op.get("content", "")))
        )

    final_lines: list[str] = []
    current_section: str | None = None
    remaining = list(bullets_to_add)
    for line in lines:
        if line.strip().startswith("##"):
            if current_section is not None:
                section_adds = [b for s, b in remaining if s == current_section]
                final_lines.extend(section_adds)
                remaining = [(s, b) for s, b in remaining if s != current_section]
            current_section = (
                line.strip()[2:].strip().lower().replace(" ", "_").replace("&", "and")
            )
        final_lines.append(line)

    if current_section is not None:
        section_adds = [b for s, b in remaining if s == current_section]
        final_lines.extend(section_adds)
        remaining = [(s, b) for s, b in remaining if s != current_section]

    if remaining:
        others_idx = next(
            (i for i, line in enumerate(final_lines) if line.strip() == "## OTHERS"),
            -1,
        )
        remainder = [b for _, b in remaining]
        if others_idx >= 0:
            final_lines[others_idx + 1 : others_idx + 1] = remainder
        else:
            final_lines.extend(remainder)

    return "\n".join(final_lines), next_id


def get_playbook_stats(playbook_text: str) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total_bullets": 0,
        "high_performing": 0,
        "problematic": 0,
        "unused": 0,
        "by_section": {},
    }
    current_section = "general"
    for line in playbook_text.strip().splitlines():
        if line.strip().startswith("##"):
            current_section = line.strip()[2:].strip()
            continue
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        stats["total_bullets"] += 1
        if parsed["helpful"] > 5 and parsed["harmful"] < 2:
            stats["high_performing"] += 1
        elif parsed["harmful"] >= parsed["helpful"] and parsed["harmful"] > 0:
            stats["problematic"] += 1
        elif parsed["helpful"] + parsed["harmful"] == 0:
            stats["unused"] += 1
        by_section = stats["by_section"].setdefault(
            current_section, {"count": 0, "helpful": 0, "harmful": 0}
        )
        by_section["count"] += 1
        by_section["helpful"] += parsed["helpful"]
        by_section["harmful"] += parsed["harmful"]
    return stats


def approximate_token_count(text: str) -> int:
    return max(1, len(text) // 4)
