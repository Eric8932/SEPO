import re
import json
from collections import Counter


def extract_actions(text: str) -> list[str]:
    """Extract all action names after 'Action:' occurrences."""
    actions = re.findall(r'Action:\s*(\w+)', text)
    return actions


def extract_action_inputs_legacy(text: str) -> dict:
    """(Legacy) Extract and merge all JSON blocks following 'Action Input:'.

    Uses non-greedy regex {.*?} which fails on nested JSON objects
    (e.g., {"headers": {"Content-Type": "..."}}) because it stops at the first '}'.
    Kept for reference. Use extract_action_inputs() instead.
    """
    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)

    combined_dict = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError:
            pass

    return combined_dict


def _extract_braced_block(text: str, start: int) -> str | None:
    """Extract a balanced {...} block starting at position `start`.

    Uses brace-depth counting to handle arbitrarily nested JSON.
    Respects JSON string literals (skips braces inside quotes).
    Returns the matched substring including outer braces, or None on failure.
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_action_inputs(text: str) -> dict:
    """Extract and merge all JSON blocks following 'Action Input:'.

    Uses brace-depth counting to correctly handle nested JSON objects
    (e.g., {"headers": {"Content-Type": "application/json"}, "data": {...}}).
    """
    combined_dict = {}
    for m in re.finditer(r'Action Input:\s*', text):
        block = _extract_braced_block(text, m.end())
        if block is None:
            continue
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError:
            pass

    return combined_dict


def merge_action_inputs(action_inputs_list: list[dict]) -> dict:
    """Merge a list of action input dicts into a single dict."""
    combined = {}
    for d in action_inputs_list:
        if d:
            combined.update(d)
    return combined


def is_correct_format(text: str) -> bool:
    """Check if the text contains the expected Action/Action Input format."""
    pattern = re.compile(r"Action:.*?\nAction Input:.*?", re.DOTALL)
    return pattern.search(text) is not None


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> block if present. Safe for non-thinking mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def compute_score(solution: str, ground_truth: str) -> dict:
    """
    Compute score for tooluse task.

    Args:
        solution: The model's response text
        ground_truth: JSON string containing list of dicts with 'Action' and 'Action_Input' keys
                      e.g., '[{"Action": "search", "Action_Input": "{\"query\": \"test\"}"}]'

    Returns:
        dict with score, acc, pred, incorrect_format, feedback
    """
    # Strip <think>...</think> to avoid extracting actions from reasoning trace
    solution_clean = _strip_thinking(solution)
    # Parse ground truth
    try:
        gt_list = json.loads(ground_truth)
    except json.JSONDecodeError:
        # If ground_truth is already a list (passed directly), handle that case
        if isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            return {
                "score": 0.0,
                "acc": 0.0,
                "pred": "",
                "incorrect_format": 1,
                "feedback": "Failed to parse ground truth JSON",
            }
    
    # Extract ground truth actions and action inputs
    gt_actions = [item['Action'] for item in gt_list]
    gt_action_inputs_list = []
    for item in gt_list:
        try:
            parsed_input = json.loads(item['Action_Input']) if isinstance(item['Action_Input'], str) else item['Action_Input']
            gt_action_inputs_list.append(parsed_input)
        except (json.JSONDecodeError, KeyError):
            gt_action_inputs_list.append({})
    gt_action_inputs = merge_action_inputs(gt_action_inputs_list)
    
    # Extract predicted actions and action inputs from solution (after stripping <think>)
    pred_actions = extract_actions(solution_clean)
    pred_action_inputs = extract_action_inputs(solution_clean)

    # Check correctness
    actions_correct = Counter(pred_actions) == Counter(gt_actions)
    action_inputs_correct = pred_action_inputs == gt_action_inputs

    # Both must be correct for full score
    is_correct = actions_correct and action_inputs_correct
    reward = 1.0 if is_correct else 0.0

    # Check format (on cleaned solution)
    correct_format = is_correct_format(solution_clean)

    # Build prediction string for logging
    prediction = f"Actions: {pred_actions}, Inputs: {pred_action_inputs}"
    
    # Build feedback
    feedback_parts = []
    if not actions_correct:
        feedback_parts.append(f"Actions mismatch: predicted {pred_actions}, expected {gt_actions}")
    if not action_inputs_correct:
        feedback_parts.append(f"Action inputs mismatch: predicted {pred_action_inputs}, expected {gt_action_inputs}")

    if len(feedback_parts) == 0:
        feedback = "" # no feedback means correct
    else:
        feedback = "; ".join(feedback_parts)
    
    return {
        "score": reward,
        "acc": reward,
        "pred": prediction,
        "incorrect_format": 0 if correct_format else 1,
        "feedback": feedback,
    }
