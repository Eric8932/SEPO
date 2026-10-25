"""Reward scoring for MCQ in native Qwen3 thinking mode (enable_thinking=True).

Qwen3 with enable_thinking=True produces responses in the form:
    <think>
    ...chain of thought...
    </think>
    \boxed{A}

The final answer is expected inside \\boxed{} as a single option letter (A/B/C/D).
Fallback: last non-empty line after </think>, or last standalone A/B/C/D.
"""
import re


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> block and return the remainder."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_answer(text: str) -> str:
    """Extract the final answer letter from text after stripping thinking trace.

    Priority:
    1. \\boxed{X} — extract from the last \\boxed{}
    2. Last non-empty line that is a single letter A/B/C/D
    3. "answer is X" pattern
    4. Last standalone A/B/C/D in text
    """
    after_think = strip_thinking(text)

    # Priority 1: \boxed{X}
    boxed_matches = re.findall(r"\\boxed\{([A-Da-d])\}", after_think)
    if boxed_matches:
        return boxed_matches[-1].upper()

    # Priority 2: last non-empty line that is a single letter
    lines = [l.strip() for l in after_think.splitlines() if l.strip()]
    for line in reversed(lines):
        m = re.fullmatch(r"([A-Da-d])[.):]?", line)
        if m:
            return m.group(1).upper()

    # Priority 3: "answer is X" / "答案是X" pattern
    m = re.search(r"(?:answer[:\s]+|答案[是：\s]+)([A-Da-d])\b", after_think, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Priority 4: last standalone uppercase letter in the text
    matches = re.findall(r"\b([A-D])\b", after_think)
    if matches:
        return matches[-1].upper()

    return ""


def compute_score(solution: str, ground_truth: str) -> dict:
    pred = extract_answer(solution)
    reward = float(pred == ground_truth.strip().upper())
    return {
        "score": reward,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 0 if pred in ("A", "B", "C", "D") else 1,
        "feedback": "",
    }
