"""
Reward function for SynLogic logical reasoning tasks.

Uses the SynLogic verifier library to check model outputs.
Requires the SynLogic repo to be cloned at a known path (SYNLOGIC_ROOT).

Each sample's `extra_info.game_data_str` contains the full problem spec
(question, answer, difficulty, metadata). The verifier checks the model's
response against the ground truth using task-specific rules.

Reward: 1.0 if format correct AND answer correct, else 0.0
"""

import os
import re
import sys
import json
from typing import Optional

# ---------------------------------------------------------------------------
# SynLogic verifier setup (lazy import)
# ---------------------------------------------------------------------------
_SYNLOGIC_ROOT = os.environ.get(
    "SYNLOGIC_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../third_party/SynLogic")),
)
_verifier_classes = None


def _ensure_verifiers():
    """Lazy-load verifier registry from SynLogic repo."""
    global _verifier_classes
    if _verifier_classes is not None:
        return
    if _SYNLOGIC_ROOT not in sys.path:
        sys.path.insert(0, _SYNLOGIC_ROOT)
    try:
        from task2verifier import verifier_classes
        _verifier_classes = verifier_classes
    except ImportError as e:
        raise ImportError(
            f"Cannot import SynLogic verifiers. "
            f"Please clone https://github.com/MiniMax-AI/SynLogic to {_SYNLOGIC_ROOT}\n"
            f"Original error: {e}"
        )


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

THOUGHT_DELIMITER_END = "</think>"
REASONING_DELIMITER_END = "</reasoning>"


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> or <reasoning>...</reasoning> block from response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_answer_tag(text: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_solution(solution_str: str) -> str:
    """Extract the answer from model output, stripping thinking/reasoning and answer tags."""
    text = solution_str
    if THOUGHT_DELIMITER_END in text:
        text = text.split(THOUGHT_DELIMITER_END, 1)[1]
    elif REASONING_DELIMITER_END in text:
        text = text.split(REASONING_DELIMITER_END, 1)[1]
    answer = _extract_answer_tag(text)
    return answer if answer is not None else text


def _check_format(response: str) -> bool:
    """Check if response follows <think>...</think><answer>...</answer> format."""
    return (
        "<think>" in response
        and "</think>" in response
        and "<answer>" in response
        and "</answer>" in response
        and response.count("<think>") == 1
        and response.count("</think>") == 1
        and response.count("<answer>") == 1
        and response.count("</answer>") == 1
    )


def _bbeh_fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match for BBEH tasks (ported from SynLogic's bbeh_verifier)."""
    prediction = prediction.strip().lower()
    reference = str(reference).strip().lower()

    if prediction == reference:
        return True

    # Option format: (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        if len(reference) == 1:
            return prediction[1] == reference
        elif len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
            return prediction[1] == reference[1]
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        if len(prediction) == 1:
            return reference[1] == prediction

    # List format: [a, b, c]
    if reference.startswith("[") and reference.endswith("]"):
        ref_clean = reference[1:-1].replace(" ", "").replace("'", "").replace('"', "")
        if prediction.startswith("[") and prediction.endswith("]"):
            pred_clean = prediction[1:-1].replace(" ", "").replace("'", "").replace('"', "")
        else:
            pred_clean = prediction.replace(" ", "").replace("'", "").replace('"', "")
        return ref_clean == pred_clean

    # Numeric match
    try:
        if float(prediction) == float(reference):
            return True
    except (ValueError, TypeError):
        pass

    # Whitespace normalization
    if prediction.replace(" ", "") == reference.replace(" ", ""):
        return True

    # Quote normalization
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Trailing question mark
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    data_source: str = "",
) -> float:
    """Compute reward for a SynLogic task.

    Args:
        solution_str: Model's full response text.
        ground_truth: Not used directly (answer is in game_data_str).
                      Kept for interface compatibility.
        extra_info: Must contain 'game_data_str' (JSON string with question/answer/metadata).
        data_source: The full data_source string (e.g. "synlogic_sudoku").

    Returns:
        dict with 'score' (float, 0.0 or 1.0)
    """
    if extra_info is None:
        extra_info = {}

    game_data_str = extra_info.get("game_data_str", "{}")

    # Strip "synlogic_" prefix to get the verifier key
    task_name = data_source.replace("synlogic_", "") if data_source else ""

    _ensure_verifiers()

    try:
        from base.data import Data
        game_data = Data.from_json_str(game_data_str)
    except Exception:
        # If Data class fails, return 0
        return {"score": 0.0}

    # Check format (relaxed: we don't require strict format for non-thinking mode)
    # Format check: at minimum need <answer> tags to extract answer
    format_ok = _check_format(solution_str)

    if not format_ok:
        # Even without proper format, try to extract and verify
        pass

    # Extract answer from model output
    extracted = _extract_solution(solution_str)

    # --- BBEH tasks: use direct fuzzy match (BBEHVerifier's extract_answer
    #     has issues when receiving pre-extracted answer text) ---
    if task_name.startswith("bbeh_"):
        correct = _bbeh_fuzzy_match(extracted, game_data.answer)
        return {"score": 1.0 if correct else 0.0}

    # --- SynLogic tasks: use task-specific verifier ---
    if task_name and task_name in _verifier_classes:
        verifier_class = _verifier_classes[task_name]
        verifier = verifier_class()
        try:
            import io, contextlib
            # Try original extracted text first
            with contextlib.redirect_stdout(io.StringIO()):
                result = verifier.verify(game_data, extracted)
            correct = bool(result)

            # If failed and extracted doesn't contain \boxed{}, retry with \boxed{} wrapper
            # Many SynLogic verifiers require \boxed{} format for extract_answer
            if not correct and "\\boxed{" not in extracted:
                boxed = f"\\boxed{{{extracted}}}"
                with contextlib.redirect_stdout(io.StringIO()):
                    result2 = verifier.verify(game_data, boxed)
                correct = bool(result2)
        except Exception:
            correct = False
    else:
        # Fallback: exact match against ground_truth
        correct = False
        try:
            gt = ground_truth.strip()
            pred = extracted.strip()
            correct = pred == gt
        except Exception:
            pass

    score = 1.0 if correct else 0.0
    return {"score": score}
