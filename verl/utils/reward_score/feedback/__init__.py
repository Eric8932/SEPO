from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import mcq_thinking
from verl.utils.reward_score.feedback import tooluse
from verl.utils.reward_score.feedback import synlogic


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "math_dapo", "math_dapo_v2", "math_dapo_thinking", "gsm8k", "openthoughts", "openthoughts_v2", "openthoughts_thinking", "aime24", "aime25", "amc23"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        # Legacy format: enable_thinking=False, answer in <answer>X</answer>
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval_thinking"]:
        # Native Qwen3 thinking mode: enable_thinking=True, answer is last line after </think>
        results = mcq_thinking.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse", "tooluse_thinking"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    elif data_source.startswith("synlogic_"):
        results = synlogic.compute_score(solution_str, ground_truth, extra_info, data_source=data_source)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results
