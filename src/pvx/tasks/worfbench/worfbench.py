"""
Custom Inspect AI Task + Solver for WorfBench

Paper: https://arxiv.org/pdf/2410.07869
GitHub: https://github.com/zjunlp/WorfBench
Huggingface:
    Train: https://huggingface.co/datasets/zjunlp/WorFBench_train
    Test: https://huggingface.co/datasets/zjunlp/WorFBench_test
"""

import argparse
import re
from functools import partial

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, chain_of_thought, generate
from sentence_transformers import SentenceTransformer

from pvx.tasks.worfbench.eval_prompt import one_shot_example, two_shot_example
from pvx.tasks.worfbench.graph_evaluator import t_eval_graph, t_eval_nodes
from pvx.utils.inspect_utils import dicts_to_chatmessages, print_sample_input
from pvx.utils.logging_utils import setup_logging

sentence_model = SentenceTransformer("all-mpnet-base-v2")

logger = setup_logging(name="worfbench")

HELD_OUT_TASKS = ["intercodesql", "seal_tools"]

HF_PATHS = {"test": "zjunlp/WorFBench_test", "train": "zjunlp/WorFBench_train"}

SHOTS_INTRO = {
    0: "",  # wont be accessed, spliced to empty list
    1: "Here is an example.\n",
    2: "Here are two examples. You need to strictly follow the format provided in the examples.\n",
}

shot_maps = {
    0: {
        task: []
        for task in ["alfworld", "webshop", "os", "toolbench", "toolalpaca", "lumos", "wikihow"]
    },
    1: one_shot_example,
    2: two_shot_example,
}

ANSWER_PATTERN_WORD = re.compile(r"(?i)ANSWER\s*:\s*(\w+)(?:\n|$)")


@task
def worfbench_task(
    split: str = "test",
    limit: int = None,
    nshot: int = 2,
    eval_type: str = "node",
    prompt_type: str = "",
) -> Task:
    """
    Custom Task for Worfbench.

    Args:
        split (str): Choice of 'test' or 'train' splits
        limit (int): Number of samples
        nshot (int): Number of shot reasoning: [0, 1, 2]
        prompt_type (str): 'chain_of_thought'

    Returns:
        inspect.Task:
    """
    logger.info("Constructing WorfBench task")

    dataset = build_dataset(split, limit, nshot)
    logger.info("Loaded WorfBench dataset")

    return Task(
        dataset=dataset,
        solver=[
            *([chain_of_thought()] if prompt_type == "chain_of_thought" else []),
            generate(),
        ],
        scorer=worfbench_scorer(prompt_type=prompt_type, eval_type=eval_type),
    )


@scorer(
    metrics={
        "precision": [mean(), stderr()],
        "recall": [mean(), stderr()],
        "f1_score": [mean(), stderr()],
        # or: "*": [mean(), stderr()]  # applies to all keys :contentReference[oaicite:1]{index=1}
    }
)
def worfbench_scorer(prompt_type: str, eval_type: str = "node") -> Scorer:
    """
    Custom Scorer for Worfbench.

    Args:
        eval_type (str): 'node' or 'graph'

    Returns:
        inspect_ai.Scorer
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion

        # if prompt_type == 'chain_of_thought':
        #     match = re.search(ANSWER_PATTERN_WORD, state.output.completion)

        #     answer = match.groups()[0] if match else ''

        ### logic from WorfBench/evaluator/node_eval.py:eval_workflow() ###
        gold_plan = target.text
        pred_plan = answer

        # build workflow
        gold_graph_workflow = workflow_to_graph_list(gold_plan)
        pred_graph_workflow = workflow_to_graph_list(pred_plan)

        # select eval method to use: 'node' or 'graph'
        t_eval_func = None
        if eval_type == "node":
            t_eval_func = t_eval_nodes
        elif eval_type == "graph":
            t_eval_func = t_eval_graph
        else:
            raise ValueError("`eval_type` must be 'node' or 'graph'")

        # evaluate workflow
        # result: {'precision': ..., 'recall': ..., 'f1_score': ...}
        result = t_eval_func(pred_graph_workflow, gold_graph_workflow, sentence_model)

        # return score
        return Score(
            value=result,
            answer=answer,
        )

    return score


def build_dataset(split: str = "test", limit: int = None, nshot: int = 2) -> Dataset:
    """
    Builds Inspect Dataset object with parameters

    Args:
        split (str): Choice of 'test' or 'train' splits
        limit (int): Number of samples
        nshot (int): Number of shot reasoning: [0, 1, 2]

    Returns:
        inspect.dataset.Dataset:
    """
    if split not in HF_PATHS:
        raise ValueError("Split must be either 'test' or 'train'.")

    if not 0 <= nshot <= 2:
        raise ValueError("nshot must be 0, 1, or 2 for zero-shot, one-shot, two-shot respectively")

    nshot_record_to_sample = partial(record_to_sample, nshot=nshot, split=split)

    # Load from Hugging Face: zjunlp/FBench_test
    worf_test_all: Dataset = hf_dataset(
        path=HF_PATHS[split],
        split=split
        + (f"[:{limit}]" if limit else ""),  # Allow limiting samples for quick smoke tests
        sample_fields=nshot_record_to_sample,
        trust=True,
    )  # Required for some HF loading scripts

    worf_test_filtered: Dataset = worf_test_all.filter(
        lambda s: s.metadata["source"] not in HELD_OUT_TASKS, name="worfbench_filtered"
    )

    return worf_test_filtered


def record_to_sample(record, nshot: int = 2, split="test") -> Sample:
    """
    Convert test split records into InspectAI format.
    Main change is optionally adding one-shot/two-shot examples in same method as paper.
    Format is different between train and test split.

    Link: https://huggingface.co/datasets/zjunlp/WorFBench_test
    Link: https://huggingface.co/datasets/zjunlp/WorFBench_train

    Args:
        record ()
        split (str): 'train' or 'test'
        nshot (int): Number of shot reasoning: [0, 1, 2]

    Returns:
        inspect.dataset.Sample:
    """
    source = record["source"].split("/")[-1]  # train has environment/{source} for some
    conversation = record.get(
        "conversations", record.get("messages")
    )  # conversations in test, messages in train
    conversation_id = record.get("id")  # somehow train doens't have id

    # extract parts of conversation
    system_message = conversation[0]
    shots = conversation[1:-2]  # empty list for test split as len(conversation) == 3
    base_question = conversation[-2]
    answer_response = conversation[-1]

    # HELD_OUT_TASKS will be filtered out in post-Dataset filter
    if source in HELD_OUT_TASKS:
        return Sample(input="", metadata={"source": source})

    if split == "test":
        # insert shots
        shots = shot_maps[nshot][source]

        # add turn instruction if shots included
        if nshot != 0:
            base_question["content"] = "Now it's your turn.\n" + base_question["content"]

    elif split == "train":
        # strip "Here are an/two examples ..." intro text from first example and add proper intro based on number of shots
        shots[0]["content"] = SHOTS_INTRO[nshot] + shots[0]["content"].split("\n", 2)[1]

        # strip extra shots
        shots = shots[: 2 * nshot]

        # remove "Now it's your turn." in base question if zero-shot
        if nshot == 0:
            base_question["content"] = base_question["content"].split("\n")[1]

    # combine back into list of dicts
    prompt_msgs = [system_message, *shots, base_question]

    return Sample(
        # The model sees this big block of text
        input=dicts_to_chatmessages(prompt_msgs),  # omit system and omit gold
        # The model is graded against this answer
        target=answer_response["content"],
        # We store the ID to track specific cases (like the Squirrel ones later)
        metadata={"source": source},
        id=conversation_id,  # None for traiining
    )


"""
Pulled from WorFBench GitHub

WorfBench/evaluator/node_eval.py:workflow_to_graph_list
"""


def workflow_to_graph_list(workflow: str) -> list[str]:
    try:
        if "Node" not in workflow:
            print("workflow is not in the right format")
            return []

        node_pattern = re.compile(r"\d+[:.] (.+)")
        node_matches = node_pattern.findall(workflow)

        node_workflow = [match.strip() for match in node_matches]
        if len(node_workflow) != 0 and (
            "Finish" in node_workflow[-1] or "finish" in node_workflow[-1]
        ):
            # node_workflow = node_workflow[:-1]
            pass
        elif len(node_workflow) == 0:
            print("node_workflow is empty")

        node_workflow.insert(0, "START")
        node_workflow.append("END")

        edge_pattern = re.compile(r"\(\s*(\d+|START)\s*,\s*(\d+|END)\s*\)")

        edge_matches = edge_pattern.findall(workflow)

        if len(edge_matches) == 0:
            print("edge_workflow is empty")
            return []

        edge_workflow = []
        for _i, match in enumerate(edge_matches):
            edge = list(match)
            if "START" in edge:
                edge[edge.index("START")] = "0"
            if "END" in edge:
                edge_num = len(node_workflow) - 1
                edge[edge.index("END")] = str(edge_num)
            edge = tuple(map(int, edge))  # Convert back to tuple after modification
            edge_workflow.append(edge)

        # print(f"edge_workflow: {edge_workflow}")
        if len(edge_workflow) == 0:
            print("edge_workflow is empty")
            return []

        return {"nodes": node_workflow, "edges": edge_workflow}
    except Exception as e:
        print(e)
        return []


# 3. VERIFICATION BLOCK (Run this file directly to test)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-S",
        "--split",
        type=str,
        choices=["test", "train"],
        default="test",
        help="'test' or 'train'",
    )
    ap.add_argument("-L", "--limit", type=int, default=2, help="Number of samples")
    ap.add_argument(
        "-N", "--nshot", type=int, default=2, help="Number of shot reasoning: [0, 1, 2]"
    )
    args = ap.parse_args()

    print("🔄 Fetching WorfBench from Hugging Face...")

    # Manually load the dataset to verify it looks correct
    ds = build_dataset(args.split, args.limit, args.nshot)

    # Grab the first sample
    sample = ds[0]

    print("\n✅ SUCCESS! Here is what the data looks like:\n")
    print(f"--- [NUMBER SAMPLES (How many samples does the dataset have)] ---\n{len(ds)}\n")
    print(f"--- [INPUT (What the model sees)] ---\n{print_sample_input(sample.input)}\n")
    print(f"--- [TARGET (The correct answer)] ---\n{sample.target}")
    print("\n------------------------------------------------")
    print("You are ready to connect a model.")
