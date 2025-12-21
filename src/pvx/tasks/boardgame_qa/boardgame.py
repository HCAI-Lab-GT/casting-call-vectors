import os

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import chain_of_thought, generate, system_message
from inspect_ai.scorer import pattern

from pvx.utils.inspect_utils import print_sample_input

HF_PATH = "tasksource/Boardgame-QA"
@task
def boardgame_loader(limit: int, prompt_type: str='') -> Task:
    '''
    
    Args:
        limit (int)
        prompt_type (str): 'chain_of_thought'
    
    Returns:
        inspect_ai.Task
    '''
    split = "test" if not limit else f"test[:{limit}]"
    
    return Task(
        # Load from Hugging Face: tasksource/Boardgame-QA
        dataset=hf_dataset(
            path=HF_PATH,
            split=split,  # Allow limiting samples for quick smoke tests
            sample_fields=record_to_sample,
            trust=True,  # Required for some HF loading scripts
        ),
        solver=[
            system_message(
                "You are a helpful logic assistant. "
                "Solve the logic puzzle step-by-step. "
                "The last line poses the statement in question and you have to idenitfy if the statement is provably true, provably false, or neither. "
                "You must provide a final answer and the answer must be one of the following words: 'proved', 'disproved', or 'unknown', for provably true, provable false, or neither provably true or false respectively."
            ),
            *([chain_of_thought()] if prompt_type == 'chain_of_thought' else []),
            generate(),
        ],
        # scorer=model_graded_fact(),
        scorer=pattern(
            pattern="\\b(proved|disproved|unknown)\\b",
        )
    )

def record_to_sample(record):
    return Sample(
        # The model sees this big block of text
        input=record["example"],
        # The model is graded against this answer
        target=record["label"],
        # We store the ID to track specific cases (like the Squirrel ones later)
        metadata={},
        id=record.get("idx", "unknown")
    )

# 3. VERIFICATION BLOCK (Run this file directly to test)
if __name__ == "__main__":
    print("🔄 Fetching BoardgameQA from Hugging Face...")

    # Manually load the dataset to verify it looks correct
    ds = hf_dataset(
        path=HF_PATH, split="test", sample_fields=record_to_sample, trust=True
    )

    # Grab the first sample
    sample = ds[0]

    print("\n✅ SUCCESS! Here is what the data looks like:\n")
    print(f"--- [NUMBER SAMPLES (How many samples does the dataset have)] ---\n{len(ds)}\n")
    print(f"--- [INPUT (What the model sees)] ---\n{print_sample_input(sample.input)}\n")
    print(f"--- [TARGET (The correct answer)] ---\n{sample.target}")
    print("\n------------------------------------------------")
    print("You are ready to connect a model.")
