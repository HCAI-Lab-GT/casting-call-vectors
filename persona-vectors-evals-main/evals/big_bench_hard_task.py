"""
Big Bench Hard Evaluation

This evaluation module is the basic framework implementation of the Big Bench Hard Task
from HuggingFace to be used with inspect-ai. We create a function to convert the dataset to the appropriate set of samples used for evaluation with input and target and set up the task to be ran through the 
inspect ai framework.

Key Features:
    - Sample Creation: We take each record from the dataset and convert it to a sample to be used by the framework
    - Utilize a system message to explain to the model how to answer the question and what to explain in the response
    - We include a chain of thought and generate as a part of our solver to allow for us to examine the logs
    - We score with the 'match' scorer which checks if the actual answer is within the model's answer.

Dependencies:
    - ollama: For local model inference
    - openai: For API-based model inference
    - inspect-ai: Evaluation Framework for loading dataset and performing model evaluation
    
Environment Variables:
    - LITELLM_API_KEY: Required when using OpenAI-compatible endpoints
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import system_message, chain_of_thought, generate


# 1. DEFINE THE DATA MAPPING
def record_to_sample(record):
    """
    Converts an individual record entry in the dataset to a sample
        
    This method takes an individual record and pulls the input and target
    to create a sample which can be used by the evaluation framework.
        
    Args:
        record Dict[str, str]: A dictionary representing a single row from a HuggingFace dataset. 
        
    Returns:
        sample: inspect-_ai.dataset.Sample: An inspect_ai Sample type used for the ealuation framework to evaluate easily.
        
        
    Example:
        >>> record = {'input': 'Question A', 'target': '(A)'}
        >>> sample = Sample(input='Question A', target='(A)')
    """
    return Sample(
        input=record["input"],
        target=record["target"],
        metadata={
            "subset": "logical_deduction_five_objects" # Tracks which task this is
        }
    )


@task
def bbh_loader():
    # 2. SELECT YOUR SUBSET
    # Common BBH options: 
    # - 'date_understanding'
    # - 'logical_deduction_five_objects'
    # - 'movie_recommendation'
    # - 'ruin_names'
    """
    Creates a task for each record in the dataset to perform inference and evaluation.
    
    Takes the dataset to run the evluation and specified the solver algorithms to be used, 
    such as chain of thought and generate to generate a response which can then be evaluated
    by the scorer function.
        

        
    Returns:
        Task: inspect_ai.Task: An inspect_ai Task type which represents the dataset being used along with the solver and scorer functions.
        
        
    Example:
        >>> dataset: maveriq/bigbenchhard
        >>> task = Task(dataset, solver=[system_message, chain_of_thought, generate], scorer=match)
    """
    
    
    TASK_SUBSET = "logical_deduction_five_objects"

    return Task(
        dataset=hf_dataset(
            path="maveriq/bigbenchhard",
            name=TASK_SUBSET,  # <--- This loads the specific sub-task
            split="train",     # BBH often only has a 'train' split on HF
            sample_fields=record_to_sample,
            trust=True
        ),
        solver=[
            # BBH is hard, so we explicitly ask for step-by-step reasoning
            system_message("You are a helpful reasoning assistant. Think step-by-step. Your answer must be the letter that corresponds to the option in parantheses: (A), (B), (C), (D), (E)."),
            chain_of_thought(),
            generate()
        ],
        # BBH answers are usually short phrases or words, so match() works well
        scorer=match(ignore_case=True)
    )