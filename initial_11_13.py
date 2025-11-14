# %%
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from persona_vectors.PersonaDataset import PersonaDataset
from persona_vectors.ModelWithPersona import ModelWithPersona
import pandas as pd

# trait_list = [
#     "sarcastic",
#     "verbose",
#     "evil",
#     "kind",
#     "analytical",
#     "empathetic",
#     "humorous",
#     "optimistic",
#     "pessimistic",
#     "creative",
#     "brave",
#     "cautious",
# ]

# model = 'openai/gpt-oss-120b'

# failures = ''

# for trait in tqdm(trait_list):
#     try:
#         dataset = PersonaDataset(trait=trait, num_questions=100, client=None, model=model)
#         dataset.generate_dataset()
#     except Exception as e:
#         failures += f"{trait}: {e}\n"

# with open('initial_11_13_failures.txt', 'w') as f:
#     print(failures)
#     f.write(failures)



## BOARDGAME QA

# splits = {'train': 'Boardgame-QA_train.csv', 'validation': 'Boardgame-QA_valid.csv', 'test': 'Boardgame-QA_test.csv'}
# df = pd.read_csv("hf://datasets/1-800-LLMs/Boardgame-QA/" + splits["train"])
# df = df[['example', 'proof']]
# print(df.head())
# df.to_csv("experiments/boardgame_qa_train.csv", index=False)

df = pd.read_csv("experiments/boardgame_qa_train.csv")
df.sample(frac=1).reset_index(drop=True)

def process_single_example(args):
    """Process a single example with the given persona inference model."""
    row, persona_inference, alpha, lock = args
    prompt = row['example'] + "\nAfter considering the problem carefully, end your answer with \"YES\" or \"NO\"."
    
    # Use lock to ensure thread-safe GPU access if needed
    with lock:
        answer = persona_inference.inference_with_persona(prompt=prompt, alpha=alpha, temperature=0.8, max_new_tokens=1000)
    
    ground_truth = row['proof'].find("\"yes\"") != -1
    predicted_answer = None
    if "YES" in answer.upper():
        predicted_answer = True
    elif "NO" in answer.upper():
        predicted_answer = False
    
    return predicted_answer is not None and (predicted_answer == ground_truth)

def benchmark_with_persona(json_filepath:str, max_n:int=15000, df: pd.DataFrame=df, alpha: int=0.5, max_workers: int=4):
    """
    Benchmark with persona using parallel processing.
    
    Args:
        json_filepath: Path to persona initialization JSON
        max_n: Maximum number of examples to process
        df: DataFrame containing examples
        alpha: Alpha value for persona steering
        max_workers: Number of parallel threads (default: 4)
    """
    persona_inference = ModelWithPersona.from_json(json_filepath=json_filepath)
    scores = []
    n = min(max_n, len(df))
    
    # Create a lock for thread-safe GPU access
    gpu_lock = threading.Lock()
    
    # Prepare arguments for parallel processing
    rows_to_process = [(row, persona_inference, alpha, gpu_lock) for idx, row in df.iterrows() if idx < n]
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_example, args) for args in rows_to_process]
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Boardgame-QA file {json_filepath}"):
            try:
                score = future.result()
                scores.append(score)
            except Exception as e:
                print(f"Error processing example: {e}")
                scores.append(False)
    
    accuracy = sum(scores) / len(scores) if scores else 0
    print(f"Accuracy: {accuracy}")
    return scores

n = 40
df = df.loc[:n].copy()

# Adjust max_workers based on your GPU memory and available threads
# Lower values (2-4) for limited GPU memory, higher (8-16) for more memory
MAX_WORKERS = 4

for trait in [
    'analytical', 'creative', 'empathetic', 'kind', 'optimistic', 'pessimistic', 'sarcastic', 'verbose'
]:
    df[trait] = benchmark_with_persona(
        json_filepath=f"/home/pmahajan40/persona-vectors/model_with_persona/{trait}_persona_initialization.json", 
        alpha=0.3, 
        max_n=n, 
        df=df,
        max_workers=MAX_WORKERS
    )
