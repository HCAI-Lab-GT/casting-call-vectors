# %%
from tqdm import tqdm

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

def process_single_example(row, persona_inference, alpha):
    """Process a single example with the given persona inference model."""
    prompt = row['example'] + "\nAfter considering the problem carefully, end your answer with \"YES\" or \"NO\"."

    try:
        answer = persona_inference.inference_with_persona(prompt=prompt, alpha=alpha, temperature=0.8, max_new_tokens=1000)
        
        ground_truth = row['proof'].find("\"yes\"") != -1
        predicted_answer = None
        if "YES" in answer.upper():
            predicted_answer = True
        elif "NO" in answer.upper():
            predicted_answer = False
        
        return predicted_answer is not None and (predicted_answer == ground_truth)
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def benchmark_with_persona(persona = ModelWithPersona, max_n:int=15000, df: pd.DataFrame=df, alpha: float=0.5):
    """
    Benchmark with persona using sequential processing.
    
    Args:
        json_filepath: Path to persona initialization JSON
        max_n: Maximum number of examples to process
        df: DataFrame containing examples
        alpha: Alpha value for persona steering
    """
    persona_inference = persona
    scores = []
    n = min(max_n, len(df))
    
    # Process examples sequentially
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= n:
            break
        try:
            score = process_single_example(row, persona_inference, alpha)
            scores.append(score)
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            scores.append(False)

    return scores

n = 100
df = df.loc[:n-1].copy()

traits = [
    'analytical', 'creative', 'empathetic', 'kind', 'optimistic', 'pessimistic', 'sarcastic', 'verbose'
]
alphas = [0.1, 0.3, 0.5, 1.0, 2.0]

total_iterations = len(traits) * len(alphas)
with tqdm(total=total_iterations, desc="Processing traits and alphas") as pbar:
    for trait in traits:
        persona = ModelWithPersona.from_json(json_filepath=f"/home/pmahajan40/persona-vectors/model_with_persona/{trait}_persona_initialization.json")
        for alpha in alphas:
            alpha_str = str(alpha).replace('.', '_')
            df[f'{trait}_{alpha_str}'] = benchmark_with_persona(
                persona=persona, 
                alpha=alpha, 
                max_n=n, 
                df=df
            )
            pbar.update(1)
            pbar.set_postfix({'trait': trait, 'alpha': alpha})
            
df.to_csv("experiments/boardgame_qa_with_persona_initial_11_13.csv", index=False)
