import pandas as pd
from persona_vector_generation.PersonaInference import PersonaInference
import asyncio
from tqdm import tqdm

# Login using e.g. `huggingface-cli login` to access this dataset
# splits = {'train': 'Boardgame-QA_train.csv', 'validation': 'Boardgame-QA_valid.csv', 'test': 'Boardgame-QA_test.csv'}
# df = pd.read_csv("hf://datasets/1-800-LLMs/Boardgame-QA/" + splits["train"])
# df = df[['example', 'proof']]
# print(df.head())
# df.to_csv("boardgame_qa_train.csv", index=False)

df = pd.read_csv("boardgame_qa_train.csv")
df.sample(frac=1).reset_index(drop=True)

def benchmark_with_persona(trait:str, trait_file: str=None, max_n:int=15000, df: pd.DataFrame=df, alpha: int=0.5):
    persona_inference = PersonaInference(trait=trait, trait_file=trait_file)
    scores = []
    n = min(max_n, len(df))
    for index, row in tqdm(df.iterrows(), total=n, desc=f"Evaluating Boardgame-QA with trait {trait}"):
        prompt = row['example'] + "\nAfter considering the problem carefully, end your answer with \"YES\" or \"NO\"."
        answer = persona_inference.inference_with_persona(prompt=row['example'], alpha=0.5, temperature=0.8, max_new_tokens=1000)
        ground_truth = row['proof'].find("\"yes\"") != -1
        predicted_answer = None
        if "YES" in answer.upper():
            predicted_answer = True
        elif "NO" in answer.upper():
            predicted_answer = False
        scores.append(predicted_answer is not None and (predicted_answer == ground_truth))
    accuracy = sum(scores) / len(scores)
    print(f"Accuracy with trait {trait}: {accuracy}")
    return scores
n = 100
df = df.loc[:n].copy()
df['control'] = benchmark_with_persona(trait="verbose", trait_file='persona_vectors/verbose_persona_vector.json', max_n=n, alpha=0.0, df=df)
df['evil'] = benchmark_with_persona(trait="evil", trait_file='persona_vectors/evil_persona_vector.json', max_n=n, alpha=0.3,df=df)
df['sarcastic'] = benchmark_with_persona(trait="sarcastic", trait_file='persona_vectors/sarcastic_persona_vector.json', alpha=0.3, max_n=n)
# df['evil'] = benchmark_with_persona(trait="evil", max_n=20)
# df['kind'] = benchmark_with_persona(trait="evil", max_n=20)
# df['cautious'] = benchmark_with_persona(trait="evil", max_n=20)
# df['brave'] = benchmark_with_persona(trait="evil", max_n=20)
df.to_csv("scratch.csv", index=False)