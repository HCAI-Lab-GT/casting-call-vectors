"""
This is a simple script to test the vocational responses with the system prompts and questions generated from `generate_vocational_personas.py`. It loads the persona definitions and runs inference to see how the model responds to the vocational prompts.
"""

import argparse
import json
import os
from itertools import product
from tempfile import tempdir

from openai import OpenAI
from tqdm import tqdm


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def save_jsonl(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_client():
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not base_url:
        raise EnvironmentError("OPENAI_BASE_URL is not set in the environment.")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    return OpenAI(base_url=base_url, api_key=api_key)


def infer(client, system_prompt, question, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    )
    return response.choices[0].message.content


def extract_system_prompts(persona_data, num_sys_prompts=None):
    system_prompts = [entry["pos"] for entry in persona_data["instruction"]]
    if num_sys_prompts is not None:
        num_sys = min(num_sys_prompts, 5, len(system_prompts))
        system_prompts = system_prompts[:num_sys]
    return system_prompts


def extract_questions(persona_data, num_questions=None):
    questions = persona_data["questions"]
    if num_questions is not None:
        questions = questions[:num_questions]
    return questions


def derive_output_file(persona_path, output_dir):
    persona_name = os.path.splitext(os.path.basename(persona_path))[0]
    return os.path.join(output_dir, f"{persona_name}_responses.jsonl")


def run_inference(client, system_prompts, questions, model="gpt-4o"):
    records = []
    combinations = list(product(enumerate(system_prompts), enumerate(questions)))

    for (sp_idx, sys_prompt), (q_idx, question) in tqdm(
        combinations,
        desc="Running inference",
        total=len(combinations),
    ):
        response = infer(client, sys_prompt, question, model=model)
        records.append(
            {
                "sys_prompt": sys_prompt,
                "question": question,
                "response": response,
            }
        )

    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Test vocational persona responses")
    parser.add_argument(
        "--persona_path",
        default="persona_data/vocational_personas/instructions/actors.json",
        help="Path to the generated persona file",
    )
    parser.add_argument(
        "--output_path",
        default="persona_data/vocational_personas/sample_responses",
        help="Directory to save the output JSONL file",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=None,
        help="Number of questions to use (subset). If not specified, all questions are used.",
    )
    parser.add_argument(
        "--num_sys_prompts",
        type=int,
        default=None,
        help="Number of system prompts to use (max 5). If not specified, all are used.",
    )

    parser.add_argument(
        "--model",
        default="qwen/qwen3-235b-a22b-2507",
        help="Model to use for inference (e.g., gpt-4o, qwen/qwen3-235b-a22b-2507, etc.)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load persona data
    persona_data = load_json(args.persona_path)

    # Extract system prompts and questions
    system_prompts = extract_system_prompts(persona_data, args.num_sys_prompts)
    questions = extract_questions(persona_data, args.num_questions)

    # Derive output filename from persona path
    output_file = derive_output_file(args.persona_path, args.output_path)

    # Initialize OpenAI client
    client = create_client()

    # Run inference for all combinations of system prompts and questions
    print(
        f"Running inference for {len(system_prompts)} system prompts x {len(questions)} questions = {len(system_prompts) * len(questions)} combinations"
    )

    records = run_inference(client, system_prompts, questions, model=args.model)

    # Save results
    save_jsonl(records, output_file)
    print(f"Saved {len(records)} responses to {output_file}")


if __name__ == "__main__":
    main()
