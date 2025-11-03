import json
import asyncio
from pathlib import Path
from prompts import PROMPTS
from mlx_lm import load, generate
from mlx_lm.tuner import train

class TraitGenerator:

    def __init__(self, model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit", output_dir='./traits_output'):
        self.model, self.tokenizer = load(model_id)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
    
    async def inference_request(self, user_prompt: str):
        system_prompt = "You are an expert AI evaluator and dataset designer."
        user_prompt = user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply chat template (converts to proper format for the model)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=15000,
            verbose=False
        )

        text = response.strip()

        print(text)

        # try parsing the JSON directly (the model is instructed to output pure JSON)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print("⚠️ Model did not return clean JSON. Saving raw text for inspection.")
            data = {"raw_text": text}
    
        return data
    
    async def generate_trait_instruction(self, trait: str) -> dict:
        system_prompt = "You are an expert AI evaluator and dataset designer."
        user_prompt = PROMPTS["trait_instruction"].format(
            TRAIT=trait
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply chat template (converts to proper format for the model)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=15000,
            verbose=False
        )

        text = response.strip()

        print(text)

        # try parsing the JSON directly (the model is instructed to output pure JSON)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print("⚠️ Model did not return clean JSON. Saving raw text for inspection.")
            data = {"raw_text": text}
    
        return data
    
    
    async def generate_positive_negative_samples(self, trait: str, trait_instruction: str, question_instruction: str = "") -> dict:
        system_prompt = "You are an expert AI evaluator and dataset designer."
        user_prompt = PROMPTS["generate_trait"].format(
            TRAIT=trait,
            trait_instruction=trait_instruction,
            question_instruction=question_instruction or ""
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply chat template (converts to proper format for the model)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=15000,
            verbose=False
        )

        text = response.strip()

        print(text)

        # try parsing the JSON directly (the model is instructed to output pure JSON)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print("⚠️ Model did not return clean JSON. Saving raw text for inspection.")
            data = {"raw_text": text}

        # save
        out_file = self.output_dir / f"{trait.replace(' ', '_')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Generated dataset for trait '{trait}' → {out_file}")
        return data
    
    async def generate_pos_neg_examples_v2(self, trait: str) -> dict:
        trait_instruction_map = await self.inference_request(user_prompt=PROMPTS["trait_instruction"].format(TRAIT=trait))
        trait_instruction = trait_instruction_map['trait_instruction']

        question_instruction_map = await self.inference_request(user_prompt=PROMPTS["question_instruction"].format(TRAIT=trait))
        question_instruction = question_instruction_map['question_instruction']

        positive_negative_examples_map = await self.inference_request(user_prompt=PROMPTS["generate_trait"].format(TRAIT=trait, trait_instruction=trait_instruction, question_instruction=question_instruction))

        out_file = self.output_dir / f"{trait.replace(' ', '_')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(positive_negative_examples_map, f, indent=2, ensure_ascii=False)

        print(f"✅ Generated dataset for trait '{trait}' → {out_file}")
        return positive_negative_examples_map




if __name__ == "__main__":
    generator = TraitGenerator(model_id="mlx-community/Qwen2.5-7B-Instruct-4bit")

    # trait_instruction = asyncio.run(generator.generate_trait_instruction(trait="sarcastic"))

    # trait_instruction = asyncio.run(generator.inference_request(user_prompt=PROMPTS["question_instruction"].format(TRAIT="sarcastic")))

    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="sarcastic"))

    print(trait_instruction)
    # trait_data = asyncio.run(generator.generate(
    #     trait="sarcastic",
    #     trait_instruction="Tends to make witty, mocking remarks and uses irony to convey humor.",
    #     question_instruction="Include both social and factual scenarios where sarcasm could naturally appear."
    # ))

    # print(trait_data.keys())

    
    
