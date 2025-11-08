import json
import asyncio
from pathlib import Path
from data_generation.prompts import PROMPTS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TraitGenerator:

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-32B-Instruct", output_dir='./traits_output'):
        # Load model and tokenizer using Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_response(self, messages: list, max_tokens: int = 15000) -> str:
        """Helper method to generate responses using Hugging Face transformers"""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated tokens (excluding the input prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    async def inference_request(self, user_prompt: str):
        system_prompt = "You are an expert AI evaluator and dataset designer."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate response using the helper method
        text = self._generate_response(messages)

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

        # Generate response using the helper method
        text = self._generate_response(messages)

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

        # Generate response using the helper method
        text = self._generate_response(messages)

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
    
    def extract_pos_neg_question_pairs(self, trait_data: dict):
        """
        Extract pairs of (positive, negative, question) from trait data.
        Returns a list of objects with pos, neg, and question attributes.
        """
        from collections import namedtuple
        
        Pair = namedtuple('Pair', ['pos', 'neg', 'question'])
        pairs = []
        
        instructions = trait_data.get('instruction', [])
        questions = trait_data.get('questions', [])
        
        # Create pairs by pairing each instruction pair with each question
        for instruction_pair in instructions:
            for question in questions:
                pair = Pair(
                    pos=instruction_pair.get('pos', ''),
                    neg=instruction_pair.get('neg', ''),
                    question=question
                )
                pairs.append(pair)
        
        return pairs
    
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
    generator = TraitGenerator(model_id="Qwen/Qwen2.5-32B-Instruct")
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="evil"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="kind"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="brave"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="cautious"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="stoic"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="efficient"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="verbose"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="analytical"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="lazy"))
    trait_instruction = asyncio.run(generator.generate_pos_neg_examples_v2(trait="cynical"))
    print(trait_instruction)


    
    
