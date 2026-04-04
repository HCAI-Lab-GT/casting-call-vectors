import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pvx import setup_logging
from pvx.utils.response_generation import ResponseGeneration

logger = setup_logging(name="create-gold-standard-prompts")

SYSTEM_TEMPLATE = (
	"You are {role_name}, your description is: {role_description_and_catchphrases}. Now please\n"
	"answer some questions to accurately show your personality traits! Your speaking style should\n"
	"fully imitate the personality role assigned to you! Please do not expose that you are an artificial\n"
	"intelligence model or a language model, you must always remember that you are only assigned\n"
	"one personality role. Don't be verbose or too formal or polite when speaking."
)

ROLE_DESCRIPTION_PROMPT = """
First Step (generate description):
You are a role description model. Please use a sentence or a paragraph to describe the
role I give you. Including but not limited to: the role's personality description, the
role's life experience, the role's personality changes, the role's main story line,
the role's important events, etc. The name of the role should not appear in the
description, and the description should not be too long. Please start with ``The role's
description is: '' and then refer to it as ``the role''.
Second Step (convert from third-person description to second-person description):
Please change the third person of this sentence to the second person, and start with ``Your
description is:''.

Role name: {role}

Return only the final second-person description text.
""".strip()

CATCHPHRASES_PROMPT = """
I will give you some role names in movies and TV shows, and you need to tell me the
catchphrases of this role. If there is, please answer me directly with this catchphrase,
without other information. If not, please answer ``no''. Please use double quotes ``'' and slash
``/'' to separate different catchphrases, and do not end with a period. For example, if I ask you:
for the role ``judge'', what are the catchphrases? You only need to answer me:
``Justice has been served'?'' or ``no''. If there are multiple catchphrases, please separate them with a
slash ``/''.

Role name: {role}

Return only the catchphrase text(s) or "no".
""".strip()

FEW_SHOT_PROMPT = """
You are building a gold-label roleplaying evaluation dataset.

Role: {role}
Role description: {role_description}
Catchphrases and speaking cues: {catchphrases_description}

Below are the dialogue prompts to use as the few-shot user prompts:
{dialogue_prompts_block}

Generate one short in-character assistant answer for each prompt in the same order.
The assistant must not mention being an AI model.

Return valid JSON only:
{{"assistant_answers": ["...", "..."]}}
""".strip()

DIALOGUE_PROMPTS_PROMPT = """
You are building a gold-label roleplaying evaluation dataset.

Role: {role}
Role description: {role_description}
Catchphrases and speaking cues: {catchphrases_description}

Generate {count} dialogue prompts that a user could ask to reveal this role's personality in conversation.
Each prompt must be a single user turn, direct and concrete, and should not explicitly ask the model to "act as" the role.

Return valid JSON only:
{{"dialogue_prompts": ["..."]}}
""".strip()


def parse_json_response(response: str) -> dict[str, Any]:
	cleaned = response.strip()
	if cleaned.startswith("```"):
		cleaned = cleaned.strip("`").strip()
		if cleaned.startswith("json"):
			cleaned = cleaned[4:].strip()
	start = cleaned.find("{")
	end = cleaned.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError("Model response did not contain a JSON object")
	return json.loads(cleaned[start : end + 1])


def parse_text_response(response: str) -> str:
	cleaned = response.strip()
	if cleaned.startswith("```"):
		cleaned = cleaned.strip("`").strip()
		lowered = cleaned.lower()
		for prefix in ("json", "text", "markdown"):
			if lowered.startswith(prefix):
				cleaned = cleaned[len(prefix) :].strip()
				break
	if cleaned.startswith("{") and cleaned.endswith("}"):
		try:
			payload = json.loads(cleaned)
		except json.JSONDecodeError:
			return cleaned
		if isinstance(payload, dict):
			for key in (
				"role_description",
				"catchphrases_description",
				"catchphrases",
				"text",
				"response",
			):
				value = payload.get(key)
				if isinstance(value, str) and value.strip():
					return value.strip()
			if len(payload) == 1:
				only_value = next(iter(payload.values()))
				if isinstance(only_value, str) and only_value.strip():
					return only_value.strip()
	return cleaned.strip()


def generate_payload(
	generator: ResponseGeneration,
	label: str,
	prompt: str,
	temperature: float,
	max_new_tokens: int,
) -> dict[str, Any]:
	last_error: Exception | None = None
	messages = [
		{"role": "system", "content": "You are an expert dataset designer."},
		{"role": "user", "content": prompt},
	]
	for attempt in range(1, 4):
		_, response = generator(messages=messages, temperature=temperature, max_new_tokens=max_new_tokens)
		try:
			return parse_json_response(response)
		except Exception as exc:
			last_error = exc
			logger.warning("Failed to parse %s on attempt %d: %s", label, attempt, exc)
	raise ValueError(f"Failed to generate parseable JSON for {label}") from last_error


def generate_text_response(
	generator: ResponseGeneration,
	label: str,
	prompt: str,
	temperature: float,
	max_new_tokens: int,
) -> str:
	last_error: Exception | None = None
	messages = [
		{"role": "system", "content": "You are an expert dataset designer."},
		{"role": "user", "content": prompt},
	]
	for attempt in range(1, 4):
		_, response = generator(messages=messages, temperature=temperature, max_new_tokens=max_new_tokens)
		try:
			parsed = parse_text_response(response)
			if parsed:
				return parsed
			raise ValueError("Model response was empty")
		except Exception as exc:
			last_error = exc
			logger.warning("Failed to parse %s on attempt %d: %s", label, attempt, exc)
	raise ValueError(f"Failed to generate parseable text for {label}") from last_error


def render_prompt(system_instruction: str, demonstrations: list[dict[str, str]]) -> str:
	parts = [f"System Instruction:\n{system_instruction}"]
	for demo in demonstrations:
		parts.append(f"User Prompt:\n{demo['user_prompt']}")
		parts.append(f"Assistant Prompt:\n{demo['assistant_prompt']}")
	return "\n".join(parts)


def build_messages(system_instruction: str, demonstrations: list[dict[str, str]]) -> list[dict[str, str]]:
	messages = [{"role": "system", "content": system_instruction}]
	for demo in demonstrations:
		messages.append({"role": "user", "content": demo["user_prompt"]})
		messages.append({"role": "assistant", "content": demo["assistant_prompt"]})
	return messages


def main() -> None:
	parser = argparse.ArgumentParser(description="Create gold-label prompts for one or more roles")
	parser.add_argument("-r", "--roles", nargs="+", required=True, help="Role names to generate")
	parser.add_argument("-b", "--backend", default="openai", choices=["openai", "vllm", "hf_local"])
	parser.add_argument("-m", "--model", default="openai/gpt-4.1-mini")
	parser.add_argument("--base_url", default="https://openrouter.ai/api/v1")
	parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
	parser.add_argument("--local_model", default="Qwen/Qwen2.5-7B-Instruct")
	parser.add_argument("--device", default=None)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--max_new_tokens", type=int, default=2048)
	parser.add_argument("--num_dialogue_prompts", type=int, default=5)
	parser.add_argument("--output_dir", default="./persona_data/gold_labels_prompts_dataset")
	parser.add_argument("--overwrite", action="store_true")
	args = parser.parse_args()

	generator = ResponseGeneration(
		backend=args.backend,
		model=args.model,
		local_model=args.local_model,
		base_url=args.base_url,
		api_key_env=args.api_key_env,
		device=args.device,
	)
	output_dir = Path(args.output_dir)

	for role in args.roles:
		role_prompt = ROLE_DESCRIPTION_PROMPT.format(role=role)
		role_description = generate_text_response(
			generator,
			"role description",
			role_prompt,
			args.temperature,
			args.max_new_tokens,
		).strip()

		catch_prompt = CATCHPHRASES_PROMPT.format(
			role=role,
		)
		catchphrases_description = generate_text_response(
			generator,
			"catchphrases",
			catch_prompt,
			args.temperature,
			args.max_new_tokens,
		).strip()

		dialogue_prompt = DIALOGUE_PROMPTS_PROMPT.format(
			role=role,
			role_description=role_description,
			catchphrases_description=catchphrases_description,
			count=args.num_dialogue_prompts,
		)
		dialogue_payload = generate_payload(generator, "dialogue prompts", dialogue_prompt, args.temperature, args.max_new_tokens)
		dialogue_prompts = [str(prompt).strip() for prompt in dialogue_payload["dialogue_prompts"]]

		dialogue_prompts_block = "\n".join(
			f"{index}. {prompt}" for index, prompt in enumerate(dialogue_prompts, start=1)
		)
		few_shot_prompt = FEW_SHOT_PROMPT.format(
			role=role,
			role_description=role_description,
			catchphrases_description=catchphrases_description,
			dialogue_prompts_block=dialogue_prompts_block,
		)
		few_shot_payload = generate_payload(
			generator,
			"few-shot demonstrations",
			few_shot_prompt,
			args.temperature,
			args.max_new_tokens,
		)

		assistant_answers = [str(answer).strip() for answer in few_shot_payload.get("assistant_answers", [])]
		if not assistant_answers and "few_shot_demonstrations" in few_shot_payload:
			assistant_answers = [
				str(demo.get("assistant_prompt", "")).strip()
				for demo in list(few_shot_payload["few_shot_demonstrations"])
			]

		if len(assistant_answers) != len(dialogue_prompts):
			raise ValueError(
				"Few-shot answer count does not match dialogue prompt count: "
				f"{len(assistant_answers)} vs {len(dialogue_prompts)}"
			)

		demonstrations = [
			{"user_prompt": prompt, "assistant_prompt": answer}
			for prompt, answer in zip(dialogue_prompts, assistant_answers)
		]

		role_description_and_catchphrases = (
			f"{role_description} Signature catchphrases and speaking cues: {catchphrases_description}"
		)
		system_instruction = SYSTEM_TEMPLATE.format(
			role_name=role,
			role_description_and_catchphrases=role_description_and_catchphrases,
		)
		final_prompt = {
			"id": 1,
			"dialogue_prompts": dialogue_prompts,
			"messages": build_messages(system_instruction, demonstrations),
			"rendered_prompt": render_prompt(system_instruction, demonstrations),
		}

		output_path = output_dir / f"{role.replace('/', '_')}_gold_label.json"
		if output_path.exists() and not args.overwrite:
			raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")
		output_path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"role": role,
			"backend": args.backend,
			"model": args.model,
			"base_url": args.base_url,
			"generated_at": datetime.now(timezone.utc).isoformat(),
			"role_description": role_description,
			"catchphrases_description": catchphrases_description,
			"role_description_and_catchphrases": role_description_and_catchphrases,
			"few_shot_demonstrations": demonstrations,
			"dialogue_prompts": dialogue_prompts,
			"generation_prompts": {
				"role_description_prompt": role_prompt,
				"catchphrases_prompt": catch_prompt,
				"dialogue_prompts_prompt": dialogue_prompt,
				"few_shot_answers_prompt": few_shot_prompt,
			},
			"gold_label_prompt": final_prompt,
		}
		output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
		logger.info("Saved gold-label prompts for %s to %s", role, output_path)


if __name__ == "__main__":
	main()
