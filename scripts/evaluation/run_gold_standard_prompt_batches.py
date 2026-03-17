import argparse
import subprocess
import sys
from pathlib import Path

from pvx import setup_logging

logger = setup_logging(name="run-gold-standard-prompt-batches")


def discover_roles(role_dir: Path) -> list[str]:
	roles: list[str] = []
	seen: set[str] = set()
	for path in sorted(role_dir.glob("*.json")):
		if "_dataset" in path.stem:
			continue
		role = path.stem
		if role in seen:
			continue
		seen.add(role)
		roles.append(role)
	return roles


def chunk_roles(roles: list[str], batch_size: int) -> list[list[str]]:
	return [roles[index : index + batch_size] for index in range(0, len(roles), batch_size)]


def output_exists(role: str, output_dir: Path) -> bool:
	output_path = output_dir / f"{role.replace('/', '_')}_gold_label.json"
	return output_path.exists()


def build_command(args: argparse.Namespace, roles: list[str]) -> list[str]:
	command = [
		sys.executable,
		str(args.generator_script),
		"--roles",
		*roles,
		"--backend",
		args.backend,
		"--model",
		args.model,
		"--base_url",
		args.base_url,
		"--api_key_env",
		args.api_key_env,
		"--local_model",
		args.local_model,
		"--temperature",
		str(args.temperature),
		"--max_new_tokens",
		str(args.max_new_tokens),
		"--num_dialogue_prompts",
		str(args.num_dialogue_prompts),
		"--output_dir",
		str(args.output_dir),
	]
	if args.device:
		command.extend(["--device", args.device])
	if args.overwrite:
		command.append("--overwrite")
	return command


def main() -> None:
	parser = argparse.ArgumentParser(description="Run gold-standard prompt generation in batches")
	parser.add_argument("--roles", nargs="*", default=None, help="Optional explicit role list")
	parser.add_argument("--role_dir", default="./persona_data/role_datasets")
	parser.add_argument("--output_dir", default="./persona_data/gold_labels_prompts_dataset")
	parser.add_argument(
		"--generator_script",
		default="./scripts/evaluation/create_gold_standard_prompts.py",
	)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--max_batches", type=int, default=None)
	parser.add_argument("--start_batch", type=int, default=0)
	parser.add_argument("-b", "--backend", default="openai", choices=["openai", "vllm", "hf_local"])
	parser.add_argument("-m", "--model", default="openai/gpt-4.1-mini")
	parser.add_argument("--base_url", default="https://openrouter.ai/api/v1")
	parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
	parser.add_argument("--local_model", default="Qwen/Qwen2.5-7B-Instruct")
	parser.add_argument("--device", default=None)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--max_new_tokens", type=int, default=2048)
	parser.add_argument("--num_dialogue_prompts", type=int, default=5)
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	if args.batch_size <= 0:
		raise ValueError("--batch_size must be positive")
	if args.max_batches is not None and args.max_batches <= 0:
		raise ValueError("--max_batches must be positive")
	if args.start_batch < 0:
		raise ValueError("--start_batch cannot be negative")

	role_dir = Path(args.role_dir)
	output_dir = Path(args.output_dir)
	roles = args.roles or discover_roles(role_dir)
	if not args.overwrite:
		roles = [role for role in roles if not output_exists(role, output_dir)]

	batches = chunk_roles(roles, args.batch_size)
	if args.start_batch:
		batches = batches[args.start_batch :]
	if args.max_batches is not None:
		batches = batches[: args.max_batches]

	if not batches:
		logger.info("No roles to process")
		return

	if args.dry_run:
		logger.info("Dry run enabled; batch role lists will be printed but nothing will be executed")

	for batch_index, batch_roles in enumerate(batches, start=args.start_batch + 1):
		if args.dry_run:
			logger.info("Batch %d roles: %s", batch_index, ", ".join(batch_roles))
			continue
		logger.info("Running batch %d with roles: %s", batch_index, ", ".join(batch_roles))
		command = build_command(args, batch_roles)
		subprocess.run(command, check=True)


if __name__ == "__main__":
	main()