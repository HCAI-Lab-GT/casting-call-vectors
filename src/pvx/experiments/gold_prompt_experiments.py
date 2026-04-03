from collections import defaultdict
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import pandas as pd
import argparse

from pvx import setup_logging
from pvx.utils.response_generation import ResponseGeneration

from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.roles_layers.assistant_axis_persona_model import AssistantAxisPersonaModel
from pvx.implementations.roles_layers.role_layers_persona_model import RoleLayersPersonaModel
from pvx.implementations.judges.llm_as_judge import LLMJudge, PROMPT_TEMPLATE, GOLD_COMPARATOR_PROMPT_TEMPLATE

logger = setup_logging(name="gold-prompt-experiments")

class GoldPromptExperiments():
    def __init__(self, 
                 persona_model: type[AbstractPersonaModel] = None, 
                 target_model_id: str = "allenai/Olmo-3-7B-Instruct",
                 judge_model_id: str = "gpt-4.1-mini",
                 json_filepath: str = "./persona_data/persona_models.json",
                 safetensors_dir: str = "./persona_data/model_inits/",
                 assistant_axis_pt_dir: str = "./assistant_axis_vectors",
                 assistant_axis_layer: int | None = None,
                 assistant_axis_alpha: float = 2.5,
                 gold_prompts_dir: str = "./persona_data/gold_labels_prompts_dataset",
                 validation_questions_file: str = "./configs/validation_questions.jsonl",
                 save_dir: str = "./experiment_data/gold_prompt_experiments/"):
        '''
        Initialize the GoldPromptExperiments class.
        
        Args:
            persona_model (type[AbstractPersonaModel], optional): A class of a persona model to use for
                generating responses. If None, a default RolePersonaModel will be used.
            judge_model_id (str, optional): The model ID to use for the LLM judge. Defaults to "gpt-4-0613".
            json_filepath (str, optional): The filepath to the JSON file containing persona model data. Defaults to "./persona_data/persona_models.json".
        '''
        self.json_filepath = json_filepath
        self.safetensors_dir = safetensors_dir
        self.assistant_axis_pt_dir = assistant_axis_pt_dir
        self.assistant_axis_layer = assistant_axis_layer
        self.assistant_axis_alpha = assistant_axis_alpha
        self.gold_prompts_dir = Path(gold_prompts_dir)
        self.validation_questions_file = Path(validation_questions_file)
        self.target_model_id = target_model_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.result_columns = [
            "role",
            "layer",
            "sample_count",
            "alpha",
            "temperature",
            "question",
            "assistant_axis",
            "baseline",
            "steered",
            "assistant_axis_score",
            "baseline_score",
            "steered_score",
            "cmp_emotional_register",
            "cmp_vocab_choice",
            "cmp_social_dynamic",
            "cmp_motivation",
            "cmp_worldview_alignment",
        ]
        self._gold_baseline_messages: dict[str, list[dict[str, str]]] = {}
        self._gold_role_descriptions: dict[str, str] = {}

        self.persona_model = persona_model or RoleLayersPersonaModel
        self.persona_models = defaultdict(dict) # dict of dicts to hold persona models for each role, layer, and dataset count config
        
        self.role_judge = LLMJudge(model=judge_model_id, prompt_template=PROMPT_TEMPLATE)
        self.comparator_judge = LLMJudge(model=judge_model_id, prompt_template=GOLD_COMPARATOR_PROMPT_TEMPLATE)
        self.comparator_judge.judge_func = self.comparator_judge._aggregate_gold_comparator_score
        
        self.generate_response = ResponseGeneration()

    def _get_role_save_path(self, role: str) -> Path:
        safe_role = role.replace("/", "_")
        return self.save_dir / f"Comparison_GoldStandard_{safe_role}.csv"

    def _load_validation_questions(self) -> list[str]:
        if not self.validation_questions_file.exists():
            raise FileNotFoundError(
                f"Validation questions file not found: {self.validation_questions_file}"
            )

        questions: list[str] = []
        with open(self.validation_questions_file, "r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {self.validation_questions_file}: {exc}"
                    ) from exc

                question = str(record.get("question", "")).strip()
                if question:
                    questions.append(question)

        if not questions:
            raise ValueError(
                f"No valid questions were found in {self.validation_questions_file}"
            )

        logger.info("Loaded %s validation questions from %s", len(questions), self.validation_questions_file)
        return questions

    def _empty_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.result_columns)

    def _load_existing_results(self, save_path: Path) -> pd.DataFrame:
        if not save_path.exists():
            return self._empty_results_df()

        try:
            if save_path.stat().st_size == 0:
                logger.warning("Results file exists but is empty; reinitializing: %s", save_path)
                return self._empty_results_df()

            loaded_df = pd.read_csv(save_path)
            if loaded_df.empty:
                return self._empty_results_df()

            return loaded_df
        except pd.errors.EmptyDataError:
            logger.warning("Results file contains no parseable CSV rows; reinitializing: %s", save_path)
            return self._empty_results_df()

    def _load_gold_baseline_messages(self, role: str) -> list[dict[str, str]]:
        if role in self._gold_baseline_messages:
            return deepcopy(self._gold_baseline_messages[role])

        gold_path = self.gold_prompts_dir / f"{role.replace('/', '_')}_gold_label.json"
        if not gold_path.exists():
            raise FileNotFoundError(
                f"Gold prompt file not found for role '{role}' at {gold_path}. "
                "Generate it with scripts/evaluation/create_gold_standard_prompts.py first."
            )

        with open(gold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self._gold_role_descriptions[role] = str(payload.get("role_description", "")).strip()

        messages: list[dict[str, str]] = []
        if isinstance(payload.get("gold_label_prompt"), dict):
            messages = payload["gold_label_prompt"].get("messages", [])
        elif isinstance(payload.get("gold_label_prompts"), list) and payload["gold_label_prompts"]:
            messages = payload["gold_label_prompts"][0].get("messages", [])

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"Gold prompt file for role '{role}' does not contain baseline messages: {gold_path}"
            )

        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role_name = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role_name not in {"system", "user", "assistant"}:
                continue
            if not content:
                continue
            normalized_messages.append({"role": role_name, "content": content})

        if normalized_messages and normalized_messages[-1]["role"] == "user":
            logger.info("Dropping trailing user turn from gold baseline for role '%s'", role)
            normalized_messages = normalized_messages[:-1]

        if not normalized_messages:
            raise ValueError(
                f"Gold prompt file for role '{role}' has no valid baseline messages: {gold_path}"
            )

        self._gold_baseline_messages[role] = normalized_messages
        return deepcopy(normalized_messages)
        
    def load_models(self, roles: list[str], layers: list[int], sample_counts: list[int]):
        '''
        Load or create persona models for each role and layer combination.
        
        Args:
            roles (list[str]): A list of roles to create persona models for.
            layers (list[int]): A list of layers to create persona models for.
            sample_counts (list[int]): A list of dataset counts to use for creating the persona models.
        '''
        persona_configs = list(product(roles, layers, sample_counts))
        
        # load persona models for each config
        for role, layer, sample_count in persona_configs:
            try:
                model = self.persona_model.load_or_create(
                        target_model_id=self.target_model_id,
                        concept=role,
                        layer=layer,
                        target_pairs=sample_count,
                        json_filepath=self.json_filepath,
                        only_load=True,
                        safetensors_dir=self.safetensors_dir,
                    )
                self.persona_models[role][(layer, sample_count)] = model
                
            except FileNotFoundError as e:
                logger.error(e)
                raise
            
            except Exception as e:
                logger.error(e)
                continue
    
    def evaluate_model(self, questions: list = None, 
                       roles=None, layers=None, sample_counts=None, 
                       alphas=None, temperatures=None, max_new_tokens=2000,
                       baseline_temperature=.2,
                       save_each=True, judge_responses=False) -> dict:
        '''
        Evaluates model on variety of parameters on questions.
        Scores baseline and steered results as well as comparisons
        
        Args:
            roles (list[str]): A list of roles to create persona models for.
            layers (list[int]): A list of layers to create persona models for.
            sample_counts (list[int]): A list of dataset counts to use for creating the persona models.
            alphas (list[int]): A list of alphas to create persona models for.
            temperatures (list[int]): A list of temperatures counts to use for creating the persona models.
            max_new_tokens (int): max generated tokens both baseline and steered
            save_each (bool): whether to save csv in progress (default=True)
            questions (list, optional): Explicit list of questions. If empty or None, questions are
                loaded from self.validation_questions_file.
        '''
        
        logger.info("Starting Gold Prompt Experiments with model %s", self.target_model_id)
        
        # instantiation params
        roles = roles or ["Lawyer"]
        layers = layers or [16]
        sample_counts = sample_counts or [40]
        temperatures = temperatures or [0.2]

        # load models for all config into self.persona_models
        self.load_models(roles=roles, layers=layers, sample_counts=sample_counts)

        # evaluation params
        alphas = alphas or [2.0]
        temperatures = temperatures or [0.1]
        role_questions = questions or self._load_validation_questions()
        
        run_configs = list(product(alphas, temperatures))
        all_results: list[pd.DataFrame] = []
                
        for role in roles:
            entry_configs = list(product(self.persona_models[role].items(), run_configs))
            
            logger.info("=== Processing role: %s ===", role)
            baseline_seed_messages = self._load_gold_baseline_messages(role)
            role_description = self._gold_role_descriptions.get(role, "")
            role_save_path = self._get_role_save_path(role)
            role_save_path.parent.mkdir(parents=True, exist_ok=True)

            if not role_save_path.exists():
                self._empty_results_df().to_csv(role_save_path, index=False)
                logger.info("Created role results file: %s", role_save_path)

            # assistant_axis_layer = self.assistant_axis_layer if self.assistant_axis_layer is not None else layers[0]
            assistant_axis_layer = 16
            
            assistant_axis_model = AssistantAxisPersonaModel.load_or_create(
                target_model_id=self.target_model_id,
                concept=role,
                layer=assistant_axis_layer,
                target_pairs=sample_counts[0],
                json_filepath=self.json_filepath,
                safetensors_dir=self.assistant_axis_pt_dir,
            )
            
            pd_results = self._load_existing_results(role_save_path) if save_each else self._empty_results_df()
            
            for question in role_questions:
                # Skip generating assistant axis and basleine if all entries already asked
                pregen = pd_results[pd_results["question"] == question]
                if len(pregen) >= len(entry_configs):
                    logger.info("Skipped question as all entries already generated.")
                    continue
                
                # If question previously asked before, load in Baseline response to avoid asking
                if len(pregen) >= 1:
                    prev_entry = pregen.iloc[0]
                    # assistant_axis_response = pregen["assistant_axis"]
                    # assistant_axis_score = pregen["assistant_axis_score"]
                    
                    # logger.info(
                    #     "Loaded former AssistantAxis response (assistant_axis slot) for role '%s' on layer %s, alpha %s, temperature %s:",
                    #     role,
                    #     assistant_axis_layer,
                    #     self.assistant_axis_alpha,
                    #     baseline_temperature,
                    # )
                    # logger.info(assistant_axis_response)
                
                    
                    baseline_response = prev_entry["baseline"]
                    baseline_score = prev_entry["baseline_score"]
                    
                    logger.info("Loaded former Gold Standard response for role '%s' on temperature %s:", role, baseline_temperature)
                    logger.info(baseline_response)
                
                # Otherwise, retrieve assistance_axis and gold standard
                else:
                    # # Use Assistant Axis generation in place of plain assistant_axis baseline.
                    # assistant_axis_response = assistant_axis_model.generate(
                    #     prompt=question,
                    #     alpha=self.assistant_axis_alpha,
                    #     max_new_tokens=max_new_tokens,
                    #     temperature=baseline_temperature,
                    # )
                    # logger.info(
                    #     "AssistantAxis response (assistant_axis slot) for role '%s' on layer %s, alpha %s, temperature %s:",
                    #     role,
                    #     assistant_axis_layer,
                    #     self.assistant_axis_alpha,
                    #     baseline_temperature,
                    # )
                    # logger.info(assistant_axis_response)
                
                    # # calculate judge score for AssistantAxis response in assistant_axis slot
                    # if judge_responses:
                    #     assistant_axis_score = self.role_judge(role=role, role_description=role_description, question=question, answer=assistant_axis_response)
                    #     logger.info(
                    #         "AssistantAxis judge score for role '%s' on layer %s, alpha %s, temperature %s: %s",
                    #         role,
                    #         assistant_axis_layer,
                    #         self.assistant_axis_alpha,
                    #         baseline_temperature,
                    #         assistant_axis_score,
                    #     )
                    # else:
                    #     assistant_axis_score = -1
                    #     logger.info("AssistantAxis judge scoring skipped on GPU.")

                    # Get prompted role answer for prompted baseline
                    baseline_messages = deepcopy(baseline_seed_messages)
                    baseline_messages.append({"role": "user", "content": question})
                    baseline_response = self.generate_response(messages=baseline_messages,
                                                            max_new_tokens=max_new_tokens,
                                                            temperature=baseline_temperature)[1]
                    logger.info("Gold Standard response for role '%s' on temperature %s:", role, baseline_temperature)
                    logger.info(baseline_response)
                    
                    # calculate judge score for baseline response
                    if judge_responses:
                        baseline_score = self.role_judge(role=role, role_description=role_description, question=question, answer=baseline_response)
                        logger.info("Baseline judge score for role '%s' on temperature %s: %s", role, baseline_temperature, baseline_score)
                    else:
                        baseline_score = -1
                        logger.info("Baseline judge scoring skipped on GPU.")


                # calculate judge score for each config
                for (instant_params, model), (alpha, temperature) in entry_configs:
                    model: AbstractPersonaModel
                    
                    if role_save_path.exists():
                        # if save_each is true, load existing results to check if this config has already been evaluated
                        if save_each:
                            pd_results = self._load_existing_results(role_save_path)

                        if ((pd_results["role"] == model.concept) &
                            (pd_results["layer"] == model.layer_steering) &
                            (pd_results["sample_count"] == model.target_pairs) &
                            (pd_results["alpha"] == alpha) &
                            (pd_results["temperature"] == temperature) &
                            (pd_results["question"] == question)).any():
                            
                            logger.info(
                                "Skipping already evaluated config: role= %s, layer=%s, sample_count=%s, alpha=%s, temperature=%s",
                                role, model.layer_steering, model.target_pairs, alpha, temperature)
                            continue
                        
                    layers, sample_count = instant_params
                    
                    # Use Assistant Axis generation in place of plain assistant_axis baseline.
                    assistant_axis_response = assistant_axis_model.generate(
                        prompt=question,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    logger.info("AssistantAxis response for role '%s' on layer %s, sample_count %s, alpha %s, temperature %s:", 
                                role, model.layer_steering, model.target_pairs, alpha, temperature)
                    logger.info(assistant_axis_response)
                    
                    # calculate judge score for AssistantAxis response in assistant_axis slot
                    if judge_responses:
                        assistant_axis_score = self.role_judge(role=role, role_description=role_description, question=question, answer=assistant_axis_response)
                        logger.info("AssistantAxis judge score for role '%s' on layer %s, alpha %s, temperature %s: %s",
                                    role, model.layer_steering, alpha, temperature, assistant_axis_score)
                    else:
                        assistant_axis_score = -1
                        logger.info("AssistantAxis judge scoring skipped on GPU.")
                    
                    # generate prompted role answer for steered response
                    steered_response = model.generate(
                        prompt=question,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    logger.info("Steered response for role '%s' on layer %s, sample_count %s, alpha %s, temperature %s:", 
                                role, model.layer_steering, model.target_pairs, alpha, temperature)
                    logger.info(steered_response)
                    
                    if judge_responses:
                        # calculate judge score for steered response
                        steered_score = self.role_judge(role=role, role_description=role_description, question=question, answer=steered_response)
                        logger.info("Steered judge score for role '%s' on layer %s, sample_count %s, alpha %s, temperature %s: %s", 
                                    role, model.layer_steering, model.target_pairs, alpha, temperature, steered_score)
                        
                        # calculate comparative scores (multi-dimensional Likert alignment vs gold baseline)
                        comparative_scores = self.comparator_judge(role=role, role_description=role_description, question=question, baseline=baseline_response, answer=steered_response)
                        logger.info("Comparative scores for role '%s': %s", role, comparative_scores)
                    
                    else:
                        steered_score = -1
                        logger.info("Steered judge scoring skipped on GPU.")
                        
                        comparative_scores = defaultdict(lambda: defaultdict(lambda: -1))
                        logger.info("Comparative judge scoring skipped on GPU.")

                        

                    # log results to dataframe
                    new_result = {
                        "role": model.concept,
                        "layer": model.layer_steering,
                        "sample_count": model.target_pairs,
                        "alpha": alpha,
                        "temperature": temperature,
                        "question": question,
                        "assistant_axis": assistant_axis_response,
                        "baseline": baseline_response,
                        "steered": steered_response,
                        "assistant_axis_score": assistant_axis_score,
                        "baseline_score": baseline_score,
                        "steered_score": steered_score,
                        "cmp_emotional_register": comparative_scores["style"]["emotional_register"],
                        "cmp_vocab_choice":        comparative_scores["style"]["vocab_choice"],
                        "cmp_social_dynamic":      comparative_scores["style"]["social_dynamic"],
                        "cmp_motivation":          comparative_scores["content"]["motivation"],
                        "cmp_worldview_alignment": comparative_scores["content"]["worldview_alignment"],
                    }
                    new_result_df = pd.DataFrame([new_result])
                    if pd_results.empty:
                        pd_results = new_result_df
                    else:
                        pd_results = pd.concat([pd_results, new_result_df], ignore_index=True)
                    
                    # save results after each config if save_each is true
                    if save_each:
                        pd_results.sort_values(by=["role", "layer", "sample_count", "alpha", "temperature"], inplace=True)
                        pd_results.to_csv(role_save_path, index=False)
                        logger.info("Saved progress to %s", role_save_path)
                        
            pd_results.sort_values(by=["role", "layer", "sample_count", "alpha", "temperature"], inplace=True)
            pd_results.to_csv(role_save_path, index=False)
            logger.info("Results saved to %s", role_save_path)
            all_results.append(pd_results.copy())
            logger.info("=== Concluded processing role: %s ===", role)
            
                        
        logger.info("=== Final Results ===")

        if not all_results:
            return self._empty_results_df()

        return pd.concat(all_results, ignore_index=True)
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model", type=str, default="allenai/Olmo-3-7B-Instruct", help="HF model typically",
    )
    ap.add_argument(
        "-r", "--roles", nargs="+", type=str, default=["Lawyers"], help="Roles of the persona dataset"
    )
    ap.add_argument(
        "-l", "--layers", nargs="+", type=int, default=[16], help="List of layers to extract activations from (default: 14)"
    )
    ap.add_argument(
        "-s", "--sample_counts", nargs="+", type=int, default=[20, 40, 50], help="List of dataset counts to extract activations from (default: 14)"
    )
    ap.add_argument(
        "-a", "--alphas", nargs="+", type=float, default=[2.5], help="Alpha value for persona steering"
    )
    ap.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.2], help="Temperature for sampling"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="Optional single question override; if omitted, questions are loaded from --questions_file",
    )
    ap.add_argument(
        "--questions_file",
        type=str,
        default="./configs/validation_questions.jsonl",
        help="JSONL file containing validation questions with a 'question' field",
    )
    ap.add_argument(
        "-f",
        "--safetensors_dir",
        type=str,
        default="./persona_data/model_inits/",
        help="Directory containing model layer initialization safetensors files",
    )
    ap.add_argument(
        "--pt_dir",
        type=str,
        default="persona_data/assistant-axis/olmo-3-7b-instruct/vectors/",
        # default="../assistant-axis-pvx/outputs/olmo-3-7b-instruct/vectors/",
        help="Directory containing Assistant Axis .pt files, one per role",
    )
    ap.add_argument(
        "--assistant_axis_layer",
        type=int,
        default=16,
        help="Layer to use for Assistant Axis assistant_axis replacement (defaults to first value in --layers)",
    )
    ap.add_argument(
        "--assistant_axis_alpha",
        type=float,
        default=2.5,
        help="Alpha to use for Assistant Axis assistant_axis replacement",
    )
    ap.add_argument(
        "-d",
        "--save_dir",
        type=str,
        default="./experiment_data/gold_prompt_experiments/",
        help="Directory to save results to, or an explicit CSV output path",
    )
    ap.add_argument(
        "--gold_prompts_dir",
        type=str,
        default="./persona_data/gold_labels_prompts_dataset",
        help="Directory containing {role}_gold_label.json files",
    )
    ap.add_argument(
        "-j",
        "--judge_responses",
        action="store_true",
        help="Judge after generated responses",
    )
    args = ap.parse_args()
    
    experiment = GoldPromptExperiments(
        persona_model=RoleLayersPersonaModel,
        target_model_id=args.model, 
        safetensors_dir=args.safetensors_dir,
        assistant_axis_pt_dir=args.pt_dir,
        assistant_axis_layer=args.assistant_axis_layer, # DEPRECATED - uses 16
        assistant_axis_alpha=args.assistant_axis_alpha, # DEPRECATED - uses reg alpha
        gold_prompts_dir=args.gold_prompts_dir,
        validation_questions_file=args.questions_file,
        save_dir=args.save_dir
    )
    
    experiment.evaluate_model(
        questions=[args.question] if args.question else None,
        roles=args.roles, layers=args.layers, sample_counts=args.sample_counts,
        alphas=args.alphas, temperatures=args.temperatures, max_new_tokens=args.max_new_tokens,
        save_each=True, judge_responses=args.judge_responses
    )