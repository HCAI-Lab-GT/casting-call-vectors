from collections import defaultdict
from itertools import product
from pathlib import Path
import pandas as pd
import argparse

from pvx import setup_logging
from pvx.utils.response_generation import ResponseGeneration

from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.roles_layers.role_layers_persona_model import RoleLayersPersonaModel
from pvx.implementations.judges.llm_as_judge import LLMJudge, PROMPT_TEMPLATE, COMPARATOR_PROMPT_TEMPLATE

logger = setup_logging(name="gold-prompt-experiments")

class GoldPromptExperiments():
    def __init__(self, 
                 persona_model: type[AbstractPersonaModel] = None, 
                 target_model_id: str = "allenai/Olmo-3-7B-Instruct",
                 judge_model_id: str = "gpt-4.1-mini",
                 json_filepath: str = "./persona_data/persona_models.json",
                 safetensors_dir: str = "./persona_data/model_experiments_inits/",
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
        self.target_model_id = target_model_id
        self.save_path = Path(save_dir, f'{target_model_id.split("/")[1]}_results.csv')

        self.persona_model = persona_model or RoleLayersPersonaModel
        self.persona_models = defaultdict(dict) # dict of dicts to hold persona models for each role, layer, and dataset count config
        
        self.role_judge = LLMJudge(model=judge_model_id, prompt_template=PROMPT_TEMPLATE)
        self.comparator_judge = LLMJudge(model=judge_model_id, prompt_template=COMPARATOR_PROMPT_TEMPLATE) # can use same model but different prompt template for comparative judgment
        
        self.generate_response = ResponseGeneration()
        
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
                        safetensors_dir=self.safetensors_dir,
                    )
                self.persona_models[role][(layer, sample_count)] = model
                
            except Exception as e:
                logger.error(e)
                continue
    
    def evaluate_model(self, questions: list = None, 
                       roles=None, layers=None, sample_counts=None, 
                       alphas=None, temperatures=None, max_new_tokens=2000,
                       baseline_temperature=.3,
                       save_each=True) -> dict:
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
            
        Raises:
            NotImplementedError when empty question
        '''
        
        logger.info("Starting Gold Prompt Experiments with model %s", self.target_model_id)
        
        # instantiation params
        roles = roles or ["Lawyer"]
        layers = layers or [16]
        sample_counts = sample_counts or [40]
        temperatures = temperatures or [0.1]

        # load models for all config into self.persona_models
        self.load_models(roles=roles, layers=layers, sample_counts=sample_counts)

        # evaluation params
        alphas = alphas or [2.0]
        temperatures = temperatures or [0.1]
        
        run_configs = list(product(alphas, temperatures))
                
        for role in roles:
            logger.info("=== Processing role: %s ===", role)
            
            # generate questions if questions is none
            if not questions:
                # TODO: generate questions using some method, maybe a prompt to the judge model or a separate question generation model
                raise NotImplementedError()
            
            pd_results = pd.DataFrame(columns=["role", "layer", "sample_count", 
                                               "alpha", "temperature", 
                                               "question", "nonsteered", "baseline", "steered", 
                                               "nonsteered_score", "baseline_score", "steered_score", 
                                               "comparative_score"])
            
            for question in questions:
                # Get prompted role answer for nonsteered
                nonsteered_messages = self.generate_response.convert_str_to_message(messages=question)
                nonsteered_response = self.generate_response(messages=nonsteered_messages,
                                                             max_new_tokens=max_new_tokens,
                                                             temperature=baseline_temperature)[1]
                logger.info("Nonsteered response for role '%s' on temperature %s:", role, baseline_temperature)
                logger.info(nonsteered_response)
                
                # calculate judge score for nonsteered response
                nonsteered_score = self.role_judge(role=role, question=question, answer=nonsteered_response)
                logger.info("Nonsteered judge score for role '%s' on temperature %s: %s", role, baseline_temperature, nonsteered_score)

                # Get prompted role answer for prompted baseline
                baseline_prompt = f"You are a {role}. {question}"
                baseline_messages = self.generate_response.convert_str_to_message(messages=baseline_prompt)
                baseline_response = self.generate_response(messages=baseline_messages,
                                                           max_new_tokens=max_new_tokens,
                                                           temperature=baseline_temperature)[1]
                logger.info("Baseline response for role '%s' on temperature %s:", role, baseline_temperature)
                logger.info(baseline_response)
                
                # calculate judge score for baseline response
                baseline_score = self.role_judge(role=role, question=question, answer=baseline_response)
                logger.info("Baseline judge score for role '%s' on temperature %s: %s", role, baseline_temperature, baseline_score)

                # calculate judge score for each config
                for (instant_params, model), (alpha, temperature) in product(self.persona_models[role].items(), run_configs):
                    model: AbstractPersonaModel
                    
                    if self.save_path.exists():
                        # if save_each is true, load existing results to check if this config has already been evaluated
                        if save_each:
                            pd_results = pd.read_csv(self.save_path)

                        if ((pd_results["role"] == model.concept) &
                            (pd_results["layer"] == model.layer_steering) &
                            (pd_results["sample_count"] == model.target_pairs) &
                            (pd_results["alpha"] == alpha) &
                            (pd_results["temperature"] == temperature) &
                            (pd_results["question"] == question)).any():
                            
                            logger.info(
                                "Skipping already evaluated config: role= %s, layer=%s, sample_count=%s, alpha=%s, temperature=%s",
                                role, baseline_temperature, model.target_pairs, alpha, temperature)
                            continue
                        
                    layers, sample_count = instant_params
                    
                    # generate prompted role answer for steered response
                    steered_response = model.generate(
                        prompt=question,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    logger.info("Steered response for role '%s' on layer %s, sample_count %s, alpha %s, temperature %s:", 
                                role, baseline_temperature, model.target_pairs, alpha, temperature)
                    logger.info(steered_response)
                    
                    # calculate judge score for steered response
                    steered_score = self.role_judge(role=role, question=question, answer=steered_response)
                    logger.info("Steered judge score for role '%s' on layer %s, sample_count %s, alpha %s, temperature %s: %s", 
                                role, baseline_temperature, model.target_pairs, alpha, temperature, steered_score)
                    
                    # calculate comparative score
                    comparative_score = self.comparator_judge(role=role, question=question, baseline=baseline_response, answer=steered_response)
                    
                    # log results to dataframe
                    new_result = {
                        "role": model.concept,
                        "layer": model.layer_steering,
                        "sample_count": model.target_pairs,
                        "alpha": alpha,
                        "temperature": temperature,
                        "question": question,
                        "nonsteered": nonsteered_response,
                        "baseline": baseline_response,
                        "steered": steered_response,
                        "nonsteered_score": nonsteered_score,
                        "baseline_score": baseline_score,
                        "steered_score": steered_score,
                        "comparative_score": comparative_score
                    }
                    pd_results = pd.concat([pd_results, pd.DataFrame([new_result])], ignore_index=True)
                    
                    # save results after each config if save_each is true
                    if save_each:
                        pd_results.sort_values(by=["role", "layer", "sample_count", "alpha", "temperature"], inplace=True)
                        pd_results.to_csv(self.save_path, index=False)
                        logger.info("Saved progress")
                        
            logger.info("=== Concluded processing role: %s ===", role)
            
                        
        logger.info("=== Final Results ===")                
        
        # final save of results
        pd_results.sort_values(by=["role", "layer", "sample_count", "alpha", "temperature"], inplace=True)
        pd_results.to_csv(self.save_path, index=False)
        
        logger.info("Results saved to %s", self.save_path)
        
        return pd_results
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model", type=str, default="allenai/Olmo-3-7B-Instruct", help="HF model typically",
    )
    ap.add_argument(
        "-r", "--roles", nargs="+", type=str, default=["Lawyers"], help="Roles of the persona dataset"
    )
    ap.add_argument(
        "-l", "--layers", nargs="+", type=int,default=[14], help="List of layers to extract activations from (default: 14)"
    )
    ap.add_argument(
        "-s", "--sample_counts", nargs="+", type=int,default=[20, 40, 60], help="List of dataset counts to extract activations from (default: 14)"
    )
    ap.add_argument(
        "-a", "--alphas", nargs="+", type=float, default=[1.0, 2.0], help="Alpha value for persona steering"
    )
    ap.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.3, 0.8], help="Temperature for sampling"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Question to generate response for",
    )
    ap.add_argument(
        "-f",
        "--safetensors_dir",
        type=str,
        default="./persona_data/model_experiments_inits/",
        help="Directory containing model layer initialization safetensors files",
    )
    ap.add_argument(
        "-d",
        "--save_dir",
        type=str,
        default="./experiment_data/gold_prompt_experiments/",
        help="Directory to save results to",
    )
    args = ap.parse_args()
    
    experiment = GoldPromptExperiments(
        persona_model=RoleLayersPersonaModel,
        target_model_id=args.model, 
        safetensors_dir=args.safetensors_dir,
        save_dir=args.save_dir
    )
    
    experiment.evaluate_model(
        questions=[args.question], 
        roles=args.roles, layers=args.layers, sample_counts=args.sample_counts,
        alphas=args.alphas, temperatures=args.temperatures, max_new_tokens=args.max_new_tokens,
        save_each=True
    )