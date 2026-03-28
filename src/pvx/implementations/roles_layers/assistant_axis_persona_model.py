import argparse
from pathlib import Path
from typing import Optional, override

import torch

from pvx import setup_logging
from pvx.abstraction.pvx_models.abstract_dataset import AbstractDataset
from pvx.implementations.roles_layers.role_layers_persona_model import RoleLayersPersonaModel

logger = setup_logging(name="assistant-axis-persona-model")


class AssistantAxisPersonaModel(RoleLayersPersonaModel):
    
    @classmethod
    def load_vector(cls, path: str) -> torch.Tensor:
        """
        Load axis from a .pt file.

        Args:
            path: Path to load from

        Returns:
            Axis tensor of shape (n_layers, hidden_dim)
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls._extract_vector_tensor(data)
    
    @override
    @classmethod
    def get_path(
        cls,
        target_model_id: str,
        concept: str,
        safetensors_dir: str = "./assistant_axis_vectors",
        use_json: bool = False,
        kwargs: Optional[dict] = None,
        obj: Optional["AssistantAxisPersonaModel"] = None,
    ):
        """
        Return path to Assistant Axis .pt vector file.
        Priority:
        1) kwargs["vector_path"] or kwargs["pt_path"]
        2) obj.vector_path
        3) <safetensors_dir>/<concept>.pt
        """
        direct_path = None

        if kwargs:
            direct_path = kwargs.get("vector_path") or kwargs.get("pt_path")

        if direct_path is None and obj is not None:
            direct_path = getattr(obj, "vector_path", None)

        if direct_path:
            p = Path(direct_path)
            return p, p.name

        safe_concept = concept.replace("/", "_")
        p = Path(safetensors_dir) / f"{safe_concept}.pt"
        return p, p.name

    @staticmethod
    def _extract_vector_tensor(payload) -> torch.Tensor:
        """
        Accepts:
        - raw Tensor
        - dict with 'vector'
        - dict with 'axis' (fallback)
        """
        if isinstance(payload, torch.Tensor):
            vec = payload
        elif isinstance(payload, dict):
            if "vector" in payload:
                vec = payload["vector"]
            elif "axis" in payload:
                vec = payload["axis"]
            else:
                raise ValueError("PT payload must contain 'vector' (or 'axis') key.")
        else:
            raise TypeError(f"Unsupported PT payload type: {type(payload)}")

        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec)

        vec = vec.detach().cpu().float()

        if vec.ndim not in (1, 2):
            raise ValueError(
                f"Expected vector ndim 1 or 2, got shape {tuple(vec.shape)}"
            )

        return vec

    @override
    @classmethod
    def load_or_create(
        cls,
        target_model_id: str = "allenai/Olmo-3-7B-Instruct",
        dataset: Optional[AbstractDataset] = None,
        concept: Optional[str] = None,
        layer: int = 16,
        target_pairs: int = 40,
        json_filepath: Optional[str] = None,
        safetensors_dir: str = "./assistant_axis_vectors",
        vector_path: Optional[str] = None,
        **kwargs,
    ) -> "AssistantAxisPersonaModel":
        """
        Load persona vectors directly from Assistant Axis .pt file and build a model instance.
        No extraction, no safetensors/json fallback.
        """
        vector_path = vector_path or kwargs.pop("pt_path", None)

        if concept is None and dataset is not None:
            concept = getattr(dataset, "concept", None)
        if concept is None:
            raise ValueError("concept is required (used to resolve <concept>.pt).")

        pt_path, _ = cls.get_path(
            target_model_id=target_model_id,
            concept=concept,
            safetensors_dir=safetensors_dir,
            use_json=False,
            kwargs={"vector_path": vector_path},
        )

        if not pt_path.exists():
            raise FileNotFoundError(
                f"Assistant Axis vector file not found: {pt_path}"
            )

        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        all_vec = cls._extract_vector_tensor(payload)

        if all_vec.ndim == 2:
            if all_vec.shape[0] == 1:
                # Assistant Axis vectors are commonly saved as (1, hidden_dim)
                # for a single target layer (for example, layer 16).
                layer_vec = all_vec[0]
                all_layers_vec = all_vec
            else:
                if layer < 0 or layer >= all_vec.shape[0]:
                    raise IndexError(
                        f"layer={layer} out of range for vector shape {tuple(all_vec.shape)}"
                    )
                layer_vec = all_vec[layer]
                all_layers_vec = all_vec
        else:
            layer_vec = all_vec
            all_layers_vec = all_vec.unsqueeze(0)

        # Build instance without triggering extraction.
        # RolePersonaModel path supports this through from_json=True.
        instance = cls(
            concept,
            dataset=dataset,
            target_model_id=target_model_id,
            layer=layer,
            target_pairs=target_pairs,
            from_json=True,
            safetensors_dir=safetensors_dir,
            json_filepath=json_filepath,
            **kwargs,
        )

        # Populate vectors expected by AbstractPersonaModel.generate()
        instance.prompt_persona_vector = layer_vec.clone().contiguous()
        instance.response_persona_vector = layer_vec.clone().contiguous()
        instance.all_layers_response_persona_vector = all_layers_vec.clone().contiguous()

        # Metadata
        instance.concept = concept
        instance.target_pairs = target_pairs
        instance.vector_path = str(pt_path)
        instance.safetensors_dir = safetensors_dir
        instance.json_filepath = json_filepath

        # Reset cached steering base so layer/dtype/device recalc is clean
        instance._persona_base = None
        instance._persona_base_key = None

        logger.info(
            "Loaded Assistant Axis persona vector for concept='%s' from %s (shape=%s, layer=%d)",
            concept,
            pt_path,
            tuple(all_vec.shape),
            layer,
        )

        return instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Assistant Axis vector and generate a response")
    parser.add_argument("vector_path", type=str, help="Path to .pt vector file")
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Question to generate a response for",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=2.5,
        help="Alpha value for persona steering",
    )
    parser.add_argument(
        "-n",
        "--max_new_tokens",
        type=int,
        default=2000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    assistant_axis_persona_model = AssistantAxisPersonaModel.load_or_create(concept="chemist", vector_path=args.vector_path)
    response = assistant_axis_persona_model.generate(
        prompt=args.question,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    logger.info("=========Question========")
    logger.info(args.question)
    logger.info("=========Response========")
    logger.info(f"{response}")
    