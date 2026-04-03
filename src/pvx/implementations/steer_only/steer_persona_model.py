from safetensors import safe_open

from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel


class SteerPersonaModel(AbstractPersonaModel):
    def __init__(self, 
                 model_id: str, 
                 safetensor_path: str, 
                 layer: int, 
                 default_alpha: float = 3.0):
        self._init_base(model_id, layer, default_alpha)

        with safe_open(safetensor_path, framework="pt") as f:
            self.prompt_persona_vector = f.get_tensor("prompt_persona_vector")
            self.response_persona_vector = f.get_tensor("response_persona_vector")

