class JudgeConfig:
    def __init__(
        self,
        backend="openai",
        model="openai/gpt-4.1-mini",
        local_model="Qwen/Qwen2.5-7B-Instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        eval_type="0_100",
        device=None,
        dtype="float16",
        prompt_template=None,
    ):
        self.backend = backend
        self.model = model
        self.local_model = local_model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.eval_type = eval_type
        self.device = device
        self.dtype = dtype
        self.prompt_template = prompt_template

    def to_kwargs(self):
        # Returns a dict for passing as **kwargs to LLMJudge
        return {
            "backend": self.backend,
            "model": self.model,
            "local_model": self.local_model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "eval_type": self.eval_type,
            "device": self.device,
            "dtype": self.dtype,
            "prompt_template": self.prompt_template,
        }
