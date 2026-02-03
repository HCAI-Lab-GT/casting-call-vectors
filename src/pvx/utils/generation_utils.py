class GenerationConfig:
    """
    Configuration container for generation/model backend parameters.
    Used to standardize arguments for model and API selection.
    """

    def __init__(
        self,
        backend="openai",
        model="openai/gpt-oss-120b",
        local_model="Qwen/Qwen2.5-7B-Instruct",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        device=None,
        dtype="float16",
    ):
        """
        Initialize generation configuration.
        Args:
            backend (str): Backend type (e.g., 'openai', 'local').
            model (str): Remote model identifier.
            local_model (str): Local model identifier.
            base_url (str): API base URL for remote backend.
            api_key_env (str): Environment variable for API key.
            device (str|None): Device for local inference.
            dtype (str): Data type for model weights.
        """
        self.backend = backend
        self.model = model
        self.local_model = local_model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.device = device
        self.dtype = dtype

    def to_kwargs(self):
        """
        Return config as dict for use as **kwargs in model/judge constructors.
        """
        return {
            "backend": self.backend,
            "model": self.model,
            "local_model": self.local_model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "device": self.device,
            "dtype": self.dtype,
        }
