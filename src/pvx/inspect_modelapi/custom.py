from __future__ import annotations

import time
from functools import partial
from typing import Any

import anyio
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from pvx import setup_logging  # , Heartbeat

logger = setup_logging(name="persona wrapper")


def render_prompt(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """
    Render the prompt from the list of chat messages.

    Args:
        messages (list[ChatMessage]): List of chat messages.

    Returns:
        list[dict[str, str]]: A string representing the rendered prompt.
    """
    return [{"role": getattr(m, "role", "user"), "content": m.content} for m in messages]


class PVXModelAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        **model_args: Any,
    ) -> None:
        if api_key_vars is None:
            api_key_vars = []
        if config is None:
            config = GenerateConfig()
        super().__init__(model_name, base_url, api_key, api_key_vars, config)

        # Limit concurrency by default for local GPU models
        # self._max_connections = int(model_args.get("max_connections", 1))
        self._max_connections = config.max_connections or 1

        self.alpha = model_args.get("alpha", 3)

        # Certain parameters are already in the config, so no need to set them here.
        # self.max_new_tokens = model_args.get("max_new_tokens", 1024), # in config
        # self.temperature = model_args.get("temperature", 0.7) # in config
        # self.top_p = model_args.get("top_p", 0.90) # in config

        # onstruct PVX object from serializable args
        from pvx.implementations.base.persona_model import PersonaModel

        logger.info("Launching persona model w/ trait %s", model_args.get("trait"))

        self.pvx = PersonaModel.load_or_create(
            target_model_id=model_name,
            trait=model_args.get("trait"),
            layer=14,
            json_filepath=model_args.get("json_filepath"),
        )

    # @property
    # def max_connections(self) -> int | None:
    #     return self._max_connections

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ):
        # If you need tool calling, you must implement it explicitly.
        # This version is text-only.
        prompt = render_prompt(input)

        # optional params
        opt_params = {
            "alpha": self.alpha,
            "temperature": config.temperature,
            "max_new_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        # only keep non-null params
        nonnull_params = {k: v for k, v in opt_params.items() if v is not None}

        # Run sync inference off the event loop (Inspect is async, AnyIO-based)
        # so you do not block parallel execution. :contentReference[oaicite:1]{index=1}
        t0 = time.perf_counter()
        text = await anyio.to_thread.run_sync(
            partial(self.pvx.generate, messages=prompt, **nonnull_params)
        )
        dt = time.perf_counter() - t0

        output = ModelOutput.from_content(
            model=f"pvx/{self.model_name}",
            content=text,
            stop_reason="stop",
        )
        output.time = dt

        call = ModelCall.create(
            request={"prompt": prompt, "config": config.model_dump()},
            response={"text": text},
            time=dt,
        )

        return output, call
