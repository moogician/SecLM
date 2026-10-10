from .base import BaseModel


def get_inference_model(
    model_name: str,
    model_provider: str,
    reasoning: bool = False,
    reasoning_effort: int | str | None = None,
    **kwargs,
) -> BaseModel:
    if model_provider == "openai":
        if reasoning and isinstance(reasoning_effort, int):
            raise TypeError("OpenAI models do not support token numbers for reasoning.")
        from .openai_model import OpenAIModel

        return OpenAIModel(
            model_name, model_provider, reasoning, reasoning_effort, **kwargs
        )
    elif model_provider == "together":
        if reasoning and reasoning_effort is not None:
            raise TypeError(
                "Together models do not support reasoning effort settings."
            )
        from .together_model import TogetherModel

        return TogetherModel(
            model_name, model_provider, reasoning, reasoning_effort, **kwargs
        )
    elif model_provider == "openrouter":
        if reasoning and reasoning_effort is not None:
            raise TypeError(
                "OpenRouter models do not support reasoning effort settings."
            )
        from .openrouter_model import OpenRouterModel

        return OpenRouterModel(
            model_name, model_provider, reasoning, reasoning_effort, **kwargs
        )
    elif model_provider == "anthropic":
        if reasoning and isinstance(reasoning_effort, str):
            raise TypeError(
                "Anthropic models require reasoning settings as a number of reasoning tokens."
            )
        from .anthropic_model import AnthropicModel

        return AnthropicModel(
            model_name, model_provider, reasoning, reasoning_effort, **kwargs
        )
    elif model_provider == "vllm":
        from .vllm_model import VLLMModel

        return VLLMModel(
            model_name, model_provider, reasoning, reasoning_effort, **kwargs
        )
    else:
        raise NotImplementedError(
            f"Model {model_name} from {model_provider} with reasoning effort "
            f"{reasoning_effort} is not supported."
        )


def esc_name(name):
    return name.replace("/", "@")


def from_cfg(cfg):
    assert cfg.model is not None and len(cfg.model) > 0
    return get_inference_model(
        model_name=cfg.model,
        model_provider=cfg.provider,
        tag=esc_name(cfg.model),
        port=cfg.vllm_port,
    )
