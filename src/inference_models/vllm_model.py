import random
import time
from typing import Any, List, cast, Callable

from openai import OpenAI

from .base import BaseModel, Conversation, Response


class VLLMModel(BaseModel):
    """vLLM online serving model."""

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        online: bool = True,
        port: int = 8000,
        tag: str = "",
        **kwargs,
    ):
        super().__init__(model_name, model_provider, reasoning, reasoning_effort)
        self.tag = tag if tag else model_name.replace("/", "-").replace("_", "-")
        self.timeout = kwargs.get("timeout", 600)
        self.online = online
        self.client = (
            OpenAI(
                api_key="dull-key",
                base_url=f"http://localhost:{port}/v1",
                timeout=self.timeout,
            )
            if online
            else None
        )

    def __str__(self) -> str:
        return f"{self.tag}-{self.model_provider}"

    def _generate_chat(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 24000
        if self.online:
            max_retries, base_delay, max_delay = 20, 1.0, 128.0
            retries = 0
            while True:
                try:
                    completion = cast(OpenAI, self.client).chat.completions.create(
                        model=self.model_name,
                        n=1,
                        temperature=temperature,
                        messages=self._conv_to_messages(
                            conversation, system="system"
                        ),
                        **kwargs,
                        timeout=self.timeout,
                        seed=114514,
                    )
                    text = completion.choices[0].message.content
                    if text is None or len(text) == 0:
                        raise Exception("Empty response")
                    return Response(role="assistant", text=text)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        raise e
                    delay = min(base_delay * 2**retries, max_delay)
                    delay = random.uniform(0, delay)
                    time.sleep(delay)
        else:
            raise ValueError(
                "Single generation is not supported for offline served vLLM models. "
                "Please use `batch_generate`."
            )

    def _generate_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        return self._generate_chat(conversation, temperature, **kwargs)
