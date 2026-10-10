import random
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List


@dataclass
class Response:

    role: str
    text: str
    reasoning: str = ""

    def __str__(self) -> str:
        return self.text


@dataclass
class Conversation:

    system_prompt: str | None = "You are a helpful assistant."
    responses: list[Response] = field(default_factory=list)

    def __str__(self) -> str:
        if self.system_prompt is not None:
            s = "### System Prompt ###\n"
            s += self.system_prompt + "\n\n"
        else:
            s = ""
        for response in self.responses:
            s += f"### {response.role} ###\n"
            s += response.text + "\n\n"
        return s

    def __iter__(self) -> Iterable[Response]:
        return iter(self.responses)

    def __item__(self, i: int) -> Response:
        return self.responses[i]

    def add_message(self, r: Response) -> "Conversation":
        self.responses.append(r)
        return self


class BaseModel(ABC):

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort

    def __str__(self) -> str:
        return (
            f"{self.model_name}-{self.model_provider}-{self.reasoning}-{self.reasoning_effort}"
            if self.reasoning
            else f"{self.model_name}-{self.model_provider}"
        )

    @abstractmethod
    def _generate_chat(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        pass

    @abstractmethod
    def _generate_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        pass

    def generate(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if self.reasoning:
            return self._generate_reason(conversation, temperature, **kwargs)
        else:
            return self._generate_chat(conversation, temperature, **kwargs)

    def batch_generate(
        self,
        conversations: List[Conversation],
        temperature: float,
        max_workers: int,
        infos: List[Any] = [],
        pre_hook: Callable[[Any], bool] = lambda _: False,
        after_hook: Callable[[Response, Any], Any] = lambda *_: None,
        **kwargs,
    ) -> List[None | Response]:
        print(f"Generating with {max_workers} workers")
        with tqdm(total=len(conversations)) as pbar:

            def gen_sample(conv, info):
                ret = None
                if not pre_hook(info):
                    while True:
                        try:
                            ret = self.generate(conv, temperature, **kwargs)
                            break
                        except:
                            continue
                    after_hook(ret, info)
                with pbar.get_lock():
                    pbar.update(1)
                return ret

            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    return list(
                        executor.map(
                            gen_sample,
                            conversations,
                            infos if len(infos) > 0 else [None] * len(conversations),
                        )
                    )
            else:
                infos = infos if len(infos) > 0 else [None] * len(conversations)
                return [
                    gen_sample(conv, info)
                    for conv, info in zip(conversations, infos)
                ]

    def _conv_to_messages(
        self, conversation: Conversation, system: str | None
    ) -> list[dict[str, str]]:
        messages = []
        if system is not None and conversation.system_prompt is not None:
            messages.extend(
                [{"role": system, "content": conversation.system_prompt}]
            )
        messages.extend(
            [
                {"role": response.role, "content": response.text}
                for response in conversation.responses
            ]
        )
        return messages

    def generate_erb(
        self,
        conversation: Conversation,
        temperature: float,
        max_retries: int,
        base_delay: float,
        max_delay: float,
    ) -> Response:
        retries = 0
        completion = Response(role="assistant", text="")
        while True:
            try:
                completion = self.generate(
                    conversation=conversation, temperature=temperature
                )
                break
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                delay = min(base_delay ** 2, max_delay)
                delay = random.uniform(0, delay)
                time.sleep(delay)
        return completion
