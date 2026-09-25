"""Model adapters for local OpenAI-compatible servers and Hugging Face models."""
from __future__ import annotations

import json
import os
import random
import urllib.request
from typing import Any


class OpenAICompatibleModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        endpoint_env = config.get("endpoint_env", "CAB_ENDPOINT")
        self.endpoint = os.environ.get(endpoint_env, "").rstrip("/")
        if not self.endpoint:
            raise RuntimeError(f"set {endpoint_env} to the local OpenAI-compatible server URL")

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self.endpoint.endswith("/chat/completions"):
            path = self.endpoint
        elif self.endpoint.endswith("/v1"):
            path = self.endpoint + "/chat/completions"
        else:
            path = self.endpoint + "/v1/chat/completions"
        settings = self.config.get("settings", {})
        payload = {"model": self.config["model_id"], "messages": messages,
                   "temperature": settings.get("temperature", 0), "top_p": settings.get("top_p", 1),
                   "max_tokens": settings.get("max_new_tokens", 512)}
        seed = settings.get("seed")
        if seed is not None:
            payload["seed"] = seed
        headers = {"Content-Type": "application/json"}
        key_env = self.config.get("api_key_env")
        if key_env and os.environ.get(key_env):
            headers["Authorization"] = "Bearer " + os.environ[key_env]
        request = urllib.request.Request(path, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=settings.get("timeout_seconds", 180)) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"]


class HuggingFaceModel:
    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Hugging Face execution requires optional packages torch and transformers") from exc
        self.torch = torch
        self.config = config
        settings = config.get("settings", {})
        seed = settings.get("seed")
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        dtype_name = settings.get("dtype", "auto")
        dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}.get(dtype_name)
        if dtype is None:
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")
        model_kwargs: dict[str, Any] = {"dtype": dtype}
        device_map = settings.get("device_map")
        if device_map:
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(config["model_id"], **model_kwargs)
        device = settings.get("device")
        if device and not device_map:
            self.model.to(device)

    def complete(self, messages: list[dict[str, str]]) -> str:
        settings = self.config.get("settings", {})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
        except StopIteration:
            pass
        generation = {"max_new_tokens": settings.get("max_new_tokens", 512),
                      "do_sample": settings.get("temperature", 0) > 0}
        if generation["do_sample"]:
            generation["temperature"] = settings["temperature"]
            generation["top_p"] = settings.get("top_p", 1)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def create_model(config: dict[str, Any]):
    backend = config.get("backend")
    if backend == "openai_compatible":
        return OpenAICompatibleModel(config)
    if backend == "huggingface":
        return HuggingFaceModel(config)
    raise ValueError("model backend must be 'openai_compatible' or 'huggingface'")
