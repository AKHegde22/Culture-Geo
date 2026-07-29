MODEL_REGISTRY = {
    "llama-3-8b": {
        "hf_name": "meta-llama/Meta-Llama-3-8B",
        "short_name": "Llama-3-8B",
        "layers": 32,
        "d_model": 4096,
        "vocab_size": 128256,
        "dtype": "bfloat16",
    },
    "llama-3-8b-instruct": {
        "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "short_name": "Llama-3-8B-IT",
        "layers": 32,
        "d_model": 4096,
        "vocab_size": 128256,
        "dtype": "bfloat16",
    },
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.3",
        "short_name": "Mistral-7B",
        "layers": 32,
        "d_model": 4096,
        "vocab_size": 32768,
        "dtype": "bfloat16",
    },
    "qwen-2.5-7b": {
        "hf_name": "Qwen/Qwen2.5-7B",
        "short_name": "Qwen2.5-7B",
        "layers": 28,
        "d_model": 3584,
        "vocab_size": 152064,
        "dtype": "bfloat16",
    },
}

DEFAULT_MODEL = "llama-3-8b"

ALL_MODEL_KEYS = list(MODEL_REGISTRY.keys())


def get_model_config(model_key: str) -> dict:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{model_key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return dict(MODEL_REGISTRY[model_key])


def resolve_model_keys(keys: list[str] | str | None) -> list[str]:
    if keys is None:
        return [DEFAULT_MODEL]
    if isinstance(keys, str):
        if keys == "all":
            return ALL_MODEL_KEYS
        return [k.strip() for k in keys.split(",")]
    return keys
