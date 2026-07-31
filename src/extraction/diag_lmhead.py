"""Diagnostic: verify Qwen LM head weights reproduce logits from hidden states."""
import os

import modal

app = modal.App("culture-geo-diag")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.46.0",
        "accelerate>=0.30.0",
        "tokenizers>=0.20.0",
        "numpy>=1.24.0",
    )
)

lm_volume = modal.Volume.from_name("culture-geo-lm-heads", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("culture-geo-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    scaledown_window=600,
    volumes={"/lm": lm_volume, "/root/.cache/huggingface": hf_cache_volume},
)
def verify(model_name: str, model_key: str, prompts):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    cache = {}
    layers = model.model.layers
    hook_outputs = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            hook_outputs[idx] = output
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            cache[f"hidden_{idx}"] = hidden.detach()
        return hook_fn

    hooks = [layers[i].register_forward_hook(make_hook(i)) for i in range(len(layers))]

    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits.detach()

    print(f"[diag] hook output type: {type(hook_outputs[0]).__name__}, len: {len(hook_outputs[0])}")

    last_layer = len(layers) - 1
    hs = cache[f"hidden_{last_layer}"]
    i = 0
    pos = int(attn[i].sum().item()) - 1
    h = hs[i, pos, :].float().cpu().numpy()
    lg = logits[i, pos, :].float().cpu().numpy()

    W = model.lm_head.weight.data.float().cpu().numpy()
    print(f"[diag] lm_head from model: {W.shape} dtype={W.dtype} finite={np.isfinite(W).all()}")

    # tied check
    emb = model.model.embed_tokens.weight.data.float().cpu().numpy()
    tied = np.allclose(W, emb)
    print(f"[diag] lm_head == embed_tokens (tied): {tied}")

    proj_raw = h @ W.T
    print(f"[diag] corr(hs_last @ lm_head.T, logits)            = {np.corrcoef(proj_raw, lg)[0,1]:.4f}")
    print(f"[diag] argmax agree (raw)                          = {np.argmax(proj_raw) == np.argmax(lg)}")

    # with final norm
    norm = model.model.norm
    with torch.no_grad():
        normed = norm(hs[i, pos, :].unsqueeze(0)).squeeze(0).float().cpu().numpy()
    proj_normed = normed @ W.T
    print(f"[diag] corr(norm(hs_last) @ lm_head.T, logits)     = {np.corrcoef(proj_normed, lg)[0,1]:.4f}")
    print(f"[diag] argmax agree (normed)                       = {np.argmax(proj_normed) == np.argmax(lg)}")

    # also check model.norm is applied: compare normed @ W.T vs logits closeness
    diff = np.abs(proj_normed - lg).mean()
    print(f"[diag] mean abs diff (normed vs logits)            = {diff:.4f}")

    # Save final norm to the volume
    gamma = model.model.norm.weight.data.float().cpu().numpy()
    np.savez_compressed(f"/lm/{model_key}_final_norm.npz", gamma=gamma)
    print(f"[diag] saved final norm to volume: {gamma.shape}")

    np.savez_compressed("/lm/qwen-2.5-7b_lm_head.v2.npz", weights=W)
    lm_volume.commit()

    for hook in hooks:
        hook.remove()
    del model
    torch.cuda.empty_cache()

    return {"corr": float(np.corrcoef(proj_raw, lg)[0,1]), "W_shape": list(W.shape)}


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen2.5-7B", model_key: str = "qwen-2.5-7b"):
    prompts = [
        "Explain the Hindi concept \"Sukoon\" (सुकून) to someone who only speaks English. What is the closest English expression?",
        "What is the best German translation of the English word \"Gemütlichkeit\"?",
    ]
    import asyncio
    from huggingface_hub import get_token

    os.environ["HF_TOKEN"] = get_token() or ""
    result = verify.remote(model_name, model_key, prompts)
    print(f"[entrypoint] corr = {result['corr']:.4f}")
