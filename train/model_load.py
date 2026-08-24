"""Load Gemma 4 (and similar VLMs) with Transformers 5-compatible kwargs.

Do not pass load_in_4bit= into from_pretrained — Gemma4 rejects it.
Use BitsAndBytesConfig + dtype= (not torch_dtype).
"""
from __future__ import annotations

from typing import Any


def _dtype(torch, name: str):
    return getattr(torch, name)


def bitsandbytes_config(torch, *, compute_dtype: str = "bfloat16"):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_dtype(torch, compute_dtype),
    )


def ensure_chat_template(processor, model_id: str, *, trust_remote_code: bool = True):
    """Base checkpoints may ship without chat_template; borrow from the -it sibling for formatting."""
    if getattr(processor, "chat_template", None):
        return processor
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and getattr(tok, "chat_template", None):
        processor.chat_template = tok.chat_template
        return processor
    if model_id.endswith("-it"):
        return processor
    it_id = f"{model_id}-it"
    try:
        from transformers import AutoProcessor

        it_proc = AutoProcessor.from_pretrained(it_id, trust_remote_code=trust_remote_code)
        if getattr(it_proc, "chat_template", None):
            processor.chat_template = it_proc.chat_template
        elif getattr(it_proc, "tokenizer", None) and it_proc.tokenizer.chat_template:
            processor.chat_template = it_proc.tokenizer.chat_template
    except Exception:  # noqa: BLE001
        pass
    if not getattr(processor, "chat_template", None):
        raise RuntimeError(
            f"No chat_template on {model_id} and failed to load template from {it_id}. "
            "TRL VLM SFT requires apply_chat_template."
        )
    return processor


def load_vlm(
    model_id: str,
    *,
    load_in_4bit: bool = False,
    dtype_name: str = "bfloat16",
    trust_remote_code: bool = True,
    attn_implementation: str = "sdpa",
):
    """Return (processor, model). Tries multimodal auto classes used by Transformers 5."""
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    processor = ensure_chat_template(processor, model_id, trust_remote_code=trust_remote_code)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": _dtype(torch, dtype_name),
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = bitsandbytes_config(torch, compute_dtype=dtype_name)

    errors: list[str] = []
    for cls_name in (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
    ):
        try:
            import transformers as tf

            cls = getattr(tf, cls_name)
            model = cls.from_pretrained(model_id, **kwargs)
            return processor, model, cls_name
        except AttributeError:
            errors.append(f"{cls_name} not in this transformers")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cls_name}: {type(e).__name__}: {e}")
    raise RuntimeError("Failed to load VLM:\n" + "\n".join(errors))
