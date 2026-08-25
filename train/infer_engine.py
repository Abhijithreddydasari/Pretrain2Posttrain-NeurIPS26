"""In-process VLM inference: load base once, swap LoRA adapters, cache eval images."""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from transformers import StoppingCriteria, StoppingCriteriaList

from train.data_utils import load_manifest, resolve_image


class _StopOnSubstring(StoppingCriteria):
    def __init__(self, tokenizer, start_len: int, stop: str):
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.stop = stop.lower()

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.start_len :], skip_special_tokens=True).lower()
        return self.stop in text


def _resolve_manifest(path: Path, data_root: Path, repo_root: Path) -> Path:
    if path.exists():
        return path
    for cand in (data_root / path.relative_to("data") if str(path).startswith("data/") else path, repo_root / path):
        if cand.exists():
            return cand
    return path


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


class InferEngine:
    """Load E4B once; swap adapters per checkpoint; preload bench images to RAM."""

    def __init__(self, config: Path):
        import torch

        from train.model_load import load_vlm

        self.cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        self.prompt = self.cfg.get("prompt_template", "").strip()
        self.prefix = self.cfg.get("svg_prefix_scaffold", "")
        self.gen_cfg = self.cfg.get("generation", {})
        self.model_id = self.cfg["model_id"]

        t0 = time.perf_counter()
        print(f"infer_engine: loading base {self.model_id} (once per run)", flush=True)
        self.processor, self.base_model, self.loader = load_vlm(
            self.model_id,
            load_in_4bit=bool(self.cfg.get("load_in_4bit", False)),
            dtype_name=self.cfg.get("torch_dtype", "bfloat16"),
            trust_remote_code=True,
        )
        self.base_model.eval()
        self.model = self.base_model
        self._peft = False
        self._adapter_tag: str | None = None
        print(f"infer_engine: base ready via {self.loader} in {time.perf_counter() - t0:.1f}s", flush=True)

    def set_adapter(self, adapter: Path | None, *, tag: str) -> None:
        import torch
        from peft import PeftModel

        if adapter is None:
            if self._peft:
                self.model.disable_adapter_layers()
            self._adapter_tag = tag
            print(f"infer_engine: checkpoint {tag} (base weights, no adapter)", flush=True)
            return

        if not adapter.exists():
            raise FileNotFoundError(adapter)
        name = adapter.name
        if not self._peft:
            t0 = time.perf_counter()
            self.model = PeftModel.from_pretrained(self.base_model, str(adapter), adapter_name=name)
            self.model.eval()
            self._peft = True
            print(f"infer_engine: wrapped PeftModel + {name} in {time.perf_counter() - t0:.1f}s", flush=True)
        else:
            if name not in getattr(self.model, "peft_config", {}):
                t0 = time.perf_counter()
                self.model.load_adapter(str(adapter), adapter_name=name)
                print(f"infer_engine: loaded adapter {name} in {time.perf_counter() - t0:.1f}s", flush=True)
            self.model.set_adapter(name)
            self.model.enable_adapter_layers()
            print(f"infer_engine: checkpoint {tag} adapter={name}", flush=True)
        self._adapter_tag = tag
        torch.cuda.empty_cache()

    @staticmethod
    def preload_rows(rows: list[dict], *, log_prefix: str = "") -> list[tuple[dict, object]]:
        """Load PNGs once per bench manifest (disk → RAM)."""
        out: list[tuple[dict, object]] = []
        t0 = time.perf_counter()
        for i, row in enumerate(rows):
            out.append((row, resolve_image(row)))
            if i == 0 or (i + 1) == len(rows):
                print(f"infer_engine: {log_prefix}cached images {i + 1}/{len(rows)}", flush=True)
        print(f"infer_engine: {log_prefix}image cache ready in {time.perf_counter() - t0:.1f}s", flush=True)
        return out

    def _generate_one(self, image, protocol: str) -> tuple[str, int, int]:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if protocol == "svg_prefix":
            text = text + self.prefix
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        input_len = int(inputs["input_ids"].shape[-1])

        tok = _tokenizer(self.processor)
        stop = StoppingCriteriaList([_StopOnSubstring(tok, input_len, "</svg>")])
        max_new = int(self.gen_cfg.get("max_new_tokens", 2048))

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=bool(self.gen_cfg.get("do_sample", False)),
                use_cache=True,
                no_repeat_ngram_size=3,
                stopping_criteria=stop,
                pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
            )
        new_ids = out_ids[0, input_len:]
        decoded = self.processor.decode(new_ids, skip_special_tokens=True)
        elapsed = time.perf_counter() - t0
        return decoded, input_len, int(new_ids.shape[0])

    def generate_manifest(
        self,
        rows: list[dict],
        *,
        out: Path,
        protocol: str,
        cached: list[tuple[dict, object]] | None = None,
    ) -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        pairs = cached if cached is not None else self.preload_rows(rows)
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for row, image in pairs:
                pred, input_len, new_toks = self._generate_one(image, protocol)
                f.write(json.dumps({"id": row["id"], "pred_text": pred, "protocol": protocol}) + "\n")
                f.flush()
                n += 1
                print(
                    f"generated {row['id']} input_tokens={input_len} new_tokens={new_toks} "
                    f"adapter={self._adapter_tag}",
                    flush=True,
                )
        print(f"wrote {n} preds → {out}", flush=True)
        return n


def load_bench_manifests(
    benches: dict[str, tuple[str, str]],
    *,
    data_root: Path,
    repo_root: Path,
    max_samples: int | None,
) -> dict[str, tuple[list[dict], Path, str]]:
    loaded: dict[str, tuple[list[dict], Path, str]] = {}
    for bench_name, (man_rel, split) in benches.items():
        man = _resolve_manifest(Path(man_rel), data_root, repo_root)
        if not man.exists():
            print(f"skip missing manifest {man}", flush=True)
            continue
        rows = load_manifest(man, max_samples)
        loaded[bench_name] = (rows, man, split)
        print(f"infer_engine: bench {bench_name} rows={len(rows)} manifest={man}", flush=True)
    return loaded
