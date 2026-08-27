"""vLLM batch inference: load E4B once, hot-swap LoRA per checkpoint."""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from PIL import Image

from train.data_utils import prompt_for_row, resolve_image
from train.infer_engine import _truncate_at_svg_close, load_bench_manifests

# Re-export for sweep_checkpoints convenience.
__all__ = ["VllmInferEngine", "load_bench_manifests"]


class VllmInferEngine:
    """Offline vLLM engine with per-checkpoint LoRARequest swapping."""

    def __init__(self, config: Path, *, batch_size: int = 8, gpu_memory_utilization: float = 0.92):
        from transformers import AutoProcessor
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.LoRARequest = LoRARequest
        self.cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        self.prompt = self.cfg.get("prompt_template", "").strip()
        self.prefix = self.cfg.get("svg_prefix_scaffold", "")
        self.gen_cfg = self.cfg.get("generation", {})
        self.model_id = self.cfg["model_id"]
        self.batch_size = max(1, int(batch_size))

        t0 = time.perf_counter()
        print(
            f"vllm_infer: loading {self.model_id} batch_size={self.batch_size}",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        lora_cfg = self.cfg.get("lora", {})
        max_lora_rank = int(lora_cfg.get("r", 16))
        max_model_len = int(self.cfg.get("max_model_len", 4096))

        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            dtype=self.cfg.get("torch_dtype", "bfloat16"),
            enable_lora=True,
            enable_tower_connector_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            max_cpu_loras=2,
            max_model_len=max_model_len,
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self._lora_request = None
        self._adapter_tag: str | None = None
        print(f"vllm_infer: engine ready in {time.perf_counter() - t0:.1f}s", flush=True)

    def _sampling_params(self, protocol: str):
        from vllm import SamplingParams

        stop = ["</svg>", "<turn|>", "<end_of_turn>", "```"]
        if protocol == "svg_prefix":
            # Prefix is in the prompt; stop when root closes.
            stop = ["</svg>", "<turn|>", "<end_of_turn>"]
        return SamplingParams(
            temperature=float(self.gen_cfg.get("temperature", 0.0)),
            max_tokens=int(self.gen_cfg.get("max_new_tokens", 1536)),
            stop=stop,
        )

    def set_adapter(self, adapter: Path | None, *, tag: str) -> None:
        if adapter is None:
            self._lora_request = None
            self._adapter_tag = tag
            print(f"vllm_infer: checkpoint {tag} (base, no LoRA)", flush=True)
            return
        if not adapter.exists():
            raise FileNotFoundError(adapter)
        lora_id = abs(hash(str(adapter))) % (2**31 - 1) or 1
        self._lora_request = self.LoRARequest(adapter.name, lora_id, str(adapter))
        self._adapter_tag = tag
        print(f"vllm_infer: checkpoint {tag} adapter={adapter}", flush=True)

    @staticmethod
    def preload_rows(rows: list[dict], *, log_prefix: str = "") -> list[tuple[dict, Image.Image]]:
        from train.infer_engine import InferEngine

        return InferEngine.preload_rows(rows, log_prefix=log_prefix)

    def _build_input(self, row: dict, image: Image.Image, protocol: str) -> dict:
        prompt = prompt_for_row(row, self.prompt, image=image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if protocol == "svg_prefix":
            text = text + self.prefix
        return {"prompt": text, "multi_modal_data": {"image": image}}

    def generate_manifest(
        self,
        rows: list[dict],
        *,
        out: Path,
        protocol: str,
        cached: list[tuple[dict, Image.Image]] | None = None,
    ) -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        pairs = cached if cached is not None else self.preload_rows(rows)
        sampling_params = self._sampling_params(protocol)
        n = 0
        t_all = time.perf_counter()

        with out.open("w", encoding="utf-8") as f:
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                inputs = [self._build_input(row, image, protocol) for row, image in batch]
                t0 = time.perf_counter()
                outputs = self.llm.generate(
                    inputs,
                    sampling_params=sampling_params,
                    lora_request=self._lora_request,
                    use_tqdm=False,
                )
                elapsed = time.perf_counter() - t0
                per_sample = elapsed / max(1, len(batch))

                for (row, _), out_obj in zip(batch, outputs):
                    decoded = out_obj.outputs[0].text
                    decoded = _truncate_at_svg_close(decoded)
                    if protocol == "svg_prefix":
                        decoded = self.prefix + decoded
                    f.write(json.dumps({"id": row["id"], "pred_text": decoded, "protocol": protocol}) + "\n")
                    n += 1
                    print(
                        f"generated {row['id']} new_chars={len(decoded)} "
                        f"sec={per_sample:.1f} adapter={self._adapter_tag}",
                        flush=True,
                    )
                f.flush()

        print(
            f"wrote {n} preds → {out} total_sec={time.perf_counter() - t_all:.1f}",
            flush=True,
        )
        return n
