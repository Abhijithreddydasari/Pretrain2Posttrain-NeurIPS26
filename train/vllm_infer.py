"""vLLM batch inference: load E4B once, hot-swap LoRA per checkpoint."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml
from PIL import Image

from train.data_utils import prompt_for_row, resolve_image
from train.infer_engine import _truncate_at_svg_close, load_bench_manifests
from train.model_load import ensure_chat_template

# Re-export for sweep_checkpoints convenience.
__all__ = ["VllmInferEngine", "load_bench_manifests"]


_TAG_RE = re.compile(
    r"<!--[\s\S]*?-->|<\?[^>]*\?>|<!\[CDATA\[[\s\S]*?\]\]>|<![^>]*>|"
    r"</?\s*[A-Za-z_][\w:.-]*(?:\s+(?:[^<>\"']|\"[^\"]*\"|'[^']*')*)?\s*/?>",
    flags=re.IGNORECASE,
)
_TAG_NAME_RE = re.compile(r"<\s*(/?)\s*([A-Za-z_][\w:.-]*)", flags=re.IGNORECASE)


def recover_svg_prefix(text: str, *, max_open_tags: int = 256) -> tuple[str | None, str | None]:
    """Close a deterministic well-formed prefix of an unterminated SVG.

    Recovery never replaces the raw prediction. It is stored separately for a
    secondary renderability analysis. Content after the last complete tag, or
    after the first mismatched closing tag, is discarded.
    """
    start = re.search(r"<svg\b", text or "", flags=re.IGNORECASE)
    if not start:
        return None, None
    fragment = text[start.start() :]
    stack: list[str] = []
    cursor = 0
    kept_end = 0
    saw_root = False
    units: list[str] = []
    unit_ends: list[int] = []
    stack_snapshots: list[list[str]] = []
    recovery = "closed_open_tags"

    for match in _TAG_RE.finditer(fragment):
        between = fragment[cursor : match.start()]
        if "<" in between:
            break
        token = match.group(0)
        cursor = match.end()
        kept_end = cursor
        if token.startswith(("<!--", "<?", "<!")):
            continue
        named = _TAG_NAME_RE.match(token)
        if not named:
            break
        closing = bool(named.group(1))
        name = named.group(2).lower()
        self_closing = token.rstrip().endswith("/>")
        if not saw_root:
            if closing or name != "svg":
                break
            saw_root = True
        if closing:
            if not stack or stack[-1] != name:
                kept_end = match.start()
                break
            stack.pop()
            if not stack:
                return fragment[:cursor], "already_closed"
        elif not self_closing:
            stack.append(name)
            if len(stack) > max_open_tags:
                return None, "too_many_open_tags"

        # Detect only exact, adjacent XML-unit repetition (three copies). This
        # avoids guessing from merely similar geometry while catching common
        # autoregressive loops. Keep the first copy and close its open stack.
        unit = fragment[unit_ends[-1] if unit_ends else 0 : cursor]
        units.append(unit)
        unit_ends.append(cursor)
        stack_snapshots.append(list(stack))
        n_units = len(units)
        for block_len in range(1, min(8, n_units // 3) + 1):
            a = units[n_units - 3 * block_len : n_units - 2 * block_len]
            b = units[n_units - 2 * block_len : n_units - block_len]
            c = units[n_units - block_len :]
            if a == b == c and sum(len(x) for x in a) >= 32:
                keep_index = n_units - 2 * block_len - 1
                # Never cut away the root SVG unit.
                if keep_index >= 0:
                    kept_end = unit_ends[keep_index]
                    stack = stack_snapshots[keep_index]
                    recovery = "trimmed_exact_repetition_and_closed_open_tags"
                    break
        if recovery.startswith("trimmed_"):
            break

    if not saw_root or not stack or kept_end <= 0:
        return None, None
    recovered = fragment[:kept_end].rstrip() + "".join(f"</{name}>" for name in reversed(stack))
    return recovered, recovery


def _load_existing_predictions(path: Path, expected_ids: set[str], protocol: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                print(f"vllm_infer: ignore malformed resume row {path}:{line_no}", flush=True)
                continue
            row_id = str(row.get("id", ""))
            if row_id in expected_ids and row.get("protocol", protocol) == protocol:
                rows[row_id] = row
    return rows


class VllmInferEngine:
    """Offline vLLM engine with per-checkpoint LoRARequest swapping."""

    def __init__(
        self,
        config: Path,
        *,
        batch_size: int = 64,
        gpu_memory_utilization: float = 0.92,
        max_new_tokens: int | None = None,
        enforce_eager: bool = True,
    ):
        from transformers import AutoProcessor
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.LoRARequest = LoRARequest
        self.cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        self.prompt = self.cfg.get("prompt_template", "").strip()
        self.prefix = self.cfg.get("svg_prefix_scaffold", "")
        self.gen_cfg = dict(self.cfg.get("generation", {}))
        if max_new_tokens is not None:
            self.gen_cfg["max_new_tokens"] = int(max_new_tokens)
        self.model_id = self.cfg["model_id"]
        self.batch_size = max(1, int(batch_size))

        t0 = time.perf_counter()
        print(
            f"vllm_infer: loading {self.model_id} batch_size={self.batch_size}",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.processor = ensure_chat_template(self.processor, self.model_id, trust_remote_code=True)
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
            enforce_eager=enforce_eager,
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
            include_stop_str_in_output=True,
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
        resume: bool = True,
    ) -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        all_pairs = cached if cached is not None else self.preload_rows(rows)
        expected_ids = {str(row["id"]) for row, _ in all_pairs}
        if len(expected_ids) != len(all_pairs):
            raise ValueError("evaluation manifest contains duplicate ids")
        existing = _load_existing_predictions(out, expected_ids, protocol) if resume else {}
        pairs = [(row, image) for row, image in all_pairs if str(row["id"]) not in existing]
        sampling_params = self._sampling_params(protocol)
        n_new = 0
        t_all = time.perf_counter()
        print(
            f"vllm_infer: resume kept={len(existing)} remaining={len(pairs)} total={len(expected_ids)}",
            flush=True,
        )

        mode = "a" if existing else "w"
        with out.open(mode, encoding="utf-8") as f:
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
                    choice = out_obj.outputs[0]
                    decoded = choice.text
                    stop_reason = getattr(choice, "stop_reason", None)
                    finish_reason = getattr(choice, "finish_reason", None)
                    decoded = _truncate_at_svg_close(decoded)
                    if protocol == "svg_prefix":
                        decoded = self.prefix + decoded
                    # Some vLLM versions strip the matched stop string despite
                    # include_stop_str_in_output. Restore only the observed stop.
                    if str(stop_reason).lower() == "</svg>" and "</svg>" not in decoded.lower():
                        decoded += "</svg>"
                    decoded = _truncate_at_svg_close(decoded)
                    recovered_svg, recovery = recover_svg_prefix(decoded)
                    record = {
                        "id": row["id"],
                        "pred_text": decoded,
                        "protocol": protocol,
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "new_tokens": len(getattr(choice, "token_ids", []) or []),
                        "has_svg_open": bool(re.search(r"<svg\b", decoded, flags=re.IGNORECASE)),
                        "has_svg_close": "</svg>" in decoded.lower(),
                        "hit_length_limit": str(finish_reason).lower() == "length",
                    }
                    if recovered_svg and recovered_svg != decoded:
                        record["recovered_svg"] = recovered_svg
                        record["recovery"] = recovery
                    f.write(json.dumps(record) + "\n")
                    existing[str(row["id"])] = record
                    n_new += 1
                    print(
                        f"generated {row['id']} new_chars={len(decoded)} "
                        f"sec={per_sample:.1f} finish={finish_reason} stop={stop_reason} "
                        f"closed={record['has_svg_close']} adapter={self._adapter_tag}",
                        flush=True,
                    )
                f.flush()

        missing = expected_ids - set(existing)
        if missing:
            raise RuntimeError(f"prediction file incomplete: missing {len(missing)}/{len(expected_ids)} ids")
        # Canonicalize order and remove any duplicate rows accumulated by an
        # interrupted append before declaring the checkpoint complete.
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row, _ in all_pairs:
                f.write(json.dumps(existing[str(row["id"])]) + "\n")
        tmp.replace(out)
        print(
            f"complete {len(existing)} preds → {out} new={n_new} "
            f"total_sec={time.perf_counter() - t_all:.1f}",
            flush=True,
        )
        return len(existing)
