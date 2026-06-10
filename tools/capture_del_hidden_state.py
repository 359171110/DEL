#!/usr/bin/env python3
import argparse
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark as benchmark_module
from arguments import Arguments
from benchmark import BenchmarkArguments
from generate import load_model_and_tokenizer, setup
from self_speculation.generator_base import GenerationConfig
import self_speculation.llama_model_utils as llama_model_utils
import transformers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a DEL single-token hidden state before lm_head."
    )
    parser.add_argument("--model", required=True, help="Local model path")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. gsm8k")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--num-speculations", type=int, default=6)
    parser.add_argument("--sample", default="False", choices=["True", "False"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./logs")
    parser.add_argument(
        "--capture-path",
        default="logs/captured_del_hidden.pt",
        help="Where to save the captured tensor payload",
    )
    return parser


def bool_arg(value: str) -> bool:
    return value.lower() == "true"


def stack_functions() -> list[str]:
    frames = []
    for frame_info in inspect.stack()[2:8]:
        frames.append(frame_info.function)
    return frames


def main() -> int:
    args_ns = build_parser().parse_args()
    capture_path = Path(args_ns.capture_path)
    capture_path.parent.mkdir(parents=True, exist_ok=True)

    args = Arguments(
        model=args_ns.model,
        seed=args_ns.seed,
        output_dir=args_ns.output_dir,
    )
    bench_args = BenchmarkArguments(
        dataset=args_ns.dataset,
        num_samples=args_ns.num_samples,
    )
    gen_config = GenerationConfig(
        max_steps=args_ns.max_steps,
        exit_layer=args_ns.exit_layer,
        num_speculations=args_ns.num_speculations,
        generation_strategy="DEL_speculative",
        sample=bool_arg(args_ns.sample),
    )

    original_safe_lm_head = llama_model_utils._safe_lm_head
    captured = {"done": False}

    def wrapped_safe_lm_head(
        model: transformers.LlamaForCausalLM,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not captured["done"]
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.float16
            and hidden_states.shape[1] == 1
        ):
            payload = {
                "tensor": hidden_states.detach().cpu(),
                "meta": {
                    "shape": list(hidden_states.shape),
                    "dtype": str(hidden_states.dtype),
                    "device": str(hidden_states.device),
                    "is_contiguous": bool(hidden_states.is_contiguous()),
                    "stride": list(hidden_states.stride()),
                    "storage_offset": int(hidden_states.storage_offset()),
                    "call_stack_functions": stack_functions(),
                },
            }
            torch.save(payload, capture_path)
            captured["done"] = True
            print(
                json.dumps(
                    {
                        "capture_path": str(capture_path),
                        "meta": payload["meta"],
                    },
                    ensure_ascii=False,
                )
            )
        return original_safe_lm_head(model, hidden_states)

    llama_model_utils._safe_lm_head = wrapped_safe_lm_head
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        setup(args, device=device)
        transformers.utils.logging.set_verbosity_error()
        model, tokenizer = load_model_and_tokenizer(args, device=device)
        metrics = benchmark_module.benchmark(model, tokenizer, bench_args, gen_config, seed=args.seed)
        print(json.dumps({"benchmark_metrics": metrics}, ensure_ascii=False))
    finally:
        llama_model_utils._safe_lm_head = original_safe_lm_head

    if not captured["done"]:
        raise RuntimeError("No matching DEL single-token hidden state was captured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
