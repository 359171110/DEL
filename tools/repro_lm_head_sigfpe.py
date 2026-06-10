#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import transformers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated lm_head attribution cases.")
    parser.add_argument("--model", required=True, help="Local model path")
    parser.add_argument(
        "--tensor-path",
        help="Optional torch.save file produced by capture_del_hidden_state.py",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for child cases",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for child cases, usually cuda",
    )
    parser.add_argument(
        "--output-jsonl",
        default="logs/lm_head_attribution_results.jsonl",
        help="Where to append matrix results",
    )
    parser.add_argument(
        "--case",
        help="Internal use: run a single case in the current process",
    )
    return parser


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def tensor_metadata(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
    }


def load_model(model_path: str, dtype: torch.dtype, device: str):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        use_safetensors=True,
        device_map=None,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)
    return model


def load_hidden_tensor(tensor_path: str, dtype: torch.dtype, device: str) -> torch.Tensor:
    loaded = torch.load(tensor_path, map_location="cpu")
    if isinstance(loaded, dict) and "tensor" in loaded:
        loaded = loaded["tensor"]
    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"Unsupported tensor payload type: {type(loaded)!r}")
    return loaded.to(device=device, dtype=dtype)


def make_case_tensor(
    case_name: str,
    model,
    dtype: torch.dtype,
    device: str,
    tensor_path: str | None,
) -> torch.Tensor:
    hidden_size = model.lm_head.weight.shape[1]

    if case_name == "synthetic-seq1":
        return torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
    if case_name == "synthetic-seq2":
        return torch.randn(1, 2, hidden_size, device=device, dtype=dtype)
    if case_name == "synthetic-view-seq1":
        base = torch.randn(1, 2, hidden_size, device=device, dtype=dtype)
        return base[:, :1, :]
    if case_name == "synthetic-clone-seq1":
        base = torch.randn(1, 2, hidden_size, device=device, dtype=dtype)
        return base[:, :1, :].clone()
    if case_name == "synthetic-contiguous-seq1":
        base = torch.randn(1, 2, hidden_size, device=device, dtype=dtype)
        return base[:, :1, :].contiguous()
    if case_name == "synthetic-double-then-slice":
        base = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
        return torch.cat([base, base], dim=1)
    if case_name == "captured-original":
        if not tensor_path:
            raise ValueError("--tensor-path is required for captured-original")
        return load_hidden_tensor(tensor_path, dtype=dtype, device=device)
    if case_name == "captured-contiguous":
        if not tensor_path:
            raise ValueError("--tensor-path is required for captured-contiguous")
        return load_hidden_tensor(tensor_path, dtype=dtype, device=device).contiguous()
    if case_name == "captured-clone":
        if not tensor_path:
            raise ValueError("--tensor-path is required for captured-clone")
        return load_hidden_tensor(tensor_path, dtype=dtype, device=device).clone()
    raise ValueError(f"Unsupported case: {case_name}")


def run_single_case(args) -> int:
    dtype = parse_dtype(args.dtype)
    model = load_model(args.model, dtype=dtype, device=args.device)
    tensor = make_case_tensor(args.case, model, dtype, args.device, args.tensor_path)

    meta: Dict[str, Any] = {
        "case": args.case,
        "tensor": tensor_metadata(tensor),
        "lm_head_weight_shape": list(model.lm_head.weight.shape),
        "lm_head_weight_dtype": str(model.lm_head.weight.dtype),
        "device": args.device,
    }

    torch.cuda.synchronize()
    with torch.inference_mode():
        logits = model.lm_head(tensor)
        torch.cuda.synchronize()
    meta["logits_shape"] = list(logits.shape)
    print(json.dumps(meta, ensure_ascii=False))
    return 0


def case_matrix(tensor_path: str | None) -> List[str]:
    cases = [
        "synthetic-seq1",
        "synthetic-seq2",
        "synthetic-view-seq1",
        "synthetic-clone-seq1",
        "synthetic-contiguous-seq1",
        "synthetic-double-then-slice",
    ]
    if tensor_path:
        cases.extend(
            [
                "captured-original",
                "captured-contiguous",
                "captured-clone",
            ]
        )
    return cases


def run_matrix(args) -> int:
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = case_matrix(args.tensor_path)

    for case_name in cases:
        cmd = [
            sys.executable,
            __file__,
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--device",
            args.device,
            "--case",
            case_name,
        ]
        if args.tensor_path:
            cmd.extend(["--tensor-path", args.tensor_path])

        env = os.environ.copy()
        env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

        record: Dict[str, Any] = {
            "case": case_name,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                record["parsed"] = json.loads(proc.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                pass

        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = "ok" if proc.returncode == 0 else f"crash({proc.returncode})"
        print(f"[{status}] {case_name}")

    print(f"Results written to {output_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.case:
        return run_single_case(args)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
