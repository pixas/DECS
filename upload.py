#!/usr/bin/env python3
"""Upload a local checkpoint to Hugging Face with metadata sanitization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover
    HfApi = None  # type: ignore[assignment]


DEFAULT_SOURCE = (
    "~/checkpoints/"
    "deepscaler_mix_r1_distill_qwen1.5b_grpo_proc_length_a001_b001_c001_n16_"
    "nozeroadv_fm0_constant_dapo_adp_global_step_260"
)
DEFAULT_MODEL_NAME = "DECS_1.5B"
DEFAULT_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

JSON_METADATA_NAMES = {
    "config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "adapter_config.json",
    "model.safetensors.index.json",
}
TEXT_METADATA_SUFFIXES = {".yaml", ".yml"}

PATH_LIKE_KEYWORDS = ("path", "file", "dir", "cache", "checkpoint", "ckpt", "local")
PATH_REPLACEMENTS = [
    (re.compile(r"/home/[^/\s]+"), "/home/<user>"),
    (re.compile(r"/mnt/petrelfs/[^/\s]+"), "/mnt/petrelfs/<user>"),
    (re.compile(r"/mnt/phwfile/[^/\s]+"), "/mnt/phwfile/<group>"),
    (re.compile(r"([A-Za-z]:\\Users\\)[^\\]+"), r"\1<user>"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local checkpoint to Hugging Face Hub."
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE,
        help="Local checkpoint directory. Default is your DECS 1.5B checkpoint path.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model repository name on your Hugging Face account.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Full repo id, e.g. 'your-hf-username/DECS_1.5B'. Overrides --username.",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Hugging Face username used with --model-name when --repo-id is not set.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model id used in README metadata.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload as a private model repo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare staging files only, do not upload.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep temporary staging directory for inspection.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload DECS_1.5B model with sanitized metadata and README",
        help="Commit message for Hugging Face upload.",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable metadata de-identification.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Do not generate README.md automatically.",
    )
    return parser.parse_args()


def looks_like_local_path(value: str) -> bool:
    if value.startswith(("http://", "https://", "hf://")):
        return False
    if value.startswith(("~", "/")):
        return True
    if re.match(r"^[A-Za-z]:\\", value):
        return True
    return any(x in value for x in ("/home/", "/mnt/", "\\Users\\"))


def basename_from_path_like(value: str) -> str:
    if re.match(r"^[A-Za-z]:\\", value):
        return PureWindowsPath(value).name
    return Path(value).name


def sanitize_path_string(value: str, home_path: str, username: str) -> str:
    result = value
    if home_path:
        result = result.replace(home_path, "<HOME>")
    if username:
        result = result.replace(f"/{username}/", "/<user>/")
        result = result.replace(f"\\{username}\\", "\\<user>\\")
    for pattern, replacement in PATH_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def sanitize_json_obj(
    obj: Any,
    *,
    key: str | None,
    home_path: str,
    username: str,
    model_name: str,
) -> Any:
    if isinstance(obj, dict):
        return {
            k: sanitize_json_obj(
                v,
                key=k,
                home_path=home_path,
                username=username,
                model_name=model_name,
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            sanitize_json_obj(
                item,
                key=key,
                home_path=home_path,
                username=username,
                model_name=model_name,
            )
            for item in obj
        ]
    if not isinstance(obj, str):
        return obj

    normalized = sanitize_path_string(obj, home_path=home_path, username=username)
    key_lower = (key or "").lower()

    if key_lower == "_name_or_path" and looks_like_local_path(obj):
        return model_name

    if any(token in key_lower for token in PATH_LIKE_KEYWORDS) and looks_like_local_path(
        obj
    ):
        file_name = basename_from_path_like(obj)
        return file_name or "<redacted-local-path>"

    return normalized


def sanitize_json_file(path: Path, home_path: str, username: str, model_name: str) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    sanitized = sanitize_json_obj(
        data,
        key=None,
        home_path=home_path,
        username=username,
        model_name=model_name,
    )
    if sanitized == data:
        return False

    path.write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def sanitize_text_file(path: Path, home_path: str, username: str) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return False
    sanitized = sanitize_path_string(content, home_path=home_path, username=username)
    if sanitized == content:
        return False
    path.write_text(sanitized, encoding="utf-8")
    return True


def sanitize_metadata(staging_dir: Path, model_name: str) -> list[Path]:
    changed: list[Path] = []
    home_path = os.path.expanduser("~")
    username = os.path.basename(home_path.rstrip("/"))

    for file_path in staging_dir.rglob("*"):
        if not file_path.is_file():
            continue
        name_lower = file_path.name.lower()
        suffix_lower = file_path.suffix.lower()

        if name_lower in JSON_METADATA_NAMES:
            if sanitize_json_file(file_path, home_path, username, model_name):
                changed.append(file_path)
            continue

        if suffix_lower == ".json" and any(
            marker in name_lower for marker in ("config", "args", "trainer", "training")
        ):
            if sanitize_json_file(file_path, home_path, username, model_name):
                changed.append(file_path)
            continue

        if suffix_lower in TEXT_METADATA_SUFFIXES and any(
            marker in name_lower for marker in ("config", "args", "trainer", "training")
        ):
            if sanitize_text_file(file_path, home_path, username):
                changed.append(file_path)

    return changed


def build_readme(repo_id: str, model_name: str, base_model: str, source_dir: Path) -> str:
    today = dt.date.today().isoformat()
    source_name = source_dir.name
    return f"""---
language:
- zh
- en
pipeline_tag: text-generation
tags:
- deepscaler
- reasoning
- grpo
- qwen2
base_model: {base_model}
license: other
---

# {model_name}

{model_name} is a reasoning-focused causal language model built from `{base_model}` and further trained with a GRPO-style setup on mixed data.

## Model Summary

- Base model: `{base_model}`
- Upload source: local checkpoint `{source_name}`
- Upload date: `{today}`
- Recommended use: long-form reasoning and mathematical/problem-solving style generation

## Quick Start (Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {{"role": "user", "content": "Solve: If x^2 - 5x + 6 = 0, what are x values?"}}
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

## Quick Start (vLLM)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{repo_id}", trust_remote_code=True)
sampling = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=512)
prompt = "Please reason step by step: what is 37 * 48?"
outputs = llm.generate([prompt], sampling_params=sampling)
print(outputs[0].outputs[0].text)
```

## Notes

- This model may produce incorrect or unverifiable reasoning. Always validate outputs in high-stakes settings.
- Performance can vary by prompt style and decoding parameters.
- License and acceptable-use constraints should follow the upstream base model and your deployment policy.

## Privacy Sanitization

Before uploading, local/private metadata is sanitized:

- Absolute local paths are de-identified.
- Username-like path segments are masked.
- Path-like fields in config metadata are rewritten to public-safe values.

## Citation

If you use this model, please cite this repository and the upstream base model:

- `{repo_id}`
- `{base_model}`
"""


def resolve_repo_id(
    args: argparse.Namespace,
    api: Any,
    token: str | None,
) -> str:
    if args.repo_id:
        return args.repo_id
    if args.username:
        return f"{args.username}/{args.model_name}"
    if args.dry_run:
        return f"your-hf-username/{args.model_name}"
    if api is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        )
    info = api.whoami(token=token)
    username = info.get("name") or info.get("user")
    if not username:
        raise RuntimeError(
            "Cannot infer Hugging Face username. Use --username or --repo-id."
        )
    return f"{username}/{args.model_name}"


def copy_checkpoint(source_dir: Path, payload_dir: Path) -> None:
    shutil.copytree(source_dir, payload_dir, dirs_exist_ok=True)


def summarize_payload(payload_dir: Path) -> tuple[int, float]:
    files = [p for p in payload_dir.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in files)
    return len(files), total_size / (1024**3)


def run(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not args.dry_run and not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set --token or env HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
        )

    api = HfApi(token=token) if HfApi is not None else None
    repo_id = resolve_repo_id(args, api=api, token=token)

    temp_dir_obj = None
    if args.keep_staging:
        staging_root = Path(tempfile.mkdtemp(prefix="decs_hf_upload_"))
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="decs_hf_upload_")
        staging_root = Path(temp_dir_obj.name)

    payload_dir = staging_root / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    copy_checkpoint(source_dir, payload_dir)

    changed_files: list[Path] = []
    if not args.no_sanitize:
        changed_files = sanitize_metadata(payload_dir, model_name=args.model_name)

    if not args.no_readme:
        readme_path = payload_dir / "README.md"
        readme_path.write_text(
            build_readme(
                repo_id=repo_id,
                model_name=args.model_name,
                base_model=args.base_model,
                source_dir=source_dir,
            ),
            encoding="utf-8",
        )

    file_count, size_gb = summarize_payload(payload_dir)

    print(f"[info] Source: {source_dir}")
    print(f"[info] Repo: {repo_id}")
    print(f"[info] Staging dir: {payload_dir}")
    print(f"[info] Payload files: {file_count}, total size: {size_gb:.2f} GiB")
    if changed_files:
        print("[info] Sanitized metadata files:")
        for path in changed_files:
            print(f"  - {path.relative_to(payload_dir)}")
    else:
        print("[info] No metadata file changed during sanitization.")

    if args.dry_run:
        print("[dry-run] Skip upload.")
        if not args.keep_staging:
            print("[dry-run] Staging directory will be cleaned up automatically.")
        return

    if api is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        )

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(payload_dir),
        commit_message=args.commit_message,
        token=token,
    )
    print(f"[done] Upload complete: https://huggingface.co/{repo_id}")

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()


if __name__ == "__main__":
    run(parse_args())
