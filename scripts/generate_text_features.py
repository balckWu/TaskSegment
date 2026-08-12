from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasksegment.configs import TEXT_GROUP_NAMES


GROUP_ORDER = tuple(TEXT_GROUP_NAMES)


def _mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor | None,
) -> torch.Tensor:
    valid = attention_mask.bool()
    if special_tokens_mask is not None:
        valid = valid & ~special_tokens_mask.bool()
    mask = valid.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def encode_group(
    prompts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    special_tokens_mask = batch.pop("special_tokens_mask", None)
    batch = {key: value.to(device) for key, value in batch.items()}
    if special_tokens_mask is not None:
        special_tokens_mask = special_tokens_mask.to(device)

    output = model(**batch, output_hidden_states=True, return_dict=True)
    last_four = torch.stack(output.hidden_states[-4:], dim=0).mean(dim=0)
    sentence_vectors = _mean_pool(
        last_four,
        batch["attention_mask"],
        special_tokens_mask,
    )
    return sentence_vectors.mean(dim=0)


def load_prompts(path: Path) -> Dict[str, Dict[str, List[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError("Prompt JSON must be an object keyed by domain")
    for domain, groups in obj.items():
        if not isinstance(groups, dict):
            raise ValueError(f"domain={domain} must map to an object")
        for group in GROUP_ORDER:
            values = groups.get(group)
            if not isinstance(values, list) or not values:
                raise ValueError(f"domain={domain} group={group} must be a non-empty string list")
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise ValueError(f"domain={domain} group={group} contains an invalid prompt")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate five structured medical text summary tokens per domain"
    )
    parser.add_argument("--prompt-json", default="./text_prompts.json")
    parser.add_argument("--output-dir", default="./text_features_llm")
    parser.add_argument("--model", default="dmis-lab/biobert-large-cased-v1.1")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        help="BioBERT works reliably with the slow WordPiece tokenizer; leave disabled unless needed.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device
    )
    prompts = load_prompts(Path(args.prompt_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading text encoder: {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=bool(args.use_fast_tokenizer),
        local_files_only=bool(args.local_files_only),
    )
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=bool(args.local_files_only),
    ).to(device).eval()
    hidden_dim = int(model.config.hidden_size)

    for domain, groups in prompts.items():
        tokens = []
        for group in GROUP_ORDER:
            token = encode_group(
                groups[group],
                tokenizer,
                model,
                device,
                max_length=args.max_length,
            )
            tokens.append(token.unsqueeze(0))

        # Keep the encoder's original geometry. Common-direction centering is
        # applied deterministically by load_text_bank at train and inference.
        text_features = torch.cat(tokens, dim=0).float().cpu().contiguous()
        if not torch.isfinite(text_features).all():
            raise RuntimeError(f"Non-finite features generated for {domain}")

        save_obj = {
            "schema_version": 3,
            "text_features": text_features,
            "text_mask": torch.ones(len(GROUP_ORDER), dtype=torch.long),
            "text_group_summary_count": len(GROUP_ORDER),
            "group_names": list(GROUP_ORDER),
            "prompts": groups,
            "hidden_dim": hidden_dim,
            "encoder": args.model,
            "embedding_type": "five_group_last4_mean_no_special_tokens",
        }
        save_path = output_dir / f"text_features_{domain}.pt"
        torch.save(save_obj, save_path)
        print(f"saved {domain}: {save_path} shape={tuple(text_features.shape)}")

    print("Done. Run scripts/audit_text_features.py before training.")


if __name__ == "__main__":
    main()
