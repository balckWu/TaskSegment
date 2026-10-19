from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from PIL import Image

from .model import Iris2D, segmentation_loss
from .project import DOMAINS, MODALITY, project_utilities, source_hashes
from .protocol import default_data_root, load_recipe, set_cosine_lr, verify_reference


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def precision(device, amp):
    return torch.autocast("cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()


def get_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for a small smoke test.")
    return torch.device(name)


def data_manifest(dataset, data_root):
    base = Path(data_root).resolve()
    rows = []
    for sample in dataset.samples:
        row = dict(sample)
        for name in ("image", "mask"):
            # Keep logical paths under data_root; symlinks may resolve outside it.
            row[name] = Path(sample[name]).absolute().relative_to(base).as_posix()
        rows.append(row)
    return rows


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_datasets(args, data_module):
    roots = {d: str(Path(args.data_root).resolve() / d) for d in DOMAINS}
    datasets = {
        split: data_module.MultiOrganDataset(
            roots, data_module.get_image_transform(), split=split,
            image_size=(args.image_size, args.image_size), seed=args.seed,
            cache_in_memory=args.cache_dataset, use_cpu_clahe=False,
        ) for split in ("train", "val", "test")
    }
    seen = {}
    for split, dataset in datasets.items():
        for domain in DOMAINS:
            if not dataset.organ_to_indices.get(domain):
                raise ValueError(f"Missing or empty domain: {domain}/{split}")
        for sample in dataset.samples:
            for key in ("image", "mask"):
                real = str(Path(sample[key]).resolve())
                previous = seen.setdefault((key, real), split)
                if previous != split:
                    raise ValueError(f"Same {key} file appears in {previous} and {split}: {real}")
    if not getattr(args, "smoke", False):
        expected = {"thyroid": (855, 285, 286), "TN3K": (2447, 432, 614),
                    "BUSI_WHU": (555, 186, 186), "BUS-BRA": (1312, 280, 283),
                    "OTU": (656, 164, 382), "prostate": (67, 14, 16)}
        for column, split in enumerate(("train", "val", "test")):
            for domain, counts in expected.items():
                actual = len(datasets[split].organ_to_indices[domain])
                if actual != counts[column]:
                    raise ValueError(f"Paper split count mismatch {domain}/{split}: {actual} != {counts[column]}")
    manifests = {s: data_manifest(d, args.data_root) for s, d in datasets.items()}
    return datasets, manifests


def select_supports(dataset, count, seed):
    """Match TSV3 prototypes.py: sorted domains, seeded positive sampling."""
    rng = np.random.default_rng(seed)
    manifest = {}
    for domain in sorted(DOMAINS):
        pool = np.asarray(dataset.organ_to_positive_indices.get(domain, []), dtype=np.int64)
        if len(pool) < count:
            raise ValueError(f"{domain} has {len(pool)} positive training pairs; need {count}")
        manifest[domain] = sorted(int(i) for i in rng.choice(pool, size=count, replace=False))
    return manifest


def load_batch(dataset, indices, device):
    pairs = [dataset[i] for i in indices]
    return (torch.stack([p[0] for p in pairs]).to(device),
            torch.stack([p[1] for p in pairs]).to(device))


@torch.inference_mode()
def build_prototypes(model, train, supports, device):
    model.eval()
    tokens = {}
    for domain, indices in supports.items():
        # Per-support encoding makes the result independent of support batch size.
        parts = []
        for index in indices:
            image, mask = load_batch(train, [index], device)
            parts.append(model.encode_support(image, mask).float())
        tokens[domain] = torch.stack(parts).mean(0)
    return tokens


@torch.inference_mode()
def evaluate(model, query, prototypes, device, metric_module, batch_size=4, output=None, save_predictions=False):
    model.eval()
    rows = []
    domains = {}
    for domain in sorted(DOMAINS):
        indices = query.get_organ_indices(domain)
        for start in range(0, len(indices), batch_size):
            group = indices[start:start + batch_size]
            images, masks = load_batch(query, group, device)
            logits = model.segment(images, prototypes[domain])
            predicted = (logits.float().softmax(1)[:, 1] >= .5).cpu().numpy().astype(np.uint8)
            truth = masks.cpu().numpy().astype(np.uint8)
            for index, pred, target in zip(group, predicted, truth):
                info = query.get_sample_info(index)
                rows.append(dict(domain=domain, sample_id=info["sample_id"],
                    dice=metric_module.binary_dice(pred, target),
                    iou=metric_module.binary_iou(pred, target),
                    hd95=metric_module.binary_hd95(pred, target)))
                if save_predictions:
                    folder = Path(output) / "predictions" / domain
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pred * 255).save(folder / f"{info['sample_id']}.png")
        local = [r for r in rows if r["domain"] == domain]
        domains[domain] = {m: float(np.mean([r[m] for r in local])) for m in ("dice", "iou", "hd95")}
        domains[domain]["count"] = len(local)
    result = {"per_domain": domains, "macro": {
        m: float(np.mean([v[m] for v in domains.values()])) for m in ("dice", "iou", "hd95")}}
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        with (Path(output) / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def schedule(domains, count):
    result = []
    for _ in range(count // len(domains)):
        chunk = list(domains)
        random.shuffle(chunk)
        result.extend(chunk)
    tail = list(domains)
    random.shuffle(tail)
    return result + tail[:count % len(domains)]


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def common_arguments(parser):
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dataset", action="store_true")


def train_main():
    recipe, controlled = load_recipe()
    parser = argparse.ArgumentParser(description="Independent paper-based Iris-2D, TSV3 matched-budget protocol")
    common_arguments(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=recipe["model"]["base_channels"])
    parser.add_argument("--context-queries", type=int, default=recipe["model"]["context_queries"])
    parser.add_argument("--attention-depth", type=int, default=recipe["model"]["attention_depth"])
    parser.add_argument("--support-count", type=int, default=recipe["support_count"], help="Iris one-shot training")
    parser.add_argument("--query-count", type=int, default=controlled["batch_size"])
    parser.add_argument("--eval-support-count", type=int, default=recipe["eval_support_count"])
    parser.add_argument("--eval-batch-size", type=int, default=controlled["val_batch_size"])
    parser.add_argument("--epochs", type=int, default=controlled["epochs"])
    parser.add_argument("--episodes-per-epoch", type=int, default=controlled["episodes_per_epoch"])
    parser.add_argument("--grad-accum-steps", type=int, default=controlled["grad_accum_steps"])
    parser.add_argument("--lr", type=float, default=recipe["lr"])
    parser.add_argument("--min-lr", type=float, default=recipe["min_lr"])
    parser.add_argument("--weight-decay", type=float, default=recipe["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=controlled["warmup_epochs"])
    parser.add_argument("--patience", type=int, default=controlled["patience"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume this output directory's checkpoint.last.pth")
    parser.add_argument("--smoke", action="store_true", help="One episode/epoch on all real validation cases; no paper claim")
    args = parser.parse_args()
    verify_reference()
    for name in ("support_count", "query_count", "eval_support_count", "eval_batch_size", "epochs", "episodes_per_epoch", "grad_accum_steps", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.image_size <= 0 or args.image_size % 16:
        parser.error("--image-size must be positive and divisible by 16")
    if args.min_lr <= 0 or args.lr < args.min_lr or args.weight_decay < 0 or args.warmup_epochs < 0:
        parser.error("Require lr >= min-lr > 0, nonnegative weight decay and warmup")
    if args.smoke:
        args.epochs, args.episodes_per_epoch = 1, 1
    elif args.image_size != 512:
        parser.error("Formal baseline requires 512x512; other sizes require --smoke")
    if not args.smoke:
        required = {"lr": recipe["lr"], "min_lr": recipe["min_lr"],
                    "weight_decay": recipe["weight_decay"], "warmup_epochs": controlled["warmup_epochs"],
                    "epochs": controlled["epochs"], "patience": controlled["patience"],
                    "episodes_per_epoch": controlled["episodes_per_epoch"],
                    "grad_accum_steps": controlled["grad_accum_steps"], "query_count": controlled["batch_size"],
                    "no_amp": not controlled["amp"]}
        differences = [name for name, value in required.items() if getattr(args, name) != value]
        if differences:
            parser.error("Formal run must match the audited baseline protocol: " + ", ".join(differences))
    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Use a fresh output or --resume.")
    output.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    data, augmentation, metrics = project_utilities()
    data.set_seed(args.seed, deterministic=True)
    datasets, manifests = load_datasets(args, data)
    train, val = datasets["train"], datasets["val"]
    if args.smoke:
        val.organ_to_indices = {d: ids[:2] for d, ids in val.organ_to_indices.items()}
    supports = select_supports(train, args.eval_support_count, args.seed)
    for domain in DOMAINS:
        if len(train.organ_to_positive_indices[domain]) <= args.support_count:
            raise ValueError(f"{domain} needs more positive training pairs than support-count")
    hashes = source_hashes()
    split_hash = fingerprint(manifests)
    config = dict(base_channels=args.base_channels, context_queries=args.context_queries,
                  attention_depth=args.attention_depth)
    model = Iris2D(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, bad_epochs = 0, -math.inf, 0
    history = []
    protocol = vars(args).copy()
    ignored = {"resume", "device", "cache_dataset", "eval_batch_size"}
    if args.resume:
        state = torch.load(output / "checkpoint.last.pth", map_location="cpu", weights_only=True)
        if state["source_hashes"] != hashes or state["split_fingerprint"] != split_hash:
            raise ValueError("Sources or split manifest changed; resume refused")
        if any(protocol[k] != state["protocol"][k] for k in protocol if k not in ignored):
            raise ValueError("Training protocol changed; use the original arguments to resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best, bad_epochs = state["epoch"], state["best_val_dice"], state["bad_epochs"]
        history = state["history"]
        restore_rng(state["rng"])
    else:
        write_json(output / "splits.json", manifests)
        write_json(output / "support_manifest.json", {
            d: [manifests["train"][i] for i in indices] for d, indices in supports.items()})
        write_json(output / "protocol.json", {
            "implementation": "independent paper-based Iris 2D adaptation, TSV3 matched-budget",
            "args": protocol, "model": model.config, "source_hashes": hashes,
            "split_fingerprint": split_hash, "parameters": sum(p.numel() for p in model.parameters()),
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"},
            "optimizer": recipe["optimizer"], "lr_schedule": recipe["lr_schedule"],
            "reference_training_protocol": controlled,
        })
        write_json(output / "checkpoint.pth.config.json", {
            "method": "iris2d", "training_args": protocol, "model_config": model.config,
            "optimizer": recipe["optimizer"], "lr_schedule": recipe["lr_schedule"],
            "effective_batch_size": args.query_count * args.grad_accum_steps,
            "selection_metric": "validation_domain_macro_dice", "source_hashes": hashes})
    print(f"Iris-2D independent adaptation | parameters={sum(p.numel() for p in model.parameters()):,} | eval K={args.eval_support_count}", flush=True)
    print(f"optimizer=AdamW lr={args.lr:g} min_lr={args.min_lr:g} warmup={args.warmup_epochs} schedule={recipe['lr_schedule']} weight_decay={args.weight_decay:g}", flush=True)
    if bad_epochs >= args.patience:
        print("This run already reached early stopping; no additional epochs.", flush=True)
        return
    for epoch in range(start, args.epochs):
        before = time.time()
        lr = set_cosine_lr(optimizer, epoch, args.epochs, args.lr, args.min_lr, args.warmup_epochs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for step, domain in enumerate(schedule(sorted(DOMAINS), args.episodes_per_epoch)):
            support_indices, query_indices = train.sample_episode_indices(domain, args.support_count, args.query_count)
            if set(support_indices) & set(query_indices):
                raise ValueError(f"Overlapping support/query training indices in {domain}")
            xs, ys = load_batch(train, support_indices, device)
            xq, yq = load_batch(train, query_indices, device)
            xs, ys = augmentation.augment_episode_medical(xs, ys[:, None].float(), MODALITY.get(domain, "generic"))
            xq, yq = augmentation.augment_episode_medical(xq, yq[:, None].float(), MODALITY.get(domain, "generic"))
            if not (ys.flatten(1).sum(1) > 0).all():
                raise RuntimeError("Augmentation removed a support foreground")
            window_start = (step // args.grad_accum_steps) * args.grad_accum_steps
            window_size = min(args.grad_accum_steps, args.episodes_per_epoch - window_start)
            with precision(device, amp):
                loss = segmentation_loss(model(xs, ys, xq), yq[:, 0].long())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, episode {step+1}")
            scaler.scale(loss / window_size).backward()
            if (step + 1) % args.grad_accum_steps == 0 or step + 1 == args.episodes_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["max_grad_norm"], error_if_nonfinite=not amp)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            if (step + 1) % 100 == 0:
                print(f"epoch={epoch+1} episode={step+1}/{args.episodes_per_epoch} loss={loss_sum/(step+1):.5f}", flush=True)
        prototypes = build_prototypes(model, train, supports, device)
        validation = evaluate(model, val, prototypes, device, metrics, args.eval_batch_size)
        dice = validation["macro"]["dice"]
        improved = dice > best
        best, bad_epochs = (dice, 0) if improved else (best, bad_epochs + 1)
        history.append(dict(epoch=epoch+1, loss=loss_sum / args.episodes_per_epoch,
            val_dice=dice, val_iou=validation["macro"]["iou"], val_hd95=validation["macro"]["hd95"], lr=lr))
        state = dict(schema=1, epoch=epoch+1, model_config=model.config, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
            best_val_dice=best, bad_epochs=bad_epochs, history=history,
            support_indices=supports, split_fingerprint=split_hash, source_hashes=hashes,
            protocol=protocol, validation=validation,
            prototypes={d: t.cpu() for d, t in prototypes.items()})
        save_checkpoint(output / "checkpoint.last.pth", state)
        if improved:
            save_checkpoint(output / "checkpoint.pth", state)
            write_json(output / "best_validation.json", validation)
        write_json(output / "history.json", history)
        with (output / "checkpoint.pth.history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "lr", "train_loss", "val_macro_dice", "val_macro_iou", "val_macro_hd95"])
            writer.writeheader()
            writer.writerows({"epoch": row["epoch"], "lr": row["lr"], "train_loss": row["loss"],
                "val_macro_dice": row["val_dice"], "val_macro_iou": row["val_iou"], "val_macro_hd95": row["val_hd95"]} for row in history)
        print(f"epoch={epoch+1} lr={lr:.12g} val Dice={dice:.5f} IoU={validation['macro']['iou']:.5f} HD95={validation['macro']['hd95']:.3f} best={best:.5f} time={time.time()-before:.1f}s", flush=True)
        if bad_epochs >= args.patience:
            print("Early stopping on validation macro Dice.", flush=True)
            break


def predict_main():
    parser = argparse.ArgumentParser(description="Evaluate a selected Iris-2D checkpoint; no test-set selection")
    common_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--support-count", type=int, default=None, help="Default: training evaluation count; optionally 1 for a separate one-shot result")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    verify_reference()
    if args.eval_batch_size <= 0 or (args.support_count is not None and args.support_count <= 0):
        parser.error("Batch and support counts must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {output}")
    device = get_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["protocol"]["smoke"]:
        print("WARNING: this is a smoke checkpoint, not a formal baseline.", flush=True)
    if source_hashes() != state["source_hashes"]:
        raise ValueError("Project or baseline sources changed since training")
    data, _, metrics = project_utilities()
    args.seed = state["protocol"]["seed"]
    args.image_size = state["protocol"]["image_size"]
    args.smoke = state["protocol"]["smoke"]
    data.set_seed(args.seed, deterministic=True)
    datasets, manifests = load_datasets(args, data)
    if fingerprint(manifests) != state["split_fingerprint"]:
        raise ValueError("Dataset metadata differs from training; test refused")
    if args.smoke:
        datasets["test"].organ_to_indices = {d: ids[:2] for d, ids in datasets["test"].organ_to_indices.items()}
    model = Iris2D(**state["model_config"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    count = args.support_count or state["protocol"]["eval_support_count"]
    # For unchanged K, use exactly the saved training-support manifest. One-shot
    # is a separate seeded positive-support draw; checkpoint selection stays K=4.
    supports = state["support_indices"] if count == state["protocol"]["eval_support_count"] else select_supports(datasets["train"], count, args.seed)
    prototypes = build_prototypes(model, datasets["train"], supports, device)
    result = evaluate(model, datasets["test"], prototypes, device, metrics,
        args.eval_batch_size, output, args.save_predictions)
    result.update(seed=args.seed, checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_epoch=state["epoch"], eval_support_count=count,
        split_fingerprint=state["split_fingerprint"], source_hashes=state["source_hashes"],
        model_config=state["model_config"], training_protocol=state["protocol"],
        smoke=state["protocol"]["smoke"], metric_size=args.image_size, threshold=.5, postprocess=False,
        support_manifest={d: [manifests["train"][i] for i in ids] for d, ids in supports.items()})
    write_json(output / "metrics.json", result)
    print(json.dumps(result["macro"], indent=2), flush=True)
