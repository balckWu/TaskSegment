from __future__ import annotations

from pathlib import Path

import torch

from tasksegment.checkpoint import load_checkpoint_model
from tasksegment.configs import DEFAULT_MODEL_CONFIG
from tasksegment.models import TaskSegmentModel


def _config(**updates):
    cfg = dict(DEFAULT_MODEL_CONFIG)
    cfg.update(updates)
    return cfg


def test_shared_task_tokens_are_domain_and_support_agnostic():
    model = TaskSegmentModel(
        **_config(
            use_semantic_text=False,
            use_task_specific_representation=False,
            use_task_conditioned_decoder=True,
        )
    ).eval()

    first = model.encode_task(None, None, output_batch_size=2)["task_tokens"]
    second = model.encode_task(
        torch.randn(2, 1, 32, 32),
        torch.ones(2, 1, 32, 32),
        Ftext=torch.randn(2, 5, 1024),
        domain_ids=torch.tensor([0, 5]),
        output_batch_size=2,
    )["task_tokens"]

    assert first is not None
    assert first.shape == (2, 10, 1024)
    assert torch.equal(first, second)
    assert torch.equal(first[0], first[1])


def test_unconditioned_decoder_ignores_task_tokens():
    model = TaskSegmentModel(
        **_config(
            use_semantic_text=False,
            use_task_conditioned_decoder=False,
        )
    ).eval()
    xq = torch.randn(1, 1, 32, 32)
    query = model.encode_query(xq)
    tokens_a = torch.randn(1, 10, 1024)
    tokens_b = torch.randn(1, 10, 1024)

    with torch.inference_mode():
        out_a = model.segment_with_task(query_feats=query, task_tokens=tokens_a)["pred_masks"]
        out_b = model.segment_with_task(query_feats=query, task_tokens=tokens_b)["pred_masks"]
        out_none = model.segment_with_task(query_feats=query, task_tokens=None)["pred_masks"]

    assert torch.equal(out_a, out_b)
    assert torch.equal(out_a, out_none)


def test_grouped_checkpoints_load_without_prototype_bank(tmp_path: Path):
    cases = (
        (
            "no_task_specific_representation",
            _config(
                use_semantic_text=False,
                use_task_specific_representation=False,
                use_task_conditioned_decoder=True,
            ),
        ),
        (
            "no_task_conditioned_decoder",
            _config(
                use_semantic_text=False,
                use_task_specific_representation=True,
                use_task_conditioned_decoder=False,
            ),
        ),
    )

    for ablation, cfg in cases:
        model = TaskSegmentModel(**cfg)
        checkpoint = {
            "schema_version": 5,
            "architecture": "TaskSegment_DomainTextV3",
            "model_state_dict": model.state_dict(),
            "model_config": cfg,
            "prototype_bank": {},
            "train_config": {"ablation_name": ablation},
        }
        path = tmp_path / f"{ablation}.pth"
        torch.save(checkpoint, path)
        restored, bank, restored_cfg, raw = load_checkpoint_model(path, torch.device("cpu"))
        assert restored.ablation_name == ablation
        assert not bank.bank
        assert restored_cfg["use_task_specific_representation"] == cfg["use_task_specific_representation"]
        assert restored_cfg["use_task_conditioned_decoder"] == cfg["use_task_conditioned_decoder"]
        assert raw["schema_version"] == 5
