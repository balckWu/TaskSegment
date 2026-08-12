import numpy as np
import torch

from tasksegment.checkpoint import migrate_model_config
from tasksegment.configs import DEFAULT_LOSS_CONFIG
from tasksegment.models.decoder import IrisMaskDecoder2D
from tasksegment.models.task_encoder import TaskEncoder2D
from tasksegment.training.evaluation import postprocess_mask
from tasksegment.training.losses import (
    deep_supervision_loss,
    dense_guidance_loss,
    focal_tversky_loss,
    segmentation_loss,
    soft_dice_loss,
)
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou


def test_task_encoder_uses_five_text_summaries():
    encoder = TaskEncoder2D(
        hidden_dim=16,
        num_query_tokens=4,
        context_upscale=4,
        num_heads=4,
        text_dim=16,
        text_group_summary_count=5,
        domain_names=("a", "b"),
        group_norm_groups=4,
    ).eval()
    support = torch.randn(2, 16, 2, 2)
    mask = (torch.rand(2, 1, 32, 32) > 0.8).float()
    text = torch.randn(2, 5, 16)
    with torch.inference_mode():
        visual = encoder.forward_visual(support, mask)
        domain_ids = encoder.domain_ids_from_names(["a", "b"])
        task = encoder.compose_task_tokens(visual, text, domain_ids=domain_ids)
    assert visual.shape == (2, 5, 16)
    assert task.shape == (2, 10, 16)



def test_domain_embedding_changes_condition_tokens():
    encoder = TaskEncoder2D(
        hidden_dim=16,
        num_query_tokens=4,
        context_upscale=4,
        num_heads=4,
        text_dim=16,
        text_group_summary_count=5,
        domain_names=("a", "b"),
        group_norm_groups=4,
    ).eval()
    visual = torch.randn(2, 5, 16)
    shared_text = torch.randn(1, 5, 16).expand(2, -1, -1)
    ids = encoder.domain_ids_from_names(["a", "b"])
    with torch.inference_mode():
        task = encoder.compose_task_tokens(visual, shared_text, domain_ids=ids)
    condition = task[:, 5:]
    assert not torch.allclose(condition[0], condition[1])

def test_small_decoder_returns_three_aux_heads():
    decoder = IrisMaskDecoder2D(
        hidden_dim=16,
        num_heads=4,
        num_query_tokens=4,
        encoder_filters=(4, 8, 16, 16, 16),
        dense_guidance_dim=8,
        group_norm_groups=4,
        text_group_summary_count=5,
        use_deep_supervision=True,
    ).train()
    pyramid = {
        "bottleneck": torch.randn(1, 16, 1, 1),
        "skip4": torch.randn(1, 16, 2, 2),
        "skip3": torch.randn(1, 16, 4, 4),
        "skip2": torch.randn(1, 8, 8, 8),
        "skip1": torch.randn(1, 4, 16, 16),
    }
    task = torch.randn(1, 10, 16)
    out = decoder(pyramid, task)
    assert out["logits"].shape == (1, 2, 16, 16)
    assert len(out["aux_logits"]) == 3


def test_model_config_migration_removes_dicev1_text_options():
    cfg = migrate_model_config(
        {
            "encoder_batchnorm": False,
            "text_token_mode": "all",
            "max_text_tokens": 100,
            "dense_guidance_strength": 0.75,
        }
    )
    assert cfg["encoder_norm_type"] == "none"
    assert "text_token_mode" not in cfg
    assert "max_text_tokens" not in cfg
    assert cfg["dense_guidance_strength"] == 0.75


def test_dice_oriented_losses_are_finite():
    logits = torch.randn(2, 2, 32, 32, requires_grad=True)
    target = (torch.rand(2, 32, 32) > 0.7).long()
    main = segmentation_loss(logits, target, DEFAULT_LOSS_CONFIG)
    assert torch.isfinite(main)
    assert torch.isfinite(soft_dice_loss(logits, target))
    assert torch.isfinite(focal_tversky_loss(logits, target))
    main.backward()
    assert logits.grad is not None


def test_aux_and_dense_losses():
    target = (torch.rand(2, 32, 32) > 0.7).long()
    aux = [
        torch.randn(2, 2, 4, 4),
        torch.randn(2, 2, 8, 8),
        torch.randn(2, 2, 16, 16),
    ]
    value = deep_supervision_loss(aux, target, DEFAULT_LOSS_CONFIG)
    assert torch.isfinite(value)
    dense = dense_guidance_loss(
        {"dense_guidance_logits_lowres": torch.randn(2, 1, 4, 4)},
        target.unsqueeze(1).float(),
    )
    assert torch.isfinite(dense)


def test_overlap_metrics_and_empty_cases():
    empty = np.zeros((16, 16), dtype=np.uint8)
    one = empty.copy()
    one[4:8, 4:8] = 1
    assert binary_dice(empty, empty) == 1.0
    assert binary_iou(empty, empty) == 1.0
    assert binary_dice(empty, one) == 0.0
    assert binary_hd95(empty, empty) == 0.0
    assert binary_hd95(empty, one) > 0.0


def test_postprocess_min_area_is_effective():
    pred = np.zeros((32, 32), dtype=np.uint8)
    pred[2:4, 2:4] = 1
    pred[10:20, 10:20] = 1
    out = postprocess_mask(pred, min_area=10)
    assert out[2:4, 2:4].sum() == 0
    assert out[10:20, 10:20].sum() == 100
