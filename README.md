# TaskSegment

Integrated source-code repository for the TaskSegment ultrasound segmentation experiments.

## Layout

- `tasksegment/`: main model, training and inference code
- `scripts/`: training, prediction, baseline and audit scripts
- `tools/`: visualization/export helpers
- `configs/`: experiment configuration
- `third_party/`: baseline source code retained from the earlier TSV3 tree

## Main ablations

The integrated training/evaluation code supports:

- `full`
- `no_text_conditioning`
- `no_visual_prototype`
- `no_dense_guidance`
- `no_task_specific_representation`
- `no_task_conditioned_decoder`

Use `python scripts/train.py --help` and `python scripts/predict.py --help` for the exact current arguments.

Datasets, checkpoints, predictions, paper artifacts and logs are intentionally excluded from this repository.
