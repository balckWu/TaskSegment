# Reproducing the KBS manuscript

## Identity of the reported code

The paper calls the full model **US-TextSeg**. Historical code and checkpoint
classes still use `TaskSegment`; renaming those would break compatibility.

| Paper row | Historical source/run | Release entry point |
|---|---|---|
| US-TextSeg | TSV3 / `tasksegment_full2` | `scripts/train.py --ablation full` |
| w/o Text Conditioning | TSV4 / `tasksegment_no_text_conditioning` | `--ablation no_text_conditioning` |
| w/o Support-derived Visual Tokens | TSV4 / `tasksegment_no_visual_prototype` | `--ablation no_visual_prototype` |
| w/o Dense Guidance | TSV4 / `tasksegment_no_dense_guidance_nobank` | `--ablation no_dense_guidance` |
| w/o Task-conditioned Decoder | TSV5 / `tasksegment_no_task_conditioned_decoder` | `--ablation no_task_conditioned_decoder` |
| Iris-2D | TSV3 / `iris2d` | `scripts/launch_iris2d.py` |

`tasksegment_full2` was trained under the older `no_prototype_bank_blending`
experiment name. Its saved checkpoint explicitly records blending disabled,
K=4, five text groups and the loss/settings in the manuscript. Release `full`
retains that formulation. The integrated decoder also supports the later TSV5
ablations; compatibility is checked against the original checkpoints rather than
inferred from directory names. `docs/server_source_map.json` records the audit.

## Data and text

Arrange each dataset as follows, supplying the same image-level splits used by
the experiments. Masks use the corresponding image stem.

```text
data/<domain>/<train|val|test>/images/
data/<domain>/<train|val|test>/masks/
```

| Domain key | Train | Validation | Test |
|---|---:|---:|---:|
| thyroid | 855 | 285 | 286 |
| TN3K | 2447 | 432 | 614 |
| BUSI_WHU | 555 | 186 | 186 |
| BUS-BRA | 1312 | 280 | 283 |
| OTU | 656 | 164 | 382 |
| prostate | 67 | 14 | 16 |

Obtain public datasets from their original providers as cited in the manuscript.
The thyroid and prostate datasets are private and are not distributed here.
Folder counts alone do not establish patient independence. Without the authorized
private data, the complete six-domain evaluation cannot be independently rerun.

Generate the five structured text groups from the supplied `text_prompts.json`:

```bash
python scripts/generate_text_features.py --device cpu \
  --prompt-json text_prompts.json --output-dir text_features_llm
python scripts/audit_text_features.py --help
```

The encoder is `dmis-lab/biobert-large-cased-v1.1`. Generation averages the last
four hidden layers, excludes special tokens and pools sentences within each group.
The common direction is removed by the text-bank loader, not during generation.
For offline execution, supply a local encoder path with `--model` and
`--local-files-only`. Text-encoder assets are not included.

## US-TextSeg and ablations

Each task uses one GPU. A direct seed-42 command is:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --data-root ./data --text-dir ./text_features_llm \
  --save-path paper_artifacts_kbs/runs/tasksegment_full/seed_42/checkpoint.pth \
  --seed 42 --deterministic --ablation full \
  --epochs 150 --batch-size 4 --episodes-per-epoch 1500 \
  --grad-accum-steps 4 --lr 5e-5 --min-lr 1e-6 \
  --weight-decay 1e-4 --warmup-epochs 5 --patience 20 \
  --prototype-eval-support-samples 4 --cache-dataset --preload-val-cache --amp

CUDA_VISIBLE_DEVICES=0 python -u scripts/predict.py \
  --model-path paper_artifacts_kbs/runs/tasksegment_full/seed_42/checkpoint.pth \
  --data-root ./data --text-dir ./text_features_llm \
  --save-dir paper_artifacts_kbs/runs/tasksegment_full/seed_42/predictions \
  --seed 42 --device cuda:0 --threshold 0.5 --num-vis-per-organ 0
```

Repeat with seeds 3407 and 2026 and distinct output directories. Select the
checkpoint by validation domain-macro Dice. Support manifests use positive
training examples only, remain fixed per seed, and their tokens are recomputed
for the applicable checkpoint. Inference reads this checkpoint's saved support
state; do not build supports from test masks.

Replace `full` with the four ablation flags in the mapping above and give each
variant its own output path. The text ablation removes semantic, domain and group
signals together. The decoder ablation also removes dense guidance, so their
effects are not independent additive contributions. Optional historical variants
such as `with_prototype_bank` and `no_semantic_text` are not main-paper rows.

The single-run CLI defaults now use batch size 4 and five warmup epochs. Older
CLI defaults of 2 and 3 did not describe the completed runs, whose launchers
explicitly passed 4 and 5. No completed result was replaced or retrained.

For managed scheduling, edit dataset/text/asset paths in `configs/kbs_3seeds.json`:

```bash
python scripts/paper.py plan --config configs/kbs_3seeds.json \
  --experiments tasksegment_full --seeds 42 --show-commands
python scripts/paper.py launch --config configs/kbs_3seeds.json \
  --experiments tasksegment_full --seeds 42
```

The recipe schedules serially on GPU 0 by default. Configure other GPU IDs if
needed. Each process still uses one card. Existing training/inference scripts and
the internal state-dict names are retained.

## Baselines

The same recipe includes U-Net, U-Net++, Swin-Unet, TransUNet, CLIPSeg, LViT and
SAM-Med2D. Their existing setup scripts and local pretrained paths remain required:
TransUNet uses `R50+ViT-B_16.npz`, CLIPSeg its pretrained model, LViT its text
encoder/repository, and SAM-Med2D its repository/checkpoint. LViT trains at its
native 224 resolution; evaluation logits are resized to 512. Do not interpret
these differing pretraining, architecture and loss choices as controlled text-only
comparisons. Use the environments named in the recipe, or adapt environment names
to your installation without changing the experimental settings.

Iris-2D has a separate, source-verified launcher:

```bash
python scripts/preflight_iris2d.py
python scripts/check_iris2d.py
python scripts/launch_iris2d.py --gpu 0 --dry-run
python scripts/launch_iris2d.py --gpu 0 --data-root ./data
```

It uses one training support, four queries and four fixed evaluation supports;
AdamW lr=1e-4, five warmup epochs, 1500 episodes/epoch, four-episode accumulation,
150 epochs maximum and patience 20. CE and Dice are equally weighted. Do not
substitute the US-TextSeg loss. The source files, reference JSON and the
controlled protocol fingerprint are preserved exactly from its completed
checkpoint. Changes to these inputs require reviewing/retraining the experiment;
do not disable provenance checks to reuse a mismatched checkpoint.

## Results and checks

All metrics use a 512x512 grid, threshold 0.5 and no morphology. HD95 is in
resized-image pixels. Empty masks and boundary-distance definitions follow
`tasksegment/training/metrics.py`. Average images within each domain; average
six domains equally within each seed; finally calculate the three-seed mean
and sample SD (n-1). Never convert mean Dice to mean IoU algebraically.

```bash
python scripts/kbs_results.py
python scripts/kbs_results.py --run-roots \
  /path/TSV3/paper_artifacts/runs /path/TSV4/paper_artifacts/runs \
  /path/TSV5/paper_artifacts/runs --output outputs/kbs_summary.json
```

The first command checks the identifier-free archive. The second reads actual
completed run directories, including nested baseline summaries and Iris JSON,
rejects missing/duplicate seeds or domains, and verifies the manuscript values.
For a new public-recipe rerun, use its `paper_artifacts_kbs/runs` plus the Iris
run directory's parent. Historical and released full/dense-ablation aliases are
recognized; providing two copies of the same run fails instead of choosing one.

The October 19 audit verified the archived domain summaries, not a fresh inference
on all 1,767 test images. The result archive contains no patient filenames,
images, masks, raw support manifests, clinical predictions or credentials.

Run the numerical/protocol checks in a suitable Torch environment:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
python scripts/check_iris2d.py
```

The validation record distinguishes unit/synthetic checks, strict historical
checkpoint loading, and archived numerical verification from a full re-evaluation.
