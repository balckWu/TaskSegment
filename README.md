# US-TextSeg

Implementation for **US-TextSeg: Structured Text-Guided Multi-Domain Ultrasound
Segmentation with Support-Derived Visual Prototypes**, prepared for
Knowledge-Based Systems (KBS).

The repository URL and internal `TaskSegment` Python/checkpoint names are retained
for compatibility with the completed experiments. The paper's method name is
**US-TextSeg**. The manuscript release is `kbs-v9-20261019`.

## What this release contains

- US-TextSeg and its four reported ablations, with episodic training and fixed
  **K=4 positive training supports** per domain for validation/testing.
- The completed **Iris-2D** baseline, copied from the server implementation and
  checked against the source fingerprints saved in its seed-42 checkpoint.
- Existing U-Net, U-Net++, Swin-Unet, TransUNet, CLIPSeg, LViT and SAM-Med2D
  training/evaluation entry points.
- A portable single-GPU recipe, aggregate results for three seeds, and a
  standard-library script that reproduces the numerical tables.

This is closed-set segmentation across six predefined domains. No test mask is
used as a prompt, support example or model-selection input. No cross-epoch
prototype-bank blending is used for the reported full model. Iris-2D is an
independent 2D adaptation, not the original Iris authors' volumetric implementation.

## Start here

Use Python 3.10 and install the dependencies in an isolated environment:

```bash
pip install -r requirements.txt
```

The manuscript reports PyTorch 2.5.1 / CUDA 12.1 for the experiments. The release
audit used the server's current Python 3.10.20 / PyTorch 2.7.1+cu118 environment;
see the validation record for the exact checks and their limits. Optional baseline
dependencies and pretrained assets have separate setup scripts in `scripts/`.

Verify the archived results **without PyTorch or datasets**:

```bash
python scripts/kbs_results.py
```

It validates 39 completed runs, 234 domain records and 402 manuscript reference
values. Macro metrics average the six domains equally within each seed, then use
the sample standard deviation across seeds. Iris values originally supplied to the
manuscript at six-decimal precision are checked at that precision.

Read **[the reproduction guide](docs/PAPER_REPRODUCTION.md)** for dataset layout,
text features, training, testing, baselines, historical run names and source mapping.

Preview a US-TextSeg run without starting training:

```bash
python scripts/paper.py plan --config configs/kbs_3seeds.json \
  --experiments tasksegment_full --seeds 42 --show-commands
```

The new recipe writes to `paper_artifacts_kbs/`, separate from historical outputs.
`configs/paper_3seeds.json` retains the historical protocol used by Iris's strict
fingerprint checks; use `configs/kbs_3seeds.json` for the portable paperlab plan.

## Repository layout

| Path | Purpose |
|---|---|
| `tasksegment/` | US-TextSeg model, support protocol, losses and evaluation |
| `iris2d/` | Server-verified independent Iris-2D implementation |
| `scripts/` | Training, inference, result verification and setup |
| `configs/kbs_3seeds.json` | Single-GPU KBS recipe for eight main methods and four ablations |
| `configs/iris2d_*.json` | Iris recipe and source/protocol verification |
| `paper_results/` | Domain-level aggregates only; no patient identifiers |
| `docs/` | Reproduction guide and server-source/validation records |
| `tests/` | Model, protocol and result aggregation checks |
| `third_party/` | Existing third-party baseline code; retain its notices |

Datasets, private clinical images, patient manifests, weights, predictions,
credentials and training logs are excluded. Access to the two institutional
datasets remains subject to the manuscript's ethics and data-use restrictions.
The public aggregate results are archived evaluations; this release does not
claim a fresh full-dataset evaluation or new statistical significance.
