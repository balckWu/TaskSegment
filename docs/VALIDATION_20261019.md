# KBS release validation - 2026-10-19

## Environment and scope

Validation ran on the experiment server with Python 3.10.20, PyTorch 2.7.1+cu118
and pytest 9.1.1. Model checks used CPU, with GPUs hidden from the test process.
This is the server's current audit environment; it is not evidence that the
historical training environment reported in the manuscript used this Torch version.

No experiments were retrained, no running GPU job was changed, and no private
image, annotation, prediction, checkpoint or support manifest was published.

## Archived numerical results

- Nine main-comparison methods and four reported ablations.
- Three seeds (42, 3407, 2026), six domains: 39 runs / 234 domain records.
- 402 mean/SD values checked against the v8 manuscript reference.
- Existing values agree to manuscript precision, including the Iris-2D values
  originally supplied to the manuscript at six-decimal precision.
- All expected domain test counts match; duplicate/missing seed-domain records
  are rejected. Macro statistics use equal domain weights and sample SD.

These checks read the actual archived server summaries. They do not recompute
metrics from 1,767 test images or constitute a new significance analysis.

## Historical checkpoint compatibility

The released implementation strictly loaded the actual seed-42 checkpoint for
US-TextSeg, each of its four reported ablations, and Iris-2D. For each checkpoint,
the historical source tree and release source were run in separate processes
on the same fixed synthetic 64x64 inputs. Their output tensors were bitwise equal.

| Model | Historical tree | Strict load | Synthetic outputs |
|---|---|---|---|
| US-TextSeg (`tasksegment_full2`) | TSV3 | passed | bitwise equal |
| w/o Text Conditioning | TSV4 | passed | bitwise equal |
| w/o Support-derived Visual Tokens | TSV4 | passed | bitwise equal |
| w/o Dense Guidance | TSV4 | passed | bitwise equal |
| w/o Task-conditioned Decoder | TSV5 | passed | bitwise equal |
| Iris-2D | TSV3 | passed | bitwise equal |

These probes check checkpoint topology and the exercised forward paths. They
do not establish bitwise equality for full-resolution CUDA training or constitute
a fresh clinical/test-set evaluation. The source and controlled-protocol
fingerprints saved in Iris-2D's seed-42 checkpoint match the release exactly.

## Automated checks

- 28 pytest checks passed: model/loss/metrics, support-independent grouped ablations, checkpoint roundtrip,
  command generation, scheduler, database and aggregate-result tests.
- Five additional Iris numerical checks, including support pooling gradients,
  support averaging, checkpoint roundtrip, and learning-rate equality with the
  four image baselines.
- Iris source/protocol preflight passed; launcher dry-run passed.
- The portable KBS plan generates all twelve paperlab experiments across the
  three seeds (36 jobs); the separate Iris launcher supplies the remaining 3 runs.

The old scheduler tests referred to superseded semantic-only/bank ablations and
fixed GPU IDs. They now validate the actual KBS ablation set and portable recipe.
The new result tests explicitly check domain balancing, n-1 SD, invalid inputs,
both historical CSV count-column formats and Iris's K=4 evaluation budget.

An empty, unused `tasksegment/models/paper_code` gitlink without a corresponding
`.gitmodules` entry was removed from the release index. No model imports that path;
the actual model implementations are ordinary files in `tasksegment/models/`.
