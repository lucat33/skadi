# Development notes

How the model code in this repository came to be: the steps in the order they were
taken, the decisions behind them and the problems met on the way. The data
exploration that preceded it is in `notebooks/01_explore_data.ipynb`.

## 1. Fixing the target

The Skadi document (Sec. 3.2–3.3) specifies a U-Net with a MobileNetV3-Small encoder,
a 2x512x512 input (HH, HV), three output classes, a weighted cross-entropy + Dice loss
and INT8 quantisation-aware training. The goal added on top is to do better than
TinyIceNet (arXiv:2603.03075).

"Better" only means something if both models are measured in the same way, so the first
step was to read the TinyIceNet paper for its exact protocol:

* AI4Arctic ready-to-train, HH and HV only;
* every scene resized to 512x512 (about 800 m per pixel), for training and for testing;
* six SOD classes, F1 on the 20 official test scenes: 75.168 % (FP32), 75.216 % (8-bit QAT);
* 146.6 k parameters, 2.97 GMACs.

Consequences for the design:

* the benchmark is on six SOD classes, the Skadi product has three. The network
  predicts six and the three are obtained by summing probabilities, so one model
  serves both purposes;
* the cache is built at 512x512 with the same resize, even though a fixed
  downsampling factor would be geometrically cleaner;
* the paper does not say how the F1 is averaged. We use the AutoICE definition
  (weighted by class support), which matches their wording and their comparison with
  MMSeaIce.

## 2. Data pipeline

`skadi/preprocess.py` converts each netCDF scene into one small `.npz`. Details that
needed care:

* SAR channels contain NaN along the swath borders. Averaging them as zeros would
  darken the border pixels, so the area resize divides by the fraction of valid input.
* Labels are resized with nearest neighbour to keep them exactly as charted.
* The test scenes keep their charts in separate `*_reference.nc` files; the loader
  looks for them, and `scripts/download_data.py test` fetches the test article.
* The latitude/longitude tie-point grid has dimension names that suggest (samples,
  lines), but the array is laid out like the image: the 1200x5185 scene opened in the
  exploration notebook comes with a 6x21 grid. It is used as stored, without transposing.
* Longitude is stored as sine and cosine because scenes near the date line would
  otherwise be interpolated across the ±180° jump.

The whole training set is about 1 GB at this size, so it is loaded in RAM and batches
are drawn by indexing. No DataLoader workers are involved, which matters on Windows.

## 3. Models

* `SkadiUNet`: MobileNetV3-Small encoder truncated after the last residual block,
  four decoder stages with skip connections built from depthwise-separable
  convolutions, classifier at half resolution. First version of the decoder used
  widths (96, 64, 48, 32) and cost 0.60 GMACs; since that is five times below the
  baseline, the widths were raised to (128, 96, 64, 48): 942 k parameters, 0.91 GMACs.
* `TinyIceNet`: re-implemented from the text of the paper. The paper says "three
  contracting blocks" but also "9 convolution layers, only the last one 1x1" and lists
  four filter counts (16, 32, 64, 64). Four double-conv blocks with three poolings give
  146,599 parameters with a 7-way head, i.e. the 146.6 k of the paper, and 2.91 GMACs
  against the stated 2.97. A unit test pins these numbers.

## 4. Training recipe

AdamW, warm-up and cosine decay, mixed precision, EMA of the weights, best epoch chosen
on 30 held-out training scenes. Augmentation runs batched on the GPU and includes the
three effects named in the Skadi document: speckle injection, incidence-angle-dependent
intensity scaling and geometric transformations.

## 5. Checking the code without the dataset

The development machine was a CPU-only cloud container without the dataset: the
storage server behind the figshare download links answers on a non-standard port that
was not reachable from there. So `scripts/make_synthetic_data.py` writes fake scenes
with the same variables and file layout as AI4Arctic (including NaN borders, masked
regions and separate reference files for the test scenes), and everything was run on
those. They exist to exercise the code; scores on them mean nothing.

Problems found this way:

* **Crash with very small datasets.** With fewer scenes than the validation size the
  training set was empty. The validation set is now capped at a quarter of the scenes.
* **Low validation F1 despite a low training loss.** On 11 synthetic training scenes
  the validation F1 stayed around 45 % while the loss kept falling. Evaluating the
  checkpoint on the training scenes gave 80 %, with open water almost never predicted:
  in that synthetic set open water was 3 % of the training pixels but 23 % of the
  validation pixels. A data artefact, not a bug — but to exclude the most dangerous
  alternative, a test was added that warps an image equal to its own label map and
  checks that image and labels still coincide after the augmentation.
* **Quantisation ranges updated during validation.** The fake-quantisation observers
  kept collecting statistics in evaluation mode, so validation scenes influenced the
  activation ranges. Observers are now disabled before every validation pass.
* **Learning rate.** The first default (2e-3) gave an unstable validation curve on the
  synthetic runs; the default is now 1e-3. This will need confirming on real data.

## 6. A comparison on synthetic scenes that changed the model

To see whether both architectures train sensibly with the same recipe, each was
trained for 600 iterations (batch 4) on 34 synthetic scenes, validated on 6 and tested
on 6 more. These scores say nothing about real ice — the synthetic classes are just
different mean backscatter values with noise — but the comparison exposed a real flaw.

| Run | Test F1 on the synthetic scenes |
|---|---:|
| TinyIceNet re-implementation | 87.0 |
| Skadi U-Net, first conv = mean of the RGB filters | 65.0 |
| Skadi U-Net, first conv = R filters for HH, G filters for HV | 97.2 |

The first Skadi U-Net was clearly behind the much smaller baseline. The cause was the
way the ImageNet stem had been adapted to two channels: averaging the RGB filters gives
HH and HV *identical* filters, so at the start the network only sees HH + HV, and the
colour-opponent filters, which are the ones that respond to a difference between
channels, average out to almost nothing. Assigning the filters of different colours to
the two polarisations keeps that diversity. Nothing else was changed between the
second and the third row.

Two related changes to the training loop:

* the EMA now averages the BatchNorm statistics together with the weights instead of
  copying them from the live model;
* every epoch both the live weights and the EMA weights are validated and the better
  set is saved, because the average lags behind during the first epochs.

A lesson on procedure: two training processes started side by side on the 4-core
machine ran about twenty times slower than one (each wants all cores). One at a time.

## 7. Quantisation and product generation

`skadi/quantize.py` compares FP32, post-training quantisation and QAT, and exports the
INT8 model (1.27 MB). `skadi/infer.py` turns a scene into OW/FYI/MYI polygons with
confidence and an out-of-distribution score.

The first QAT runs on the synthetic model were disappointing: 97.2 % in FP32 became
73.0 % after conversion. Splitting the problem showed that

* the converted integer model and the fake-quantised one agreed, so the conversion was
  not at fault;
* with float weights and quantised activations the loss was almost the same, so the
  activations were the issue, not the weights;
* PyTorch's default x86 configuration gives activations 7 bits (`reduce_range`), not 8.
  With real 8-bit activations and no retraining at all the score went from 76 to 90.

The script now uses an explicit 8-bit configuration and starts QAT from a calibration
pass (without it the first epoch began at 48 % validation F1, with it at 86 %). After
160 QAT iterations the synthetic model reaches 91.1 % on the test scenes and the
converted model matches the simulated one on validation (96.81 vs 96.84).

## 8. What is still open

* **The training on AI4Arctic has not been run.** `python scripts/run_experiments.py`
  does it and writes `docs/RESULTS.md`. Until then there is no evidence that the model
  beats TinyIceNet.
* Hyper-parameters were set without seeing real validation curves.
* Not implemented: NIC charts as extra training data, Lee filter, CFAR detector, FPGA
  synthesis.
