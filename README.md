**English** | [日本語](README.ja.md)

# tatara

![tatara — forging shogi NNUE in Rust](docs/tatara-hero.jpg)

**A fast Rust trainer for shogi NNUE evaluation networks.**

tatara trains shogi NNUE (Efficiently Updatable Neural Network) evaluation
networks on NVIDIA GPUs. The trainer, data pipeline, and CUDA Driver API runtime
are written in Rust, while the native GPU backend compiles hand-fused CUDA C++
kernels with NVCC into an embedded fat binary.

The default `native` feature uses only the NVCC-built kernels and
portable Rust host runtime. The
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) Rust → PTX backend remains
available as an opt-in numerical and performance oracle. Keeping both paths
preserves the GPU/CPU equivalence-test coverage used to validate the native
backend. The `oxide-parity` feature enables parity and benchmark comparisons
across both device backends; invoke it with
`--no-default-features --features oxide-parity`. It builds the NVCC fat binary
only, so generate the cuda-oxide PTX first with
`bash scripts/build-kernels.sh` (see
[docs/native-cuda-benchmark.md](docs/native-cuda-benchmark.md)).

Hand-fusing the GPU kernels makes it **very fast** — the measured cuda-oxide
backend out-throughputs its upstream CUDA C++ trainer
[bullet-shogi](https://github.com/SH11235/bullet-shogi), while the native backend
keeps the same fused training design.

**cuda-oxide vs bullet-shogi (measured on RTX 3080 Ti)**: for LayerStack, even the
bit-identical default path is **+37%**, and stacking the opt-in FP16 modes reaches
up to **~2.1×**. For Simple (HalfKP `512x2-8-64`) it is around **+20%** on the
default and around **+55%** with `--all-optim` (FP16/TF32).

**Throughput** (`--batch-size 65536`, pos/s; fp32 → `--all-optim` (fp16/tf32)). The
`--all-optim` gain is larger on older, more bandwidth-bound GPUs and on bigger nets:

| arch / config (feature) | RTX 5090 | RTX 3080 Ti | RTX 4070 Ti |
|---|---|---|---|
| LayerStack `--ft-out 1536` (`halfka-hm-merged`) | 2.45M → 3.13M (+28%) | 0.99M → 1.59M (+61%) | 1.26M → 2.14M (+70%) |
| LayerStack `--ft-out 768` (`halfka-hm-merged`) | 4.24M → 5.10M (+20%) | 2.05M → 2.97M (+45%) | 2.69M → 4.14M (+54%) |
| Simple `256x2-32-32` (`halfkp`) | 11.29M → 13.37M (+18%) | 7.30M → 10.10M (+38%) | 5.79M → 7.85M (+35%) |
| Simple `256x2-32-32` (`kp`) | – | – | 6.69M → 6.15M (−8%) |

The RTX 4070 Ti (12 GB) column is the mean of two runs of the command below with the
teacher file in the page cache and `--superbatches 3` (`--batches-per-superbatch 60` for
LayerStack, 150 for Simple); the two runs differ by at most 0.2%. For `kp`, whose
feature transformer has only 1,710 rows, `--all-optim` was slower than fp32. `–` means not
measured.

Each position is 40 bytes in the teacher file, so training at 6M positions/s reads about
240 MB/s, and storage that reads more slowly sets a lower limit: 100 MB/s allows about
2.5M positions/s. A hard disk may not keep up; an SSD, fast network storage, or a teacher
file that fits in the page cache does.

Measurement command (teacher-data / progress paths are environment-specific;
compare pos/s with and without `--all-optim`):

```sh
# LayerStack (halfka-hm-merged, needs progress)
cargo run --release --bin nnue-train -- \
  --data /path/to/teacher.psv \
  --feature-set halfka-hm-merged --batch-size 65536 --superbatches 8 [--all-optim] \
  layerstack --progress-coeff /path/to/progress.bin --ft-out {1536|768} --l1 {16|8} --l2 32

# HalfKP (simple, no progress needed; the simple trainer requires --win-rate-model)
cargo run --release --bin nnue-train -- \
  --data /path/to/teacher.psv \
  --feature-set halfkp --batch-size 65536 --superbatches 8 [--all-optim] \
  --scale 290 --win-rate-model \
  --wrm-in-offset 0 --wrm-target-offset 0 \
  --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
  simple --arch 256x2-32-32
```

The simple trainer only supports the win-rate-model loss (its int8 output
layer cannot represent the centipawn-scale output that the plain sigmoid loss
converges to). The `--wrm-*` values above degenerate the WRM to a plain
sigmoid, so this configuration uses the same value for `--scale`, every
`--wrm-*-scaling`, and `--wrm-nnue2score`. When `simple --fv-scale` is omitted,
`--scale` must equal `--wrm-nnue2score` whether or not weights are loaded. The
exported `fv_scale` comes from a v9+ `--resume` checkpoint when available, then
from the `--init-from` input net if it carries one, and otherwise from
`round(127 × QB / --scale)`. Set `simple --fv-scale <N>` to always write `N` and
waive that equality requirement. A v9+ checkpoint preserves the value without
respecifying it; when resuming a v8 or earlier checkpoint, repeat an override or
the value falls back to the `--scale`-derived value.

The derived value is a nominal scale based on the training labels, not
necessarily the playing-strength optimum. Determine the optimum through games
and bake it in with `--fv-scale`, or set the engine-side option.

*tatara (踏鞴), the traditional Japanese furnace that smelts iron sand (raw
material) into tamahagane steel — forging a net out of raw data.*

> **NVIDIA only** — both GPU backends target NVIDIA CUDA; ROCm / AMD is out of
> scope. To train comparable shogi NNUE nets on an AMD GPU, see the upstream
> [bullet-shogi](https://github.com/SH11235/bullet-shogi), which has both CUDA
> and HIP backends.

## What you can train

A trained NNUE is defined by two independent choices: the **architecture**
(a subcommand), which fixes the network structure, and the **input feature set**
(`--feature-set`), which fixes how a board position is turned into an input
vector.

### Architecture

| Architecture | Subcommand | Structure |
|---|---|---|
| **LayerStack** | `layerstack` | Specializes the output layer per position bucket. `--bucket-mode progresskpabs` uses progress coefficients and 2–9 buckets (`progresskpabsq16` is the same routing with integer Q16 arithmetic; `progresskpabsq16ek` adds a ninth bucket for positions where both kings have entered the middle of the board); `--bucket-mode kingrank9` uses YaneuraOu KingRank9 with exactly 9 buckets and no progress coefficients. FT output `--ft-out` (default 1536) → `--l1` (default 16) → `--l2` (default 32) |
| **Simple** | `simple` | A plain NNUE with no bucket split (FT → 2 hidden layers → single output). Layer dimensions are set with `--arch <l1>x2-<l2>-<l3>` (`l1` = FT output, `l2`/`l3` = hidden layers; default `256x2-32-32`); activation crelu / screlu / pairwise |

### Input feature set

`--feature-set` selects one of six (default `halfka-hm-merged`). They differ in
how the kings are included as features:

| `--feature-set` | King handling |
|---|---|
| `halfkp` | Kings themselves are not included as piece features |
| `halfka-split` | Kings are included; own-king and enemy-king features have separate slots |
| `halfka-merged` | Kings are included; own-king and enemy-king features share one slot |
| `halfka-hm-split` | `halfka-split` plus a left-right mirror so the king always sits on one side of the board, compressing king squares 81 → 45 |
| `halfka-hm-merged` (default) | `halfka-merged` plus the same left-right mirror king-square compression |
| `kp` | The king square does not select a weight block: pieces (kings excluded) and both kings are independent features, 1710 inputs in total. The `K+P` feature list that YaneuraOu's KP256 uses |

The default `halfka-hm-merged` applies to shogi the same design as Stockfish's
**HalfKAv2_hm** (left-right king-square mirroring + own-king/enemy-king features
sharing one slot).

A separate binary, `progress-kpabs-train`, is a KP-abs progress trainer that
produces `progress.bin`, the bucket coefficients for LayerStack. The approach of
learning game progress and assigning it to output buckets is based on an idea
from [a post by nodchip](https://nodchip.hatenablog.com/entry/2026/02/04/000000).

### Networks YaneuraOu loads

A `--feature-set` alone does not name a network: the width of the feature
transformer and the two hidden layers are part of it, and a YaneuraOu build is
compiled for exactly one such network. These four `simple` networks are
written in the file layout of the YaneuraOu evaluation file of the same name,
so a build of the listed edition loads the `.bin` without conversion. All four
use the default `--activation crelu`.

| Network | `--feature-set` | `simple --arch` | YaneuraOu `YANEURAOU_EDITION` |
|---|---|---|---|
| `halfkp_128x2_32_32` | `halfkp` | `128x2-32-32` | `YANEURAOU_ENGINE_NNUE_halfkp_128x2_32_32` |
| `halfkp_256x2_32_32` | `halfkp` | `256x2-32-32` | `YANEURAOU_ENGINE_NNUE` |
| `halfkp_768x2_16_64` | `halfkp` | `768x2-16-64` | `YANEURAOU_ENGINE_NNUE_halfkp_768x2_16_64` |
| `kp_256x2_32_32` | `kp` | `256x2-32-32` | `YANEURAOU_ENGINE_NNUE_KP256` |

The names follow the network naming of the
[shielune/YaneuraOu](https://github.com/shielune/YaneuraOu) fork
(`NNUE_<feature>_<FT>x2_<H1>_<H2>`, described in its
`source/eval/nnue/architectures/README.md`), whose Makefile generates the
architecture header from the edition name. An engine built for one network
refuses a `.bin` of another with `NNUE hash mismatch` and `failed to read nn.bin`.

`--init-from` also reads `.bin` files that other trainers wrote for one of
these networks. A file is accepted when all of these hold:

- its version word, network hash, FT hash and FC hash equal the ones computed
  for the `--feature-set`, `simple --arch` and `--activation` you pass;
- the data after the architecture string is exactly the length that network
  needs, with nothing missing and nothing left over;
- the architecture string does not name a different activation. The hashes do
  not encode it, so a string with `SqrClippedReLU` is refused unless
  `--activation screlu`, and one with a `/2x2]-Pairwise` feature block unless
  `--activation pairwise`.

The architecture string is otherwise not what identifies the network. It may
differ from tatara's in a `ModelType=` prefix, in `AffineTransform` where tatara
writes `AffineTransformSparseInput`, and in lacking `,fv_scale=`; these are
accepted silently. Any other difference, such as a string that gives the wrong
layer widths, is accepted with a warning on stderr that shows the two texts.
A file with a different feature set or layer widths has different hashes and is
refused.

A file without `,fv_scale=` does not supply an `fv_scale`: the export then uses
the same value as a run without `--init-from`, `round(127 × QB / --scale)`
unless `simple --fv-scale` is given, and the startup log line `fv_scale:` says
so.

## Setup

### Requirements

- **OS** — Linux is first-class; Windows is supported via WSL2, with experimental
  native Windows support through `native`; macOS cannot build the GPU
  crates
- **NVIDIA GPU** (see the backend-specific support matrix in `docs/setup.md`)
- **CUDA Toolkit 12.x** (verified with 12.9)
- **NVCC** for the default `native` backend; **LLVM 21+** and
  `cargo-oxide` for the opt-in cuda-oxide oracle
- **Rust nightly** (pinned in `rust-toolchain.toml`)

For the native CUDA C++ build command, cuda-oxide setup, detailed per-OS
instructions, and the supported-GPU matrix, see [docs/setup.md](docs/setup.md).

### Build and train

For building the kernels and running the smoke test, see
[docs/setup.md](docs/setup.md); for how to run training, see
[docs/training-quickstart.md](docs/training-quickstart.md). To skip the build and
train in a container, see "Train with Docker" below.

### Train with Docker

The image on GitHub Container Registry holds `nnue-train` and the other tools of this
workspace, so a machine with an NVIDIA driver that supports CUDA 12.9 and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
can train without a toolchain.

```bash
docker pull ghcr.io/rshogi/tatara:latest

# GPU smoke test, no data needed; ends with "[smoke/simple] PASSED"
docker run --rm --gpus all ghcr.io/rshogi/tatara:latest simple
```

Everything after the image name goes to `nnue-train`. Mount the teacher data
read-only and an output directory that outlives the container, and give both
by their paths inside the container. `--data` takes one PSV file, so concatenate
shards first (`cat shard-*.psv > train.psv`). This trains the `kp_256x2_32_32`
network; for another one, change `--feature-set` and `--arch` as in
"Networks YaneuraOu loads" above:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v /path/to/data:/data:ro -v /path/to/out:/out \
  ghcr.io/rshogi/tatara:latest \
  --data /data/train.psv --test-data /data/val.psv \
  --output /out --net-id kp256 \
  --feature-set kp --superbatches <N> --batch-size 65536 --threads <N> \
  --scale 290 --win-rate-model \
  --wrm-in-offset 0 --wrm-target-offset 0 \
  --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
  simple --arch 256x2-32-32
```

- `--test-data` is the held-out file whose loss is printed after every superbatch.
  Leave it out to train without validation.
- `--threads` is the number of data-loading threads. Always set it: the data loader is
  usually the bottleneck, and the number of physical cores is a good starting point.
- A superbatch is `--batches-per-superbatch` batches (default 6104), about 400 million
  positions at `--batch-size 65536`.
- Every `--save-rate` superbatches (default 20) and at the end, the output directory
  gets `<net-id>-<superbatch>.bin` (the quantised net) and `<net-id>-<superbatch>.ckpt`
  (resume state).
- The startup log prints `fv_scale: 28` for `--scale 290`. YaneuraOu does not read
  `fv_scale` from the file; give the same number to the engine as the USI option
  `FV_SCALE`.
- The other tools are in the same image: `docker run --rm --entrypoint net_to_yo
  ghcr.io/rshogi/tatara:latest --help`.

Choosing GPUs, other GPU generations, and building the image yourself are covered under
"Docker image" in [docs/setup.md](docs/setup.md#docker-image); the training options are in
[docs/training-quickstart.md](docs/training-quickstart.md).

## Documentation

- [Dense QAT continuation](docs/dense-qat.md): compare ordinary and quantization-aware tails from the same raw checkpoint.

- [Setup guide](docs/setup.md) — per-OS guidance, native CUDA / cuda-oxide build
  setup, supported-GPU matrix, CUDA toolkit root resolution
- [Training quickstart](docs/training-quickstart.md) — per-architecture training
  examples + key CLI options + resume / checkpoint workflow
- [Game-progress buckets: preparing `progress.bin`](docs/progress-bin.md) —
  training the LayerStack bucket coefficients and surveying the bucket split
- [End-to-end training benchmark](docs/bench-pos.md) — reproducible real-data
  throughput measurement on Linux/WSL and native Windows
- [Held-out validation](docs/held-out-validation.md) — `test_loss` / `test_acc`
  setup, choosing the held-out source, and reading the metrics
- [FT precision diagnostics](docs/precision-diagnostics.md) — sampled moment
  storage errors and master / forward-copy changes at selected training steps
- [Training schedules](docs/training-schedule.md) — scheduling the learning rate
  (`--lr-schedule`) and the WDL lambda (`--wdl` / `--start-wdl` / `--end-wdl`)
  across a run
- [Fused kernel catalog](docs/kernels/fused-pattern-catalog.md) — which kernel
  does what
- [Arch string](docs/arch-string.md) — how the architecture-description string
  embedded in the quantised `.bin` header is assembled and checked at load time
  (Japanese only)
- [Converting between tatara and YaneuraOu LayerStack nets](docs/net-to-yaneuraou.md) —
  `net_to_yo` / `net_from_yo` usage, supported architecture, and binary layout (Japanese only)

## Using the trained net

The quantised `.bin` that tatara produces is designed to be loaded by the
[rshogi](https://github.com/SH11235/rshogi) engine. The `.bin` header and the
SCReLU / Pairwise activations are specific to this project, so other shogi
engines such as YaneuraOu cannot necessarily load it as-is; depending on the
architecture, some nets need additional inference code before they can be
loaded. A supported LayerStack net can be converted to and from the YaneuraOu
format with [`net_to_yo` / `net_from_yo`](docs/net-to-yaneuraou.md).
The `simple` networks listed under "Networks YaneuraOu loads" are loaded by a
YaneuraOu build of the matching edition without conversion. YaneuraOu takes the
output scale from its `FV_SCALE` option, not from the file.
Pre-trained reference nets are attached to the
[GitHub Releases](https://github.com/SH11235/tatara/releases). To train your
own net, see the [setup guide](docs/setup.md).

## Glossary

| Abbreviation | Meaning |
|---|---|
| **QAT** | Quantization-aware training — training with quantization in the forward pass |
| **STE** | Straight-through estimator — using a surrogate gradient through rounding |
| **NNUE** | Efficiently Updatable Neural Network — a lightweight evaluation function used by shogi / chess engines |
| **FT** | Feature Transformer — the NNUE's sparse-input → dense layer |
| **LayerStack shared parameters** | `l1_shared_weight` and `l1_shared_bias` are the bucket-independent L1 terms initialized by `--init-l1-shared`; `--stack-shared-delta` adds zero-initialized shared terms to L2/L3 during training |
| **PSV** | PackedSfenValue — a training-data format from bullet-shogi (one position + score + WDL) |
| **DL** | Deep learning — the model-derived score label stored alongside the base search score in a dual-label PSV |
| **HCPE** | HuffmanCodedPosAndEval — the 38-byte Apery / dlshogi position, score, move, and game-result format |
| **KP / KP-abs** | King-Piece relative feature and its absolute-value variant (for progress / entering-king detection) |
| **Q16** | Fixed-point values scaled by 65536 (16 fractional bits). `--bucket-mode progresskpabsq16` and `progresskpabsq16ek` quantise the progress coefficients and bucket thresholds this way |
| **bucket** | Per-output-bucket weight separation (branching by game phase / progress) |
| **PSQT** | Piece-Square Table — a linear per-piece-per-square evaluation table. The LayerStack `--psqt` option adds a per-bucket PSQT output to the network output so the dense path only has to learn non-material structure |
| **EffectBucket** | Feature-set extension that maps each base feature row to `base_index * NB + effect_bucket`, where `EffectBucket=<config>` in the arch string records the bucket count and whether king features are bucketed |
| **FT factorizer** | Training-time virtual FT rows shared across king buckets (LayerStack; **ON by default**, disable with `--no-ft-factorize`). Each virtual row learns the king-bucket-independent component of one piece plane and is folded into the real rows on export, so the `.bin` shape and inference are unchanged. The same idea as nnue-pytorch's feature factorizer (whose default feature set `HalfKAv2_hm^` is also factorized). Works together with `--psqt` (the PSQT block gets the same virtual rows); `--init-from` automatically disables it (logged, since a quantised `.bin` has no virtual rows) |
| **CReLU / SCReLU / Pairwise** | NNUE activation functions. CReLU = Clipped ReLU, SCReLU = Squared Clipped ReLU, Pairwise = elementwise product of the first and second halves, halving the input dimension. Selected by `--activation` on the `simple` architecture |
| **RAdam / Ranger** | Rectified Adam / Ranger optimizer (Ranger = RAdam + lookahead) |
| **WRM** | Win-rate model loss (from bullet `--win-rate-model`) |
| **QA / QB / FV_SCALE** | Quantisation scale constants. QA = the FT weight / bias quantisation multiplier (on the `simple` architecture it is set by the activation: 127 for CReLU / Pairwise, 255 for SCReLU); QB = dense-weight scale (64). FV_SCALE converts the net output back to a centipawn evaluation. On `simple`, activation outputs are always on a 127 scale, so the default is `round(127 × QB / training scale)`; `--fv-scale` overrides it, while omission retains the value from a v9+ resume checkpoint, then from an `--init-from` input net that carries one |
| **WDL** | Win/Draw/Loss — the game-result target (1.0 / 0.5 / 0.0) blended against the teacher score by the WDL lambda; see [docs/training-schedule.md](docs/training-schedule.md) |
| **FNV-1a** | Fowler–Noll–Vo hash, variant 1a — a simple non-cryptographic hash. The trainer uses its 64-bit form for the recipe fingerprint stored in resume checkpoints |
| **CV** | Coefficient of variation — sample standard deviation divided by the mean, reported as a percentage for benchmark run-to-run variability |
| **SPRT** | Sequential Probability Ratio Test — a method that plays two nets against each other and sequentially tests the strength difference. Used to confirm the quality of a trained net |
| **superbatch** | A bullet term: the unit of "multiple batches treated as one, advancing the lr/wdl scheduler" |
| **PTX** | Parallel Thread Execution — a virtual ISA for NVIDIA GPUs. CUDA C++ / Rust → PTX (`.ptx` text) → the CUDA driver's JIT compiles it to SASS (real machine code) for execution. It is portable across generations (PTX built for sm_80 runs forward-compatibly on sm_86/89/90). See the supported-GPU matrix in `docs/setup.md` |
| **SASS** | Per-generation real machine code for NVIDIA GPUs. The terminal form the CUDA driver JIT produces from PTX. This repository does not handle it directly |
| **sm_XX** | The compute capability of an NVIDIA GPU (e.g. sm_75 = Turing, sm_86 = Ampere RTX 30xx). Used to specify the target architecture when generating PTX (`CUDA_OXIDE_TARGET=sm_86`, etc.) |

## Related repositories

- [rshogi](https://github.com/SH11235/rshogi) — the shogi engine that loads and plays with the NNUE trained here
- [bullet](https://github.com/jw1912/bullet) — upstream (NNUE training framework)
- [bullet-shogi](https://github.com/SH11235/bullet-shogi) — a shogi-oriented fork of bullet
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) — the Rust → PTX rustc backend

## License

MIT (see [LICENSE](LICENSE)).
For the scope of code taken from bullet-shogi / bullet / cuda-oxide and license
compatibility, see [ATTRIBUTION.md](ATTRIBUTION.md).
