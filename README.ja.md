[English](README.md) | **日本語**

# tatara

![tatara — Rust で鍛える将棋 NNUE](docs/tatara-hero.jpg)

**将棋 NNUE 評価関数を高速に学習する Rust 製トレーナー。**

tatara は将棋の NNUE (Efficiently Updatable Neural Network) 評価関数を NVIDIA
GPU で学習するツール。trainer、data pipeline、CUDA Driver API runtime は
Rust で書かれ、native GPU backend は hand-fuse した CUDA C++ kernel を NVCC で
fat binary に compile して実行ファイルへ埋め込む。

既定の `native` feature は cuda-oxide を使わず、NVCC で build した
kernel と portable Rust host runtime だけを使う。
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) の Rust → PTX backend は opt-in の
数値・性能 oracle として保守する。両経路を残すことで、native backend の検証に使う
GPU/CPU 等価テスト資産も維持できる。`oxide-parity` feature は数値 parity と benchmark
比較のため両方の device backend を有効化し、
`--no-default-features --features oxide-parity` で指定する。build されるのは NVCC
fat binary のみなので、cuda-oxide 側の PTX は事前に
`bash scripts/build-kernels.sh` で生成しておく
([docs/native-cuda-benchmark.md](docs/native-cuda-benchmark.md) 参照)。

GPU kernel を hand-fuse することで **極めて高速** — 実測した cuda-oxide backend は
上流の CUDA C++ trainer [bullet-shogi](https://github.com/SH11235/bullet-shogi)
を上回る throughput を出し、native backend も同じ fused training 設計を使う。

**cuda-oxide vs bullet-shogi (RTX 3080 Ti 実測)**: LayerStack は bit-identical な既定経路でも
**+37%**、opt-in の FP16 モードを積むと最大 **~2.1×**。Simple (HalfKP
`512x2-8-64`) は既定経路で約 **+20%**、`--all-optim`(FP16/TF32) で約 **+55%**。

**実効 throughput** (`--batch-size 65536`, pos/s; fp32 → `--all-optim`(fp16/tf32))。
`--all-optim` の効果は帯域律速の強い旧世代 GPU ほど、また net が大きいほど大きい:

| arch / 構成 (feature) | RTX 3080 Ti (Ampere) | RTX 4070 Ti (Ada) | RTX 5090 (Blackwell) |
|---|---|---|---|
| LayerStack `--ft-out 1536` (`halfka-hm-merged`) | 0.99M → 1.59M (+61%) | 1.26M → 2.14M (+70%) | 2.45M → 3.13M (+28%) |
| LayerStack `--ft-out 768` (`halfka-hm-merged`) | 2.05M → 2.97M (+45%) | 2.69M → 4.14M (+54%) | 4.24M → 5.10M (+20%) |
| Simple `256x2-32-32` (`halfkp`) | 7.30M → 10.10M (+38%) | 5.79M → 7.85M (+35%) | 11.29M → 13.37M (+18%) |
| Simple `256x2-32-32` (`kp`) | – | 6.69M → 6.15M (−8%) | – |

RTX 4070 Ti (12 GB) の列は、下の計測コマンドを教師 file を page cache に載せた状態で
`--superbatches 3` (`--batches-per-superbatch` は LayerStack が 60、Simple が 150) で
2 回回した平均。2 回の差は最大 0.2%。特徴変換器が 1,710 行しかない `kp` では、
`--all-optim` が fp32 より遅かった。`–` は未測定。

教師 file の 1 局面は 40 バイトなので、600 万局面/秒で学習すると約 240 MB/s を読む。
これより遅いストレージではそれが上限になり、100 MB/s なら約 250 万局面/秒までになる。
ハードディスクでは追いつかないことがある。SSD、速いネットワークストレージ、page cache に
収まる教師 file なら律速にならない。

計測コマンド (教師データ / progress は環境依存 path、`--all-optim` の有無で pos/s を比較):

```sh
# LayerStack (halfka-hm-merged, progress 必要)
cargo run --release --bin nnue-train -- \
  --data /path/to/teacher.psv \
  --feature-set halfka-hm-merged --batch-size 65536 --superbatches 8 [--all-optim] \
  layerstack --progress-coeff /path/to/progress.bin --ft-out {1536|768} --l1 {16|8} --l2 32

# HalfKP (simple, progress 不要; simple トレーナは --win-rate-model 必須)
cargo run --release --bin nnue-train -- \
  --data /path/to/teacher.psv \
  --feature-set halfkp --batch-size 65536 --superbatches 8 [--all-optim] \
  --scale 290 --win-rate-model \
  --wrm-in-offset 0 --wrm-target-offset 0 \
  --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
  simple --arch 256x2-32-32
```

simple トレーナは win-rate-model loss のみ対応 (int8 出力層が plain sigmoid loss の
収束する centipawn スケール出力を表現できない)。上記の `--wrm-*` は WRM を plain
sigmoid へ恒等退化させるため、この設定では `--scale`、各
`--wrm-*-scaling`、`--wrm-nnue2score` に同じ値を使う。`simple --fv-scale` を
省略した場合は weight 読み込みの有無にかかわらず
`--scale = --wrm-nnue2score` が必須。書き出す `fv_scale` は v9 以降の
`--resume` checkpoint に保存された値、`--init-from` の入力 net の値 (net が持つ場合)、
`round(127 × QB / --scale)` の導出値の順に解決する。
`simple --fv-scale <N>` を明示すると常に `N` を書き出し、この等値要件を免除する。
v9 以降の checkpoint では再指定不要。v8 以前から resume する場合は override を
再指定しないと `--scale` からの導出値へ戻る。

導出値は学習ラベル基準の公称値であり、対局での最適値とは限らない。対局で最適値を
測って `--fv-scale` で焼き込むか、エンジン側 option で指定する。

*tatara(踏鞴)、砂鉄（raw material）から玉鋼を精錬する日本の伝統的なたたら炉 — raw data から net を鍛え上げる。*

> **NVIDIA only** — どちらの GPU backend も NVIDIA CUDA を対象とし、ROCm / AMD は対象外。
> AMD GPU で類似の NNUE 学習を行いたい場合は CUDA / HIP 両 backend を持つ
> 上流の [bullet-shogi](https://github.com/SH11235/bullet-shogi) を参照。

## 学習できる NNUE

学習する NNUE は、ネットワーク構造を決める **アーキテクチャ**(サブコマンド)と、
盤面をどう入力ベクトルに変換するかを決める **入力 feature set**(`--feature-set`)を
独立に選ぶ。

### アーキテクチャ

| アーキ | サブコマンド | 構造 |
|---|---|---|
| **LayerStack** | `layerstack` | 局面の進行度で出力層を bucket 別に専用化(`--num-buckets`、既定 9。Stockfish の "LayerStacks" と同じ発想)。`--bucket-mode progresskpabsq16ek` は双方の玉が盤の中段まで進んだ局面を 9 つ目の bucket に分ける。FT 出力 `--ft-out`(既定 1536)→ `--l1`(既定 16)→ `--l2`(既定 32)|
| **Simple** | `simple` | bucket 分割のない素の NNUE(FT → 隠れ 2 層 → 単一出力)。層次元は `--arch <l1>x2-<l2>-<l3>` で指定(`l1` = FT 出力、`l2`/`l3` = 隠れ層、既定 `256x2-32-32`)、活性化 crelu / screlu / pairwise |

### 入力 feature set

`--feature-set` で 6 種から選べる(既定 `halfka-hm-merged`)。玉をどう特徴に含めるかが違う:

| `--feature-set` | 玉の扱い |
|---|---|
| `halfkp` | 玉自体は駒の特徴に含めない |
| `halfka-split` | 玉も含める。自玉用と敵玉用の特徴枠を別々に持つ |
| `halfka-merged` | 玉も含める。自玉と敵玉で特徴枠を 1 つに共有する |
| `halfka-hm-split` | `halfka-split` に加え、玉が常に盤の片側へ来るよう左右反転し、玉マスを 81 → 45 に圧縮 |
| `halfka-hm-merged`(既定) | `halfka-merged` + 同じ左右反転による玉マス圧縮 |
| `kp` | 玉マスで重みブロックを切り替えない。駒(玉以外)と両玉のマスをそれぞれ独立した特徴にする(計 1710 入力)。YaneuraOu の KP256 が使う `K+P` の特徴リスト |

既定の `halfka-hm-merged` は、Stockfish の **HalfKAv2_hm**(玉マスの左右反転 +
自玉・敵玉の特徴枠を 1 つに共有)と同じ設計を将棋に適用したもの。

別バイナリ `progress-kpabs-train` は LayerStack の bucket 係数 `progress.bin` を
生成する KP-abs progress trainer。進行度を学習して出力 bucket に割り当てる手法は
[nodchip 氏の記事](https://nodchip.hatenablog.com/entry/2026/02/04/000000) の
アイデアに基づく。

### YaneuraOu で読めるネットワーク

`--feature-set` だけではネットワークは決まらない。特徴変換器と 2 つの隠れ層の幅も
ネットワークの一部である。次の 4 つの `simple` ネットワークは、同名の YaneuraOu 評価関数
ファイルと同じ file 構造で書き出すので、YaneuraOu が変換なしで読み込める。
4 つとも既定の `--activation crelu` を使う。

| ネットワーク | サブコマンド | `--feature-set` | サブコマンドの後ろの option |
|---|---|---|---|
| `halfkp_128x2_32_32` | `simple` | `halfkp` | `--arch 128x2-32-32` |
| `halfkp_256x2_32_32` | `simple` | `halfkp` | `--arch 256x2-32-32` |
| `halfkp_768x2_16_64` | `simple` | `halfkp` | `--arch 768x2-16-64` |
| `kp_256x2_32_32` | `simple` | `kp` | `--arch 256x2-32-32` |

`layerstack` のネットワークも 3 つ、NAGISA の評価関数ファイルと同じ file 構造で書き出せる。
9 個の stack を持ち、進行度で stack (bucket 0〜7) を選ぶ。進行度には、学習に使った
`progress.bin` が要る。`progress8ek` の 2 つは、両玉が盤の中央まで入った局面を bucket 8
に振り分ける。`progress8kpabs` は stack 8 を使わない。

| ネットワーク | サブコマンド | `--feature-set` | サブコマンドの後ろの option |
|---|---|---|---|
| `halfkahm2_1024_15_64_progress8ek` | `layerstack` | `halfka-hm-merged` | `--ft-out 1024 --l1 16 --l2 64 --bucket-mode progresskpabsq16ek --progress-coeff <progress.bin>` |
| `halfkahm2_2304_15_64_progress8ek` | `layerstack` | `halfka-hm-merged` | `--ft-out 2304 --l1 16 --l2 64 --bucket-mode progresskpabsq16ek --progress-coeff <progress.bin>` |
| `halfkahm2_1024_15_64_progress8kpabs` | `layerstack` | `halfka-hm-merged` | `--ft-out 1024 --l1 16 --l2 64 --bucket-mode progresskpabsq16pad --progress-coeff <progress.bin>` |

`nnue-train` が書く `.bin` は、`progress8ek` のネットワークなら
`net_to_yo --assume-progresskpabsq16ek`、`progress8kpabs` なら
`net_to_yo --assume-progresskpabsq16pad` で変換する
([詳細](docs/net-to-yaneuraou.md))。変換後のファイルの Network 名は KingRank9 の net と
同じ `_K3K3` で終わるので、どの bucket 規則かはエンジンに教える必要がある。評価関数の
directory に `nn.bin` と並べて `progress.bin` を置く。`progress8ek` のときは
`eval_options.txt` に `LS_BUCKET_MODE progress8ek` と書く。既定の `progress8kpabs` のままだと、
エンジンは警告なしに読み込み、別の stack を選ぶ。`progress8kpabs` はエンジンの
`LS_BUCKET_MODE` の既定値である。`progress8kpabs` のネットワークの stack 8 には学習中に
局面が一つも送られないので、そこでは何も学習されない。

ネットワーク名は [shielune/YaneuraOu](https://github.com/shielune/YaneuraOu) フォークの
命名 (`NNUE_<feature>_<FT>x2_<H1>_<H2>`、同フォークの
`source/eval/nnue/architectures/README.md` に説明がある) に従う。YaneuraOu に別の
ネットワークの `.bin` を読ませると、`NNUE hash mismatch` と `failed to read nn.bin` で
拒否される。

他の学習器がこれらのネットワーク向けに書いた `.bin` も `--init-from` で読める。次の条件を
すべて満たすファイルを受け入れる。

- version、network hash、FT hash、FC hash が、指定した `--feature-set`、`simple --arch`、
  `--activation` から計算した値と一致する。
- architecture 文字列より後ろのデータの長さが、そのネットワークに必要な長さと過不足なく
  一致する。
- architecture 文字列が別の活性化を示していない。hash は活性化を含まないため、
  `SqrClippedReLU` を含む文字列は `--activation screlu` でなければ、`/2x2]-Pairwise` の
  特徴変換ブロックを持つ文字列は `--activation pairwise` でなければ拒否する。

それ以外では、ネットワークの同定に architecture 文字列を使わない。`ModelType=` の接頭辞、
tatara の `AffineTransformSparseInput` に当たる箇所が `AffineTransform` であること、
`,fv_scale=` が無いこと、の違いは黙って受け入れる。これ以外の違い (例えば層の幅の誤記)
は受け入れた上で、2 つの文字列を示す警告を stderr に出す。feature set や層の幅が違う
ファイルは hash が異なるため拒否される。

`,fv_scale=` を持たないファイルは `fv_scale` を供給しない。その場合の書き出しは
`--init-from` なしの実行と同じ値、`simple --fv-scale` を指定しなければ
`round(127 × QB / --scale)` になり、起動ログの `fv_scale:` の行にその旨が出る。

## セットアップ

### 環境要件

- **OS** — Linux 一級サポート、Windows は WSL2 経由に加え
  `native` で native Windows を実験的にサポート、macOS は GPU ビルド非対応
- **NVIDIA GPU** (backend 別の対応表は `docs/setup.ja.md` 参照)
- **CUDA Toolkit 12.x** (12.9 で動作確認)
- 既定の `native` backend は **NVCC**、opt-in の cuda-oxide oracle は
  **LLVM 21+** と `cargo-oxide`
- **Rust nightly** (`rust-toolchain.toml` で固定)

native CUDA C++ の build command、cuda-oxide のセットアップ、OS 別の詳細手順、
サポート GPU マトリクスは [docs/setup.ja.md](docs/setup.ja.md) を参照。

### ビルドと学習

kernel のビルドと smoke test は [docs/setup.ja.md](docs/setup.ja.md)、学習の回し方は
[docs/training-quickstart.ja.md](docs/training-quickstart.ja.md) を参照。build せずに
コンテナで学習する場合は、下の「Docker で学習する」を参照。

### Docker で学習する

GitHub Container Registry の image に `nnue-train` とこの workspace の各ツールが入っている。
CUDA 12.9 に対応した NVIDIA driver と
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
がある machine なら、toolchain 無しで学習できる。

```bash
docker pull ghcr.io/rshogi/tatara:latest

# GPU の smoke test。データ不要で、最後に "[smoke/simple] PASSED" と出る
docker run --rm --gpus all ghcr.io/rshogi/tatara:latest simple
```

image 名より後ろの引数はすべて `nnue-train` に渡る。教師データは読み取り専用で、出力先は
container が消えても残る directory で mount し、どちらも container 内の path で指定する。
`--data` は PSV を 1 file だけ取るので、shard は先に連結する (`cat shard-*.psv > train.psv`)。
次は `kp_256x2_32_32` を学習する例。別のネットワークにするときは、上の
「YaneuraOu で読めるネットワーク」の表のとおり `--feature-set` と `--arch` を変える:

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

- `--test-data` は検証用の file で、損失をスーパーバッチごとに表示する。省けば検証なしで学習する。
- `--threads` は教師データを読み込む thread 数。必ず指定する。読み込みが律速になりやすく、
  物理 core 数が出発点になる。
- 1 スーパーバッチは `--batches-per-superbatch` バッチ (既定 6104) で、`--batch-size 65536`
  なら約 4 億局面。
- `--save-rate` スーパーバッチごと (既定 20) と最後に、出力先に `<net-id>-<superbatch>.bin`
  (量子化した net) と `<net-id>-<superbatch>.ckpt` (再開用の状態) ができる。
- 起動時の log に、`--scale 290` なら `fv_scale: 28` と出る。YaneuraOu は file から `fv_scale` を
  読まないので、同じ値を USI option `FV_SCALE` で渡す。
- ほかのツールも同じ image に入っている:
  `docker run --rm --entrypoint net_to_yo ghcr.io/rshogi/tatara:latest --help`。

GPU の選び方、他の世代の GPU、image を自分で build する方法は
[docs/setup.ja.md](docs/setup.ja.md#docker-image) の「Docker image」、学習の option は
[docs/training-quickstart.ja.md](docs/training-quickstart.ja.md) にある。

## ドキュメント

- [Setup guide](docs/setup.ja.md) — OS 別の案内、native CUDA / cuda-oxide の build
  セットアップ、サポート GPU マトリクス、CUDA toolkit root 解決
- [Training quickstart](docs/training-quickstart.ja.md) — アーキ別の学習例 + 主要
  CLI option + resume / checkpoint 運用
- [局面進行度 bucket: `progress.bin` の用意](docs/progress-bin.ja.md) — LayerStack
  の bucket 係数の学習と bucket 分布の確認
- [held-out validation](docs/held-out-validation.ja.md) — `test_loss` / `test_acc`
  の有効化、held-out source の選び方、指標の読み方
- [Training schedules](docs/training-schedule.ja.md) — 学習率 (`--lr-schedule`) と
  WDL lambda (`--wdl` / `--start-wdl` / `--end-wdl`) のスケジューリング
- [Fused kernel catalog](docs/kernels/fused-pattern-catalog.md) — どの kernel
  が何を担うか
- [Arch string](docs/arch-string.md) — 量子化 `.bin` header に埋め込むアーキ
  記述文字列の組み立てと load 時照合 (日本語のみ)
- [YaneuraOu 用 LayerStack net 変換](docs/net-to-yaneuraou.md) —
  `net_to_yo` の使い方、対応アーキ、binary layout

## 学習した net の使い方

tatara が出力する量子化 `.bin` は [rshogi](https://github.com/SH11235/rshogi)
エンジンでロードする前提の format。`.bin` header と SCReLU / Pairwise 活性化は
本プロジェクト固有なので、YaneuraOu など他の将棋エンジンでそのまま読めるとは
限らず、アーキテクチャによっては推論部の追加実装をしないとロードできない。
対応する 1536-16-32 LayerStack は
[`net_to_yo`](docs/net-to-yaneuraou.md) で変換できる。
「YaneuraOu で読めるネットワーク」の表にある `simple` ネットワークは、YaneuraOu にそのまま
読める。出力の scale は file ではなく YaneuraOu の `FV_SCALE` option で
決まる。
学習済みの参考 net は
[GitHub Releases](https://github.com/SH11235/tatara/releases) に添付している。
自分で net を学習する場合の環境構築は [docs/setup.ja.md](docs/setup.ja.md) を参照。

## 用語 (glossary)

| 略語 | 意味 |
|---|---|
| **NNUE** | Efficiently Updatable Neural Network — 将棋 / チェスエンジンで使われる軽量評価関数 |
| **FT** | Feature Transformer — NNUE の入力 sparse → dense 層 |
| **LayerStack shared parameter** | `l1_shared_weight` / `l1_shared_bias` は `--init-l1-shared` で初期化する bucket 非依存の L1 項。`--stack-shared-delta` は学習時の L2/L3 にゼロ初期化した shared 項を追加する |
| **PSV** | PackedSfenValue — bullet-shogi 由来の学習データ format (1 局面 + score + WDL) |
| **HCPE** | HuffmanCodedPosAndEval — Apery / dlshogi の 38-byte 局面・score・指し手・対局結果 format |
| **KP / KP-abs** | King-Piece relative feature と絶対値版 (progress / 入玉判定用) |
| **Q16** | 65536 倍 (小数部 16 bit) の固定小数点。`--bucket-mode progresskpabsq16` と `progresskpabsq16ek` は progress 係数と bucket 閾値をこの形で量子化する |
| **bucket** | per-output-bucket 重み分離 (game phase / progress で分岐) |
| **PSQT** | Piece-Square Table — 駒種×マスごとの線形評価テーブル。LayerStack の `--psqt` で per-bucket PSQT 出力を network 出力に加算し、dense 経路は非マテリアル構造の学習に専念できる |
| **CReLU / SCReLU / Pairwise** | NNUE の活性化関数。CReLU = Clipped ReLU、SCReLU = Squared Clipped ReLU、Pairwise = 前半と後半の要素積で入力次元を半減。`simple` アーキの `--activation` で選択 |
| **RAdam / Ranger** | Rectified Adam / Ranger optimizer (Ranger = RAdam + lookahead) |
| **WRM** | Win-rate model loss (bullet `--win-rate-model` 由来) |
| **QA / QB / FV_SCALE** | 量子化スケール定数。QA = FT weight / bias の量子化 multiplier (`simple` アーキでは活性化で決まる: CReLU / Pairwise は 127、SCReLU は 255)、QB = dense weight の scale (64)。FV_SCALE は net 出力を centipawn 評価値へ戻す係数。`simple` の活性化出力は常に 127-scale のため、既定値は `round(127 × QB / 学習 scale)`。`--fv-scale` で上書きでき、省略時は v9 以降の resume checkpoint、次に `--init-from` の入力 net (値を持つ場合) から値を引き継ぐ |
| **WDL** | Win/Draw/Loss — 対局結果ターゲット (1.0 / 0.5 / 0.0)。WDL lambda で教師 score と blend する。[docs/training-schedule.ja.md](docs/training-schedule.ja.md) を参照 |
| **FNV-1a** | Fowler–Noll–Vo hash の 1a 版。暗号用途ではない簡単なハッシュで、学習器は resume 用 checkpoint に保存するレシピの指紋に 64 bit 版を使う |
| **SPRT** | Sequential Probability Ratio Test — 2 つの net を対局させ棋力差を逐次検定する手法。学習済 net の品質確認に使う |
| **superbatch** | bullet 用語で「複数 batch を 1 単位として lr/wdl scheduler を進める」単位 |
| **PTX** | Parallel Thread Execution — NVIDIA GPU 向け仮想 ISA。CUDA C++ / Rust → PTX (`.ptx` テキスト) → CUDA driver の JIT が SASS (実機機械語) に compile して実行。世代非依存に配布可 (sm_80 向け PTX を sm_86/89/90 が forward-compat で実行できる)。`docs/setup.ja.md` のサポート GPU マトリクス参照 |
| **SASS** | NVIDIA GPU の世代別実機機械語。PTX から CUDA driver JIT が生成する終端形式。本リポでは直接扱わない |
| **sm_XX** | NVIDIA GPU の compute capability (例: sm_75 = Turing、sm_86 = Ampere RTX 30xx)。PTX 生成時の target アーキ指定 (`CUDA_OXIDE_TARGET=sm_86` 等) に使う |

## 関連リポジトリ

- [rshogi](https://github.com/SH11235/rshogi) — 本リポで学習した NNUE をロードして対局する将棋エンジン
- [bullet](https://github.com/jw1912/bullet) — 上流 (NNUE training framework)
- [bullet-shogi](https://github.com/SH11235/bullet-shogi) — bullet の将棋向け fork
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) — Rust → PTX rustc backend

## License

MIT (see [LICENSE](LICENSE))。
bullet-shogi / bullet / cuda-oxide からの取り込み範囲とライセンス互換性は
[ATTRIBUTION.md](ATTRIBUTION.md) を参照。
