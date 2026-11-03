//! Simple 4 層 NNUE quantised binary の save / load。
//!
//! `ArchKind::Simple` (bucket 無し 4 層 dense アーキ) の量子化 weight を 1 本の
//! binary に (de)serialise する **GPU 非依存・pure CPU** module。LayerStack の
//! [`crate::layerstack_weights`] と並ぶ、もう一方のアーキ用フォーマット。
//!
//! ## Simple architecture (bucket 無し / PSQT 無し / skip 無し / L1 shared 無し)
//!
//! - FT: `ft_in → ft_out`、weight + bias 共有 stm/nstm (`ft_in` は feature set 依存、
//!   `ft_out` は CLI 設定値)
//! - per-perspective FT post 活性化 → `concat(stm, nstm)` = `l1_in`。CReLU / SCReLU は
//!   elementwise で `l1_in = 2*ft_out`、Pairwise は前半×後半の積で 1 perspective が
//!   半減し `l1_in = ft_out`
//! - L1 dense: `l1_in → l1_out` + bias → CReLU / SCReLU
//! - L2 dense: `l1_out → l2_out` + bias → CReLU / SCReLU
//! - L3 dense: `l2_out → 1` + bias → スカラ net_output
//!
//! ## file layout (top-level、little-endian)
//!
//! 1. header: `NNUE_VERSION` (u32) + `network_hash` (u32) + `arch_len` (u32) + `arch_str`
//! 2. `ft_hash` (u32)
//! 3. FT biases: `i16` × `ft_out` (raw、scale = QA)
//! 4. FT weights: `i16` × `ft_in * ft_out` (raw、scale = QA)
//! 5. `fc_hash` (u32) — dense 層群の topology hash
//! 6. L1: bias `i32` × `l1_out` (scale = `127*QB`)、weight `i8` × `l1_out * pad32(l1_in)` (scale = QB)
//! 7. L2: bias `i32` × `l2_out`、weight `i8` × `l2_out * pad32(l1_out)`
//! 8. L3: bias `i32` × 1、weight `i8` × `pad32(l2_out)`
//!
//! ## 量子化 scale
//!
//! | layer | bias scale | weight scale |
//! |---|---|---|
//! | FT | QA (i16) | QA (i16) |
//! | L1 / L2 / L3 | `127*QB` (i32) | QB (i8) |
//!
//! QA は活性化で決まる: CReLU / Pairwise → 127、SCReLU → 255。dense 層の入力は
//! 活性化後で常に 127-scale (CReLU の出力、SCReLU の `x²>>9`、Pairwise の 2 CReLU
//! 積 `×127/128` いずれも 127-scale) のため、dense bias scale は活性化に依らず
//! `127*QB`。
//!
//! ## pad32
//!
//! dense 層の i8 weight は入力次元を 32 の倍数に pad して書く (SIMD load align)。
//! padding byte は 0。
//!
//! ## 重み layout
//!
//! 全 weight は **row-major** (`w[out * in_dim + in_idx]`)。FT は
//! `ft_w[feat * ft_out + out]`。

use std::io::{self, Read, Write};

use shogi_features::FeatureSetSpec;

// =============================================================================
// constants
// =============================================================================

/// Simple アーキ量子化 binary の format version magic。
///
/// 値は YaneuraOu upstream `source/eval/nnue/nnue_common.h` の `kVersion`
/// (= `0x7AF32F16`) と一致させる。YaneuraOu の versioning モデルでは version magic
/// は形式世代スタンプであり、アーキの弁別子ではない (全 NNUE ファイルで単一値、
/// アーキは file 内 `kHashValue = FeatureTransformer::GetHashValue() ^
/// Network::GetHashValue()` で識別する)。本 format の `network_hash` は同 hash 機構の
/// 移植。
pub const NNUE_VERSION: u32 = 0x7AF32F16;

/// dense 層 weight の量子化 multiplier。
pub const QB: i32 = 64;

/// FT 活性化後 (= dense 層入力) の量子化スケール。
///
/// FT の活性化出力は活性化関数に依らず常に 127-scale (CReLU の出力、SCReLU の
/// `x²>>9`、Pairwise の 2 CReLU 積 ×127/128 いずれも 127-scale)。
/// `SimpleActivation::qa()` (SCReLU=255) は FT 重み量子化用であり、dense bias
/// scale と fv_scale の算出にはこの定数を用いる。
pub const FT_OUTPUT_QA: i32 = 127;

/// 推論側 evaluation scale `fv_scale` を算出する。
///
/// `fv_scale = round(FT_OUTPUT_QA × QB / scale)`。量子化ネットワークの生出力を
/// centipawn 相当へ換算する除数で、`.bin` ヘッダの arch 文字列 (`build_arch_str`)
/// に書き込まれ推論エンジンが参照する。
///
/// FT 活性化出力は活性化関数に依らず 127-scale ([`FT_OUTPUT_QA`]) のため、
/// fv_scale も活性化非依存。`scale` は学習 loss の sigmoid scale で、呼び出し側で
/// 有限・正値を保証すること。
pub fn simple_fv_scale(scale: f32) -> i32 {
    ((FT_OUTPUT_QA * QB) as f32 / scale).round() as i32
}

/// pad to multiple of 32 (SIMD alignment)。
#[inline]
pub fn pad32(x: usize) -> usize {
    x.div_ceil(32) * 32
}

// =============================================================================
// SimpleActivation
// =============================================================================

/// Simple アーキの FT post 活性化関数。FT 出力の量子化 multiplier QA と arch 文字列の
/// 活性化トークンを決める。
///
/// `CReLU` / `SCReLU` は FT 出力を elementwise に変換する (出力次元 = 入力次元)。
/// `Pairwise` は前半 `[0, ft_out/2)` と後半 `[ft_out/2, ft_out)` の対応 index 同士の
/// 積を取り **出力次元を半減** する (`combined_dim` が L1 入力次元として連動する)。
/// L1 / L2 の dense 層出力は `Pairwise` でも CReLU で活性化する (pairwise 乗算は
/// FT post 固有で、単一ベクトルの dense 出力には適用しない)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimpleActivation {
    /// Clipped ReLU (`clamp(x, 0, 1)`)。
    CReLU,
    /// Squared Clipped ReLU (`clamp(x, 0, 1)²`)。
    SCReLU,
    /// Pairwise Clipped ReLU: `clamp(a, 0, 1) * clamp(b, 0, 1) * (127/128)`、`a` は
    /// 前半 index、`b` は対応する後半 index。FT 出力次元を半減する。
    Pairwise,
}

impl SimpleActivation {
    /// CLI / arch identity が扱う canonical 名。
    pub const fn canonical_name(self) -> &'static str {
        match self {
            SimpleActivation::CReLU => "crelu",
            SimpleActivation::SCReLU => "screlu",
            SimpleActivation::Pairwise => "pairwise",
        }
    }

    /// canonical 名から逆引きする。未知の名前は `None`。
    pub fn from_canonical_name(name: &str) -> Option<SimpleActivation> {
        match name {
            "crelu" => Some(SimpleActivation::CReLU),
            "screlu" => Some(SimpleActivation::SCReLU),
            "pairwise" => Some(SimpleActivation::Pairwise),
            _ => None,
        }
    }

    /// FT 出力の量子化 multiplier QA。CReLU / Pairwise は 127、SCReLU は 255。
    ///
    /// Pairwise は 2 つの CReLU 出力の積で、各因子が CReLU と同じ `[0, 1]` 値域・
    /// 127-scale 量子化を持つため QA = 127。
    pub const fn qa(self) -> i32 {
        match self {
            SimpleActivation::CReLU | SimpleActivation::Pairwise => 127,
            SimpleActivation::SCReLU => 255,
        }
    }

    /// FT post の出力次元が入力 (1 perspective あたり `ft_out`) に対して何分の 1 か。
    /// CReLU / SCReLU は 1、Pairwise は 2 (前半×後半の積で半減)。
    pub const fn ft_output_divisor(self) -> usize {
        match self {
            SimpleActivation::CReLU | SimpleActivation::SCReLU => 1,
            SimpleActivation::Pairwise => 2,
        }
    }

    /// arch 文字列の dense 層 (L1 / L2) 活性化トークン (nnue-pytorch 系の層名)。
    /// Pairwise の dense 層は CReLU で活性化するため `ClippedReLU`。
    const fn arch_token(self) -> &'static str {
        match self {
            SimpleActivation::CReLU | SimpleActivation::Pairwise => "ClippedReLU",
            SimpleActivation::SCReLU => "SqrClippedReLU",
        }
    }
}

// =============================================================================
// SimpleId — Simple ネットの構造化 identity
// =============================================================================

/// Simple ネットの identity を成す構造化フィールド。`load` はこの値から計算した
/// version・hash・本体の長さ・活性化と file の内容を照合する。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimpleId {
    /// 入力 feature set (FT 入力次元 `ft_in` と feature hash の源)。
    pub feature_set: FeatureSetSpec,
    /// per-perspective 活性化関数。
    pub activation: SimpleActivation,
    /// FT 出力 (accumulator) 次元。
    pub ft_out: usize,
    /// L1 dense 出力 (隠れ層 1) 次元。
    pub l1_out: usize,
    /// L2 dense 出力 (隠れ層 2) 次元。
    pub l2_out: usize,
}

impl SimpleId {
    /// FT 入力次元 (`feature_set` 依存)。
    pub fn ft_in(&self) -> usize {
        self.feature_set.ft_in()
    }

    /// L1 dense の入力次元 (= stm/nstm の FT post 出力を concat した次元)。
    /// CReLU / SCReLU は `2 * ft_out`、Pairwise は pairwise 乗算で 1 perspective が
    /// `ft_out / 2` に縮むため `ft_out`。
    pub fn combined_dim(&self) -> usize {
        2 * self.ft_out / self.activation.ft_output_divisor()
    }
}

// =============================================================================
// arch_str + hash
// =============================================================================

/// arch 文字列の identity 部 (feature / 層次元 / 活性化を含み、`fv_scale` を含まない)。
/// `load` は他の trainer が書いたファイルの文字列をこの部分と見比べ、食い違いを警告する。
fn arch_identity(id: &SimpleId) -> String {
    let ft_in = id.ft_in();
    let ft_out = id.ft_out;
    let l1_out = id.l1_out;
    let l2_out = id.l2_out;
    let l1_input = id.combined_dim();
    let act = id.activation.arch_token();
    // FT ブロックは活性化で 2 形式に分かれる。CReLU / SCReLU は `[ft_in->ft_outx2]`、
    // Pairwise は FT 出力を pairwise 乗算で半減するため `[ft_in->ft_out/2x2]-Pairwise`。
    // 推論エンジンは FT ブロックの `/2` と `-Pairwise` suffix で pairwise を識別し、
    // `->Nx2]` から FT 次元を読む。
    let ft_block = match id.activation {
        SimpleActivation::Pairwise => format!("[{ft_in}->{ft_out}/2x2]-Pairwise"),
        SimpleActivation::CReLU | SimpleActivation::SCReLU => format!("[{ft_in}->{ft_out}x2]"),
    };
    format!(
        "Features={}{}{},\
         Network=AffineTransform[1<-{}](\
         {}[{}](\
         AffineTransform[{}<-{}](\
         {}[{}](\
         AffineTransformSparseInput[{}<-{}](\
         InputSlice[{}(0:{})])))))",
        id.feature_set.arch_feature_name(),
        id.feature_set.arch_friend_marker(),
        ft_block,
        l2_out,   // Output layer input
        act,      // L2 出力の活性化
        l2_out,   // L2 output features
        l2_out,   // L3 input / L2 output
        l1_out,   // L2 input features
        act,      // L1 出力の活性化
        l1_out,   // L1 output features
        l1_out,   // L2 input / L1 output
        l1_input, // L1 input (FT post 出力の concat 次元)
        l1_input, // InputSlice size
        l1_input, // InputSlice range
    )
}

/// arch 文字列全体 (`<identity>,fv_scale=<N>`)。`fv_scale` は推論時の評価値
/// スケールで identity には含めない (同一 topology でも変動する)。
pub fn build_arch_str(id: &SimpleId, fv_scale: i32) -> String {
    format!("{},fv_scale={}", arch_identity(id), fv_scale)
}

/// 他の trainer が書いたファイルの arch 文字列を `expected` と照合した結果。
#[derive(Debug, PartialEq, Eq)]
struct ArchTextReport {
    /// 末尾の `,fv_scale=N` token の値。token が無ければ `None`。
    fv_scale: Option<i32>,
    /// 文字列が示す活性化 ([`activation_named_by`])。
    activation: SimpleActivation,
    /// 許容する書き方の違い ([`tolerated_form`]) を除いても `expected` の arch 文字列と
    /// 食い違う箇所の説明。食い違いが無ければ `None`。
    difference: Option<String>,
}

/// arch 文字列が示す活性化。`SqrClippedReLU` を含めば SCReLU、FT ブロックが
/// `/2x2]-Pairwise` で終われば Pairwise、どちらも無ければ CReLU。
///
/// network hash は活性化を含まない (CReLU と SCReLU の 256x2-32-32 は hash も本体の
/// バイト数も同一) ため、FT の量子化 scale (QA) を決めるこの区別は文字列だけが担う。
/// 両方の印を持つ文字列は tatara が書く形ではなく QA を決められないので `Err`。
fn activation_named_by(arch_text: &str) -> io::Result<SimpleActivation> {
    let squared = arch_text.contains("SqrClippedReLU");
    let pairwise = arch_text.contains("/2x2]-Pairwise");
    match (squared, pairwise) {
        (true, true) => Err(invalid(format!(
            "arch_str names both SqrClippedReLU and a Pairwise feature block, so its \
             activation is ambiguous: `{arch_text}`"
        ))),
        (true, false) => Ok(SimpleActivation::SCReLU),
        (false, true) => Ok(SimpleActivation::Pairwise),
        (false, false) => Ok(SimpleActivation::CReLU),
    }
}

/// 同じネットワークを指しうる書き方の違いを畳んだ arch 文字列。
///
/// - 先頭の `ModelType=<name>;` を落とす (YaneuraOu 系の trainer が付ける)。
/// - `AffineTransformSparseInput[` を `AffineTransform[` に揃える (疎入力に対する最適化の
///   有無の違いで、重みの並びは同じ)。
fn tolerated_form(arch_text: &str) -> String {
    let body = match arch_text
        .strip_prefix("ModelType=")
        .and_then(|rest| rest.split_once(';'))
    {
        Some((_, rest)) => rest,
        None => arch_text,
    };
    body.replace("AffineTransformSparseInput[", "AffineTransform[")
}

/// 2 つの文字列が最初に食い違う位置の前後を抜き出して説明する。
fn describe_first_difference(file: &str, want: &str) -> String {
    let at = file
        .chars()
        .zip(want.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let excerpt =
        |text: &str| -> String { text.chars().skip(at.saturating_sub(24)).take(56).collect() };
    format!(
        "the file has `...{}` where this network is written as `...{}`",
        excerpt(file),
        excerpt(want)
    )
}

/// file の arch 文字列を解釈する。`,fv_scale=N` があれば値を検証して取り出し、
/// 無ければ `None` とする。
fn inspect_arch_text(arch_str: &str, expected: &SimpleId) -> io::Result<ArchTextReport> {
    let (arch_text, fv_scale) = match arch_str.rsplit_once(",fv_scale=") {
        Some((arch_text, fv_scale_str)) => {
            let fv_scale: i32 = fv_scale_str
                .parse()
                .map_err(|_| invalid(format!("invalid fv_scale in arch_str: `{fv_scale_str}`")))?;
            if fv_scale <= 0 {
                return Err(invalid(format!(
                    "invalid fv_scale in arch_str: `{fv_scale_str}` (must be positive)"
                )));
            }
            (arch_text, Some(fv_scale))
        }
        None => (arch_str, None),
    };
    let activation = activation_named_by(arch_text)?;
    let file_form = tolerated_form(arch_text);
    let expected_form = tolerated_form(&arch_identity(expected));
    let difference =
        (file_form != expected_form).then(|| describe_first_difference(&file_form, &expected_form));
    Ok(ArchTextReport {
        fv_scale,
        activation,
        difference,
    })
}

/// `id` のネットワークで arch 文字列より後ろ (`ft_hash` から末尾まで) が占めるバイト数。
fn body_len(id: &SimpleId) -> usize {
    let ft = 4 + 2 * (id.ft_out + id.ft_in() * id.ft_out);
    let fc_hash = 4;
    let l1 = 4 * id.l1_out + id.l1_out * pad32(id.combined_dim());
    let l2 = 4 * id.l2_out + id.l2_out * pad32(id.l1_out);
    let l3 = 4 + pad32(id.l2_out);
    ft + fc_hash + l1 + l2 + l3
}

/// dense 層群の topology hash (nnue-pytorch 系)。
/// 層次元 (`l1_input` / `l1_out` / `l2_out`) のみに依存する。`l1_input` は L1 dense の
/// 入力次元 (= FT post 出力の concat、`SimpleId::combined_dim`)。
pub const fn compute_fc_hash(l1_input: usize, l1_out: usize, l2_out: usize) -> u32 {
    // InputSlice hash (L1 dense input = FT post 出力の concat 次元)。
    let mut prev_hash: u32 = 0xEC42E90D;
    prev_hash ^= l1_input as u32;

    // FC 層の出力次元列: L1=l1_out / L2=l2_out / 出力=1。L1/L2 は活性化付き。
    // const fn なので index ベースの `while` で回す。
    let layer_sizes = [(l1_out, true), (l2_out, true), (1_usize, false)];
    let mut i = 0;
    while i < layer_sizes.len() {
        let (out_features, has_relu) = layer_sizes[i];
        let mut layer_hash: u32 = 0xCC03DAE4;
        layer_hash = layer_hash.wrapping_add(out_features as u32);
        layer_hash ^= prev_hash >> 1;
        layer_hash ^= prev_hash << 31;
        if has_relu {
            layer_hash = layer_hash.wrapping_add(0x538D24C7);
        }
        prev_hash = layer_hash;
        i += 1;
    }
    prev_hash
}

/// FT (FeatureTransformer) layer hash: `feature_hash ^ (ft_out * 2)`。
/// LayerStack と違い `ft_out` が runtime 値のため引数で受ける。
pub fn ft_hash(feature_hash: u32, ft_out: usize) -> u32 {
    feature_hash ^ (ft_out as u32 * 2)
}

/// network hash: `compute_fc_hash ^ ft_hash`。
pub fn network_hash(id: &SimpleId) -> u32 {
    compute_fc_hash(id.combined_dim(), id.l1_out, id.l2_out)
        ^ ft_hash(id.feature_set.feature_hash(), id.ft_out)
}

// =============================================================================
// SimpleWeights — トレーナー側 weight 表現 (f32、row-major)
// =============================================================================

/// Simple アーキの全 weight (f32、host 側保持)。
///
/// Layout (`ft_in` = `id.ft_in()`、`ft_out` / `l1_out` / `l2_out` = `id` の値、
/// `l1_in` = `id.combined_dim()` = CReLU/SCReLU は `2*ft_out`・Pairwise は `ft_out`):
/// - `ft_w`: `(ft_in, ft_out)` row-major、`ft_w[feat * ft_out + out]`
/// - `ft_b`: `(ft_out)` (stm/nstm 共有)
/// - `l1_w`: `(l1_out, l1_in)` row-major
/// - `l1_b`: `(l1_out)`
/// - `l2_w`: `(l2_out, l1_out)` row-major
/// - `l2_b`: `(l2_out)`
/// - `l3_w`: `(l2_out)` (出力 1 次元なので out 軸省略)
/// - `l3_b`: `(1)`
#[derive(Debug, Clone)]
pub struct SimpleWeights {
    /// この weight の構造化 identity。
    pub id: SimpleId,
    /// 推論時の評価値スケール。通常は [`simple_fv_scale`] で算出し、arch 文字列に
    /// `,fv_scale=N` として記録する。他の trainer が書いたファイルはこの token を
    /// 持たないことがあり、その場合 `load` は値を作らず `None` を返す。`None` の
    /// weight を `save_quantised` すると token も書かない。
    pub fv_scale: Option<i32>,
    pub ft_w: Vec<f32>,
    pub ft_b: Vec<f32>,
    pub l1_w: Vec<f32>,
    pub l1_b: Vec<f32>,
    pub l2_w: Vec<f32>,
    pub l2_b: Vec<f32>,
    pub l3_w: Vec<f32>,
    pub l3_b: Vec<f32>,
}

impl SimpleWeights {
    /// 全 buffer を 0 で初期化した新規 instance。
    pub fn zeroed(id: SimpleId, fv_scale: i32) -> Self {
        Self {
            id,
            fv_scale: Some(fv_scale),
            ft_w: vec![0.0; id.ft_in() * id.ft_out],
            ft_b: vec![0.0; id.ft_out],
            l1_w: vec![0.0; id.l1_out * id.combined_dim()],
            l1_b: vec![0.0; id.l1_out],
            l2_w: vec![0.0; id.l2_out * id.l1_out],
            l2_b: vec![0.0; id.l2_out],
            l3_w: vec![0.0; id.l2_out],
            l3_b: vec![0.0; 1],
        }
    }

    /// 各 weight buffer の要素数が `id` の層次元と整合するか検証する。
    /// 不整合は `save_quantised` 内の index 計算が panic / 末尾切り捨てに
    /// なるため、書き出し前に `InvalidInput` で弾く。
    fn check_buffer_lengths(&self) -> io::Result<()> {
        let id = &self.id;
        let groups = [
            ("ft_w", self.ft_w.len(), id.ft_in() * id.ft_out),
            ("ft_b", self.ft_b.len(), id.ft_out),
            ("l1_w", self.l1_w.len(), id.l1_out * id.combined_dim()),
            ("l1_b", self.l1_b.len(), id.l1_out),
            ("l2_w", self.l2_w.len(), id.l2_out * id.l1_out),
            ("l2_b", self.l2_b.len(), id.l2_out),
            ("l3_w", self.l3_w.len(), id.l2_out),
            ("l3_b", self.l3_b.len(), 1),
        ];
        for (name, got, want) in groups {
            if got != want {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("SimpleWeights.{name} has {got} elements, expected {want}"),
                ));
            }
        }
        Ok(())
    }

    /// Simple quantised binary を `writer` に書き出す。weight buffer の長さが
    /// `id` の層次元と不整合なら、書き出し前に `InvalidInput` で返す。
    pub fn save_quantised<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.check_buffer_lengths()?;
        let id = &self.id;

        // ---- header ----
        writer.write_all(&NNUE_VERSION.to_le_bytes())?;
        writer.write_all(&network_hash(id).to_le_bytes())?;
        let arch_bytes = match self.fv_scale {
            Some(fv_scale) => build_arch_str(id, fv_scale),
            None => arch_identity(id),
        }
        .into_bytes();
        writer.write_all(&(arch_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&arch_bytes)?;

        // ---- FT hash ----
        writer.write_all(&ft_hash(id.feature_set.feature_hash(), id.ft_out).to_le_bytes())?;

        // ---- FT biases / weights (i16, scale = QA) ----
        // FT weight/bias は training 中 clamp されない (i16 飽和域 ±i16::MAX/QA まで
        // 開放) ため、recipe によっては量子化で silent clip されうる。export 側で
        // 飽和件数を数え、発生していれば警告する (clamp 自体は write_i16_quantised が
        // 行うので数値破綻はしないが、無言の情報損失に気付けるようにする)。
        let qa = id.activation.qa() as f64;
        warn_if_i16_saturates("ft_b", &self.ft_b, qa);
        warn_if_i16_saturates("ft_w", &self.ft_w, qa);
        write_i16_quantised(writer, &self.ft_b, qa)?;
        write_i16_quantised(writer, &self.ft_w, qa)?;

        // ---- dense topology hash ----
        writer
            .write_all(&compute_fc_hash(id.combined_dim(), id.l1_out, id.l2_out).to_le_bytes())?;

        // ---- dense layers (bias i32 scale 127*QB, weight i8 scale QB) ----
        let bias_scale = (FT_OUTPUT_QA * QB) as f64;
        let weight_scale = QB as f64;
        // dense weight は学習中に ±127/QB へ clamp される。loss と出力スケールの不整合で
        // 特に出力層 (l3) が clamp 端へ張り付くと評価値が破綻するため、層別に飽和率を
        // 警告する (i16 FT 飽和警告と対の export 側 guard)。
        warn_if_i8_saturates("l1_w", &self.l1_w, weight_scale);
        warn_if_i8_saturates("l2_w", &self.l2_w, weight_scale);
        warn_if_i8_saturates("l3_w (output)", &self.l3_w, weight_scale);
        // L1: (l1_out, combined_dim)
        write_i32_bias(writer, &self.l1_b, bias_scale)?;
        write_i8_weight(
            writer,
            &self.l1_w,
            id.l1_out,
            id.combined_dim(),
            weight_scale,
        )?;
        // L2: (l2_out, l1_out)
        write_i32_bias(writer, &self.l2_b, bias_scale)?;
        write_i8_weight(writer, &self.l2_w, id.l2_out, id.l1_out, weight_scale)?;
        // L3: (1, l2_out)
        write_i32_bias(writer, &self.l3_b, bias_scale)?;
        write_i8_weight(writer, &self.l3_w, 1, id.l2_out, weight_scale)?;

        Ok(())
    }

    /// Simple quantised binary を parse する。`expected` は要求する identity
    /// (feature set / 活性化 / 層次元)。
    ///
    /// 次のいずれかに当たる file は `InvalidData` (短い file は `UnexpectedEof`) で reject する。
    ///
    /// - version が [`NNUE_VERSION`] でない。
    /// - network hash・FT hash・dense 層群の hash のどれかが `expected` から計算した値と違う。
    ///   feature set と層次元の違いはここで落ちる。
    /// - arch 文字列より後ろの長さが `expected` のネットワークに要るバイト数と違う
    ///   (足りない、または余る)。
    /// - arch 文字列が示す活性化が `expected.activation` と違う。hash は活性化を含まない
    ///   ため、CReLU と SCReLU の取り違えを防ぐのはこの照合だけである。
    /// - `,fv_scale=N` の `N` が正の整数でない。
    ///
    /// arch 文字列そのものは同一性の根拠にしない。hash と長さが合うなら、`ModelType=` の
    /// 接頭辞、`AffineTransform` と `AffineTransformSparseInput` の違い、`,fv_scale=` の
    /// 欠落だけが異なるファイルは黙って受け入れ、それ以外の違い (例えば次元の誤記) は
    /// 受け入れた上で stderr に警告する。`,fv_scale=` が無いファイルの
    /// [`SimpleWeights::fv_scale`] は `None` になる。
    pub fn load<R: Read>(reader: &mut R, expected: SimpleId) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];

        // version
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != NNUE_VERSION {
            return Err(invalid(format!(
                "unknown NNUE version: {version:#x} (expected {NNUE_VERSION:#x})"
            )));
        }

        // network_hash
        reader.read_exact(&mut buf4)?;
        let file_network_hash = u32::from_le_bytes(buf4);
        let expected_network_hash = network_hash(&expected);
        if file_network_hash != expected_network_hash {
            return Err(invalid(format!(
                "network_hash mismatch: file {file_network_hash:#x}, \
                 expected {expected_network_hash:#x}"
            )));
        }

        // arch_str
        reader.read_exact(&mut buf4)?;
        let arch_len = u32::from_le_bytes(buf4) as usize;
        if arch_len == 0 || arch_len > 16384 {
            return Err(invalid(format!("invalid arch_len: {arch_len}")));
        }
        let mut arch_bytes = vec![0u8; arch_len];
        reader.read_exact(&mut arch_bytes)?;
        let arch_str = String::from_utf8(arch_bytes)
            .map_err(|e| invalid(format!("arch_str is not valid UTF-8: {e}")))?;
        // version と hash が合えば YaneuraOu は文字列を見ずに読み込むため、ここでも文字列の
        // 食い違いは拒否の理由にしない。ただし活性化は hash に現れないので文字列で確かめる。
        let arch = inspect_arch_text(&arch_str, &expected)?;
        if arch.activation != expected.activation {
            return Err(invalid(format!(
                "activation mismatch: the arch_str of the file names {}, expected {}: `{arch_str}`",
                arch.activation.canonical_name(),
                expected.activation.canonical_name()
            )));
        }
        let fv_scale = arch.fv_scale;

        // ft_hash
        reader.read_exact(&mut buf4)?;
        let file_ft_hash = u32::from_le_bytes(buf4);
        let expected_ft_hash = ft_hash(expected.feature_set.feature_hash(), expected.ft_out);
        if file_ft_hash != expected_ft_hash {
            return Err(invalid(format!(
                "ft_hash mismatch: file {file_ft_hash:#x}, expected {expected_ft_hash:#x}"
            )));
        }

        // FT biases / weights (i16, scale = QA)
        let qa = expected.activation.qa() as f32;
        let ft_b = read_i16_quantised(reader, expected.ft_out, qa)?;
        let ft_w = read_i16_quantised(reader, expected.ft_in() * expected.ft_out, qa)?;

        // dense topology hash
        reader.read_exact(&mut buf4)?;
        let file_fc_hash = u32::from_le_bytes(buf4);
        let expected_fc_hash =
            compute_fc_hash(expected.combined_dim(), expected.l1_out, expected.l2_out);
        if file_fc_hash != expected_fc_hash {
            return Err(invalid(format!(
                "fc_hash mismatch: file {file_fc_hash:#x}, expected {expected_fc_hash:#x}"
            )));
        }

        // dense layers
        let bias_scale = (FT_OUTPUT_QA * QB) as f32;
        let weight_scale = QB as f32;
        let l1_b = read_i32_bias(reader, expected.l1_out, bias_scale)?;
        let l1_w = read_i8_weight(
            reader,
            expected.l1_out,
            expected.combined_dim(),
            weight_scale,
        )?;
        let l2_b = read_i32_bias(reader, expected.l2_out, bias_scale)?;
        let l2_w = read_i8_weight(reader, expected.l2_out, expected.l1_out, weight_scale)?;
        let l3_b = read_i32_bias(reader, 1, bias_scale)?;
        let l3_w = read_i8_weight(reader, 1, expected.l2_out, weight_scale)?;

        // 期待する本体を読み切った後に 1 バイトでも残っていれば、別のネットワークの
        // ファイルか壊れたファイルなので拒否する。
        let mut probe = [0u8; 1];
        loop {
            match reader.read(&mut probe) {
                Ok(0) => break,
                Ok(_) => {
                    return Err(invalid(format!(
                        "trailing bytes after the {} bytes this network needs after the arch_str",
                        body_len(&expected)
                    )));
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }

        if let Some(difference) = arch.difference {
            eprintln!(
                "[nnue-format] warning: the arch_str of this file does not describe the requested \
                 network ({difference}). The file is loaded because its version, network hash, \
                 FT hash, FC hash and body length all match the requested network; those, not the \
                 text, identify the network."
            );
        }

        Ok(Self {
            id: expected,
            fv_scale,
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l2_w,
            l2_b,
            l3_w,
            l3_b,
        })
    }
}

// =============================================================================
// quantise / dequantise helpers
// =============================================================================

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn clamp_round(v: f64, lo: f64, hi: f64) -> f64 {
    let r = v.round();
    if r < lo {
        lo
    } else if r > hi {
        hi
    } else {
        r
    }
}

/// f32 列を `scale` 倍して i16 量子化し、raw little-endian で書く。
fn write_i16_quantised<W: Write>(writer: &mut W, values: &[f32], scale: f64) -> io::Result<()> {
    for &v in values {
        let q = clamp_round(scale * v as f64, i16::MIN as f64, i16::MAX as f64) as i16;
        writer.write_all(&q.to_le_bytes())?;
    }
    Ok(())
}

/// `round(scale·v)` が i16 範囲 `[-32768, 32767]` を超える要素数を数える。`scale`
/// は FT 量子化スケール QA。値域に収まれば 0。`write_i16_quantised` の clamp が
/// 情報を落とす要素を export 前に把握するための pure helper。
fn count_i16_saturations(values: &[f32], scale: f64) -> usize {
    values
        .iter()
        .filter(|&&v| {
            let q = (scale * v as f64).round();
            q < i16::MIN as f64 || q > i16::MAX as f64
        })
        .count()
}

/// FT テンソルの i16 量子化で飽和が起きていれば stderr に警告する。発生件数 0 なら
/// 無出力。飽和は `write_i16_quantised` が i16 範囲に clamp するため数値破綻には
/// ならないが、weight が i16 表現域 (±i16::MAX/QA) を超えて育った場合に silent な
/// 情報損失となるので気付けるようにする。
fn warn_if_i16_saturates(name: &str, values: &[f32], scale: f64) {
    let n = count_i16_saturations(values, scale);
    if n > 0 {
        let bound = i16::MAX as f64 / scale;
        eprintln!(
            "[nnue-format] warning: {name} has {n}/{} elements saturating i16 quantisation \
             (|w| > {bound:.4}); values are clamped on export (silent precision loss). \
             Consider tightening the training-time weight clamp for this tensor.",
            values.len()
        );
    }
}

/// `round(scale·v)` が学習時の対称 i8 clamp 端 `±127` に達した要素数を数える。
/// `scale` は dense weight の量子化スケール QB。Simple trainer は dense weight を
/// `±127/QB` に clamp するため、loss と出力 scale が不整合な run では範囲外へは出ず、
/// clamp 端に張り付く。export の表現域 `[-128, 127]` の外だけを数えるとその状態を
/// 見逃すので、`|round(scale·v)| >= 127` を飽和として扱う。
fn count_i8_clamp_saturations(values: &[f32], scale: f64) -> usize {
    values
        .iter()
        .filter(|&&v| {
            let q = (scale * v as f64).round();
            q <= -(i8::MAX as f64) || q >= i8::MAX as f64
        })
        .count()
}

/// dense 層の weight が学習時の i8 clamp 端 `±127/QB` に達していれば層名付きで警告する。
/// 件数 0 なら無出力。出力層で張り付き率が高いと、学習が要求する出力スケールを i8 dense
/// が構成できず、量子化ネットの評価値が破綻する (loss と dense clamp の不整合の兆候)。
fn warn_if_i8_saturates(name: &str, values: &[f32], scale: f64) {
    let n = count_i8_clamp_saturations(values, scale);
    if n > 0 {
        let pct = 100.0 * n as f64 / values.len() as f64;
        eprintln!(
            "[nnue-format] warning: {name} has {n}/{} ({pct:.1}%) elements at or beyond the \
             symmetric int8 training clamp (|round(scale·w)| >= 127). A large saturating \
             fraction in the output layer means the trained weights cannot form the requested \
             evaluation scale (loss / dense clamp scale mismatch).",
            values.len()
        );
    }
}

/// `n` 個の i16 (raw LE) を読み、`scale` で割って f32 列に戻す。
fn read_i16_quantised<R: Read>(reader: &mut R, n: usize, scale: f32) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 2];
    for _ in 0..n {
        reader.read_exact(&mut buf)?;
        out.push(i16::from_le_bytes(buf) as f32 / scale);
    }
    Ok(out)
}

/// bias 列を `scale` 倍して i32 量子化し書く。
fn write_i32_bias<W: Write>(writer: &mut W, values: &[f32], scale: f64) -> io::Result<()> {
    for &v in values {
        let q = clamp_round(scale * v as f64, i32::MIN as f64, i32::MAX as f64) as i32;
        writer.write_all(&q.to_le_bytes())?;
    }
    Ok(())
}

/// `n` 個の i32 bias を読み、`scale` で割って戻す。
fn read_i32_bias<R: Read>(reader: &mut R, n: usize, scale: f32) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        reader.read_exact(&mut buf)?;
        out.push(i32::from_le_bytes(buf) as f32 / scale);
    }
    Ok(out)
}

/// row-major weight `(out_dim, in_dim)` を `pad32(in_dim)` 幅に pad しながら
/// `scale` 倍して i8 量子化し書く。padding byte は 0。
fn write_i8_weight<W: Write>(
    writer: &mut W,
    weights: &[f32],
    out_dim: usize,
    in_dim: usize,
    scale: f64,
) -> io::Result<()> {
    let padded_in = pad32(in_dim);
    for out in 0..out_dim {
        for in_idx in 0..padded_in {
            let q: i8 = if in_idx < in_dim {
                clamp_round(scale * weights[out * in_dim + in_idx] as f64, -128.0, 127.0) as i8
            } else {
                0
            };
            writer.write_all(&[q as u8])?;
        }
    }
    Ok(())
}

/// `pad32(in_dim)` 幅で書かれた i8 weight を読み、padding を捨て row-major
/// `(out_dim, in_dim)` の f32 に戻す。
fn read_i8_weight<R: Read>(
    reader: &mut R,
    out_dim: usize,
    in_dim: usize,
    scale: f32,
) -> io::Result<Vec<f32>> {
    let padded_in = pad32(in_dim);
    let mut out = vec![0.0_f32; out_dim * in_dim];
    let mut buf = [0u8; 1];
    for o in 0..out_dim {
        for in_idx in 0..padded_in {
            reader.read_exact(&mut buf)?;
            if in_idx < in_dim {
                out[o * in_dim + in_idx] = buf[0] as i8 as f32 / scale;
            }
        }
    }
    Ok(out)
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_features::FeatureSet;

    #[test]
    fn count_i16_saturations_counts_only_out_of_range() {
        // scale = QA = 127。|round(127·w)| > 32767 ⇔ |w| ≳ 258.0 のみ飽和。
        let qa = FT_OUTPUT_QA as f64;
        assert_eq!(
            count_i16_saturations(&[0.0, 1.98, -1.98, 100.0, -257.9], qa),
            0
        );
        assert_eq!(count_i16_saturations(&[300.0, -300.0, 1.0], qa), 2);
        assert_eq!(count_i16_saturations(&[], qa), 0);
    }

    #[test]
    fn count_i8_clamp_saturations_flags_boundary_pinned_dense_weights() {
        // dense weight scale = QB = 64。clamp 境界は |w| ≈ 127/64 ≈ 1.98
        // 正常な dense weight は 0 件、loss / dense clamp 不整合で境界へ張り付いた
        // weight は export 範囲内の ±127 でも飽和として数える。
        let qb = QB as f64;
        assert_eq!(count_i8_clamp_saturations(&[0.0, 1.97, -1.97, 0.5], qb), 0);
        assert_eq!(
            count_i8_clamp_saturations(&[127.0 / 64.0, -127.0 / 64.0, 1.0], qb),
            2
        );
        assert_eq!(count_i8_clamp_saturations(&[], qb), 0);
    }

    #[test]
    fn count_i8_clamp_saturations_boundary_exact() {
        // scale=1.0 で w を量子化値そのものに使う。±126 は未飽和、±127 は学習 clamp
        // 端なので飽和。round 後判定も固定する: 126.4→126、126.5→127。
        assert_eq!(count_i8_clamp_saturations(&[126.0, -126.0], 1.0), 0);
        assert_eq!(count_i8_clamp_saturations(&[127.0, -127.0], 1.0), 2);
        assert_eq!(count_i8_clamp_saturations(&[126.4], 1.0), 0);
        assert_eq!(count_i8_clamp_saturations(&[126.5], 1.0), 1);
    }

    #[test]
    fn count_i16_saturations_boundary_exact() {
        // round(scale·w) が i16 端点ちょうどに乗るケース。scale=1.0 で w を量子化値
        // そのものに使う。32767 / -32768 は範囲内 (0 件)、32768 / -32769 は飽和。
        assert_eq!(count_i16_saturations(&[32767.0, -32768.0], 1.0), 0);
        assert_eq!(count_i16_saturations(&[32768.0, -32769.0], 1.0), 2);
        // round 後判定: 32767.4→32767 (範囲内)、32767.5→32768 (飽和)。
        assert_eq!(count_i16_saturations(&[32767.4], 1.0), 0);
        assert_eq!(count_i16_saturations(&[32767.5], 1.0), 1);
    }

    #[test]
    fn count_i16_saturations_screlu_scale_255() {
        // SCReLU は FT scale = QA = 255。飽和境界は |w| ≈ 32767/255 ≈ 128.5。
        // 同じ |w| でも scale=127 では飽和しないが scale=255 では飽和しうる。
        let qa = SimpleActivation::SCReLU.qa() as f64;
        assert_eq!(qa, 255.0);
        // round(255·128.4)=32742 ≤ 32767 で範囲内、round(255·129.0)=32895 で飽和。
        assert_eq!(count_i16_saturations(&[128.4, -128.4], qa), 0);
        assert_eq!(count_i16_saturations(&[129.0, -129.0], qa), 2);
        // 端点ちょうど: 32767/255 は範囲内、32768/255 は飽和。
        assert_eq!(count_i16_saturations(&[(32767.0 / 255.0) as f32], qa), 0);
        assert_eq!(count_i16_saturations(&[(32768.0 / 255.0) as f32], qa), 1);
    }

    fn test_id(activation: SimpleActivation) -> SimpleId {
        SimpleId {
            feature_set: FeatureSet::HalfKaHmMerged.spec(),
            activation,
            ft_out: 256,
            l1_out: 32,
            l2_out: 32,
        }
    }

    /// 256x2-32-32 / CReLU の Simple ネットで、YaneuraOu が評価関数ファイルの照合に使う
    /// 3 つの値 (network hash / FT hash / 本体のバイト数) を固定する。
    ///
    /// 期待値は YaneuraOu 形式で配布されている K+P と HalfKP の 256x2-32-32 の評価関数
    /// ファイルのヘッダから読んだ値。YaneuraOu はアーキ文字列を解釈せず `network hash` だけを
    /// 照合するので、この 3 つが合えば同じ形で読み込まれる。本体は arch 文字列以降
    /// (`ft_hash` から末尾まで) で、arch 文字列の長さは含まない。
    fn assert_matches_distributed_net(
        feature_set: FeatureSet,
        network_hash_want: u32,
        ft_hash_want: u32,
        body_len_want: usize,
    ) {
        let id = SimpleId {
            feature_set: feature_set.spec(),
            activation: SimpleActivation::CReLU,
            ft_out: 256,
            l1_out: 32,
            l2_out: 32,
        };
        assert_eq!(network_hash(&id), network_hash_want, "network hash");
        assert_eq!(
            ft_hash(id.feature_set.feature_hash(), 256),
            ft_hash_want,
            "ft hash"
        );

        let weights = SimpleWeights::zeroed(id, 28);
        let mut bytes = Vec::new();
        weights.save_quantised(&mut bytes).unwrap();
        let arch_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(bytes.len() - (12 + arch_len), body_len_want, "body length");
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            network_hash_want
        );
        assert_eq!(
            u32::from_le_bytes(bytes[12 + arch_len..16 + arch_len].try_into().unwrap()),
            ft_hash_want
        );
    }

    #[test]
    fn kp_256x2_32_32_header_matches_the_distributed_net() {
        assert_matches_distributed_net(FeatureSet::Kp, 0x5C64_64A9, 0x3F57_15FF, 893_740);
    }

    #[test]
    fn halfkp_256x2_32_32_header_matches_the_distributed_net() {
        assert_matches_distributed_net(FeatureSet::HalfKp, 0x3E5A_A6EE, 0x5D69_D7B8, 64_216_876);
    }

    #[test]
    fn kp_arch_string_has_no_perspective_marker() {
        let id = SimpleId {
            feature_set: FeatureSet::Kp.spec(),
            activation: SimpleActivation::CReLU,
            ft_out: 256,
            l1_out: 32,
            l2_out: 32,
        };
        assert!(build_arch_str(&id, 28).starts_with("Features=K+P[1710->256x2],Network="));
        let halfkp = SimpleId {
            feature_set: FeatureSet::HalfKp.spec(),
            ..id
        };
        assert!(build_arch_str(&halfkp, 28).starts_with("Features=HalfKP(Friend)[125388->256x2],"));
    }

    #[test]
    fn simple_fv_scale_uses_ft_output_qa() {
        // fv_scale = round(FT_OUTPUT_QA(127) × QB(64) / scale)、活性化に非依存。
        // scale=290 → round(8128/290) = round(28.03) = 28
        assert_eq!(simple_fv_scale(290.0), 28);
        // scale=600 → round(8128/600) = round(13.55) = 14
        assert_eq!(simple_fv_scale(600.0), 14);
        // 回帰防止: 旧実装は SCReLU で QA=255 を使い scale=290 で 56 になっていた。
        assert_ne!(simple_fv_scale(290.0), 56);
    }

    #[test]
    fn pad32_correct() {
        assert_eq!(pad32(0), 0);
        assert_eq!(pad32(1), 32);
        assert_eq!(pad32(8), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(33), 64);
        assert_eq!(pad32(512), 512);
    }

    /// 全 [`SimpleActivation`] variant (テストの網羅対象)。
    const ALL_ACTIVATIONS: [SimpleActivation; 3] = [
        SimpleActivation::CReLU,
        SimpleActivation::SCReLU,
        SimpleActivation::Pairwise,
    ];

    #[test]
    fn activation_name_round_trips() {
        for act in ALL_ACTIVATIONS {
            assert_eq!(
                SimpleActivation::from_canonical_name(act.canonical_name()),
                Some(act)
            );
        }
        assert_eq!(SimpleActivation::from_canonical_name("bogus"), None);
    }

    #[test]
    fn combined_dim_halves_for_pairwise() {
        // CReLU / SCReLU は L1 入力 = 2*ft_out、Pairwise は ft_out (pairwise 乗算で半減)。
        assert_eq!(test_id(SimpleActivation::CReLU).combined_dim(), 512);
        assert_eq!(test_id(SimpleActivation::SCReLU).combined_dim(), 512);
        assert_eq!(test_id(SimpleActivation::Pairwise).combined_dim(), 256);
    }

    #[test]
    fn zeroed_save_load_round_trips() {
        for act in ALL_ACTIVATIONS {
            let id = test_id(act);
            let original = SimpleWeights::zeroed(id, 13);
            let mut buf = Vec::new();
            original.save_quantised(&mut buf).unwrap();
            let loaded =
                SimpleWeights::load(&mut std::io::Cursor::new(&buf), id).expect("load zeroed");

            assert_eq!(loaded.id, id);
            assert_eq!(loaded.fv_scale, Some(13));
            assert_eq!(loaded.ft_w.len(), id.ft_in() * id.ft_out);
            assert_eq!(loaded.l1_w.len(), id.l1_out * id.combined_dim());
            assert_eq!(loaded.l2_w.len(), id.l2_out * id.l1_out);
            assert_eq!(loaded.l3_w.len(), id.l2_out);
            assert_eq!(loaded.l3_b.len(), 1);
            assert!(loaded.ft_w.iter().all(|&x| x == 0.0));
            assert!(loaded.l1_w.iter().all(|&x| x == 0.0));
            assert!(loaded.l3_b.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn quantisation_grid_values_round_trip_exactly() {
        // 量子化格子上の値 (k/QA, k/QB, k/(127*QB)) は厳密に round-trip する。
        let id = test_id(SimpleActivation::CReLU);
        let qa = id.activation.qa() as f32;
        let qb = QB as f32;
        let bias = (FT_OUTPUT_QA * QB) as f32;
        let mut w = SimpleWeights::zeroed(id, 16);
        for (i, v) in w.ft_w.iter_mut().enumerate() {
            *v = ((i % 7) as i32 - 3) as f32 / qa;
        }
        for (i, v) in w.ft_b.iter_mut().enumerate() {
            *v = ((i % 5) as i32 - 2) as f32 / qa;
        }
        for (i, v) in w.l1_w.iter_mut().enumerate() {
            *v = ((i % 9) as i32 - 4) as f32 / qb;
        }
        for (i, v) in w.l2_w.iter_mut().enumerate() {
            *v = ((i % 11) as i32 - 5) as f32 / qb;
        }
        for (i, v) in w.l3_w.iter_mut().enumerate() {
            *v = ((i % 13) as i32 - 6) as f32 / qb;
        }
        for (i, v) in w.l1_b.iter_mut().enumerate() {
            *v = ((i % 3) as i32 - 1) as f32 / bias;
        }
        w.l2_b[0] = 7.0 / bias;
        w.l3_b[0] = -9.0 / bias;

        let mut buf = Vec::new();
        w.save_quantised(&mut buf).unwrap();
        let loaded = SimpleWeights::load(&mut std::io::Cursor::new(&buf), id).unwrap();

        assert_eq!(loaded.ft_w, w.ft_w);
        assert_eq!(loaded.ft_b, w.ft_b);
        assert_eq!(loaded.l1_w, w.l1_w);
        assert_eq!(loaded.l1_b, w.l1_b);
        assert_eq!(loaded.l2_w, w.l2_w);
        assert_eq!(loaded.l2_b, w.l2_b);
        assert_eq!(loaded.l3_w, w.l3_w);
        assert_eq!(loaded.l3_b, w.l3_b);
    }

    #[test]
    fn load_rejects_feature_set_mismatch() {
        let merged = test_id(SimpleActivation::CReLU);
        let mut split = merged;
        split.feature_set = FeatureSet::HalfKaSplit.spec();
        let mut buf = Vec::new();
        SimpleWeights::zeroed(merged, 16)
            .save_quantised(&mut buf)
            .unwrap();
        let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), split)
            .expect_err("feature set mismatch must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_dimension_mismatch() {
        let id = test_id(SimpleActivation::CReLU);
        let mut wider = id;
        wider.l1_out = 64;
        let mut buf = Vec::new();
        SimpleWeights::zeroed(id, 16)
            .save_quantised(&mut buf)
            .unwrap();
        let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), wider)
            .expect_err("dimension mismatch must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_activation_mismatch() {
        // feature set / 層次元が同一でも活性化が異なれば reject (arch 文字列が示す活性化)。
        let crelu = test_id(SimpleActivation::CReLU);
        let screlu = test_id(SimpleActivation::SCReLU);
        let mut buf = Vec::new();
        SimpleWeights::zeroed(crelu, 16)
            .save_quantised(&mut buf)
            .unwrap();
        let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), screlu)
            .expect_err("activation mismatch must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_non_positive_fv_scale_in_arch_str() {
        let id = test_id(SimpleActivation::CReLU);
        for fv_scale in [0, -16] {
            let mut buf = Vec::new();
            SimpleWeights::zeroed(id, fv_scale)
                .save_quantised(&mut buf)
                .unwrap();

            let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), id)
                .expect_err("non-positive fv_scale must reject");
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);
            assert!(
                err.to_string().contains(&format!("`{fv_scale}`"))
                    && err.to_string().contains("must be positive"),
                "error should identify the invalid fv_scale: {err}"
            );
        }
    }

    #[test]
    fn load_rejects_truncated_input() {
        let id = test_id(SimpleActivation::CReLU);
        let mut buf = Vec::new();
        SimpleWeights::zeroed(id, 16)
            .save_quantised(&mut buf)
            .unwrap();
        buf.truncate(buf.len() / 2);
        assert!(SimpleWeights::load(&mut std::io::Cursor::new(&buf), id).is_err());
    }

    #[test]
    fn save_rejects_mismatched_buffer_length() {
        // weight buffer 長が id 次元と不整合なら save が InvalidInput で弾く。
        let id = test_id(SimpleActivation::CReLU);

        // 長すぎる buffer。
        let mut too_long = SimpleWeights::zeroed(id, 16);
        too_long.l1_b.push(0.0);
        assert_eq!(
            too_long
                .save_quantised(&mut Vec::new())
                .expect_err("too-long buffer must reject")
                .kind(),
            io::ErrorKind::InvalidInput
        );

        // 短すぎる buffer (検証が無ければ index panic / 末尾切り捨てになる経路)。
        let mut too_short = SimpleWeights::zeroed(id, 16);
        too_short.ft_w.pop();
        assert_eq!(
            too_short
                .save_quantised(&mut Vec::new())
                .expect_err("too-short buffer must reject")
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn load_rejects_invalid_utf8_arch_str() {
        // arch_str は version(4) + network_hash(4) + arch_len(4) の後ろから始まる。
        // 先頭バイトを不正 UTF-8 (0xFF) に潰すと load が InvalidData で reject する。
        let id = test_id(SimpleActivation::CReLU);
        let mut buf = Vec::new();
        SimpleWeights::zeroed(id, 16)
            .save_quantised(&mut buf)
            .unwrap();
        buf[12] = 0xFF;
        let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), id)
            .expect_err("invalid UTF-8 arch_str must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn file_size_matches_layout() {
        for act in ALL_ACTIVATIONS {
            let id = test_id(act);
            let mut buf = Vec::new();
            SimpleWeights::zeroed(id, 16)
                .save_quantised(&mut buf)
                .unwrap();

            let arch_len = build_arch_str(&id, 16).len();
            let header = 4 + 4 + 4 + arch_len; // version + network_hash + arch_len + arch_str
            let ft = 4 + (id.ft_out + id.ft_in() * id.ft_out) * 2; // ft_hash + i16 ft_b/ft_w
            let fc_hash = 4;
            let l1 = id.l1_out * 4 + id.l1_out * pad32(id.combined_dim());
            let l2 = id.l2_out * 4 + id.l2_out * pad32(id.l1_out);
            let l3 = 4 + pad32(id.l2_out);
            assert_eq!(
                buf.len(),
                header + ft + fc_hash + l1 + l2 + l3,
                "file size mismatch for {act:?}"
            );
        }
    }

    #[test]
    fn arch_str_contains_structured_fields() {
        let id = test_id(SimpleActivation::SCReLU);
        let s = build_arch_str(&id, 27);
        assert!(s.contains("HalfKaHmMerged(Friend)[73305->256x2]"));
        assert!(s.contains("SqrClippedReLU[32]"));
        assert!(s.contains("AffineTransformSparseInput[32<-512]"));
        assert!(s.contains("InputSlice[512(0:512)]"));
        assert!(s.ends_with(",fv_scale=27"));
    }

    #[test]
    fn arch_str_pairwise_uses_halved_l1_input_and_suffix() {
        // Pairwise は FT ブロックに `/2` と `-Pairwise` suffix を持ち、L1 入力次元
        // (AffineTransformSparseInput / InputSlice) が ft_out (= 256) に半減する。
        let id = test_id(SimpleActivation::Pairwise);
        let s = build_arch_str(&id, 27);
        assert!(s.contains("HalfKaHmMerged(Friend)[73305->256/2x2]-Pairwise"));
        assert!(s.contains("ClippedReLU[32]"));
        assert!(s.contains("AffineTransformSparseInput[32<-256]"));
        assert!(s.contains("InputSlice[256(0:256)]"));
        assert!(s.ends_with(",fv_scale=27"));
    }

    #[test]
    fn arch_str_parentheses_balanced() {
        // nnue-pytorch 系の arch 文字列は層の入れ子を括弧で表すため、
        // `(` と `)` の数が一致していなければ malformed。
        for act in ALL_ACTIVATIONS {
            let s = build_arch_str(&test_id(act), 16);
            let opens = s.matches('(').count();
            let closes = s.matches(')').count();
            assert_eq!(opens, closes, "arch_str parentheses unbalanced: `{s}`");
        }
    }

    // ---- 他の trainer が書いたファイルの受け入れ / 拒否 ----

    fn kp_id(
        activation: SimpleActivation,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
    ) -> SimpleId {
        SimpleId {
            feature_set: FeatureSet::Kp.spec(),
            activation,
            ft_out,
            l1_out,
            l2_out,
        }
    }

    /// `id` の零重みネットを `fv_scale` 付きで書き出した bytes。
    fn saved(id: SimpleId, fv_scale: i32) -> Vec<u8> {
        let mut buf = Vec::new();
        SimpleWeights::zeroed(id, fv_scale)
            .save_quantised(&mut buf)
            .unwrap();
        buf
    }

    /// arch 文字列だけを `new_arch` に差し替えた bytes。長さ欄も合わせて書き換える。
    fn with_arch_str(bytes: &[u8], new_arch: &str) -> Vec<u8> {
        let old_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut out = bytes[..8].to_vec();
        out.extend_from_slice(&(new_arch.len() as u32).to_le_bytes());
        out.extend_from_slice(new_arch.as_bytes());
        out.extend_from_slice(&bytes[12 + old_len..]);
        out
    }

    fn arch_str_of(bytes: &[u8]) -> String {
        let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        String::from_utf8(bytes[12..12 + len].to_vec()).unwrap()
    }

    fn load_bytes(bytes: &[u8], expected: SimpleId) -> io::Result<SimpleWeights> {
        SimpleWeights::load(&mut std::io::Cursor::new(bytes), expected)
    }

    /// `arch_str` を持つ file を `expected` で読み、受け入れて警告も無いことを確かめる。
    fn assert_accepted_silently(bytes: &[u8], expected: SimpleId) -> SimpleWeights {
        let arch_str = arch_str_of(bytes);
        let report = inspect_arch_text(&arch_str, &expected).unwrap();
        assert_eq!(report.difference, None, "arch_str: {arch_str}");
        load_bytes(bytes, expected).unwrap_or_else(|e| panic!("`{arch_str}` must load: {e}"))
    }

    #[test]
    fn load_accepts_model_type_prefix() {
        let id = kp_id(SimpleActivation::CReLU, 256, 32, 32);
        let canonical = build_arch_str(&id, 28);
        let bytes = with_arch_str(&saved(id, 28), &format!("ModelType=Standard;{canonical}"));
        let loaded = assert_accepted_silently(&bytes, id);
        assert_eq!(loaded.fv_scale, Some(28));
    }

    #[test]
    fn load_accepts_plain_affine_transform_for_the_sparse_input_layer() {
        let id = test_id(SimpleActivation::CReLU);
        let canonical = build_arch_str(&id, 28);
        assert!(canonical.contains("AffineTransformSparseInput[32<-512]"));
        let plain = canonical.replace("AffineTransformSparseInput[", "AffineTransform[");
        let bytes = with_arch_str(&saved(id, 28), &plain);
        let loaded = assert_accepted_silently(&bytes, id);
        assert_eq!(loaded.fv_scale, Some(28));
    }

    #[test]
    fn load_accepts_a_missing_fv_scale_and_does_not_invent_one() {
        for act in ALL_ACTIVATIONS {
            let id = test_id(act);
            let bytes = with_arch_str(&saved(id, 28), &arch_identity(&id));
            let loaded = assert_accepted_silently(&bytes, id);
            assert_eq!(loaded.fv_scale, None, "{act:?}");
            assert_eq!(loaded.id, id);
        }
    }

    #[test]
    fn load_accepts_all_tolerated_differences_together() {
        // 配布されている 128x2-32-32 の HalfKP は `ModelType=` 付き・素の
        // `AffineTransform`・`fv_scale` 無しで書かれている。
        let id = SimpleId {
            feature_set: FeatureSet::HalfKp.spec(),
            activation: SimpleActivation::CReLU,
            ft_out: 128,
            l1_out: 32,
            l2_out: 32,
        };
        let foreign = format!(
            "ModelType=Standard;{}",
            arch_identity(&id).replace("AffineTransformSparseInput[", "AffineTransform[")
        );
        assert!(foreign.starts_with("ModelType=Standard;Features=HalfKP(Friend)[125388->128x2]"));
        let bytes = with_arch_str(&saved(id, 28), &foreign);
        let loaded = assert_accepted_silently(&bytes, id);
        assert_eq!(loaded.fv_scale, None);
    }

    #[test]
    fn load_accepts_a_misdescribed_arch_str_and_reports_the_difference() {
        // 768x2-16-64 の本体に 256x2-32-32 の文字列が付いたファイル。hash と本体の長さは
        // 768x2-16-64 のものなので読み込めるが、文字列の違いは報告される。
        let id = kp_id(SimpleActivation::CReLU, 768, 16, 64);
        let wrong_text = arch_identity(&kp_id(SimpleActivation::CReLU, 256, 32, 32));
        let bytes = with_arch_str(&saved(id, 28), &wrong_text);

        let report = inspect_arch_text(&arch_str_of(&bytes), &id).unwrap();
        let difference = report.difference.expect("a dimension misprint is reported");
        assert!(
            difference.contains("[1710->256x2]") && difference.contains("[1710->768x2]"),
            "the report must show both dimension texts: {difference}"
        );
        assert_eq!(report.fv_scale, None);

        let loaded = load_bytes(&bytes, id).expect("hashes and length match, so it loads");
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.fv_scale, None);
        assert_eq!(loaded.ft_w.len(), 1710 * 768);
    }

    #[test]
    fn tolerated_differences_do_not_hide_other_differences() {
        // 許容する 3 種の違いに加えて次元の誤記があれば、差異は報告される。
        let id = kp_id(SimpleActivation::CReLU, 768, 16, 64);
        let wrong = arch_identity(&kp_id(SimpleActivation::CReLU, 256, 32, 32))
            .replace("AffineTransformSparseInput[", "AffineTransform[");
        let report = inspect_arch_text(&format!("ModelType=Standard;{wrong}"), &id).unwrap();
        assert!(report.difference.is_some());
    }

    #[test]
    fn load_rejects_a_wrong_version() {
        let id = test_id(SimpleActivation::CReLU);
        let mut bytes = saved(id, 28);
        bytes[..4].copy_from_slice(&(NNUE_VERSION ^ 1).to_le_bytes());
        let err = load_bytes(&bytes, id).expect_err("wrong version must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("version"), "{err}");
    }

    #[test]
    fn load_rejects_each_wrong_hash() {
        let id = kp_id(SimpleActivation::CReLU, 256, 32, 32);
        let good = saved(id, 28);
        let arch_len = u32::from_le_bytes(good[8..12].try_into().unwrap()) as usize;
        let ft_hash_at = 12 + arch_len;
        let fc_hash_at = ft_hash_at + 4 + 2 * (id.ft_out + id.ft_in() * id.ft_out);
        for (name, at) in [
            ("network_hash", 4),
            ("ft_hash", ft_hash_at),
            ("fc_hash", fc_hash_at),
        ] {
            let mut bytes = good.clone();
            bytes[at] ^= 0x01;
            let err = load_bytes(&bytes, id).expect_err("a wrong hash must reject");
            assert_eq!(err.kind(), io::ErrorKind::InvalidData, "{name}");
            assert!(err.to_string().contains(name), "{name}: {err}");
        }
    }

    #[test]
    fn load_rejects_wrong_body_length() {
        let id = kp_id(SimpleActivation::CReLU, 256, 32, 32);
        let good = saved(id, 28);
        assert_eq!(good.len() - (12 + arch_str_of(&good).len()), body_len(&id));
        assert!(load_bytes(&good, id).is_ok());

        // 1 バイト足りない、または 1 バイト余るファイル。
        let short = &good[..good.len() - 1];
        let err = load_bytes(short, id).expect_err("a short body must reject");
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);

        let mut long = good.clone();
        long.push(0);
        let err = load_bytes(&long, id).expect_err("trailing bytes must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("trailing bytes"), "{err}");
    }

    #[test]
    fn load_rejects_wrong_body_length_even_with_tolerated_text() {
        let id = test_id(SimpleActivation::CReLU);
        let mut bytes = with_arch_str(&saved(id, 28), &arch_identity(&id));
        bytes.extend_from_slice(&[0; 32]);
        assert!(load_bytes(&bytes, id).is_err());
    }

    #[test]
    fn load_rejects_an_activation_the_text_contradicts() {
        for (written, wanted) in [
            (SimpleActivation::SCReLU, SimpleActivation::CReLU),
            (SimpleActivation::CReLU, SimpleActivation::SCReLU),
        ] {
            let bytes = saved(test_id(written), 28);
            let err = load_bytes(&bytes, test_id(wanted)).expect_err("activation must reject");
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);
            assert!(err.to_string().contains("activation mismatch"), "{err}");

            // fv_scale が無くても、許容する違いがあっても結果は変わらない。
            let bare = with_arch_str(
                &bytes,
                &format!("ModelType=Standard;{}", arch_identity(&test_id(written))),
            );
            assert!(load_bytes(&bare, test_id(wanted)).is_err());
        }
    }

    #[test]
    fn load_rejects_an_arch_str_naming_both_squared_relu_and_pairwise() {
        let id = test_id(SimpleActivation::Pairwise);
        let both = format!("{},SqrClippedReLU[32]", arch_identity(&id));
        let bytes = with_arch_str(&saved(id, 28), &both);
        let err = load_bytes(&bytes, id).expect_err("an ambiguous activation must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_a_pairwise_net_read_as_crelu_and_the_reverse() {
        // Pairwise は L1 入力次元が半減するので hash が変わる。文字列を書き換えても通らない。
        let pairwise = test_id(SimpleActivation::Pairwise);
        let crelu = test_id(SimpleActivation::CReLU);
        let bytes = saved(pairwise, 28);
        let relabelled = with_arch_str(&bytes, &build_arch_str(&crelu, 28));
        assert!(load_bytes(&relabelled, crelu).is_err());
        assert!(load_bytes(&saved(crelu, 28), pairwise).is_err());
    }

    #[test]
    fn load_rejects_a_different_feature_set_or_dimension_whatever_the_text_says() {
        let kp = kp_id(SimpleActivation::CReLU, 256, 32, 32);
        let halfkp = SimpleId {
            feature_set: FeatureSet::HalfKp.spec(),
            ..kp
        };
        let kp_bytes = saved(kp, 28);
        assert!(load_bytes(&kp_bytes, halfkp).is_err());
        // 文字列を期待する形に書き換えても hash が違うので落ちる。
        let relabelled = with_arch_str(&kp_bytes, &build_arch_str(&halfkp, 28));
        let err = load_bytes(&relabelled, halfkp).expect_err("feature set must reject");
        assert!(err.to_string().contains("network_hash"), "{err}");

        for narrower in [
            SimpleId { ft_out: 128, ..kp },
            SimpleId { l1_out: 16, ..kp },
            SimpleId { l2_out: 64, ..kp },
        ] {
            let relabelled = with_arch_str(&kp_bytes, &build_arch_str(&narrower, 28));
            assert!(load_bytes(&relabelled, narrower).is_err(), "{narrower:?}");
        }
    }

    #[test]
    fn load_rejects_a_malformed_fv_scale_even_if_everything_else_matches() {
        let id = test_id(SimpleActivation::CReLU);
        for suffix in [",fv_scale=abc", ",fv_scale=", ",fv_scale=0", ",fv_scale=-3"] {
            let bytes = with_arch_str(&saved(id, 28), &format!("{}{suffix}", arch_identity(&id)));
            let err = load_bytes(&bytes, id).expect_err(suffix);
            assert_eq!(err.kind(), io::ErrorKind::InvalidData, "{suffix}");
        }
    }

    #[test]
    fn saving_a_net_without_fv_scale_writes_no_token_and_reloads_identically() {
        let id = test_id(SimpleActivation::SCReLU);
        let mut weights = SimpleWeights::zeroed(id, 28);
        weights.fv_scale = None;
        let mut bytes = Vec::new();
        weights.save_quantised(&mut bytes).unwrap();
        assert!(!arch_str_of(&bytes).contains("fv_scale"));

        let loaded = assert_accepted_silently(&bytes, id);
        assert_eq!(loaded.fv_scale, None);
        let mut again = Vec::new();
        loaded.save_quantised(&mut again).unwrap();
        assert_eq!(again, bytes);
    }

    #[test]
    fn a_file_with_fv_scale_loads_exactly_as_before() {
        for act in ALL_ACTIVATIONS {
            let id = test_id(act);
            let bytes = saved(id, 41);
            let loaded = assert_accepted_silently(&bytes, id);
            assert_eq!(loaded.fv_scale, Some(41), "{act:?}");
        }
    }

    #[test]
    fn fc_hash_depends_only_on_dimensions() {
        // topology hash は層次元 (l1_input / l1_out / l2_out) のみに依存し、
        // feature set / 活性化に依らない。
        let a = compute_fc_hash(256, 32, 32);
        assert_eq!(a, compute_fc_hash(256, 32, 32));
        assert_ne!(a, compute_fc_hash(512, 32, 32));
        assert_ne!(a, compute_fc_hash(256, 64, 32));
    }
}
