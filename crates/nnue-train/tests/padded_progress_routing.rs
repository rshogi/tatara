//! `BucketMode::ProgressKpAbsQ16Padded` が、読み込んだ進行度係数で局面を bucket 0..=7 に
//! だけ振り分け、9 stack 目 (bucket 8) へ一つも送らないことを確かめる。
//!
//! 進行度係数はプロセス全体で一つしか読み込めないので、この test は専用の integration test
//! binary にしてある (係数を読み込まない単体 test と同じ process に混ぜない)。

use nnue_train::dataloader::{BucketMode, BucketedPrefetchedLoader, PsvFileLoader};
use shogi_features::{FeatureSet, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs};
use shogi_format::{Color, ShogiBoard};

const SAMPLE: &[u8] = include_bytes!("../../shogi-format/tests/data/sample.psv");

/// 係数を再現可能な疑似乱数で作る。値域は、sample.psv の局面の logit 和が 8 つの bucket の
/// 大半に散らばる程度の広さに取る。
fn synthetic_coefficients() -> Vec<f64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 1.6
        })
        .collect()
}

/// 進行度 (0..=255) を、閾値 `round(ln(k / (256 - k)) * 65536)` を超えた個数として求め、
/// `進行度 * 8 / 256` で 8 段に落とす。実装の閾値表を使わない独立した参照実装。
fn reference_bucket(board: &ShogiBoard, coefficients: &[f64]) -> u8 {
    let mut sum = 0_i64;
    ShogiProgressKPAbs::for_each_active_index_board(board, |index| {
        sum += (coefficients[index] * 65536.0).round() as i64;
    });
    let progress = (1..256)
        .filter(|&k| {
            let p = f64::from(k) / 256.0;
            ((p / (1.0 - p)).ln() * 65536.0).round() as i64 <= sum
        })
        .count();
    (progress * 8 / 256) as u8
}

#[test]
fn padded_progress_routes_only_to_the_first_eight_stacks() {
    let coefficients = synthetic_coefficients();
    let bytes: Vec<u8> = coefficients.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShogiProgressKPAbs::load_from_bytes(&bytes).expect("load synthetic coefficients");

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("sample.psv");
    std::fs::write(&path, SAMPLE).expect("write sample PSV");

    // 局面ごとの参照 bucket。
    let mut reference_reader = PsvFileLoader::new(&path).expect("open sample PSV");
    let mut boards = Vec::new();
    while let Some(psv) = reference_reader.next_psv().expect("read sample PSV") {
        boards.push(psv.decode());
    }
    assert_eq!(boards.len(), SAMPLE.len() / 40);
    let expected: Vec<i32> = boards
        .iter()
        .map(|board| i32::from(reference_bucket(board, &coefficients)))
        .collect();

    // dataloader を通した bucket が参照実装と局面ごとに一致する。
    let mut loader = BucketedPrefetchedLoader::spawn(
        &path,
        boards.len(),
        None,
        None,
        1,
        BucketMode::ProgressKpAbsQ16Padded,
        FeatureSet::HalfKaHmMerged.spec(),
        true,
        9,
        std::fs::metadata(&path).expect("sample metadata").len(),
        false,
    )
    .expect("spawn padded-progress loader");
    let (batch, buckets) = loader
        .next_batch()
        .expect("load batch")
        .expect("full batch");
    assert_eq!(batch.n_positions, boards.len());
    assert_eq!(buckets, expected);

    // 係数が効いていて (中央の bucket 4 だけではない)、bucket 8 は空のまま。
    let used: std::collections::BTreeSet<i32> = buckets.iter().copied().collect();
    assert!(used.len() >= 4, "buckets used: {used:?}");
    assert!(
        used.iter().all(|&bucket| (0..8).contains(&bucket)),
        "{used:?}"
    );
    loader.recycle((batch, buckets));

    // 相入玉の局面 (先手玉が 5 段目以内、後手玉が 5 段目以降) でも bucket 8 へは行かない。
    // 玉の位置だけを差し替える (進行度の和は玉の位置にも依るので、参照実装で計算し直す)。
    let mut mutual_entering_king = 0;
    for board in &boards {
        let mut moved = board.clone();
        moved.side_to_move = Color::Black;
        moved.black_king_sq = shogi_format::types::Square::new(4, 3);
        moved.white_king_sq = shogi_format::types::Square::new(4, 5);
        let bucket = BucketMode::ProgressKpAbsQ16Padded.bucket_board(&moved, 9);
        assert!(bucket < 8, "bucket {bucket}");
        assert_eq!(bucket, reference_bucket(&moved, &coefficients));
        assert_eq!(
            BucketMode::ProgressKpAbsQ16EnteringKing.bucket_board(&moved, 9),
            8
        );
        mutual_entering_king += 1;
    }
    assert_eq!(mutual_entering_king, boards.len());
}
