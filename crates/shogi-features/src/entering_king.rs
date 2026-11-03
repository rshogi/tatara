//! LayerStack の相入玉 bucket 割当。

use shogi_format::ShogiBoard;

/// 相入玉 bucket を持つ割当の固定 bucket 数。進行度で分ける 8 bucket に、相入玉専用の
/// 1 bucket を加えた 9。
pub const ENTERING_KING_NUM_BUCKETS: usize = 9;

/// 相入玉局面を受ける bucket の index (最後の bucket)。
pub const ENTERING_KING_BUCKET: u8 = (ENTERING_KING_NUM_BUCKETS - 1) as u8;

/// 相入玉でない局面を進行度で分ける bucket 数 (index `0..ENTERING_KING_BUCKET`)。
pub const ENTERING_KING_PROGRESS_BUCKETS: usize = ENTERING_KING_NUM_BUCKETS - 1;

/// 双方の玉が盤の中段 (5 段目) までを含めて敵陣側へ進んでいるか。
///
/// `Square::rank` は 1 段目を 0 とする 0 始まりで、先手の敵陣は小さい側にある。
/// 先手玉が 5 段目以内 (`rank <= 4`) かつ後手玉が 5 段目以降 (`rank >= 4`) のとき真。
/// 玉がちょうど 5 段目にいる場合も含む。片方の玉だけが進んでいる局面は偽。
pub fn is_mutual_entering_king_board(board: &ShogiBoard) -> bool {
    debug_assert!(board.black_king_sq.is_valid());
    debug_assert!(board.white_king_sq.is_valid());

    board.black_king_sq.rank() <= 4 && board.white_king_sq.rank() >= 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_format::Color;
    use shogi_format::types::Square;

    /// 玉 2 枚だけの局面を SFEN の盤面部から作る。玉以外の駒は読み飛ばす。
    fn board_from_sfen(sfen: &str) -> ShogiBoard {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            ..Default::default()
        };
        let placement = sfen.split_whitespace().next().expect("SFEN board field");
        for (rank, row) in placement.split('/').enumerate() {
            let mut file_from_left = 0usize;
            for ch in row.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    file_from_left += empty as usize;
                    continue;
                }
                let file = 8 - file_from_left;
                match ch {
                    'K' => board.black_king_sq = Square::new(file as u8, rank as u8),
                    'k' => board.white_king_sq = Square::new(file as u8, rank as u8),
                    _ => {}
                }
                file_from_left += 1;
            }
            assert_eq!(file_from_left, 9, "SFEN rank must contain nine squares");
        }
        assert!(board.black_king_sq.is_valid(), "SFEN must contain K");
        assert!(board.white_king_sq.is_valid(), "SFEN must contain k");
        board
    }

    /// 先手玉を上から `black_row` 行目の左から `black_col` 列目 (どちらも 0 始まり)、
    /// 後手玉を `white_row` 行目の `white_col` 列目に置いた SFEN。
    fn sfen_with_kings(black: (usize, usize), white: (usize, usize)) -> String {
        let rows: Vec<String> = (0..9)
            .map(|row| {
                (0..9)
                    .map(|col| {
                        if (row, col) == black {
                            'K'
                        } else if (row, col) == white {
                            'k'
                        } else {
                            '1'
                        }
                    })
                    .collect()
            })
            .collect();
        format!("{} b - 1", rows.join("/"))
    }

    #[test]
    fn bucket_constants_are_consistent() {
        assert_eq!(ENTERING_KING_NUM_BUCKETS, 9);
        assert_eq!(ENTERING_KING_BUCKET, 8);
        assert_eq!(ENTERING_KING_PROGRESS_BUCKETS, 8);
    }

    #[test]
    fn fixtures_around_the_fifth_rank() {
        // (先手玉, 後手玉, 期待値)。座標は SFEN の (上からの行, 左からの列)。
        // 行 4 が 5 段目。
        let fixtures = [
            // どちらもちょうど 5 段目。
            ((4, 4), (4, 5), true),
            // 先手玉が 5 段目、後手玉がそれより敵陣側 (6 段目)。
            ((4, 4), (5, 4), true),
            // 先手玉が 4 段目、後手玉が 5 段目。
            ((3, 4), (4, 5), true),
            // 先手玉が 6 段目で止まっている。後手玉が 1 段目まで進んでいても偽。
            ((5, 4), (8, 4), false),
            // 後手玉が 4 段目で止まっている。先手玉が 1 段目まで進んでいても偽。
            ((0, 4), (3, 4), false),
            // 双方 5 段目から 1 段手前 (先手 6 段目、後手 4 段目) で互いに入れ違い。
            ((5, 4), (3, 4), false),
            // 平手初期配置の玉位置。
            ((8, 4), (0, 4), false),
        ];
        for (black, white, expected) in fixtures {
            let sfen = sfen_with_kings(black, white);
            assert_eq!(
                is_mutual_entering_king_board(&board_from_sfen(&sfen)),
                expected,
                "{sfen}"
            );
        }
    }

    #[test]
    fn matches_the_rank_rule_for_every_king_placement() {
        // 先手玉は上から 5 行目まで、後手玉は 5 行目から下、という条件を行番号で直接書く。
        let mut checked = 0usize;
        for black_row in 0..9 {
            for black_col in 0..9 {
                for white_row in 0..9 {
                    for white_col in 0..9 {
                        if (black_row, black_col) == (white_row, white_col) {
                            continue;
                        }
                        let sfen = sfen_with_kings((black_row, black_col), (white_row, white_col));
                        let expected = black_row <= 4 && white_row >= 4;
                        assert_eq!(
                            is_mutual_entering_king_board(&board_from_sfen(&sfen)),
                            expected,
                            "{sfen}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 81 * 80);
    }
}
