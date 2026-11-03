#!/usr/bin/env bash
# fork (origin) に上流 (upstream) の変更を取り込む。手順の全体は CLAUDE.md の
# 「上流 (SH11235/tatara) の取り込み」にある。
#
#     scripts/sync-upstream.sh status    上流にだけあるコミット、fork の変更と重なるファイル、
#                                        merge の事前試算を表示する (作業ツリーは変えない)
#     scripts/sync-upstream.sh prepare   origin/master から sync/upstream-<sha> を切り、
#                                        upstream/main をマージコミットで取り込む
#
# 比べる ref は FORK_REF (既定 origin/master) と UPSTREAM_REF (既定 upstream/main) で
# 差し替えられる。衝突時の表示を、実際の履歴を触らずに試すときに使う。
set -euo pipefail

FORK_REPO=rshogi/tatara
UPSTREAM_REPO=SH11235/tatara
FORK_REF=${FORK_REF:-origin/master}
UPSTREAM_REF=${UPSTREAM_REF:-upstream/main}

cd "$(dirname "$0")/.."

mode=${1:-}
case "$mode" in
    status | prepare) ;;
    *)
        echo "usage: $0 status|prepare" >&2
        exit 2
        ;;
esac

# 別のリポジトリで動かして、関係の無い branch を取り込まないための確認。
origin_url=$(git remote get-url origin)
case "$origin_url" in
    *"$FORK_REPO"*) ;;
    *)
        echo "origin が $FORK_REPO ではない: $origin_url" >&2
        exit 2
        ;;
esac
if git remote get-url upstream >/dev/null 2>&1; then
    case "$(git remote get-url upstream)" in
        *"$UPSTREAM_REPO"*) ;;
        *)
            echo "upstream が $UPSTREAM_REPO ではない: $(git remote get-url upstream)" >&2
            exit 2
            ;;
    esac
else
    git remote add upstream "https://github.com/$UPSTREAM_REPO.git"
fi

git fetch --quiet --prune origin
git fetch --quiet --prune upstream

fork=$FORK_REF
up=$UPSTREAM_REF
base=$(git merge-base "$fork" "$up")
count=$(git rev-list --count "$fork..$up")

if [ "$count" -eq 0 ]; then
    echo "取り込むコミットは無い ($fork は $up を含んでいる)"
    exit 0
fi

# 上流が変えたときに、取り込みの前後で追加の作業が要るファイル。"path|理由"。
watch=(
    ".github/workflows/checks.yaml|fork では integration.yaml に改めている。同じ変更を integration.yaml の checks job に写す"
    "CLAUDE.md|branch 名は master のまま、上流の追記だけ取り込む"
    "rust-toolchain.toml|ローカルに新しい nightly を入れる"
    "crates/cuda-native-runtime/kernels|CUDA kernel が変わった。kp と halfkp の smoke と短い学習を走らせる"
    "crates/shogi-features/src/feature_set.rs|fork の kp との共存を確認する (FeatureSet::ALL の長さと match の腕)"
    "Dockerfile|fork の Dockerfile と衝突する"
)

echo "上流にだけあるコミット ($count 件):"
git log --format='  %h %s' "$fork..$up"

echo
upstream_files=$(git diff --name-only "$base" "$up")
echo "上流が変えたファイル: $(printf '%s\n' "$upstream_files" | grep -c .) 個"
overlap=$(comm -12 <(printf '%s\n' "$upstream_files" | sort) <(git diff --name-only "$base" "$fork" | sort))
if [ -n "$overlap" ]; then
    echo "fork の変更と同じファイル (自動で合わさることが多いが、結果は読む):"
    printf '%s\n' "$overlap" | sed 's/^/  /'
fi

echo
echo "追加の作業が要るもの:"
found=0
for entry in "${watch[@]}"; do
    path=${entry%%|*}
    reason=${entry#*|}
    if ! git diff --quiet "$base" "$up" -- "$path"; then
        echo "  - $path: $reason"
        found=1
    fi
done
[ "$found" -eq 1 ] || echo "  (なし)"

echo
if trial=$(git merge-tree --write-tree --name-only "$fork" "$up" 2>&1); then
    echo "merge の事前試算: 衝突なし"
else
    echo "merge の事前試算: 衝突あり。衝突するファイル:"
    printf '%s\n' "$trial" | sed -n '2,/^$/p' | sed '/^$/d; s/^/  /'
fi

[ "$mode" = prepare ] || exit 0

# 未追跡ファイルは数えない。merge が上書きする場合は git が自分で止める。
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "追跡しているファイルに変更がある。commit か stash してから prepare を実行する" >&2
    exit 2
fi
branch=sync/upstream-$(git rev-parse --short "$up")
if git show-ref --verify --quiet "refs/heads/$branch"; then
    echo "branch $branch が既にある。続きをやるか、不要なら消してから実行する" >&2
    exit 2
fi

message=$(mktemp)
trap 'rm -f "$message"' EXIT
{
    echo "upstream: $UPSTREAM_REPO ${up#upstream/} ($(git rev-parse --short "$up")) を取り込む"
    echo
    echo "取り込むコミット:"
    git log --format='- %h %s' "$fork..$up"
} >"$message"

git switch --quiet -c "$branch" "$fork"
echo
if git merge --no-ff -F "$message" "$up"; then
    echo
    echo "取り込みが済んだ: $branch"
    echo "次は検証、push、PR の順 (CLAUDE.md の手順 3 以降)"
else
    echo
    echo "衝突した。衝突を解消して git add のあと git commit する (merge の続き)。" >&2
    echo "やめるときは git merge --abort のあと git switch - && git branch -D $branch" >&2
    exit 1
fi
