# Trainer image: `nnue-train` and the small tools built from the same workspace.
#
#   docker build -t tatara .
#   docker run --rm --gpus all tatara simple        # GPU smoke test, no data needed
#   docker run --rm --gpus all tatara --help
#
# The build needs no GPU, only nvcc. The container needs a GPU at run time,
# which the NVIDIA container runtime provides together with the driver.
ARG CUDA_VERSION=12.9.1
ARG UBUNTU=ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build

# `git` is not used to fetch anything: bins/nnue_train/build.rs runs it to stamp
# the commit and a dirty flag into the binary (`nnue-train build-info`), and the
# rescore sidecar fingerprint includes that commit. The build context therefore
# has to be a clean checkout that carries its `.git` directory. Without `.git`
# the stamp is `unknown`; with files changed or missing from the context it is
# `<sha>-dirty`. `.dockerignore` may only list paths that git ignores for the
# same reason.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# `--default-toolchain none`: the pinned nightly is installed below, from the
# channel named in rust-toolchain.toml.
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain none --profile minimal --no-modify-path
ENV PATH=/root/.cargo/bin:$PATH

# rust-toolchain.toml also lists rust-src, rustc-dev, rust-analyzer, clippy and
# llvm-tools. They serve the opt-in cuda-oxide backend and development; the
# default `native` backend needs none of them, and honouring the file would
# download about a gigabyte of components. Setting RUSTUP_TOOLCHAIN overrides
# the file, so only the channel is read from it, here and in the build step.
# This layer depends on that one file and stays cached across source changes.
COPY rust-toolchain.toml /tmp/rust-toolchain.toml
RUN channel="$(sed -nE 's/^channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' /tmp/rust-toolchain.toml)" \
 && test -n "${channel}" \
 && rustup toolchain install "${channel}" --profile minimal --no-self-update \
 && printf '%s\n' "${channel}" > /opt/rust-channel

# nvcc compiles the kernels for one virtual architecture and embeds them as PTX
# only; the driver JIT-compiles that PTX for the GPU it finds, so any GPU of this
# generation or newer runs it. `docker build` has no GPU for build.rs to query
# with nvidia-smi, hence the explicit value. It must be numeric (`80`, not
# `sm_80` or `8.0`). Use 75 to run on Turing GPUs.
ARG TATARA_CUDA_COMPUTE=80
ENV TATARA_CUDA_COMPUTE=${TATARA_CUDA_COMPUTE}

COPY . /src
WORKDIR /src

# build.rs falls back to `unknown` when git cannot report the commit or the
# tree state, and the binary is then built without saying so. Fail here instead.
# A modified tree is allowed (the stamp says `-dirty`) unless REQUIRE_CLEAN_TREE
# is true, which CI sets so that a stray file in the context cannot go unnoticed.
ARG REQUIRE_CLEAN_TREE=false
RUN if ! git rev-parse --verify HEAD >/dev/null 2>&1; then \
      echo "git cannot read a commit from the build context." >&2; \
      echo "Build from a checkout whose .git is a full directory: a source archive" >&2; \
      echo "has no .git, and a linked worktree's .git is a pointer file to a path" >&2; \
      echo "outside the context." >&2; \
      exit 1; \
    fi \
 && changes="$(git status --porcelain --untracked-files=normal)" \
 && if [ "${REQUIRE_CLEAN_TREE}" = true ] && [ -n "${changes}" ]; then \
      echo "the build context differs from its commit:" >&2; \
      echo "${changes}" | head -20 >&2; \
      exit 1; \
    fi

# Every workspace member, so the converters and the progress trainers come
# along with `nnue-train`. `--locked` fails on a Cargo.lock that has drifted
# from Cargo.toml instead of re-resolving it. No cache mounts: build.rs stamps
# the commit and tree state into `nnue-trainer`, and a `target/` carried over
# from an earlier build could keep a stamp that no longer matches the source.
# The whole build takes well under a minute.
RUN export RUSTUP_TOOLCHAIN="$(cat /opt/rust-channel)" \
 && cargo build --release --locked --workspace \
 && mkdir -p /out \
 && find target/release -maxdepth 1 -type f -perm -u+x ! -name '*.so' ! -name '*.d' \
      -exec cp {} /out/ \; \
 && ls -l /out \
 && test -x /out/nnue-train

# The binary links libcublas and reaches the driver through the CUDA Driver
# API; the NVIDIA container runtime injects the driver. The runtime image
# carries libcublas and nothing of the toolchain. Keep CUDA_VERSION at or below
# what the host driver supports.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
LABEL org.opencontainers.image.title="tatara" \
      org.opencontainers.image.description="Shogi NNUE trainer (nnue-train) and its converter tools" \
      org.opencontainers.image.licenses="MIT"
COPY --from=build /out/ /usr/local/bin/
WORKDIR /work

# `nnue-train` takes a subcommand (`simple` or `layerstack`) and has no sensible
# default. Other tools in /usr/local/bin run with `--entrypoint <name>`.
ENTRYPOINT ["nnue-train"]
