#!/usr/bin/env bash
#SBATCH -J dex_vla_critic_wam
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 04:00:00
#SBATCH -o logs/pace_gpu_%j.out
#SBATCH -e logs/pace_gpu_%j.err
# ==============================================================================
# Georgia Tech PACE-ICE Heavy GPU Suite for `dex-vla-critic-wam`
# 1. Configures scratch storage (`~/scratch/huggingface`, `~/scratch/capstone_datasets`)
#    so large datasets (`HenryZ07/RFT-reference-trajectory`, `birbirll/vega-pipette-fail`)
#    and VLM weights (`OpenGVLab/InternVL3-2B-hf`) never overflow the 30 GB home quota.
# 2. Runs 100-epoch GPU training of `DPINetWorldModel`, `PointTransformerWorldModel`,
#    `MultiViewObjectPoseNet`, and `Fast1HzNeuralCritic`.
# 3. Benchmarks the full 2B `OpenGVLab/InternVL3-2B-hf` in 4-bit NF4 (`bitsandbytes`)
#    with `torch.backends.cuda.enable_cudnn_sdp(False)` and Perceiver token caching.
# 4. Renders all four 720p MP4 showcase videos into `demo_videos/`.
# ==============================================================================
set -euo pipefail

SCRATCH="${SCRATCH:-$HOME/scratch}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_DIR="${ENV_DIR:-$SCRATCH/capstone_env}"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export VLM_CACHE_ROOT="$HF_HOME"
export UV_CACHE_DIR="${SCRATCH}/uv_cache"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/models" "$REPO_DIR/results" "$REPO_DIR/demo_videos" "$SCRATCH/capstone_datasets" "$HF_HOME" "$UV_CACHE_DIR"

module load python/3.12.5 2>/dev/null || module load anaconda3/2023.03 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda/12.6.1 2>/dev/null || true

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || pip install --user uv
if [ ! -d "$ENV_DIR" ]; then
    uv venv "$ENV_DIR" --python 3.12
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    uv pip install -r "$REPO_DIR/requirements.txt"
else
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
fi

echo "===================================================================="
echo " PACE-ICE Node: $(hostname)"
echo " GPU Assigned:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "===================================================================="

# 1. Extract bundled DexYCB 25-sequence benchmark pack directly on PACE-ICE (Zero SCP needed!)
DEXYCB_TARGET="$SCRATCH/capstone_datasets/dexycb"
if [ ! -d "$DEXYCB_TARGET/20200709-subject-01" ]; then
    echo "Extracting bundled DexYCB benchmark packs to $DEXYCB_TARGET ..."
    mkdir -p "$DEXYCB_TARGET"
    tar -xzf "$REPO_DIR/data/dexycb_pace_labels.tar.gz" -C "$DEXYCB_TARGET"
    tar -xzf "$REPO_DIR/data/dexycb_pace_rgb.tar.gz" -C "$DEXYCB_TARGET"
fi

# 2. Download RFT-reference-trajectory directly from HuggingFace on PACE-ICE's 10 Gbps network
python - <<'PY'
import os
from huggingface_hub import snapshot_download

scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
target = os.path.join(scratch, "capstone_datasets", "RFT-reference-trajectory")
os.makedirs(target, exist_ok=True)

# Download all 191 simulation logs + metadata + first 12 success & 10 failure episodes (fast ~20s download on PACE)
allow_patterns = [
    "meta/*",
    "logs/*.log",
    "failure at last/logs/*.log",
    "data/chunk-000/episode_00000*.parquet",
    "data/chunk-000/episode_00001[0-5].parquet",
    "failure at last/data/chunk-000/episode_00000*.parquet",
    "videos/chunk-000/*/episode_00000*.mp4",
    "videos/chunk-000/*/episode_00001[0-5].mp4",
    "failure at last/videos/chunk-000/*/episode_00000*.mp4",
]
print(f"Downloading HenryZ07/RFT-reference-trajectory directly from HuggingFace -> {target} ...")
snapshot_download(
    repo_id="HenryZ07/RFT-reference-trajectory",
    repo_type="dataset",
    local_dir=target,
    allow_patterns=allow_patterns,
    max_workers=8,
)
print("HuggingFace direct dataset download complete!")
PY

python "$REPO_DIR/scripts/run_comprehensive_benchmarks.py" \
    --dexycb_root "$DEXYCB_TARGET" \
    --rft_root "$SCRATCH/capstone_datasets/RFT-reference-trajectory" \
    --urdf_path "$REPO_DIR/assets/vega_u_wuji_hand2.urdf" \
    --full_pace \
    --epochs_wm 100 \
    --epochs_critic 100

python "$REPO_DIR/scripts/render_experiment_videos.py"

echo "[COMPLETE] PACE-ICE GPU Training, Benchmarking & Video Rendering Finished!"
