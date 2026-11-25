#!/bin/bash
#SBATCH --job-name=dex_vla_wam
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=24G
#SBATCH --output=logs/pace_gpu_%j.out
#SBATCH --error=logs/pace_gpu_%j.err
# ==============================================================================
# Georgia Tech PACE-ICE Fast-Allocation GPU Suite for `dex-vla-critic-wam`
# - Uses `#SBATCH --gres=gpu:1` (matches VLM LaneB point.sbatch/profile_vram.sbatch)
#   so SLURM allocates ANY immediately available GPU (V100, RTX_6000, A40, A100, L40S, H100)
# - Uses `--time=00:45:00` and `--mem=24G` so SLURM backfill schedules it immediately
# - Reuses existing `~/scratch/vlm_placement_env` (or `~/scratch/lanea_env`) if present
# ==============================================================================
set -euo pipefail

SCRATCH="${SCRATCH:-$HOME/scratch}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export VLM_CACHE_ROOT="$HF_HOME"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export UV_CACHE_DIR="${SCRATCH}/uv_cache"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/models" "$REPO_DIR/results" "$REPO_DIR/demo_videos" "$SCRATCH/capstone_datasets" "$HF_HOME" "$PIP_CACHE_DIR"

module load anaconda3/2023.03 2>/dev/null || module load python/3.12.5 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda/12.6.1 2>/dev/null || true

# Reuse existing PACE scratch environment from VLM project if available for instant startup
if [ -d "$SCRATCH/vlm_placement_env" ]; then
    echo "Reusing existing Conda environment: $SCRATCH/vlm_placement_env"
    source activate "$SCRATCH/vlm_placement_env" 2>/dev/null || conda activate "$SCRATCH/vlm_placement_env" 2>/dev/null || export PATH="$SCRATCH/vlm_placement_env/bin:$PATH"
    pip install --quiet opencv-python-headless pyarrow pyyaml huggingface_hub
elif [ -d "$SCRATCH/lanea_env" ]; then
    echo "Reusing existing venv: $SCRATCH/lanea_env"
    # shellcheck disable=SC1091
    source "$SCRATCH/lanea_env/bin/activate"
    pip install --quiet opencv-python-headless pyarrow pyyaml huggingface_hub
else
    ENV_DIR="$SCRATCH/capstone_env"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || pip install --user uv
    if [ ! -d "$ENV_DIR" ]; then
        uv venv "$ENV_DIR"
        # shellcheck disable=SC1091
        source "$ENV_DIR/bin/activate"
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        uv pip install -r "$REPO_DIR/requirements.txt"
    else
        # shellcheck disable=SC1091
        source "$ENV_DIR/bin/activate"
    fi
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
