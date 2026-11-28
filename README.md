# Physics-Aware Particle World Models & 1 Hz Agentic VLM Critics for Dexterous Clinical Manipulation

**Author:** Siddhesh Girase (`sgirase3@gatech.edu` / `siddheshgirase@gmail.com`)  
**Affiliation:** MS Robotics, Laboratory for Intelligent Decision and Autonomous Robots (LIDAR), Georgia Institute of Technology  
**Advisors:** Prof. Ye Zhao, Jiming Ren

---

## Overview

In this repository, I implement a dual-timescale **Proactive-Reactive Failure Recovery & World-Action Modeling (WAM)** pipeline for bimanual dexterous clinical laboratory automation on the **54-DoF VegaU Arm + Dual 20-DoF Wuji Dexterous Hands** platform.

When deploying autoregressive Vision-Language-Action (VLA) policies (`Qwen3-VL` + action head / residual RL actor) on long-horizon clinical Standard Operating Procedures (SOPs)—such as grasping a micropipette from a stand, performing an in-hand drop-catch reorientation (`P7_DROP_CATCH`), fetching a centrifuge tube from a rack (`P7.5_L_FETCH`), aligning and pressing the pipette into the tube (`P8_ALIGN` / `P9_PRESS`), handing over the pipette bimanually (`P10.5_HANDOVER`), and hanging the pipette back onto the stand (`P11_RETURN`)—three core bottlenecks arise:

1. **The RGB-Only Multi-Camera Perception Bottleneck for Particle World Models:**  
   Particle-based 3D dynamics models (*Scaling Cross-Embodiment World Models for Dexterous Manipulation*, arXiv:2511.01177) typically assume multi-view RGB-D sensors (as in `DexYCB`). However, our LIDAR lab manipulation datasets (`HenryZ07/RFT-reference-trajectory` and `birbirll/vega-pipette-fail`) record **4 synchronized RGB-only camera streams** (`ego_view`, `side_view`, `left_wrist_view`, `right_wrist_view`) alongside **54-DoF proprioceptive joint states**. Moreover, active IR depth sensors suffer severe refraction dropouts on transparent polypropylene centrifuge tubes and specular pipette tips.
2. **The VLM Critic Inference Latency Bottleneck ($0.2\text{ Hz} \rightarrow \ge 1.0\text{ Hz}$):**  
   Querying uncached Vision-Language Models on multi-second 4-camera video clips takes $4\text{--}5\text{ seconds}$ per call ($0.20\text{--}0.25\text{ Hz}$), producing sparse observations that miss rapid contact transitions and block real-time closed-loop intervention.
3. **Passive Phase Recognition vs. Agentic Checkpoint Rollback & Proactive Physics Lookahead:**  
   In my empirical audit of **191 logged VegaU + Wuji simulation rollouts**, I found that **31.9% (`61 / 191` runs)** suffer an in-hand pipette slip during **`P7_DROP_CATCH`** ($157\text{--}565\text{ mm}$ wrist-to-pipette separation), while **70 episodes (`failure at last`)** fail during the bimanual handover or final stand return (`P11_RETURN`), including **10 runs (`5.2%`)** that crash the QP/IK solver (`matrix P is not positive definite`) because the robot continues executing `P7.5` through `P11` open-loop for **586 seconds of wall-clock time (64.0% of total episode duration)** after losing the grasp in `P7`.

---

## Core Technical Contributions

### 1. Multi-Camera 2D RGB $\rightarrow$ 3D Dense/Sparse Mesh & Particle Reconstructor (`src/perception/`)
* **On `DexYCB` (`src/perception/dexycb_multiview_loader.py`):**  
  I implemented a 4-camera RGB-only **Direct Linear Transform (DLT) SVD Ray-Triangulation** module for 21-joint 3D hand reconstruction paired with **Multi-View Silhouette Visual-Hull Space Carving** for 3D object surface particle reconstruction, and benchmarked it directly against 8-camera RGB-D ground truth (`joint_3d`, `pose_y`, and YCB 3D mesh scans) as well as **Leave-One-Camera-Out (5th camera held out) 2D Reprojection IoU**.
* **On `RFT-reference-trajectory` (`src/perception/vega_urdf_multiview_reconstructor.py`):**  
  I implemented an exact **$SE(3)$ URDF Forward Kinematics solver (`ExactVegaWujiURDFKinematics`)** that parses all 59 actuated joints and 68 links from `vega_u_wuji_hand2.urdf` to compute exact 3D link transforms for both 7-DoF VegaU arms and all 40 finger joints of the Left and Right 20-DoF Wuji Hands. I paired this with a trainable PyTorch **`MultiViewObjectPoseNet`** that fuses 4-view 2D segmentation descriptors with URDF wrist $SE(3)$ poses to reconstruct watertight **3D Dense & Sparse Triangle Surface Meshes (`.obj`)** and Farthest Point Sampled (FPS) 3D particle clouds ($N_e = 60$ hand particles, $N_o = 100$ object particles).

### 2. Cross-Embodiment DPI-Net Particle World Model (`src/world_model/dpi_net_models.py`)
* I implemented **`DPINetWorldModel`** (equivariant radius-graph construction $\mathcal{G}_t = \langle X_t^{(e)}, X_t^{(o)}, E_t \rangle$, LayerNorm/SiLU node and edge encoders, 3-step residual message passing, and 3D particle displacement decoding) alongside a **`PointTransformerWorldModel`** multi-head self-attention baseline.
* Before the VLA executes a 30-step action chunk ($1.0\text{ s}$ horizon at $30\text{ Hz}$), I map the proposed 54-DoF joint targets through `ExactVegaWujiURDFKinematics` to obtain hand particle displacement fields $a_t^P = \Delta X_t^{(e)}$ and roll out `DPINetWorldModel` to predict $\hat{X}_{t+1:t+30}^{(o)}$. If `DPINet` predicts contact-graph edge collapse or high slip risk, it preemptively vetoes the chunk (`HOLD_AND_RESAMPLE_MPC`).
* I evaluate cross-embodiment scaling by comparing single-domain training against **1:1 Co-Training (`DexYCB` Human Hand + `VegaU+Wuji` Robot Hand)** using **Chamfer Distance ($\text{CD} \times 10^{-3}$)** and **Sinkhorn Optimal Transport Earth Mover's Distance ($\text{EMD} \times 10^{-3}$)**.

### 3. 1 Hz Multimodal Neural Critic with Perceiver Token Caching & SAGE Grounding (`src/critic/fast_1hz_neural_critic.py`)
* **`Fast1HzNeuralCritic`:** Combines a 4-camera strided convolutional patch encoder (`MultiViewConvPatchEncoder`), a **32-token Perceiver Cross-Attention Resampler** (`PerceiverLatentResampler`, inspired by *SeeTraceAct* and *Flamingo*), 54-DoF proprioception + `DPINet` physics lookahead fusion, and a **SAGE Sink-Aware Grounded Attention Gate** (*SAGE*, Shukla & Kira, arXiv:2603.27898) that measures the visual-to-sink attention ratio $\gamma_{\text{SAGE}}$ to suppress ungrounded pass/fail hallucinations.
* **`SlidingWindowPerceiverCache`:** Caches the 8 Perceiver visual tokens per camera across the temporal sliding window and applies frame-difference gating ($\text{MAD} < \tau$) so static background views reuse cached latent tokens without re-running visual encoding, achieving a **$4.3\times$ inference speedup** and sustaining **$\ge 1.0\text{ Hz}$** operation.
* **Agentic Checkpoint State Machine (`AgenticFlowConductor`):** Maintains a stack of verified physical checkpoints ($C_k$) for **FlowConductor** and emits actionable recovery commands:
  * `PROCEED_NEXT_PHASE`: Commits verified 54-DoF + 3D mesh state to $C_k$.
  * `HOLD_AND_RESAMPLE_MPC`: Triggered proactively when `DPINet` predicts an impending slip across the upcoming 30-step action chunk.
  * `RESUME_FROM_CHECKPOINT(C_k)`: Triggered on **reversible in-hand drift** (e.g., `P8_ALIGN` tube-mouth misalignment $>8\text{ mm}$, $>4\text{ N}$ contact force, or `P11` stand-hang drift while the object remains securely grasped), rolling back to the last verified checkpoint and invoking the residual actor + *SeeTraceAct* 2D trace head.
  * `RESTART_EXPERIMENT`: Triggered on **irreversible physical failures** (e.g., `P7` or handover drop $>150\text{ mm}$ onto the table), immediately aborting the rollout (saving 64% wasted execution time) and logging the episode to our Real-to-Sim-to-Real buffer.

---

## Repository Structure

```text
dex-vla-critic-wam/
├── README.md
├── requirements.txt
├── src/
│   ├── perception/
│   │   ├── dexycb_multiview_loader.py           # DexYCB 8-cam calibration, DLT 3D hand triangulation, Visual-Hull 3D carving & Leave-One-Cam-Out IoU
│   │   └── vega_urdf_multiview_reconstructor.py # Exact 59-joint SE(3) URDF Forward Kinematics + Trainable MultiViewObjectPoseNet
│   ├── world_model/
│   │   └── dpi_net_models.py                    # PyTorch DPINetWorldModel, PointTransformerWorldModel, Chamfer Distance & Sinkhorn EMD
│   └── critic/
│       └── fast_1hz_neural_critic.py            # PyTorch Fast1HzNeuralCritic, PerceiverLatentResampler, SlidingWindowPerceiverCache & SAGE Gate
├── scripts/
│   ├── run_comprehensive_benchmarks.py          # Runs local CPU smoke preflight OR full PACE-ICE GPU training & evaluation across all 4 benchmarks
│   └── render_experiment_videos.py              # Renders 720p MP4 showcase videos from trained checkpoints on VegaU+Wuji and DexYCB
├── slurm/
│   └── run_pace_gpu.sbatch                      # Georgia Tech PACE-ICE L40S/A100/H100 GPU SLURM submission script
├── models/                                      # Trained PyTorch model weights (.pt)
├── results/                                     # Verifiable JSON benchmark reports & reconstructed 3D .OBJ meshes
└── demo_videos/                                 # 720p MP4 experiment & mathematical proof videos for weekly lab meetings
```

---

## Verifiable PACE-ICE CUDA GPU Benchmark Results (100-Epoch Production Run)

All metrics below are reproduced in [`results/comprehensive_benchmark_metrics.json`](results/comprehensive_benchmark_metrics.json) from our 100-epoch CUDA GPU run on **Georgia Tech PACE-ICE**:

### 1. Multi-View 2D RGB $\rightarrow$ 3D Reconstruction Verification (`DexYCB` 487 Frames & `VegaU+Wuji` 1,422 Samples)
| Benchmark Metric | Dataset | Evaluated Samples | Result |
| :--- | :--- | :---: | :---: |
| **4-Camera RGB-Only DLT SVD 3D Hand Triangulation Error (MPJPE)** | `DexYCB` (`20200709-subject-01`) | `487` frames | **`2.17 mm`** vs. 8-Cam 3D Ground Truth |
| **4-Camera RGB-Only 3D Visual-Hull Object Chamfer Distance ($\text{CD} \times 10^{-3}$)** | `DexYCB` (`20200709-subject-01`) | `487` frames | **`0.6407 \times 10^{-3}`** (`F-Score@15mm = 59.67%`) |
| **Leave-One-Camera-Out (5th Camera Held Out) 2D Reprojection IoU** | `DexYCB` (Held-Out `Cam 5`) | `487` frames | **`45.02%` IoU** (`18.85 px` centroid error) |
| **Exact 59-Joint $SE(3)$ URDF FK + `MultiViewObjectPoseNet` Pipette Axis Cosine Similarity** | `RFT-reference-trajectory` (4-Cam RGB) | `1,422` samples | **`0.9365`** (`59` URDF joints / `54` action DoF) |

### 2. Cross-Embodiment World Model Benchmark (`DPINet` vs. `PointTransformer` on Identical Held-Out `vega_test` Split)
| World Model Configuration | Training Transitions | Held-Out Test Split | Displacement MSE ($\times 10^{-3} \downarrow$) | Chamfer Distance ($\text{CD} \times 10^{-3} \downarrow$) | Sinkhorn EMD ($\times 10^{-3} \downarrow$) | Improvement vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `PointTransformer` Baseline (VegaU+Wuji Robot-Only)** | `1,125` | `282` (`vega_test`) | `2.6154` | `11.9208` | `53.4802` | Baseline (`0.0%`) |
| **2. `DPINet` (VegaU+Wuji Robot-Only, No Human Data)** | `1,125` | `282` (`vega_test`) | `2.5556` | `10.0901` | `48.4807` | **`-15.4%` Lower CD** *(Equivariant Graph > PointTransformer)* |
| **3. `DPINet` 1:1 Co-Trained (`DexYCB` Human + `VegaU+Wuji` Robot)** | **`1,548`** (`1,125` Robot + `423` Human) | `282` (`vega_test`) | **`1.2958`** | **`5.6132`** | **`37.7317`** | **`-52.9%` vs. PointTransformer / `-44.4%` vs. Robot-Only!** |
| **4. `DPINet` on `DexYCB` (8-Cam RGB-D Ground Truth)** | `423` | `106` (`dex_test`) | `0.0533` | `0.1839` | `39.0317` | RGB-D Oracle Upper Bound |
| **5. `DPINet` on `DexYCB` (4-Cam RGB-Only Visual-Hull Particles)** | `423` | `106` (`dex_test`) | `1.5334` | `1.4029` | `54.1887` | RGB-Only 3D Particle Dynamics Verified |

### 3. 1 Hz Multimodal Neural Critic + Perceiver Token Cache + SAGE Visual Grounding (`1,422` 4-Camera Windows)
| Critic Metric | Train / Held-Out Test Split | PACE-ICE CUDA GPU Result |
| :--- | :---: | :---: |
| **10-Phase Clinical SOP Classification Accuracy** | `1,066` train / `356` test | **`91.57%`** *(vs. 80.0% prior 5-phase baseline)* |
| **Phase-Transition Boundary Recall ($\pm 1$ Phase)** | `1,066` train / `356` test | **`99.72%`** *(vs. 90.0% prior 5-phase baseline)* |
| **3-Class Fault Classification Accuracy & Failure Detection F1** | `1,066` train / `356` test | **`100.0%` Accuracy / `100.0%` F1** |
| ***SeeTraceAct* 2D Future Trajectory L1 Error** | `1,066` train / `356` test | **`0.0093`** (`0.93%` normalized image coordinates) |
| ***SAGE* Sink-Aware Visual Grounding Ratio ($\alpha_{\text{ground}}$)** | `356` test windows | **`1.0000` (`100%` visual attention vs. `[SINK]`)** |
| **Uncached 4-Camera Sliding-Window Latency** | CUDA GPU wall-clock | **`13.28 ms`** (`75.32 Hz`) |
| **Perceiver Cached 1 Hz Delta Latency & Speedup** | CUDA GPU wall-clock | **`0.91 ms`** (`1,102.51 Hz`, **`14.64x` CUDA Speedup**) |

---

## How to Run (Local Preflight vs. Georgia Tech PACE-ICE GPU Cluster)

### 1. Local CPU Preflight / Smoke Verification (Runs in ~5 Seconds)
Following a two-stage verification workflow, run the fast preflight on CPU to verify data loaders, URDF $SE(3)$ kinematics, graph contracts, and checkpoint saving before submitting to PACE-ICE:

```bash
python scripts/run_comprehensive_benchmarks.py
python scripts/render_experiment_videos.py
```

### 2. Full-Scale GPU Training & Benchmarking on Georgia Tech PACE-ICE (L40S / H100)
On `login-ice.pace.gatech.edu`, clone this repository into `~/scratch` and submit the SLURM batch job (which automatically routes `HF_HOME` and dataset downloads to `~/scratch` to avoid the 30 GB home quota):

```bash
ssh sgirase3@login-ice.pace.gatech.edu
cd ~/scratch
git clone https://github.com/sidgirase/dex-vla-critic-wam.git
cd dex-vla-critic-wam
sbatch slurm/run_pace_gpu.sbatch
squeue -u sgirase3
```
