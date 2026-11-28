"""
Generates PowerPoint-Ready Annotated Demo & Mathematical Proof Videos for Run #1 (`demo_videos/run_1_demos/`)
=============================================================================================================
Every video in `demo_videos/run_1_demos/` includes:
1. Live PACE-ICE Run #1 CUDA GPU Benchmark Metrics (487 DexYCB frames, 1,422 VegaU windows, 1,548 transitions).
2. A dedicated bottom **"WHAT IS HAPPENING RIGHT NOW (PLAIN-ENGLISH NARRATION)"** banner on every frame
   that explains the exact visual & mathematical event step-by-step for PowerPoint slide presentations.

Videos produced in `demo_videos/run_1_demos/`:
1. `run1_demo_1_multiview_2d_to_3d_mesh_dpi_net.mp4`
2. `run1_demo_2_agentic_1hz_critic_checkpoint_recovery_failure.mp4`
3. `run1_demo_3_agentic_1hz_critic_full_sop_success.mp4`
4. `run1_proof_1_dlt_svd_and_visual_hull_leave_one_cam_out.mp4`
5. `run1_proof_2_se3_urdf_fk_and_moving_wrist_cam_chain.mp4`
6. `run1_proof_3_dpinet_graph_message_passing_and_cotraining.mp4`
7. `run1_proof_4_perceiver_1hz_cache_and_sage_sink_grounding.mp4`
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

import render_math_proof_videos as rmp


def _annotate_existing_video_with_run1_narration(
    src_mp4: str,
    dst_mp4: str,
    run1_header_badge: str,
    narration_schedule: List[Tuple[float, float, str, str]],
) -> None:
    """
    Takes a 1280x720 MP4 video, compresses the main viewport to 1280x636, and adds an 84-pixel
    high-contrast PowerPoint narration footer at the bottom (y=636..720) with:
    - Line 1 (Cyan/Gold): Step-by-step plain-English explanation of what is happening right now
    - Line 2 (Green/White): Exact Run #1 PACE-ICE CUDA mathematical verification proof
    """
    cap = cv2.VideoCapture(src_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    writer = cv2.VideoWriter(dst_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))

    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        prog = float(f_idx) / float(max(total_frames - 1, 1))

        # Resize original 1280x720 frame into top 1280x634 area
        top_view = cv2.resize(frame, (1280, 634), interpolation=cv2.INTER_AREA)

        # Draw Run #1 PACE-ICE CUDA badge in top-right corner
        cv2.rectangle(top_view, (875, 6), (1272, 34), (25, 65, 40), -1)
        cv2.rectangle(top_view, (875, 6), (1272, 34), (90, 255, 140), 1)
        cv2.putText(top_view, run1_header_badge, (885, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 255, 180), 1, cv2.LINE_AA)

        # Find active narration for current progress
        line1 = narration_schedule[-1][2]
        line2 = narration_schedule[-1][3]
        for p_start, p_end, l1, l2 in narration_schedule:
            if p_start <= prog <= p_end:
                line1, line2 = l1, l2
                break

        footer = np.full((86, 1280, 3), (14, 17, 24), dtype=np.uint8)
        cv2.line(footer, (0, 0), (1280, 0), (90, 210, 255), 2)
        cv2.rectangle(footer, (12, 10), (195, 36), (40, 80, 140), -1)
        cv2.putText(footer, "WHAT IS HAPPENING:", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 240, 130), 1, cv2.LINE_AA)
        cv2.putText(footer, line1, (208, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 248, 255), 1, cv2.LINE_AA)

        cv2.rectangle(footer, (12, 46), (195, 72), (25, 75, 50), -1)
        cv2.putText(footer, "PACE RUN #1 PROOF:", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 255, 165), 1, cv2.LINE_AA)
        cv2.putText(footer, line2, (208, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 255, 195), 1, cv2.LINE_AA)

        out_frame = np.vstack([top_view, footer])
        writer.write(out_frame)
        f_idx += 1

    cap.release()
    writer.release()


def main() -> None:
    # First re-render the 4 math proof videos so Proof 3 has the exact Run #1 PACE-ICE CUDA table
    rmp.main()

    demo_dir = os.path.join(_REPO_ROOT, "demo_videos")
    proof_dir = os.path.join(demo_dir, "math_proofs")
    run1_dir = os.path.join(demo_dir, "run_1_demos")
    os.makedirs(run1_dir, exist_ok=True)

    # 1. Annotate Demo 1 (Multi-View 2D RGB -> 3D Dense/Sparse Mesh -> DPI-Net Particle Rollout)
    _annotate_existing_video_with_run1_narration(
        os.path.join(demo_dir, "demo_1_multiview_2d_to_3d_mesh_dpi_net.mp4"),
        os.path.join(run1_dir, "run1_demo_1_multiview_2d_to_3d_mesh_dpi_net.mp4"),
        "PACE-ICE RUN #1 (CUDA): CD = 5.61x10^-3 (-52.9%)",
        [
            (
                0.00, 0.33,
                "Stage 1 (Left): 4 synchronized 2D RGB cameras segment the blue pipette & yellow tube without depth sensors.",
                "Exact 59-joint SE(3) URDF Forward Kinematics computes 3D Wuji finger poses while MultiViewObjectPoseNet triangulates object pose (0.9365 axis cos-sim).",
            ),
            (
                0.33, 0.66,
                "Stage 2 (Top-Right): Toggle switches between Dense (1,840 tris) and Sparse (360 tris) 3D reconstructed surface meshes (.OBJ).",
                "Farthest Point Sampling (FPS) extracts N_e=60 hand particles (Gold) and N_o=100 object particles (Blue/Cyan) in a shared 3D frame.",
            ),
            (
                0.66, 1.01,
                "Stage 3 (Bottom-Right): DPI-Net builds a 3D radius contact graph (green edges) and predicts T+30 future object trajectory (magenta arrows).",
                "Run #1 PACE-ICE CUDA (1,548 transitions): 1:1 Human+Robot Co-Trained DPI-Net reduces Chamfer Distance by 52.9% vs PointTransformer (11.92 -> 5.61x10^-3).",
            ),
        ],
    )

    # 2. Annotate Demo 2 (1Hz Agentic VLM Critic + DPI-Net Failure Interception & Checkpoint Rollback)
    _annotate_existing_video_with_run1_narration(
        os.path.join(demo_dir, "demo_2_agentic_1hz_critic_checkpoint_recovery_failure.mp4"),
        os.path.join(run1_dir, "run1_demo_2_agentic_1hz_critic_checkpoint_recovery_failure.mp4"),
        "PACE-ICE RUN #1 (CUDA): 100% Fault F1 @ 0.91ms",
        [
            (
                0.00, 0.38,
                "Act 1 (Nominal Execution on Failure Episode 000003): Robot executes P5->P10 while 1Hz Critic commits verified checkpoints C_k.",
                "Run #1 PACE-ICE CUDA (1,422 windows): Fast1HzNeuralCritic achieves 91.57% 10-phase accuracy & 99.72% transition recall at 0.91 ms cached latency.",
            ),
            (
                0.38, 0.68,
                "Act 2 (Fault Detection during P11 Hang-Back): Pipette slips off the stand bar! DPI-Net contact edges collapse (162 -> 49 edges).",
                "Slip Risk spikes to 0.85 and SAGE-grounded 1Hz Critic (100.0% Fault F1 on PACE) immediately triggers RESUME_FROM_CHECKPOINT(CKPT_P10_RELEASE)!",
            ),
            (
                0.68, 1.01,
                "Act 3 (Offline Trace-Driven Rollback Visualization): State rewinds to verified checkpoint CKPT_P10_RELEASE instead of crashing at P11.",
                "Prevents the 70.6s (64% episode) wasted open-loop execution and QP/IK solver crash ('matrix P is not positive definite') found in our 191-log audit.",
            ),
        ],
    )

    # 3. Annotate Demo 3 (Full 10-Phase SOP Success Verification at 1Hz)
    _annotate_existing_video_with_run1_narration(
        os.path.join(demo_dir, "demo_3_agentic_1hz_critic_full_sop_success.mp4"),
        os.path.join(run1_dir, "run1_demo_3_agentic_1hz_critic_full_sop_success.mp4"),
        "PACE-ICE RUN #1 (CUDA): 91.57% 10-Phase Acc",
        [
            (
                0.00, 0.35,
                "Phases P1 -> P6 (Right-Hand Pipette Grasp & Unrack): Critic verifies grasp closure and commits CKPT_P5_R_GRIP and CKPT_P6_UNRACK.",
                "Perceiver Sliding-Window Cache reuses static camera tokens, running in 0.91 ms on PACE CUDA (14.64x faster than uncached 13.28 ms).",
            ),
            (
                0.35, 0.70,
                "Phases P7 -> P9 (In-Hand Drop-Catch, Left-Hand Tube Fetch, Bimanual Align & Press): Critic monitors tip-to-tube alignment.",
                "SeeTraceAct 2D future wrist-trace prediction achieves 0.0093 L1 error across 356 held-out PACE test windows while SAGE maintains 100% visual grounding.",
            ),
            (
                0.70, 1.01,
                "Phases P10 -> P11 (Release, Handover & Stand Return): All 10 clinical SOP checkpoints verified -> FlowConductor signals SOP COMPLETE.",
                "Outperforms prior 5-phase tube critic (80.0% acc / 90.0% recall) by scaling to all 10 bimanual phases at 91.57% accuracy and 99.72% transition recall.",
            ),
        ],
    )

    # 4. Annotate Math Proof 1 (DLT SVD Triangulation & Held-Out Cam 5 Visual Hull)
    _annotate_existing_video_with_run1_narration(
        os.path.join(proof_dir, "proof_1_dlt_svd_and_visual_hull_leave_one_cam_out.mp4"),
        os.path.join(run1_dir, "run1_proof_1_dlt_svd_and_visual_hull_leave_one_cam_out.mp4"),
        "PACE RUN #1 (487 Frames): MPJPE = 2.17 mm",
        [
            (
                0.00, 0.50,
                "Left & Center: 4 calibrated RGB cameras shoot 2D keypoint rays into 3D space. We stack A_c = [u P_3 - P_1; v P_3 - P_2] and solve SVD(A).",
                "PACE-ICE Run #1 (487 DexYCB frames): 4th singular value sigma_4 -> 0 proves ray intersection, achieving 2.17 mm 3D hand error without depth maps!",
            ),
            (
                0.50, 1.01,
                "Right Panel (Held-Out 5th Camera Proof): 3D Visual Hull carved from Cams 1-4 is projected onto Held-Out Cam 5 (never used in 3D carving).",
                "PACE-ICE Run #1: Reconstructed 3D object achieves CD = 0.6407x10^-3 (F-Score@15mm = 59.67%) and 45.02% 2D silhouette overlap IoU on Held-Out Cam 5.",
            ),
        ],
    )

    # 5. Annotate Math Proof 2 (Exact 59-Joint SE(3) URDF FK & Dynamic Moving Wrist Cameras)
    _annotate_existing_video_with_run1_narration(
        os.path.join(proof_dir, "proof_2_se3_urdf_fk_and_moving_wrist_cam_chain.mp4"),
        os.path.join(run1_dir, "run1_proof_2_se3_urdf_fk_and_moving_wrist_cam_chain.mp4"),
        "PACE RUN #1 (1,422 Samples): Axis CosSim = 0.9365",
        [
            (
                0.00, 0.55,
                "Left & Center: Wrist and head cameras move continuously with the robot arms. How do we keep them calibrated without recalibrating online?",
                "We multiply the live 59-joint URDF SE(3) chain T_{world<-wrist}(q_t) (updated at 1,000 Hz from motor encoders) by constant mount T_{wrist<-cam}!",
            ),
            (
                0.55, 1.01,
                "Right Panel (Relative Slip Tracking): Pipette 3D pose x_pip(t) is triangulated independently from 2D masks and compared to wrist x_wrist(q_t).",
                "When d_rel(t) = ||x_pip(t) - x_wrist(q_t)||_2 jumps above 135-150 mm (late episode slip), the physical slip is caught immediately!",
            ),
        ],
    )

    # 6. Annotate Math Proof 3 (Equivariant DPI-Net Radius Graph & 5-Model Co-Training)
    _annotate_existing_video_with_run1_narration(
        os.path.join(proof_dir, "proof_3_dpinet_graph_message_passing_and_cotraining.mp4"),
        os.path.join(run1_dir, "run1_proof_3_dpinet_graph_message_passing_and_cotraining.mp4"),
        "PACE RUN #1 (1,548 Samples): -52.9% CD via Co-Train",
        [
            (
                0.00, 0.50,
                "Left: Human MANO hands and 20-DoF Wuji robot hands are both converted into 60 hand dots (Gold) + 100 object dots (Blue) with r=7.5cm contact edges.",
                "Because local particle contact physics is identical across embodiments, DPI-Net can co-train 1:1 on DexYCB human data + VegaU robot data!",
            ),
            (
                0.50, 1.01,
                "Right: All 3 robot world models evaluated on the EXACT same 282 held-out vega_test transitions from our 100-epoch PACE-ICE CUDA run.",
                "PointTransformer: 11.92x10^-3 -> Robot-Only DPI-Net: 10.09x10^-3 (-15.4%) -> 1:1 Co-Trained DPI-Net: 5.61x10^-3 (-44.4% vs Robot-Only, -52.9% vs PT)!",
            ),
        ],
    )

    # 7. Annotate Math Proof 4 (1Hz Perceiver Cache & SAGE Sink-Token Grounding)
    _annotate_existing_video_with_run1_narration(
        os.path.join(proof_dir, "proof_4_perceiver_1hz_cache_and_sage_sink_grounding.mp4"),
        os.path.join(run1_dir, "run1_proof_4_perceiver_1hz_cache_and_sage_sink_grounding.mp4"),
        "PACE RUN #1 (CUDA): 0.91ms (14.64x Cache Speedup)",
        [
            (
                0.00, 0.50,
                "Top-Left: 8-slot Perceiver Ring Buffer caches 32 compressed visual tokens per step and only re-encodes cameras with active motion.",
                "PACE-ICE Run #1 CUDA Latency: Drops from 13.28 ms (uncached) to 0.91 ms (cached = 1,102.5 Hz, 14.64x speedup) — well under the 1,000 ms (1Hz) target!",
            ),
            (
                0.50, 1.01,
                "Bottom-Left & Right: Vanilla VLMs dump 68% attention onto [BOS] sink tokens and hallucinate success. SAGE subtracts beta=2.5 from the sink logit.",
                "PACE-ICE Run #1: SAGE achieves 100% visual-to-sink grounding on CUDA, driving 100% Fault Detection F1 and triggering RESUME_FROM_CHECKPOINT.",
            ),
        ],
    )

    print("[SUCCESS] Rendered all 7 annotated Run #1 presentation videos in:", run1_dir)
    for fn in sorted(os.listdir(run1_dir)):
        p = os.path.join(run1_dir, fn)
        print(f"  - {fn} ({os.path.getsize(p) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
