"""
KITTI Visual Odometry Pipeline
Synthetic Dataset Edition
Author: Seun
"""

import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# PATHS - Edit only these two lines if you move the folder
# ================================================================
BASE_PATH  = 'dataset'
PLOT_PATH  = 'trajectory_result.png'
# ================================================================

# ── STAGE 1: CALIBRATION ────────────────────────────────────────

def load_calib(filepath):
    calib = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])
    P0 = calib['P0'].reshape(3, 4)
    P1 = calib['P1'].reshape(3, 4)
    return P0, P1

def extract_intrinsics(P0, P1):
    K  = P0[0:3, 0:3]
    fx = K[0, 0]
    tx = P1[0, 3]
    B  = -tx / fx
    return K, B

# ── STAGE 2: STEREO DEPTH ───────────────────────────────────────

def compute_depth(imgL, imgR, K, B):
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disp   = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disp[disp <= 0] = 0.1
    fx    = K[0, 0]
    depth = (fx * B) / disp
    depth[depth > 100] = 100
    return depth

# ── STAGE 3: FEATURE DETECTION AND MATCHING ─────────────────────

def detect_and_match(img1, img2):
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return [], []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2

# ── STAGE 4: POSE ESTIMATION ────────────────────────────────────

def estimate_pose(pts1, pts2, K, depth):
    if len(pts1) < 8:
        return np.eye(4)
    pts3d = []
    pts2d = []
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    for (u, v), pt2 in zip(pts1, pts2):
        ui, vi = int(u), int(v)
        if 0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]:
            Z = depth[vi, ui]
            if 0.1 < Z < 80:
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                pts3d.append([X, Y, Z])
                pts2d.append(pt2)
    if len(pts3d) < 6:
        return np.eye(4)
    pts3d = np.array(pts3d, dtype=np.float64)
    pts2d = np.array(pts2d, dtype=np.float64)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        reprojectionError=8.0,
        iterationsCount=200
    )
    if not success:
        return np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.flatten()
    return T

# ── STAGE 5: LOAD GROUND TRUTH ──────────────────────────────────

def load_poses(filepath):
    poses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            T = np.array([float(x) for x in line.split()])
            T = T.reshape(3, 4)
            pose = np.eye(4)
            pose[:3, :] = T
            poses.append(pose)
    return poses

# ── MAIN PIPELINE ───────────────────────────────────────────────

def run_pipeline(base_path):
    seq_path   = os.path.join(base_path, 'sequences', '00')
    calib_path = os.path.join(seq_path,  'calib.txt')
    poses_path = os.path.join(base_path, 'poses', '00.txt')
    img0_path  = os.path.join(seq_path,  'image_0')
    img1_path  = os.path.join(seq_path,  'image_1')

    P0, P1   = load_calib(calib_path)
    K, B     = extract_intrinsics(P0, P1)
    print(f"Calibration loaded | fx={K[0,0]:.1f} | B={B:.4f}m")

    gt_poses = load_poses(poses_path)
    print(f"Ground truth loaded | {len(gt_poses)} poses")

    images = sorted(os.listdir(img0_path))
    print(f"Found {len(images)} frames")

    est_trajectory = []
    gt_trajectory  = []
    current_pose   = np.eye(4)

    print("\nRunning pipeline...")
    for i in range(len(images) - 1):
        imgL      = cv2.imread(os.path.join(img0_path, images[i]),     0)
        imgR      = cv2.imread(os.path.join(img1_path, images[i]),     0)
        imgL_next = cv2.imread(os.path.join(img0_path, images[i + 1]), 0)

        depth      = compute_depth(imgL, imgR, K, B)
        pts1, pts2 = detect_and_match(imgL, imgL_next)
        delta_pose = estimate_pose(pts1, pts2, K, depth)

        current_pose = current_pose @ np.linalg.inv(delta_pose)

        est_trajectory.append(current_pose[:3, 3].copy())
        gt_trajectory.append(gt_poses[i][:3, 3].copy())

        if i % 10 == 0:
            print(f"  Frame {i:03d}/{len(images)-1} | "
                  f"matches={len(pts1)} | "
                  f"pos=({current_pose[0,3]:.1f}, {current_pose[2,3]:.1f})")

    return np.array(est_trajectory), np.array(gt_trajectory), K, B

# ── STAGE 6: PLOT ───────────────────────────────────────────────

def plot_trajectory(est, gt, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f1a')

    ax = axes[0]
    ax.set_facecolor('#0f0f1a')
    ax.plot(gt[:,  0], gt[:,  2], 'c-',  linewidth=2.5, label='Ground Truth')
    ax.plot(est[:, 0], est[:, 2], 'r--', linewidth=2,   label='Estimated VO')
    ax.scatter(gt[0, 0],  gt[0, 2],  c='lime',   s=100, zorder=5, label='Start')
    ax.scatter(gt[-1, 0], gt[-1, 2], c='yellow', s=100, zorder=5, label='End')
    ax.set_title("Bird's Eye Trajectory", color='white', fontsize=14)
    ax.set_xlabel('X (metres)', color='white')
    ax.set_ylabel('Z (metres)', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    ax2 = axes[1]
    ax2.set_facecolor('#0f0f1a')
    errors = np.linalg.norm(est - gt, axis=1)
    ax2.plot(errors, color='orange', linewidth=2)
    ax2.fill_between(range(len(errors)), errors, alpha=0.3, color='orange')
    ax2.set_title('Position Error Over Time', color='white', fontsize=14)
    ax2.set_xlabel('Frame', color='white')
    ax2.set_ylabel('ATE (metres)', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333355')
    ate = np.mean(errors)
    ax2.axhline(ate, color='red', linestyle='--', alpha=0.7)
    ax2.text(1, ate + 0.1, f'Mean ATE: {ate:.2f}m', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"\nPlot saved to: {save_path}")
    print(f"Mean ATE = {ate:.4f} metres")

# ── RUN ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    est, gt, K, B = run_pipeline(BASE_PATH)
    plot_trajectory(est, gt, PLOT_PATH)
    print("\nDone. Open trajectory_result.png to see your results.")
