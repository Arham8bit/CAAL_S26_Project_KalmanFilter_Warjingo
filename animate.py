"""
3D Full-Body Walking Animation — Kalman Filter Milestone 2
===========================================================
Section 8 Requirement:
  "Create 3D full-body walking animations for:
    • Measured data
    • LKF estimates  
    • EKF estimates
   Clearly convey motion continuity and estimation quality."

Generates 5 animations:
  1. animation_measured.gif    — Noisy sensor data
  2. animation_lkf.gif         — LKF filtered (optimal for Cartesian sensor)
  3. animation_ekf.gif         — EKF (SHOWS DIVERGENCE starting ~frame 650)
  4. animation_ekf_full.gif    — EKF complete divergence sequence
  5. animation_comparison.gif  — Three-panel (SHOWS DIVERGENCE)

Run: python3 animate.py
Output: animations/ folder
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# ================================================================
#  CONFIGURATION - UPDATED TO 3040 FRAMES
# ================================================================
CONFIG = {
    'max_frames_normal': 3040,   # UPDATED: Now runs full 3040 frames
    'max_frames_full': 3040,     # UPDATED: Complete sequence to 3040
    'step': 3,                   # Every 3rd frame (faster rendering)
    'fps': 20,                   # Frames per second
    'dpi': 80,                   # Slightly lower for faster encoding
    'rotate_camera': True,       # Slowly rotate view
    'rotation_speed': 0.10,      # Degrees per frame
}

# ================================================================
#  PATHS
# ================================================================
LKF_CSV = "lkf_results.csv"
EKF_CSV = "ekf_results.csv"
OUT_DIR = "animations/"
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================
#  CONSTANTS
# ================================================================
DT = 0.01
NUM_JOINTS = 23

JOINT_NAMES = [
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
]

# Skeleton connectivity
SKELETON = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),    # spine + head
    (5,7),(7,8),(8,9),(9,10),                # R arm
    (5,11),(11,12),(12,13),(13,14),          # L arm
    (0,15),(15,16),(16,17),(17,18),          # R leg
    (0,19),(19,20),(20,21),(21,22),          # L leg
]

# ================================================================
#  ANATOMICAL COLORING
# ================================================================
def bone_color(a, b):
    s = min(a, b)
    if s in (0,1,2,3,4,5): return '#F5DEB3'   # spine: wheat
    if s == 6: return '#FFD700'                # head: gold
    if s in (7,8,9): return '#87CEEB'          # R arm: sky blue
    if s in (11,12,13): return '#90EE90'       # L arm: light green
    if s in (15,16,17): return '#FFA07A'       # R leg: salmon
    if s in (19,20,21): return '#DDA0DD'       # L leg: plum
    return '#FFFFFF'

BONE_COLORS = [bone_color(a, b) for a, b in SKELETON]

JCOLORS = (
    ['#FFD700'] + ['#F5DEB3']*4 + ['#FFD700']*2 +
    ['#87CEEB']*4 + ['#90EE90']*4 +
    ['#FFA07A']*4 + ['#DDA0DD']*4
)

FILTER_COLORS = {
    'measured': '#E74C3C',  # Red
    'lkf': '#3498DB',       # Blue
    'ekf': '#27AE60',       # Green
}

# ================================================================
#  DATA LOADING
# ================================================================
def load_positions(csv_path, position_prefix):
    """Load position data from CSV."""
    print(f"  Loading {position_prefix} positions from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    positions = np.zeros((len(df), NUM_JOINTS, 3))
    
    for j, joint_name in enumerate(JOINT_NAMES):
        positions[:, j, 0] = df[f'{position_prefix}_{joint_name}_x'].values
        positions[:, j, 1] = df[f'{position_prefix}_{joint_name}_y'].values
        positions[:, j, 2] = df[f'{position_prefix}_{joint_name}_z'].values
    
    print(f"    Loaded {len(positions)} frames")
    return positions


def load_all_data(max_frames):
    """Load measured, LKF, and EKF position data."""
    print("\n" + "="*60)
    print("Loading animation data...")
    print("="*60)
    
    noisy = load_positions(LKF_CSV, 'noisy')
    lkf = load_positions(LKF_CSV, 'filt')
    ekf = load_positions(EKF_CSV, 'ekf')
    
    N = min(len(noisy), len(lkf), len(ekf), max_frames)
    print(f"\n  Using {N} frames ({N*DT:.1f} seconds)")
    
    return noisy[:N], lkf[:N], ekf[:N]


# ================================================================
#  AXIS SETUP
# ================================================================
def calculate_limits(data_list, padding=0.3):
    """Calculate axis limits with padding."""
    all_data = np.concatenate(data_list, axis=0)
    
    limits = []
    for axis in range(3):
        min_val = all_data[:, :, axis].min()
        max_val = all_data[:, :, axis].max()
        range_val = max_val - min_val
        limits.append((min_val - padding*range_val, max_val + padding*range_val))
    
    return limits


def style_axis(ax, title, xlim, ylim, zlim):
    """Configure 3D axis appearance."""
    ax.set_facecolor('#0B0B0B')
    
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#2A2A2A')
    
    ax.grid(True, color='#1A1A1A', linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X (m)', color='#666', fontsize=8)
    ax.set_ylabel('Y (m)', color='#666', fontsize=8)
    ax.set_zlabel('Z (m)', color='#666', fontsize=8)
    ax.tick_params(colors='#444', labelsize=7)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.view_init(elev=18, azim=30)


# ================================================================
#  SINGLE-FILTER ANIMATION
# ================================================================
def create_single_animation(positions, title, output_path, use_anatomical=False):
    """Create animation for single filter."""
    N = len(positions)
    idx = np.arange(0, N, CONFIG['step'])
    n_frames = len(idx)
    duration = n_frames / CONFIG['fps']
    
    print(f"\n{'='*60}")
    print(f"Creating: {output_path}")
    print(f"{'='*60}")
    print(f"  Source frames : {N}")
    print(f"  Step          : {CONFIG['step']}")
    print(f"  Render frames : {n_frames}")
    print(f"  Duration      : {duration:.1f}s")
    
    limits = calculate_limits([positions])
    xlim, ylim, zlim = limits
    
    fig = plt.figure(figsize=(10, 9))
    fig.patch.set_facecolor('#090909')
    
    ax = fig.add_subplot(111, projection='3d')
    style_axis(ax, title, xlim, ylim, zlim)
    
    fig.text(0.5, 0.96, f'{title}  —  Milestone 2 Section 8',
             ha='center', fontsize=13, fontweight='bold', color='white')
    
    txt_time = fig.text(0.5, 0.92, '', ha='center', fontsize=9, color='#AAA')
    
    if use_anatomical:
        patches = [
            mpatches.Patch(color='#F5DEB3', label='Spine'),
            mpatches.Patch(color='#FFD700', label='Head'),
            mpatches.Patch(color='#87CEEB', label='R Arm'),
            mpatches.Patch(color='#90EE90', label='L Arm'),
            mpatches.Patch(color='#FFA07A', label='R Leg'),
            mpatches.Patch(color='#DDA0DD', label='L Leg'),
        ]
        fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=7.5,
                   facecolor='#1A1A1A', edgecolor='#333', labelcolor='white',
                   framealpha=0.5, bbox_to_anchor=(0.5, 0.02))
    
    pos0 = positions[0]
    bone_lines = []
    
    if use_anatomical:
        for (a, b), col in zip(SKELETON, BONE_COLORS):
            ln, = ax.plot([pos0[a,0], pos0[b,0]], [pos0[a,1], pos0[b,1]],
                         [pos0[a,2], pos0[b,2]], color=col, lw=2.0, alpha=0.92)
            bone_lines.append(ln)
        joint_colors = JCOLORS
    else:
        filter_name = title.split()[0].lower()
        color = FILTER_COLORS.get(filter_name, '#FFFFFF')
        for (a, b) in SKELETON:
            ln, = ax.plot([pos0[a,0], pos0[b,0]], [pos0[a,1], pos0[b,1]],
                         [pos0[a,2], pos0[b,2]], color=color, lw=2.0, alpha=0.90)
            bone_lines.append(ln)
        joint_colors = [color] * NUM_JOINTS
    
    scatter = ax.scatter(pos0[:,0], pos0[:,1], pos0[:,2],
                        c=joint_colors, s=24, alpha=0.95, 
                        depthshade=False, zorder=5)
    
    start_time = time.time()
    
    def update(anim_idx):
        if anim_idx % max(1, n_frames // 20) == 0:
            pct = (anim_idx / n_frames) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / max(anim_idx, 1)) * (n_frames - anim_idx)
            print(f"  Progress: {pct:5.1f}% | Elapsed: {elapsed:5.0f}s | ETA: {eta:5.0f}s",
                  end='\r', flush=True)
        
        source_idx = idx[anim_idx]
        time_sec = source_idx * DT
        pos = positions[source_idx]
        
        for bone_idx, (a, b) in enumerate(SKELETON):
            bone_lines[bone_idx].set_data_3d(
                [pos[a,0], pos[b,0]], [pos[a,1], pos[b,1]], [pos[a,2], pos[b,2]]
            )
        
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        txt_time.set_text(f'Time: {time_sec:.2f}s  |  Frame: {source_idx}/{N}')
        
        if CONFIG['rotate_camera']:
            azim = 30 + anim_idx * CONFIG['rotation_speed']
            ax.view_init(elev=18, azim=azim)
        
        return bone_lines + [scatter]
    
    print(f"\n  Rendering {n_frames} frames...")
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=int(1000 / CONFIG['fps']), blit=True, repeat=True
    )
    
    print(f"\n  Encoding to GIF...")
    writer = animation.PillowWriter(fps=CONFIG['fps'], bitrate=1800)
    ani.save(output_path, writer=writer, dpi=CONFIG['dpi'])
    plt.close(fig)
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n  ✓ Complete in {elapsed:.0f}s")
    print(f"    File: {output_path}")
    print(f"    Size: {file_size:.1f} MB")


# ================================================================
#  THREE-PANEL COMPARISON ANIMATION
# ================================================================
def create_comparison_animation(noisy, lkf, ekf, output_path):
    """Create side-by-side comparison: Measured | LKF | EKF"""
    N = len(lkf)
    idx = np.arange(0, N, CONFIG['step'])
    n_frames = len(idx)
    duration = n_frames / CONFIG['fps']
    
    print(f"\n{'='*60}")
    print(f"Creating THREE-PANEL COMPARISON: {output_path}")
    print(f"{'='*60}")
    print(f"  Source frames : {N}")
    print(f"  Render frames : {n_frames}")
    print(f"  Duration      : {duration:.1f}s")
    
    limits = calculate_limits([noisy, lkf, ekf])
    xlim, ylim, zlim = limits
    
    fig = plt.figure(figsize=(21, 7.5))
    fig.patch.set_facecolor('#090909')
    
    ax_n = fig.add_subplot(131, projection='3d')
    ax_l = fig.add_subplot(132, projection='3d')
    ax_e = fig.add_subplot(133, projection='3d')
    
    style_axis(ax_n, 'Measured (Noisy Sensor)', xlim, ylim, zlim)
    style_axis(ax_l, 'LKF Filtered', xlim, ylim, zlim)
    style_axis(ax_e, 'EKF Filtered (DIVERGING)', xlim, ylim, zlim)
    
    fig.text(0.5, 0.985,
             '3D Full-Body Walking  —  Measured | LKF | EKF (Shows Divergence)',
             ha='center', fontsize=13, fontweight='bold', color='white')
    
    fig.text(0.175, 0.01, 'Raw sensor (noisy)',
             ha='center', fontsize=8, color='#E74C3C')
    fig.text(0.500, 0.01, 'Linear Kalman Filter (optimal)',
             ha='center', fontsize=8, color='#3498DB')
    fig.text(0.825, 0.01, 'Extended Kalman Filter (diverges ~650)',
             ha='center', fontsize=8, color='#27AE60')
    
    txt_time = fig.text(0.5, 0.955, '', ha='center', fontsize=9, color='#AAA')
    
    patches = [
        mpatches.Patch(color='#F5DEB3', label='Spine'),
        mpatches.Patch(color='#FFD700', label='Head'),
        mpatches.Patch(color='#87CEEB', label='R Arm'),
        mpatches.Patch(color='#90EE90', label='L Arm'),
        mpatches.Patch(color='#FFA07A', label='R Leg'),
        mpatches.Patch(color='#DDA0DD', label='L Leg'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=7.5,
               facecolor='#1A1A1A', edgecolor='#333', labelcolor='white',
               framealpha=0.5, bbox_to_anchor=(0.5, 0.03))
    
    datasets = [noisy, lkf, ekf]
    axes = [ax_n, ax_l, ax_e]
    bones_list = [[] for _ in range(3)]
    scatters = []
    
    for k in range(3):
        ax = axes[k]
        pos0 = datasets[k][0]
        
        for (a, b), col in zip(SKELETON, BONE_COLORS):
            ln, = ax.plot([pos0[a,0], pos0[b,0]], [pos0[a,1], pos0[b,1]],
                         [pos0[a,2], pos0[b,2]], color=col, lw=1.8, alpha=0.92)
            bones_list[k].append(ln)
        
        sc = ax.scatter(pos0[:,0], pos0[:,1], pos0[:,2],
                       c=JCOLORS, s=22, alpha=0.95, depthshade=False, zorder=5)
        scatters.append(sc)
    
    start_time = time.time()
    
    def update(anim_idx):
        if anim_idx % max(1, n_frames // 20) == 0:
            pct = (anim_idx / n_frames) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / max(anim_idx, 1)) * (n_frames - anim_idx)
            print(f"  Progress: {pct:5.1f}% | Elapsed: {elapsed:5.0f}s | ETA: {eta:5.0f}s",
                  end='\r', flush=True)
        
        source_idx = idx[anim_idx]
        time_sec = source_idx * DT
        
        for k in range(3):
            pos = datasets[k][source_idx]
            
            for bone_idx, (a, b) in enumerate(SKELETON):
                bones_list[k][bone_idx].set_data_3d(
                    [pos[a,0], pos[b,0]], [pos[a,1], pos[b,1]], [pos[a,2], pos[b,2]]
                )
            
            scatters[k]._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
            
            if CONFIG['rotate_camera']:
                azim = 30 + anim_idx * CONFIG['rotation_speed']
                axes[k].view_init(elev=18, azim=azim)
        
        txt_time.set_text(f'Time: {time_sec:.2f}s  |  Frame: {source_idx}/{N}')
        
        return scatters + [b for bones in bones_list for b in bones]
    
    print(f"\n  Rendering {n_frames} frames...")
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=int(1000 / CONFIG['fps']), blit=True, repeat=True
    )
    
    print(f"\n  Encoding to GIF...")
    writer = animation.PillowWriter(fps=CONFIG['fps'], bitrate=1800)
    ani.save(output_path, writer=writer, dpi=CONFIG['dpi'])
    plt.close(fig)
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n  ✓ Complete in {elapsed:.0f}s")
    print(f"    File: {output_path}")
    print(f"    Size: {file_size:.1f} MB")


# ================================================================
#  MAIN
# ================================================================
def main():
    print("\n" + "="*60)
    print("  3D GAIT ANIMATION GENERATOR")
    print("  Milestone 2 - Section 8")
    print("  RUNNING FULL 3040 FRAMES (30.4 seconds)")
    print("="*60)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key:20s} : {value}")
    
    for filepath in [LKF_CSV, EKF_CSV]:
        if not os.path.exists(filepath):
            print(f"\n✗ ERROR: '{filepath}' not found!")
            print("  Run ./lkf and ./ekf first.")
            return
    
    # ============================================================
    # PART 1: MAIN ANIMATIONS (FULL 3040 FRAMES)
    # ============================================================
    print("\n" + "="*60)
    print("PART 1: Main animations (0-3040 frames, FULL SEQUENCE)")
    print("="*60)
    
    noisy, lkf, ekf = load_all_data(CONFIG['max_frames_normal'])
    
    create_single_animation(
        noisy, 'Measured (Noisy Sensor)',
        OUT_DIR + 'animation_measured.gif', use_anatomical=True
    )
    
    create_single_animation(
        lkf, 'LKF Filtered',
        OUT_DIR + 'animation_lkf.gif', use_anatomical=True
    )
    
    create_single_animation(
        ekf, 'EKF Filtered (Shows Divergence)',
        OUT_DIR + 'animation_ekf.gif', use_anatomical=True
    )
    
    create_comparison_animation(
        noisy, lkf, ekf,
        OUT_DIR + 'animation_comparison.gif'
    )
    
    # ============================================================
    # PART 2: FULL EKF DIVERGENCE SEQUENCE (FULL 3040 FRAMES)
    # ============================================================
    print("\n" + "="*60)
    print("PART 2: Complete EKF divergence (0-3040 frames)")
    print("="*60)
    
    _, _, ekf_full = load_all_data(CONFIG['max_frames_full'])
    
    create_single_animation(
        ekf_full, 'EKF Filtered (Complete Divergence)',
        OUT_DIR + 'animation_ekf_full.gif', use_anatomical=True
    )
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("  ✓ ANIMATION GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {OUT_DIR}")
    print("\nGenerated files:")
    
    for filename in sorted(os.listdir(OUT_DIR)):
        if filename.endswith('.gif'):
            filepath = os.path.join(OUT_DIR, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  • {filename:35s} ({size_mb:5.1f} MB)")
    
    print("\n" + "="*60)
    print("KEY DELIVERABLES:")
    print("  • animation_comparison.gif  ← THREE-PANEL showing divergence")
    print("  • animation_ekf.gif         ← EKF showing divergence start")
    print("  • animation_ekf_full.gif    ← Complete divergence sequence")
    print("\nTIMELINE (FULL 3040 FRAMES = 30.4 seconds):")
    print("  Frame 0-650    : EKF tracking normally")
    print("  Frame 650-1500 : Divergence starts and accelerates")
    print("  Frame 1500-3040: Complete divergence visible")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()