"""
Kalman Filter Milestone 3 - Plots
Karan Kumar - 30212 | Team Alpha | IBA Karachi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Loading data...")
noisy  = pd.read_csv("NoisyValues.csv")
true   = pd.read_csv("TrueValues.csv")
lkf    = pd.read_csv("LKF_asm_output.csv")
ekf    = pd.read_csv("EKF_asm_output.csv")
lkf_m2 = pd.read_csv("LKF_output.csv")
ekf_m2 = pd.read_csv("EKF_output.csv")

JOINT = "pelvis"
T     = len(lkf)
time  = np.arange(T) * 0.01

os.makedirs("plots_m3", exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(f"plots_m3/{name}.png", dpi=150)
    plt.close()
    print(f"  Saved: plots_m3/{name}.png")

# Plot 1: Position time series
print("Plot 1: Position time series...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"LKF-ASM — Position Time Series ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, lkf[f"{JOINT}_p{axis}"], label="LKF-ASM Estimated", color="blue", linewidth=1.2)
    ax.plot(time, noisy[f"{JOINT}_{axis}"], label="Noisy", color="red", linewidth=0.8, alpha=0.5)
    ax.plot(time, true[f"{JOINT}_{axis}"], label="True", color="green", linewidth=1.0, linestyle="--")
    ax.set_ylabel(f"p{axis} (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("01_position_timeseries")

# Plot 2: Velocity time series
print("Plot 2: Velocity time series...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"LKF-ASM — Velocity Time Series ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, lkf[f"{JOINT}_v{axis}"], color="blue", linewidth=1.2)
    ax.set_ylabel(f"v{axis} (m/s)")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("02_velocity_timeseries")

# Plot 3: Acceleration time series
print("Plot 3: Acceleration time series...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"LKF-ASM — Acceleration Time Series ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, lkf[f"{JOINT}_a{axis}"], color="darkorange", linewidth=1.2)
    ax.set_ylabel(f"a{axis} (m/s²)")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("03_acceleration_timeseries")

# Plot 4: Jerk time series
print("Plot 4: Jerk time series...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"LKF-ASM — Jerk Time Series ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, lkf[f"{JOINT}_j{axis}"], color="purple", linewidth=1.2)
    ax.set_ylabel(f"j{axis} (m/s³)")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("04_jerk_timeseries")

# Plot 5: True vs Noisy vs LKF-ASM
print("Plot 5: True vs Noisy vs LKF-ASM...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"True vs Noisy vs LKF-ASM Estimated Position ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, true[f"{JOINT}_{axis}"], label="True", color="green", linewidth=1.5, linestyle="--")
    ax.plot(time, noisy[f"{JOINT}_{axis}"], label="Noisy", color="red", linewidth=0.8, alpha=0.6)
    ax.plot(time, lkf[f"{JOINT}_p{axis}"], label="LKF-ASM", color="blue", linewidth=1.2)
    ax.set_ylabel(f"p{axis} (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("05_true_vs_noisy_vs_lkf")

# Plot 6: True vs Noisy vs EKF-ASM
print("Plot 6: True vs Noisy vs EKF-ASM...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"True vs Noisy vs EKF-ASM Estimated Position ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, true[f"{JOINT}_{axis}"], label="True", color="green", linewidth=1.5, linestyle="--")
    ax.plot(time, noisy[f"{JOINT}_{axis}"], label="Noisy", color="red", linewidth=0.8, alpha=0.6)
    ax.plot(time, ekf[f"{JOINT}_p{axis}"], label="EKF-ASM", color="purple", linewidth=1.2)
    ax.set_ylabel(f"p{axis} (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("06_true_vs_noisy_vs_ekf")

# Plot 7: LKF vs EKF comparison
print("Plot 7: LKF vs EKF comparison...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"LKF-ASM vs EKF-ASM Position Comparison ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, true[f"{JOINT}_{axis}"], label="True", color="green", linewidth=1.5, linestyle="--")
    ax.plot(time, lkf[f"{JOINT}_p{axis}"], label="LKF-ASM", color="blue", linewidth=1.2)
    ax.plot(time, ekf[f"{JOINT}_p{axis}"], label="EKF-ASM", color="purple", linewidth=1.2, linestyle="-.")
    ax.set_ylabel(f"p{axis} (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("07_lkf_vs_ekf")

# Plot 8: Error comparison
print("Plot 8: Error comparison...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Position Error: LKF-ASM vs EKF-ASM ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    lkf_err = np.abs(lkf[f"{JOINT}_p{axis}"].values - true[f"{JOINT}_{axis}"].values)
    ekf_err = np.abs(ekf[f"{JOINT}_p{axis}"].values - true[f"{JOINT}_{axis}"].values)
    ax.plot(time, lkf_err, label=f"LKF Error (mean={lkf_err.mean():.4f})", color="blue", linewidth=1.0)
    ax.plot(time, ekf_err, label=f"EKF Error (mean={ekf_err.mean():.4f})", color="purple", linewidth=1.0)
    ax.set_ylabel(f"|error| {axis} (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("08_error_comparison")

# Plot 9: M2 vs M3 LKF
print("Plot 9: M2 vs M3 LKF comparison...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Milestone-2 (C++) vs Milestone-3 (ASM) — LKF Position ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, lkf_m2[f"{JOINT}_p{axis}"], label="M2 C++", color="blue", linewidth=1.5)
    ax.plot(time, lkf[f"{JOINT}_p{axis}"], label="M3 ASM", color="red", linewidth=1.0, linestyle="--")
    ax.set_ylabel(f"p{axis} (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("09_m2_vs_m3_lkf")

# Plot 10: M2 vs M3 EKF
print("Plot 10: M2 vs M3 EKF comparison...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Milestone-2 (C++) vs Milestone-3 (ASM) — EKF Position ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    ax.plot(time, ekf_m2[f"{JOINT}_p{axis}"], label="M2 C++", color="purple", linewidth=1.5)
    ax.plot(time, ekf[f"{JOINT}_p{axis}"], label="M3 ASM", color="red", linewidth=1.0, linestyle="--")
    ax.set_ylabel(f"p{axis} (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
save("10_m2_vs_m3_ekf")

# Plot 11: M2 vs M3 numerical difference LKF
print("Plot 11: M2 vs M3 numerical difference (LKF)...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Numerical Difference: M2 C++ vs M3 ASM — LKF ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    diff = np.abs(lkf[f"{JOINT}_p{axis}"].values - lkf_m2[f"{JOINT}_p{axis}"].values)
    ax.plot(time, diff, color="darkred", linewidth=0.8)
    ax.set_ylabel(f"|diff| p{axis}")
    if diff.max() > 0:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Max diff: {diff.max():.2e}", fontsize=10)
axes[-1].set_xlabel("Time (s)")
save("11_m2_vs_m3_diff_lkf")

# Plot 12: M2 vs M3 numerical difference EKF
print("Plot 12: M2 vs M3 numerical difference (EKF)...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Numerical Difference: M2 C++ vs M3 ASM — EKF ({JOINT})", fontsize=14)
for ax, axis in zip(axes, ['x', 'y', 'z']):
    diff = np.abs(ekf[f"{JOINT}_p{axis}"].values - ekf_m2[f"{JOINT}_p{axis}"].values)
    ax.plot(time, diff, color="darkred", linewidth=0.8)
    ax.set_ylabel(f"|diff| p{axis}")
    if diff.max() > 0:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Max diff: {diff.max():.2e}", fontsize=10)
axes[-1].set_xlabel("Time (s)")
save("12_m2_vs_m3_diff_ekf")

print("\nAll plots saved in 'plots_m3/' folder!")
