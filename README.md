# Kalman Filter — RISC-V Assembly Implementation
### CAAL S26 | Institute of Business Administration, Karachi

**Team Warjingo** | Muhammad Ismail (31689) · Mehdi Ali (30507) · Muhammad Ismail (30917) · Arham Awan (30934)

🔗 [GitHub Repository](https://github.com/Arham8bit/CAAL_S26_Project_KalmanFilter_Warjingo)

---

## Overview

A full 23-joint 3D human gait tracking system implementing a **Linear Kalman Filter (LKF)** and **Extended Kalman Filter (EKF)**, progressing across four milestones from mathematical derivation through C++ to scalar and vectorised RISC-V assembly.

```
State dimension  N = 276  (12 states × 23 joints)
Measurement dim  M = 69   (3 measurements × 23 joints)
Timesteps        T = 3040
Δt = 0.01 s    σⱼ = 1.0    σᵣ = 0.5
```

---

## Milestones

### Milestone 1 — Mathematical Foundation

Full first-principles derivation of the filter model.

**State vector** (per joint, 12-dimensional):
```
x = [pₓ vₓ aₓ jₓ  pᵧ vᵧ aᵧ jᵧ  p_z v_z a_z j_z]ᵀ
```
Position, velocity, acceleration, and jerk per axis — jerk is included because it captures smooth yet rapidly changing real-world motion that acceleration-only models miss.

**Constant-jerk discrete model** (derived via Taylor expansion):
```
p(t+Δt) = p + vΔt + a(Δt²/2) + j(Δt³/6)
v(t+Δt) = v + aΔt + j(Δt²/2)
a(t+Δt) = a + jΔt
j(t+Δt) = j
```

This gives a 4×4 block `F_axis`; the full **12×12 state transition matrix F** is block-diagonal with three copies (x, y, z axes are independent).

**Process noise Q = G·σⱼ²·Gᵀ** where `G = [Δt³/6, Δt²/2, Δt, 1]ᵀ` per axis — noise enters only at the jerk level and propagates upward.

**LKF measurement model** — sensor gives direct Cartesian position, so H (3×12) simply extracts pₓ, pᵧ, p_z from the state.

**EKF measurement model** — spherical coordinates (range r, azimuth θ, elevation φ) require a nonlinear h(x); the Jacobian H_n (3×12) is derived analytically and recomputed at each timestep:
```
∂r/∂pₓ = pₓ/r       ∂θ/∂pₓ = -pᵧ/ρ²        ∂φ/∂pₓ = -(p_z·pₓ)/(r²ρ)
∂r/∂pᵧ = pᵧ/r       ∂θ/∂pᵧ =  pₓ/ρ²        ∂φ/∂pᵧ = -(pᵧ·p_z)/(r²ρ)
∂r/∂p_z = p_z/r      ∂θ/∂p_z = 0             ∂φ/∂p_z =  ρ/r²
```
where r = √(pₓ²+pᵧ²+p_z²) and ρ = √(pₓ²+pᵧ²).

---

### Milestone 2 — C++ Reference Implementation

LKF and EKF implemented in C++ with a hand-rolled `Matrix` class (no external libraries). Establishes the numerical baseline all later milestones verify against. Covariance updated in Joseph form for numerical stability.

---

### Milestone 3 — RISC-V Scalar Assembly (RV64GD)

Full function-by-function translation into hand-written scalar assembly. All computation in `.s` files; only CSV I/O delegated to C helpers.

Key design decisions:
- **Handle-table architecture** — 33 matrix pointers in a heap array; `s3` holds the base throughout the filter loop
- **Zero-skipping in `multiply_mat`** — F and H are >99% zeros; `feq.d` skips the inner j-loop when `A[i][k] == 0`, reducing effective work ~100×
- **`fmadd.d` / `fnmsub.d`** at every multiply-accumulate point to match C++ `fma()` rounding exactly
- **`solve_system`** — LU decomposition with partial pivoting, 112-byte stack frame, all 13 callee-saved registers used

**Result: bit-identical output to C++ across all 3040 × 276 = 838,240 values (max error = 0.0).**

---

### Milestone 4 — RISC-V Vector Assembly (RVV 1.0)

All hot-path kernels replaced with RVV 1.0 instructions (`VLEN=256`, `e64`, `LMUL=m1`, VLMAX=4 doubles).

| Function | RVV Instructions | Reduction |
|---|---|---|
| `zero_mem` | `vfmv.v.f` + `vse64.v` | 2.33× |
| `mat_add` / `mat_sub` | `vle64.v` + `vfadd/sub.vv` + `vse64.v` | 3.33× |
| `mat_transpose` | `vle64.v` + `vsse64.v` (strided) | 3.14× |
| `mat_mul` (inner j-loop) | `vfmacc.vf` | 3.43× |
| `lu_solve` (6 inner loops) | `vfnmsac.vf` + `vfdiv.vf` | 2.86× |

`vfmacc.vf` is a fused IEEE 754 operation identical in rounding to scalar `fmadd.d`, and the ikj accumulation order is preserved — guaranteeing bit-identical output.

Predicted hardware speedup (Amdahl's Law): **LKF ≈ 2.7×, EKF ≈ 2.1×**

> QEMU 8.2 emulates each vector instruction as VLMAX sequential micro-ops, so wall-clock time is ~4× slower under emulation — this is an emulator artifact, not a hardware result.

**Result: max error = 0.0 for both filters across all 3040 frames.**

---

## Build & Run

**Prerequisites:**
```bash
sudo apt install gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu
# QEMU 8.2 built from source (stock 6.2 does not support RVV)
```

**Milestone 3 — Scalar:**
```bash
cd Milestone-3
make clean && make run       # build, run, verify
```

**Milestone 4 — Vector:**
```bash
cd Milestone-4
make all                     # build M3 reference + M4 vector
make run_vector              # run under QEMU with -cpu rv64,v=on,vlen=256
./verify_m4                  # compare M4 vs M3 output (tolerance 1e-9)
make benchmark               # timed comparison
```

---

## References

1. RISC-V International, *RISC-V "V" Vector Extension*, v1.0 (2021)
2. IEEE 754-2008, *Standard for Floating-Point Arithmetic*
3. A. Becker, *Kalman Filter from the Ground Up*, 2023
4. Li & Jilkov, *A Survey of Maneuvering Target Tracking*, 2003
5. QEMU Project, *QEMU 8.2 User Mode Emulation Documentation*, 2024
6. Team Warjingo, *Milestone 3 Report — RISC-V Scalar Assembly*, April 2026
