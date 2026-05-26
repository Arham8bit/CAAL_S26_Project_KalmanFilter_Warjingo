

.equ NUM_JOINTS,       23
.equ STATE_PER_JOINT,  12
.equ MEAS_PER_JOINT,   3
.equ N,                276
.equ M,                69
.equ NN,               76176
.equ MM,               4761
.equ NM,               19044
.equ NUM_PTRS,         33

# Pointer table offsets (index * 8)
.equ PT_F,         0
.equ PT_Q,         8
.equ PT_H,        16
.equ PT_Ht,       24
.equ PT_R,        32
.equ PT_x,        40
.equ PT_P,        48
.equ PT_Ft,       56
.equ PT_xp,       64
.equ PT_Pp,       72
.equ PT_FP,       80
.equ PT_FPFt,     88
.equ PT_S,        96
.equ PT_HP,      104
.equ PT_HPHt,    112
.equ PT_PHt,     120
.equ PT_PHtt,    128
.equ PT_Ktsol,   136
.equ PT_K,       144
.equ PT_z,       152
.equ PT_Hx,      160
.equ PT_y,       168
.equ PT_Ky,      176
.equ PT_KH,      184
.equ PT_IKH,     192
.equ PT_IKHt,    200
.equ PT_IKHP,    208
.equ PT_IKHPIt,  216
.equ PT_KR,      224
.equ PT_KRKt,    232
.equ PT_Kt,      240
.equ PT_Imat,    248
.equ PT_FILE,    256

    .section .rodata
csv_path:       .string "NoisyValues.csv"
out_path:       .string "LKF_vector_output.csv"
msg_reading:    .string "[LKF-VEC] Reading dataset...\n"
msg_ts:         .string "[LKF-VEC] Timesteps: %d, Cols: %d\n"
msg_building:   .string "[LKF-VEC] Building matrices...\n"
msg_running:    .string "[LKF-VEC] Running filter...\n"
msg_progress:   .string "[LKF-VEC] t=%d/%d\n"
msg_done:       .string "[LKF-VEC] Done. Output saved.\n"

    .section .text


#                 UTILITY: zero_mem(ptr, count)

    .globl zero_mem
zero_mem:
    # VECTORISED: fills 'count' doubles with 0.0 using vse64.v
    fcvt.d.w ft0, zero              # ft0 = 0.0
    li      t0, 0                   # i = 0
zm_loop:
    bge     t0, a1, zm_done
    sub     t1, a1, t0              # remaining
    vsetvli t2, t1, e64, m1, ta, ma # VL = min(remaining, VLMAX)
    vfmv.v.f v0, ft0                # broadcast 0.0 to all lanes
    slli    t3, t0, 3               # byte offset
    add     t4, a0, t3
    vse64.v v0, (t4)                # store VL zeros
    add     t0, t0, t2              # i += VL
    j       zm_loop
zm_done:
    ret

#                    MATH FUNCTIONS


# --- mat_add(A, B, C, size) ---
    .globl mat_add
mat_add:
    # VECTORISED: C[i] = A[i] + B[i], i = 0..size-1
    li      t0, 0
ma_loop:
    bge     t0, a3, ma_done
    sub     t1, a3, t0              # remaining
    vsetvli t2, t1, e64, m1, ta, ma
    slli    t3, t0, 3
    add     t4, a0, t3
    add     t5, a1, t3
    add     t6, a2, t3
    vle64.v v0, (t4)                # load A chunk
    vle64.v v1, (t5)                # load B chunk
    vfadd.vv v2, v0, v1             # C = A + B
    vse64.v v2, (t6)                # store C chunk
    add     t0, t0, t2              # i += VL
    j       ma_loop
ma_done:
    ret

#  mat_sub(A, B, C, size) 
    .globl mat_sub
mat_sub:
    # VECTORISED: C[i] = A[i] - B[i], i = 0..size-1
    li      t0, 0
ms_loop:
    bge     t0, a3, ms_done
    sub     t1, a3, t0
    vsetvli t2, t1, e64, m1, ta, ma
    slli    t3, t0, 3
    add     t4, a0, t3
    add     t5, a1, t3
    add     t6, a2, t3
    vle64.v v0, (t4)
    vle64.v v1, (t5)
    vfsub.vv v2, v0, v1
    vse64.v v2, (t6)
    add     t0, t0, t2
    j       ms_loop
ms_done:
    ret

#  mat_transpose(A, B, rows, cols) 
    .globl mat_transpose
mat_transpose:
    # VECTORISED: B(cols x rows) = A(rows x cols)^T
    # Contiguous row loads from A, strided column stores to B
    # a0=A, a1=B, a2=rows, a3=cols
    slli    a4, a2, 3               # a4 = stride = rows * 8 bytes
    li      t0, 0                   # i = 0 (row counter)
mt_o:
    bge     t0, a2, mt_d
    li      t1, 0                   # j = 0 (col counter)
mt_i:
    bge     t1, a3, mt_nr
    sub     t2, a3, t1              # remaining cols
    vsetvli t3, t2, e64, m1, ta, ma
    # Load contiguous chunk A[i][j .. j+vl-1]
    mul     t4, t0, a3
    add     t4, t4, t1              # i*cols + j
    slli    t4, t4, 3
    add     t5, a0, t4
    vle64.v v0, (t5)
    # Strided store to B[j][i], B[j+1][i], ...
    mul     t4, t1, a2
    add     t4, t4, t0              # j*rows + i
    slli    t4, t4, 3
    add     t5, a1, t4
    vsse64.v v0, (t5), a4           # stride = rows*8
    add     t1, t1, t3              # j += vl
    j       mt_i
mt_nr:
    addi    t0, t0, 1
    j       mt_o
mt_d:
    ret

#  VECTORISED mat_mul(A, B, C, rowsA, colsA, colsB)
#  ikj loop order.  Inner j-loop vectorised with vfmacc.vf.
#  Zero-skip retained for sparse matrices.
    .globl mat_mul
mat_mul:
    addi    sp, sp, -48
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    mv      s0, a0                  # A
    mv      s1, a1                  # B
    mv      s2, a2                  # C
    mv      s3, a3                  # rowsA
    mv      s4, a4                  # colsA
    mv      s5, a5                  # colsB
    # --- Zero C (vectorised) ---
    mul     t0, s3, s5
    fcvt.d.w ft3, zero
    li      t1, 0
mmz:
    bge     t1, t0, mmzd
    sub     t2, t0, t1
    vsetvli t3, t2, e64, m1, ta, ma
    vfmv.v.f v0, ft3
    slli    t4, t1, 3
    add     t5, s2, t4
    vse64.v v0, (t5)
    add     t1, t1, t3
    j       mmz
mmzd:
    # --- Main ikj loop ---
    li      t0, 0                   # i
mmi:
    bge     t0, s3, mmd
    li      t1, 0                   # k
mmk:
    bge     t1, s4, mmni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s0, t2
    fld     ft0, 0(t3)             # ft0 = A[i][k]
    fmv.d.x ft7, zero
    feq.d   a6, ft0, ft7
    bnez    a6, mmnk               # zero-skip
    mul     t4, t1, s5             # k * colsB
    mul     t5, t0, s5             # i * colsB
    li      t6, 0                  # j = 0
mmj:
    bge     t6, s5, mmnk
    sub     a6, s5, t6
    vsetvli a7, a6, e64, m1, ta, ma
    # &B[k][j]
    add     a6, t4, t6
    slli    a6, a6, 3
    add     a6, s1, a6
    vle64.v v1, (a6)               # B[k][j..j+vl-1]
    # &C[i][j]
    add     a6, t5, t6
    slli    a6, a6, 3
    add     a6, s2, a6
    vle64.v v2, (a6)               # C[i][j..j+vl-1]
    vfmacc.vf v2, ft0, v1          # C += A[i][k] * B[k][j..]
    vse64.v v2, (a6)
    add     t6, t6, a7             # j += vl
    j       mmj
mmnk:
    addi    t1, t1, 1
    j       mmk
mmni:
    addi    t0, t0, 1
    j       mmi
mmd:
    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    addi    sp, sp, 48
    ret

#  VECTORISED lu_solve(A, B, X, n, m) 
#  Row swaps, elimination, fwd/back sub all vectorised over j dimension.
    .globl lu_solve
lu_solve:
    addi    sp, sp, -112
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)
    sd      s6, 56(sp)
    sd      s7, 64(sp)
    sd      s8, 72(sp)
    sd      s9, 80(sp)
    sd      s10, 88(sp)
    sd      s11, 96(sp)
    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4

    # --- Allocate LU (n*n) ---
    mul     a0, s3, s3
    slli    a0, a0, 3
    call    malloc
    mv      s5, a0
    # Vectorised copy A -> LU
    mul     t0, s3, s3
    li      t1, 0
lca:
    bge     t1, t0, lcad
    sub     t2, t0, t1
    vsetvli t3, t2, e64, m1, ta, ma
    slli    t4, t1, 3
    add     t5, s0, t4
    add     t6, s5, t4
    vle64.v v0, (t5)
    vse64.v v0, (t6)
    add     t1, t1, t3
    j       lca
lcad:

    # --- Allocate pivot (n ints) ---
    slli    a0, s3, 2
    call    malloc
    mv      s6, a0
    li      t0, 0
lip:
    bge     t0, s3, lipd
    slli    t1, t0, 2
    add     t2, s6, t1
    sw      t0, 0(t2)
    addi    t0, t0, 1
    j       lip
lipd:

    # --- Allocate Bp and Y (n*m each) ---
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s7, a0
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s8, a0

    # ========== LU DECOMPOSITION ==========
    li      s9, 0                   # k (column)
lkl:
    bge     s9, s3, lkd

    # Pivot search (scalar - sequential max)
    mul     t0, s9, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)
    fsgnjx.d ft1, ft0, ft0
    mv      s10, s9
    addi    t0, s9, 1
lps:
    bge     t0, s3, lpd
    mul     t1, t0, s3
    add     t1, t1, s9
    slli    t1, t1, 3
    add     t2, s5, t1
    fld     ft2, 0(t2)
    fsgnjx.d ft3, ft2, ft2
    fle.d   t3, ft3, ft1
    bnez    t3, lpn
    fmv.d   ft1, ft3
    mv      s10, t0
lpn:
    addi    t0, t0, 1
    j       lps
lpd:
    beq     s10, s9, lns

    # Swap pivot indices
    slli    t0, s9, 2
    add     t1, s6, t0
    slli    t0, s10, 2
    add     t2, s6, t0
    lw      t3, 0(t1)
    lw      t4, 0(t2)
    sw      t4, 0(t1)
    sw      t3, 0(t2)

    # Vectorised row swap: LU[s9] <-> LU[s10]
    li      t0, 0
lsl:
    bge     t0, s3, lns
    sub     t1, s3, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s3
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s5, t3              # &LU[s9][t0]
    mul     t3, s10, s3
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t5, s5, t3              # &LU[s10][t0]
    vle64.v v0, (t4)
    vle64.v v1, (t5)
    vse64.v v1, (t4)
    vse64.v v0, (t5)
    add     t0, t0, t2
    j       lsl
lns:

    # Elimination: vectorised inner j-loop
    addi    s11, s9, 1              # i = k+1
lei:
    bge     s11, s3, led
    # Compute multiplier (scalar)
    mul     t0, s11, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)             # LU[i][k]
    mul     t2, s9, s3
    add     t2, t2, s9
    slli    t2, t2, 3
    add     t3, s5, t2
    fld     ft1, 0(t3)             # LU[k][k]
    fdiv.d  ft0, ft0, ft1
    fsd     ft0, 0(t1)
    # Vectorised: LU[i][j] -= ft0 * LU[k][j], j=k+1..n-1
    addi    t4, s9, 1
lej:
    bge     t4, s3, leni
    sub     t5, s3, t4
    vsetvli t6, t5, e64, m1, ta, ma
    mul     a6, s11, s3
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s5, a6             # &LU[i][j]
    vle64.v v0, (a7)
    mul     a6, s9, s3
    add     a6, a6, t4
    slli    a6, a6, 3
    add     t5, s5, a6             # &LU[k][j]
    vle64.v v1, (t5)
    vfnmsac.vf v0, ft0, v1         # LU[i][j] -= mult * LU[k][j]
    vse64.v v0, (a7)
    add     t4, t4, t6
    j       lej
leni:
    addi    s11, s11, 1
    j       lei
led:
    addi    s9, s9, 1
    j       lkl
lkd:

    # ========== PERMUTE B ROWS (vectorised) ==========
    li      t0, 0
lpi:
    bge     t0, s3, lpdd
    slli    t1, t0, 2
    add     t2, s6, t1
    lw      t3, 0(t2)              # piv[i]
    li      t4, 0
lpj:
    bge     t4, s4, lpni
    sub     t5, s4, t4
    vsetvli t6, t5, e64, m1, ta, ma
    mul     a6, t3, s4
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s1, a6             # &B[piv[i]][j]
    vle64.v v0, (a7)
    mul     a6, t0, s4
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s7, a6             # &Bp[i][j]
    vse64.v v0, (a7)
    add     t4, t4, t6
    j       lpj
lpni:
    addi    t0, t0, 1
    j       lpi
lpdd:

    # ========== FORWARD SUBSTITUTION (vectorised over j) ==========
    # Restructured: i outer, k middle, j inner (vectorised)
    # Y[i][j] = Bp[i][j] - sum_{k<i} L[i][k]*Y[k][j]
    li      s9, 0                   # i = 0
lfi:
    bge     s9, s3, lfd
    # Copy Bp row i -> Y row i (vectorised)
    li      t0, 0
lf_cp:
    bge     t0, s4, lf_cpd
    sub     t1, s4, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s4
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s7, t3             # &Bp[i][j]
    add     t5, s8, t3             # &Y[i][j]
    vle64.v v0, (t4)
    vse64.v v0, (t5)
    add     t0, t0, t2
    j       lf_cp
lf_cpd:
    # For k = 0..i-1: Y[i][j] -= L[i][k] * Y[k][j]
    li      s10, 0                 # k = 0
lfk:
    bge     s10, s9, lfni          # k < i
    mul     t0, s9, s3
    add     t0, t0, s10
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)            # L[i][k]
    li      t0, 0                  # j = 0
lfj_v:
    bge     t0, s4, lfk_nx
    sub     t1, s4, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s4
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s8, t3             # &Y[i][j]
    mul     t5, s10, s4
    add     t5, t5, t0
    slli    t5, t5, 3
    add     t6, s8, t5             # &Y[k][j]
    vle64.v v0, (t4)
    vle64.v v1, (t6)
    vfnmsac.vf v0, ft0, v1        # Y[i][j] -= L[i][k]*Y[k][j]
    vse64.v v0, (t4)
    add     t0, t0, t2
    j       lfj_v
lfk_nx:
    addi    s10, s10, 1
    j       lfk
lfni:
    addi    s9, s9, 1
    j       lfi
lfd:

    # ========== BACK SUBSTITUTION (vectorised over j) ==========
    # Copy Y -> X first, then update X in-place
    mul     t0, s3, s4
    li      t1, 0
lb_cp:
    bge     t1, t0, lb_cpd
    sub     t2, t0, t1
    vsetvli t3, t2, e64, m1, ta, ma
    slli    t4, t1, 3
    add     t5, s8, t4             # &Y
    add     t6, s2, t4             # &X
    vle64.v v0, (t5)
    vse64.v v0, (t6)
    add     t1, t1, t3
    j       lb_cp
lb_cpd:

    addi    s9, s3, -1             # i = n-1
lbi:
    bltz    s9, lbd
    # For k = i+1..n-1: X[i][j] -= U[i][k]*X[k][j]
    addi    s10, s9, 1
lbk:
    bge     s10, s3, lb_div
    mul     t0, s9, s3
    add     t0, t0, s10
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)            # U[i][k]
    li      t0, 0
lbj_v:
    bge     t0, s4, lbk_nx
    sub     t1, s4, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s4
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s2, t3             # &X[i][j]
    mul     t5, s10, s4
    add     t5, t5, t0
    slli    t5, t5, 3
    add     t6, s2, t5             # &X[k][j]
    vle64.v v0, (t4)
    vle64.v v1, (t6)
    vfnmsac.vf v0, ft0, v1        # X[i][j] -= U[i][k]*X[k][j]
    vse64.v v0, (t4)
    add     t0, t0, t2
    j       lbj_v
lbk_nx:
    addi    s10, s10, 1
    j       lbk
lb_div:
    # Divide X row i by U[i][i] (vectorised)
    mul     t0, s9, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft1, 0(t1)            # U[i][i]
    li      t0, 0
lb_dv:
    bge     t0, s4, lbni
    sub     t1, s4, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s4
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s2, t3
    vle64.v v0, (t4)
    vfdiv.vf v0, v0, ft1          # X[i][j] /= U[i][i]
    vse64.v v0, (t4)
    add     t0, t0, t2
    j       lb_dv
lbni:
    addi    s9, s9, -1
    j       lbi
lbd:

    # --- Free workspace ---
    mv      a0, s5
    call    free
    mv      a0, s6
    call    free
    mv      a0, s7
    call    free
    mv      a0, s8
    call    free
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    ld      s6, 56(sp)
    ld      s7, 64(sp)
    ld      s8, 72(sp)
    ld      s9, 80(sp)
    ld      s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 112
    ret

#
#                         MAIN
# 
    .globl main
main:
    addi    sp, sp, -64
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    # sp+48 = rows, sp+52 = cols (temps for read_csv)

    #     - Read CSV     
    la      a0, msg_reading
    call    printf
    la      a0, csv_path
    addi    a1, sp, 48
    addi    a2, sp, 52
    call    read_csv
    mv      s0, a0              # s0 = noisy data (flat)
    lw      s1, 48(sp)          # s1 = T
    lw      s2, 52(sp)          # s2 = cols
    la      a0, msg_ts
    mv      a1, s1
    mv      a2, s2
    call    printf

    #      Allocate pointer table (33 pointers)     
    li      a0, NUM_PTRS*8
    call    malloc
    mv      s3, a0              # s3 = ptbl

    la      a0, msg_building
    call    printf

    #      Allocate all matrices     
    # Helper macro pattern: malloc size, store in ptbl

    # F (NN doubles)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_F(s3)

    # Q
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Q(s3)

    # H (M*N)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_H(s3)

    # Ht (N*M)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Ht(s3)

    # R (MM)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_R(s3)

    # x (N)
    li      a0, N*8
    call    malloc
    sd      a0, PT_x(s3)

    # P (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_P(s3)

    # Ft (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Ft(s3)

    # x_pred (N)
    li      a0, N*8
    call    malloc
    sd      a0, PT_xp(s3)

    # P_pred (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Pp(s3)

    # FP (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_FP(s3)

    # FPFt (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_FPFt(s3)

    # S (MM)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_S(s3)

    # HP (MN)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_HP(s3)

    # HPHt (MM)
    li      a0, MM*8
    call    malloc
    sd      a0, PT_HPHt(s3)

    # PHt (NM)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_PHt(s3)

    # PHt_t (MN)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_PHtt(s3)

    # Kt_sol (MN)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Ktsol(s3)

    # K (NM)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_K(s3)

    # z (M)
    li      a0, M*8
    call    malloc
    sd      a0, PT_z(s3)

    # Hx (M)
    li      a0, M*8
    call    malloc
    sd      a0, PT_Hx(s3)

    # y (M)
    li      a0, M*8
    call    malloc
    sd      a0, PT_y(s3)

    # Ky (N)
    li      a0, N*8
    call    malloc
    sd      a0, PT_Ky(s3)

    # KH (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_KH(s3)

    # IKH (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKH(s3)

    # IKHt (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHt(s3)

    # IKH_P (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHP(s3)

    # IKH_P_It (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKHPIt(s3)

    # KR (NM)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_KR(s3)

    # KRKt (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_KRKt(s3)

    # Kt (MN)
    li      a0, NM*8
    call    malloc
    sd      a0, PT_Kt(s3)

    # I_mat (NN)
    li      a0, NN*8
    call    malloc
    sd      a0, PT_Imat(s3)

    #      BUILD F     
    ld      a0, PT_F(s3)
    li      a1, NN
    call    zero_mem

    ld      s4, PT_F(s3)        # s4 = F (temp use before filter loop)

    # Load FP constants
    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0             # DT = 0.01
    fmul.d  ft1, ft0, ft0       # DT^2
    li      t0, 0x3FE0000000000000
    fmv.d.x ft8, t0             # 0.5
    fmul.d  ft2, ft8, ft1       # DT^2/2
    fmul.d  ft3, ft1, ft0       # DT^3
    li      t0, 0x3FC5555555555555
    fmv.d.x ft9, t0             # 1/6
    fmul.d  ft4, ft9, ft3       # DT^3/6
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0             # 1.0

    li      t0, 0               # jt = 0
bF_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, bF_done
    li      t2, 0               # axis = 0
bF_ax:
    li      t3, 3
    bge     t2, t3, bF_njt
    # base = jt*12 + axis*4
    li      t4, 12
    mul     t4, t0, t4
    slli    t5, t2, 2
    add     t4, t4, t5          # base

    # Row 0: F[b][b]=1, F[b][b+1]=dt, F[b][b+2]=dt2/2, F[b][b+3]=dt3/6
    li      a1, N
    mul     a2, t4, a1          # base*N
    add     a3, a2, t4          # base*N + base
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)          # [b][b]=1
    addi    a3, t4, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)          # [b][b+1]=dt
    addi    a3, t4, 2
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft2, 0(a4)          # [b][b+2]=dt2/2
    addi    a3, t4, 3
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft4, 0(a4)          # [b][b+3]=dt3/6

    # Row 1
    addi    a5, t4, 1
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)          # 1
    addi    a3, a5, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)          # dt
    addi    a3, a5, 2
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft2, 0(a4)          # dt2/2

    # Row 2
    addi    a5, t4, 2
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)          # 1
    addi    a3, a5, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)          # dt

    # Row 3
    addi    a5, t4, 3
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)          # 1

    addi    t2, t2, 1
    j       bF_ax
bF_njt:
    addi    t0, t0, 1
    j       bF_jt
bF_done:

    #      BUILD Q     
    ld      s4, PT_Q(s3)
    mv      a0, s4
    li      a1, NN
    call    zero_mem

    # Reload constants (clobbered by call)
    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0
    fmul.d  ft1, ft0, ft0
    li      t0, 0x3FE0000000000000
    fmv.d.x ft8, t0
    fmul.d  ft2, ft8, ft1
    fmul.d  ft3, ft1, ft0
    li      t0, 0x3FC5555555555555
    fmv.d.x ft9, t0
    fmul.d  ft4, ft9, ft3
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0
    # g[0]=ft4=dt3/6, g[1]=ft2=dt2/2, g[2]=ft0=dt, g[3]=ft5=1.0
    # Store g on stack (sp+48..sp+79, reusing temp space)
    fsd     ft4, 48(sp)
    fsd     ft2, 56(sp)
    li      a0, 32
    call    malloc
    mv      s4, a0              # s4 = g array temp

    # Reload constants AGAIN
    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0
    fmul.d  ft1, ft0, ft0
    li      t0, 0x3FE0000000000000
    fmv.d.x ft8, t0
    fmul.d  ft2, ft8, ft1
    fmul.d  ft3, ft1, ft0
    li      t0, 0x3FC5555555555555
    fmv.d.x ft9, t0
    fmul.d  ft4, ft9, ft3
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0

    fsd     ft4, 0(s4)          # g[0]
    fsd     ft2, 8(s4)          # g[1]
    fsd     ft0, 16(s4)         # g[2]
    fsd     ft5, 24(s4)         # g[3]

    ld      a5, PT_Q(s3)        # a5 = Q pointer
    li      t0, 0
bQ_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, bQ_done
    li      t2, 0
bQ_ax:
    li      t3, 3
    bge     t2, t3, bQ_njt
    li      t4, 0               # i
bQ_i:
    li      t5, 4
    bge     t4, t5, bQ_nax
    li      t6, 0               # j
bQ_j:
    li      a1, 4
    bge     t6, a1, bQ_ni
    slli    a2, t4, 3
    add     a3, s4, a2
    fld     ft0, 0(a3)
    slli    a2, t6, 3
    add     a3, s4, a2
    fld     ft1, 0(a3)
    fmul.d  ft2, ft0, ft1      # g[i]*g[j] (sigma=1)

    li      a1, 12
    mul     a2, t0, a1
    slli    a3, t2, 2
    add     a2, a2, a3
    add     a2, a2, t4          # row = jt*12+axis*4+i

    mul     a3, t0, a1
    slli    a4, t2, 2
    add     a3, a3, a4
    add     a3, a3, t6          # col = jt*12+axis*4+j

    li      a1, N
    mul     a4, a2, a1
    add     a4, a4, a3
    slli    a4, a4, 3
    add     a4, a5, a4
    fsd     ft2, 0(a4)

    addi    t6, t6, 1
    j       bQ_j
bQ_ni:
    addi    t4, t4, 1
    j       bQ_i
bQ_nax:
    addi    t2, t2, 1
    j       bQ_ax
bQ_njt:
    addi    t0, t0, 1
    j       bQ_jt
bQ_done:
    mv      a0, s4
    call    free                # free g

    #      BUILD H     
    ld      a0, PT_H(s3)
    li      a1, NM
    call    zero_mem
    ld      s4, PT_H(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0
    li      t0, 0
bH_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, bH_done
    li      a1, N
    # H[(j*3)*N + j*12] = 1
    li      t2, 3
    mul     t3, t0, t2          # j*3
    mul     t4, t3, a1          # (j*3)*N
    li      t2, 12
    mul     t5, t0, t2          # j*12
    add     t6, t4, t5
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)

    # H[(j*3+1)*N + j*12+4] = 1
    addi    t6, t3, 1
    mul     t6, t6, a1
    addi    a2, t5, 4
    add     t6, t6, a2
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)

    # H[(j*3+2)*N + j*12+8] = 1
    addi    t6, t3, 2
    mul     t6, t6, a1
    addi    a2, t5, 8
    add     t6, t6, a2
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)

    addi    t0, t0, 1
    j       bH_lp
bH_done:

    #      BUILD R (diagonal, 0.25)     
    ld      a0, PT_R(s3)
    li      a1, MM
    call    zero_mem
    ld      s4, PT_R(s3)
    li      t0, 0x3FD0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
bR_lp:
    li      t2, M
    bge     t1, t2, bR_done
    li      a1, M
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       bR_lp
bR_done:

    #      BUILD I_mat (N*N identity)     
    ld      a0, PT_Imat(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_Imat(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
bI_lp:
    li      t2, N
    bge     t1, t2, bI_done
    li      a1, N
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       bI_lp
bI_done:

    #      INIT x from first measurement     
    ld      a0, PT_x(s3)
    li      a1, N
    call    zero_mem
    ld      s4, PT_x(s3)
    li      t0, 0
ix_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, ix_done
    li      a1, 3
    mul     a2, t0, a1
    slli    a2, a2, 3
    add     a3, s0, a2
    fld     ft0, 0(a3)
    fld     ft1, 8(a3)
    fld     ft2, 16(a3)
    li      a1, 12
    mul     a2, t0, a1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)          # x[j*12+0]
    fsd     ft1, 32(a3)         # x[j*12+4]  (4*8=32)
    fsd     ft2, 64(a3)         # x[j*12+8]  (8*8=64)
    addi    t0, t0, 1
    j       ix_lp
ix_done:

    #      INIT P = I     
    ld      a0, PT_P(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_P(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
iP_lp:
    li      t2, N
    bge     t1, t2, iP_done
    li      a1, N
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       iP_lp
iP_done:

    #      Compute Ht = H^T     
    ld      a0, PT_H(s3)
    ld      a1, PT_Ht(s3)
    li      a2, M
    li      a3, N
    call    mat_transpose

    #      Compute Ft = F^T     
    ld      a0, PT_F(s3)
    ld      a1, PT_Ft(s3)
    li      a2, N
    li      a3, N
    call    mat_transpose

    #      Open output     
    la      a0, out_path
    call    open_output
    sd      a0, PT_FILE(s3)

    #      FILTER LOOP     
    la      a0, msg_running
    call    printf

    li      s4, 0               # s4 = t
filter_loop:
    bge     s4, s1, filter_done

    #      PREDICTION: x_pred = F * x     
    ld      a0, PT_F(s3)
    ld      a1, PT_x(s3)
    ld      a2, PT_xp(s3)
    li      a3, N
    li      a4, N
    li      a5, 1
    call    mat_mul

    #      FP = F * P     
    ld      a0, PT_F(s3)
    ld      a1, PT_P(s3)
    ld      a2, PT_FP(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    #      FPFt = FP * Ft     
    ld      a0, PT_FP(s3)
    ld      a1, PT_Ft(s3)
    ld      a2, PT_FPFt(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    #      P_pred = FPFt + Q     
    ld      a0, PT_FPFt(s3)
    ld      a1, PT_Q(s3)
    ld      a2, PT_Pp(s3)
    li      a3, NN
    call    mat_add

    #      Load z from noisy[t]     
    mv      a0, s4
    mv      a1, s2
    mul     a2, a0, a1
    slli    a2, a2, 3
    add     a2, s0, a2          # &noisy[t*cols]
    ld      a3, PT_z(s3)
    li      t0, 0
lz_lp:
    li      t1, M
    bge     t0, t1, lz_done
    slli    t2, t0, 3
    add     t3, a2, t2
    add     t4, a3, t2
    fld     ft0, 0(t3)
    fsd     ft0, 0(t4)
    addi    t0, t0, 1
    j       lz_lp
lz_done:

    #      HP = H * P_pred     
    ld      a0, PT_H(s3)
    ld      a1, PT_Pp(s3)
    ld      a2, PT_HP(s3)
    li      a3, M
    li      a4, N
    li      a5, N
    call    mat_mul

    #      HPHt = HP * Ht     
    ld      a0, PT_HP(s3)
    ld      a1, PT_Ht(s3)
    ld      a2, PT_HPHt(s3)
    li      a3, M
    li      a4, N
    li      a5, M
    call    mat_mul

    #      S = HPHt + R     
    ld      a0, PT_HPHt(s3)
    ld      a1, PT_R(s3)
    ld      a2, PT_S(s3)
    li      a3, MM
    call    mat_add

    #      PHt = P_pred * Ht     
    ld      a0, PT_Pp(s3)
    ld      a1, PT_Ht(s3)
    ld      a2, PT_PHt(s3)
    li      a3, N
    li      a4, N
    li      a5, M
    call    mat_mul

    #      PHt_t = PHt^T (MxN)     
    ld      a0, PT_PHt(s3)
    ld      a1, PT_PHtt(s3)
    li      a2, N
    li      a3, M
    call    mat_transpose

    #      Solve S * Kt_sol = PHt_t     
    ld      a0, PT_S(s3)
    ld      a1, PT_PHtt(s3)
    ld      a2, PT_Ktsol(s3)
    li      a3, M
    li      a4, N
    call    lu_solve

    #      K = Kt_sol^T (NxM)     
    ld      a0, PT_Ktsol(s3)
    ld      a1, PT_K(s3)
    li      a2, M
    li      a3, N
    call    mat_transpose

    #      Hx = H * x_pred     
    ld      a0, PT_H(s3)
    ld      a1, PT_xp(s3)
    ld      a2, PT_Hx(s3)
    li      a3, M
    li      a4, N
    li      a5, 1
    call    mat_mul

    #      y = z - Hx     
    ld      a0, PT_z(s3)
    ld      a1, PT_Hx(s3)
    ld      a2, PT_y(s3)
    li      a3, M
    call    mat_sub

    #      Ky = K * y     
    ld      a0, PT_K(s3)
    ld      a1, PT_y(s3)
    ld      a2, PT_Ky(s3)
    li      a3, N
    li      a4, M
    li      a5, 1
    call    mat_mul

    #      x = x_pred + Ky     
    ld      a0, PT_xp(s3)
    ld      a1, PT_Ky(s3)
    ld      a2, PT_x(s3)
    li      a3, N
    call    mat_add

    #      KH = K * H     
    ld      a0, PT_K(s3)
    ld      a1, PT_H(s3)
    ld      a2, PT_KH(s3)
    li      a3, N
    li      a4, M
    li      a5, N
    call    mat_mul

    #      IKH = I - KH     
    ld      a0, PT_Imat(s3)
    ld      a1, PT_KH(s3)
    ld      a2, PT_IKH(s3)
    li      a3, NN
    call    mat_sub

    #      IKHt = IKH^T     
    ld      a0, PT_IKH(s3)
    ld      a1, PT_IKHt(s3)
    li      a2, N
    li      a3, N
    call    mat_transpose

    #      IKH_P = IKH * P_pred     
    ld      a0, PT_IKH(s3)
    ld      a1, PT_Pp(s3)
    ld      a2, PT_IKHP(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    #      IKH_P_It = IKH_P * IKHt     
    ld      a0, PT_IKHP(s3)
    ld      a1, PT_IKHt(s3)
    ld      a2, PT_IKHPIt(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    #      KR = K * R     
    ld      a0, PT_K(s3)
    ld      a1, PT_R(s3)
    ld      a2, PT_KR(s3)
    li      a3, N
    li      a4, M
    li      a5, M
    call    mat_mul

    #      Kt = K^T     
    ld      a0, PT_K(s3)
    ld      a1, PT_Kt(s3)
    li      a2, N
    li      a3, M
    call    mat_transpose

    #      KRKt = KR * Kt     
    ld      a0, PT_KR(s3)
    ld      a1, PT_Kt(s3)
    ld      a2, PT_KRKt(s3)
    li      a3, N
    li      a4, M
    li      a5, N
    call    mat_mul

    #      P = IKH_P_It + KRKt     
    ld      a0, PT_IKHPIt(s3)
    ld      a1, PT_KRKt(s3)
    ld      a2, PT_P(s3)
    li      a3, NN
    call    mat_add

    #      Write row     
    ld      a0, PT_FILE(s3)
    ld      a1, PT_x(s3)
    li      a2, N
    call    write_row

    #      Progress every 500 steps     
    li      t0, 500
    rem     t1, s4, t0
    bnez    t1, no_print
    la      a0, msg_progress
    mv      a1, s4
    mv      a2, s1
    call    printf
no_print:

    addi    s4, s4, 1
    j       filter_loop

filter_done:
    ld      a0, PT_FILE(s3)
    call    close_output

    la      a0, msg_done
    call    printf

    li      a0, 0
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    addi    sp, sp, 64
    ret

