
# =========================================================================
#  Extended Kalman Filter - RISC-V RV64GD Assembly
#  Karan Kumar - 30212 | Team Alpha | Milestone 3
#
#  Architecture:
#    s0 -> raw sensor data base address
#    s1 -> total frames count
#    s2 -> columns per frame
#    s3 -> matrix handle table base
#    s4 -> current frame index
#
#  Handle Table Layout:
#    [0]=F [1]=Q [2]=R [3]=state [4]=cov [5]=Ft
#    [6]=state_p [7]=cov_p [8]=tmp_fp [9]=tmp_fpft [10]=innov_cov
#    [11]=jac [12]=jacT [13]=tmp_jp [14]=tmp_jpjt
#    [15]=tmp_pjt [16]=tmp_pjtt [17]=gain_sol
#    [18]=gain [19]=meas_sph [20]=pred_sph [21]=innov [22]=gain_innov
#    [23]=tmp_kj [24]=tmp_ikj [25]=tmp_ikjt [26]=tmp_ikjp [27]=tmp_ikjpt
#    [28]=tmp_kr [29]=tmp_krkt [30]=gain_t [31]=eye [32]=output_file
# =========================================================================

.equ JOINTS,          23
.equ DIM_PER_JOINT,   12
.equ MEAS_DIM_JOINT,  3
.equ STATE_DIM,       276
.equ MEAS_DIM,        69
.equ SQ_STATE,        76176
.equ SQ_MEAS,         4761
.equ CROSS_DIM,       19044
.equ HANDLE_COUNT,    33

.equ H_F,        0
.equ H_Q,        8
.equ H_R,       16
.equ H_st,      24
.equ H_cv,      32
.equ H_Ft,      40
.equ H_stp,     48
.equ H_cvp,     56
.equ H_fp,      64
.equ H_fpft,    72
.equ H_S,       80
.equ H_jac,     88
.equ H_jacT,    96
.equ H_jp,     104
.equ H_jpjt,   112
.equ H_pjt,    120
.equ H_pjtt,   128
.equ H_ksol,   136
.equ H_K,      144
.equ H_zsph,   152
.equ H_hsph,   160
.equ H_inn,    168
.equ H_kinn,   176
.equ H_kj,     184
.equ H_ikj,    192
.equ H_ikjt,   200
.equ H_ikjp,   208
.equ H_ikjpt,  216
.equ H_kr,     224
.equ H_krkt,   232
.equ H_Kt,     240
.equ H_eye,    248
.equ H_fout,   256

    .section .rodata
# trigonometric constants for atan2 approximation
c_pio4:     .double 0.7853981633974483
c_a1:       .double 0.2447
c_a2:       .double 0.0663
c_one:      .double 1.0
c_pi:       .double 3.14159265358979323846
c_halfpi:   .double 1.5707963267948966
c_epsilon:  .double 1.0e-9

path_input:     .string "NoisyValues.csv"
path_output:    .string "EKF_asm_output.csv"
str_load:       .string "[EKF] Loading data...\n"
str_info:       .string "[EKF] Frames=%d Cols=%d\n"
str_init:       .string "[EKF] Initializing system...\n"
str_start:      .string "[EKF] Processing...\n"
str_step:       .string "[EKF] Frame %d/%d\n"
str_finish:     .string "[EKF] Complete.\n"

    .section .text

# ===================== clear_doubles(ptr, count) =========================
    .globl clear_doubles
clear_doubles:
    li      t0, 0
    fcvt.d.w ft0, zero
.Lcd_lp:
    bge     t0, a1, .Lcd_dn
    slli    t1, t0, 3
    add     t2, a0, t1
    fsd     ft0, 0(t2)
    addi    t0, t0, 1
    j       .Lcd_lp
.Lcd_dn:
    ret

# ===================== vec_add(A, B, C, len) =============================
    .globl vec_add
vec_add:
    li      t0, 0
.Lva_lp:
    bge     t0, a3, .Lva_dn
    slli    t1, t0, 3
    add     t2, a0, t1
    add     t3, a1, t1
    add     t4, a2, t1
    fld     ft0, 0(t2)
    fld     ft1, 0(t3)
    fadd.d  ft2, ft0, ft1
    fsd     ft2, 0(t4)
    addi    t0, t0, 1
    j       .Lva_lp
.Lva_dn:
    ret

# ===================== vec_sub(A, B, C, len) =============================
    .globl vec_sub
vec_sub:
    li      t0, 0
.Lvs_lp:
    bge     t0, a3, .Lvs_dn
    slli    t1, t0, 3
    add     t2, a0, t1
    add     t3, a1, t1
    add     t4, a2, t1
    fld     ft0, 0(t2)
    fld     ft1, 0(t3)
    fsub.d  ft2, ft0, ft1
    fsd     ft2, 0(t4)
    addi    t0, t0, 1
    j       .Lvs_lp
.Lvs_dn:
    ret

# ===================== transpose_mat(src, dst, nrow, ncol) ===============
    .globl transpose_mat
transpose_mat:
    li      t0, 0
.Ltm_r:
    bge     t0, a2, .Ltm_dn
    li      t1, 0
.Ltm_c:
    bge     t1, a3, .Ltm_nr
    mul     t2, t0, a3
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, a0, t2
    mul     t4, t1, a2
    add     t4, t4, t0
    slli    t4, t4, 3
    add     t5, a1, t4
    fld     ft0, 0(t3)
    fsd     ft0, 0(t5)
    addi    t1, t1, 1
    j       .Ltm_c
.Ltm_nr:
    addi    t0, t0, 1
    j       .Ltm_r
.Ltm_dn:
    ret

# ===================== multiply_mat(A, B, C, rA, cA, cB) =================
    .globl multiply_mat
multiply_mat:
    addi    sp, sp, -48
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4
    mv      s5, a5
    mul     t0, s3, s5
    li      t1, 0
    fcvt.d.w ft3, zero
.Lmm_clr:
    bge     t1, t0, .Lmm_clrd
    slli    t2, t1, 3
    add     t3, s2, t2
    fsd     ft3, 0(t3)
    addi    t1, t1, 1
    j       .Lmm_clr
.Lmm_clrd:
    li      t0, 0
.Lmm_r:
    bge     t0, s3, .Lmm_dn
    li      t1, 0
.Lmm_m:
    bge     t1, s4, .Lmm_nr
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s0, t2
    fld     ft0, 0(t3)
    fmv.d.x ft7, zero
    feq.d   a6, ft0, ft7
    bnez    a6, .Lmm_nm
    mul     t4, t1, s5
    mul     t5, t0, s5
    li      t6, 0
.Lmm_c:
    bge     t6, s5, .Lmm_nm
    add     a6, t4, t6
    slli    a6, a6, 3
    add     a7, s1, a6
    fld     ft1, 0(a7)
    add     a6, t5, t6
    slli    a6, a6, 3
    add     a7, s2, a6
    fld     ft2, 0(a7)
    fmadd.d ft2, ft0, ft1, ft2
    fsd     ft2, 0(a7)
    addi    t6, t6, 1
    j       .Lmm_c
.Lmm_nm:
    addi    t1, t1, 1
    j       .Lmm_m
.Lmm_nr:
    addi    t0, t0, 1
    j       .Lmm_r
.Lmm_dn:
    ld      s0,  0(sp)
    ld      s1,  8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    addi    sp, sp, 48
    ret

# ===================== solve_system(A, B, X, n, m) =======================
    .globl solve_system
solve_system:
    addi    sp, sp, -112
    sd      ra,   0(sp)
    sd      s0,   8(sp)
    sd      s1,  16(sp)
    sd      s2,  24(sp)
    sd      s3,  32(sp)
    sd      s4,  40(sp)
    sd      s5,  48(sp)
    sd      s6,  56(sp)
    sd      s7,  64(sp)
    sd      s8,  72(sp)
    sd      s9,  80(sp)
    sd      s10, 88(sp)
    sd      s11, 96(sp)
    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4
    mul     a0, s3, s3
    slli    a0, a0, 3
    call    malloc
    mv      s5, a0
    mul     t0, s3, s3
    li      t1, 0
.Lss_cp:
    bge     t1, t0, .Lss_cpd
    slli    t2, t1, 3
    add     t3, s0, t2
    add     t4, s5, t2
    fld     ft0, 0(t3)
    fsd     ft0, 0(t4)
    addi    t1, t1, 1
    j       .Lss_cp
.Lss_cpd:
    slli    a0, s3, 2
    call    malloc
    mv      s6, a0
    li      t0, 0
.Lss_pv:
    bge     t0, s3, .Lss_pvd
    slli    t1, t0, 2
    add     t2, s6, t1
    sw      t0, 0(t2)
    addi    t0, t0, 1
    j       .Lss_pv
.Lss_pvd:
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s7, a0
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s8, a0

    # factorization
    li      s9, 0
.Lss_fk:
    bge     s9, s3, .Lss_fkd
    mul     t0, s9, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)
    fsgnjx.d ft1, ft0, ft0
    mv      s10, s9
    addi    t0, s9, 1
.Lss_ps:
    bge     t0, s3, .Lss_psd
    mul     t1, t0, s3
    add     t1, t1, s9
    slli    t1, t1, 3
    add     t2, s5, t1
    fld     ft2, 0(t2)
    fsgnjx.d ft3, ft2, ft2
    fle.d   t3, ft3, ft1
    bnez    t3, .Lss_psn
    fmv.d   ft1, ft3
    mv      s10, t0
.Lss_psn:
    addi    t0, t0, 1
    j       .Lss_ps
.Lss_psd:
    beq     s10, s9, .Lss_ns
    slli    t0, s9, 2
    add     t1, s6, t0
    slli    t0, s10, 2
    add     t2, s6, t0
    lw      t3, 0(t1)
    lw      t4, 0(t2)
    sw      t4, 0(t1)
    sw      t3, 0(t2)
    li      t0, 0
.Lss_sw:
    bge     t0, s3, .Lss_ns
    mul     t1, s9, s3
    add     t1, t1, t0
    slli    t1, t1, 3
    add     t2, s5, t1
    mul     t3, s10, s3
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s5, t3
    fld     ft4, 0(t2)
    fld     ft5, 0(t4)
    fsd     ft5, 0(t2)
    fsd     ft4, 0(t4)
    addi    t0, t0, 1
    j       .Lss_sw
.Lss_ns:
    addi    s11, s9, 1
.Lss_ei:
    bge     s11, s3, .Lss_eid
    mul     t0, s11, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)
    mul     t2, s9, s3
    add     t2, t2, s9
    slli    t2, t2, 3
    add     t3, s5, t2
    fld     ft1, 0(t3)
    fdiv.d  ft0, ft0, ft1
    fsd     ft0, 0(t1)
    addi    t4, s9, 1
.Lss_ej:
    bge     t4, s3, .Lss_eni
    mul     t5, s11, s3
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, s5, t5
    fld     ft2, 0(t6)
    mul     a6, s9, s3
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s5, a6
    fld     ft3, 0(a7)
    fnmsub.d ft2, ft0, ft3, ft2
    fsd     ft2, 0(t6)
    addi    t4, t4, 1
    j       .Lss_ej
.Lss_eni:
    addi    s11, s11, 1
    j       .Lss_ei
.Lss_eid:
    addi    s9, s9, 1
    j       .Lss_fk
.Lss_fkd:

    # permute
    li      t0, 0
.Lss_pi:
    bge     t0, s3, .Lss_pid
    slli    t1, t0, 2
    add     t2, s6, t1
    lw      t3, 0(t2)
    li      t4, 0
.Lss_pj:
    bge     t4, s4, .Lss_pni
    mul     t5, t3, s4
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, s1, t5
    fld     ft0, 0(t6)
    mul     a6, t0, s4
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s7, a6
    fsd     ft0, 0(a7)
    addi    t4, t4, 1
    j       .Lss_pj
.Lss_pni:
    addi    t0, t0, 1
    j       .Lss_pi
.Lss_pid:

    # forward sub
    li      t0, 0
.Lss_fi:
    bge     t0, s3, .Lss_fid
    li      t1, 0
.Lss_fj:
    bge     t1, s4, .Lss_fni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s7, t2
    fld     ft0, 0(t3)
    li      t4, 0
.Lss_fk2:
    bge     t4, t0, .Lss_fst
    mul     t5, t0, s3
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, s5, t5
    fld     ft1, 0(t6)
    mul     a6, t4, s4
    add     a6, a6, t1
    slli    a6, a6, 3
    add     a7, s8, a6
    fld     ft2, 0(a7)
    fnmsub.d ft0, ft1, ft2, ft0
    addi    t4, t4, 1
    j       .Lss_fk2
.Lss_fst:
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s8, t2
    fsd     ft0, 0(t3)
    addi    t1, t1, 1
    j       .Lss_fj
.Lss_fni:
    addi    t0, t0, 1
    j       .Lss_fi
.Lss_fid:

    # back sub
    addi    t0, s3, -1
.Lss_bi:
    bltz    t0, .Lss_bid
    li      t1, 0
.Lss_bj:
    bge     t1, s4, .Lss_bni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s8, t2
    fld     ft0, 0(t3)
    addi    t4, t0, 1
.Lss_bk:
    bge     t4, s3, .Lss_bdv
    mul     t5, t0, s3
    add     t5, t5, t4
    slli    t5, t5, 3
    add     t6, s5, t5
    fld     ft1, 0(t6)
    mul     a6, t4, s4
    add     a6, a6, t1
    slli    a6, a6, 3
    add     a7, s2, a6
    fld     ft2, 0(a7)
    fnmsub.d ft0, ft1, ft2, ft0
    addi    t4, t4, 1
    j       .Lss_bk
.Lss_bdv:
    mul     t5, t0, s3
    add     t5, t5, t0
    slli    t5, t5, 3
    add     t6, s5, t5
    fld     ft1, 0(t6)
    fdiv.d  ft0, ft0, ft1
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s2, t2
    fsd     ft0, 0(t3)
    addi    t1, t1, 1
    j       .Lss_bj
.Lss_bni:
    addi    t0, t0, -1
    j       .Lss_bi
.Lss_bid:
    mv      a0, s5
    call    free
    mv      a0, s6
    call    free
    mv      a0, s7
    call    free
    mv      a0, s8
    call    free
    ld      ra,   0(sp)
    ld      s0,   8(sp)
    ld      s1,  16(sp)
    ld      s2,  24(sp)
    ld      s3,  32(sp)
    ld      s4,  40(sp)
    ld      s5,  48(sp)
    ld      s6,  56(sp)
    ld      s7,  64(sp)
    ld      s8,  72(sp)
    ld      s9,  80(sp)
    ld      s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 112
    ret

# =================== EKF-SPECIFIC FUNCTIONS ==============================

# --- approx_atan(z) -> fa0 ---
    .globl approx_atan
approx_atan:
    fsgnjx.d ft0, fa0, fa0
    la      t0, c_pio4
    fld     ft1, 0(t0)
    la      t0, c_a1
    fld     ft2, 0(t0)
    la      t0, c_a2
    fld     ft3, 0(t0)
    la      t0, c_one
    fld     ft4, 0(t0)
    fmul.d  ft5, ft1, fa0
    fsub.d  ft6, ft0, ft4
    fmul.d  ft7, ft3, ft0
    fadd.d  ft7, ft2, ft7
    fmul.d  ft6, fa0, ft6
    fmul.d  ft6, ft6, ft7
    fsub.d  fa0, ft5, ft6
    ret

# --- approx_atan2(y, x) -> fa0 ---
#   Uses fs0-fs3 callee-saved to survive inner call
    .globl approx_atan2
approx_atan2:
    addi    sp, sp, -40
    sd      ra,  0(sp)
    fsd     fs0, 8(sp)
    fsd     fs1, 16(sp)
    fsd     fs2, 24(sp)
    fsd     fs3, 32(sp)
    fmv.d   fs0, fa0
    fmv.d   fs1, fa1
    la      t0, c_pi
    fld     fs2, 0(t0)
    la      t0, c_halfpi
    fld     fs3, 0(t0)
    fcvt.d.w ft0, zero

    feq.d   t0, fs1, ft0
    feq.d   t1, fs0, ft0
    and     t2, t0, t1
    bnez    t2, .La2_zero
    bnez    t0, .La2_xz

    fsgnjx.d ft1, fs1, fs1
    fsgnjx.d ft2, fs0, fs0
    flt.d   t0, ft1, ft2
    bnez    t0, .La2_yb

    fdiv.d  fa0, fs0, fs1
    call    approx_atan
    fcvt.d.w ft0, zero
    flt.d   t0, fs1, ft0
    beqz    t0, .La2_ret
    fle.d   t1, ft0, fs0
    beqz    t1, .La2_sp
    fadd.d  fa0, fa0, fs2
    j       .La2_ret
.La2_sp:
    fsub.d  fa0, fa0, fs2
    j       .La2_ret

.La2_yb:
    fdiv.d  fa0, fs1, fs0
    call    approx_atan
    fsub.d  fa0, fs3, fa0
    fcvt.d.w ft0, zero
    flt.d   t0, fs0, ft0
    beqz    t0, .La2_ybx
    fsgnjn.d fa0, fa0, fa0
.La2_ybx:
    flt.d   t0, fs1, ft0
    beqz    t0, .La2_ret
    fle.d   t1, ft0, fs0
    beqz    t1, .La2_ybs
    fadd.d  fa0, fa0, fs2
    j       .La2_ret
.La2_ybs:
    fsub.d  fa0, fa0, fs2
    j       .La2_ret

.La2_xz:
    fcvt.d.w ft0, zero
    flt.d   t0, ft0, fs0
    bnez    t0, .La2_ph
    fsgnjn.d fa0, fs3, fs3
    j       .La2_ret
.La2_ph:
    fmv.d   fa0, fs3
    j       .La2_ret
.La2_zero:
    fcvt.d.w fa0, zero
.La2_ret:
    ld      ra,  0(sp)
    fld     fs0, 8(sp)
    fld     fs1, 16(sp)
    fld     fs2, 24(sp)
    fld     fs3, 32(sp)
    addi    sp, sp, 40
    ret

# --- state_to_spherical(x_state, out_sph) ---
#   Converts state vector positions to (r, theta, phi) per joint
    .globl state_to_spherical
state_to_spherical:
    addi    sp, sp, -64
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    fsd     fs0, 32(sp)
    fsd     fs1, 40(sp)
    fsd     fs2, 48(sp)
    mv      s0, a0
    mv      s1, a1
    li      s2, 0
.Ls2s_lp:
    li      t0, JOINTS
    bge     s2, t0, .Ls2s_dn
    li      t0, 12
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s0, t1
    fld     fs0, 0(t2)
    fld     fs1, 32(t2)
    fld     fs2, 64(t2)
    fmul.d  ft0, fs0, fs0
    fmul.d  ft1, fs1, fs1
    fmul.d  ft2, fs2, fs2
    fadd.d  ft0, ft0, ft1
    fadd.d  ft0, ft0, ft2
    fsqrt.d ft3, ft0
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     ft3, 0(t2)
    fmv.d   fa0, fs1
    fmv.d   fa1, fs0
    call    approx_atan2
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     fa0, 8(t2)
    fmul.d  ft0, fs0, fs0
    fmul.d  ft1, fs1, fs1
    fadd.d  ft0, ft0, ft1
    fsqrt.d ft3, ft0
    fmv.d   fa0, fs2
    fmv.d   fa1, ft3
    call    approx_atan2
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     fa0, 16(t2)
    addi    s2, s2, 1
    j       .Ls2s_lp
.Ls2s_dn:
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    fld     fs0, 32(sp)
    fld     fs1, 40(sp)
    fld     fs2, 48(sp)
    addi    sp, sp, 64
    ret

# --- meas_to_spherical(cart_row, out_sph) ---
#   Converts flat measurement (69 Cartesian) to spherical
    .globl meas_to_spherical
meas_to_spherical:
    addi    sp, sp, -64
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    fsd     fs0, 32(sp)
    fsd     fs1, 40(sp)
    fsd     fs2, 48(sp)
    mv      s0, a0
    mv      s1, a1
    li      s2, 0
.Lm2s_lp:
    li      t0, JOINTS
    bge     s2, t0, .Lm2s_dn
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s0, t1
    fld     fs0, 0(t2)
    fld     fs1, 8(t2)
    fld     fs2, 16(t2)
    fmul.d  ft0, fs0, fs0
    fmul.d  ft1, fs1, fs1
    fmul.d  ft2, fs2, fs2
    fadd.d  ft0, ft0, ft1
    fadd.d  ft0, ft0, ft2
    fsqrt.d ft3, ft0
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     ft3, 0(t2)
    fmv.d   fa0, fs1
    fmv.d   fa1, fs0
    call    approx_atan2
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     fa0, 8(t2)
    fmul.d  ft0, fs0, fs0
    fmul.d  ft1, fs1, fs1
    fadd.d  ft0, ft0, ft1
    fsqrt.d ft3, ft0
    fmv.d   fa0, fs2
    fmv.d   fa1, ft3
    call    approx_atan2
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     fa0, 16(t2)
    addi    s2, s2, 1
    j       .Lm2s_lp
.Lm2s_dn:
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    fld     fs0, 32(sp)
    fld     fs1, 40(sp)
    fld     fs2, 48(sp)
    addi    sp, sp, 64
    ret

# --- compute_jac(x_state, jac_out) ---
#   Computes 69x276 Jacobian matrix
    .globl compute_jac
compute_jac:
    addi    sp, sp, -48
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    mv      s0, a0
    mv      s1, a1
    mv      a0, s1
    li      a1, CROSS_DIM
    call    clear_doubles
    li      s2, 0
.Lcj_lp:
    li      t0, JOINTS
    bge     s2, t0, .Lcj_dn
    li      t0, 12
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s0, t1
    fld     ft0, 0(t2)
    fld     ft1, 32(t2)
    fld     ft2, 64(t2)
    fmul.d  ft3, ft0, ft0
    fmul.d  ft4, ft1, ft1
    fmul.d  ft5, ft2, ft2
    fadd.d  ft6, ft3, ft4
    fadd.d  ft7, ft6, ft5
    fsqrt.d ft8, ft7
    fsqrt.d ft9, ft6
    la      t0, c_epsilon
    fld     ft10, 0(t0)
    flt.d   t0, ft8, ft10
    bnez    t0, .Lcj_nx
    flt.d   t0, ft9, ft10
    bnez    t0, .Lcj_nx

    li      t0, 3
    mul     s3, s2, t0
    li      t0, 12
    mul     s4, s2, t0

    # dr/dpx
    fdiv.d  ft3, ft0, ft8
    li      a0, STATE_DIM
    mul     a1, s3, a0
    add     a1, a1, s4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)
    # dr/dpy
    fdiv.d  ft3, ft1, ft8
    li      a0, STATE_DIM
    mul     a1, s3, a0
    addi    a3, s4, 4
    add     a1, a1, a3
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)
    # dr/dpz
    fdiv.d  ft3, ft2, ft8
    li      a0, STATE_DIM
    mul     a1, s3, a0
    addi    a3, s4, 8
    add     a1, a1, a3
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)
    # dtheta/dpx = -py/rho2
    fsgnjn.d ft3, ft1, ft1
    fdiv.d  ft3, ft3, ft6
    addi    a3, s3, 1
    li      a0, STATE_DIM
    mul     a1, a3, a0
    add     a1, a1, s4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)
    # dtheta/dpy = px/rho2
    fdiv.d  ft3, ft0, ft6
    addi    a3, s3, 1
    li      a0, STATE_DIM
    mul     a1, a3, a0
    addi    a4, s4, 4
    add     a1, a1, a4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)
    # dphi/dpx = -(px*pz)/(r2*rho)
    fmul.d  ft3, ft0, ft2
    fsgnjn.d ft3, ft3, ft3
    fmul.d  ft4, ft7, ft9
    fdiv.d  ft3, ft3, ft4
    addi    a3, s3, 2
    li      a0, STATE_DIM
    mul     a1, a3, a0
    add     a1, a1, s4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)
    # dphi/dpy = -(py*pz)/(r2*rho)
    fmul.d  ft3, ft1, ft2
    fsgnjn.d ft3, ft3, ft3
    fmul.d  ft4, ft7, ft9
    fdiv.d  ft3, ft3, ft4
    addi    a3, s3, 2
    li      a0, STATE_DIM
    mul     a1, a3, a0
    addi    a4, s4, 4
    add     a1, a1, a4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)
    # dphi/dpz = rho/r2
    fdiv.d  ft3, ft9, ft7
    addi    a3, s3, 2
    li      a0, STATE_DIM
    mul     a1, a3, a0
    addi    a4, s4, 8
    add     a1, a1, a4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)
.Lcj_nx:
    addi    s2, s2, 1
    j       .Lcj_lp
.Lcj_dn:
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    addi    sp, sp, 48
    ret

# ========================= MAIN =========================================
    .globl main
main:
    addi    sp, sp, -64
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)

    la      a0, str_load
    call    printf
    la      a0, path_input
    addi    a1, sp, 48
    addi    a2, sp, 52
    call    load_csv
    mv      s0, a0
    lw      s1, 48(sp)
    lw      s2, 52(sp)
    la      a0, str_info
    mv      a1, s1
    mv      a2, s2
    call    printf

    li      a0, HANDLE_COUNT*8
    call    malloc
    mv      s3, a0

    la      a0, str_init
    call    printf

    # --- allocate all matrices ---
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_F(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_Q(s3)
    li a0, SQ_MEAS*8
    call malloc
    sd a0, H_R(s3)
    li a0, STATE_DIM*8
    call malloc
    sd a0, H_st(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_cv(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_Ft(s3)
    li a0, STATE_DIM*8
    call malloc
    sd a0, H_stp(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_cvp(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_fp(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_fpft(s3)
    li a0, SQ_MEAS*8
    call malloc
    sd a0, H_S(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_jac(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_jacT(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_jp(s3)
    li a0, SQ_MEAS*8
    call malloc
    sd a0, H_jpjt(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_pjt(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_pjtt(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_ksol(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_K(s3)
    li a0, MEAS_DIM*8
    call malloc
    sd a0, H_zsph(s3)
    li a0, MEAS_DIM*8
    call malloc
    sd a0, H_hsph(s3)
    li a0, MEAS_DIM*8
    call malloc
    sd a0, H_inn(s3)
    li a0, STATE_DIM*8
    call malloc
    sd a0, H_kinn(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_kj(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_ikj(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_ikjt(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_ikjp(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_ikjpt(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_kr(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_krkt(s3)
    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_Kt(s3)
    li a0, SQ_STATE*8
    call malloc
    sd a0, H_eye(s3)

    # === BUILD F (same as LKF) ===
    ld a0, H_F(s3)
    li a1, SQ_STATE
    call clear_doubles
    ld s4, H_F(s3)
    li t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0
    fmul.d ft1, ft0, ft0
    li t0, 0x3FE0000000000000
    fmv.d.x ft8, t0
    fmul.d ft2, ft8, ft1
    fmul.d ft3, ft1, ft0
    li t0, 0x3FC5555555555555
    fmv.d.x ft9, t0
    fmul.d ft4, ft9, ft3
    li t0, 0x3FF0000000000000
    fmv.d.x ft5, t0
    li t0, 0
.Lef_jt:
    li t1, JOINTS
    bge t0, t1, .Lef_dn
    li t2, 0
.Lef_ax:
    li t3, 3
    bge t2, t3, .Lef_njt
    li t4, 12
    mul t4, t0, t4
    slli t5, t2, 2
    add t4, t4, t5
    li a1, STATE_DIM
    mul a2, t4, a1
    add a3, a2, t4
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft5, 0(a4)
    addi a3, t4, 1
    add a3, a2, a3
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft0, 0(a4)
    addi a3, t4, 2
    add a3, a2, a3
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft2, 0(a4)
    addi a3, t4, 3
    add a3, a2, a3
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft4, 0(a4)
    addi a5, t4, 1
    mul a2, a5, a1
    add a3, a2, a5
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft5, 0(a4)
    addi a3, a5, 1
    add a3, a2, a3
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft0, 0(a4)
    addi a3, a5, 2
    add a3, a2, a3
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft2, 0(a4)
    addi a5, t4, 2
    mul a2, a5, a1
    add a3, a2, a5
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft5, 0(a4)
    addi a3, a5, 1
    add a3, a2, a3
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft0, 0(a4)
    addi a5, t4, 3
    mul a2, a5, a1
    add a3, a2, a5
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft5, 0(a4)
    addi t2, t2, 1
    j .Lef_ax
.Lef_njt:
    addi t0, t0, 1
    j .Lef_jt
.Lef_dn:

    # === BUILD Q ===
    ld a0, H_Q(s3)
    li a1, SQ_STATE
    call clear_doubles
    li a0, 32
    call malloc
    mv s4, a0
    li t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0
    fmul.d ft1, ft0, ft0
    li t0, 0x3FE0000000000000
    fmv.d.x ft8, t0
    fmul.d ft2, ft8, ft1
    fmul.d ft3, ft1, ft0
    li t0, 0x3FC5555555555555
    fmv.d.x ft9, t0
    fmul.d ft4, ft9, ft3
    li t0, 0x3FF0000000000000
    fmv.d.x ft5, t0
    fsd ft4, 0(s4)
    fsd ft2, 8(s4)
    fsd ft0, 16(s4)
    fsd ft5, 24(s4)
    ld a5, H_Q(s3)
    li t0, 0
.Leq_jt:
    li t1, JOINTS
    bge t0, t1, .Leq_dn
    li t2, 0
.Leq_ax:
    li t3, 3
    bge t2, t3, .Leq_njt
    li t4, 0
.Leq_i:
    li t5, 4
    bge t4, t5, .Leq_nax
    li t6, 0
.Leq_j:
    li a1, 4
    bge t6, a1, .Leq_ni
    slli a2, t4, 3
    add a3, s4, a2
    fld ft0, 0(a3)
    slli a2, t6, 3
    add a3, s4, a2
    fld ft1, 0(a3)
    fmul.d ft2, ft0, ft1
    li a1, 12
    mul a2, t0, a1
    slli a3, t2, 2
    add a2, a2, a3
    add a2, a2, t4
    mul a3, t0, a1
    slli a4, t2, 2
    add a3, a3, a4
    add a3, a3, t6
    li a1, STATE_DIM
    mul a4, a2, a1
    add a4, a4, a3
    slli a4, a4, 3
    add a4, a5, a4
    fsd ft2, 0(a4)
    addi t6, t6, 1
    j .Leq_j
.Leq_ni:
    addi t4, t4, 1
    j .Leq_i
.Leq_nax:
    addi t2, t2, 1
    j .Leq_ax
.Leq_njt:
    addi t0, t0, 1
    j .Leq_jt
.Leq_dn:
    mv a0, s4
    call free

    # === BUILD R (0.25, 0.01, 0.01 per joint) ===
    ld a0, H_R(s3)
    li a1, SQ_MEAS
    call clear_doubles
    ld s4, H_R(s3)
    li t0, 0x3FD0000000000000
    fmv.d.x ft0, t0
    li t0, 0x3F847AE147AE147B
    fmv.d.x ft1, t0
    li t1, 0
.Ler_lp:
    li t2, JOINTS
    bge t1, t2, .Ler_dn
    li a1, 3
    mul a2, t1, a1
    li a1, MEAS_DIM
    mul a3, a2, a1
    add a3, a3, a2
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft0, 0(a4)
    addi a2, a2, 1
    mul a3, a2, a1
    add a3, a3, a2
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft1, 0(a4)
    addi a2, a2, 1
    mul a3, a2, a1
    add a3, a3, a2
    slli a3, a3, 3
    add a4, s4, a3
    fsd ft1, 0(a4)
    addi t1, t1, 1
    j .Ler_lp
.Ler_dn:

    # === BUILD I ===
    ld a0, H_eye(s3)
    li a1, SQ_STATE
    call clear_doubles
    ld s4, H_eye(s3)
    li t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li t1, 0
.Lei_lp:
    li t2, STATE_DIM
    bge t1, t2, .Lei_dn
    li a1, STATE_DIM
    mul a2, t1, a1
    add a2, a2, t1
    slli a2, a2, 3
    add a3, s4, a2
    fsd ft0, 0(a3)
    addi t1, t1, 1
    j .Lei_lp
.Lei_dn:

    # === INIT STATE ===
    ld a0, H_st(s3)
    li a1, STATE_DIM
    call clear_doubles
    ld s4, H_st(s3)
    li t0, 0
.Lex_lp:
    li t1, JOINTS
    bge t0, t1, .Lex_dn
    li a1, 3
    mul a2, t0, a1
    slli a2, a2, 3
    add a3, s0, a2
    fld ft0, 0(a3)
    fld ft1, 8(a3)
    fld ft2, 16(a3)
    li a1, 12
    mul a2, t0, a1
    slli a2, a2, 3
    add a3, s4, a2
    fsd ft0, 0(a3)
    fsd ft1, 32(a3)
    fsd ft2, 64(a3)
    addi t0, t0, 1
    j .Lex_lp
.Lex_dn:

    # === INIT COV = I ===
    ld a0, H_cv(s3)
    li a1, SQ_STATE
    call clear_doubles
    ld s4, H_cv(s3)
    li t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li t1, 0
.Lec_lp:
    li t2, STATE_DIM
    bge t1, t2, .Lec_dn
    li a1, STATE_DIM
    mul a2, t1, a1
    add a2, a2, t1
    slli a2, a2, 3
    add a3, s4, a2
    fsd ft0, 0(a3)
    addi t1, t1, 1
    j .Lec_lp
.Lec_dn:

    # === Ft = F^T ===
    ld a0, H_F(s3)
    ld a1, H_Ft(s3)
    li a2, STATE_DIM
    li a3, STATE_DIM
    call transpose_mat

    # === OPEN OUTPUT ===
    la a0, path_output
    call csv_open
    sd a0, H_fout(s3)

    la a0, str_start
    call printf

    # ===================== FILTER LOOP ===================================
    li s4, 0
.Lefl_lp:
    bge s4, s1, .Lefl_dn

    # predict state
    ld a0, H_F(s3)
    ld a1, H_st(s3)
    ld a2, H_stp(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, 1
    call multiply_mat

    # predict covariance
    ld a0, H_F(s3)
    ld a1, H_cv(s3)
    ld a2, H_fp(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, STATE_DIM
    call multiply_mat

    ld a0, H_fp(s3)
    ld a1, H_Ft(s3)
    ld a2, H_fpft(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, STATE_DIM
    call multiply_mat

    ld a0, H_fpft(s3)
    ld a1, H_Q(s3)
    ld a2, H_cvp(s3)
    li a3, SQ_STATE
    call vec_add

    # measurement -> spherical
    mv a0, s4
    mv a1, s2
    mul a2, a0, a1
    slli a2, a2, 3
    add a0, s0, a2
    ld a1, H_zsph(s3)
    call meas_to_spherical

    # Jacobian at predicted state
    ld a0, H_stp(s3)
    ld a1, H_jac(s3)
    call compute_jac

    # JacT
    ld a0, H_jac(s3)
    ld a1, H_jacT(s3)
    li a2, MEAS_DIM
    li a3, STATE_DIM
    call transpose_mat

    # JP = Jac * Pp
    ld a0, H_jac(s3)
    ld a1, H_cvp(s3)
    ld a2, H_jp(s3)
    li a3, MEAS_DIM
    li a4, STATE_DIM
    li a5, STATE_DIM
    call multiply_mat

    # JPJt
    ld a0, H_jp(s3)
    ld a1, H_jacT(s3)
    ld a2, H_jpjt(s3)
    li a3, MEAS_DIM
    li a4, STATE_DIM
    li a5, MEAS_DIM
    call multiply_mat

    # S = JPJt + R
    ld a0, H_jpjt(s3)
    ld a1, H_R(s3)
    ld a2, H_S(s3)
    li a3, SQ_MEAS
    call vec_add

    # PJt = Pp * JacT
    ld a0, H_cvp(s3)
    ld a1, H_jacT(s3)
    ld a2, H_pjt(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, MEAS_DIM
    call multiply_mat

    # PJt^T
    ld a0, H_pjt(s3)
    ld a1, H_pjtt(s3)
    li a2, STATE_DIM
    li a3, MEAS_DIM
    call transpose_mat

    # solve S * Ksol = PJt^T
    ld a0, H_S(s3)
    ld a1, H_pjtt(s3)
    ld a2, H_ksol(s3)
    li a3, MEAS_DIM
    li a4, STATE_DIM
    call solve_system

    # K = Ksol^T
    ld a0, H_ksol(s3)
    ld a1, H_K(s3)
    li a2, MEAS_DIM
    li a3, STATE_DIM
    call transpose_mat

    # h(x_pred)
    ld a0, H_stp(s3)
    ld a1, H_hsph(s3)
    call state_to_spherical

    # innovation = z_sph - h(x_pred)
    ld a0, H_zsph(s3)
    ld a1, H_hsph(s3)
    ld a2, H_inn(s3)
    li a3, MEAS_DIM
    call vec_sub

    # K * innovation
    ld a0, H_K(s3)
    ld a1, H_inn(s3)
    ld a2, H_kinn(s3)
    li a3, STATE_DIM
    li a4, MEAS_DIM
    li a5, 1
    call multiply_mat

    # state = pred + K*innov
    ld a0, H_stp(s3)
    ld a1, H_kinn(s3)
    ld a2, H_st(s3)
    li a3, STATE_DIM
    call vec_add

    # KJ = K * Jac
    ld a0, H_K(s3)
    ld a1, H_jac(s3)
    ld a2, H_kj(s3)
    li a3, STATE_DIM
    li a4, MEAS_DIM
    li a5, STATE_DIM
    call multiply_mat

    # IKJ = I - KJ
    ld a0, H_eye(s3)
    ld a1, H_kj(s3)
    ld a2, H_ikj(s3)
    li a3, SQ_STATE
    call vec_sub

    # IKJt
    ld a0, H_ikj(s3)
    ld a1, H_ikjt(s3)
    li a2, STATE_DIM
    li a3, STATE_DIM
    call transpose_mat

    # IKJP = IKJ * Pp
    ld a0, H_ikj(s3)
    ld a1, H_cvp(s3)
    ld a2, H_ikjp(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, STATE_DIM
    call multiply_mat

    # IKJPt = IKJP * IKJt
    ld a0, H_ikjp(s3)
    ld a1, H_ikjt(s3)
    ld a2, H_ikjpt(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, STATE_DIM
    call multiply_mat

    # KR = K * R
    ld a0, H_K(s3)
    ld a1, H_R(s3)
    ld a2, H_kr(s3)
    li a3, STATE_DIM
    li a4, MEAS_DIM
    li a5, MEAS_DIM
    call multiply_mat

    # Kt
    ld a0, H_K(s3)
    ld a1, H_Kt(s3)
    li a2, STATE_DIM
    li a3, MEAS_DIM
    call transpose_mat

    # KRKt
    ld a0, H_kr(s3)
    ld a1, H_Kt(s3)
    ld a2, H_krkt(s3)
    li a3, STATE_DIM
    li a4, MEAS_DIM
    li a5, STATE_DIM
    call multiply_mat

    # P = IKJPt + KRKt
    ld a0, H_ikjpt(s3)
    ld a1, H_krkt(s3)
    ld a2, H_cv(s3)
    li a3, SQ_STATE
    call vec_add

    # write output
    ld a0, H_fout(s3)
    ld a1, H_st(s3)
    li a2, STATE_DIM
    call csv_write_state

    # progress
    li t0, 500
    rem t1, s4, t0
    bnez t1, .Lefl_np
    la a0, str_step
    mv a1, s4
    mv a2, s1
    call printf
.Lefl_np:
    addi s4, s4, 1
    j .Lefl_lp

.Lefl_dn:
    ld a0, H_fout(s3)
    call csv_close

    la a0, str_finish
    call printf

    li a0, 0
    call exit

