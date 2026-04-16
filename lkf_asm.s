# =========================================================================
#  Linear Kalman Filter - RISC-V RV64GD Assembly
#  | Team Warjongo | Milestone 3
#
#  Architecture:
#    s0 -> raw sensor data base address
#    s1 -> total frames count
#    s2 -> columns per frame
#    s3 -> matrix handle table base
#    s4 -> current frame index
#
#  Handle Table Layout (each entry is 8 bytes):
#    [0]=F [1]=Q [2]=H [3]=Ht [4]=R [5]=state [6]=cov [7]=Ft
#    [8]=state_p [9]=cov_p [10]=tmp_fp [11]=tmp_fpft [12]=innov_cov
#    [13]=tmp_hp [14]=tmp_hpht [15]=tmp_pht [16]=tmp_phtt [17]=gain_sol
#    [18]=gain [19]=meas [20]=pred_meas [21]=innov [22]=gain_innov
#    [23]=tmp_kh [24]=tmp_ikh [25]=tmp_ikht [26]=tmp_ikhp [27]=tmp_ikhpit
#    [28]=tmp_kr [29]=tmp_krkt [30]=gain_t [31]=eye [32]=output_file
# =========================================================================

# --- System dimensions ---
.equ JOINTS,          23
.equ DIM_PER_JOINT,   12
.equ MEAS_DIM_JOINT,  3
.equ STATE_DIM,       276
.equ MEAS_DIM,        69
.equ SQ_STATE,        76176
.equ SQ_MEAS,         4761
.equ CROSS_DIM,       19044
.equ HANDLE_COUNT,    33

# --- Handle table byte offsets ---
.equ H_F,        0
.equ H_Q,        8
.equ H_H,       16
.equ H_Ht,      24
.equ H_R,       32
.equ H_st,      40
.equ H_cv,      48
.equ H_Ft,      56
.equ H_stp,     64
.equ H_cvp,     72
.equ H_fp,      80
.equ H_fpft,    88
.equ H_S,       96
.equ H_hp,     104
.equ H_hpht,   112
.equ H_pht,    120
.equ H_phtt,   128
.equ H_ksol,   136
.equ H_K,      144
.equ H_z,      152
.equ H_hxp,    160
.equ H_inn,    168
.equ H_kinn,   176
.equ H_kh,     184
.equ H_ikh,    192
.equ H_ikht,   200
.equ H_ikhp,   208
.equ H_ikhpt,  216
.equ H_kr,     224
.equ H_krkt,   232
.equ H_Kt,     240
.equ H_eye,    248
.equ H_fout,   256

    .section .rodata
path_input:     .string "NoisyValues.csv"
path_output:    .string "LKF_asm_output.csv"
str_load:       .string "[LKF] Loading data...\n"
str_info:       .string "[LKF] Frames=%d Cols=%d\n"
str_init:       .string "[LKF] Initializing system...\n"
str_start:      .string "[LKF] Processing...\n"
str_step:       .string "[LKF] Frame %d/%d\n"
str_finish:     .string "[LKF] Complete.\n"

    .section .text

# ===================== HELPER: clear_doubles(ptr, count) =================
    .globl clear_doubles
clear_doubles:
    li      t0, 0
    fcvt.d.w ft0, zero
.Lcd_loop:
    bge     t0, a1, .Lcd_end
    slli    t1, t0, 3
    add     t2, a0, t1
    fsd     ft0, 0(t2)
    addi    t0, t0, 1
    j       .Lcd_loop
.Lcd_end:
    ret

# ===================== vec_add(A, B, C, len) =============================
    .globl vec_add
vec_add:
    li      t0, 0
.Lva_loop:
    bge     t0, a3, .Lva_end
    slli    t1, t0, 3
    add     t2, a0, t1
    add     t3, a1, t1
    add     t4, a2, t1
    fld     ft0, 0(t2)
    fld     ft1, 0(t3)
    fadd.d  ft2, ft0, ft1
    fsd     ft2, 0(t4)
    addi    t0, t0, 1
    j       .Lva_loop
.Lva_end:
    ret

# ===================== vec_sub(A, B, C, len) =============================
    .globl vec_sub
vec_sub:
    li      t0, 0
.Lvs_loop:
    bge     t0, a3, .Lvs_end
    slli    t1, t0, 3
    add     t2, a0, t1
    add     t3, a1, t1
    add     t4, a2, t1
    fld     ft0, 0(t2)
    fld     ft1, 0(t3)
    fsub.d  ft2, ft0, ft1
    fsd     ft2, 0(t4)
    addi    t0, t0, 1
    j       .Lvs_loop
.Lvs_end:
    ret

# ===================== transpose_mat(src, dst, nrow, ncol) ===============
    .globl transpose_mat
transpose_mat:
    li      t0, 0
.Ltm_row:
    bge     t0, a2, .Ltm_end
    li      t1, 0
.Ltm_col:
    bge     t1, a3, .Ltm_nrow
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
    j       .Ltm_col
.Ltm_nrow:
    addi    t0, t0, 1
    j       .Ltm_row
.Ltm_end:
    ret

# ===================== multiply_mat(A, B, C, rA, cA, cB) =================
# Uses fmadd.d and zero-skip optimization
    .globl multiply_mat
multiply_mat:
    addi    sp, sp, -48
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    mv      s0, a0          # src_a
    mv      s1, a1          # src_b
    mv      s2, a2          # dst_c
    mv      s3, a3          # num_rows
    mv      s4, a4          # inner_dim
    mv      s5, a5          # num_cols

    # clear output matrix
    mul     t0, s3, s5
    li      t1, 0
    fcvt.d.w ft3, zero
.Lmm_clr:
    bge     t1, t0, .Lmm_clr_done
    slli    t2, t1, 3
    add     t3, s2, t2
    fsd     ft3, 0(t3)
    addi    t1, t1, 1
    j       .Lmm_clr
.Lmm_clr_done:

    # row loop
    li      t0, 0
.Lmm_row:
    bge     t0, s3, .Lmm_finish
    li      t1, 0
.Lmm_mid:
    bge     t1, s4, .Lmm_nrow
    # fetch A[row][mid]
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s0, t2
    fld     ft0, 0(t3)
    # sparsity: skip if zero
    fmv.d.x ft7, zero
    feq.d   a6, ft0, ft7
    bnez    a6, .Lmm_nmid
    # precompute offsets
    mul     t4, t1, s5      # mid * num_cols
    mul     t5, t0, s5      # row * num_cols
    li      t6, 0
.Lmm_col:
    bge     t6, s5, .Lmm_nmid
    # B[mid][col]
    add     a6, t4, t6
    slli    a6, a6, 3
    add     a7, s1, a6
    fld     ft1, 0(a7)
    # C[row][col]
    add     a6, t5, t6
    slli    a6, a6, 3
    add     a7, s2, a6
    fld     ft2, 0(a7)
    fmadd.d ft2, ft0, ft1, ft2
    fsd     ft2, 0(a7)
    addi    t6, t6, 1
    j       .Lmm_col
.Lmm_nmid:
    addi    t1, t1, 1
    j       .Lmm_mid
.Lmm_nrow:
    addi    t0, t0, 1
    j       .Lmm_row
.Lmm_finish:
    ld      s0,  0(sp)
    ld      s1,  8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    addi    sp, sp, 48
    ret

# ===================== solve_system(A, B, X, n, m) =======================
# LU factorization with partial pivoting
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
    mv      s0, a0          # coeff matrix
    mv      s1, a1          # rhs matrix
    mv      s2, a2          # solution
    mv      s3, a3          # size
    mv      s4, a4          # rhs cols

    # workspace: LU copy
    mul     a0, s3, s3
    slli    a0, a0, 3
    call    malloc
    mv      s5, a0
    # copy coefficients
    mul     t0, s3, s3
    li      t1, 0
.Lss_cp:
    bge     t1, t0, .Lss_cp_done
    slli    t2, t1, 3
    add     t3, s0, t2
    add     t4, s5, t2
    fld     ft0, 0(t3)
    fsd     ft0, 0(t4)
    addi    t1, t1, 1
    j       .Lss_cp
.Lss_cp_done:

    # pivot indices
    slli    a0, s3, 2
    call    malloc
    mv      s6, a0
    li      t0, 0
.Lss_piv:
    bge     t0, s3, .Lss_piv_done
    slli    t1, t0, 2
    add     t2, s6, t1
    sw      t0, 0(t2)
    addi    t0, t0, 1
    j       .Lss_piv
.Lss_piv_done:

    # permuted rhs
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s7, a0

    # temp for forward sub
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s8, a0

    # === FACTORIZATION ===
    li      s9, 0
.Lss_fact:
    bge     s9, s3, .Lss_fact_done
    # find pivot
    mul     t0, s9, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)
    fsgnjx.d ft1, ft0, ft0
    mv      s10, s9
    addi    t0, s9, 1
.Lss_piv_search:
    bge     t0, s3, .Lss_piv_found
    mul     t1, t0, s3
    add     t1, t1, s9
    slli    t1, t1, 3
    add     t2, s5, t1
    fld     ft2, 0(t2)
    fsgnjx.d ft3, ft2, ft2
    fle.d   t3, ft3, ft1
    bnez    t3, .Lss_piv_next
    fmv.d   ft1, ft3
    mv      s10, t0
.Lss_piv_next:
    addi    t0, t0, 1
    j       .Lss_piv_search
.Lss_piv_found:

    # swap rows if needed
    beq     s10, s9, .Lss_no_swap
    slli    t0, s9, 2
    add     t1, s6, t0
    slli    t0, s10, 2
    add     t2, s6, t0
    lw      t3, 0(t1)
    lw      t4, 0(t2)
    sw      t4, 0(t1)
    sw      t3, 0(t2)
    li      t0, 0
.Lss_swap:
    bge     t0, s3, .Lss_no_swap
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
    j       .Lss_swap
.Lss_no_swap:

    # elimination
    addi    s11, s9, 1
.Lss_elim_i:
    bge     s11, s3, .Lss_elim_done
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
.Lss_elim_j:
    bge     t4, s3, .Lss_elim_ni
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
    j       .Lss_elim_j
.Lss_elim_ni:
    addi    s11, s11, 1
    j       .Lss_elim_i
.Lss_elim_done:
    addi    s9, s9, 1
    j       .Lss_fact
.Lss_fact_done:

    # === PERMUTE RHS ===
    li      t0, 0
.Lss_perm_i:
    bge     t0, s3, .Lss_perm_done
    slli    t1, t0, 2
    add     t2, s6, t1
    lw      t3, 0(t2)
    li      t4, 0
.Lss_perm_j:
    bge     t4, s4, .Lss_perm_ni
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
    j       .Lss_perm_j
.Lss_perm_ni:
    addi    t0, t0, 1
    j       .Lss_perm_i
.Lss_perm_done:

    # === FORWARD SUBSTITUTION ===
    li      t0, 0
.Lss_fwd_i:
    bge     t0, s3, .Lss_fwd_done
    li      t1, 0
.Lss_fwd_j:
    bge     t1, s4, .Lss_fwd_ni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s7, t2
    fld     ft0, 0(t3)
    li      t4, 0
.Lss_fwd_k:
    bge     t4, t0, .Lss_fwd_st
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
    j       .Lss_fwd_k
.Lss_fwd_st:
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s8, t2
    fsd     ft0, 0(t3)
    addi    t1, t1, 1
    j       .Lss_fwd_j
.Lss_fwd_ni:
    addi    t0, t0, 1
    j       .Lss_fwd_i
.Lss_fwd_done:

    # === BACK SUBSTITUTION ===
    addi    t0, s3, -1
.Lss_bk_i:
    bltz    t0, .Lss_bk_done
    li      t1, 0
.Lss_bk_j:
    bge     t1, s4, .Lss_bk_ni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s8, t2
    fld     ft0, 0(t3)
    addi    t4, t0, 1
.Lss_bk_k:
    bge     t4, s3, .Lss_bk_div
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
    j       .Lss_bk_k
.Lss_bk_div:
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
    j       .Lss_bk_j
.Lss_bk_ni:
    addi    t0, t0, -1
    j       .Lss_bk_i
.Lss_bk_done:

    # free workspace
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

    # load dataset
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

    # allocate handle table
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

    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_H(s3)

    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_Ht(s3)

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
    sd a0, H_hp(s3)

    li a0, SQ_MEAS*8
    call malloc
    sd a0, H_hpht(s3)

    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_pht(s3)

    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_phtt(s3)

    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_ksol(s3)

    li a0, CROSS_DIM*8
    call malloc
    sd a0, H_K(s3)

    li a0, MEAS_DIM*8
    call malloc
    sd a0, H_z(s3)

    li a0, MEAS_DIM*8
    call malloc
    sd a0, H_hxp(s3)

    li a0, MEAS_DIM*8
    call malloc
    sd a0, H_inn(s3)

    li a0, STATE_DIM*8
    call malloc
    sd a0, H_kinn(s3)

    li a0, SQ_STATE*8
    call malloc
    sd a0, H_kh(s3)

    li a0, SQ_STATE*8
    call malloc
    sd a0, H_ikh(s3)

    li a0, SQ_STATE*8
    call malloc
    sd a0, H_ikht(s3)

    li a0, SQ_STATE*8
    call malloc
    sd a0, H_ikhp(s3)

    li a0, SQ_STATE*8
    call malloc
    sd a0, H_ikhpt(s3)

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

    # === CONSTRUCT F ===
    ld      a0, H_F(s3)
    li      a1, SQ_STATE
    call    clear_doubles
    ld      s4, H_F(s3)

    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft0, t0             # dt=0.01
    fmul.d  ft1, ft0, ft0       # dt^2
    li      t0, 0x3FE0000000000000
    fmv.d.x ft8, t0             # 0.5
    fmul.d  ft2, ft8, ft1       # dt^2/2
    fmul.d  ft3, ft1, ft0       # dt^3
    li      t0, 0x3FC5555555555555
    fmv.d.x ft9, t0             # 1/6
    fmul.d  ft4, ft9, ft3       # dt^3/6
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0             # 1.0

    li      t0, 0
.Lbf_jt:
    li      t1, JOINTS
    bge     t0, t1, .Lbf_end
    li      t2, 0
.Lbf_ax:
    li      t3, 3
    bge     t2, t3, .Lbf_njt
    li      t4, 12
    mul     t4, t0, t4
    slli    t5, t2, 2
    add     t4, t4, t5

    li      a1, STATE_DIM
    mul     a2, t4, a1
    # row 0
    add     a3, a2, t4
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)
    addi    a3, t4, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)
    addi    a3, t4, 2
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft2, 0(a4)
    addi    a3, t4, 3
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft4, 0(a4)
    # row 1
    addi    a5, t4, 1
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)
    addi    a3, a5, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)
    addi    a3, a5, 2
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft2, 0(a4)
    # row 2
    addi    a5, t4, 2
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)
    addi    a3, a5, 1
    add     a3, a2, a3
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)
    # row 3
    addi    a5, t4, 3
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)

    addi    t2, t2, 1
    j       .Lbf_ax
.Lbf_njt:
    addi    t0, t0, 1
    j       .Lbf_jt
.Lbf_end:

    # === CONSTRUCT Q ===
    ld      a0, H_Q(s3)
    li      a1, SQ_STATE
    call    clear_doubles

    li      a0, 32
    call    malloc
    mv      s4, a0

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
    fsd     ft4,  0(s4)
    fsd     ft2,  8(s4)
    fsd     ft0, 16(s4)
    fsd     ft5, 24(s4)

    ld      a5, H_Q(s3)
    li      t0, 0
.Lbq_jt:
    li      t1, JOINTS
    bge     t0, t1, .Lbq_end
    li      t2, 0
.Lbq_ax:
    li      t3, 3
    bge     t2, t3, .Lbq_njt
    li      t4, 0
.Lbq_i:
    li      t5, 4
    bge     t4, t5, .Lbq_nax
    li      t6, 0
.Lbq_j:
    li      a1, 4
    bge     t6, a1, .Lbq_ni
    slli    a2, t4, 3
    add     a3, s4, a2
    fld     ft0, 0(a3)
    slli    a2, t6, 3
    add     a3, s4, a2
    fld     ft1, 0(a3)
    fmul.d  ft2, ft0, ft1
    li      a1, 12
    mul     a2, t0, a1
    slli    a3, t2, 2
    add     a2, a2, a3
    add     a2, a2, t4
    mul     a3, t0, a1
    slli    a4, t2, 2
    add     a3, a3, a4
    add     a3, a3, t6
    li      a1, STATE_DIM
    mul     a4, a2, a1
    add     a4, a4, a3
    slli    a4, a4, 3
    add     a4, a5, a4
    fsd     ft2, 0(a4)
    addi    t6, t6, 1
    j       .Lbq_j
.Lbq_ni:
    addi    t4, t4, 1
    j       .Lbq_i
.Lbq_nax:
    addi    t2, t2, 1
    j       .Lbq_ax
.Lbq_njt:
    addi    t0, t0, 1
    j       .Lbq_jt
.Lbq_end:
    mv      a0, s4
    call    free

    # === CONSTRUCT H ===
    ld      a0, H_H(s3)
    li      a1, CROSS_DIM
    call    clear_doubles
    ld      s4, H_H(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft5, t0
    li      t0, 0
.Lbh_lp:
    li      t1, JOINTS
    bge     t0, t1, .Lbh_end
    li      a1, STATE_DIM
    li      t2, 3
    mul     t3, t0, t2
    mul     t4, t3, a1
    li      t2, 12
    mul     t5, t0, t2
    add     t6, t4, t5
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)
    addi    t6, t3, 1
    mul     t6, t6, a1
    addi    a2, t5, 4
    add     t6, t6, a2
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)
    addi    t6, t3, 2
    mul     t6, t6, a1
    addi    a2, t5, 8
    add     t6, t6, a2
    slli    t6, t6, 3
    add     a2, s4, t6
    fsd     ft5, 0(a2)
    addi    t0, t0, 1
    j       .Lbh_lp
.Lbh_end:

    # === CONSTRUCT R ===
    ld      a0, H_R(s3)
    li      a1, SQ_MEAS
    call    clear_doubles
    ld      s4, H_R(s3)
    li      t0, 0x3FD0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
.Lbr_lp:
    li      t2, MEAS_DIM
    bge     t1, t2, .Lbr_end
    li      a1, MEAS_DIM
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       .Lbr_lp
.Lbr_end:

    # === CONSTRUCT IDENTITY ===
    ld      a0, H_eye(s3)
    li      a1, SQ_STATE
    call    clear_doubles
    ld      s4, H_eye(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
.Lbi_lp:
    li      t2, STATE_DIM
    bge     t1, t2, .Lbi_end
    li      a1, STATE_DIM
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       .Lbi_lp
.Lbi_end:

    # === INIT STATE ===
    ld      a0, H_st(s3)
    li      a1, STATE_DIM
    call    clear_doubles
    ld      s4, H_st(s3)
    li      t0, 0
.Lis_lp:
    li      t1, JOINTS
    bge     t0, t1, .Lis_end
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
    fsd     ft0, 0(a3)
    fsd     ft1, 32(a3)
    fsd     ft2, 64(a3)
    addi    t0, t0, 1
    j       .Lis_lp
.Lis_end:

    # === INIT COVARIANCE = I ===
    ld      a0, H_cv(s3)
    li      a1, SQ_STATE
    call    clear_doubles
    ld      s4, H_cv(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
.Lic_lp:
    li      t2, STATE_DIM
    bge     t1, t2, .Lic_end
    li      a1, STATE_DIM
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       .Lic_lp
.Lic_end:

    # === PRECOMPUTE Ht, Ft ===
    ld      a0, H_H(s3)
    ld      a1, H_Ht(s3)
    li      a2, MEAS_DIM
    li      a3, STATE_DIM
    call    transpose_mat

    ld      a0, H_F(s3)
    ld      a1, H_Ft(s3)
    li      a2, STATE_DIM
    li      a3, STATE_DIM
    call    transpose_mat

    # === OPEN OUTPUT ===
    la      a0, path_output
    call    csv_open
    sd      a0, H_fout(s3)

    la      a0, str_start
    call    printf

    # ===================== FILTER LOOP ===================================
    li      s4, 0
.Lfl_loop:
    bge     s4, s1, .Lfl_end

    # -- predict state --
    ld a0, H_F(s3)
    ld a1, H_st(s3)
    ld a2, H_stp(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, 1
    call multiply_mat

    # -- predict covariance --
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

    # -- load measurement --
    mv      a0, s4
    mv      a1, s2
    mul     a2, a0, a1
    slli    a2, a2, 3
    add     a2, s0, a2
    ld      a3, H_z(s3)
    li      t0, 0
.Llz:
    li      t1, MEAS_DIM
    bge     t0, t1, .Llz_end
    slli    t2, t0, 3
    add     t3, a2, t2
    add     t4, a3, t2
    fld     ft0, 0(t3)
    fsd     ft0, 0(t4)
    addi    t0, t0, 1
    j       .Llz
.Llz_end:

    # -- innovation covariance S = H*Pp*Ht + R --
    ld a0, H_H(s3)
    ld a1, H_cvp(s3)
    ld a2, H_hp(s3)
    li a3, MEAS_DIM
    li a4, STATE_DIM
    li a5, STATE_DIM
    call multiply_mat

    ld a0, H_hp(s3)
    ld a1, H_Ht(s3)
    ld a2, H_hpht(s3)
    li a3, MEAS_DIM
    li a4, STATE_DIM
    li a5, MEAS_DIM
    call multiply_mat

    ld a0, H_hpht(s3)
    ld a1, H_R(s3)
    ld a2, H_S(s3)
    li a3, SQ_MEAS
    call vec_add

    # -- Kalman gain --
    ld a0, H_cvp(s3)
    ld a1, H_Ht(s3)
    ld a2, H_pht(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, MEAS_DIM
    call multiply_mat

    ld a0, H_pht(s3)
    ld a1, H_phtt(s3)
    li a2, STATE_DIM
    li a3, MEAS_DIM
    call transpose_mat

    ld a0, H_S(s3)
    ld a1, H_phtt(s3)
    ld a2, H_ksol(s3)
    li a3, MEAS_DIM
    li a4, STATE_DIM
    call solve_system

    ld a0, H_ksol(s3)
    ld a1, H_K(s3)
    li a2, MEAS_DIM
    li a3, STATE_DIM
    call transpose_mat

    # -- innovation --
    ld a0, H_H(s3)
    ld a1, H_stp(s3)
    ld a2, H_hxp(s3)
    li a3, MEAS_DIM
    li a4, STATE_DIM
    li a5, 1
    call multiply_mat

    ld a0, H_z(s3)
    ld a1, H_hxp(s3)
    ld a2, H_inn(s3)
    li a3, MEAS_DIM
    call vec_sub

    # -- state update --
    ld a0, H_K(s3)
    ld a1, H_inn(s3)
    ld a2, H_kinn(s3)
    li a3, STATE_DIM
    li a4, MEAS_DIM
    li a5, 1
    call multiply_mat

    ld a0, H_stp(s3)
    ld a1, H_kinn(s3)
    ld a2, H_st(s3)
    li a3, STATE_DIM
    call vec_add

    # -- covariance update (Joseph form) --
    ld a0, H_K(s3)
    ld a1, H_H(s3)
    ld a2, H_kh(s3)
    li a3, STATE_DIM
    li a4, MEAS_DIM
    li a5, STATE_DIM
    call multiply_mat

    ld a0, H_eye(s3)
    ld a1, H_kh(s3)
    ld a2, H_ikh(s3)
    li a3, SQ_STATE
    call vec_sub

    ld a0, H_ikh(s3)
    ld a1, H_ikht(s3)
    li a2, STATE_DIM
    li a3, STATE_DIM
    call transpose_mat

    ld a0, H_ikh(s3)
    ld a1, H_cvp(s3)
    ld a2, H_ikhp(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, STATE_DIM
    call multiply_mat

    ld a0, H_ikhp(s3)
    ld a1, H_ikht(s3)
    ld a2, H_ikhpt(s3)
    li a3, STATE_DIM
    li a4, STATE_DIM
    li a5, STATE_DIM
    call multiply_mat

    ld a0, H_K(s3)
    ld a1, H_R(s3)
    ld a2, H_kr(s3)
    li a3, STATE_DIM
    li a4, MEAS_DIM
    li a5, MEAS_DIM
    call multiply_mat

    ld a0, H_K(s3)
    ld a1, H_Kt(s3)
    li a2, STATE_DIM
    li a3, MEAS_DIM
    call transpose_mat

    ld a0, H_kr(s3)
    ld a1, H_Kt(s3)
    ld a2, H_krkt(s3)
    li a3, STATE_DIM
    li a4, MEAS_DIM
    li a5, STATE_DIM
    call multiply_mat

    ld a0, H_ikhpt(s3)
    ld a1, H_krkt(s3)
    ld a2, H_cv(s3)
    li a3, SQ_STATE
    call vec_add

    # -- write output --
    ld a0, H_fout(s3)
    ld a1, H_st(s3)
    li a2, STATE_DIM
    call csv_write_state

    # -- progress --
    li      t0, 500
    rem     t1, s4, t0
    bnez    t1, .Lfl_nop
    la      a0, str_step
    mv      a1, s4
    mv      a2, s1
    call    printf
.Lfl_nop:
    addi    s4, s4, 1
    j       .Lfl_loop

.Lfl_end:
    ld      a0, H_fout(s3)
    call    csv_close

    la      a0, str_finish
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

