

.equ NUM_JOINTS,       23
.equ STATE_PER_JOINT,  12
.equ MEAS_PER_JOINT,   3
.equ N,                276
.equ M,                69
.equ NN,               76176
.equ MM,               4761
.equ NM,               19044
.equ NUM_PTRS,         33

.equ PT_F,         0
.equ PT_Q,         8
.equ PT_R,        16
.equ PT_x,        24
.equ PT_P,        32
.equ PT_Ft,       40
.equ PT_xp,       48
.equ PT_Pp,       56
.equ PT_FP,       64
.equ PT_FPFt,     72
.equ PT_S,        80
.equ PT_Jac,      88
.equ PT_JacT,     96
.equ PT_JP,      104
.equ PT_JPJt,    112
.equ PT_PJt,     120
.equ PT_PJt_t,   128
.equ PT_Ktsol,   136
.equ PT_K,       144
.equ PT_zsph,    152
.equ PT_hx,      160
.equ PT_y,       168
.equ PT_Ky,      176
.equ PT_KJ,      184
.equ PT_IKJ,     192
.equ PT_IKJt,    200
.equ PT_IKJP,    208
.equ PT_IKJPIt,  216
.equ PT_KR,      224
.equ PT_KRKt,    232
.equ PT_Kt,      240
.equ PT_Imat,    248
.equ PT_FILE,    256

    .section .rodata
atan_pi4:       .double 0.7853981633974483
atan_c1:        .double 0.2447
atan_c2:        .double 0.0663
const_one:      .double 1.0
const_zero:     .double 0.0
const_pi:       .double 3.14159265358979323846
const_pi2:      .double 1.5707963267948966
const_half:     .double 0.5
const_sixth:    .double 0.16666666666666666
const_dt:       .double 0.01
const_r_var:    .double 0.25
const_a_var:    .double 0.01
const_eps:      .double 1.0e-9
csv_path:       .string "NoisyValues.csv"
out_path:       .string "EKF_vector_output.csv"
msg_reading:    .string "[EKF-VEC] Reading dataset...\n"
msg_ts:         .string "[EKF-VEC] Timesteps: %d, Cols: %d\n"
msg_building:   .string "[EKF-VEC] Building matrices...\n"
msg_running:    .string "[EKF-VEC] Running filter...\n"
msg_progress:   .string "[EKF-VEC] t=%d/%d\n"
msg_done:       .string "[EKF-VEC] Done. Output saved.\n"

    .section .text

#     
#                 UTILITY: zero_mem(ptr, count)
#     
    .globl zero_mem
zero_mem:
    # VECTORISED: fills 'count' doubles with 0.0
    fcvt.d.w ft0, zero
    li      t0, 0
zm_loop:
    bge     t0, a1, zm_done
    sub     t1, a1, t0
    vsetvli t2, t1, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    slli    t3, t0, 3
    add     t4, a0, t3
    vse64.v v0, (t4)
    add     t0, t0, t2
    j       zm_loop
zm_done:
    ret

#     
#                    MATH FUNCTIONS
#     

    .globl mat_add
mat_add:
    li      t0, 0
ma_loop:
    bge     t0, a3, ma_done
    sub     t1, a3, t0
    vsetvli t2, t1, e64, m1, ta, ma
    slli    t3, t0, 3
    add     t4, a0, t3
    add     t5, a1, t3
    add     t6, a2, t3
    vle64.v v0, (t4)
    vle64.v v1, (t5)
    vfadd.vv v2, v0, v1
    vse64.v v2, (t6)
    add     t0, t0, t2
    j       ma_loop
ma_done:
    ret

    .globl mat_sub
mat_sub:
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

    .globl mat_transpose
mat_transpose:
    slli    a4, a2, 3
    li      t0, 0
mt_o:
    bge     t0, a2, mt_d
    li      t1, 0
mt_i:
    bge     t1, a3, mt_nr
    sub     t2, a3, t1
    vsetvli t3, t2, e64, m1, ta, ma
    mul     t4, t0, a3
    add     t4, t4, t1
    slli    t4, t4, 3
    add     t5, a0, t4
    vle64.v v0, (t5)
    mul     t4, t1, a2
    add     t4, t4, t0
    slli    t4, t4, 3
    add     t5, a1, t4
    vsse64.v v0, (t5), a4
    add     t1, t1, t3
    j       mt_i
mt_nr:
    addi    t0, t0, 1
    j       mt_o
mt_d:
    ret

    .globl mat_mul
mat_mul:
    addi    sp, sp, -48
    sd      s0, 0(sp)
    sd      s1, 8(sp)
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
    # Zero C (vectorised)
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
    li      t0, 0
mmi:
    bge     t0, s3, mmd
    li      t1, 0
mmk:
    bge     t1, s4, mmni
    mul     t2, t0, s4
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t3, s0, t2
    fld     ft0, 0(t3)
    fmv.d.x ft7, zero
    feq.d   a6, ft0, ft7
    bnez    a6, mmnk
    mul     t4, t1, s5
    mul     t5, t0, s5
    li      t6, 0
mmj:
    bge     t6, s5, mmnk
    sub     a6, s5, t6
    vsetvli a7, a6, e64, m1, ta, ma
    add     a6, t4, t6
    slli    a6, a6, 3
    add     a6, s1, a6
    vle64.v v1, (a6)
    add     a6, t5, t6
    slli    a6, a6, 3
    add     a6, s2, a6
    vle64.v v2, (a6)
    vfmacc.vf v2, ft0, v1
    vse64.v v2, (a6)
    add     t6, t6, a7
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
    # Alloc LU
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
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s7, a0
    mul     a0, s3, s4
    slli    a0, a0, 3
    call    malloc
    mv      s8, a0
    # LU decomposition
    li      s9, 0
lkl:
    bge     s9, s3, lkd
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
    slli    t0, s9, 2
    add     t1, s6, t0
    slli    t0, s10, 2
    add     t2, s6, t0
    lw      t3, 0(t1)
    lw      t4, 0(t2)
    sw      t4, 0(t1)
    sw      t3, 0(t2)
    # Vectorised row swap
    li      t0, 0
lsl:
    bge     t0, s3, lns
    sub     t1, s3, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s3
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s5, t3
    mul     t3, s10, s3
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t5, s5, t3
    vle64.v v0, (t4)
    vle64.v v1, (t5)
    vse64.v v1, (t4)
    vse64.v v0, (t5)
    add     t0, t0, t2
    j       lsl
lns:
    addi    s11, s9, 1
lei:
    bge     s11, s3, led
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
    # Vectorised elimination
    addi    t4, s9, 1
lej:
    bge     t4, s3, leni
    sub     t5, s3, t4
    vsetvli t6, t5, e64, m1, ta, ma
    mul     a6, s11, s3
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s5, a6
    vle64.v v0, (a7)
    mul     a6, s9, s3
    add     a6, a6, t4
    slli    a6, a6, 3
    add     t5, s5, a6
    vle64.v v1, (t5)
    vfnmsac.vf v0, ft0, v1
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
    # Vectorised permute B rows
    li      t0, 0
lpi:
    bge     t0, s3, lpdd
    slli    t1, t0, 2
    add     t2, s6, t1
    lw      t3, 0(t2)
    li      t4, 0
lpj:
    bge     t4, s4, lpni
    sub     t5, s4, t4
    vsetvli t6, t5, e64, m1, ta, ma
    mul     a6, t3, s4
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s1, a6
    vle64.v v0, (a7)
    mul     a6, t0, s4
    add     a6, a6, t4
    slli    a6, a6, 3
    add     a7, s7, a6
    vse64.v v0, (a7)
    add     t4, t4, t6
    j       lpj
lpni:
    addi    t0, t0, 1
    j       lpi
lpdd:
    # Vectorised forward substitution
    li      s9, 0
lfi:
    bge     s9, s3, lfd
    li      t0, 0
lf_cp:
    bge     t0, s4, lf_cpd
    sub     t1, s4, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s4
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s7, t3
    add     t5, s8, t3
    vle64.v v0, (t4)
    vse64.v v0, (t5)
    add     t0, t0, t2
    j       lf_cp
lf_cpd:
    li      s10, 0
lfk:
    bge     s10, s9, lfni
    mul     t0, s9, s3
    add     t0, t0, s10
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)
    li      t0, 0
lfj_v:
    bge     t0, s4, lfk_nx
    sub     t1, s4, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s4
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s8, t3
    mul     t5, s10, s4
    add     t5, t5, t0
    slli    t5, t5, 3
    add     t6, s8, t5
    vle64.v v0, (t4)
    vle64.v v1, (t6)
    vfnmsac.vf v0, ft0, v1
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
    # Vectorised back substitution
    mul     t0, s3, s4
    li      t1, 0
lb_cp:
    bge     t1, t0, lb_cpd
    sub     t2, t0, t1
    vsetvli t3, t2, e64, m1, ta, ma
    slli    t4, t1, 3
    add     t5, s8, t4
    add     t6, s2, t4
    vle64.v v0, (t5)
    vse64.v v0, (t6)
    add     t1, t1, t3
    j       lb_cp
lb_cpd:
    addi    s9, s3, -1
lbi:
    bltz    s9, lbd
    addi    s10, s9, 1
lbk:
    bge     s10, s3, lb_div
    mul     t0, s9, s3
    add     t0, t0, s10
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft0, 0(t1)
    li      t0, 0
lbj_v:
    bge     t0, s4, lbk_nx
    sub     t1, s4, t0
    vsetvli t2, t1, e64, m1, ta, ma
    mul     t3, s9, s4
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t4, s2, t3
    mul     t5, s10, s4
    add     t5, t5, t0
    slli    t5, t5, 3
    add     t6, s2, t5
    vle64.v v0, (t4)
    vle64.v v1, (t6)
    vfnmsac.vf v0, ft0, v1
    vse64.v v0, (t4)
    add     t0, t0, t2
    j       lbj_v
lbk_nx:
    addi    s10, s10, 1
    j       lbk
lb_div:
    mul     t0, s9, s3
    add     t0, t0, s9
    slli    t0, t0, 3
    add     t1, s5, t0
    fld     ft1, 0(t1)
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
    vfdiv.vf v0, v0, ft1
    vse64.v v0, (t4)
    add     t0, t0, t2
    j       lb_dv
lbni:
    addi    s9, s9, -1
    j       lbi
lbd:
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
#              EKF-SPECIFIC FUNCTIONS
#     

# manual_atan(z) 
#   fa0 = z, returns fa0 = atan(z)
#   atan(z) ~ (pi/4)*z - z*(|z|-1)*(0.2447 + 0.0663*|z|)
    .globl manual_atan
manual_atan:
    fsgnjx.d ft0, fa0, fa0      # ft0 = |z|
    la      t0, atan_pi4
    fld     ft1, 0(t0)          # pi/4
    la      t0, atan_c1
    fld     ft2, 0(t0)          # 0.2447
    la      t0, atan_c2
    fld     ft3, 0(t0)          # 0.0663
    la      t0, const_one
    fld     ft4, 0(t0)          # 1.0
    fmul.d  ft5, ft1, fa0       # pi/4 * z
    fsub.d  ft6, ft0, ft4       # |z| - 1
    fmul.d  ft7, ft3, ft0       # 0.0663 * |z|
    fadd.d  ft7, ft2, ft7       # 0.2447 + 0.0663*|z|
    fmul.d  ft6, fa0, ft6       # z * (|z| - 1)
    fmul.d  ft6, ft6, ft7       # z*(|z|-1)*(...)
    fsub.d  fa0, ft5, ft6
    ret

# manual_atan2(y, x) 
#   fa0 = y, fa1 = x, returns fa0
#   Uses fs0-fs3 (callee-saved) to survive calls to manual_atan
    .globl manual_atan2
manual_atan2:
    addi    sp, sp, -40
    sd      ra,  0(sp)
    fsd     fs0, 8(sp)
    fsd     fs1, 16(sp)
    fsd     fs2, 24(sp)
    fsd     fs3, 32(sp)

    fmv.d   fs0, fa0             # fs0 = y
    fmv.d   fs1, fa1             # fs1 = x
    la      t0, const_pi
    fld     fs2, 0(t0)          # fs2 = PI
    la      t0, const_pi2
    fld     fs3, 0(t0)          # fs3 = PI/2
    fcvt.d.w ft0, zero           # ft0 = 0.0

    # x==0 && y==0 ?
    feq.d   t0, fs1, ft0
    feq.d   t1, fs0, ft0
    and     t2, t0, t1
    bnez    t2, a2_zero
    # x==0?
    bnez    t0, a2_xzero
    # |x| vs |y|
    fsgnjx.d ft1, fs1, fs1      # |x|
    fsgnjx.d ft2, fs0, fs0      # |y|
    flt.d   t0, ft1, ft2        # |x| < |y|?
    bnez    t0, a2_ybig

    #  |x| >= |y| 
    fdiv.d  fa0, fs0, fs1       # y/x
    call    manual_atan
    # check x < 0
    fcvt.d.w ft0, zero
    flt.d   t0, fs1, ft0
    beqz    t0, a2_ret
    # y >= 0?
    fle.d   t1, ft0, fs0
    beqz    t1, a2_subpi
    fadd.d  fa0, fa0, fs2
    j       a2_ret
a2_subpi:
    fsub.d  fa0, fa0, fs2
    j       a2_ret

a2_ybig:
    # |y| > |x| 
    fdiv.d  fa0, fs1, fs0       # x/y
    call    manual_atan
    fsub.d  fa0, fs3, fa0       # PI/2 - atan(r)
    # y < 0?
    fcvt.d.w ft0, zero
    flt.d   t0, fs0, ft0
    beqz    t0, a2_yb_cx
    fsgnjn.d fa0, fa0, fa0      # -angle
a2_yb_cx:
    flt.d   t0, fs1, ft0        # x < 0?
    beqz    t0, a2_ret
    fle.d   t1, ft0, fs0        # y >= 0?
    beqz    t1, a2_yb_sp
    fadd.d  fa0, fa0, fs2
    j       a2_ret
a2_yb_sp:
    fsub.d  fa0, fa0, fs2
    j       a2_ret

a2_xzero:
    fcvt.d.w ft0, zero
    flt.d   t0, ft0, fs0        # y > 0?
    bnez    t0, a2_poshalf
    fsgnjn.d fa0, fs3, fs3      # -PI/2
    j       a2_ret
a2_poshalf:
    fmv.d   fa0, fs3             # PI/2
    j       a2_ret
a2_zero:
    fcvt.d.w fa0, zero
a2_ret:
    ld      ra,  0(sp)
    fld     fs0, 8(sp)
    fld     fs1, 16(sp)
    fld     fs2, 24(sp)
    fld     fs3, 32(sp)
    addi    sp, sp, 40
    ret

# compute_hx(x_state, hx_out) 
#   a0 = x (Nx1), a1 = hx (Mx1)
    .globl compute_hx
compute_hx:
    addi    sp, sp, -64
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    fsd     fs0, 32(sp)
    fsd     fs1, 40(sp)
    fsd     fs2, 48(sp)
    # sp+56 used for temp

    mv      s0, a0              # x
    mv      s1, a1              # hx
    li      s2, 0               # j = 0
chx_lp:
    li      t0, 23
    bge     s2, t0, chx_dn
    li      t0, 12
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s0, t1
    fld     fs0, 0(t2)          # px (callee-saved)
    fld     fs1, 32(t2)         # py
    fld     fs2, 64(t2)         # pz
    # r = sqrt(px^2+py^2+pz^2)
    fmul.d  ft0, fs0, fs0
    fmul.d  ft1, fs1, fs1
    fmul.d  ft2, fs2, fs2
    fadd.d  ft0, ft0, ft1
    fadd.d  ft0, ft0, ft2
    fsqrt.d ft3, ft0             # r
    # store r
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     ft3, 0(t2)
    # theta = atan2(py, px)
    fmv.d   fa0, fs1
    fmv.d   fa1, fs0
    call    manual_atan2
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     fa0, 8(t2)          # theta
    # rho = sqrt(px^2+py^2)
    fmul.d  ft0, fs0, fs0
    fmul.d  ft1, fs1, fs1
    fadd.d  ft0, ft0, ft1
    fsqrt.d ft3, ft0             # rho
    # phi = atan2(pz, rho)
    fmv.d   fa0, fs2
    fmv.d   fa1, ft3
    fsd     ft3, 56(sp)         # save rho (not needed after call)
    call    manual_atan2
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     fa0, 16(t2)         # phi
    addi    s2, s2, 1
    j       chx_lp
chx_dn:
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    fld     fs0, 32(sp)
    fld     fs1, 40(sp)
    fld     fs2, 48(sp)
    addi    sp, sp, 64
    ret

#  cart_to_spherical(noisy_row, z_sph) 
#   a0 = noisy row (69 Cartesian doubles)
#   a1 = z_sph (69 spherical doubles)
    .globl cart_to_spherical
cart_to_spherical:
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
c2s_lp:
    li      t0, 23
    bge     s2, t0, c2s_dn
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s0, t1
    fld     fs0, 0(t2)          # px
    fld     fs1, 8(t2)          # py
    fld     fs2, 16(t2)         # pz
    # r
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
    # theta = atan2(py, px)
    fmv.d   fa0, fs1
    fmv.d   fa1, fs0
    call    manual_atan2
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     fa0, 8(t2)
    # rho
    fmul.d  ft0, fs0, fs0
    fmul.d  ft1, fs1, fs1
    fadd.d  ft0, ft0, ft1
    fsqrt.d ft3, ft0
    # phi = atan2(pz, rho)
    fmv.d   fa0, fs2
    fmv.d   fa1, ft3
    call    manual_atan2
    li      t0, 3
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s1, t1
    fsd     fa0, 16(t2)
    addi    s2, s2, 1
    j       c2s_lp
c2s_dn:
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    fld     fs0, 32(sp)
    fld     fs1, 40(sp)
    fld     fs2, 48(sp)
    addi    sp, sp, 64
    ret

# build_jacobian(x_state, J) 
#   a0 = x (Nx1), a1 = J (MxN)
    .globl build_jacobian
build_jacobian:
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
    li      a1, NM
    call    zero_mem

    li      s2, 0
bj_lp:
    li      t0, 23
    bge     s2, t0, bj_dn
    li      t0, 12
    mul     t1, s2, t0
    slli    t1, t1, 3
    add     t2, s0, t1
    fld     ft0, 0(t2)          # px
    fld     ft1, 32(t2)         # py
    fld     ft2, 64(t2)         # pz
    # r2, rho2
    fmul.d  ft3, ft0, ft0
    fmul.d  ft4, ft1, ft1
    fmul.d  ft5, ft2, ft2
    fadd.d  ft6, ft3, ft4       # rho2
    fadd.d  ft7, ft6, ft5       # r2
    fsqrt.d ft8, ft7             # r
    fsqrt.d ft9, ft6             # rho
    # skip if r or rho < eps
    la      t0, const_eps
    fld     ft10, 0(t0)
    flt.d   t0, ft8, ft10
    bnez    t0, bj_nx
    flt.d   t0, ft9, ft10
    bnez    t0, bj_nx

    li      t0, 3
    mul     s3, s2, t0           # row0 = jt*3
    li      t0, 12
    mul     s4, s2, t0           # col_base = jt*12

    # dr/dpx = px/r -> J[row0*N + col_base]
    fdiv.d  ft3, ft0, ft8
    li      a0, N
    mul     a1, s3, a0
    add     a1, a1, s4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)

    # dr/dpy = py/r -> J[row0*N + col_base+4]
    fdiv.d  ft3, ft1, ft8
    li      a0, N
    mul     a1, s3, a0
    addi    a3, s4, 4
    add     a1, a1, a3
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)

    # dr/dpz = pz/r -> J[row0*N + col_base+8]
    fdiv.d  ft3, ft2, ft8
    li      a0, N
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
    li      a0, N
    mul     a1, a3, a0
    add     a1, a1, s4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)

    # dtheta/dpy = px/rho2
    fdiv.d  ft3, ft0, ft6
    addi    a3, s3, 1
    li      a0, N
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
    li      a0, N
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
    li      a0, N
    mul     a1, a3, a0
    addi    a4, s4, 4
    add     a1, a1, a4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)

    # dphi/dpz = rho/r2
    fdiv.d  ft3, ft9, ft7
    addi    a3, s3, 2
    li      a0, N
    mul     a1, a3, a0
    addi    a4, s4, 8
    add     a1, a1, a4
    slli    a1, a1, 3
    add     a2, s1, a1
    fsd     ft3, 0(a2)

bj_nx:
    addi    s2, s2, 1
    j       bj_lp
bj_dn:
    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    addi    sp, sp, 48
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

    la      a0, msg_reading
    call    printf
    la      a0, csv_path
    addi    a1, sp, 48
    addi    a2, sp, 52
    call    read_csv
    mv      s0, a0
    lw      s1, 48(sp)
    lw      s2, 52(sp)
    la      a0, msg_ts
    mv      a1, s1
    mv      a2, s2
    call    printf

    # Alloc pointer table
    li      a0, NUM_PTRS*8
    call    malloc
    mv      s3, a0

    la      a0, msg_building
    call    printf

    #  Allocate matrices 
    li      a0, NN*8
    call    malloc
    sd      a0, PT_F(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_Q(s3)

    li      a0, MM*8
    call    malloc
    sd      a0, PT_R(s3)

    li      a0, N*8
    call    malloc
    sd      a0, PT_x(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_P(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_Ft(s3)

    li      a0, N*8
    call    malloc
    sd      a0, PT_xp(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_Pp(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_FP(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_FPFt(s3)

    li      a0, MM*8
    call    malloc
    sd      a0, PT_S(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_Jac(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_JacT(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_JP(s3)

    li      a0, MM*8
    call    malloc
    sd      a0, PT_JPJt(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_PJt(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_PJt_t(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_Ktsol(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_K(s3)

    li      a0, M*8
    call    malloc
    sd      a0, PT_zsph(s3)

    li      a0, M*8
    call    malloc
    sd      a0, PT_hx(s3)

    li      a0, M*8
    call    malloc
    sd      a0, PT_y(s3)

    li      a0, N*8
    call    malloc
    sd      a0, PT_Ky(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_KJ(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKJ(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKJt(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKJP(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_IKJPIt(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_KR(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_KRKt(s3)

    li      a0, NM*8
    call    malloc
    sd      a0, PT_Kt(s3)

    li      a0, NN*8
    call    malloc
    sd      a0, PT_Imat(s3)

    # BUILD F (same as LKF) 
    ld      a0, PT_F(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_F(s3)
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
    li      t0, 0
ebF_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, ebF_done
    li      t2, 0
ebF_ax:
    li      t3, 3
    bge     t2, t3, ebF_njt
    li      t4, 12
    mul     t4, t0, t4
    slli    t5, t2, 2
    add     t4, t4, t5
    li      a1, N
    mul     a2, t4, a1
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
    addi    a5, t4, 3
    mul     a2, a5, a1
    add     a3, a2, a5
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft5, 0(a4)
    addi    t2, t2, 1
    j       ebF_ax
ebF_njt:
    addi    t0, t0, 1
    j       ebF_jt
ebF_done:

    #  BUILD Q (same as LKF) 
    ld      a0, PT_Q(s3)
    li      a1, NN
    call    zero_mem
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
    fsd     ft4, 0(s4)
    fsd     ft2, 8(s4)
    fsd     ft0, 16(s4)
    fsd     ft5, 24(s4)
    ld      a5, PT_Q(s3)
    li      t0, 0
ebQ_jt:
    li      t1, NUM_JOINTS
    bge     t0, t1, ebQ_done
    li      t2, 0
ebQ_ax:
    li      t3, 3
    bge     t2, t3, ebQ_njt
    li      t4, 0
ebQ_i:
    li      t5, 4
    bge     t4, t5, ebQ_nax
    li      t6, 0
ebQ_j:
    li      a1, 4
    bge     t6, a1, ebQ_ni
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
    li      a1, N
    mul     a4, a2, a1
    add     a4, a4, a3
    slli    a4, a4, 3
    add     a4, a5, a4
    fsd     ft2, 0(a4)
    addi    t6, t6, 1
    j       ebQ_j
ebQ_ni:
    addi    t4, t4, 1
    j       ebQ_i
ebQ_nax:
    addi    t2, t2, 1
    j       ebQ_ax
ebQ_njt:
    addi    t0, t0, 1
    j       ebQ_jt
ebQ_done:
    mv      a0, s4
    call    free

    # BUILD R (diagonal: 0.25, 0.01, 0.01 per joint)
    ld      a0, PT_R(s3)
    li      a1, MM
    call    zero_mem
    ld      s4, PT_R(s3)
    li      t0, 0x3FD0000000000000
    fmv.d.x ft0, t0             # 0.25 (range var)
    li      t0, 0x3F847AE147AE147B
    fmv.d.x ft1, t0             # 0.01 (angle var)
    li      t1, 0
ebR_lp:
    li      t2, NUM_JOINTS
    bge     t1, t2, ebR_done
    # R[(j*3)*M + j*3] = 0.25
    li      a1, 3
    mul     a2, t1, a1
    li      a1, M
    mul     a3, a2, a1
    add     a3, a3, a2
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft0, 0(a4)
    # R[(j*3+1)*M + j*3+1] = 0.01
    addi    a2, a2, 1
    mul     a3, a2, a1
    add     a3, a3, a2
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft1, 0(a4)
    # R[(j*3+2)*M + j*3+2] = 0.01
    addi    a2, a2, 1
    mul     a3, a2, a1
    add     a3, a3, a2
    slli    a3, a3, 3
    add     a4, s4, a3
    fsd     ft1, 0(a4)
    addi    t1, t1, 1
    j       ebR_lp
ebR_done:

    # BUILD I_mat 
    ld      a0, PT_Imat(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_Imat(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
ebI_lp:
    li      t2, N
    bge     t1, t2, ebI_done
    li      a1, N
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       ebI_lp
ebI_done:

    #  INIT x 
    ld      a0, PT_x(s3)
    li      a1, N
    call    zero_mem
    ld      s4, PT_x(s3)
    li      t0, 0
eix_lp:
    li      t1, NUM_JOINTS
    bge     t0, t1, eix_done
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
    j       eix_lp
eix_done:

    #  INIT P = I 
    ld      a0, PT_P(s3)
    li      a1, NN
    call    zero_mem
    ld      s4, PT_P(s3)
    li      t0, 0x3FF0000000000000
    fmv.d.x ft0, t0
    li      t1, 0
eiP_lp:
    li      t2, N
    bge     t1, t2, eiP_done
    li      a1, N
    mul     a2, t1, a1
    add     a2, a2, t1
    slli    a2, a2, 3
    add     a3, s4, a2
    fsd     ft0, 0(a3)
    addi    t1, t1, 1
    j       eiP_lp
eiP_done:

    # Ft = F^T 
    ld      a0, PT_F(s3)
    ld      a1, PT_Ft(s3)
    li      a2, N
    li      a3, N
    call    mat_transpose

    #  Open output 
    la      a0, out_path
    call    open_output
    sd      a0, PT_FILE(s3)

    #  FILTER LOOP 
    la      a0, msg_running
    call    printf

    li      s4, 0
efilter_loop:
    bge     s4, s1, efilter_done

    #  PREDICTION: x_pred = F * x 
    ld      a0, PT_F(s3)
    ld      a1, PT_x(s3)
    ld      a2, PT_xp(s3)
    li      a3, N
    li      a4, N
    li      a5, 1
    call    mat_mul

    #  FP = F * P 
    ld      a0, PT_F(s3)
    ld      a1, PT_P(s3)
    ld      a2, PT_FP(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    #  FPFt = FP * Ft 
    ld      a0, PT_FP(s3)
    ld      a1, PT_Ft(s3)
    ld      a2, PT_FPFt(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    #  P_pred = FPFt + Q 
    ld      a0, PT_FPFt(s3)
    ld      a1, PT_Q(s3)
    ld      a2, PT_Pp(s3)
    li      a3, NN
    call    mat_add

    # Convert noisy[t] to spherical -> z_sph 
    mv      a0, s4
    mv      a1, s2
    mul     a2, a0, a1
    slli    a2, a2, 3
    add     a0, s0, a2          # &noisy[t*cols]
    ld      a1, PT_zsph(s3)
    call    cart_to_spherical

    #  Build Jacobian at x_pred 
    ld      a0, PT_xp(s3)
    ld      a1, PT_Jac(s3)
    call    build_jacobian

    #      JacT = Jac^T     
    ld      a0, PT_Jac(s3)
    ld      a1, PT_JacT(s3)
    li      a2, M
    li      a3, N
    call    mat_transpose

    #      JP = Jac * P_pred (MxN)     
    ld      a0, PT_Jac(s3)
    ld      a1, PT_Pp(s3)
    ld      a2, PT_JP(s3)
    li      a3, M
    li      a4, N
    li      a5, N
    call    mat_mul

    #      JPJt = JP * JacT (MxM)     
    ld      a0, PT_JP(s3)
    ld      a1, PT_JacT(s3)
    ld      a2, PT_JPJt(s3)
    li      a3, M
    li      a4, N
    li      a5, M
    call    mat_mul

    #      S = JPJt + R     
    ld      a0, PT_JPJt(s3)
    ld      a1, PT_R(s3)
    ld      a2, PT_S(s3)
    li      a3, MM
    call    mat_add

    #      PJt = P_pred * JacT (NxM)     
    ld      a0, PT_Pp(s3)
    ld      a1, PT_JacT(s3)
    ld      a2, PT_PJt(s3)
    li      a3, N
    li      a4, N
    li      a5, M
    call    mat_mul

    #      PJt_t = PJt^T (MxN)     
    ld      a0, PT_PJt(s3)
    ld      a1, PT_PJt_t(s3)
    li      a2, N
    li      a3, M
    call    mat_transpose

    #      Solve S * Kt_sol = PJt_t     
    ld      a0, PT_S(s3)
    ld      a1, PT_PJt_t(s3)
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

    #      hx = h(x_pred)     
    ld      a0, PT_xp(s3)
    ld      a1, PT_hx(s3)
    call    compute_hx

    #      y = z_sph - hx     
    ld      a0, PT_zsph(s3)
    ld      a1, PT_hx(s3)
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

    #      KJ = K * Jac     
    ld      a0, PT_K(s3)
    ld      a1, PT_Jac(s3)
    ld      a2, PT_KJ(s3)
    li      a3, N
    li      a4, M
    li      a5, N
    call    mat_mul

    #      IKJ = I - KJ     
    ld      a0, PT_Imat(s3)
    ld      a1, PT_KJ(s3)
    ld      a2, PT_IKJ(s3)
    li      a3, NN
    call    mat_sub

    #      IKJt = IKJ^T     
    ld      a0, PT_IKJ(s3)
    ld      a1, PT_IKJt(s3)
    li      a2, N
    li      a3, N
    call    mat_transpose

    #      IKJP = IKJ * P_pred     
    ld      a0, PT_IKJ(s3)
    ld      a1, PT_Pp(s3)
    ld      a2, PT_IKJP(s3)
    li      a3, N
    li      a4, N
    li      a5, N
    call    mat_mul

    #      IKJPIt = IKJP * IKJt     
    ld      a0, PT_IKJP(s3)
    ld      a1, PT_IKJt(s3)
    ld      a2, PT_IKJPIt(s3)
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

    #      P = IKJPIt + KRKt     
    ld      a0, PT_IKJPIt(s3)
    ld      a1, PT_KRKt(s3)
    ld      a2, PT_P(s3)
    li      a3, NN
    call    mat_add

    #      Write row     
    ld      a0, PT_FILE(s3)
    ld      a1, PT_x(s3)
    li      a2, N
    call    write_row

    #      Progress     
    li      t0, 500
    rem     t1, s4, t0
    bnez    t1, eno_print
    la      a0, msg_progress
    mv      a1, s4
    mv      a2, s1
    call    printf
eno_print:
    addi    s4, s4, 1
    j       efilter_loop

efilter_done:
    ld      a0, PT_FILE(s3)
    call    close_output
    la      a0, msg_done
    call    printf

    ld      ra,  0(sp)
    ld      s0,  8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    addi    sp, sp, 64
    li      a0, 0
    call    exit

