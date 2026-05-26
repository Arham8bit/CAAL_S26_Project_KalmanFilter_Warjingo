AS      = riscv64-linux-gnu-as
GCC     = riscv64-linux-gnu-gcc
RUN_S   = ~/qemu-8.2.2/build/qemu-riscv64
RUN_V   = ~/qemu-8.2.2/build/qemu-riscv64 -cpu rv64,v=on,vlen=256

# --- Milestone 3 scalar targets (reference) ---
lkf_asm.o: lkf_asm.s
	$(AS) -o lkf_asm.o lkf_asm.s

lkf_io.o: lkf_io.c
	$(GCC) -c -O2 -o lkf_io.o lkf_io.c

lkf_asm: lkf_asm.o lkf_io.o
	$(GCC) -o lkf_asm lkf_asm.o lkf_io.o -static -lm

ekf_asm.o: ekf_asm.s
	$(AS) -o ekf_asm.o ekf_asm.s

ekf_io.o: ekf_io.c
	$(GCC) -c -O2 -o ekf_io.o ekf_io.c

ekf_asm: ekf_asm.o ekf_io.o
	$(GCC) -o ekf_asm ekf_asm.o ekf_io.o -static -lm

# --- Milestone 4 vector targets ---
lkf_vector.o: lkf_vector.s
	$(AS) -march=rv64gcv -o lkf_vector.o lkf_vector.s

ekf_vector.o: ekf_vector.s
	$(AS) -march=rv64gcv -o ekf_vector.o ekf_vector.s

lkf_vector: lkf_vector.o lkf_io.o
	$(GCC) -o lkf_vector lkf_vector.o lkf_io.o -static -lm

ekf_vector: ekf_vector.o ekf_io.o
	$(GCC) -o ekf_vector ekf_vector.o ekf_io.o -static -lm

# --- Verification ---
verify_m4: verify_m4.c
	gcc -O2 -o verify_m4 verify_m4.c -lm

# --- Build groups ---
all: scalar vector verify_m4

scalar: lkf_asm ekf_asm

vector: lkf_vector ekf_vector

# --- Run scalar (M3 reference) ---
run_scalar: scalar
	$(RUN_S) ./lkf_asm
	$(RUN_S) ./ekf_asm

# --- Run vector (M4) ---
run_vector: vector
	$(RUN_V) ./lkf_vector
	$(RUN_V) ./ekf_vector

# --- Full pipeline: build all, run both, verify, time ---
run: all
	@echo "===== Running M3 Scalar LKF ====="
	$(RUN_S) ./lkf_asm
	@echo "===== Running M4 Vector LKF ====="
	$(RUN_V) ./lkf_vector
	@echo "===== Running M3 Scalar EKF ====="
	$(RUN_S) ./ekf_asm
	@echo "===== Running M4 Vector EKF ====="
	$(RUN_V) ./ekf_vector
	@echo "===== Numerical Verification ====="
	./verify_m4

# --- Timing comparison ---
benchmark: all
	@echo "===== Timing M3 Scalar LKF ====="
	time $(RUN_S) ./lkf_asm
	@echo "===== Timing M4 Vector LKF ====="
	time $(RUN_V) ./lkf_vector
	@echo "===== Timing M3 Scalar EKF ====="
	time $(RUN_S) ./ekf_asm
	@echo "===== Timing M4 Vector EKF ====="
	time $(RUN_V) ./ekf_vector

clean:
	rm -f *.o lkf_asm ekf_asm lkf_vector ekf_vector verify_m4 \
	      *_output.csv *_asm_output.csv *_vector_output.csv

.PHONY: all scalar vector run run_scalar run_vector benchmark clean
