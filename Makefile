AS = riscv64-linux-gnu-as
GCC = riscv64-linux-gnu-gcc
RUN = qemu-riscv64

all: lkf_asm ekf_asm

lkf_asm.o: lkf_asm.s
	$(AS) -o lkf_asm.o lkf_asm.s

lkf_helpers.o: lkf_helpers.c
	$(GCC) -c -O2 -o lkf_helpers.o lkf_helpers.c

lkf_asm: lkf_asm.o lkf_helpers.o
	$(GCC) -o lkf_asm lkf_asm.o lkf_helpers.o -static -lm

ekf_asm.o: ekf_asm.s
	$(AS) -o ekf_asm.o ekf_asm.s

ekf_helpers.o: ekf_helpers.c
	$(GCC) -c -O2 -o ekf_helpers.o ekf_helpers.c

ekf_asm: ekf_asm.o ekf_helpers.o
	$(GCC) -o ekf_asm ekf_asm.o ekf_helpers.o -static -lm

lkf_cpp: Lkf.cpp
	g++ -O2 -o lkf_cpp Lkf.cpp -lm

ekf_cpp: Ekf.cpp
	g++ -O2 -o ekf_cpp Ekf.cpp -lm

verify_all: verify_all.c
	gcc -O2 -o verify_all verify_all.c -lm

run: lkf_asm ekf_asm lkf_cpp ekf_cpp verify_all
	$(RUN) ./lkf_asm
	$(RUN) ./ekf_asm
	./lkf_cpp
	./ekf_cpp
	./verify_all

clean:
	rm -f *.o lkf_asm ekf_asm lkf_cpp ekf_cpp verify_all *_asm_output.csv *_output.csv

.PHONY: all run clean
