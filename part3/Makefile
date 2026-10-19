GEM5_DIR := /root/gem5
GEM5 := $(GEM5_DIR)/build/RISCV/gem5.opt
CONFIG := $(GEM5_DIR)/configs/deprecated/example/se.py

SRC := main.c
BIN := main
HOST_BIN := main_host

CC := riscv64-linux-gnu-gcc
HOST_CC := gcc

COMMON_FLAGS := -std=c11 -O2 -Wall -Wextra
RISCV_FLAGS := $(COMMON_FLAGS) -static -march=rv64gcv -mabi=lp64d
LDLIBS := -lm

INPUT ?= /workspace/data/input_gray.pgm
OUTPUT ?= /workspace/P3/reconstructed.pgm
BLOCKS ?= 1

.PHONY: all build host-test gem5-test run stats clean

all: build

build: $(BIN)

$(BIN): $(SRC)
	$(CC) $(RISCV_FLAGS) $(SRC) -o $(BIN) $(LDLIBS)

$(HOST_BIN): $(SRC)
	$(HOST_CC) $(COMMON_FLAGS) $(SRC) -o $(HOST_BIN) $(LDLIBS)

host-test: $(HOST_BIN)
	./$(HOST_BIN) ../data/input_gray.pgm reconstructed_host.pgm 1

gem5-test: BLOCKS=1
gem5-test: run

run: $(BIN)
	rm -rf m5out
	$(GEM5) $(CONFIG) \
		-c $(CURDIR)/$(BIN) \
		-o "$(INPUT) $(OUTPUT) $(BLOCKS)" \
		--cpu-type=TimingSimpleCPU \
		--mem-size=256MB \
		--caches \
		--l1i_size=32kB \
		--l1i_assoc=8 \
		--l1d_size=32kB \
		--l1d_assoc=8 \
		--cacheline=32
	$(MAKE) stats

stats:
	grep -E "simSeconds|simTicks|numCycles|simInsts|simOps|hostSeconds|cpi|overallMisses|overallMissRate" m5out/stats.txt

clean:
	rm -f $(BIN) $(HOST_BIN)
	rm -f reconstructed.pgm reconstructed_host.pgm
	rm -rf m5out
