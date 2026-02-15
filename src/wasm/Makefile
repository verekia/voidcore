CLANG ?= /opt/homebrew/opt/llvm/bin/clang
WASM_OPT = wasm-opt
CFLAGS = --target=wasm32-unknown-unknown -O3 -flto -fno-strict-aliasing -msimd128 -nostdlib \
         -Wl,--no-entry -Wl,--export-dynamic -Wl,--import-memory -Wl,--initial-memory=33554432 \
         -Wl,-z,stack-size=262144
SRCS = voidcore.c math.c arena.c cull.c sort.c bvh.c
OUT = ../../public/voidcore.wasm

all: $(OUT)

$(OUT): $(SRCS) math.h arena.h cull.h sort.h bvh.h
	$(CLANG) $(CFLAGS) $(SRCS) -o $(OUT)
	$(WASM_OPT) -O3 $(OUT) -o $(OUT)

clean:
	rm -f $(OUT)

.PHONY: all clean
