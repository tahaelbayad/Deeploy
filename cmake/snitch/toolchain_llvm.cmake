set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)

set(LLVM_TAG llvm)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}/${LLVM_TAG}-objdump)
set(CMAKE_LINKER ${TOOLCHAIN_PREFIX}/ld.lld)
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

add_compile_options(
    -I/scratch/msc24h7/llvm-project/INSTALL/picolibc/riscv/include
    -I/scratch/msc24h7/llvm-project/INSTALL/lib/clang/15.0.0/include
    -target riscv32-unknown-elf
    -MMD
    -MP
    -march=rv32imaf_zfh_xdma_xssr_xfrep_xfalthalf_xfquarter_xfaltquarter_xfvecsingle_xfvechalf_xfvecalthalf_xfvecquarter_xfvecaltquarter_xfauxhalf_xfauxalthalf_xfauxquarter_xfauxaltquarter_xfauxvechalf_xfauxvecalthalf_xfauxvecaltquarter_xfexpauxvechalf_xfexpauxvecalthalf_xfexpauxvecquarter_xfexpauxvecaltquarter
    -mcpu=snitch
    -menable-experimental-extensions
    -mabi=ilp32f
    -mcmodel=medany
    -fno-builtin-printf
    -fno-builtin-sqrtf
    -fno-common
    -fopenmp
    -ftls-model=local-exec
    -DNUM_CORES=${NUM_CORES}
)

add_link_options(
    -target riscv32-unknown-elf
    -MMD
    -MP
    -fuse-ld=lld
    -nostartfiles
    -nostdlib
    -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/12.0.1/lib/
    -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32imaf/ilp32f
    -L/scratch/msc24h7/llvm-project/INSTALL/picolibc/riscv/lib
)

link_libraries(
    -lc
    -lclang_rt.builtins-riscv32
)

# Required by math library to avoid conflict with stdint definition
add_compile_definitions(__LINK_LLD)
add_compile_definitions(__TOOLCHAIN_LLVM__)
add_compile_definitions(__DEFINED_uint64_t)
