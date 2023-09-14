set -e

MACE_ROOT_DIR=$(pwd)

# build for arm linux aarch64
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/qnx
fi

if [[ -z "$QNX_BIN_DIR" ]]; then
    echo "please set \$QNX_BIN_DIR as your qnx toolchain root"
    exit 1
fi

MACE_ENABLE_CPU=ON

MACE_ENABLE_CODE_MODE=OFF
if [[ "$RUNMODE" == "code" ]]; then
    MACE_ENABLE_CODE_MODE=ON
fi

DMACE_ENABLE_BFLOAT16=OFF
if [[ "$BFLOAT16" == "ON" ]]; then
    DMACE_ENABLE_BFLOAT16=ON
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DQNX=True \
      -DCROSSTOOL_ROOT=${QNX_BIN_DIR} \
      -DCMAKE_TOOLCHAIN_FILE=${MACE_ROOT_DIR}/cmake/toolchains/qnx.cmake \
      -DMACE_ENABLE_NEON=OFF         \
      -DMACE_ENABLE_QUANTIZE=OFF     \
      -DMACE_ENABLE_OPENCL=OFF       \
      -DMACE_ENABLE_CPU=${MACE_ENABLE_CPU}                \
      -DMACE_ENABLE_BFLOAT16=${DMACE_ENABLE_BFLOAT16}     \
      -DMACE_ENABLE_TESTS=ON         \
      -DMACE_ENABLE_BENCHMARKS=ON    \
      -DMACE_ENABLE_CODE_MODE=${MACE_ENABLE_CODE_MODE}    \
      -DCMAKE_INSTALL_PREFIX=install \
      ../../..
make -j$(nproc) VERBOSE=1 && make install
cd ../../..