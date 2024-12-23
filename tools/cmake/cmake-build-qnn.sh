set -ex


export QNX_BIN_DIR=/home/develop/software/
export QNX_HOST=$QNX_BIN_DIR/qnn/host/linux/x86_64
export QNX_TARGET=$QNX_BIN_DIR/qnn/target/qnx7
export QNX_TOOLS_DIR=$QNX_HOST
export QNX_TARGET_DIR=$QNX_TARGET


MACE_ROOT_DIR=$(pwd)

# build for arm linux aarch64
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/qnn
fi

if [[ -z "$QNX_BIN_DIR" ]]; then
    echo "please set \$QNX_BIN_DIR as your qnx toolchain root"
    exit 1
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DQNN=True \
      -DCROSSTOOL_ROOT=${QNX_BIN_DIR} \
      -DCMAKE_TOOLCHAIN_FILE=${MACE_ROOT_DIR}/cmake/toolchains/qnn.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DMACE_ENABLE_NEON=ON     \
      -DMACE_ENABLE_QUANTIZE=ON   \
      -DMACE_ENABLE_OPENCL=OFF       \
      -DMACE_ENABLE_CPU=ON                \
      -DMACE_ENABLE_BFLOAT16=OFF     \
      -DMACE_ENABLE_TESTS=OFF         \
      -DMACE_ENABLE_BENCHMARKS=OFF    \
      -DMACE_ENABLE_CODE_MODE=OFF    \
      -DMACE_ENABLE_HEXAGON_DSP=ON    \
      -DMACE_ENABLE_RPCMEM=ON    \
      -DCMAKE_INSTALL_PREFIX=install \
      ../../..

make $@ VERBOSE=1 -j && make install
cd ../../..
