set(CMAKE_SYSTEM_NAME QNN)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "${CROSSTOOL_ROOT}/qnn/host/linux/x86_64/usr/bin/ntoaarch64-gcc")
set(CMAKE_CXX_COMPILER "${CROSSTOOL_ROOT}/qnn/host/linux/x86_64/usr/bin/ntoaarch64-g++")
set(CMAKE_FIND_ROOT_PATH "${CROSSTOOL_ROOT}/qnn/target/qnx7")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_CXX_FLAGS "-D_XOPEN_SOURCE=500 -D_QNX_SOURCE ${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(MACE_CC_FLAGS "-D_XOPEN_SOURCE=500 -D_QNX_SOURCE ${MACE_CC_FLAGS}")