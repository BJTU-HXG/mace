cd ../hexagonsdk354
source setup_sdk_env.source
cd ../hexagon_nn
make tree VERBOSE=1 V=android_Release_aarch64 CDSP_FLAG=1
make tree VERBOSE=1 V=hexagon_Release_dynamic_toolv83_v66 V66=1
cp ./android_Release_aarch64/libcdsprpc.so ./android_Release_aarch64/libhexagon_controller.so ../mace/third_party/nnlib/arm64-v8a/
cp ./hexagon_Release_dynamic_toolv83_v66/libhexagon_nn_skel.so ../mace/third_party/nnlib/v66/
cd ../mace
