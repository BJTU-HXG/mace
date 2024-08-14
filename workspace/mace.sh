#!/bin/bash

MACE_ROOT="/home/NIO/mace"
NN_ROOT="/home/NIO/hexagon_nn"
SDK_ROOT="/home/NIO/hexagonsdk354"

echo "Current Path:" $(pwd)

function source_env(){
    source $SDK_ROOT/setup_sdk_env.source
    echo "=========Source Env========="
}

function convert_model(){
    shift
    for arg in "$@"
    do
        case $arg in
            --config=*)
                yml="${arg#*=}"
                ;;
            --quantize_stat=*)
                is_quantize="${arg#*=}"
                ;;
            *)
        esac
    done
    cd $MACE_ROOT
    if [ -n "$is_quantize" ]; then
        python tools/python/convert.py --config=$yml --quantize_stat
    else
        python tools/python/convert.py --config=$yml
    fi
    echo "=========Convert model=========="
}

function run_model(){
    shift
    for arg in "$@"
    do
        case $arg in
            --config=*)
                yml="${arg#*=}"
                ;;
            --target_abi=*)
                abi="${arg#*=}"
                ;;
            --vlog_level=*)
                log_level="${arg#*=}"
                ;;
            --accuracy_log=*)
                acc_log="${arg#*=}"
                ;;
            *)
        esac
    done
    cd $MACE_ROOT
    python tools/python/run_model.py --config=$yml --validate --target_abi=$abi --vlog_level=$log_level --accuracy_log=$acc_log
    echo "==========Run model==========="
}

function cc_nnlib(){
    source_env
    cd $NN_ROOT
    make tree VERBOSE=1 V=hexagon_Release_dynamic_toolv83_v66 V66=1
    cp hexagon_Release_dynamic_toolv83_v66/libhexagon_nn_skel.so ../mace/third_party/nnlib/v66
    echo "==========CC hexagon nnlib============"
}

function cc_android(){
    source_env
    cd $NN_ROOT
    make tree VERBOSE=1 V=android_Release_aarch64 CDSP_FLAG=1
    cp android_Release_aarch64/libhexagon_controller.so ../mace/third_party/nnlib/arm64-v8a
    cp android_Release_aarch64/libcdsprpc.so ../mace/third_party/nnlib/arm64-v8a
    echo "==========CC android nnlib========="
}


case $1 in
    source_env)
        source_env
        ;;
    convert)
        convert_model "$@"
        ;;
    run_model)
        run_model "$@"
        ;;
    cc_nnlib)
        cc_nnlib "$@"
        ;;
    cc_android)
        cc_android "$@"
        ;;
    *)
        echo "Operation dismatch!"
esac