#!/bin/bash

rm -r ./build
mkdir ./build

ABI=arm64-v8a
PLATFORM=android-31
ANDROID_NDK=~/Android/Sdk/ndk/28.0.12433566/
SNPE_ROOT=~/Qualcomm/SNPE/snpe-1.60.0.3313/
AOSP=~/aosp/pixel_3xl_android12_r34/

while [ $# -gt 0 ]; do
    name=$1
    shift
    suffix_shift=1

    if [ $name = "--abi" ]; then
        ABI=$1
        echo "set ABI: $ABI"
    elif [ $name = "--platform" ]; then
        PLATFORM=$1
        echo "set PLATFORM: $PLATFORM"
    elif [ $name = "--ndk" ]; then
        ANDROID_NDK=$1
        echo "set ANDROID_NDK: $ANDROID_NDK"
    elif [ $name = "--snpe" ]; then
        SNPE_ROOT=$1
        echo "set SNPE_ROOT: $SNPE_ROOT"
    elif [ $name = "--aosp" ]; then
        AOSP=$1
        echo "set AOSP: $AOSP"
    else
        suffix_shift=0
        echo "invalid arg: $name"
    fi

    if [ $suffix_shift -gt 0 ]; then
        shift
    fi
done

echo "\n"

cmake \
    -S ./ \
    -B ./build \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ABI \
    -DANDROID_NDK=$ANDROID_NDK \
    -DANDROID_PLATFORM=$PLATFORM \
    -DSNPE_ROOT=$SNPE_ROOT \
    -DAOSP=$AOSP

cd ./build
make