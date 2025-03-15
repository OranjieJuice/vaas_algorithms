cmake_minimum_required(VERSION 3.4.1)
project(mobilenet_ssd)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/libs/${ANDROID_ABI}/)
add_compile_options(-Wall -Wextra -std=c++14 -fexceptions -frtti)

if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(SNPE_LIBRARY_DIRECTORY ${SNPE_ROOT}/lib/aarch64-android-clang6.0)
    set(AOSP_VNDK_DIRECTORY ${AOSP}/prebuilts/vndk/v28/arm64/arch-arm64-armv8-a/shared/vndk-sp)
elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set(SNPE_LIBRARY_DIRECTORY ${SNPE_ROOT}/lib/arm-android-clang6.0)
    set(AOSP_VNDK_DIRECTORY ${AOSP}/prebuilts/vndk/v28/arm/arch-arm-armv7-a-neon/shared/vndk-sp)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/qcom/include
    ${PROJECT_SOURCE_DIR}/../common/snpe_task/include
    ${PROJECT_SOURCE_DIR}/../common/utils/include
    ${SNPE_ROOT}/include/zdl
    ${AOSP}/system/core/libcutils/include
)

LINK_DIRECTORIES(
        ${SNPE_LIBRARY_DIRECTORY}
        ${AOSP_VNDK_DIRECTORY}
)

add_library(
    mobilenetssd
    SHARED
    ${PROJECT_SOURCE_DIR}/qcom/AlgoMobilenetSSD.cpp
    ${PROJECT_SOURCE_DIR}/AlgoMobilenetSSDInterface.cpp
    ${PROJECT_SOURCE_DIR}/../common/snpe_task/SNPETask.cpp
    ${PROJECT_SOURCE_DIR}/../common/utils/VAASAlgoUtils.cpp
)

target_link_libraries(
    mobilenetssd
    log
    c++
    cutils
    c++_shared
    SNPE
    calculator
    calculator_htp
    hta
    PlatformValidatorShared
    snpe_dsp_domains_v2
)