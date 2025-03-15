#include "AlgoImgCls.h"
#include <string>
#include <cutils/properties.h>
#include <iostream>
#include <fstream>

namespace vaas_algorithms {

AlgoImgCls AlgoImgCls::msInstance;

AlgoImgCls::AlgoImgCls() {}

AlgoImgCls::~AlgoImgCls() {}

int AlgoImgCls::init() {
    ALOGI(TAG, "AlgoImgCls init finish");
    std::string dlc = "/odm/etc/camera/shufflenetv2.dlc";
    std::string outputDir = "/data/data/com.android.camera2/save/vaas";
    mClassNum = 1000;
    mElementSize = BUFFER_ELEMENT_SIZE.find(BufferType::BUFFER_TYPE_FLOAT32)->second;
    uint32_t bytes = mClassNum * mElementSize;
    mOutput.resize(bytes);

    char key[256] = "persist.com.android.camera.imgcls.runtime";
    char value[256];
    property_get(key, value, "dsp");
    ALOGI(TAG, "use runtime %s", value);
    zdl::DlSystem::Runtime_t runtime = SNPETask::str2Runtime(value);
    zdl::DlSystem::ProfilingLevel_t profilingLevel = zdl::DlSystem::ProfilingLevel_t::DETAILED;
    ALOGI(TAG, "build snpe");
    mpSNPETask = std::make_unique<SNPETask>(dlc, runtime, profilingLevel, outputDir, false, false, 32, 1, true);
    ALOGI(TAG, "AlgoImgCls init finish");
    return EXIT_SUCCESS;
}

int AlgoImgCls::process(BufferInfos& inputInfos, ImageClassificationInfo& outputInfo) {
    ALOGI(TAG, "AlgoImgCls process start");
    std::vector<uint8_t*> inputs(inputInfos.num, nullptr);
    std::vector<uint32_t> inputsBytes(inputInfos.num, 0);

    for (uint32_t i = 0; i < inputInfos.num; ++i) {
        inputsBytes[i] = inputInfos.pBufferInfo[i].getBufferSize();
        inputs[i] = inputInfos.pBufferInfo[i].pBuffer;
    }

    process(inputs.data(), inputsBytes.data(), inputInfos.num, outputInfo);
    ALOGI(TAG, "AlgoImgCls process finish");
    return EXIT_SUCCESS;
}

int AlgoImgCls::process(uint8_t** pInputs, uint32_t* pInputsBytes, uint32_t inputNum, ImageClassificationInfo& outputInfo) {
    uint8_t* pOutput = mOutput.data();
    uint32_t outputSize = mOutput.size();
    mpSNPETask->process(pInputs, pInputsBytes, inputNum, &pOutput, &outputSize, 1);
    return executeClass(reinterpret_cast<float*>(pOutput), outputSize / sizeof(float), outputInfo);
}

int AlgoImgCls::deinit() {
    mpSNPETask.reset();
    return EXIT_SUCCESS;
}

AlgoImgCls* AlgoImgCls::getInstance() {
    return &msInstance;
}

template <typename T>
int AlgoImgCls::executeClass(T* pOutput, uint32_t outputSize, ImageClassificationInfo& outputInfo) {
    for (uint32_t i = 0; i < outputSize; ++i) {
        if (pOutput[i] > pOutput[outputInfo.label]) {
            outputInfo.label = i;
        }

        outputInfo.confidence += exp(pOutput[i]);
    }

    outputInfo.confidence = exp(pOutput[outputInfo.label]) / outputInfo.confidence;
    ALOGI(TAG, "classification label: %u, confidence: %f", outputInfo.label, outputInfo.confidence);
    return EXIT_SUCCESS;
}

}