#include "AlgoImgCls.h"
#include <string>
#include <cutils/properties.h>

namespace vaas_algorithms {

AlgoImgCls AlgoImgCls::msInstance = AlgoImgCls();

AlgoImgCls::AlgoImgCls() {}

AlgoImgCls::~AlgoImgCls() {}

int AlgoImgCls::init() {
    std::string dlc = "/odm/etc/camera/shufflenetv2.dlc";
    std::string outputDir = "/data/data/com.android.camera2/save/vaas";
    char key[256] = "persist.com.android.camera.runtime";
    char value[256];
    property_get(key, value, "dsp");
    ALOGI(TAG, "use runtime %s", value);
    zdl::DlSystem::Runtime_t runtime = str2Runtime(value);

    zdl::DlSystem::ProfilingLevel_t profilingLevel = zdl::DlSystem::ProfilingLevel_t::DETAILED;
    ALOGI(TAG, "build snpe");
    mpSNPETask = std::make_unique<SNPETask>(dlc, runtime, profilingLevel, outputDir, false, false, 32, 1, true);
    ALOGI(TAG, "vaasInit finish");
    return EXIT_SUCCESS;
}

int AlgoImgCls::process(BufferInfo inputInfo, BufferInfo outputInfo, bool bReuseBuffer) {
    uint8_t** pInputs = inputInfo.pBuffer;
    std::vector<uint32_t> inputsBytes(inputInfo.num, 0);
    for (uint32_t i = 0; i < inputInfo.num; ++i) {
        inputsBytes[i] = inputInfo.pWidth[i] * inputInfo.pHeight[i] * 3 * sizeof(float);
    }

    uint8_t** poutputs = outputInfo.pBuffer;
    std::vector<uint32_t> outputsBytes(outputInfo.num, 0);
    for (uint32_t i = 0; i < inputInfo.num; ++i) {
        outputsBytes[i] = outputInfo.pWidth[i] * outputInfo.pHeight[i] * sizeof(float);
    }

    process(pInputs, inputsBytes.data(), inputInfo.num,
            poutputs, outputsBytes.data(), outputInfo.num);
    
    return EXIT_SUCCESS;
}

int AlgoImgCls::process(float* pInput, uint32_t inputSize, float* pOutput, uint32_t outputSize) {
    std::string outputDir = "/data/data/com.android.camera2/save/vaas";
    
    std::string savePath = outputDir + "/Result/";
    mpSNPETask->process(pInput, inputSize, pOutput, outputSize);
    // mpSNPETask->saveOutputMap(outputTensor, savePath);
    return EXIT_SUCCESS;
}

int AlgoImgCls::process(uint8_t** pInputs, uint32_t* pInputsBytes, uint32_t inputNum,
                        uint8_t** pOutputs, uint32_t* pOutputsBytes, uint32_t outputNum) {
    mpSNPETask->process(pInputs, pInputsBytes, inputNum, pOutputs, pOutputsBytes, outputNum);

    for (int i = 0; i < outputNum; ++i) {
        executeClass(reinterpret_cast<float*>(pOutputs[i]), pOutputsBytes[i] / sizeof(float));
    }

    return EXIT_SUCCESS;
}

int AlgoImgCls::deinit() {
    mpSNPETask.reset();
    return EXIT_SUCCESS;
}

AlgoImgCls* AlgoImgCls::getInstance() {
    return &msInstance;
}

template <typename T>
int AlgoImgCls::executeClass(T* pOutput, uint32_t outputSize) {
    float conf = 0;
    float maxVal = -FLT_MAX;
    int label = -1;

    for (int i = 0; i < outputSize; ++i) {
        if (pOutput[i] > maxVal) {
            maxVal = pOutput[i];
            label = i;
        }

        conf += exp(pOutput[i]);
    }

    conf = exp(maxVal) / conf;
    ALOGI(TAG, "classification label: %d, confidence: %f", label, conf);
    return EXIT_SUCCESS;
}

}