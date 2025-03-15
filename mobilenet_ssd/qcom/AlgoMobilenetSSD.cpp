#include "AlgoMobilenetSSD.h"
#include <string>
#include <cutils/properties.h>

namespace vaas_algorithms {

AlgoMobilenetSSD AlgoMobilenetSSD::msInstance;

AlgoMobilenetSSD::AlgoMobilenetSSD() {}

AlgoMobilenetSSD::~AlgoMobilenetSSD() {}

int AlgoMobilenetSSD::init() {
    std::string dlc = "/odm/etc/camera/mobilenet-v1-ssd-mp-0_675.dlc";
    std::string outputDir = "/data/data/com.android.camera2/save/vaas";
    mBoxNum = 3000;
    mClassNum = 21;
    mPosNum = 4;
    mSingleElementSize = BUFFER_ELEMENT_SIZE.find(BufferType::BUFFER_TYPE_FLOAT32)->second;
    uint32_t elementNum = mBoxNum * (mClassNum + mPosNum);
    uint32_t bytes = elementNum * mSingleElementSize;
    mOutput.resize(bytes);
    mOutputInfo = std::vector<BufferInfo>{
            BufferInfo(mOutput.data(), elementNum, 1, 0, 0, 0, ImageType::IMAGE_TYPE_BIN, BufferType::BUFFER_TYPE_FLOAT32)};
    mOutputInfos = BufferInfos(mOutputInfo.size(), mOutputInfo.data());

    char key[256] = "persist.com.android.camera.mssd.runtime";
    char value[256];
    property_get(key, value, "dsp");
    ALOGI(TAG, "use runtime %s", value);
    zdl::DlSystem::Runtime_t runtime = SNPETask::str2Runtime(value);
    zdl::DlSystem::ProfilingLevel_t profilingLevel = zdl::DlSystem::ProfilingLevel_t::DETAILED;

    ALOGI(TAG, "build snpe");
    mpSNPETask = std::make_unique<SNPETask>(dlc, runtime, profilingLevel, outputDir, false, false, 32, 1, true);
    ALOGI(TAG, "vaasInit finish");
    return EXIT_SUCCESS;
}

int AlgoMobilenetSSD::process(BufferInfos& inputInfos, MobilenetSSDConfig& config) {
    std::vector<uint8_t*> inputs(inputInfos.num, nullptr);
    std::vector<uint32_t> inputsBytes(inputInfos.num, 0);

    for (uint32_t i = 0; i < inputInfos.num; ++i) {
        inputsBytes[i] = inputInfos.pBufferInfo[i].getBufferSize();
        inputs[i] = inputInfos.pBufferInfo[i].pBuffer;
    }

    return process(inputs.data(), inputsBytes.data(), inputInfos.num, config);
}

int AlgoMobilenetSSD::process(uint8_t** pInputs, uint32_t* pInputsBytes, uint32_t inputNum, MobilenetSSDConfig& config) {
    uint8_t* pModelOutput = mOutput.data();
    uint32_t modelOutputSize = mOutput.size();
    mpSNPETask->process(pInputs, pInputsBytes, inputNum, &pModelOutput, &modelOutputSize, 1);
    std::vector<std::set<BoxInfo>> sortedBoxInfos(mClassNum);
    std::vector<BoxInfo> pickedBoxInfos;

    for (uint32_t i = 0; i < mBoxNum; ++i) {
        pickBox(reinterpret_cast<float*>(pModelOutput) + i * (mClassNum + mPosNum), sortedBoxInfos, config.scoreThreshold);
    }

    for (uint32_t i = 1; i < mClassNum; ++i) {
        nms(sortedBoxInfos[i], pickedBoxInfos, config.iouThreshold, config.topK);
    }

    for (const BoxInfo& boxInfo:pickedBoxInfos) {
        ALOGI(TAG, "label: %d, confidence: %f, box: %f %f %f %f",
                boxInfo.label, boxInfo.confidence, boxInfo[0], boxInfo[1], boxInfo[2], boxInfo[3]);
    }

    config.processObjectCallback(config.pObject, &pickedBoxInfos);
    return EXIT_SUCCESS;
}

int AlgoMobilenetSSD::deinit() {
    mpSNPETask.reset();
    return EXIT_SUCCESS;
}

AlgoMobilenetSSD* AlgoMobilenetSSD::getInstance() {
    return &msInstance;
}

template <typename T>
int AlgoMobilenetSSD::pickBox(T* pModelOutput, std::vector<std::set<BoxInfo>>& sortedBoxInfos, float scoreThreshold) {
    uint32_t label = 0;
    float conf = pModelOutput[0];

    for (uint32_t i = 1; i < mClassNum; ++i) {
        if (pModelOutput[i] > conf) {
            conf = pModelOutput[i];
            label = i;
        }
    }

    if (!std::isnan(conf) && conf > scoreThreshold && label > 0) {
        ALOGD(TAG, "classification label: %u, confidence: %f", label, conf);
        std::vector<float> pos(mPosNum);
        std::copy(pModelOutput + mClassNum, pModelOutput + mClassNum + mPosNum, pos.begin());
        sortedBoxInfos[label].insert(BoxInfo(label, conf, pos));

        for (uint32_t i = 0; i < mPosNum; ++i) {
            ALOGD(TAG, "box[%u]: %f", i, pModelOutput[mClassNum + i]);
        }
    }

    return EXIT_SUCCESS;
}

}