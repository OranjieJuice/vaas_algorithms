#ifndef __ALGO_MOBILENET_SSD_INTERFACE_H
#define __ALGO_MOBILENET_SSD_INTERFACE_H

#include "VAASAlgoUtils.h"

namespace vaas_algorithms{

struct MobilenetSSDConfig {
    float scoreThreshold;
    float iouThreshold;
    int topK;
    void* pObject;
    processObjectCallback_t processObjectCallback;
    MobilenetSSDConfig() :
            scoreThreshold(0.6),
            iouThreshold(0.5),
            topK(-1),
            pObject(nullptr),
            processObjectCallback(nullptr) {}
    MobilenetSSDConfig(
            float scoreThreshold,
            float iouThreshold,
            int topK,
            void* pObject,
            processObjectCallback_t processObjectCallback) :
            scoreThreshold(scoreThreshold),
            iouThreshold(iouThreshold),
            topK(topK),
            pObject(pObject),
            processObjectCallback(processObjectCallback) {}
};

extern "C" int algoMobilenetSSDInit();
extern "C" int algoMobilenetSSDProcess(BufferInfos& inputInfos, MobilenetSSDConfig& config);
extern "C" int algoMobilenetSSDDeinit();

}

#endif // __ALGO_MOBILENET_SSD_INTERFACE_H