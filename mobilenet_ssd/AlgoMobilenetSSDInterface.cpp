#include "AlgoMobilenetSSDInterface.h"
#include "AlgoMobilenetSSD.h"

namespace vaas_algorithms {

extern "C"
int algoMobilenetSSDInit() {
    return AlgoMobilenetSSD::getInstance()->init();
}

extern "C"
int algoMobilenetSSDProcess(BufferInfo& inputInfo, MobilenetSSDConfig& config) {
    return AlgoMobilenetSSD::getInstance()->process(inputInfo, config);
}

extern "C"
int algoMobilenetSSDDeinit() {
    return AlgoMobilenetSSD::getInstance()->deinit();
}

}