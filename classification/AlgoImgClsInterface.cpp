#include "AlgoImgClsInterface.h"
#include "AlgoImgCls.h"

namespace vaas_algorithms {

extern "C"
int algoImgClsInit() {
    return AlgoImgCls::getInstance()->init();
}

extern "C"
int algoImgClsProcess(BufferInfos& inputInfos, ImageClassificationInfo& outputInfo) {
    return AlgoImgCls::getInstance()->process(inputInfos, outputInfo);
}

extern "C"
int algoImgClsDeinit() {
    return AlgoImgCls::getInstance()->deinit();
}

}