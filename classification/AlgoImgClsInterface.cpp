#include "AlgoImgClsInterface.h"
#include "AlgoImgCls.h"

namespace vaas_algorithms {

extern "C"
int algoImgClsInit() {
    return AlgoImgCls::getInstance()->init();
}

extern "C"
int algoImgClsProcess(BufferInfo inputInfo, BufferInfo outputInfo, bool bReuseBuffer) {
    return AlgoImgCls::getInstance()->process(inputInfo, outputInfo, bReuseBuffer);
}

extern "C"
int algoImgClsDeinit() {
    return AlgoImgCls::getInstance()->deinit();
}

}