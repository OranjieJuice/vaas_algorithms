#ifndef __ALGO_IMG_CLS_INTERFACE_H
#define __ALGO_IMG_CLS_INTERFACE_H

#include "VAASAlgoUtils.h"

namespace vaas_algorithms {

extern "C" int algoImgClsInit();
extern "C" int algoImgClsProcess(BufferInfos& inputInfos, ImageClassificationInfo& outputInfo);
extern "C" int algoImgClsDeinit();

}

#endif // __ALGOIMGCLSINTERFACE_H