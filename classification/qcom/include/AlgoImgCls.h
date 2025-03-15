#ifndef __ALGO_IMG_CLS_H
#define __ALGO_IMG_CLS_H

#include "SNPETask.h"
#include <memory>

namespace vaas_algorithms {

class AlgoImgCls {
public:
    AlgoImgCls();
    ~AlgoImgCls();
    int init();
    int process(BufferInfos& inputInfos, ImageClassificationInfo& outputInfo);
    int deinit();
    static AlgoImgCls* getInstance();
private:
    template<typename T> int executeClass(T* pOutput, uint32_t outputSize, ImageClassificationInfo& outputInfo);
    int process(uint8_t** pInputs, uint32_t* pInputsBytes, uint32_t inputNum, ImageClassificationInfo& outputInfo);
    static AlgoImgCls msInstance;
    std::unique_ptr<SNPETask> mpSNPETask;
    uint32_t mClassNum;
    uint32_t mElementSize;
    std::vector<uint8_t> mOutput;
    std::vector<BufferInfo> mOutputInfo;
    BufferInfos mOutputInfos;
};

}

#endif // __ALGO_IMG_CLS_H