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
    int process(BufferInfo inputInfo, BufferInfo outputInfo, bool bReuse);
    int deinit();
    static AlgoImgCls* getInstance();
private:
    template<typename T> int executeClass(T* pOutput, uint32_t outputSize);
    int process(float* pInput, uint32_t inputSize, float* pOutput, uint32_t outputSize);
    int process(uint8_t** pInputs, uint32_t* pInputsBytes, uint32_t inputNum,
                uint8_t** pOutputs, uint32_t* pOutputsBytes, uint32_t outputNum);
    static AlgoImgCls msInstance;
    std::unique_ptr<SNPETask> mpSNPETask;
};

}

#endif // __ALGO_IMG_CLS_H