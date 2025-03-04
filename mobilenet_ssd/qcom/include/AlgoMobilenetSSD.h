#ifndef __ALGO_MOBILENET_SSD_H
#define __ALGO_MOBILENET_SSD_H

#include "SNPETask.h"
#include "AlgoMobilenetSSDInterface.h"
#include <memory>

#ifdef TAG
#undef TAG
#endif
#define TAG "VAAS_SNPE"

namespace vaas_algorithms {

class AlgoMobilenetSSD {
public:
    AlgoMobilenetSSD();
    ~AlgoMobilenetSSD();
    int init();
    int process(BufferInfo& inputInfo, MobilenetSSDConfig& config);
    int deinit();
    static AlgoMobilenetSSD* getInstance();
private:
    template<typename T> int pickBox(T* pOutput, std::vector<std::set<BoxInfo>>& sortedBoxInfo, float scoreThreshold);
    int process(uint8_t** pInputs, uint32_t* pInputsBytes, uint32_t inputNum, MobilenetSSDConfig& config);
    static AlgoMobilenetSSD msInstance;
    std::unique_ptr<SNPETask> mpSNPETask;
    int mBoxNum;
    int mClassNum;
    int mPosNum;
    int mSingleElementSize;
    std::vector<uint8_t> mModelOutput;
};

}

#endif // __ALGO_MOBILENET_SSD_H