#ifndef __ALGO_MOBILENET_SSD_H
#define __ALGO_MOBILENET_SSD_H

#include "SNPETask.h"
#include "AlgoMobilenetSSDInterface.h"
#include <memory>

namespace vaas_algorithms {

class AlgoMobilenetSSD {
public:
    AlgoMobilenetSSD();
    ~AlgoMobilenetSSD();
    int init();
    int process(BufferInfos& inputInfos, MobilenetSSDConfig& config);
    int deinit();
    static AlgoMobilenetSSD* getInstance();
private:
    template<typename T> int pickBox(T* pOutput, std::vector<std::set<BoxInfo>>& sortedBoxInfo, float scoreThreshold);
    int process(uint8_t** pInputs, uint32_t* pInputsBytes, uint32_t inputNum, MobilenetSSDConfig& config);
    static AlgoMobilenetSSD msInstance;
    std::unique_ptr<SNPETask> mpSNPETask;
    uint32_t mBoxNum;
    uint32_t mClassNum;
    uint32_t mPosNum;
    uint32_t mSingleElementSize;
    std::vector<uint8_t> mOutput;
    std::vector<BufferInfo> mOutputInfo;
    BufferInfos mOutputInfos;
};

}

#endif // __ALGO_MOBILENET_SSD_H