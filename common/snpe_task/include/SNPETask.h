#ifndef __SNPE_TASK_H
#define __SNPE_TASK_H

#include <string>
#include "SNPE/SNPE.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "DiagLog/Options.hpp"
#include "VAASAlgoUtils.h"
#include "VAASAlgoLog.h"

#ifdef TAG
#undef TAG
#endif
#define TAG "VAAS_SNPE"

namespace vaas_algorithms {

class SNPETask {
public:
    int init(
            std::string& dlc,
            zdl::DlSystem::Runtime_t runtime,
            zdl::DlSystem::ProfilingLevel_t profilingLevel,
            bool bUseUserSuppliedBuffers);
    int process(
            const unsigned char* data,
            uint32_t dataSize,
            zdl::DlSystem::TensorMap& outputTensor);
    int process (
            std::vector<float>& input,
            zdl::DlSystem::TensorMap& outputTensor);
    int process (
            float* pInput,
            uint32_t inputSize,
            float* pOutput,
            uint32_t outputSize);
    int process(
            std::unique_ptr<zdl::DlSystem::ITensor>& inputTensor,
            zdl::DlSystem::TensorMap& outputTensor);
    int process(
            uint8_t** pInput, uint32_t* pInputSize, uint32_t inputNum,
            uint8_t** pOutput, uint32_t* pOutputSize, uint32_t outputNum);
    int saveOutputMap (zdl::DlSystem::TensorMap& outputTensor, const std::string& savePath);
    zdl::DlSystem::TensorShape getTensorShape();
    int getBatchSize();
    int deinit();
    int startDiagLog(std::string& outputDir);
    int stopDiagLog();
    static zdl::DlSystem::Runtime_t str2Runtime(const char* str);
    static std::string runtime2Str(zdl::DlSystem::Runtime_t runtime);
    SNPETask();
    SNPETask(
            std::string& dlc,
            zdl::DlSystem::Runtime_t runtime,
            zdl::DlSystem::ProfilingLevel_t profilingLevel,
            std::string& outputDir,
            bool bTfNBuffer,
            bool bStaticQuantization,
            uint32_t bitWidth,
            uint32_t resizableDim,
            bool bUseUserSuppliedBuffers);
    ~SNPETask();

private:
    void checkRuntime(zdl::DlSystem::Runtime_t& runtime);
    std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(
            std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
            zdl::DlSystem::Runtime_t runtime,
            zdl::DlSystem::ProfilingLevel_t profilingLevel,
            bool bUseUserSuppliedBuffers);
    std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainer(std::string& dlc);
    std::unique_ptr<zdl::DlSystem::ITensor> createInputTensor(
            const zdl::DlSystem::TensorShape& shape,
            const unsigned char* data,
            uint32_t dataSize);
    std::unique_ptr<zdl::DlSystem::ITensor> createInputTensor(std::vector<float>& input);
    std::unique_ptr<zdl::DlSystem::ITensor> createInputTensor(float* pInput, uint32_t inputSize);
    int createUserBuffer(
            zdl::DlSystem::UserBufferMap& userBufferMap,
            std::unordered_map<std::string, std::unique_ptr<zdl::DlSystem::IUserBuffer>>& backedUserBuffers,
            uint8_t* pBuffers,
            uint32_t bufferSize,
            const char* name);
    std::unique_ptr<zdl::SNPE::SNPE> mpSNPE;
    zdl::DlSystem::TensorShape mTensorShape;
    zdl::DlSystem::Optional<zdl::DiagLog::IDiagLog*> mLoggerOptional;
    bool mbTfNBuffer;
    bool mbStaticQuantization;
    uint32_t mBitWidth;
    uint32_t mResizableDim;
};

}

#endif // __SNPE_TASK_H