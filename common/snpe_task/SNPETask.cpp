#include <fstream>
#include <math.h>
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "SNPETask.h"

namespace vaas_algorithms {

SNPETask::SNPETask() {}

SNPETask::SNPETask(std::string& dlc,
                   zdl::DlSystem::Runtime_t runtime,
                   zdl::DlSystem::ProfilingLevel_t profilingLevel,
                   std::string& outputDir,
                   bool bTfNBuffer,
                   bool bStaticQuantization,
                   uint32_t bitWidth,
                   uint32_t resizableDim,
                   bool bUseUserSuppliedBuffers) {
    ALOGI(TAG, "SNPETask constructor start");
    mbTfNBuffer = bTfNBuffer;
    mbStaticQuantization = bStaticQuantization;
    mBitWidth = bitWidth;
    mResizableDim = resizableDim;
    init(dlc, runtime, profilingLevel, bUseUserSuppliedBuffers);
    startDiagLog(outputDir);
    ALOGI(TAG, "SNPETask constructor finish");
}

SNPETask::~SNPETask() {
    ALOGI(TAG, "SNPETask destructor start");
    stopDiagLog();
    deinit();
    ALOGI(TAG, "SNPETask destructor finish");
}

int SNPETask::init(std::string& dlc,
                   zdl::DlSystem::Runtime_t runtime,
                   zdl::DlSystem::ProfilingLevel_t profilingLevel,
                   bool bUseUserSuppliedBuffers) {
    ALOGI(TAG, "SNPE init start");
    std::unique_ptr <zdl::DlContainer::IDlContainer> container = loadContainer(dlc);
    checkRuntime(runtime);
    mpSNPE = setBuilderOptions(container, runtime, profilingLevel, bUseUserSuppliedBuffers);
    mTensorShape = mpSNPE->getInputDimensions();
    zdl::DlSystem::Version_t snpeVersion = zdl::SNPE::SNPEFactory::getLibraryVersion();
    mLoggerOptional = mpSNPE->getDiagLogInterface();

    ALOGI(TAG, "snpe version: %s, model version: %s", snpeVersion.asString().c_str(), mpSNPE->getModelVersion().c_str());
    ALOGI(TAG, "SNPE init finish");
    return EXIT_SUCCESS;
}

int SNPETask::startDiagLog(std::string& outputDir) {
    ALOGI(TAG, "startDiagLog start");
    zdl::DiagLog::Options options =  mLoggerOptional->getOptions();
    options.LogFileDirectory = outputDir;
    mLoggerOptional->setOptions(options);
    mLoggerOptional->start();
    ALOGI(TAG, "startDiagLog finish");
    return EXIT_SUCCESS;
}

int SNPETask::stopDiagLog() {
    ALOGI(TAG, "stopDiagLog start");
    mLoggerOptional->stop();
    zdl::SNPE::SNPEFactory::terminateLogging();
    ALOGI(TAG, "stopDiagLog finish");
    return EXIT_SUCCESS;
}

int SNPETask::deinit() {
    ALOGI(TAG, "snpe deinit start");
    mpSNPE.reset();
    ALOGI(TAG, "snpe deinit finish");
    return EXIT_SUCCESS;
}

int SNPETask::process(const unsigned char* data,
                      uint32_t dataSize,
                      zdl::DlSystem::TensorMap& outputTensor) {
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = createInputTensor(mTensorShape, data, dataSize);
    process(inputTensor, outputTensor);
    return EXIT_SUCCESS;
}

int SNPETask::process(std::vector<float>& input,
                      zdl::DlSystem::TensorMap& outputTensor) {
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = createInputTensor(input);
    process(inputTensor, outputTensor);
    return EXIT_SUCCESS;
}

int SNPETask::process(float* pInput,
                      uint32_t inputSize,
                      float* pOutput,
                      uint32_t outputSize) {
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = createInputTensor(pInput, inputSize);
    zdl::DlSystem::TensorMap outputTensor;
    process(inputTensor, outputTensor);
    ALOGI(TAG, "outputTensor size: %zd", outputTensor.size());
    return EXIT_SUCCESS;
}

int SNPETask::process(std::unique_ptr<zdl::DlSystem::ITensor>& inputTensor,
                      zdl::DlSystem::TensorMap& outputTensor) {
    ALOGI(TAG, "SNPE process start");
    mpSNPE->execute(inputTensor.get(), outputTensor);
    ALOGI(TAG, "SNPE process finish");
    return EXIT_SUCCESS;
}

int SNPETask::process(uint8_t** pInput, uint32_t* pInputSize, uint32_t inputNum,
                      uint8_t** pOutput, uint32_t* pOutputSize, uint32_t outputNum) {
    ALOGI(TAG, "SNPE process start");
    if ((pInput == nullptr) || (pOutput == nullptr)) {
        ALOGE(TAG, "pInput(%p) or pOutput(%p) is nullptr", pInput, pOutput);
        return EXIT_FAILURE;
    }

    const zdl::DlSystem::Optional<zdl::DlSystem::StringList>& inputNamesOpt = mpSNPE->getInputTensorNames();
    const zdl::DlSystem::Optional<zdl::DlSystem::StringList>& outputNamesOpt = mpSNPE->getOutputTensorNames();

    if (!inputNamesOpt || !outputNamesOpt) {
        ALOGE(TAG, "get inputNamesOpt(%d) or outputNamesOpt(%d) failed", !inputNamesOpt, !inputNamesOpt);
        return EXIT_FAILURE;
    }

    zdl::DlSystem::StringList inputNames = *inputNamesOpt, outputNames = *outputNamesOpt;
    uint32_t inputNamesSize = inputNames.size(), outputNamesSize = outputNames.size();

    for (uint32_t i = 0; i < inputNamesSize; ++i) {
        ALOGI(TAG, "input, i: %u, name: %s", i, inputNames.at(i));
    }

    for (uint32_t i = 0; i < outputNamesSize; ++i) {
        ALOGI(TAG, "output, i: %u, name: %s", i, outputNames.at(i));
    }

    if ((inputNamesSize > inputNum) || (outputNamesSize > outputNum)) {
        ALOGE(TAG, "inputNamesSize(%u) > inputNum(%u) or outputNamesSize(%u) > outputNum(%u)",
                inputNamesSize, inputNum, outputNamesSize, outputNum);
        return EXIT_FAILURE;
    }

    zdl::DlSystem::UserBufferMap inputUserBufferMap, outputUserBufferMap;
    std::unordered_map<std::string, std::unique_ptr<zdl::DlSystem::IUserBuffer>> inputBackedUserBuffer, outputBackedUserBuffer;

    for (uint32_t i = 0; i < inputNamesSize; ++i) {
        ALOGI(TAG, "create input, i: %u, input: %p, size: %u, name: %s", i, pInput[i], pInputSize[i], inputNames.at(i));
        if (createUserBuffer(inputUserBufferMap, inputBackedUserBuffer, pInput[i], pInputSize[i], inputNames.at(i)) == EXIT_FAILURE) {
            ALOGE(TAG, "create input user buffer failed, name: %s", inputNames.at(i));
            return EXIT_FAILURE;
        }
    }

    for (uint32_t i = 0; i < outputNamesSize; ++i) {
        ALOGI(TAG, "create output, i: %u, output: %p, size: %u, name: %s", i, pOutput[i], pOutputSize[i], outputNames.at(i));
        if (createUserBuffer(outputUserBufferMap, outputBackedUserBuffer, pOutput[i], pOutputSize[i], outputNames.at(i)) == EXIT_FAILURE) {
            ALOGE(TAG, "create output user buffer failed, name: %s", outputNames.at(i));
            return EXIT_FAILURE;
        }
    }

    ALOGI(TAG, "inputUserBufferMap: %p, outputUserBufferMap: %p",
            inputUserBufferMap.getUserBuffer(inputNames.at(0)), outputUserBufferMap.getUserBuffer(outputNames.at(0)));
    mpSNPE->execute(inputUserBufferMap, outputUserBufferMap);
    ALOGI(TAG, "SNPE process end");
    return EXIT_SUCCESS;
}

void SNPETask::checkRuntime(zdl::DlSystem::Runtime_t& runtime) {
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        ALOGE(TAG, "Request runtime %s is not available on this paltform, use cpu instead.", runtime2Str(runtime).c_str());
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }
}

std::unique_ptr<zdl::SNPE::SNPE> SNPETask::setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                             zdl::DlSystem::Runtime_t runtime,
                                                             zdl::DlSystem::ProfilingLevel_t profilingLevel,
                                                             bool bUseUserSuppliedBuffers) {
    ALOGI(TAG, "setBuilderOptions start");
    zdl::SNPE::SNPEBuilder builder(container.get());
    std::unique_ptr<zdl::SNPE::SNPE> snpe = builder.setRuntimeProcessor(runtime)
                                                   .setProfilingLevel(profilingLevel)
                                                   .setUseUserSuppliedBuffers(bUseUserSuppliedBuffers)
                                                   .setOutputLayers({})
                                                   .build();
    ALOGI(TAG, "setBuilderOptions finish");
    return snpe;
}

std::unique_ptr<zdl::DlContainer::IDlContainer> SNPETask::loadContainer(std::string& dlc) {
    ALOGI(TAG, "loadContainer start");
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(dlc);
    ALOGI(TAG, "loadContainer finish");
    return container;
}

std::unique_ptr<zdl::DlSystem::ITensor> SNPETask::createInputTensor(const zdl::DlSystem::TensorShape& shape,
                                                                    const unsigned char* data,
                                                                    uint32_t dataSize) {
    ALOGI(TAG, "createInputTensor start");
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(shape, data, dataSize);
    ALOGI(TAG, "createInputTensor finish");
    return inputTensor;
}

std::unique_ptr<zdl::DlSystem::ITensor> SNPETask::createInputTensor(std::vector<float>& input) {
    ALOGI(TAG, "createInputTensor start");
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(mTensorShape);
    std::copy(input.begin(), input.end(), inputTensor->begin());
    ALOGI(TAG, "createInputTensor finish");
    return inputTensor;
}

std::unique_ptr<zdl::DlSystem::ITensor> SNPETask::createInputTensor(float* pInput, uint32_t size) {
    ALOGI(TAG, "createInputTensor start");
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(mTensorShape);

    if (size != inputTensor->getSize()) {
        ALOGE(TAG, "input size (%u) is not equal to required (%zd)", size, inputTensor->getSize());
    }

    std::copy(pInput, pInput + size, inputTensor->begin());
    ALOGI(TAG, "createInputTensor finish");
    return inputTensor;
}

int SNPETask::copyOutputTensor(float* pOutputTensor, uint32_t outputSize, zdl::DlSystem::TensorMap& outputTensor) {
    // std::copy(outputTensor)
    return EXIT_SUCCESS;
}

int SNPETask::createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                               std::unordered_map<std::string, std::unique_ptr<zdl::DlSystem::IUserBuffer>>& backedUserBuffers,
                               uint8_t* pBuffers,
                               uint32_t bufferSize,
                               const char* name) {
    ALOGI(TAG, "createUserBuffer start");
    zdl::DlSystem::Optional<zdl::DlSystem::IBufferAttributes*> bufferAttributesOpt = mpSNPE->getInputOutputBufferAttributes(name);

    if (!bufferAttributesOpt) {
        ALOGE(TAG, "get bufferAttributesOpt failed");
        return EXIT_FAILURE;
    }

    zdl::DlSystem::TensorShape bufferShape = (*bufferAttributesOpt)->getDims();
    uint32_t dimNum = bufferShape.rank();
    std::vector<size_t> strides(dimNum);
    strides.back() = mBitWidth / 8;

    for (uint32_t i = dimNum - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * ((bufferShape[i] == 0) ? mResizableDim : bufferShape[i]);
        ALOGI(TAG, "i: %u, bufferShape: %lu", i, bufferShape[i]);
    }

    uint32_t requiredBufferSize = strides[0] * ((bufferShape[0] == 0) ? mResizableDim : bufferShape[0]);

    if (bufferSize < requiredBufferSize) {
        ALOGE(TAG, "bufferSize(%u) < requiredBufferSize(%u)", bufferSize, requiredBufferSize);
        return EXIT_FAILURE;
    }

    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> bUserBufferEncoding;

    if (mbTfNBuffer) {
        if (((*bufferAttributesOpt)->getEncodingType() == zdl::DlSystem::UserBufferEncoding::ElementType_t::FLOAT)
                && mbStaticQuantization) {
            ALOGE(TAG, "current model's encoding type not support TfNBuffer");
            return EXIT_FAILURE;
        }

        const zdl::DlSystem::UserBufferEncodingTfN* pUserBufferEncodingTfN =
                dynamic_cast<const zdl::DlSystem::UserBufferEncodingTfN*>((*bufferAttributesOpt)->getEncoding());
        uint64_t stepEquivalentTo0 = pUserBufferEncodingTfN->getStepExactly0();
        float quantizedStepSize = pUserBufferEncodingTfN->getQuantizedStepSize();
        bUserBufferEncoding = std::make_unique<zdl::DlSystem::UserBufferEncodingTfN>(stepEquivalentTo0, quantizedStepSize, mBitWidth);
    } else {
        bUserBufferEncoding = std::make_unique<zdl::DlSystem::UserBufferEncodingFloat>();
    }

    std::unique_ptr<zdl::DlSystem::IUserBuffer> pUserBuffer =
            zdl::SNPE::SNPEFactory::getUserBufferFactory().createUserBuffer(pBuffers,
                                                                            requiredBufferSize,
                                                                            strides,
                                                                            bUserBufferEncoding.get());
    userBufferMap.add(name, pUserBuffer.get());
    backedUserBuffers[name] = std::move(pUserBuffer);
    ALOGI(TAG, "createUserBuffer end");
    return EXIT_SUCCESS;
}

int SNPETask::saveOutputMap(zdl::DlSystem::TensorMap& outputTensor, const std::string& savePath) {
    ALOGI(TAG, "saveOutputMap start");
    zdl::DlSystem::StringList tensorNames = outputTensor.getTensorNames();
    int batchSize = mTensorShape[0];

    for (auto tensorName:tensorNames) {
        zdl::DlSystem::ITensor* output = outputTensor.getTensor(tensorName);
        uint32_t batchChunk  = output->getSize() / batchSize;

        for (int i = 0; i < batchSize; i++) {
            std::string file = savePath  + "/batch_" + std::to_string(i) + ".raw";

            if (!ensureDirectory(savePath)) {
                ALOGE(TAG, "Failed to create output directory: %s, error code: %s", savePath.c_str(), std::strerror(errno));
                return EXIT_FAILURE;
            }

            std::ofstream os(file);
            int k = 0, label = 0;
            float conf = 0, maxVal = *output->cbegin();

            for (auto iter = output->cbegin() + i * batchChunk; iter != output->cbegin() + (i + 1) * batchChunk; iter++) {
                float f = *iter;
                os.write(reinterpret_cast<char*>(&f), sizeof(float));

                if (f > maxVal) {
                    maxVal = f;
                    label = k;
                }

                conf += exp(f);
                k++;
            }

            conf = exp(maxVal) / conf;
            ALOGI(TAG, "classification label: %d, confidence: %f", label, conf);
        }
    }

    ALOGI(TAG, "saveOutputMap finish");
    return EXIT_SUCCESS;
}

zdl::DlSystem::TensorShape SNPETask::getTensorShape() {
    return mTensorShape;
}

int SNPETask::getBatchSize() {
    return mTensorShape[0];
}

}