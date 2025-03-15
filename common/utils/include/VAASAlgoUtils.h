#ifndef __VAAS_ALGO_UTILS_H
#define __VAAS_ALGO_UTILS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include "VAASAlgoLog.h"

#ifdef TAG
#undef TAG
#endif
#define TAG "VAAS_ALGO"

namespace vaas_algorithms {

enum ImageType {
    IMAGE_TYPE_DEFAULT = 0,
    IMAGE_TYPE_NV21 = 1,
    IMAGE_TYPE_RGB = 2,
    IMAGE_TYPE_BIN = 3,
};

enum BufferType {
    BUFFER_TYPE_DEFAULT = 0,
    BUFFER_TYPE_UINT8 = 1,
    BUFFER_TYPE_INT8 = 2,
    BUFFER_TYPE_UINT16 = 3,
    BUFFER_TYPE_INT16 = 4,
    BUFFER_TYPE_UINT32 = 5,
    BUFFER_TYPE_INT32 = 6,
    BUFFER_TYPE_FLOAT32 = 7,
};

static const std::unordered_map<BufferType, uint32_t> BUFFER_ELEMENT_SIZE = {
    {BufferType::BUFFER_TYPE_UINT8, sizeof(uint8_t)},
    {BufferType::BUFFER_TYPE_INT8, sizeof(int8_t)},
    {BufferType::BUFFER_TYPE_UINT16, sizeof(uint16_t)},
    {BufferType::BUFFER_TYPE_INT16, sizeof(int16_t)},
    {BufferType::BUFFER_TYPE_UINT32, sizeof(uint32_t)},
    {BufferType::BUFFER_TYPE_INT32, sizeof(int32_t)},
    {BufferType::BUFFER_TYPE_FLOAT32, sizeof(float)}
};

struct BufferInfo {
    uint8_t* pBuffer;
    uint32_t width;
    uint32_t height;
    uint32_t row;
    uint32_t col;
    uint32_t rotate90;
    ImageType imageType;
    BufferType bufferType;

    BufferInfo() :
            pBuffer(nullptr),
            width(0),
            height(0),
            row(0),
            col(0),
            rotate90(0),
            imageType(ImageType::IMAGE_TYPE_DEFAULT),
            bufferType(BufferType::BUFFER_TYPE_DEFAULT) {}

    BufferInfo(
            uint8_t* pBuffer,
            uint32_t width,
            uint32_t height,
            uint32_t row,
            uint32_t col,
            uint32_t rotate90,
            ImageType imageType,
            BufferType bufferType) :
            pBuffer(pBuffer),
            width(width),
            height(height),
            row(row),
            col(col),
            rotate90(rotate90),
            imageType(imageType),
            bufferType(bufferType) {}

    uint32_t getRGBSize() {
        return 3 * getRGBChannelSize();
    }

    uint32_t getRGBChannelSize() {
        return height * width;
    }

    uint32_t getYUVSize() {
        return getYChannelSize() * 3 / 2;
    }

    uint32_t getYChannelSize() {
        return col * row;
    }

    uint32_t getBufferSize() {
        if (bufferType == BufferType::BUFFER_TYPE_DEFAULT) {
            ALOGE(TAG, "getBufferSize failed, bufferType: %d", bufferType);
            return 0;
        }

        auto it = BUFFER_ELEMENT_SIZE.find(bufferType);
        uint32_t size = it == BUFFER_ELEMENT_SIZE.end() ? 0 : it->second;

        switch (imageType) {
            case ImageType::IMAGE_TYPE_NV21:
                size *= getYUVSize();
                break;
            case ImageType::IMAGE_TYPE_BIN:
                size *= getRGBChannelSize();
                break;
            case ImageType::IMAGE_TYPE_RGB:
                size *= getRGBSize();
                break;
            default:
                ALOGE(TAG, "unsupport image type, imageType: %d", imageType);
                return 0;
                break;
        }

        return size;
    }

    static uint32_t calCol(uint32_t width) {
        return width;
    }

    static uint32_t calRow(uint32_t height) {
        return (height + 16 + 63) & ~63;
    }
};

struct BufferInfos {
    uint32_t num;
    BufferInfo* pBufferInfo;
    BufferInfos() : num(0), pBufferInfo(nullptr) {}
    BufferInfos(uint32_t num, BufferInfo* pBufferInfo) : num(num), pBufferInfo(pBufferInfo) {}
};

struct ImageClassificationInfo {
    uint32_t label;
    float confidence;
    ImageClassificationInfo() : label(0), confidence(0.0f) {}
    ImageClassificationInfo(uint32_t label, float confidence) : label(label), confidence(confidence) {}
};

struct BoxInfo {
    uint32_t label;
    float confidence;
    std::vector<float> position;
    BoxInfo() : label(0), confidence(0.0f) {}
    BoxInfo(uint32_t label, float confidence) : label(label), confidence(confidence) {}
    BoxInfo(uint32_t label, float confidence, std::vector<float>& position) : label(label), confidence(confidence), position(std::move(position)) {}

    bool operator<(const BoxInfo& boxInfoB) const {
        return confidence > boxInfoB.confidence;
    }

    float operator[](uint32_t index) const {
        return position[index];
    }
};

typedef void (*processObjectCallback_t)(void* pObject, void* pData);
bool ensureDirectory(const std::string& dir);
void nms(std::set<BoxInfo>& sortedBoxInfos, std::vector<BoxInfo>& pickedBoxInfos, float iouThreshold, int topK);
float calculateIOU(const BoxInfo &boxInfoA, const BoxInfo &boxInfoB, float delta = 1e-5);

}

#endif // __VAAS_ALGO_UTILS_H