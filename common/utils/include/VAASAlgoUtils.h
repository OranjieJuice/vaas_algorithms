#ifndef __VAAS_ALGO_UTILS_H
#define __VAAS_ALGO_UTILS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include "DlSystem/DlEnums.hpp"

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

struct BufferInfo {
    uint8_t** pBuffer;
    uint32_t* pWidth;
    uint32_t* pHeight;
    uint32_t* pRow;
    uint32_t* pCol;
    uint32_t num;
    ImageType imageType;
    BufferType bufferType;
    BufferInfo() :
            pBuffer(nullptr),
            pWidth(nullptr),
            pHeight(nullptr),
            pRow(nullptr),
            pCol(nullptr),
            num(0),
            imageType(ImageType::IMAGE_TYPE_RGB),
            bufferType(BufferType::BUFFER_TYPE_UINT8) {}
    BufferInfo(uint8_t** pBuffer,
               uint32_t* pWidth,
               uint32_t* pHeight,
               uint32_t* pRow,
               uint32_t* pCol,
               uint32_t num,
               ImageType imageType,
               BufferType bufferType) :
               pBuffer(pBuffer),
               pWidth(pWidth),
               pHeight(pHeight),
               pRow(pRow),
               pCol(pCol),
               num(num),
               imageType(imageType),
               bufferType(bufferType) {}
};

struct BoxInfo {
    int label;
    float confidence;
    std::vector<float> position;
    BoxInfo() : label(0), confidence(0) {}
    BoxInfo(int label, float confidence) : label(label), confidence(confidence) {}
    BoxInfo(int label, float confidence, std::vector<float>& position) : label(label), confidence(confidence), position(std::move(position)) {}

    bool operator<(const BoxInfo& boxInfoB) const {
        return confidence > boxInfoB.confidence;
    }

    float operator[](int index) const {
        return position[index];
    }
};

typedef void (*processObjectCallback_t)(void* pObject, void* pData);

zdl::DlSystem::Runtime_t str2Runtime(const char* str);
std::string runtime2Str(zdl::DlSystem::Runtime_t runtime);
bool ensureDirectory(const std::string& dir);
void nms(std::set<BoxInfo>& sortedBoxInfos, std::vector<BoxInfo>& pickedBoxInfos, float iouThreshold, int topK);
float calculateIOU(const BoxInfo &boxInfoA, const BoxInfo &boxInfoB, float delta = 1e-5);

}

#endif // __VAAS_ALGO_UTILS_H