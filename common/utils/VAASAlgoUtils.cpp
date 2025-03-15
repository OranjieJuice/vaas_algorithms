#include <iostream>
#include <sys/stat.h>
#include "VAASAlgoUtils.h"
#include "VAASAlgoLog.h"

namespace vaas_algorithms {

bool ensureDirectory(const std::string& dir) {
    auto i = dir.find_last_of('/');
    std::string prefix = dir.substr(0, i);

    if (dir.empty() || dir == "." || dir == "..") {
        return true;
    }

    if ((i != std::string::npos) && !ensureDirectory(prefix)) {
        return false;
    }

    int rc = mkdir(dir.c_str(), S_IRWXU | S_IXGRP | S_IROTH | S_IXOTH);
    if ((rc == -1) && (errno != EEXIST)) {
        return false;
    } else {
        struct stat st;
        if (stat(dir.c_str(), &st) == -1) {
            return false;
        }

        return S_ISDIR(st.st_mode);
    }
}

void nms(std::set<BoxInfo>& sortedBoxInfos, std::vector<BoxInfo>& pickedBoxInfos, float iouThreshold, int topK) {
    int start = pickedBoxInfos.size();

    for (const BoxInfo& boxInfo:sortedBoxInfos) {
        if (static_cast<int>(pickedBoxInfos.size()) - start >= topK) {
            break;
        }

        bool bPick = true;

        for (uint32_t i = start; i < pickedBoxInfos.size() && bPick; ++i) {
            bPick &= calculateIOU(boxInfo, pickedBoxInfos[i]) < iouThreshold;
        }

        if (bPick) {
            pickedBoxInfos.push_back(std::move(boxInfo));
        }
    }
}

float calculateIOU(const BoxInfo &boxInfoA, const BoxInfo &boxInfoB, float delta) {
    float left = fmax(boxInfoA[0], boxInfoB[0]), right = fmin(boxInfoA[2], boxInfoB[2]);
    float up   = fmax(boxInfoA[1], boxInfoB[1]), down  = fmin(boxInfoA[3], boxInfoB[3]);
    float areaA = (boxInfoA[2] - boxInfoA[0]) * (boxInfoA[3] - boxInfoA[1]);
    float areaB = (boxInfoB[2] - boxInfoB[0]) * (boxInfoB[3] - boxInfoB[1]);
    float areaOverlap = fmax(0.0f, right - left) * fmax(0.0f, down - up);
    return areaOverlap / (areaA + areaB - areaOverlap + delta);
}

}