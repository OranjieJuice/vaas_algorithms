#include <iostream>
#include <sys/stat.h>
#include "VAASAlgoUtils.h"
#include "VAASAlgoLog.h"

namespace vaas_algorithms {

zdl::DlSystem::Runtime_t str2Runtime(const char* str) {
    zdl::DlSystem::Runtime_t runtime;

    if (strcmp(str, "cpu") == 0) {
        runtime = zdl::DlSystem::Runtime_t::CPU;
    } else if (strcmp(str, "gpu") == 0) {
        runtime = zdl::DlSystem::Runtime_t::GPU;
    } else if (strcmp(str, "dsp") == 0) {
        runtime = zdl::DlSystem::Runtime_t::DSP;
    } else if (strcmp(str, "aip") == 0) {
        runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
    } else {
        runtime = zdl::DlSystem::Runtime_t::CPU;
        ALOGE(TAG, "Runtime %s is not available, return cpu", str);
    }

    return runtime;
}

std::string runtime2Str(zdl::DlSystem::Runtime_t runtime) {
    std::string str;

    switch (runtime) {
        case zdl::DlSystem::Runtime_t::CPU:
            str = "cpu";
            break;
        case zdl::DlSystem::Runtime_t::GPU:
            str = "gpu";
            break;
        case zdl::DlSystem::Runtime_t::DSP:
            str = "dsp";
            break;
        case zdl::DlSystem::Runtime_t::AIP_FIXED8_TF:
            str = "aip";
            break;
        default:
            str = "cpu";
            ALOGE(TAG, "Runtime %d is not available, return cpu.", (int) runtime);
            break;
    }

    return str;
}

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
        if (pickedBoxInfos.size() - start >= topK) {
            break;
        }

        bool bPick = true;

        for (int i = start; i < pickedBoxInfos.size() && bPick; ++i) {
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