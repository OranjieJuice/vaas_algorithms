#ifndef __VAAS_ALGO_LOG_H
#define __VAAS_ALGO_LOG_H

#include <android/log.h>

namespace vaas_algorithms {

#define ALOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, __VA_ARGS__)
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, __VA_ARGS__)

}

#endif // __VAAS_ALGO_LOG_H