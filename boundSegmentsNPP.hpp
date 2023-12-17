
/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// https://blog.csdn.net/ice__snow/article/details/79699388
#pragma once

#include "opencv2/opencv.hpp"


cv::cuda::GpuMat nppiLabelMarker(cv::cuda::GpuMat src);
