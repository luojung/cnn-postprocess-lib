
#pragma once

#include <math.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "utils/os.hpp"
#include "utils/macro.h"

using namespace std;

namespace RSMat {
    
// mat 角度计算
POSTINF_DECL cv::cuda::GpuMat arctan2_cuda(cv::cuda::GpuMat& angle_v, cv::cuda::GpuMat& angle_h);
POSTINF_DECL cv::cuda::GpuMat sin_cuda(cv::cuda::GpuMat& angle);
POSTINF_DECL cv::cuda::GpuMat cos_cuda(cv::cuda::GpuMat& angle);
POSTINF_DECL void angle_remainder(cv::cuda::GpuMat& angle_mat, double remainder);

}