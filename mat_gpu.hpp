
#pragma once
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include "utils/macro.h"

using namespace std;
using namespace cv;

#define CV_NAN 0

namespace RSMat {

// **********基于opencv cuda 封装的接口***********
POSTINF_DECL cv::cuda::GpuMat equal_cuda(cv::InputArray src1, cv::InputArray src2);

POSTINF_DECL cv::cuda::GpuMat greater_cuda(cv::InputArray src1, cv::InputArray src2);
 
POSTINF_DECL cv::cuda::GpuMat ge_cuda(cv::InputArray src1, cv::InputArray src2);

POSTINF_DECL cv::cuda::GpuMat logical_or_cuda(cv::InputArray src1, cv::InputArray src2);

POSTINF_DECL cv::cuda::GpuMat logical_and_cuda(cv::InputArray src_mat1, cv::InputArray src_mat2);

POSTINF_DECL cv::cuda::GpuMat logical_not_cuda(cv::InputArray src_mat1 );

POSTINF_DECL cv::cuda::GpuMat nan_to_num(cv::cuda::GpuMat& mat);
POSTINF_DECL cv::cuda::GpuMat zeros_like_cuda(const cv::cuda::GpuMat& src_mat1, int type);
POSTINF_DECL cv::cuda::GpuMat zeros_like_cuda(const cv::cuda::GpuMat& src_mat1);
POSTINF_DECL cv::cuda::GpuMat round(const cv::cuda::GpuMat& mat_float);

/* 输入输出 */
POSTINF_DECL bool imwrite(string file_path, cv::cuda::GpuMat out);

// 取过个波段中的最大值，输出单波段
POSTINF_DECL cv::cuda::GpuMat band_max(const std::vector<cv::cuda::GpuMat>& src_mat_vec);
POSTINF_DECL cv::cuda::GpuMat band_max2(const std::vector<cv::cuda::GpuMat>& src_mat_vec );

}