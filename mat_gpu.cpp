
#include "mat_gpu.hpp"
#include "mat_cpu.hpp"
#include "utils/os.hpp"
#include "utils/record_time.h"
#include <opencv2/core/types.hpp>
namespace RSMat {

// **********基于opencv cuda 封装的接口***********
cv::cuda::GpuMat equal_cuda(cv::InputArray src1, cv::InputArray src2) {
    cv::cuda::GpuMat d_res3;
    cv::cuda::compare(src1, src2, d_res3, cv::CMP_EQ);
    return d_res3;
}

cv::cuda::GpuMat greater_cuda(cv::InputArray src1, cv::InputArray src2) {
    cv::cuda::GpuMat d_res3;
    cv::cuda::compare(src1, src2, d_res3, cv::CMP_GT);
    return d_res3;
}

cv::cuda::GpuMat ge_cuda(cv::InputArray src1, cv::InputArray src2) {
    cv::cuda::GpuMat d_res3;
    cv::cuda::compare(src1, src2, d_res3, cv::CMP_GE);
    return d_res3;
}

cv::cuda::GpuMat logical_or_cuda(cv::InputArray src1, cv::InputArray src2) {
    cv::cuda::GpuMat d_res3;
    cv::cuda::bitwise_or(src1, src2, d_res3);
    return d_res3;
}

cv::cuda::GpuMat logical_and_cuda(cv::InputArray src_mat1, cv::InputArray src_mat2) {
    cv::cuda::GpuMat src_mat1_logic;
    cv::cuda::compare(src_mat1, cv::Scalar(0), src_mat1_logic, cv::CMP_GT);
    
    cv::cuda::GpuMat d_res3;
    cv::cuda::bitwise_and(src_mat1_logic, src_mat2, d_res3);
    return d_res3;
}

cv::cuda::GpuMat logical_not_cuda(cv::InputArray src_mat1 ) {
    cv::cuda::GpuMat src_mat1_logic;
    cv::cuda::compare(src_mat1, cv::Scalar(0), src_mat1_logic, cv::CMP_GT);
    
    cv::cuda::GpuMat d_res3;
    cv::cuda::bitwise_not(src_mat1_logic, d_res3);
    return d_res3;
}

cv::cuda::GpuMat nan_to_num(cv::cuda::GpuMat& mat) { return mat.setTo(0, equal_cuda(mat, cv::Scalar(CV_NAN))); }
cv::cuda::GpuMat zeros_like_cuda(const cv::cuda::GpuMat& src_mat1, int type) { return cv::cuda::GpuMat(src_mat1.rows, src_mat1.cols, type, cv::Scalar(0)); }
cv::cuda::GpuMat zeros_like_cuda(const cv::cuda::GpuMat& src_mat1) { return cv::cuda::GpuMat(src_mat1.rows, src_mat1.cols, src_mat1.type(), cv::Scalar(0)); }
cv::cuda::GpuMat round(const cv::cuda::GpuMat& mat_float) {
    cv::cuda::GpuMat mat_tmp;
    cv::cuda::add(mat_float, cv::Scalar( 0.5), mat_tmp);
    cv::cuda::GpuMat mat_8u;
    mat_tmp.convertTo(mat_8u, CV_8U);
    return mat_8u;
};

/* 输入输出 */
bool imwrite(string file_path, cv::cuda::GpuMat out) {
    cv::Mat h_out(out);
    return RSMat::imwrite(file_path, h_out);
}

// 取过个波段中的最大值，输出单波段
cv::cuda::GpuMat band_max(const std::vector<cv::cuda::GpuMat>& src_mat_vec);
cv::cuda::GpuMat band_max2(const std::vector<cv::cuda::GpuMat>& src_mat_vec );

}