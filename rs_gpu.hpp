/*
 * @Description: Matrix operation interface used to replace numpy based on opencv
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-28 10:10:08
 */

#pragma once

#include <math.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/cudalegacy.hpp"
#include "utils/macro.h"
using namespace std;
using namespace cv;


using namespace cv::cuda;

   namespace cv { namespace cuda { namespace device{ namespace ccl
    {
        void labelComponents(const PtrStepSzb& edges, PtrStepSzi comps, int flags, cudaStream_t stream);

        template<typename T>
        void computeEdges(const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);
    }
   }}}

namespace RSMat {

// **********基于cuda开发的接口***********
POSTINF_DECL  cv::cuda::GpuMat argmax_cuda(const vector<cv::cuda::GpuMat>& src_mat_vec);

POSTINF_DECL  float max_cuda(const cv::cuda::GpuMat& src_mat);

POSTINF_DECL  float min_cuda(const cv::cuda::GpuMat& src_mat);

POSTINF_DECL  void connectivityMask(const GpuMat& image, GpuMat& mask, const cv::Scalar& lo, const cv::Scalar& hi, cudaStream_t& stream);
POSTINF_DECL  void labelComponents(const GpuMat& mask, GpuMat& components, int flags, cudaStream_t& stream);

//栅格算法
POSTINF_DECL  cv::cuda::GpuMat nonmaxsupress_cuda(cv::cuda::GpuMat& Gm, cv::cuda::GpuMat& Gd, float th = 1.0);  //非最大值抑制算法
POSTINF_DECL  cv::cuda::GpuMat remove_small_objects(const cv::cuda::GpuMat& source_img, int area_thr, int connectivity = 8);


POSTINF_DECL  static float4 scalarToCudaType(const cv::Scalar& in);

POSTINF_DECL  void connectivityMask(const GpuMat& image, GpuMat& mask, const cv::Scalar& lo, const cv::Scalar& hi, cudaStream_t& stream);

POSTINF_DECL  void labelComponents(const GpuMat& mask, GpuMat& components, int flags, cudaStream_t& stream);

}  // namespace RSMat
