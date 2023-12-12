/*
 * @Description: Matrix operation interface used to replace numpy based on opencv
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-05-12 12:23:50
 */
#include "rs_gpu.hpp"
#include "mat_gpu.hpp"
#include "utils/os.hpp"
#include "utils/record_time.h"
#include <limits>
#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#if defined(__OPENCV_BUILD) && defined(__clang__)
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif
#if defined(__OPENCV_BUILD) && defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

#include "opencv2/cudalegacy.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/opencv_modules.hpp"
#include "boundSegmentsNPP.hpp"

namespace RSMat {


void connectivityMask(const GpuMat& image, GpuMat& mask, const cv::Scalar& lo, const cv::Scalar& hi, cudaStream_t& stream);
void labelComponents(const GpuMat& mask, GpuMat& components, int flags, cudaStream_t& stream);

//栅格算法
cv::cuda::GpuMat nonmaxsupress_cuda(cv::cuda::GpuMat& Gm, cv::cuda::GpuMat& Gd, int step, float th);  //非最大值抑制算法
cv::cuda::GpuMat remove_small_objects(const cv::cuda::GpuMat& source_img, int area_thr, int connectivity ){
    cv::cuda::GpuMat objects_to_remove = RSMat::zeros_like_cuda(source_img);
    // cv::cuda::GpuMat label_img, mask, centroids;
    cv::cuda::GpuMat npp_label= nppiLabelMarker(source_img);
    RSMat::imwrite( "../data/npp_label.tif", npp_label);
    cout<<"npp_label_size: "<< npp_label.cols<<"  "<< npp_label.rows << endl;

    cv::cuda::GpuMat mask;
    mask.create(source_img.rows, source_img.cols, CV_8UC1);

    cv::cuda::GpuMat components;
    components.create(source_img.rows, source_img.cols, CV_32SC1);

    cudaStream_t s;
    cudaStreamCreate(&s);
    RSMat::connectivityMask(source_img, mask, cv::Scalar::all(0), cv::Scalar::all(2), s);
    RSMat::labelComponents(mask, components, 0, s);

    RSMat::imwrite( "../data/mask.tif", mask);
    cv::cuda::GpuMat h_components(components);
    cout<<"component_size: "<< h_components.cols<<"  "<< h_components.rows << endl;
    // cout<<"component: "<<((uint *)h_components.data)[0] << endl;
    // cout<<"component: "<<((int *)h_components.data)[0] << endl;
    RSMat::imwrite( "../data/components.tif", h_components);


    // cv::cuda::GpuMat areas = stats.col(4);  //[:, 4];
    // for (int i = 1; i < num_labels; i++) {
    //     if (areas.at<int>(i) <= area_thr) objects_to_remove.setTo(1, label_img == i);
    // }
    // cv::cuda::GpuMat objects_to_keep = RSMat::logical_not(objects_to_remove);
    // cv::cuda::GpuMat res = RSMat::logical_and(source_img, objects_to_keep);
    // return res;

    return components;
}


static float4 scalarToCudaType(const cv::Scalar& in)
{
  return make_float4((float)in[0], (float)in[1], (float)in[2], (float)in[3]);
}

void connectivityMask(const GpuMat& image, GpuMat& mask, const cv::Scalar& lo, const cv::Scalar& hi, cudaStream_t& stream)
{
    CV_Assert(!image.empty());

    int ch = image.channels();
    CV_Assert(ch <= 4);

    int depth = image.depth();

    typedef void (*func_t)(const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);

    static const func_t suppotLookup[8][4] =
    {   //    1,    2,     3,     4
        { cv::cuda::device::ccl::computeEdges<uchar>,  0,  cv::cuda::device::ccl::computeEdges<uchar3>,  cv::cuda::device::ccl::computeEdges<uchar4>  },// CV_8U
        { 0,                                 0,  0,                                  0                                  },// CV_16U
        { cv::cuda::device::ccl::computeEdges<ushort>, 0,  cv::cuda::device::ccl::computeEdges<ushort3>, cv::cuda::device::ccl::computeEdges<ushort4> },// CV_8S
        { 0,                                 0,  0,                                  0                                  },// CV_16S
        { cv::cuda::device::ccl::computeEdges<int>,    0,  0,                                  0                                  },// CV_32S
        { cv::cuda::device::ccl::computeEdges<float>,  0,  0,                                  0                                  },// CV_32F
        { 0,                                 0,  0,                                  0                                  },// CV_64F
        { 0,                                 0,  0,                                  0                                  } // CV_16F
    };

    func_t f = suppotLookup[depth][ch - 1];
    CV_Assert(f);

    if (image.size() != mask.size() || mask.type() != CV_8UC1)
        mask.create(image.size(), CV_8UC1);

    // cudaStream_t stream = StreamAccessor::getStream(s);
    float4 culo = scalarToCudaType(lo), cuhi = scalarToCudaType(hi);
    f(image, mask, culo, cuhi, stream);
}

void labelComponents(const GpuMat& mask, GpuMat& components, int flags, cudaStream_t& stream)
{
    CV_Assert(!mask.empty() && mask.type() == CV_8U);

    if (!deviceSupports(SHARED_ATOMICS))
        CV_Error(cv::Error::StsNotImplemented, "The device doesn't support shared atomics and communicative synchronization!");

    components.create(mask.size(), CV_32SC1);

    // cudaStream_t stream = StreamAccessor::getStream(s);
    cv::cuda::device::ccl::labelComponents(mask, components, flags, stream);
}

}  // namespace RSMat
