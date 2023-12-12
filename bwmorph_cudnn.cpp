/*
 * @Description: Functions that implement some of the same functionality found in Matlab"s bwmorph.
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-05-12 16:47:51
 *  https://gist.github.com/bmabey/4dd36d9938b83742a88b6f68ac1901a6
 */

#include "bwmorph_cudnn.hpp"
#include <cudnn.h>
#include <cudnn_v8.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include "mat_gpu.hpp"
#include "rs_gpu.hpp"
#include "postprocesslib/mat_gpu.hpp"

namespace RSMat{

//基于cudnn 封装卷积接口
// void filter2D( cv::cuda::GpuMat input, cv::cuda::GpuMat& output, int ddepth, InputArray kernel){
//     cudaSetDevice(0);
//     //handle
//     cudnnHandle_t handle;
//     cudnnCreate(&handle);

//     // input
//     // Tensor<float> input({ 1, src.channels(), src.rows, src.cols });
//     // Memory::copy(image_float.count() * sizeof(float), input.gptr(), src.data);

//     cudnnTensorDescriptor_t input_descriptor;
//     cudnnCreateTensorDescriptor(&input_descriptor);
//     cudnnSetTensor4dDescriptor(input_descriptor,
//                                CUDNN_TENSOR_NHWC,
//                                CUDNN_DATA_UINT8,
//                                1, 1, input.rows, input.cols);

//     // output
//     // Tensor<float> output(input.shape());
//     // vector_set_gpu(output.count(), 0.0f, output.gptr());
//     output= RSMat::zeros_like_cuda(input);

//     cudnnTensorDescriptor_t output_descriptor;
//     cudnnCreateTensorDescriptor(&output_descriptor);
//     cudnnSetTensor4dDescriptor(output_descriptor,
//                                CUDNN_TENSOR_NHWC,
//                                CUDNN_DATA_UINT8,
//                                1, 1, input.rows, input.cols);

//     // kernel
//     // Tensor<float> kernel({ output.shape(1), input.shape(1), 3, 3 });
//     // auto kernel_size = kernel.count(2, 4);
//     // float kernel_[kernel_size] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
//     // for(auto i = 0; i < kernel.count(0, 2); ++i) {
//     //     memcpy(kernel.cptr() + i * kernel_size, kernel_, kernel_size * sizeof(float));
//     // }

//     cudnnFilterDescriptor_t kernel_descriptor;
//     cudnnCreateFilterDescriptor(&kernel_descriptor);
//     cudnnSetFilter4dDescriptor(kernel_descriptor,
//                                CUDNN_DATA_UINT8,
//                                CUDNN_TENSOR_NCHW,
//                                1, 1, input.rows, input.cols);
//     // convolution descriptor
//     cudnnConvolutionDescriptor_t conv_descriptor;
//     cudnnCreateConvolutionDescriptor(&conv_descriptor);
//     cudnnSetConvolution2dDescriptor(conv_descriptor,
//                                     1, 1, // zero-padding
//                                     1, 1, // stride
//                                     1, 1,
//                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

//     // algorithm
//     cudnnConvolutionFwdAlgoPerf_t algo;
//     cudnnGetConvolutionForwardAlgorithm_v7(handle,
//                                         input_descriptor,
//                                         kernel_descriptor,
//                                         conv_descriptor,
//                                         output_descriptor,
//                                         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
//                                         0,
//                                         &algo);

//     // workspace size && allocate memory
//     size_t workspace_size = 0;
//     cudnnGetConvolutionForwardWorkspaceSize(handle,
//                                             input_descriptor,
//                                             kernel_descriptor,
//                                             conv_descriptor,
//                                             output_descriptor,
//                                             algo,
//                                             &workspace_size);

//     void * workspace = nullptr;
//     cudaMalloc(&workspace, workspace_size);

//     // convolution
//     auto alpha = 1.0f, beta = 0.0f;
//     cudnnConvolutionForward(handle,
//                             &alpha, input_descriptor, input.data,
//                             kernel_descriptor, kernel.data,
//                             conv_descriptor, algo,
//                             workspace, workspace_size,
//                             &beta, output_descriptor, output.data);

//     // Matrix32f output_image(image.shape());
//     // cudaMemcpy(output_image.ptr(), output.data, image.count() * sizeof(float), cudaMemcpyDeviceToHost);

//     // destroy
//     cudaFree(workspace);

//     cudnnDestroyTensorDescriptor(input_descriptor);
//     cudnnDestroyTensorDescriptor(output_descriptor);
//     cudnnDestroyConvolutionDescriptor(conv_descriptor);
//     cudnnDestroyFilterDescriptor(kernel_descriptor);

//     cudnnDestroy(handle);

// }

//基于DNN 封装卷积接口，由于环境文件，未能编译成功
// cv::cuda::GpuMat convolution(cv::cuda::GpuMat input, int * kernel, int kernel_size){
//      cv::dnn::LayerParams params;
//      cv::cuda::GpuMat kernel_mat(kernel_size, kernel_size, CV_32F, kernel);
//      cv::cuda::GpuMat output;

//      cv::Ptr<cv::cuda::Convolution> conv= cv::cuda::createConvolution(cv::Size(kernel_size, kernel_size));
//      conv->convolve(input, kernel_mat, output);
//      return output;
// }

// cv::cuda::GpuMat convolution(cv::cuda::GpuMat input, cv::cuda::GpuMat kernel_mat){
//      cv::cuda::GpuMat output;

//      cv::Ptr<cv::cuda::Convolution> conv= cv::cuda::createConvolution(cv::Size(kernel_mat.rows, kernel_mat.cols));
//      conv->convolve(input, kernel_mat, output);
//      return output;
// }


cv::cuda::GpuMat _neighbors_conv(cv::cuda::GpuMat image) {
    cv::cuda::GpuMat image2;
    image.convertTo(image2, CV_8U);

    cv::cuda::GpuMat mask=RSMat::greater_cuda(image2, cv::Scalar(0));
    image2.setTo(1, mask);
    image = image2;

    unsigned char temp[9] = {1, 1, 1, 1, 0, 1, 1, 1, 1};
    // unsigned char temp[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    cv::Mat h_kernel = cv::Mat(3, 3, image2.type(), (void*)temp);

    cv::cuda::GpuMat neighborhood_count;
    neighborhood_count = RSMat::convolution(image2, h_kernel);
    std::cout<<"neighborhood_count"<<std::endl;

    // cv::filter2D(image, neighborhood_count, image.depth(), kernel1, cv::Point(1, 1), 0, cv::BORDER_CONSTANT);
    neighborhood_count.setTo(0, RSMat::logical_not_cuda(image));

    return neighborhood_count;
}


cv::cuda::GpuMat _neighbors_conv(const cv::cuda::GpuMat& image, const cv::Mat kernel) {
    cv::cuda::GpuMat image2 = RSMat::zeros_like_cuda(image);
    cv::cuda::GpuMat mask=RSMat::greater_cuda(image2, cv::Scalar(0));
    image2.setTo(1, mask);
    cv::cuda::GpuMat neighborhood_count;

    neighborhood_count = RSMat::convolution(image2, kernel);
    // cv::filter2D(image2, neighborhood_count, image2.depth(), kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    neighborhood_count.setTo(0, RSMat::logical_not_cuda(image2));
    return neighborhood_count;
}

/**
 * @description: get endpoints in an image
 * @param  { image : binary (M, N) ndarray}
 * @return {the endpoints in an image}
 */
cv::cuda::GpuMat endpoints(cv::cuda::GpuMat image) { return RSMat::equal_cuda(_neighbors_conv(image), cv::Scalar(1)); }

cv::cuda::GpuMat get_neighbors(const cv::cuda::GpuMat& image, const cv::Mat neighbor_kernel) { return _neighbors_conv(image, neighbor_kernel); }

} // namespace RSMat
