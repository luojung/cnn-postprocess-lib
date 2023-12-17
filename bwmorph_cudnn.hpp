/*
 * @Description: Functions that implement some of the same functionality found in Matlab"s bwmorph.
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-05-12 16:52:05
 *  https://gist.github.com/bmabey/4dd36d9938b83742a88b6f68ac1901a6
 */

#pragma once

#include <iostream>
// #include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "RSMat.h"

using namespace std;


namespace RSMat{

void filter2D( cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int ddepth, InputArray kernel);

// """
// Counts the neighbor pixels for each pixel of an image:
//         x = [
//             [0, 1, 0],
//             [1, 1, 1],
//             [0, 1, 0]
//         ]
//         _neighbors(x)
//         [
//             [0, 3, 0],
//             [3, 4, 3],
//             [0, 3, 0]
//         ]
// :type image: numpy.ndarray
// :param image: A two-or-three dimensional image
// :return: neighbor pixels for each pixel of an image
// """
cv::cuda::GpuMat _neighbors_conv( cv::cuda::GpuMat image);

// """
// Counts the neighbor pixels for each pixel of an image:
//         x = [
//             [0, 1, 0],
//             [1, 1, 1],
//             [0, 1, 0]
//         ]
//         _neighbors(x)
//         [
//             [0, 3, 0],
//             [3, 4, 3],
//             [0, 3, 0]
//         ]
// :type image: numpy.ndarray
// :param image: A two-or-three dimensional image
// :return: neighbor pixels for each pixel of an image
// """
cv::cuda::GpuMat _neighbors_conv(const cv::cuda::GpuMat& image, const cv::cuda::GpuMat& kernel);

cv::cuda::GpuMat convolution( const cv::cuda::GpuMat input, const cv::Mat h_kernel);

/**
 * @description: get endpoints in an image
 * @param  { image : binary (M, N) ndarray}
 * @return {the endpoints in an image}
 */
POSTINF_DECL cv::cuda::GpuMat endpoints( cv::cuda::GpuMat image);

POSTINF_DECL cv::cuda::GpuMat get_neighbors(const cv::cuda::GpuMat& image, const cv::cuda::GpuMat& neighbor_kernel);

POSTINF_DECL unsigned int count_nonZero(cv::cuda::GpuMat bw_tip);   //非零值数量

POSTINF_DECL cv::cuda::GpuMat find_tip_coords(cv::cuda::GpuMat edge_rm_thin);//获取骨架线的端点坐标

POSTINF_DECL cv::cuda::GpuMat stitch_line(cv::cuda::GpuMat tip_coords, cv::cuda::GpuMat edge_rm_thin, int nbh_thr=5, int ext_thr=20);
} // namespace RSMat
