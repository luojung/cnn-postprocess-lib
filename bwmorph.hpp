/*
 * @Description: Functions that implement some of the same functionality found
 * in Matlab"s bwmorph.
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 11:02:41
 *  https://gist.github.com/bmabey/4dd36d9938b83742a88b6f68ac1901a6
 */

#pragma once

#include "RSMat.h"
#include "utils/macro.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
namespace RSMat {

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
POSTINF_DECL cv::Mat _neighbors_conv(cv::Mat &image);

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
POSTINF_DECL cv::Mat _neighbors_conv(const cv::Mat &image,
                                     const cv::Mat &kernel);

/**
 * @description: get endpoints in an image
 * @param  { image : binary (M, N) ndarray}
 * @return {the endpoints in an image}
 */
POSTINF_DECL cv::Mat endpoints(cv::Mat &image);

POSTINF_DECL cv::Mat get_neighbors(const cv::Mat &image,
                                   const cv::Mat &neighbor_kernel);

} // namespace RSMat