/*
 * @Description: Matrix operation interface used to replace numpy based on opencv
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 17:33:08
 */

#pragma once

#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include "utils/macro.h"

using namespace std;


namespace RSMat {

//向量距离计算  余弦距离
POSTINF_DECL double cosine(cv::Mat& a, cv::Mat& b);

//向量距离计算  余弦距离
POSTINF_DECL double cosine(vector<double> a, vector<double> b);

//欧氏距离
POSTINF_DECL float euclidean(cv::Mat& Mat1, cv::Mat& Mat2);
POSTINF_DECL float euclidean(cv::Point& p1, cv::Point& p2);

// sobel梯度
POSTINF_DECL cv::Mat grad_sobel(cv::Mat input);
// Scharr梯度
POSTINF_DECL cv::Mat grad_Scharr(cv::Mat input);

POSTINF_DECL cv::Mat remove_small_holes(cv::Mat& source_img, int area_thr, int connectivity = 8);

POSTINF_DECL cv::Mat remove_small_objects(const cv::Mat& source_img, int area_thr, int connectivity = 8);

// """
// https://github.com/mubastan/canny/blob/master/canny.py
// """
// ## nonmaximum suppression
// # Gm: gradient magnitudes
// # Gd: gradient directions, -pi/2 to +pi/2
// # return: nms, gradient magnitude if local max, 0 otherwise
template<typename T>
POSTINF_DECL cv::Mat nonmaxsupress(cv::Mat& Gm, cv::Mat& Gd, float th = 1.0);

POSTINF_DECL cv::Mat nonmaxsupress(cv::Mat& Gm, cv::Mat& Gd, float th = 1.0);
};  // namespace 
