/*
 * @Description: Matrix operation interface used to replace numpy based on opencv
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 17:43:17
 */

#pragma once
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include "utils/macro.h"

using namespace std;

#define CV_NAN 0

namespace RSMat {

// mat 角度计算
POSTINF_DECL cv::Mat arctan2(cv::Mat& angle_v, cv::Mat& angle_h);

POSTINF_DECL cv::Mat sin(cv::Mat& a);

POSTINF_DECL vector<float> vec_sin(vector<float>& a);

POSTINF_DECL cv::Mat cos(cv::Mat& a);

POSTINF_DECL vector<float> vec_cos(vector<float>& a);

//方向向量的单位化(L2范数 )    np.linalg.norm
POSTINF_DECL double direction_angle_linalg_norm(std::pair<float, float>& direction);
//方向向量的单位化(L2范数 )    np.linalg.norm
POSTINF_DECL int linalg_norm(cv::Mat& Mat1);

//只支持CV_32FC1
POSTINF_DECL void angle_remainder(cv::Mat& angle_mat, float remainder);

POSTINF_DECL void angle_remainder(vector<float>& angle_vec, float remainder);

};  // namespace 
