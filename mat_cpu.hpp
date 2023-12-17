/*
 * @Description: Matrix operation interface used to replace numpy based on opencv
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 17:40:30
 */

#pragma once
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include "utils/macro.h"

using namespace std;

#define CV_NAN 0

// POSTINF_DECL cv::Mat argmax(vector<cv::Mat>& src_mat_vec);

namespace RSMat {

POSTINF_DECL cv::Mat round(const cv::Mat& mat_float);

//求坐标的x轴、y轴 或 x和y的平均值
// axis: -1(x和y的平均值)，0(x轴的平均值)，1（y轴的平均值）
POSTINF_DECL double mean(vector<cv::Point> locations, int axis = -1);
// 取过个波段中的最大值，输出单波段
POSTINF_DECL cv::Mat band_max(const vector<cv::Mat>& src_mat_vec);

// np.argmax
POSTINF_DECL cv::Mat argmax( vector<cv::Mat>& src_mat_vec);

//指定数值范围进行截断，如0-255
POSTINF_DECL int intercept(cv::Mat& src_mat, int max, int min);

// 2维数组求最大值
template <typename T>
T amax(const cv::Mat& src_mat);
POSTINF_DECL float amax(const cv::Mat& src_mat);
// template<>  POSTINF_DECL unsigned char amax<unsigned char>(const cv::Mat& src_mat);
// template<>  POSTINF_DECL float amax<float>(const cv::Mat& src_mat);
// template<>  POSTINF_DECL double amax<double>(const cv::Mat& src_mat);

// 2维8u求最小值
template <typename T>
T amin(const cv::Mat& src_mat);
POSTINF_DECL float amin(const cv::Mat& src_mat);

// mat 与或
POSTINF_DECL cv::Mat logical_or(const cv::Mat& src_mat1, const cv::Mat& src_mat2);
POSTINF_DECL cv::Mat logical_and(const cv::Mat& src_mat1, const cv::Mat& src_mat2);
POSTINF_DECL cv::Mat logical_and_(cv::Mat src_mat1, cv::Mat src_mat2);
POSTINF_DECL cv::Mat logical_not(const cv::Mat& src_mat1);
POSTINF_DECL cv::Mat logical(const cv::Mat& src_mat1);
// mat 异或
POSTINF_DECL cv::Mat logical_xor(const cv::Mat& src_mat1, const cv::Mat& src_mat2);
// mat 异或
POSTINF_DECL cv::Mat logical_xor_(cv::Mat src_mat1, cv::Mat src_mat2);

POSTINF_DECL set<int> unique(const cv::Mat& src_mat);


//// <uchar>--CV_8U// <char>---CV_8S// <short>-----CV_16S
// <ushort>---CV_16U// <int>---CV_32S// <float>--CV_32F
// <double>----CV_64F//  -   CV_8UC3
POSTINF_DECL cv::Mat zeros_like(const cv::Mat src_mat1, int type);
POSTINF_DECL cv::Mat zeros_like(const cv::Mat src_mat1);

POSTINF_DECL unsigned int count_nonzero(cv::Mat mat);

POSTINF_DECL cv::Mat nan_to_num(cv::Mat& mat);

/* * write_img- */
template<typename T>
POSTINF_DECL bool imwrite(string file_path, cv::Mat out);

POSTINF_DECL bool imwrite(string file_path, cv::Mat out);

/* *read TIFF-*/
POSTINF_DECL int imread_tiff(string file_path_name, vector<cv::Mat>& out_mats);

/* *  read_ _C1--png */
POSTINF_DECL cv::Mat imread_C1(string file_path_name);

// calculate similarity rate
POSTINF_DECL float similarity(const cv::Mat mat1, string path_mat2);

POSTINF_DECL bool is_equal(const cv::Mat mat1, const cv::Mat mat2, float thr = 0.95, string mat_name ="");
// thr,阈值，如0.9为90%相等则返回true
POSTINF_DECL bool is_equal(string path_mat1, const cv::Mat mat2,  float thr = 0.99, string mat_name ="");
POSTINF_DECL bool is_equal(string path_mat1, string path_mat2, float thr = 0.99, string mat_name ="");
POSTINF_DECL bool is_equal(string path_mat1, const cv::cuda::GpuMat mat2,  float thr = 0.99, string mat_name ="");

POSTINF_DECL bool compare_mat(cv::Mat a, cv::cuda::GpuMat b, string fn="");

};  // namespace 
