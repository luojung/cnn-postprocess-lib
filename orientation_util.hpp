/*
 * @Description:
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 17:50:09
 */

#pragma once
#include <math.h>
#include <string>
#include <vector>
#include "utils/macro.h"

#include "RSMat.h"

#define Pi 3.14159265358979323846

using namespace std;


// """
// utils for orientation related
// """
 POSTINF_DECL class OrientationUtil {
  private:
    int num_bin;
    int max_angle;
    double bin_width;
    vector<int> label_space_set_valid;
    vector<int> label_space_set_gt;
    vector<int> label_space_set_pred;
    vector<float> repAngle_list;
    vector<vector<float>> repNormVec_list;

  public:
    OrientationUtil(int num_bin = 36, double max_angle = 2 * Pi);

    // """
    // angle to tangent vector (counter-clockwise for the outer boundary)
    // Args:
    //     angle: in radius
    // """
    vector<cv::Mat> angle_to_tangVec(cv::Mat& angle);

    // """
    // angle to normal vector (pointing outer)
    // Args:
    //     angle: in radius
    // """
    vector<cv::Mat> angle_to_normVec(cv::Mat& angle);

    vector<cv::cuda::GpuMat> angle_to_normVec(cv::cuda::GpuMat& angle);

    vector<vector<float>> angle_to_normVec(vector<float>& angle) ;
    // """
    // angle to label
    // Args:
    //     angle: in radius
    // """
    cv::Mat angle_to_label(cv::Mat& angle);

    cv::cuda::GpuMat angle_to_label(cv::cuda::GpuMat& angle);
    // """
    // label to angle
    // """
    vector<float> label_to_angle(vector<int>& label);

    cv::Mat label_to_angle(const cv::Mat& label);

    cv::Mat label_to_angle_32F(const cv::Mat& label) ;

    cv::cuda::GpuMat label_to_angle(const cv::cuda::GpuMat& label);

    cv::cuda::GpuMat label_to_angle_32F(const cv::cuda::GpuMat& label);

    //  """
    //         label map to rgb color image, for display
    //         """
    cv::Mat label_to_color(cv::Mat& label);

    void get_rep_norm_vec_map_HW(int H, int W, vector<cv::Mat>& rep_norm_vec_map_h, vector<cv::Mat>& rep_norm_vec_map_v);
};
