/*
 * @Description:
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 17:50:09
 */

#include "orientation_util.hpp"


// """
// utils for orientation related
// """

OrientationUtil::OrientationUtil(int num_bin , double max_angle) {
    this->num_bin = num_bin;
    this->max_angle = max_angle;
    this->bin_width = max_angle / num_bin;

    for (int i = 1; i < num_bin + 1; i++) this->label_space_set_valid.push_back(i);

    this->label_space_set_gt.push_back(0);
    this->label_space_set_pred.push_back(255);

    this->repAngle_list = this->label_to_angle(this->label_space_set_valid);
    this->repNormVec_list = this->angle_to_normVec(this->repAngle_list);
}

// """
// angle to tangent vector (counter-clockwise for the outer boundary)
// Args:
//     angle: in radius
// """
vector<cv::Mat> OrientationUtil::angle_to_tangVec(cv::Mat& angle) {
    if (angle.type() != CV_32F && angle.type() != CV_64F) SPDLOG_LOGGER_ERROR(os::get_logger(), "angle_to_tangVec只支持CV_32F  CV_64F !!! ");
    // angle = angle % (Pi * 2); //需要取余吗
    RSMat::angle_remainder(angle, Pi * 2);

    // assert(np.amax(angle) < 2 * Pi, "np.amax(angle) = {}".format(np.amax(angle)));
    // assert(np.amin(angle) >= 0, "np.amin(angle) = {}".format(np.amin(angle)));

    vector<cv::Mat> res;
    angle = angle + Pi / 2.0;
    res.push_back(RSMat::cos(angle));
    res.push_back(RSMat::sin(angle));

    return res;
}

// """
// angle to normal vector (pointing outer)
// Args:
//     angle: in radius
// """
vector<cv::Mat> OrientationUtil::angle_to_normVec(cv::Mat& angle) {
    if (angle.type() != CV_32F && angle.type() != CV_64F) SPDLOG_LOGGER_ERROR(os::get_logger(), "angle_to_tangVec只支持CV_32F  CV_64F !!! ");

    RSMat::angle_remainder(angle, Pi * 2);
    vector<cv::Mat> res;
    res.push_back(RSMat::cos(angle));
    res.push_back(RSMat::sin(angle));

    return res;
}

vector<cv::cuda::GpuMat> OrientationUtil::angle_to_normVec(cv::cuda::GpuMat& angle) {
    if (angle.type() != CV_32F && angle.type() != CV_64F) SPDLOG_LOGGER_ERROR(os::get_logger(), "angle_to_tangVec只支持CV_32F  CV_64F !!! ");

    RSMat::angle_remainder(angle, Pi * 2);
    vector<cv::cuda::GpuMat> res;
    res.push_back(RSMat::cos_cuda(angle));
    res.push_back(RSMat::sin_cuda(angle));

    return res;
}

vector<vector<float>> OrientationUtil::angle_to_normVec(vector<float>& angle) {
    RSMat::angle_remainder(angle, Pi * 2);
    vector<vector<float>> res;
    res.push_back(RSMat::vec_cos(angle));
    res.push_back(RSMat::vec_sin(angle));

    return res;
}
// """
// angle to label
// Args:
//     angle: in radius
// """
cv::Mat OrientationUtil::angle_to_label(cv::Mat& angle) {
    if (angle.type() != CV_32F && angle.type() != CV_64F) SPDLOG_LOGGER_ERROR(os::get_logger(), "angle_to_tangVec只支持CV_32F  CV_64F !!! ");
    RSMat::angle_remainder(angle, Pi * 2);

    cv::Mat label = angle / this->bin_width;
    label = RSMat::round(label) + 1;

    if (!(RSMat::amax(label) <= this->num_bin + 1)) {
        SPDLOG_LOGGER_WARN(os::get_logger(), " amax(blurred_label) = {}", RSMat::amax(label));
    }
    label.setTo(1, label >= (this->num_bin + 1));

    return label;
}

cv::cuda::GpuMat OrientationUtil::angle_to_label(cv::cuda::GpuMat& angle) {
    if (angle.type() != CV_32F && angle.type() != CV_64F) SPDLOG_LOGGER_ERROR(os::get_logger(), "angle_to_tangVec只支持CV_32F  CV_64F !!! ");
    RSMat::angle_remainder(angle, Pi * 2);

    cv::cuda::GpuMat label;
    cv::cuda::divide(angle, cv::Scalar(this->bin_width), label);
    label = RSMat::round(label);
    cv::cuda::add(label, cv::Scalar(1), label);

    if (!(RSMat::max_cuda(label) <= this->num_bin + 1)) {
        SPDLOG_LOGGER_WARN(os::get_logger(), " amax(blurred_label) = {}", RSMat::max_cuda(label));
    }
    label.setTo(cv::Scalar(1), RSMat::ge_cuda(label , cv::Scalar(this->num_bin + 1) ));

    return label;
}

// """
// label to angle
// """
vector<float> OrientationUtil::label_to_angle(vector<int>& label) {
    vector<float> angle;
    for (auto a : label) {
        float res = (a - 1) * this->bin_width;
        if (a == 0 || a == 255) res = CV_NAN;
        angle.push_back(res);
    }

    return angle;
}
cv::Mat OrientationUtil::label_to_angle(const cv::Mat& label) {
    if (label.rows == 0 || label.cols == 0) return cv::Mat();
    cv::Mat angle;
    cv::Mat label2 = label - 1;

    label2.convertTo(angle, CV_64F);
    angle = angle * this->bin_width;

    angle.setTo(CV_NAN, label == 0);
    angle.setTo(CV_NAN, label == 255);
    return angle;
}

cv::Mat OrientationUtil::label_to_angle_32F(const cv::Mat& label) {
    if (label.rows == 0 || label.cols == 0) return cv::Mat();
    cv::Mat angle;
    cv::Mat label2 = label - 1;

    label2.convertTo(angle, CV_32F);
    angle = angle * this->bin_width;

    angle.setTo(CV_NAN, label == 0);
    angle.setTo(CV_NAN, label == 255);
    return angle;
}

cv::cuda::GpuMat OrientationUtil::label_to_angle(const cv::cuda::GpuMat& label) {
    if (label.rows == 0 || label.cols == 0) return cv::cuda::GpuMat();
    cv::cuda::GpuMat label2;
    cv::cuda::subtract(label, cv::Scalar(1), label2);

    cv::cuda::GpuMat angle;
    label2.convertTo(angle, CV_64F);
    cv::cuda::multiply(angle, cv::Scalar(this->bin_width), angle);

    angle.setTo(cv::Scalar(CV_NAN), RSMat::equal_cuda(label, cv::Scalar(1)));
    angle.setTo(cv::Scalar(CV_NAN), RSMat::equal_cuda(label, cv::Scalar(255)));
    return angle;
}
cv::cuda::GpuMat OrientationUtil::label_to_angle_32F(const cv::cuda::GpuMat& label) {
    if (label.rows == 0 || label.cols == 0) return cv::cuda::GpuMat();
    cv::cuda::GpuMat label2;
    cv::cuda::subtract(label, cv::Scalar(1), label2);

    cv::cuda::GpuMat angle;
    label2.convertTo(angle, CV_32F);
    cv::cuda::multiply(angle, cv::Scalar(this->bin_width), angle);

    angle.setTo(cv::Scalar(CV_NAN), RSMat::equal_cuda(label, cv::Scalar(1)));
    angle.setTo(cv::Scalar(CV_NAN), RSMat::equal_cuda(label, cv::Scalar(255)));
    return angle;
}

//  """
//         label map to rgb color image, for display
//         """
cv::Mat OrientationUtil::label_to_color(cv::Mat& label) {
    int H = label.rows, W = label.cols;

    cv::Mat angle;
    angle = label_to_angle(label);

    cv::Mat color1 = (angle / max_angle) * 180;
    color1.convertTo(color1, CV_8U);
    cv::Mat color2(H, W, CV_8U, cv::Scalar(255));

    cv::Mat color3;
    cv::Mat label1 = (label == 0);
    RSMat::logical_not(label1).convertTo(color3, CV_32F);
    color3 = color3 / 2;
    color3.convertTo(color3, CV_8U);

//  """
//         label map to rgb color image, for display
//         """
    cv::Mat out;
    std::vector<cv::Mat> color_vec = {color1, color2, color3};
    cv::merge(color_vec, out);
    cv::cvtColor(out, out, cv::COLOR_HSV2RGB);

    std::vector<cv::Mat> RGB_vec;
    cv::split(out, RGB_vec);
    RGB_vec[0].setTo(255, label == 255);
    RGB_vec[1].setTo(255, label == 255);
    RGB_vec[2].setTo(255, label == 255);
    cv::merge(RGB_vec, out);

    return out;
}

void OrientationUtil::get_rep_norm_vec_map_HW(int H, int W, vector<cv::Mat>& rep_norm_vec_map_h, vector<cv::Mat>& rep_norm_vec_map_v){
    for (int b = 0; b < num_bin; b++) {
        cv::Mat rep_norm_h(H, W, CV_32F, this->repNormVec_list[0][b]);
        cv::Mat rep_norm_v(H, W, CV_32F, this->repNormVec_list[1][b]);

        rep_norm_vec_map_h.push_back(rep_norm_h);
        rep_norm_vec_map_v.push_back(rep_norm_v);
    }
}
