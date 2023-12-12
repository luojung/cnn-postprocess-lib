/*
 * @Description: Matrix operation interface used to replace numpy based on opencv
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 17:33:08
 */
#include "rs_cpu.hpp"
#include "mat_cpu.hpp"
#include "utils/os.hpp"
#include "utils/record_time.h"

namespace RSMat {

//向量距离计算  余弦距离
double cosine(cv::Mat& a, cv::Mat& b) {
    double dotSum = a.dot(b);        //内积
    double normFirst = cv::norm(a);  //取模
    double normSecond = cv::norm(b);
    if (normFirst == 0 || normSecond == 0) {
        return normFirst == normSecond ? 0 : 1.0;
    }

    if (!(normFirst != 0 && normSecond != 0)) SPDLOG_LOGGER_ERROR(os::get_logger(), "assert error: normFirst != 0 && normSecond != 0");

    return 1 - dotSum / (normFirst * normSecond);

    // std::cout << "cosine fail"   << std::endl;
}
//向量距离计算  余弦距离
double cosine(vector<double> a, vector<double> b) {
    if (a.size() != b.size()) SPDLOG_LOGGER_ERROR(os::get_logger(), "assert error: a.size()==b.size()");

    double a_dot_b = 0, a_ = 0, b_ = 0;
    for (int i = 0; i < a.size(); i++) {
        a_dot_b += a[i] * b[i];
        a_ += a[i] * a[i];
        b_ += b[i] * b[i];
    }
    if (a_ == 0 || b_ == 0) {
        return b_ == b_ ? 0 : 1.0;
    }
    return 1 - a_dot_b / (sqrt(a_) * sqrt(b_));
}
//欧氏距离
float euclidean(cv::Mat& Mat1, cv::Mat& Mat2) { return cv::norm(Mat1, Mat2, cv::NORM_L2); }
float euclidean(cv::Point& p1, cv::Point& p2) {
    cv::Mat Mat1(p1);
    cv::Mat Mat2(p2);
    return cv::norm(Mat1, Mat2, cv::NORM_L2);
}
// sobel梯度
cv::Mat grad_sobel(cv::Mat input) {
    SPDLOG_LOGGER_INFO(os::get_logger(), "grad_sobel...");
    cv::Mat gray, grad_h_src_image, grad_v_src_image;
    cv::cvtColor(input, gray, cv::COLOR_RGB2GRAY);
    cv::Sobel(gray, grad_h_src_image, CV_64F, 0, 1);
    cv::Sobel(gray, grad_v_src_image, CV_64F, 1, 0);
    SPDLOG_LOGGER_DEBUG(os::get_logger(), "input[{},{}]", input.rows, input.cols);

    int rows = grad_h_src_image.rows, cols = grad_h_src_image.cols;
    cv::Mat grad_mat(rows, cols, CV_64F, cv::Scalar(0));

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            double grad =
                sqrt(grad_h_src_image.at<double>(y, x) * grad_h_src_image.at<double>(y, x) + grad_v_src_image.at<double>(y, x) * grad_v_src_image.at<double>(y, x));
            // SPDLOG_LOGGER_INFO(os::get_logger(), "grad[{},{}]{}...",y,x,grad);

            grad_mat.at<double>(y, x) = grad;
            // SPDLOG_LOGGER_INFO(os::get_logger(), "grad[{},{}]{}...",y,x,grad_mat.at<double>(y,x));
        }
    }
    SPDLOG_LOGGER_INFO(os::get_logger(), "grad_sobel done");
    return grad_mat;
}
// Scharr梯度
cv::Mat grad_Scharr(cv::Mat input) {
    SPDLOG_LOGGER_INFO(os::get_logger(), "grad_Scharr...");
    cv::Mat gray, grad_h_src_image, grad_v_src_image;
    cv::cvtColor(input, gray, cv::COLOR_RGB2GRAY);
    // cv::GaussianBlur(gray, gray, cv::Size(5, 5), 3, 3);
    cv::Scharr(gray, grad_h_src_image, CV_64F, 0, 1);
    cv::Scharr(gray, grad_v_src_image, CV_64F, 1, 0);
    SPDLOG_LOGGER_DEBUG(os::get_logger(), "input[{},{}]", input.rows, input.cols);

    int rows = grad_h_src_image.rows, cols = grad_h_src_image.cols;
    cv::Mat grad_mat(rows, cols, CV_64F, cv::Scalar(0));

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            double grad =
                sqrt(grad_h_src_image.at<double>(y, x) * grad_h_src_image.at<double>(y, x) + grad_v_src_image.at<double>(y, x) * grad_v_src_image.at<double>(y, x));
            // SPDLOG_LOGGER_INFO(os::get_logger(), "grad[{},{}]{}...",y,x,grad);

            grad_mat.at<double>(y, x) = grad;
            // SPDLOG_LOGGER_INFO(os::get_logger(), "grad[{},{}]{}...",y,x,grad_mat.at<double>(y,x));
        }
    }
    SPDLOG_LOGGER_INFO(os::get_logger(), "grad_Scharr done");
    return grad_mat;
}

cv::Mat remove_small_holes(cv::Mat& source_img, int area_thr, int connectivity ) {
    // """
    // :param connectivity: connectivity for blob
    // :param source_img:np.bool, 1 for fg, 0 for bg
    // :param area_thr: area thrshold for holes to remove
    // """
    cv::Mat bg_mask = RSMat::logical_not(source_img);
    cv::Mat holes_to_remove = RSMat::zeros_like(source_img);
    cv::Mat label_img, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(bg_mask,  //.astype(np.uint8),
                                                      label_img, stats, centroids, connectivity);
    cv::Mat areas = stats.col(4);  //[:, 4];
    for (int i = 1; i < num_labels; i++) {
        if (areas.at<int>(i) <= area_thr) holes_to_remove.setTo(1, label_img == i);
    }
    cv::Mat res = RSMat::logical_or(source_img, holes_to_remove);
    return res;
}

cv::Mat remove_small_objects(const cv::Mat& source_img, int area_thr, int connectivity ) {
    // """
    // :param source_img: np.bool, 1 for fg, 0 for bg
    // :param area_thr: area thrshold for holes to remove
    // :param connectivity: connectivity for blob
    // """
    cv::Mat objects_to_remove = RSMat::zeros_like(source_img);
    cv::Mat label_img, stats, centroids;

    int num_labels = cv::connectedComponentsWithStats(source_img, label_img, stats, centroids, connectivity);
    cv::Mat areas = stats.col(4);  //[:, 4];
    for (int i = 1; i < num_labels; i++) {
        if (areas.at<int>(i) <= area_thr) objects_to_remove.setTo(1, label_img == i);
    }
    cv::Mat objects_to_keep = RSMat::logical_not(objects_to_remove);
    cv::Mat res = RSMat::logical_and(source_img, objects_to_keep);
    return res;
}

// """
// https://github.com/mubastan/canny/blob/master/canny.py
// """
// ## nonmaximum suppression
// # Gm: gradient magnitudes
// # Gd: gradient directions, -pi/2 to +pi/2
// # return: nms, gradient magnitude if local max, 0 otherwise
template<typename T>
cv::Mat nonmaxsupress(cv::Mat& Gm, cv::Mat& Gd, float th ) {  //非最大值抑制算法
  auto start = std::chrono::system_clock::now();
//   if (Gm.type() != CV_64F || Gd.type() != CV_64F) SPDLOG_LOGGER_WARN(os::get_logger(), "nonmaxsupress only support CV_64F");

  cv::Mat nms = RSMat::zeros_like(Gm);
  int h = Gm.rows, w = Gm.cols;
  for (int x = 1; x < w - 1; x++) {
    for (int y = 1; y < h - 1; y++) {
      auto mag = Gm.at<T>(y, x);
      if (mag < th) continue;
      auto teta = Gd.at<T>(y, x);
      int dx = 0, dy = -1;  //# abs(orient) >= 1.1781, teta < -67.5 degrees and teta > 67.5 degrees
      if (abs(teta) <= 0.3927) {
        dx = 1;
        dy = 0;  //# -22.5 <= teta <= 22.5
      } else if (teta < 1.1781 && teta > 0.3927) {
        dx = 1;
        dy = -1;  //# 22.5 < teta < 67.5 degrees
      } else if (teta > -1.1781 && teta < -0.3927) {
        dx = 1;
        dy = 1;  //# -67.5 < teta < -22.5 degrees
      }
      if (mag > Gm.at<T>(y + dy, x + dx) && mag > Gm.at<T>(y - dy, x - dx) ){
            nms.at<T>(y, x) = mag;

      }
      // auto thr1= Gm.at<double>(y + dy, x + dx), thr2= Gm.at<double>(y - dy, x - dx);
      // if (mag > thr1 && mag > thr2 ){
      //     printf("x:%d y:%d mag: %f  | x:%d y:%d thr1: %f | x:%d y:%d thr2: %f teta:%f\n",x, y, mag, x + dx, y + dy, thr1, x - dx, y - dy, thr2, teta );
      // }else if (mag!=0){
      //     printf("#x:%d y:%d mag: %f | x:%d y:%d thr1: %f | x:%d y:%d thr2: %f teta:%f\n",x, y, mag, x + dx, y + dy, thr1, x - dx, y - dy, thr2, teta );
      // }
    }
  }
  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration<double,std::milli>(end - start);
  cout<<"nonmaxsupress  CPU : "<<elapsed.count()<<"ms"<<endl;
  return nms;
}

cv::Mat nonmaxsupress(cv::Mat& Gm, cv::Mat& Gd, float th ) {  //非最大值抑制算法
    CV_Assert(Gm.type()==Gd.type());
    if (Gm.type() == CV_32F ) {
        return nonmaxsupress<float>(Gm, Gd, th);
    } 
    else if (Gm.type() == CV_64F) {
        return nonmaxsupress<double>(Gm, Gd, th);
    }
}
};  // namespace 
