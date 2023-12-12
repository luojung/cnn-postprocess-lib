/*
 * @Description: Functions that implement some of the same functionality found in Matlab"s bwmorph.
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 11:02:41
 *  https://gist.github.com/bmabey/4dd36d9938b83742a88b6f68ac1901a6
 */
#include "bwmorph.hpp"

namespace RSMat {
cv::Mat _neighbors_conv(cv::Mat& image) {
    cv::Mat image2;
    image.convertTo(image2, CV_8U);
    image2.setTo(1, image2 >= 1);
    image = image2;

    unsigned char temp[3][3] = {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    cv::Mat kernel1 = cv::Mat(3, 3, CV_8UC1, temp);

    cv::Mat neighborhood_count;
    cv::filter2D(image, neighborhood_count, image.depth(), kernel1, cv::Point(1, 1), 0, cv::BORDER_CONSTANT);
    neighborhood_count.setTo(0, RSMat::logical_not(image));

    return neighborhood_count;
}

cv::Mat _neighbors_conv(const cv::Mat& image, const cv::Mat& kernel) {
    cv::Mat image2 = RSMat::zeros_like(image);
    image2.setTo(1, image >= 1);

    cv::Mat neighborhood_count;
    cv::filter2D(image2, neighborhood_count, image2.depth(), kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    neighborhood_count.setTo(0, RSMat::logical_not(image2));
    return neighborhood_count;
}

/**
 * @description: get endpoints in an image
 * @param  { image : binary (M, N) ndarray}
 * @return {the endpoints in an image}
 */
cv::Mat endpoints(cv::Mat& image) { return _neighbors_conv(image) == 1; }

cv::Mat get_neighbors(const cv::Mat& image, const cv::Mat& neighbor_kernel) { return _neighbors_conv(image, neighbor_kernel); }
}