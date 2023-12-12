/*
 * @Description: Matrix operation interface used to replace numpy based on opencv
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-04-27 17:43:17
 */

#include "angle_cpu.hpp"
#include "utils/os.hpp"
#include "utils/record_time.h"

// namespace RSMat {

// mat 角度计算
cv::Mat RSMat::arctan2(cv::Mat& angle_v, cv::Mat& angle_h) {
    TIME_RECORD(arctan2);
    unsigned int rows = angle_v.rows;
    unsigned int cols = angle_v.cols;
    cv::Mat out(rows, cols, angle_v.type());

    if (angle_v.type() == CV_32F) {
        for (unsigned int i = 0; i < rows; i++) {
            float* ptra = (float*)(angle_v.data + i * angle_v.step);
            float* ptrb = (float*)(angle_h.data + i * angle_h.step);
            float* ptrout = (float*)(out.data + i * out.step);
            for (unsigned int j = 0; j < cols; j++) {
                *ptrout = atan2(*ptra, *ptrb);
                ptra++;
                ptrb++;
                ptrout++;
            }
        }
    } else if (angle_v.type() == CV_64F) {
        for (unsigned int i = 0; i < rows; i++) {
            double* ptra = (double*)(angle_v.data + i * angle_v.step);
            double* ptrb = (double*)(angle_h.data + i * angle_h.step);
            double* ptrout = (double*)(out.data + i * out.step);
            for (unsigned int j = 0; j < cols; j++) {
                *ptrout = atan2(*ptra, *ptrb);
                ptra++;
                ptrb++;
                ptrout++;
            }
        }
    }
    TIME_ELAPSED(arctan2);
    return out;
}

cv::Mat RSMat::sin(cv::Mat& a) {
    TIME_RECORD(sin);
    unsigned int rows = a.rows;
    unsigned int cols = a.cols;
    cv::Mat out(rows, cols, a.type());

    if (a.type() == CV_32F) {
        for (unsigned int i = 0; i < rows; i++) {
            float* ptra = (float*)(a.data + i * a.step);
            float* ptrout = (float*)(out.data + i * out.step);
            for (unsigned int j = 0; j < cols; j++) {
                *ptrout = std::sin(*ptra);
                ptra++;
                ptrout++;
            }
        }
    } else if (a.type() == CV_64F) {
        for (unsigned int i = 0; i < rows; i++) {
            double* ptra = (double*)(a.data + i * a.step);
            double* ptrout = (double*)(out.data + i * out.step);
            for (unsigned int j = 0; j < cols; j++) {
                *ptrout = std::sin(*ptra);
                ptra++;
                ptrout++;
            }
        }
    }
    TIME_ELAPSED(sin);
    return out;
}
vector<float> RSMat::vec_sin(vector<float>& a) {
    unsigned int len = a.size();
    vector<float> out(len);

    for (unsigned int i = 0; i < len; i++) {
        out[i] = std::sin(a[i]);
    }
    return out;
}

cv::Mat RSMat::cos(cv::Mat& a) {
    TIME_RECORD(cos);
    unsigned int rows = a.rows;
    unsigned int cols = a.cols;
    cv::Mat out(rows, cols, a.type());

    if (a.type() == CV_32F) {
        for (unsigned int i = 0; i < rows; i++) {
            float* ptra = (float*)(a.data + i * a.step);
            float* ptrout = (float*)(out.data + i * out.step);
            for (unsigned int j = 0; j < cols; j++) {
                *ptrout = std::cos(*ptra);
                ptra++;
                ptrout++;
            }
        }
    } else if (a.type() == CV_64F) {
        for (unsigned int i = 0; i < rows; i++) {
            double* ptra = (double*)(a.data + i * a.step);
            double* ptrout = (double*)(out.data + i * out.step);
            for (unsigned int j = 0; j < cols; j++) {
                *ptrout = std::cos(*ptra);
                ptra++;
                ptrout++;
            }
        }
    }
    TIME_ELAPSED(cos);

    return out;
}
vector<float> RSMat::vec_cos(vector<float>& a) {
    unsigned int len = a.size();
    vector<float> out(len);

    for (unsigned int i = 0; i < len; i++) {
        out[i] = std::cos(a[i]);
    }
    return out;
}
//方向向量的单位化(L2范数 )    np.linalg.norm
double RSMat::direction_angle_linalg_norm(std::pair<float, float>& direction) { return sqrt(direction.first * direction.first + direction.second * direction.second); }

//方向向量的单位化(L2范数 )    np.linalg.norm
int RSMat::linalg_norm(cv::Mat& Mat1) { return cv::norm(Mat1, cv::NORM_L2); }


// opencv中，由于使用Mat.at<>访问数据时，必须正确填写相应的数据类型，
//因此必须弄清楚opencv中的数据类型与C++数据类型一一对应关系。
// CV_8U(uchar)
// CV_8S(char)
// CV_16U  (ushort)
// CV_16S  (short)
// CV_32S (int)
// CV_32F (float)
// CV_64F(double)

//只支持CV_32FC1
void RSMat::angle_remainder(cv::Mat& angle_mat, float remainder) {
    TIME_RECORD(angle_remainder);
    if (angle_mat.type() != CV_32F && angle_mat.type() != CV_64F) SPDLOG_LOGGER_ERROR(os::get_logger(), "RSMat::angle_remainder只支持CV_32F CV_64F !!! ");

    if (angle_mat.type() == CV_32F) {
        for (int i = 0; i < angle_mat.rows; i++) {
            for (int j = 0; j < angle_mat.cols; j++) {
                while (angle_mat.at<float>(i, j) > remainder) angle_mat.at<float>(i, j) = angle_mat.at<float>(i, j) - remainder;
                while (angle_mat.at<float>(i, j) < 0) angle_mat.at<float>(i, j) = angle_mat.at<float>(i, j) + remainder;
            }
        }
    } else if (angle_mat.type() == CV_64F) {
        for (int i = 0; i < angle_mat.rows; i++) {
            for (int j = 0; j < angle_mat.cols; j++) {
                while (angle_mat.at<double>(i, j) > remainder) angle_mat.at<double>(i, j) = angle_mat.at<double>(i, j) - remainder;
                while (angle_mat.at<double>(i, j) < 0) angle_mat.at<double>(i, j) = angle_mat.at<double>(i, j) + remainder;
            }
        }
    }
    TIME_ELAPSED(angle_remainder);
}

void RSMat::angle_remainder(vector<float>& angle_vec, float remainder) {
    for (int i = 0; i < angle_vec.size(); i++) {
        while (angle_vec[i] > remainder) angle_vec[i] = angle_vec[i] - remainder;
        while (angle_vec[i] < 0) angle_vec[i] = angle_vec[i] + remainder;
    }
}

// };  // namespace 
