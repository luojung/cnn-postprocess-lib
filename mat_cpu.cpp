/*
 * @Description: Matrix operation interface used to replace numpy based on opencv
 * @Author: luojun1
 * @Date: 2021-05-12 15:24:16
 * @LastEditTime: 2022-05-12 12:36:24
 */

#include "mat_cpu.hpp"
#include "block_strategy.h"
#include "remote_data.h"
#include "remote_datasource.h"
#include "utils/os.hpp"
#include "utils/record_time.h"

cv::Mat RSMat::round(const cv::Mat& mat_float) {
    cv::Mat mat_tmp = mat_float + 0.5;
    cv::Mat mat_8u;
    mat_tmp.convertTo(mat_8u, CV_8U);
    return mat_8u;
}

//求坐标的x轴、y轴 或 x和y的平均值
// axis: -1(x和y的平均值)，0(x轴的平均值)，1（y轴的平均值）
double RSMat::mean(vector<cv::Point> locations, int axis) {
    if (locations.size() == 0 || axis > 1) return 0;

    double mean_x = locations[0].x, mean_y = locations[0].y;
    for (int i = 1; i < locations.size(); i++) {
        mean_x += (locations[i].x - mean_x) / (i + 1);
        mean_y += (locations[i].y - mean_y) / (i + 1);
    }

    if (axis <= -1) {
        return (mean_x + mean_y) / 2;
    } else if (axis == 0) {
        return mean_x;
    } else if (axis == 1) {
        return mean_y;
    }
}

// 取过个波段中的最大值，输出单波段
cv::Mat RSMat::band_max(const vector<cv::Mat>& src_mat_vec) {
    TIME_RECORD(band_max);
    for (int c = 1; c < src_mat_vec.size(); c++) {
        if (src_mat_vec[0].type() != src_mat_vec[c].type()) SPDLOG_LOGGER_ERROR(os::get_logger(), "assert error: src_mat_vec[0].type()！=src_mat_vec[c].type()");
        if (src_mat_vec[0].cols != src_mat_vec[c].cols) SPDLOG_LOGGER_ERROR(os::get_logger(), "assert error: src_mat_vec[0].cols()！=src_mat_vec[c].cols()");
        if (src_mat_vec[0].rows != src_mat_vec[c].rows) SPDLOG_LOGGER_ERROR(os::get_logger(), "assert error: src_mat_vec[0].rows()！=src_mat_vec[c].rows()");
    }

    if (src_mat_vec[0].type() == CV_32F) {
        cv::Mat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_32F);
        for (int i = 0; i < src_mat_vec[0].rows; i++) {
            for (int j = 0; j < src_mat_vec[0].cols; j++) {
                int max_index = 0;
                for (int c = 0; c < src_mat_vec.size(); c++) {
                    if (src_mat_vec[c].at<float>(i, j) > src_mat_vec[max_index].at<float>(i, j)) max_index = c;
                }
                res_mat.at<float>(i, j) = src_mat_vec[max_index].at<float>(i, j);  //三个波段中 最大值的index
            }
        }
        TIME_ELAPSED(band_max);
        return res_mat;
    } else if (src_mat_vec[0].type() == CV_64F) {
        cv::Mat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_64F);
        for (int i = 0; i < src_mat_vec[0].rows; i++) {
            for (int j = 0; j < src_mat_vec[0].cols; j++) {
                int max_index = 0;
                for (int c = 0; c < src_mat_vec.size(); c++) {
                    if (src_mat_vec[c].at<double>(i, j) > src_mat_vec[max_index].at<double>(i, j)) max_index = c;
                }
                res_mat.at<double>(i, j) = src_mat_vec[max_index].at<double>(i, j);  //三个波段中 最大值的index
            }
        }
        TIME_ELAPSED(band_max);
        return res_mat;
    } else if (src_mat_vec[0].type() == CV_8U) {
        cv::Mat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_8U);
        for (int i = 0; i < src_mat_vec[0].rows; i++) {
            for (int j = 0; j < src_mat_vec[0].cols; j++) {
                int max_index = 0;
                for (int c = 0; c < src_mat_vec.size(); c++) {
                    if (src_mat_vec[c].at<uchar>(i, j) > src_mat_vec[max_index].at<uchar>(i, j)) max_index = c;
                }
                res_mat.at<uchar>(i, j) = src_mat_vec[max_index].at<uchar>(i, j);  //三个波段中 最大值的index
            }
        }
        TIME_ELAPSED(band_max);
        return res_mat;
    }
}

// np.argmax
// template <typename T>
cv::Mat RSMat::argmax(vector<cv::Mat>& src_mat_vec) {
    TIME_RECORD(argmax);
    for (int c = 1; c < src_mat_vec.size(); c++) {
        if (src_mat_vec[0].type() != src_mat_vec[c].type()) SPDLOG_LOGGER_ERROR(os::get_logger(), "assert error: src_mat_vec[0].type()！=src_mat_vec[c].type()");
        if (src_mat_vec[0].cols != src_mat_vec[c].cols) SPDLOG_LOGGER_ERROR(os::get_logger(), "assert error: src_mat_vec[0].cols()！=src_mat_vec[c].cols()");
        if (src_mat_vec[0].rows != src_mat_vec[c].rows) SPDLOG_LOGGER_ERROR(os::get_logger(), "assert error: src_mat_vec[0].rows()！=src_mat_vec[c].rows()");
    }

    cv::Mat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_8U);
    if (src_mat_vec[0].type() == CV_32F) {
        for (int i = 0; i < src_mat_vec[0].rows; i++) {
            for (int j = 0; j < src_mat_vec[0].cols; j++) {
                int max_index = 0;
                for (int c = 0; c < src_mat_vec.size(); c++) {
                    if (src_mat_vec[c].at<float>(i, j) > src_mat_vec[max_index].at<float>(i, j)) max_index = c;
                }
                res_mat.at<uchar>(i, j) = max_index;  //三个波段中 最大值的index
            }
        }
    } else if (src_mat_vec[0].type() == CV_64F) {
        for (int i = 0; i < src_mat_vec[0].rows; i++) {
            for (int j = 0; j < src_mat_vec[0].cols; j++) {
                int max_index = 0;
                for (int c = 0; c < src_mat_vec.size(); c++) {
                    if (src_mat_vec[c].at<double>(i, j) > src_mat_vec[max_index].at<double>(i, j)) max_index = c;
                }
                res_mat.at<uchar>(i, j) = max_index;  //三个波段中 最大值的index
            }
        }
    } else if (src_mat_vec[0].type() == CV_8U) {
        for (int i = 0; i < src_mat_vec[0].rows; i++) {
            for (int j = 0; j < src_mat_vec[0].cols; j++) {
                int max_index = 0;
                for (int c = 0; c < src_mat_vec.size(); c++) {
                    if (src_mat_vec[c].at<unsigned char>(i, j) > src_mat_vec[max_index].at<unsigned char>(i, j)) max_index = c;
                }
                res_mat.at<uchar>(i, j) = max_index;  //三个波段中 最大值的index
            }
        }
    }

    // SPDLOG_LOGGER_DEBUG(os::get_logger(), "argmax finished: {}");
    TIME_ELAPSED(argmax);
    return res_mat;
}

//指定数值范围进行截断，如0-255
// template <typename T>
int RSMat::intercept(cv::Mat& src_mat, int max, int min) {
    TIME_RECORD(intercept);
    int cout = 0;
    if (src_mat.type() == CV_32F) {
        for (int i = 0; i < src_mat.rows; i++) {
            for (int j = 0; j < src_mat.cols; j++) {
                if (src_mat.at<double>(i, j) > max) {
                    src_mat.at<double>(i, j) = max;
                    cout++;
                }
                if (src_mat.at<double>(i, j) < min) {
                    src_mat.at<double>(i, j) = min;
                    cout++;
                }
            }
        }
    } else if (src_mat.type() == CV_64F) {
        for (int i = 0; i < src_mat.rows; i++) {
            for (int j = 0; j < src_mat.cols; j++) {
                if (src_mat.at<float>(i, j) > max) {
                    src_mat.at<float>(i, j) = max;
                    cout++;
                }
                if (src_mat.at<float>(i, j) < min) {
                    src_mat.at<float>(i, j) = min;
                    cout++;
                }
            }
        }
    } else if (src_mat.type() == CV_8U) {
        for (int i = 0; i < src_mat.rows; i++) {
            for (int j = 0; j < src_mat.cols; j++) {
                if (src_mat.at<unsigned char>(i, j) > max) {
                    src_mat.at<unsigned char>(i, j) = max;
                    cout++;
                }
                if (src_mat.at<unsigned char>(i, j) < min) {
                    src_mat.at<unsigned char>(i, j) = min;
                    cout++;
                }
            }
        }
    }

    TIME_ELAPSED(intercept);
    return cout;
}

// 2维数组求最大值
template <typename T>
T RSMat::amax(const cv::Mat& src_mat) {
    // TIME_RECORD(amax);
    if (src_mat.rows == 0 || src_mat.cols == 0) return 0;

    T max = src_mat.at<T>(0, 0);
    for (int i = 0; i < src_mat.rows; i++) {
        for (int j = 0; j < src_mat.cols; j++) {
            if (src_mat.at<T>(i, j) > max) max = src_mat.at<T>(i, j);
        }
    }
    // TIME_ELAPSED(amax);
    return max;
}

float RSMat::amax(const cv::Mat& src_mat) {
    float max = 0;
    if (src_mat.type() == CV_32F) {
        max = amax<float>(src_mat);
        return max;
    } else if (src_mat.type() == CV_64F) {
        max = amax<double>(src_mat);
        return max;
    } else if (src_mat.type() == CV_8U) {
        max = amax<unsigned char>(src_mat);
        return max;
    }
}

// 2维8u求最小值
template <typename T>
T RSMat::amin(const cv::Mat& src_mat) {
    // TIME_RECORD(amin);
    int min = 0;
    for (int i = 0; i < src_mat.rows; i++) {
        for (int j = 0; j < src_mat.cols; j++) {
            if (src_mat.at<T>(i, j) < min) min = src_mat.at<T>(i, j);
        }
    }
    // TIME_ELAPSED(amin);
    return min;
}
float RSMat::amin(const cv::Mat& src_mat) {
    float min = 0;
    if (src_mat.type() == CV_32F) {
        min = amin<float>(src_mat);
        return min;
    } else if (src_mat.type() == CV_64F) {
        min = amin<double>(src_mat);
        return min;
    } else if (src_mat.type() == CV_8U) {
        min = amin<unsigned char>(src_mat);
        return min;
    }
}

// mat 与或
cv::Mat RSMat::logical_or(const cv::Mat& src_mat1, const cv::Mat& src_mat2) { return (src_mat1 + src_mat2) > 0; }
cv::Mat RSMat::logical_and(const cv::Mat& src_mat1, const cv::Mat& src_mat2) { return (src_mat1.mul(src_mat2)) > 0; }
cv::Mat RSMat::logical_and_(cv::Mat src_mat1, cv::Mat src_mat2) { return (src_mat1.mul(src_mat2)) > 0; }
cv::Mat RSMat::logical_not(const cv::Mat& src_mat1) { return src_mat1 == 0; }
cv::Mat RSMat::logical(const cv::Mat& src_mat1) {
    return src_mat1 != 0;  // src_mat1==0的结果是 0 ,255
}
// mat 异或
cv::Mat RSMat::logical_xor(const cv::Mat& src_mat1, const cv::Mat& src_mat2) {
    cv::Mat mat1 = logical(src_mat1);
    cv::Mat mat2 = logical(src_mat2);

    return ((mat1.setTo(1, mat1) + mat2.setTo(1, mat2)) == 1);
}
// mat 异或
cv::Mat RSMat::logical_xor_(cv::Mat src_mat1, cv::Mat src_mat2) {
    cv::Mat mat1_logic = src_mat1.setTo(1, src_mat1 != 0);
    cv::Mat mat2_logic = src_mat2.setTo(1, src_mat2 != 0);

    return ((mat1_logic + mat2_logic) == 1);
}

set<int> RSMat::unique(const cv::Mat& src_mat) {
    set<int> res;
    for (int i = 0; i < src_mat.rows; i++) {
        for (int j = 0; j < src_mat.cols; j++) {
            if (src_mat.channels() == 3) {
                for (int c = 0; c < src_mat.channels(); c++) res.insert(src_mat.at<cv::Vec3b>(i, j)[c]);
            } else {
                int* value = (int*)(src_mat.data + i * src_mat.step);
                res.insert(*(value + j));
            }
        }
    }
    return res;
}

//// <uchar>--CV_8U// <char>---CV_8S// <short>-----CV_16S
// <ushort>---CV_16U// <int>---CV_32S// <float>--CV_32F
// <double>----CV_64F//  -   CV_8UC3
cv::Mat RSMat::zeros_like(const cv::Mat src_mat1, int type) { return cv::Mat(src_mat1.rows, src_mat1.cols, type, cv::Scalar(0)); }
cv::Mat RSMat::zeros_like(const cv::Mat src_mat1) {
    return cv::Mat(src_mat1.rows, src_mat1.cols, src_mat1.type(), cv::Scalar(0));
}

unsigned int RSMat::count_nonzero(cv::Mat mat) {
    TIME_RECORD(count_nonzero);
    vector<cv::Point> tip_coords;
    cv::findNonZero(mat, tip_coords);
    return tip_coords.size();
    TIME_ELAPSED(count_nonzero);
}

cv::Mat RSMat::nan_to_num(cv::Mat& mat) { return mat.setTo(0, mat == CV_NAN); }

/* * write_img- */
template <typename T>
bool RSMat::imwrite(string file_path, cv::Mat out) {
    const char* pDstImgFileName = file_path.c_str();
    int width = out.cols, height = out.rows, nChannels = out.channels();
    if (pDstImgFileName == NULL || width < 1 || height < 1 || nChannels < 1) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "invalid para ! ");
        return false;
    }

    vector<cv::Mat> Mat_vec;
    cv::split(out, Mat_vec);
    geods::RemoteData<T> R_data(width, height, 1);
    memcpy(R_data.data_ptr(), Mat_vec[0].data, height * width * sizeof(T));
    for (int i = 1; i < nChannels; i++) {
        auto data_i = std::make_shared<geods::RemoteData<T> >(width, height, 1);
        memcpy(data_i->data_ptr(), Mat_vec[i].data, height * width * sizeof(T));
        R_data.Concate(data_i);
    }

    string suffix = os::path_suffix(file_path);
    if (suffix == "png")
        R_data.SavePng(file_path);
    else if (suffix == "tif" || suffix == "tiff")
        R_data.SaveTif(file_path);
    return true;
}

bool RSMat::imwrite(string file_path, cv::Mat out) {
    if (out.type() == CV_8U) {
        return imwrite<unsigned char>(file_path, out);
    } else if (out.type() == CV_16S) {
        return imwrite<unsigned short>(file_path, out);
    } else if (out.type() == CV_32F) {
        return imwrite<float>(file_path, out);
    } else if (out.type() == CV_64F) {
        return imwrite<double>(file_path, out);
    } else if (out.type() == CV_32S) {
        return imwrite<unsigned int>(file_path, out);
    }
}

/* *read TIFF-*/
int RSMat::imread_tiff(string file_path_name, vector<cv::Mat>& out_mats) {
    int res = 0;
    auto ds = std::make_shared<geods::RemoteDataSource>();
    ds->Open(file_path_name);
    SPDLOG_LOGGER_INFO(os::get_logger(), "imread_tiff: ds->width() :{} , ds->height():{}", ds->width(), ds->height());
    int pixelDepth = ds->GetDepth();

    if (pixelDepth == 1) {
        typedef unsigned char TYPE;
        assert(pixelDepth == sizeof(TYPE));
        void* data = malloc(ds->bands() * ds->width() * ds->height() * sizeof(TYPE));

        ds->merge_data(0, 0, ds->width(), ds->height(), (void*)data, 0);
        for (int c = 0; c < ds->bands(); c++) {
            cv::Mat Mat_i(ds->height(), ds->width(), CV_8U);
            unsigned char* dst = Mat_i.data;
            TYPE* src = (TYPE*)data + c * ds->height() * ds->width();
            memcpy(dst, src, ds->height() * ds->width() * sizeof(TYPE));
            out_mats.push_back(Mat_i);
            // SPDLOG_LOGGER_INFO(os::get_logger(), "Mat_i: {} , [1,1]={}, [23,23]={}", Mat_i.total(), Mat_i.at<TYPE>(1, 1), Mat_i.at<TYPE>(23, 23));
        }
        free(data);
        return res;
    } else if (pixelDepth == 2) {
        typedef unsigned short int TYPE;
        assert(pixelDepth == sizeof(TYPE));
        void* data = malloc(ds->bands() * ds->width() * ds->height() * sizeof(TYPE));

        ds->merge_data(0, 0, ds->width(), ds->height(), (void*)data, 0);
        for (int c = 0; c < ds->bands(); c++) {
            cv::Mat Mat_i(ds->height(), ds->width(), CV_16S);
            unsigned char* dst = Mat_i.data;
            TYPE* src = (TYPE*)data + c * ds->height() * ds->width();
            memcpy(dst, src, ds->height() * ds->width() * sizeof(TYPE));
            out_mats.push_back(Mat_i);
            // SPDLOG_LOGGER_INFO(os::get_logger(), "Mat_i: {} , [1,1]={}, [23,23]={}", Mat_i.total(), Mat_i.at<TYPE>(1, 1), Mat_i.at<TYPE>(23, 23));
        }
        free(data);
        return res;
    } else if (pixelDepth == 4) {
        typedef float TYPE;
        assert(pixelDepth == sizeof(TYPE));
        void* data = malloc(ds->bands() * ds->width() * ds->height() * sizeof(TYPE));

        ds->merge_data(0, 0, ds->width(), ds->height(), (void*)data, 0);
        for (int c = 0; c < ds->bands(); c++) {
            cv::Mat Mat_i(ds->height(), ds->width(), CV_32F);
            unsigned char* dst = Mat_i.data;
            TYPE* src = (TYPE*)data + c * ds->height() * ds->width();
            memcpy(dst, src, ds->height() * ds->width() * sizeof(TYPE));
            out_mats.push_back(Mat_i);
            // SPDLOG_LOGGER_INFO(os::get_logger(), "Mat_i: {} , [1,1]={}, [23,23]={}", Mat_i.total(), Mat_i.at<TYPE>(1, 1), Mat_i.at<TYPE>(23, 23));
        }
        free(data);
        return res;
    } else if (pixelDepth == 8) {
        typedef double TYPE;
        assert(pixelDepth == sizeof(TYPE));
        void* data = malloc(ds->bands() * ds->width() * ds->height() * sizeof(TYPE));

        ds->merge_data(0, 0, ds->width(), ds->height(), (void*)data, 0);
        for (int c = 0; c < ds->bands(); c++) {
            cv::Mat Mat_i(ds->height(), ds->width(), CV_64F);
            unsigned char* dst = Mat_i.data;
            TYPE* src = (TYPE*)data + c * ds->height() * ds->width();
            memcpy(dst, src, ds->height() * ds->width() * sizeof(TYPE));
            out_mats.push_back(Mat_i);
            // SPDLOG_LOGGER_INFO(os::get_logger(), "Mat_i: {} , [1,1]={}, [23,23]={}", Mat_i.total(), Mat_i.at<TYPE>(1, 1), Mat_i.at<TYPE>(23, 23));
        }
        free(data);
        return res;
    }
}
/* *  read_ _C1--png */
cv::Mat RSMat::imread_C1(string file_path_name) {
    vector<cv::Mat> out_mats;
    imread_tiff(file_path_name, out_mats);
    return out_mats[0];
}

// calculate similarity rate
float RSMat::similarity(const cv::Mat mat1, string path_mat2) {
    cv::Mat mat2 = RSMat::imread_C1(path_mat2);
    if (mat2.empty()) SPDLOG_LOGGER_ERROR(os::get_logger(), "failed to cv::imread :{}  ", path_mat2);
    if (mat1.empty() && mat2.empty()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "{} is empty ", os::path_basename(path_mat2));
        return 1;
    }
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims || mat1.channels() != mat2.channels()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "{} is not equal ", os::path_basename(path_mat2));
        return 0;
    }
    if (mat1.size() != mat2.size() || mat1.type() != mat2.type()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "{} is not equal ", os::path_basename(path_mat2));
        return 0;
    }
    long long nrOfElements1 = mat1.total() * mat1.elemSize();
    if (nrOfElements1 != mat2.total() * mat2.elemSize()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "{} is not equal ", os::path_basename(path_mat2));
        return 0;
    }
    long long equal_num = 0, total = 0;
    for (int i = 0; i < nrOfElements1; i++) {
        if (*(mat2.data + i) > 0) {
            total++;
            if (*(mat1.data + i) == *(mat2.data + i)) equal_num++;
        }
    }
    if (total == 0) return 1.0;

    return double(equal_num) / double(total);
}

bool RSMat::is_equal(const cv::Mat mat1, const cv::Mat mat2, float thr, string mat_name) {
    if (mat1.empty() && mat2.empty()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "both are empty ");
        return true;
    }
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims || mat1.channels() != mat2.channels()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "{}：shape is not equal ", mat_name);
        SPDLOG_LOGGER_ERROR(os::get_logger(), "：shape： {} {} {}  !=  {} {} {}", mat1.cols, mat1.rows, mat1.channels(), mat2.cols, mat2.rows, mat2.channels());
        return false;
    }
    if (mat1.size() != mat2.size()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "{}：size is not equal ", mat_name);
        return false;
    }
    if (mat1.type() != mat2.type()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "{}：type is not equal ", mat_name);
        SPDLOG_LOGGER_ERROR(os::get_logger(), "type：  {}  !=  {} ", mat1.type(), mat2.type());
        return false;
    }
    long long nrOfElements1 = mat1.total();
    if (nrOfElements1 != mat2.total()) {
        SPDLOG_LOGGER_ERROR(os::get_logger(), "{}：values are not equal ", mat_name);
        return false;
    }

    long long equal_num = 0;
    if (mat1.type() == CV_32F && mat2.type() == CV_32F) {
        for (unsigned int i = 0; i < nrOfElements1; i++) {
            if (*((float*)mat1.data + i) == *((float*)mat2.data + i))
                equal_num++;
            else if (abs(*((float*)mat1.data + i) - *((float*)mat2.data + i)) < 0.000001) {
                equal_num++;
            } else {
                // std::cout << *((float*)mat1.data + i) << " " << *((float*)mat2.data + i) << "    ";
            }
        }

    } else if (mat1.type() == CV_8U && mat2.type() == CV_8U) {
        for (unsigned int i = 0; i < nrOfElements1; i++) {
            if (*((unsigned char*)mat1.data + i) == *((unsigned char*)mat2.data + i))
                equal_num++;
            else {
                // std::cout << *(mat1.data + i) << " " << *(mat2.data + i) << "    ";
            }
        }
    } else if (mat1.type() == CV_64F && mat2.type() == CV_64F) {
        for (unsigned int i = 0; i < nrOfElements1; i++) {
            if (*((double*)mat1.data + i) == *((double*)mat2.data + i))
                equal_num++;
            else if (abs(*((float*)mat1.data + i) - *((float*)mat2.data + i)) < 0.000001) {
                equal_num++;
            } else {
                // std::cout << *((double*)mat1.data + i) << " " << *((double*)mat2.data + i) << "    ";
            }
        }
    }

    // SPDLOG_LOGGER_INFO(os::get_logger(), " equal num:{}, total:{}", equal_num, nrOfElements1);

    double rate_ = double(equal_num) / double(nrOfElements1);
    if (equal_num >= nrOfElements1 * thr) {
        SPDLOG_LOGGER_INFO(os::get_logger(), mat_name + ": equal rate:{},not_equal:{}", rate_, nrOfElements1 - equal_num);
        return true;
    } else {
        SPDLOG_LOGGER_INFO(os::get_logger(), mat_name + ":  equal rate:{},not_equal:{}", rate_, nrOfElements1 - equal_num);
        return false;
    }
}
// thr,阈值，如0.9为90%相等则返回true
bool RSMat::is_equal(string path_mat1, const cv::Mat mat2, float thr, string mat_name) {
    cv::Mat mat1 = RSMat::imread_C1(path_mat1);
    return is_equal(mat1, mat2, thr, mat_name);
}
bool RSMat::is_equal(string path_mat1, string path_mat2, float thr, string mat_name) {
    cv::Mat mat1 = RSMat::imread_C1(path_mat1);
    cv::Mat mat2 = RSMat::imread_C1(path_mat2);
    return is_equal(mat1, mat2, thr, mat_name);
}
bool RSMat::is_equal(string path_mat1, const cv::cuda::GpuMat mat2, float thr, string mat_name) {
    cv::Mat mat1 = RSMat::imread_C1(path_mat1);
    // download
    cv::Mat h_b;
    {
        auto start1 = std::chrono::system_clock::now();
        mat2.download(h_b);
        auto end1 = std::chrono::system_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
        // SPDLOG_LOGGER_INFO(os::get_logger(), "download: {}", elapsed1.count());
    }
    return is_equal(mat1, h_b, thr, mat_name);
}

bool RSMat::compare_mat(cv::Mat a, cv::cuda::GpuMat b, string fn) {
    // download
    cv::Mat h_b;
    {
        auto start1 = std::chrono::system_clock::now();
        b.download(h_b);
        auto end1 = std::chrono::system_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
        // SPDLOG_LOGGER_INFO(os::get_logger(), "download: {}", elapsed1.count());
    }

    // compare
    bool equal = RSMat::is_equal(a, h_b, 0.9999, fn);
    if (fn != "") {
        a.convertTo(a, CV_8U);
        string path = "../data/" + fn + ".tif";
        RSMat::imwrite(path, a);

        h_b.convertTo(h_b, CV_8U);
        string path_cuda = "../data/" + fn + "_cuda.tif";
        RSMat::imwrite(path_cuda, h_b);
    }

    return equal;
}
