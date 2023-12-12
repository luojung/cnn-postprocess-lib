/*
 * @Description: 
 * @Author: luojun1
 * @Date: 2022-02-16 17:54:46
 * @LastEditTime: 2022-05-08 19:53:14
 */


#ifndef GPUMAT_OP_KERNEL
#define GPUMAT_OP_KERNEL

#include <cuda.h>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include <vector>
#include "utils/record_time.h"

#define uint unsigned int
#define BLOCK_WIDTH 32
//执行核函数前，需配置block数量和thread数量
//cuda采用simt架构，由于不同gpu的sm中的cuda核心不同，需要根据sm、sp数量对block的thread数量进行调整，默认一个block中有1024个thread
inline void config_grid_block(unsigned int rows, unsigned int cols, dim3 &grid, dim3 &block){
    unsigned int block_x =BLOCK_WIDTH , block_y = BLOCK_WIDTH; 
    unsigned int grid_x = (cols+block_x-1)/block_x < 10 ? (cols+block_x-1)/block_x : 10;
    unsigned int grid_y = (rows+block_y-1)/block_y < 10 ? (rows+block_y-1)/block_y : 10;
    block = dim3(block_x, block_y);
    grid = dim3(grid_x, grid_y);
}
inline void config_grid_block(unsigned int size, dim3 &grid, dim3 &block){
    unsigned int block_x =BLOCK_WIDTH * BLOCK_WIDTH; 
    unsigned int grid_x = (size+block_x-1)/block_x < 10 ? (size+block_x-1)/block_x : 10;
    block = dim3(block_x);
    grid = dim3(grid_x);
}
template <typename T>
__global__ void  d_argmax(T** mats, int channels, unsigned int cols, unsigned int rows, unsigned char* label){
    for(unsigned int y = blockIdx.y*blockDim.y + threadIdx.y; y<rows; y+=gridDim.y*blockDim.y){
        for(unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; x<cols; x+=gridDim.x*blockDim.x){
            unsigned int index_in_1d = y*cols + x;
            unsigned char max_index = 0;
            T max = mats[0][index_in_1d];
            for(unsigned char ch=1; ch < channels; ch++){
                if(mats[ch][index_in_1d] > max){
                    max_index = ch;
                    max = mats[ch][index_in_1d];
                }
            }
            // printf("max_index: %d \n", max_index );
            label[index_in_1d]=max_index;
        }
    }

 
}

#define TILE_LEN 2000   //默认的tile宽度(一维)

template <typename T>
__global__ void  d_max(T* src_mat, unsigned int cols, unsigned int rows, T* res){
    if(cols =1&& rows ==1)
        return;

    int tile_cols = cols>(gridDim.x * blockDim.x) ? (cols+ gridDim.x * blockDim.x-1) / (gridDim.x * blockDim.x) : 2;
    int tile_rows = (rows+ gridDim.y * blockDim.y-1) / (gridDim.y * blockDim.y);

    //确定索引 
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    int idy = threadIdx.y + blockIdx.y * blockDim.y; 
    int datax = tile_cols*idx; 
    int datay = tile_rows*idx; 

    T max=src_mat[datay*cols+datax] ;
    for(unsigned int y = datay; y<datay + tile_rows && y<rows; y++){
        for(unsigned int x = datax; x<datax + tile_cols && x<cols; x++){

            unsigned int index_in_1d = y*cols + x;
            if (src_mat[index_in_1d] > max){
                max = src_mat[index_in_1d];
            };
        }
    }

    unsigned int index = idy*cols + idx;
    if(res[index] < max){
        res[index] = max;
    };
    // return d_max<T><<<gridDim, blockDim>>>(res, (cols+tile_cols-1)/tile_cols, (rows+tile_rows-1)/tile_rows, res);

    // __syncthreads();
    
    // if(x<cols && y<rows){
    //     unsigned char max_index = 0;
    //     T max = mats[0][index_in_1d];
    //     for(int ch=0; ch < channels; ch++){
    //         if(mats[ch][index_in_1d] > max){
    //             max_index = ch;
    //             max = mats[ch][index_in_1d];
    //         }
    //     }
    
    //     label[index_in_1d]=max_index;
    // }
}

template <typename T>
__global__ void  d_max(T* src_mat, unsigned int len, T* res){
    assert(gridDim.y==1 && blockDim.y==1);
    if(len==1) return;

    unsigned int threads = gridDim.x * blockDim.x;
    unsigned int tile_width = (TILE_LEN*threads) > len ? TILE_LEN : (len+ threads-1) / (threads) ;

    //根据线程id确定数据
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    unsigned int datax = tile_width*idx; 
    if(datax >= len || idx>=len) {
        return;
    }
    
    T max=src_mat[datax];
    for(unsigned int x = datax+1; x<(datax+tile_width) && x<len; x++){
        if (src_mat[x] > max){
            max = src_mat[x];
        };
    }
    
    if(res[idx] < max){
        res[idx] = max;
    };
}

template <typename T>
__global__ void  d_min(T* src_mat, unsigned int len, T* res){
    assert(gridDim.y==1 && blockDim.y==1);
    if(len==1) return;

    unsigned int threads = gridDim.x * blockDim.x;
    unsigned int tile_width = (TILE_LEN*threads) > len ? TILE_LEN : (len+ threads-1) / (threads) ;

    //确定索引 
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    unsigned int datax = tile_width*idx; 
    if(datax >= len || idx>=len) {
        return;
    }
    
    T min=src_mat[datax];
    for(unsigned int x = datax+1; x<(datax+tile_width) && x<len; x++){
        if (src_mat[x] < min){
            min = src_mat[x];
        };
    }
    
    if(res[idx] > min){
        res[idx] = min;
    };
}


template <typename T>
__global__ void  d_arctan2(T* mat_v, T* mat_h, unsigned int cols, unsigned int rows, T* res){
    for(unsigned int y = blockIdx.y*blockDim.y + threadIdx.y; y<rows; y+=gridDim.y*blockDim.y){
        for(unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; x<cols; x+=gridDim.x*blockDim.x){
            unsigned int index_in_1d = y*cols + x;
            res[index_in_1d]=atan2(mat_v[index_in_1d], mat_h[index_in_1d]);
        }
    }
}

template <typename T>
__global__ void  d_sin(T* angle, unsigned int cols, unsigned int rows, T* res){
    for(unsigned int y = blockIdx.y*blockDim.y + threadIdx.y; y<rows; y+=gridDim.y*blockDim.y){
        for(unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; x<cols; x+=gridDim.x*blockDim.x){
            unsigned int index_in_1d = y*cols + x;
            res[index_in_1d]=sin(angle[index_in_1d]);
        }
    }
}

template <typename T>
__global__ void  d_cos(T* angle, unsigned int cols, unsigned int rows, T* res){
    for(unsigned int y = blockIdx.y*blockDim.y + threadIdx.y; y<rows; y+=gridDim.y*blockDim.y){
        for(unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; x<cols; x+=gridDim.x*blockDim.x){
            unsigned int index_in_1d = y*cols + x;
            res[index_in_1d]=cos(angle[index_in_1d]);
        }
    }
}
template<typename T>
__global__ void d_remainder(T* angle, unsigned int cols, unsigned int rows, T remainder){
    for(unsigned int y = blockIdx.y*blockDim.y+threadIdx.y; y<rows; y+=gridDim.y*blockDim.y){
        for(unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; x<cols; x+=gridDim.x*blockDim.x){
            unsigned int index_in_1d = y*cols + x;
            while (angle[index_in_1d] > remainder) angle[index_in_1d] = angle[index_in_1d] - remainder;
            while (angle[index_in_1d] < 0) angle[index_in_1d] = angle[index_in_1d] + remainder;
        }
    }
}


template<typename T>
__global__ void  d_nonmaxsupress(T*  Gm, T*  Gd, T* out, unsigned int cols, unsigned int rows, int step, float th = 1.0) {  //非最大值抑制算法

    uint threadx = blockIdx.x * blockDim.x + threadIdx.x;
    uint thready = blockIdx.y * blockDim.y + threadIdx.y;
    // uint tile_width = (cols + blockDim.x* gridDim.x -1) / blockDim.x* gridDim.x;
    // uint tile_length = (rows + blockDim.y* gridDim.y -1) / blockDim.y* gridDim.y;
    
    for(uint y= thready+1; y<rows-2; y+=gridDim.y*blockDim.y){
        for(uint x= threadx+1; x<cols-2; x+=gridDim.x* blockDim.x){
            // __shared__ T Gm_data[BLOCK_WIDTH][BLOCK_WIDTH];
            // Gm_data[threadIdx.y][threadIdx.x]= Gm[y*cols+x];
            // __syncthreads();
            
            // T mag = Gm_data[threadIdx.y][threadIdx.x];
            T mag = Gm[y*step+x];
            if (mag < th) continue; 
            T teta = Gd[y*step+x];

            int dx = 0, dy = -1;  //# abs(orient) >= 1.1781, teta < -67.5 degrees and teta > 67.5 degrees
            if (abs(teta) <= T(0.3927f)) {
            // if (abs(teta) <= 0.1927) {
                dx = 1;
                dy = 0;  //# -22.5 <= teta <= 22.5
            } else if (teta < T(1.1781f) && teta > T(0.3927f)) {
            // } else if (teta < 1.1781 && teta > 0.1927) {
                dx = 1;
                dy = -1;  //# 22.5 < teta < 67.5 degrees
            } else if (teta > T(-1.1781f) && teta < T(-0.3927f)) {
                dx = 1;
                dy = 1;  //# -67.5 < teta < -22.5 degrees
            }
            
            // T thr1;
            // if(threadIdx.y+dy<0 || threadIdx.y+dy>blockDim.y || threadIdx.x+dx<0 || threadIdx.x+dx>blockDim.x)
                // thr1=Gm[(y + dy)*cols+ x + dx];
            // else
            //     thr1=Gm_data[threadIdx.y+dy][threadIdx.x+dx];

            T thr1=Gm[(y + dy)*step+ x + dx];

            // T thr2;
            // if(threadIdx.y-dy<0 || threadIdx.y-dy>blockDim.y || threadIdx.x-dx<0 || threadIdx.x-dx>blockDim.x)
                // thr2=Gm[(y - dy)*cols+ x - dx];
            // else
            //     thr2=Gm_data[threadIdx.y-dy][threadIdx.x-dx];

            T thr2=Gm[(y - dy)*step+ x - dx];

            if (mag > thr1 && mag > thr2 ){
                // printf("x:%d y:%d mag: %f  | x:%d y:%d thr1: %f | x:%d y:%d thr2: %f teta:%f\n",x, y, mag, x + dx, y + dy, thr1, x - dx, y - dy, thr2, teta );
                out[y*step+x] = mag;
            }else if (mag!=0){
                // printf("#x:%d y:%d mag: %f | x:%d y:%d thr1: %f | x:%d y:%d thr2: %f teta:%f\n",x, y, mag, x + dx, y + dy, thr1, x - dx, y - dy, thr2, teta );
            }
        }
    }
}


template<typename T>
__global__ void  d_nonmaxsupress_shm(T*  Gm, T*  Gd, T* out, unsigned int cols, unsigned int rows, int step, float th = 1.0) {  //非最大值抑制算法

    uint threadx = blockIdx.x * blockDim.x + threadIdx.x;
    uint thready = blockIdx.y * blockDim.y + threadIdx.y;
    // uint tile_width = (cols + blockDim.x* gridDim.x -1) / blockDim.x* gridDim.x;
    // uint tile_length = (rows + blockDim.y* gridDim.y -1) / blockDim.y* gridDim.y;
    
    for(uint y= thready+1; y<rows-2; y+=gridDim.y*blockDim.y){
        for(uint x= threadx+1; x<cols-2; x+=gridDim.x* blockDim.x){
            // __shared__ T Gm_data[BLOCK_WIDTH][BLOCK_WIDTH];
            // Gm_data[threadIdx.y][threadIdx.x]= Gm[y*cols+x];
            // __syncthreads();
            
            // T mag = Gm_data[threadIdx.y][threadIdx.x];
            T mag = Gm[y*step+x];
            if (mag < th) continue; 
            T teta = Gd[y*step+x];

            int dx = 0, dy = -1;  //# abs(orient) >= 1.1781, teta < -67.5 degrees and teta > 67.5 degrees
            if (abs(teta) <= T(0.3927f)) {
            // if (abs(teta) <= 0.1927) {
                dx = 1;
                dy = 0;  //# -22.5 <= teta <= 22.5
            } else if (teta < T(1.1781f) && teta > T(0.3927f)) {
            // } else if (teta < 1.1781 && teta > 0.1927) {
                dx = 1;
                dy = -1;  //# 22.5 < teta < 67.5 degrees
            } else if (teta > T(-1.1781f) && teta < T(-0.3927f)) {
                dx = 1;
                dy = 1;  //# -67.5 < teta < -22.5 degrees
            }
            
            // T thr1;
            // if(threadIdx.y+dy<0 || threadIdx.y+dy>blockDim.y || threadIdx.x+dx<0 || threadIdx.x+dx>blockDim.x)
                // thr1=Gm[(y + dy)*cols+ x + dx];
            // else
            //     thr1=Gm_data[threadIdx.y+dy][threadIdx.x+dx];

            T thr1=Gm[(y + dy)*step+ x + dx];

            // T thr2;
            // if(threadIdx.y-dy<0 || threadIdx.y-dy>blockDim.y || threadIdx.x-dx<0 || threadIdx.x-dx>blockDim.x)
                // thr2=Gm[(y - dy)*cols+ x - dx];
            // else
            //     thr2=Gm_data[threadIdx.y-dy][threadIdx.x-dx];

            T thr2=Gm[(y - dy)*step+ x - dx];

            if (mag > thr1 && mag > thr2 ){
                // printf("x:%d y:%d mag: %f  | x:%d y:%d thr1: %f | x:%d y:%d thr2: %f teta:%f\n",x, y, mag, x + dx, y + dy, thr1, x - dx, y - dy, thr2, teta );
                out[y*step+x] = mag;
            }else if (mag!=0){
                // printf("#x:%d y:%d mag: %f | x:%d y:%d thr1: %f | x:%d y:%d thr2: %f teta:%f\n",x, y, mag, x + dx, y + dy, thr1, x - dx, y - dy, thr2, teta );
            }
        }
    }
}

template<typename T>
__global__ void d_band_max(T** mats, int channels, unsigned int cols, unsigned int rows, T* label){
    uint x = blockIdx.x* blockDim.x + threadIdx.x;
    uint y = blockIdx.y* blockDim.y + threadIdx.y;

    uint thread_id = x + y*blockDim.x* gridDim.x;
    uint step = (blockDim.x* gridDim.x * blockDim.y* gridDim.y );

    for(uint index= thread_id; index < cols * rows; index+=step){
        // uint index= thread_id*tile + i;
        T max = mats[0][index];
        for(uint c = 1; c<channels; c++){
            if( mats[c][index] > max){
                max = mats[c][index];
            }
        }
        label[index] = max;
    }

    // uint width = (cols * rows + blockDim.x* gridDim.x * blockDim.y* gridDim.y -1) / (blockDim.x* gridDim.x * blockDim.y* gridDim.y );
    // for(uint i= 0; i < width; i++){
    //     uint index= thread_id*width + i;
    //     T max = mats[0][index];
    //     for(uint c = 1; c<channels; c++){
    //         if( mats[c][index] > max){
    //             max = mats[c][index];
    //         }
    //     }
    //     label[index] = max;
    // }
}

template <typename T>
__global__ void  d_band_max2(T** mats, int channels, unsigned int cols, unsigned int rows, T* label){
    for(unsigned int y = blockIdx.y*blockDim.y + threadIdx.y; y<rows; y+=gridDim.y*blockDim.y){
        for(unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; x<cols; x+=gridDim.x*blockDim.x){
            unsigned int index = y*cols + x;
            T max = mats[0][index];
            for(unsigned char ch=1; ch < channels; ch++){
                if(mats[ch][index] > max){
                    max = mats[ch][index];
                }
            }
            label[index]=max;
        }
    }
}


// //提取连通域的面积和中心点坐标
// template<typename T>
// __global__ void d_remove_small_objects(T* mats, unsigned int cols, unsigned int rows, int thr){
//     uint x = blockIdx.x* blockDim.x + threadIdx.x;
//     uint y = blockIdx.y* blockDim.y + threadIdx.y;

//     uint filter_size = thr;
//     uint 

//     for(uint y = 0; y < rows-filter_size; ++y){
//         for(uint x = 0; y < cols-filter_size; ++x){
            
//         }
//     }
// }

namespace RSMat{

    cv::cuda::GpuMat argmax_cuda(const std::vector<cv::cuda::GpuMat>& src_mat_vec){
        // 开始计时
        CUDA_KERNEL_TIME_RECORD(argmax_cuda);

        //  config blocks 
        dim3 block, grid;
        config_grid_block(src_mat_vec[0].rows, src_mat_vec[0].cols, block, grid);

        //prepare input, output
        cv::cuda::GpuMat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_8U, cv::Scalar(0));
        if(src_mat_vec[0].type()==CV_8U){
            // std::cout<<"2.1  cudaMalloc src_mats[]"<<std::endl;
            unsigned char ** d_src_mats;
            cudaMalloc((void ***)&d_src_mats, src_mat_vec.size()*sizeof(unsigned char*));

            unsigned char ** h_src_mats = new unsigned char *[src_mat_vec.size()];
            for(int i=0; i<src_mat_vec.size(); i++){
                h_src_mats[i] = (unsigned char *) src_mat_vec[i].data;
            }
            cudaMemcpy(d_src_mats, h_src_mats, src_mat_vec.size()*sizeof(unsigned char*), cudaMemcpyHostToDevice);
            
            d_argmax<unsigned char><<<grid, block>>>(d_src_mats, src_mat_vec.size(), src_mat_vec[0].cols, src_mat_vec[0].rows, res_mat.data);
            delete[] h_src_mats;
            cudaFree(d_src_mats);

        }
        else if(src_mat_vec[0].type()==CV_32F){
            float ** d_src_mats;
            cudaMalloc((void ***)&d_src_mats, src_mat_vec.size()*sizeof(float*));
            
            float ** h_src_mats =new float *[src_mat_vec.size()];
            for(int i=0; i<src_mat_vec.size(); i++){
                h_src_mats[i]= (float *) src_mat_vec[i].data;
            }
            cudaMemcpy(d_src_mats, h_src_mats, src_mat_vec.size()*sizeof(float*), cudaMemcpyHostToDevice);
            
            d_argmax<float><<<grid, block>>>(d_src_mats, src_mat_vec.size(), src_mat_vec[0].cols, src_mat_vec[0].rows, res_mat.data);
            delete[] h_src_mats;
            cudaFree(d_src_mats);
        }

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }
        
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "0cudaDeviceSynchronize returned error code %d : %s!\n", cudaStatus, cudaGetErrorString(cudaStatus));
        }
        
        CUDA_KERNEL_TIME_ELAPSED(argmax_cuda);
        return res_mat;
    }

    template <typename T>
    T max_cuda(const cv::cuda::GpuMat& src_mat){
        // 开始计时
        CUDA_KERNEL_TIME_RECORD(max_cuda);

        unsigned int rows = src_mat.rows;
        unsigned int cols = src_mat.cols;
        int block_len =1024; //320;
        int grid_len =10;  //30;

        dim3 block = dim3(block_len);
        int grid_x = (rows*cols+block_len-1)/(TILE_LEN*block_len) >grid_len ? grid_len :  (rows*cols+block_len-1)/(TILE_LEN*block_len);
        grid_x= (rows*cols+block_len-1)/(TILE_LEN*block_len) <1 ? 1 : grid_x;
        dim3 grid = dim3(grid_x);

        //prepare input, output
        cv::cuda::GpuMat res_mat(1, grid.x*block.x, src_mat.type(), cv::Scalar(0));

        d_max<T><<<grid, block>>>((T*) src_mat.data, cols*rows, (T*)res_mat.data);
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "0addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        int len = cols*rows;
        int tile_width = (TILE_LEN*grid.x*block.x) > len ? TILE_LEN : (len+ grid.x*block.x-1) / (grid.x*block.x);
        int res_mat_len = (len+tile_width-1) / tile_width;

        while(res_mat_len>1){
            if(res_mat_len<=block_len)
                block_len = block_len/TILE_LEN>0? block_len/TILE_LEN : 1;
            
            block.x=block_len;
            grid.x=(res_mat_len+block_len-1)/(TILE_LEN*block_len) >grid_len ? grid_len :  (rows*cols+block_len-1)/(TILE_LEN*block_len);
            grid.x=(res_mat_len+block_len-1)/(TILE_LEN*block_len) <1 ? 1 : grid.x;

            
            d_max<T><<<grid, block>>>((T*) res_mat.data, res_mat_len, (T*)res_mat.data);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "1addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }

            tile_width = (TILE_LEN*grid.x*block.x) > res_mat_len ? TILE_LEN : (res_mat_len+ grid.x*block.x-1) / (grid.x*block.x);
            res_mat_len = (res_mat_len+tile_width-1) / tile_width;
        }

        T max;
        cudaMemcpy(&max, res_mat.data, sizeof(T), cudaMemcpyDeviceToHost);

        CUDA_KERNEL_TIME_ELAPSED(max_cuda);
        return max;
    };

    template <typename T>
    T min_cuda(const cv::cuda::GpuMat& src_mat){
        // 开始计时
        CUDA_KERNEL_TIME_RECORD(min_cuda);

        unsigned int rows = src_mat.rows;
        unsigned int cols = src_mat.cols;
        int block_len =1024; //320;
        int grid_len =10;  //30;

        dim3 block = dim3(block_len);
        int grid_x = (rows*cols+block_len-1)/(TILE_LEN*block_len) >grid_len ? grid_len :  (rows*cols+block_len-1)/(TILE_LEN*block_len);
        grid_x= (rows*cols+block_len-1)/(TILE_LEN*block_len) <1 ? 1 : grid_x;
        dim3 grid = dim3(grid_x);

        //prepare input, output
        cv::cuda::GpuMat res_mat(1, grid.x*block.x, src_mat.type(), cv::Scalar(0));

        d_min<T><<<grid, block>>>((T*) src_mat.data, cols*rows, (T*)res_mat.data);
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "0addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        int len = cols*rows;
        int tile_width = (TILE_LEN*grid.x*block.x) > len ? TILE_LEN : (len+ grid.x*block.x-1) / (grid.x*block.x);
        int res_mat_len = (len+tile_width-1) / tile_width;

        while(res_mat_len>1){
            if(res_mat_len<=block_len)
                block_len = block_len/TILE_LEN>0? block_len/TILE_LEN : 1;
            
            block.x=block_len;
            grid.x=(res_mat_len+block_len-1)/(TILE_LEN*block_len) >grid_len ? grid_len :  (rows*cols+block_len-1)/(TILE_LEN*block_len);
            grid.x=(res_mat_len+block_len-1)/(TILE_LEN*block_len) <1 ? 1 : grid.x;

            
            d_min<T><<<grid, block>>>((T*) res_mat.data, res_mat_len, (T*)res_mat.data);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "1addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }

            tile_width = (TILE_LEN*grid.x*block.x) > res_mat_len ? TILE_LEN : (res_mat_len+ grid.x*block.x-1) / (grid.x*block.x);
            res_mat_len = (res_mat_len+tile_width-1) / tile_width;
        }

        T min;
        cudaMemcpy(&min, res_mat.data, sizeof(T), cudaMemcpyDeviceToHost);

        CUDA_KERNEL_TIME_ELAPSED(min_cuda);
        return min;
    };

    float max_cuda(const cv::cuda::GpuMat& src_mat){
        if(src_mat.type()==CV_8U){
            uchar max = max_cuda<uchar>(src_mat);
            return float(max);
        }
        else if(src_mat.type()==CV_32F){
            return max_cuda<float>(src_mat);
        }else{
            std::cout<<"max_cuda: do not support type: "<<src_mat.type() <<std::endl;
        }
    }

    float min_cuda(const cv::cuda::GpuMat& src_mat){
        if(src_mat.type()==CV_8U){
            uchar max = min_cuda<uchar>(src_mat);
            return float(max);
        }
        else if(src_mat.type()==CV_32F){
            return min_cuda<float>(src_mat);
        }else{
            std::cout<<"min_cuda: do not support type: "<<src_mat.type() <<std::endl;
        }
    }

    // mat 角度计算
    cv::cuda::GpuMat arctan2_cuda(cv::cuda::GpuMat& angle_v, cv::cuda::GpuMat& angle_h) {
        CV_Assert(angle_v.type() == angle_h.type());
        CUDA_KERNEL_TIME_RECORD(arctan2_cuda);
        unsigned int rows = angle_v.rows;
        unsigned int cols = angle_v.cols;
        dim3 block, grid;
        config_grid_block(rows, cols, grid, block);

        //prepare input, output
        cv::cuda::GpuMat res_mat(rows, cols, angle_v.type(), cv::Scalar(0));

        if (angle_v.type() == CV_32F && angle_h.type() == CV_32F) {
            d_arctan2<float><<<grid, block>>>((float*) angle_v.data,(float*) angle_h.data, cols, rows, (float*)res_mat.data);
        } 
        else if (angle_v.type() == CV_64F && angle_h.type() == CV_64F) {
            std::cout<<"arctan2_cuda suggest CV_32F, it is very inefficient for CV_64F "<<std::endl;
            d_arctan2<double><<<grid, block>>>((double*) angle_v.data, (double*) angle_h.data, cols, rows, (double*)res_mat.data);
        }else{
            std::cout<<"arctan2_cuda: do not support type: "<<angle_v.type() <<std::endl;
        }
        // cudaDeviceSynchronize();
        CUDA_KERNEL_TIME_ELAPSED(arctan2_cuda);
        return res_mat;
    }

    cv::cuda::GpuMat sin_cuda(cv::cuda::GpuMat& angle) {
        CUDA_KERNEL_TIME_RECORD(sin_cuda);
        unsigned int rows = angle.rows;
        unsigned int cols = angle.cols;
        dim3 block, grid;
        config_grid_block(rows, cols, grid, block);

        //prepare input, output
        std::cout<<"2 prepare input, output"<<std::endl;
        cv::cuda::GpuMat res_mat(rows, cols, angle.type(), cv::Scalar(0));

        if (angle.type() == CV_32F ) {
            d_sin<float><<<grid, block>>>((float*) angle.data, cols, rows, (float*)res_mat.data);
        } 
        else if (angle.type() == CV_64F) {
            std::cout<<"sin_cuda suggest CV_32F, it is very inefficient for CV_64F "<<std::endl;
            d_sin<double><<<grid, block>>>((double*) angle.data, cols, rows, (double*)res_mat.data);
        }else{
            std::cout<<"sin_cuda: do not support type: "<<angle.type() <<std::endl;
        }
        // cudaDeviceSynchronize();
        CUDA_KERNEL_TIME_ELAPSED(sin_cuda);
        return res_mat;
    }

    cv::cuda::GpuMat cos_cuda(cv::cuda::GpuMat& angle) {
        CUDA_KERNEL_TIME_RECORD(cos_cuda);
        unsigned int rows = angle.rows;
        unsigned int cols = angle.cols;
        dim3 block, grid;
        config_grid_block(rows, cols, grid, block);

        //prepare input, output
        cv::cuda::GpuMat res_mat(rows, cols, angle.type(), cv::Scalar(0));

        if (angle.type() == CV_32F ) {
            d_cos<float><<<grid, block>>>((float*) angle.data, cols, rows, (float*)res_mat.data);
        } 
        else if (angle.type() == CV_64F) {
            std::cout<<"cos_cuda suggest CV_32F, it is very inefficient for CV_64F "<<std::endl;
            d_cos<double><<<grid, block>>>((double*) angle.data, cols, rows, (double*)res_mat.data);
        }else{
            std::cout<<"cos_cuda: do not support type: "<<angle.type() <<std::endl;
        }
        // cudaDeviceSynchronize();
        CUDA_KERNEL_TIME_ELAPSED(cos_cuda);
        return res_mat;
    }

    void angle_remainder(cv::cuda::GpuMat& angle, double remainder) {
        CUDA_KERNEL_TIME_RECORD(angle_remainder);
        unsigned int rows = angle.rows;
        unsigned int cols = angle.cols;
        dim3 block, grid;
        config_grid_block(rows, cols, grid, block);

        //prepare input, output
        if (angle.type() == CV_32F ) {
            d_remainder<<<grid, block>>>((float*) angle.data, cols, rows, float(remainder));
        } 
        else if (angle.type() == CV_64F) {
            std::cout<<"angle_remainder suggest CV_32F, it is very inefficient for CV_64F "<<std::endl;
            d_remainder<double><<<grid, block>>>((double*) angle.data, cols, rows, remainder);
        }else{
            std::cout<<"angle_remainder: do not support type: "<<angle.type() <<std::endl;
        }
        // cudaDeviceSynchronize();
        CUDA_KERNEL_TIME_ELAPSED(angle_remainder);
    }

    cv::cuda::GpuMat nonmaxsupress_cuda(cv::cuda::GpuMat& Gm, cv::cuda::GpuMat& Gd, float th = 1.0) {  //非最大值抑制算法
        CV_Assert(Gm.type()==Gd.type());
        CUDA_KERNEL_TIME_RECORD(nonmaxsupress_cuda);
        uint rows=Gm.rows;
        uint cols = Gm.cols;
        dim3  grid, block;
        config_grid_block(rows, cols, grid, block);

        cv::cuda::GpuMat out = cv::cuda::GpuMat(Gm.rows, Gm.cols, Gm.type(), cv::Scalar(0));
        if (Gm.type() == CV_32F ) {
            d_nonmaxsupress<float><<<grid, block>>>((float*) Gm.data, (float*) Gd.data, (float*) out.data, cols, rows, Gm.step/sizeof(float), th);
        } 
        else if (Gm.type() == CV_64F) {
            std::cout<<"nonmaxsupress suggest CV_32F, it is very inefficient for CV_64F "<<std::endl;
            d_nonmaxsupress<double><<<grid, block>>>((double*)Gm.data, (double*)Gd.data, (double*) out.data, cols, rows, Gm.step/sizeof(double), th);
        }else{
            std::cout<<"nonmaxsupress: do not support type: "<<Gm.type() <<std::endl;
        }

        CUDA_KERNEL_TIME_ELAPSED(nonmaxsupress_cuda);
        return out;
    }
    
    cv::cuda::GpuMat band_max(const std::vector<cv::cuda::GpuMat>& src_mat_vec ){
        for (int c = 1; c < src_mat_vec.size(); c++) {
            assert(src_mat_vec[0].type()== src_mat_vec[c].type() );
            assert(src_mat_vec[0].cols == src_mat_vec[c].cols);
            assert(src_mat_vec[0].rows == src_mat_vec[c].rows);
        }

        CUDA_KERNEL_TIME_RECORD(band_max);
        //  config blocks 
        uint rows=src_mat_vec[0].rows;
        uint cols = src_mat_vec[0].cols;
        dim3  grid, block;
        config_grid_block(rows*cols, grid, block);
    
        if (src_mat_vec[0].type() == CV_32F ) {
            cv::cuda::GpuMat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_32F);

            float ** d_src_mats;
            cudaMalloc((float ***) &d_src_mats, src_mat_vec.size()* sizeof(float *) ); 
            float ** h_src_mats= new float * [src_mat_vec.size()]; 
            for(int i=0;i<src_mat_vec.size(); i++){
                h_src_mats[i] = (float *)src_mat_vec[i].data;
            }
            cudaMemcpy(d_src_mats, h_src_mats, src_mat_vec.size()* sizeof(float *), cudaMemcpyHostToDevice);

            d_band_max<<<grid, block>>>(d_src_mats, src_mat_vec.size(), rows, cols, (float *)res_mat.data);
            
            CUDA_KERNEL_TIME_ELAPSED(band_max);
            delete[] h_src_mats;
            cudaFree(d_src_mats);
            return res_mat;
        }
        else if(src_mat_vec[0].type() == CV_64F ) {
            cv::cuda::GpuMat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_64F);

            double ** d_src_mats;
            cudaMalloc((double ***) &d_src_mats, src_mat_vec.size()* sizeof(double *) ); 
            double ** h_src_mats= new double * [src_mat_vec.size()]; 
            for(int i=0;i<src_mat_vec.size(); i++){
                h_src_mats[i] = (double *)src_mat_vec[i].data;
            }
            cudaMemcpy(d_src_mats, h_src_mats, src_mat_vec.size()* sizeof(double *), cudaMemcpyHostToDevice);

            d_band_max<<<grid, block>>>(d_src_mats, src_mat_vec.size(), rows, cols, (double *)res_mat.data);
            
            CUDA_KERNEL_TIME_ELAPSED(band_max);
            delete[] h_src_mats;
            cudaFree(d_src_mats);
            return res_mat;
        }
        else if(src_mat_vec[0].type() == CV_8U ) {
            cv::cuda::GpuMat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_8U);

            unsigned char ** d_src_mats;
            cudaMalloc((unsigned char ***) &d_src_mats, src_mat_vec.size()* sizeof(unsigned char *) ); 
            unsigned char ** h_src_mats= new unsigned char * [src_mat_vec.size()]; 
            for(int i=0;i<src_mat_vec.size(); i++){
                h_src_mats[i] = (unsigned char *)src_mat_vec[i].data;
            }
            cudaMemcpy(d_src_mats, h_src_mats, src_mat_vec.size()* sizeof(unsigned char *), cudaMemcpyHostToDevice);

            d_band_max<unsigned char><<<grid, block>>>(d_src_mats, src_mat_vec.size(), rows, cols, (unsigned char *)res_mat.data);
            
            CUDA_KERNEL_TIME_ELAPSED(band_max);
            delete[] h_src_mats;
            cudaFree(d_src_mats);
            return res_mat;
        }

    }

    cv::cuda::GpuMat band_max2(const std::vector<cv::cuda::GpuMat>& src_mat_vec ){
        for (int c = 1; c < src_mat_vec.size(); c++) {
            assert(src_mat_vec[0].type()== src_mat_vec[c].type() );
            assert(src_mat_vec[0].cols == src_mat_vec[c].cols);
            assert(src_mat_vec[0].rows == src_mat_vec[c].rows);
        }

        CUDA_KERNEL_TIME_RECORD(band_max2);
        //  config blocks 
        uint rows=src_mat_vec[0].rows;
        uint cols = src_mat_vec[0].cols;
        dim3  grid, block;
        config_grid_block(rows, cols, grid, block);
    
        if (src_mat_vec[0].type() == CV_32F ) {
            cv::cuda::GpuMat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_32F);

            float ** d_src_mats;
            cudaMalloc((float ***) &d_src_mats, src_mat_vec.size()* sizeof(float *) ); 
            float ** h_src_mats= new float * [src_mat_vec.size()]; 
            for(int i=0;i<src_mat_vec.size(); i++){
                h_src_mats[i] = (float *)src_mat_vec[i].data;
            }
            cudaMemcpy(d_src_mats, h_src_mats, src_mat_vec.size()* sizeof(float *), cudaMemcpyHostToDevice);

            d_band_max2<<<grid, block>>>(d_src_mats, src_mat_vec.size(), rows, cols, (float *)res_mat.data);
            
            CUDA_KERNEL_TIME_ELAPSED(band_max2);
            delete[] h_src_mats;
            cudaFree(d_src_mats);
            return res_mat;
        }
        else if(src_mat_vec[0].type() == CV_64F ) {
            cv::cuda::GpuMat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_64F);

            double ** d_src_mats;
            cudaMalloc((double ***) &d_src_mats, src_mat_vec.size()* sizeof(double *) ); 
            double ** h_src_mats= new double * [src_mat_vec.size()]; 
            for(int i=0;i<src_mat_vec.size(); i++){
                h_src_mats[i] = (double *)src_mat_vec[i].data;
            }
            cudaMemcpy(d_src_mats, h_src_mats, src_mat_vec.size()* sizeof(double *), cudaMemcpyHostToDevice);

            d_band_max2<<<grid, block>>>(d_src_mats, src_mat_vec.size(), rows, cols, (double *)res_mat.data);
            
            CUDA_KERNEL_TIME_ELAPSED(band_max2);
            delete[] h_src_mats;
            cudaFree(d_src_mats);
            return res_mat;
        }
        else if(src_mat_vec[0].type() == CV_8U ) {
            cv::cuda::GpuMat res_mat(src_mat_vec[0].rows, src_mat_vec[0].cols, CV_8U);

            unsigned char ** d_src_mats;
            cudaMalloc((unsigned char ***) &d_src_mats, src_mat_vec.size()* sizeof(unsigned char *) ); 
            unsigned char ** h_src_mats= new unsigned char * [src_mat_vec.size()]; 
            for(int i=0;i<src_mat_vec.size(); i++){
                h_src_mats[i] = (unsigned char *)src_mat_vec[i].data;
            }
            cudaMemcpy(d_src_mats, h_src_mats, src_mat_vec.size()* sizeof(unsigned char *), cudaMemcpyHostToDevice);

            d_band_max2<unsigned char><<<grid, block>>>(d_src_mats, src_mat_vec.size(), rows, cols, (unsigned char *)res_mat.data);
            
            CUDA_KERNEL_TIME_ELAPSED(band_max2);
            delete[] h_src_mats;
            cudaFree(d_src_mats);
            return res_mat;
        }
    }

    // cv::cuda::GpuMat remove_small_objects(cv::cuda::GpuMat& source_img, int area_thr, int connectivity = 8) {
    //     cv::cuda::GpuMat objects_to_remove = RSMat::zeros_like_cuda(source_img);
    //     cv::cuda::GpuMat label_img, mask, centroids;

    //     cv::cuda::GpuMat mask;
    //     mask.create(source_img.rows, source_img.cols, CV_8UC1);

    //     cv::cuda::GpuMat components;
    //     components.create(source_img.rows, source_img.cols, CV_32SC1);
    
    //     cv::cuda::connectivityMask(source_img, mask, cv::Scalar::all(0), cv::Scalar::all(2));
    //     cv::cuda::labelComponents(mask, components);


    //     // cv::cuda::GpuMat areas = stats.col(4);  //[:, 4];
    //     // for (int i = 1; i < num_labels; i++) {
    //     //     if (areas.at<int>(i) <= area_thr) objects_to_remove.setTo(1, label_img == i);
    //     // }
    //     // cv::cuda::GpuMat objects_to_keep = RSMat::logical_not(objects_to_remove);
    //     // cv::cuda::GpuMat res = RSMat::logical_and(source_img, objects_to_keep);
    //     // return res;

    //     return components;
    // }
}

#endif