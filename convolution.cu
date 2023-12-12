/*
 * @Description: 
 * @Author: luojun1
 * @Date: 2022-02-16 17:54:46
 * @LastEditTime: 2022-05-08 19:53:14
 */


#ifndef GPUMAT_OP_KERNEL
#define GPUMAT_OP_KERNEL

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math_forward_declares.h>
#include <cinttypes>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core_c.h>
#include "opencv2/opencv.hpp"

#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include "postprocesslib/utils/record_time.h"
#include "utils/record_time.h"
#include "mat_gpu.hpp"
#include "bwmorph_cudnn.hpp"

#define uint unsigned int
#define uint8 unsigned char
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
__global__ void d_Convolution_shm(T* A, T* B, int Rows, int Cols, int step, T* kernel, int kernel_size){
    int pad = kernel_size/2;

    int _Rows=( (Rows+gridDim.y*(blockDim.y-2*pad)-1) / (gridDim.y*(blockDim.y-2*pad) ) ) * (gridDim.y*(blockDim.y-2*pad));
    int _Cols=( (Cols+gridDim.x*(blockDim.x-2*pad)-1) / (gridDim.x*(blockDim.x-2*pad) ) ) * (gridDim.x*(blockDim.x-2*pad));  // void branch divergenrence

    // int y_time =0;
    // int x_time =0;
    // int y_times = (Rows+gridDim.y*(blockDim.y-2*pad)-1 )/gridDim.y*(blockDim.y-2*pad);
    // int x_times = (Cols+gridDim.x*(blockDim.x-2*pad)-1)/gridDim.x*(blockDim.x-2*pad);

    __shared__ T shm[BLOCK_WIDTH][BLOCK_WIDTH];

    for( int y = blockIdx.y*(blockDim.y-2*pad) + threadIdx.y -pad; y<Rows+pad; y+=gridDim.y*(blockDim.y-2*pad)){
        for( int x = blockIdx.x*(blockDim.x-2*pad) + threadIdx.x -pad; x<Cols+pad; x+=gridDim.x*(blockDim.x-2*pad)){
    // for( int y = blockIdx.y*(blockDim.y-2*pad) + threadIdx.y -pad; y<_Rows+pad; y+=gridDim.y*(blockDim.y-2*pad)){
    //     for( int x = blockIdx.x*(blockDim.x-2*pad) + threadIdx.x -pad; x<_Cols+pad; x+=gridDim.x*(blockDim.x-2*pad)){
    // for( int y = blockIdx.y*(blockDim.y-2*pad) + threadIdx.y -pad; y_time<y_times; y+=gridDim.y*(blockDim.y-2*pad), y_time++){
    //     for( int x = blockIdx.x*(blockDim.x-2*pad) + threadIdx.x -pad; x_time<x_times; x+=gridDim.x*(blockDim.x-2*pad), x_time++){

            if (x < Cols && x >= 0 && y < Rows && y >= 0){
                shm[threadIdx.y][threadIdx.x] = A[y * step + x];
            }
            else{
                shm[threadIdx.y][threadIdx.x] = 0;
            }
            __syncthreads();

            if (pad<=threadIdx.y && threadIdx.y < (BLOCK_WIDTH-pad)  && pad<=threadIdx.x && threadIdx.x <(BLOCK_WIDTH-pad) &&  y<Rows+pad &&x<Cols+pad) // && row < (WB - WC + 1) && col < (WB - WC + 1))
            {
                T tmp = 0;
                for (int i = -pad; i<=pad;i++){
                    for (int j = -pad;j<=pad;j++){
                        tmp += shm[threadIdx.y + i][threadIdx.x + j] * kernel[(i+pad)*kernel_size + j+pad];
                    }
                }
                // B[y*step + x] = tmp;  // illegal
                // tmp =B[y*step + x] ; // ok
                // B[y*step + x] =  A[y * step + x]; //  illegal
                // B[y*step + x] =  0; //  illegal
                // A[y * step + x]=0;   //ok
                // 不能被读写
                B[y*Cols + x] =tmp;
            }
            __syncthreads();
        }
    }
}

template <typename T>
__global__ void d_Convolution(T* A, T* B, int Rows, int Cols, int step, T* kernel, int kernel_size){
    int pad = kernel_size/2;

    for( int y = blockIdx.y*(blockDim.y) + threadIdx.y +pad; y<Rows -pad; y+=gridDim.y*(blockDim.y)){
        for( int x = blockIdx.x*(blockDim.x) + threadIdx.x+pad; x<Cols -pad; x+=gridDim.x*(blockDim.x)){
            T tmp = 0;
            for (int i = -pad; i<=pad;i++){
                for (int j = -pad;j<=pad;j++){
                    if( 0<=(y+i) && (y+i)<Rows &&  0<=(x+i) && (x+i)<Cols )
                        tmp += A[ (y+i)*step + x+j ] * kernel[(i+pad)*kernel_size + j+pad];
                }
            }
  
            B[y*step + x] = tmp;
        }
    }
}

template <typename T>
__global__ void d_count_NonZero(T* bw_tip, unsigned int size, unsigned int* point_num){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int block_counter;

    for(unsigned int i = index; i< size; i+= blockDim.x* gridDim.x){
        if(bw_tip[i]!=0){
            atomicAdd(&block_counter, 1);
        }
    }

    __syncthreads();
    if(threadIdx.x == 0)
        atomicAdd(point_num, block_counter);
}

template <typename T>
__global__ void d_find_NonZero(T* bw_tip, int Rows, int Cols,int step, unsigned int* coods, unsigned int point_num){
    for(unsigned int y = blockIdx.y*blockDim.y+threadIdx.y; y< Rows; y+= gridDim.y*blockDim.y){
        for(unsigned int x = blockIdx.x*blockDim.x+threadIdx.x; x< Cols; x+= gridDim.x*blockDim.x){
            if(bw_tip[y*step + x] != 0){
                unsigned int point_idx=atomicAdd(&(coods[2*point_num-1]), 1);
                coods[2*point_idx] = y;
                coods[2*point_idx+1] = x;
                // printf("count: %d ", point_idx);
            }
        }
    }

    __syncthreads();
    // if(threadIdx.x == 0)
}

template <typename T>
__global__ void d_stitch_line(T* edge, int Rows, int Cols,int step, unsigned int* tip_coods,
                                     unsigned int point_num, int nbh_thr, int ext_thr){
    unsigned int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int points[20][2];
    points[0][0]=tip_coods[point_idx*2];;
    points[0][1]=tip_coods[point_idx*2+1 ];

    //tranverse 
    int i=1;
    for( ; i< 5; i++){
        bool branch = false;
        // breadth-first traversal
        unsigned int nxt_y = points[i-1][0] -1;
        unsigned int nxt_x = points[i-1][1] -1;
        if(nxt_y>0 && nxt_x>0 && edge[nxt_y* step+nxt_x ] !=0 
            && (i<2 || nxt_y!=points[i-2][0] &&  nxt_x!=points[i-2][1])){
            points[i][0]=nxt_y;
            points[i][1]=nxt_x;
            branch =true;
        }

        nxt_y = points[i-1][0] -1;
        nxt_x = points[i-1][1];
        if(nxt_y>0 && edge[nxt_y* step+nxt_x ] !=0 
            && (i<2 || nxt_y!=points[i-2][0] &&  nxt_x!=points[i-2][1])){
            if(branch){
                //select next point which keeps the direction of previous points
                float pre_angle = atan((points[i-2][0]-points[i-1][0]) / (points[i-2][1]-points[i-1][1]) );
                float nxt_angle = atan((points[i-1][0]-points[i][0]) / (points[i-1][1]-points[i][1]) );
                float nxt_angle2 = atan((points[i-1][0]-nxt_y) / (points[i-1][1]-nxt_x) );

                if(abs(pre_angle-nxt_angle) > abs(pre_angle-nxt_angle2)){
                    points[i][0]=nxt_y;
                    points[i][1]=nxt_x;
                }

            }else{
                points[i][0]=nxt_y;
                points[i][1]=nxt_x;
                branch =true;
            }
        }

        nxt_y = points[i-1][0] -1;
        nxt_x = points[i-1][1]+1;
        if(nxt_y>0 && nxt_x <Cols && edge[nxt_y* step+nxt_x ] !=0 
            && (i<2 || nxt_y!=points[i-2][0] &&  nxt_x!=points[i-2][1])){
            if(branch){
                //select next point which keeps the direction of previous points
                float pre_angle = atan((points[i-2][0]-points[i-1][0]) / (points[i-2][1]-points[i-1][1]) );
                float nxt_angle = atan((points[i-1][0]-points[i][0]) / (points[i-1][1]-points[i][1]) );
                float nxt_angle2 = atan((points[i-1][0]-nxt_y) / (points[i-1][1]-nxt_x) );

                if(abs(pre_angle-nxt_angle) > abs(pre_angle-nxt_angle2)){
                    points[i][0]=nxt_y;
                    points[i][1]=nxt_x;
                }
            }else{
                points[i][0]=nxt_y;
                points[i][1]=nxt_x;
                branch =true;
            }
        }

        nxt_y = points[i-1][0];
        nxt_x = points[i-1][1]-1;
        if(nxt_x >0 && edge[nxt_y* step+nxt_x ] !=0 
            && (i<2 || nxt_y!=points[i-2][0] &&  nxt_x!=points[i-2][1])){
            if(branch){
                //select next point which keeps the direction of previous points
                float pre_angle = atan((points[i-2][0]-points[i-1][0]) / (points[i-2][1]-points[i-1][1]) );
                float nxt_angle = atan((points[i-1][0]-points[i][0]) / (points[i-1][1]-points[i][1]) );
                float nxt_angle2 = atan((points[i-1][0]-nxt_y) / (points[i-1][1]-nxt_x) );

                if(abs(pre_angle-nxt_angle) > abs(pre_angle-nxt_angle2)){
                    points[i][0]=nxt_y;
                    points[i][1]=nxt_x;
                }
            }else{
                points[i][0]=nxt_y;
                points[i][1]=nxt_x;
                branch =true;
            }
        }

        nxt_y = points[i-1][0];
        nxt_x = points[i-1][1]+1;
        if(nxt_x <Cols && edge[nxt_y* step+nxt_x ] !=0 
            && (i<2 || nxt_y!=points[i-2][0] &&  nxt_x!=points[i-2][1])){
            if(branch){
                //select next point which keeps the direction of previous points
                 float pre_angle = atan((points[i-2][0]-points[i-1][0]) / (points[i-2][1]-points[i-1][1]) );
                float nxt_angle = atan((points[i-1][0]-points[i][0]) / (points[i-1][1]-points[i][1]) );
                float nxt_angle2 = atan((points[i-1][0]-nxt_y) / (points[i-1][1]-nxt_x) );

                if(abs(pre_angle-nxt_angle) > abs(pre_angle-nxt_angle2)){
                    points[i][0]=nxt_y;
                    points[i][1]=nxt_x;
                }
            }else{
                points[i][0]=nxt_y;
                points[i][1]=nxt_x;
                branch =true;
            }
        }

        nxt_y = points[i-1][0] +1;
        nxt_x = points[i-1][1] -1;
        if(nxt_y<Rows && nxt_x >0 && edge[nxt_y* step+nxt_x ] !=0 
            && (i<2 || nxt_y!=points[i-2][0] &&  nxt_x!=points[i-2][1])){
            if(branch){
                //select next point which keeps the direction of previous points
                float pre_angle = atan((points[i-2][0]-points[i-1][0]) / (points[i-2][1]-points[i-1][1]) );
                float nxt_angle = atan((points[i-1][0]-points[i][0]) / (points[i-1][1]-points[i][1]) );
                float nxt_angle2 = atan((points[i-1][0]-nxt_y) / (points[i-1][1]-nxt_x) );

                if(abs(pre_angle-nxt_angle) > abs(pre_angle-nxt_angle2)){
                    points[i][0]=nxt_y;
                    points[i][1]=nxt_x;
                }

            }else{
                points[i][0]=nxt_y;
                points[i][1]=nxt_x;
                branch =true;
            }
        }

        nxt_y = points[i-1][0] +1;
        nxt_x = points[i-1][1] ;
        if(nxt_y<Rows && edge[nxt_y* step+nxt_x ] !=0 
            && (i<2 || nxt_y!=points[i-2][0] &&  nxt_x!=points[i-2][1])){
            if(branch){
                //select next point which keeps the direction of previous points
                 float pre_angle = atan((points[i-2][0]-points[i-1][0]) / (points[i-2][1]-points[i-1][1]) );
                float nxt_angle = atan((points[i-1][0]-points[i][0]) / (points[i-1][1]-points[i][1]) );
                float nxt_angle2 = atan((points[i-1][0]-nxt_y) / (points[i-1][1]-nxt_x) );

                if(abs(pre_angle-nxt_angle) > abs(pre_angle-nxt_angle2)){
                    points[i][0]=nxt_y;
                    points[i][1]=nxt_x;
                }
            }else{
                points[i][0]=nxt_y;
                points[i][1]=nxt_x;
                branch =true;
            }
        }

        nxt_y = points[i-1][0] +1;
        nxt_x = points[i-1][1]+1;
        if(nxt_y<Rows && nxt_x <Cols && edge[nxt_y* step+nxt_x ] !=0 
            && (i<2 || nxt_y!=points[i-2][0] &&  nxt_x!=points[i-2][1])){
            if(branch){
                //select next point which keeps the direction of previous points
                float pre_angle = atan((points[i-2][0]-points[i-1][0]) / (points[i-2][1]-points[i-1][1]) );
                float nxt_angle = atan((points[i-1][0]-points[i][0]) / (points[i-1][1]-points[i][1]) );
                float nxt_angle2 = atan((points[i-1][0]-nxt_y) / (points[i-1][1]-nxt_x) );

                if(abs(pre_angle-nxt_angle) > abs(pre_angle-nxt_angle2)){
                    points[i][0]=nxt_y;
                    points[i][1]=nxt_x;
                }
            }else{
                points[i][0]=nxt_y;
                points[i][1]=nxt_x;
                branch =true;
            }
        }

        // breakk
        if(!branch){
            break;
        }
    }

    // calculate average value
    unsigned int center_point_y=0,center_point_x=0;
    for(int j =0; j<i; j++){
        center_point_y+=points[j][0];
        center_point_x+=points[j][1];
    }
    center_point_x=center_point_x/i;
    center_point_y=center_point_y/i;

    // calculate direction vector
    
    
    // extend line
    
}


namespace RSMat{
    /*Convolution Wrapper*/

    cv::cuda::GpuMat convolution( const cv::cuda::GpuMat input, const cv::Mat h_k){
        CV_Assert(input.type()==h_k.type());
        // 开始计时
        CUDA_KERNEL_TIME_RECORD(convolution);

        //  config blocks 
        dim3  grid, block;
        config_grid_block(input.rows, input.cols, grid, block);

        //prepare input, output
        cv::cuda::GpuMat output;
        printf("rows: %d  cols: %d steps: %d output_steps: %d \n",input.rows, input.cols, input.step, output.step);

        if(input.type()==CV_8U){
            for(int i =0; i<9;i++){
                std::cout<<+((unsigned char*)h_k.data)[i]<<" "; 
            }

            uint8* d_kernel =NULL;
            cudaMalloc((void**)&d_kernel, h_k.rows*h_k.cols*sizeof(uint8));
            cudaMemcpy(d_kernel, h_k.data, h_k.rows*h_k.cols*sizeof(uint8), cudaMemcpyHostToDevice);

            uint8* output_ptr =NULL;
            cudaMalloc((void**)&output_ptr, input.rows*input.cols*sizeof(uint8));

            d_Convolution_shm<uint8><<<grid,block>>>((uint8*)input.data, (uint8*)output_ptr, input.rows, input.cols, input.step/sizeof(uint8), d_kernel, h_k.rows);
        
            cv::cuda::GpuMat res(input.rows, input.cols, input.type(), output_ptr, input.cols*sizeof(uint8));
            output = res;
            cudaFree(d_kernel);
        }
        else if(input.type()==CV_32F){
            float* d_kernel =NULL;
            cudaMalloc((void**)&d_kernel, h_k.rows*h_k.cols*sizeof(float));
            cudaMemcpy(d_kernel, h_k.data, h_k.rows*h_k.cols*sizeof(float), cudaMemcpyHostToDevice);

            uint8* output_ptr =NULL;
            cudaMalloc((void**)&output_ptr, input.rows*input.cols*sizeof(float));

            d_Convolution_shm<float><<<grid,block>>>((float*)input.data, (float*)output.data, input.rows, input.cols, input.step/sizeof(float), d_kernel, h_k.rows);
     
            cv::cuda::GpuMat res(input.rows, input.cols, input.type(), output_ptr, input.cols*sizeof(float));
            output = res;

            cudaFree(d_kernel);
        }
        else if(input.type()==CV_32S){
            uint* d_kernel =NULL;
            cudaMalloc((void**)&d_kernel, h_k.rows*h_k.cols*sizeof(uint));
            cudaMemcpy(d_kernel, h_k.data, h_k.rows*h_k.cols*sizeof(uint), cudaMemcpyHostToDevice);

            uint8* output_ptr =NULL;
            cudaMalloc((void**)&output_ptr, input.rows*input.cols*sizeof(uint));

            d_Convolution_shm<uint><<<grid,block>>>((uint*)input.data, (uint*)output.data, input.rows, input.cols, input.step/sizeof(uint), d_kernel, h_k.rows);

            cv::cuda::GpuMat res(input.rows, input.cols, input.type(), output_ptr, input.cols*sizeof(uint));
            output = res;
            cudaFree(d_kernel);
        }else{
            std::cout<<input.type()<<" type is not supported yet "<<std::endl;
            // std::exception e;
            // throw e;
        }
        cudaDeviceSynchronize();
        printf("rows: %d  cols: %d steps: %d output_steps: %d \n",input.rows, input.cols, input.step, output.step);
        
        CUDA_KERNEL_TIME_ELAPSED(convolution);
        RSMat::imwrite( "../data/bw_tip.tif", output);
        return output;
    }

    unsigned int count_nonZero(cv::cuda::GpuMat bw_tip){  //获取骨架线的端点坐标
        CV_Assert(bw_tip.type()==CV_8U);
        // 开始计时
        CUDA_KERNEL_TIME_RECORD(count_nonZero);

        //  config blocks 
        dim3  grid, block;
        config_grid_block(bw_tip.rows*bw_tip.cols, grid, block);

        unsigned int num = 0;
        unsigned int* d_num=NULL;
        cudaMalloc((void**)&d_num, sizeof(unsigned int));
        cudaMemcpy(d_num, &num, sizeof(unsigned int), cudaMemcpyHostToDevice);

        d_count_NonZero<uint8><<<grid,block>>>((uint8*)bw_tip.data, bw_tip.rows*bw_tip.cols,d_num);
        cudaMemcpy( &num,d_num, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaFree(d_num);
        CUDA_KERNEL_TIME_ELAPSED(count_nonZero);
        
        return num;
    }

    cv::cuda::GpuMat find_tip_coords(cv::cuda::GpuMat edge_rm_thin ){  //获取骨架线的端点坐标
        // CV_Assert(edge_rm_thin.type()==CV_8U);
        // 开始计时
        CUDA_KERNEL_TIME_RECORD(find_tip_coords);

        cv::cuda::GpuMat bw_tip = RSMat::endpoints(edge_rm_thin);

        unsigned int* coods =NULL;
        unsigned int point_num = count_nonZero( bw_tip);  //获取骨架线的端点数量
        cudaMalloc((void**) &coods, point_num*2*sizeof(unsigned int));
        cout<<"point_num:"<<point_num<<endl;

        //  config blocks 
        dim3  grid, block;
        config_grid_block(bw_tip.rows, bw_tip.cols, grid, block);

        CUDA_KERNEL_TIME_RECORD(d_find_NonZero);
        d_find_NonZero<uint8><<<grid,block>>>((uint8*)bw_tip.data,  bw_tip.rows, bw_tip.cols, bw_tip.step, coods, point_num);
        CUDA_KERNEL_TIME_ELAPSED(d_find_NonZero);

        CUDA_KERNEL_TIME_ELAPSED(find_tip_coords);
  
        cv::cuda::GpuMat tip_coords(point_num, 2, CV_32S, coods, 2*sizeof(unsigned int));
        // cudaFree(coods);
        return tip_coords;
    }

    cv::cuda::GpuMat stitch_line(cv::cuda::GpuMat tip_coords, cv::cuda::GpuMat edge_rm_thin, int nbh_thr, int ext_thr){
        CUDA_KERNEL_TIME_RECORD(stitch_line);

        d_stitch_line(T* edge, int Rows, int Cols,int step, unsigned int* coods, 
                         unsigned int point_num, int nbh_thr, int ext_thr);


        CUDA_KERNEL_TIME_ELAPSED(stitch_line);

    }
}

#endif