
#pragma once
#define CHECK(call)                                                         \
    do                                                                      \
    {                                                                       \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA Error\n");                                         \
            printf("    File:   %s\n", __FILE__);                           \
            printf("    Line:   %d\n", __LINE__);                           \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)


#define TIME_RECORD(t1)        \
    auto start##t1 = std::chrono::system_clock::now(); 

#define TIME_ELAPSED(t1)                                                             \
    auto end##t1 = std::chrono::system_clock::now();                                  \
    auto elapsed##t1 = std::chrono::duration<double,std::milli>(end##t1 - start##t1); \
    std::cout << #t1<< " func elapsed time: " <<elapsed##t1.count()<<"ms"<<std::endl<< std::endl;         

#define CUDA_KERNEL_TIME_RECORD(t1)        \
    std::cout<< #t1 <<" executing ... "<<std::endl; \
    cudaEvent_t cuda_t1##t1, cuda_t2##t1;  \
    float time_ms##t1 = 0.f;               \   
    CHECK(cudaEventCreate(&cuda_t1##t1));  \
    CHECK(cudaEventCreate(&cuda_t2##t1));  \
    cudaEventRecord(cuda_t1##t1);          \


#define CUDA_KERNEL_TIME_ELAPSED(t1)                                       \
    cudaEventRecord(cuda_t2##t1);                                          \
    CHECK(cudaGetLastError());                                             \
    cudaEventSynchronize(cuda_t2##t1);                                     \   
    cudaEventElapsedTime(&time_ms##t1, cuda_t1##t1, cuda_t2##t1);          \
    std::cout << #t1<< " func GPU elapsed time: " << time_ms##t1 << " ms" << std::endl<< std::endl;\
