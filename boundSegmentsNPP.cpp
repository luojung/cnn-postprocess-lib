
/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "boundSegmentsNPP.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

// #include <ImagesCPU.h>
// #include <ImagesNPP.h>
// #include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>
#include "opencv2/opencv.hpp"

bool printfNPPinfo()
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    return true;
}

cv::cuda::GpuMat nppiLabelMarker(cv::cuda::GpuMat src){
    printfNPPinfo();
    cv::cuda::GpuMat oDeviceDst32u(src.cols, src.rows, CV_32S);

    try
    {
        const char * argv ="";
        int dev = findCudaDevice(1, &argv);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

        NppiSize oSrcSize = {src.cols, src.rows};
        NppiPoint oSrcOffset = {0, 0};

        // create struct with ROI size
        NppiSize oSizeROI = {src.cols, src.rows };
        // allocate device image of appropriately reduced size
        cv::cuda::GpuMat oDeviceDst8u(src.cols, src.rows, CV_8U);

        int nBufferSize = 0;
        Npp8u * pScratchBufferNPP = 0;

        // get necessary scratch buffer size and allocate that much device memory
        NPP_CHECK_NPP (
                           nppiLabelMarkersGetBufferSize_8u32u_C1R(oSizeROI, &nBufferSize) );

        cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

        // Now generate label markers using 8 way search mode (nppiNormInf).
        // For this particular sample image, any pixel value below 165 will be considered outside of any connected region and will be labeled 0.
        // The 8u32u version of the function is used because for this particular image the maximum number of generated labels is larger than 8-bit (256)
        // so an 8 bit per pixel output image cannot contain accurate labeling results.
        int maxLabel = 0;

        if ((nBufferSize > 0) && (pScratchBufferNPP != 0))
        {
            NPP_CHECK_NPP (
                               nppiLabelMarkers_8u32u_C1R(src.data, 0, 
                                                          reinterpret_cast<Npp32u *>(oDeviceDst32u.data), 0, oSizeROI,
                                                          165, nppiNormInf, &maxLabel, pScratchBufferNPP) );
        }


        // free scratch buffer memory
        cudaFree(pScratchBufferNPP);

        // The generated list of labels is likely to be very sparsely populated so it might be possible to compress them into a label list that
        // will fit into 8 bits.
        // 
        // Get necessary scratch buffer size and allocate that much device memory
        NPP_CHECK_NPP (
                           nppiCompressMarkerLabelsGetBufferSize_32u8u_C1R(maxLabel, &nBufferSize) );

        cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

        if ((nBufferSize > 0) && (pScratchBufferNPP != 0))
        {

            NPP_CHECK_NPP (
                               nppiCompressMarkerLabels_32u8u_C1R(reinterpret_cast<Npp32u *>(oDeviceDst32u.data), 0,
                                                                  oDeviceDst8u.data, 0, oSizeROI, maxLabel,
                                                                  &maxLabel, pScratchBufferNPP) );
        }

        // free scratch buffer memory
        cudaFree(pScratchBufferNPP);

        // Since the maximum label value after label compression is less than 256 then the label pixels were succesfully compressed from 32u in data size
        // back down to 8u in size.

        // Add segment boundaries to the final resulting labeled segments using pixel value 255 as a boundary value.
        // NPP_CHECK_NPP (
        //                    nppiBoundSegments_8u_C1IR(oDeviceDst8u.data, 0, oSizeROI, 255 ) );

        // // Scale the final label values to improve contrast in visibility of the various connected region segments in this particular result image.
        // NPP_CHECK_NPP (
        //                    nppiMulC_8u_C1IRSfs(2, oDeviceDst8u.data, 0, oSizeROI, 0) );

        // // Brighten everything for improved visibility
        // NPP_CHECK_NPP (
        //                    nppiAddC_8u_C1IRSfs(96, oDeviceDst8u.data, 0, oSizeROI, 0 ) );

        
        // exit(EXIT_SUCCESS);
    }

    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return oDeviceDst32u;
    }

    return oDeviceDst32u;
}
