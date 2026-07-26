/*-------------------------------------------------------------------
Copyright 2026 JDFTx developers

This file is part of JDFTx and is distributed under the GNU GPL v3
or later. See COPYING for details.
-------------------------------------------------------------------*/

#ifndef JDFTX_CORE_GPUAPI_H
#define JDFTX_CORE_GPUAPI_H

//Keep CUDA spellings in the implementation and translate them at this small
//backend boundary. This avoids maintaining separate CUDA and HIP kernels.
#ifdef GPU_BACKEND_HIP

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipfft/hipfft.h>
#ifdef CUSOLVER_ENABLED
#include <hipsolver/hipsolver.h>
#endif

//HIP runtime:
#define cudaDeviceProp hipDeviceProp_t
#define cudaFuncAttributes hipFuncAttributes
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaCpuDeviceId hipCpuDeviceId
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaFuncGetAttributes(attr, func) hipFuncGetAttributes((attr), reinterpret_cast<const void*>(func))
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaSetDevice hipSetDevice

//hipBLAS:
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define cublasFillMode_t hipblasFillMode_t
#define cublasCreate hipblasCreate
#define cublasDasum hipblasDasum
#define cublasDaxpy_v2 hipblasDaxpy
#define cublasDdot_v2 hipblasDdot
#define cublasDnrm2_v2 hipblasDnrm2
#define cublasDscal_v2 hipblasDscal
#define cublasDznrm2_v2 hipblasDznrm2_v2
#define cublasZaxpy_v2 hipblasZaxpy_v2
#define cublasZdotc_v2 hipblasZdotc_v2
#define cublasZdscal_v2 hipblasZdscal_v2
#define cublasZgemm hipblasZgemm_v2
#define cublasZscal_v2 hipblasZscal_v2
#define cublasZtrsm hipblasZtrsm_v2
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT

//hipFFT:
#define cufftHandle hipfftHandle
#define cufftDestroy hipfftDestroy
#define cufftExecD2Z hipfftExecD2Z
#define cufftExecZ2D hipfftExecZ2D
#define cufftExecZ2Z hipfftExecZ2Z
#define cufftPlan3d hipfftPlan3d
#define CUFFT_D2Z HIPFFT_D2Z
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define CUFFT_Z2D HIPFFT_Z2D
#define CUFFT_Z2Z HIPFFT_Z2Z

//hipSOLVER (optional):
#ifdef CUSOLVER_ENABLED
#define cusolverDnHandle_t hipsolverHandle_t
#define cusolverEigMode_t hipsolverEigMode_t
#define gesvdjInfo_t hipsolverGesvdjInfo_t
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnCreateGesvdjInfo hipsolverDnCreateGesvdjInfo
#define cusolverDnDestroyGesvdjInfo hipsolverDnDestroyGesvdjInfo
#define cusolverDnZgesvdj hipsolverDnZgesvdj
#define cusolverDnZgesvdj_bufferSize hipsolverDnZgesvdj_bufferSize
#define cusolverDnZgetrf hipsolverDnZgetrf
#define cusolverDnZgetrf_bufferSize hipsolverDnZgetrf_bufferSize
#define cusolverDnZgetrs hipsolverDnZgetrs
#define cusolverDnZheevd hipsolverDnZheevd
#define cusolverDnZheevd_bufferSize hipsolverDnZheevd_bufferSize
#define cusolverDnZpotrf hipsolverDnZpotrf
#define cusolverDnZpotrf_bufferSize hipsolverDnZpotrf_bufferSize
#define cusolverDnZpotrs hipsolverDnZpotrs
#define CUSOLVER_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#endif

#else //CUDA backend

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#ifdef CUSOLVER_ENABLED
#include <cusolverDn.h>
#endif

#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GPU_DEVICE_COMPILE
#endif

#endif //JDFTX_CORE_GPUAPI_H
