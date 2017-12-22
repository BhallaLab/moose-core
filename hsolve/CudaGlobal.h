#ifndef __CUDA_GLOBAL_H__
#define __CUDA_GLOBAL_H__

//#define USE_CUDA

#ifdef USE_CUDA

#define BLOCK_WIDTH 64
#define THREADS_PER_BLOCK 256

#ifndef DEBUG_
#define DEBUG_
#endif

#define CUDA_ERROR_CHECK
#define PIN_POINT_ERROR

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cstdio>
#include <string>

typedef unsigned int u32;
typedef unsigned long long u64;

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define cusparseSafeCall( err ) __cusparseSafeCall( err, __FILE__, __LINE__ )
 
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}
 
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}

inline void __cusparseSafeCall( cusparseStatus_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( CUSPARSE_STATUS_SUCCESS != err )
    {
        std::string errMsg;
        switch(err){
            case CUSPARSE_STATUS_SUCCESS:
                errMsg =  "the operation completed successfully.";
                break;
            case CUSPARSE_STATUS_NOT_INITIALIZED:
                errMsg =  "the library was not initialized.";
                break;
            case CUSPARSE_STATUS_INVALID_VALUE:
                errMsg =  "invalid parameters were passed (mb,nnzb<=0).";
                break;
            case CUSPARSE_STATUS_ARCH_MISMATCH:
                errMsg =  "the device only supports compute capability 2.0 and above.";
                break;
            case CUSPARSE_STATUS_MAPPING_ERROR:
                errMsg =  "the texture binding failed.";
                break;
            case CUSPARSE_STATUS_EXECUTION_FAILED:
                errMsg =  "the function failed to launch on the GPU.";
                break;
            case CUSPARSE_STATUS_INTERNAL_ERROR:
                errMsg =  "an internal operation failed.";
                break;
            case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                errMsg =  "the matrix type is not supported.";
                break;
            default:
                errMsg = "Unknown cusparse error";
                break;
        }

        fprintf( stderr, "cusparseSafeCall() failed at %s:%i : %s\n",
                 file, line, errMsg.c_str() );
        exit( -1 );
    }
#endif
 
    return;
}

#endif //USE_CUDA

#endif //  __CUDA_GLOBAL_H__
