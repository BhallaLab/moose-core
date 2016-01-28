#ifndef __CUDA_GLOBAL_H__
#define __CUDA_GLOBAL_H__

#define USE_CUDA

#ifdef USE_CUDA

#define BLOCK_WIDTH 64
#define THREADS_PER_BLOCK 512

#ifndef DEBUG_
#define DEBUG_
#endif

#ifdef DEBUG_

//#define DEBUG_VERBOSE
//#define DEBUG_STEP

#endif //DEBUG_

#define CUDA_ERROR_CHECK

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

typedef unsigned int u32;
typedef unsigned long long u64;
typedef u64 ChannelData;

const u32 STATE_BITS = 22,
          COMPARTMENT_BITS = 18,
          INSTANT_BITS = 3,
          CA_ROW_BITS = 18;

const u64 X_SHIFT_BIT =63,
          Y_SHIFT_BIT =62,
          Z_SHIFT_BIT =61,
          CA_ROW_SHIFT_BIT =43,
          INSTANT_SHIFT_BIT =40,
          COMPARTMENT_SHIFT_BIT = 22,
          STATE_SHIFT_BIT = 0;
const u64 X_MASK = 1ull << X_SHIFT_BIT,
          Y_MASK = 1ull << Y_SHIFT_BIT,
          Z_MASK = 1ull << Z_SHIFT_BIT,
          CA_ROW_MASK = u64((1ull << CA_ROW_BITS) - 1ull) << CA_ROW_SHIFT_BIT,
          INSTANT_MASK = u64((1ull << INSTANT_BITS) - 1ull) << INSTANT_SHIFT_BIT,
          COMPARTMENT_MASK = u64((1ull << COMPARTMENT_BITS) - 1ull) << COMPARTMENT_SHIFT_BIT,
          STATE_MASK = u64((1ull << STATE_BITS) - 1ull) << STATE_SHIFT_BIT;
__device__ __host__ __inline__
void pack_x(u64& data, int x)
{
    data |= ((u64)x << X_SHIFT_BIT);
}

__device__ __host__ __inline__
void pack_y(u64& data, int y)
{
    data |= ((u64)y << Y_SHIFT_BIT);
}

__device__ __host__ __inline__
void pack_z(u64& data, int z)
{
    data |= ((u64)z << Z_SHIFT_BIT);
}

__device__ __host__ __inline__
void pack_ca_row_index(u64& data, int ca_row_index)
{
    data |= ((u64)ca_row_index << CA_ROW_SHIFT_BIT);
}

__device__ __host__ __inline__
void pack_instant(u64& data, int instant)
{
    data |= ((u64)instant << INSTANT_SHIFT_BIT);
}

__device__ __host__ __inline__
void pack_compartment_index(u64& data, int compartment_index)
{
    data |= ((u64)compartment_index << COMPARTMENT_SHIFT_BIT);
}

__device__ __host__ __inline__
void pack_state_index(u64& data, int state_index)
{
    data |= ((u64)state_index << STATE_SHIFT_BIT);
}
__inline__ __device__ __host__
void print_binary(char * b, u64 data)
{
    for (int i = 63; i >= 0; i--)
        b[63-i] = ((data >> i) & 1) == 1 ? '1' : '0';
    b[64] = '\0';
}

__device__ __host__ __inline__
int get_x(u64 data)
{
    return (data & X_MASK) >> X_SHIFT_BIT;
}

__device__ __host__ __inline__
int get_y(u64 data)
{
    return (data & Y_MASK) >> Y_SHIFT_BIT;
}

__device__ __host__ __inline__
int get_z(u64 data)
{
    return (data & Z_MASK) >> Z_SHIFT_BIT;
}

__device__ __host__ __inline__
int get_ca_row_index(u64 data)
{
    return (data & CA_ROW_MASK) >> CA_ROW_SHIFT_BIT;
}

__device__ __host__ __inline__
int get_instant(u64 data)
{
    return (data & INSTANT_MASK) >> INSTANT_SHIFT_BIT;
}

__device__ __host__ __inline__
int get_compartment_index(u64 data)
{
    return (data & COMPARTMENT_MASK) >> COMPARTMENT_SHIFT_BIT;
}

__device__ __host__ __inline__
int get_state_index(u64 data)
{
    return (data & STATE_MASK) >> STATE_SHIFT_BIT;
}


#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
 
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

#endif //USE_CUDA

#endif //  __CUDA_GLOBAL_H__
