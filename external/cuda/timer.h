/*
 * =====================================================================================
 *
 *       Filename:  timer.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  Sunday 25 June 2017 10:58:10  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Micky Droch (), mickydroch@gmail.com
 *   Organization:  IIT Bombay
 *
 * =====================================================================================
 */

#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif /* GPU_TIMER_H__ */
