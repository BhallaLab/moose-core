/***
 *       Filename:  CudaKsolve.h
 *
 *    Description:  CUDA version of Ksolve.
 *
 *        Version:  0.0.1
 *        Created:  2017-06-25
 *       Revision:  none
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL2
 */

#ifndef CUDAKSOLVE_H
#define CUDAKSOLVE_H

inline void callMe( );

inline void cuda_ksolve( double* dy, double* y, const double currentTime, const double time, size_t n );

#endif /* end of include guard: CUDAKSOLVE_H */

