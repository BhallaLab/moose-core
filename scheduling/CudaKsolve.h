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

#include "CudaOdeSystem.h"
#include "../ksolve/VoxelPoolsBase.h"

inline void callMe( );

inline void cuda_ksolve( double* dy, double* y, const double currentTime, const double time, size_t n );

/**
 * @brief Convert a voxel-pool to cuda ODE system.
 *
 * @param pool
 * @param ode
 */
void voxelPoolToCudaOdeSystem( VoxelPoolsBase* pool, CudaOdeSystem* pOde );

#endif /* end of include guard: CUDAKSOLVE_H */

