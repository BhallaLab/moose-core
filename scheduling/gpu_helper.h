/*
 * =====================================================================================
 *
 *       Filename:  gpu_helper.h
 *
 *    Description:  Cuda related functions.
 *
 *        Version:  1.0
 *        Created:  Saturday 24 June 2017 06:59:50  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Micky Droch (), mickydroch@gmail.com
 *   Organization:  IIT Bombay
 *
 * =====================================================================================
 */


#ifndef  gpu_helper_INC
#define  gpu_helper_INC


inline void callMe( )
{
    printf( "Calling me\n" );
}

inline void cuda_ksolve( double* dy, double* y, const double currentTime, const double time, size_t n )
{

}

#endif   /* ----- #ifndef gpu_helper_INC  ----- */
