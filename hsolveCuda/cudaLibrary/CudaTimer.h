/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Multiscale Object Oriented Simulation Environment.
 **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/


#ifndef CUDATIMER_H_
#define CUDATIMER_H_

#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

class 	CudaTimer
{
private:
	#ifdef _WIN32
	double PCFreq = 0.0;
	__int64 timerStart = 0;
	#else
	struct timeval timerStart;
	#endif

public:
	inline void StartTimer()
	{
	#ifdef _WIN32
		LARGE_INTEGER li;

		if (!QueryPerformanceFrequency(&li))
		{
			printf("QueryPerformanceFrequency failed!\n");
		}

		PCFreq = (double)li.QuadPart/1000.0;
		QueryPerformanceCounter(&li);
		timerStart = li.QuadPart;
	#else
		gettimeofday(&timerStart, NULL);
	#endif
	}

	// time elapsed in ms
	inline double GetTimer()
	{
	#ifdef _WIN32
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return (double)(li.QuadPart-timerStart)/PCFreq;
	#else
		struct timeval timerStop, timerElapsed;
		gettimeofday(&timerStop, NULL);
		timersub(&timerStop, &timerStart, &timerElapsed);
		return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
	#endif
	}
};


#endif /* CUDATIMER_H_ */
// Saeed
