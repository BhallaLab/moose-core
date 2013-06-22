/*
 * CudaTimer.h
 *
 *  Created on: Feb 18, 2013
 *      Author: Saeed Shariati
 */

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
