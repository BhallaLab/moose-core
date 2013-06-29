#include "PlatformFunctions.hpp"

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
  #include <Windows.h>
//#else
//  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#else
    #include <sys/time.h>
#endif

struct BenchTimes bench;
struct BenchConfig benchConf;

#if defined(__APPLE__)
    void random_r( struct random_data *buf, int32_t *del ){ *del = random(); }
    void initstate_r(unsigned int seed, char *statebuf, size_t statelen, struct random_data *buf) {
    	srandom(seed);
    }
#endif

uint64 gettimeInMilli()
{

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
	LARGE_INTEGER start;
	LARGE_INTEGER proc_freq;
	QueryPerformanceFrequency(&proc_freq);
	QueryPerformanceCounter(&start);
	return (uint64)(start.QuadPart/(proc_freq.QuadPart/1000L));
#else
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (tv.tv_sec*1000 + tv.tv_usec/1000);
#endif
    
  // ULONGLONG WINAPI
  //return (ulong)GetTickCount64();
  // http://msdn.microsoft.com/en-us/library/ms644900(VS.85).aspx#high_resolution

  //FILETIME ft;
  //unsigned __int64 tmpres = 0; 
  //GetSystemTimeAsFileTime(&ft);
 
  //tmpres |= ft.dwHighDateTime;
  //tmpres <<= 32;
  //tmpres |= ft.dwLowDateTime;
  
  //tmpres /= 10; /*converting file time to unix epoch*/ 
  //tmpres -= DELTA_EPOCH_IN_MICROSECS; /*convert into microseconds*/
  //return (long)(tmpres/1000UL);
}


