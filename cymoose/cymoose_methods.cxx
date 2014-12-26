/*
 * =====================================================================================
 *
 *       Filename:  main_cython.hpp
 *
 *    Description:  Header file
 *
 *        Version:  1.0
 *        Created:  02/27/2014 03:23:20 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawar@ee.iitb.ac.in
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef WIN32
#include <sys/time.h>
#else
#include <time.h>
#endif
#include <math.h>
#include <queue>
#ifdef WIN32
#include "../external/xgetopt/XGetopt.h"
#else
#include <unistd.h> // for getopt
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef MACOSX
#include <sys/sysctl.h>
#endif // MACOSX

#include <iostream>
using namespace std;

unsigned int getNumCores()
{
	unsigned int numCPU = 0;
#ifdef WIN_32
	SYSTEM_INFO sysinfo;
	GetSystemInfo( &sysinfo );

	numCPU = sysinfo.dwNumberOfProcessors;
#endif

#ifdef LINUX
	numCPU = sysconf( _SC_NPROCESSORS_ONLN );
#endif

#ifdef MACOSX
	int mib[4];
	size_t len = sizeof(numCPU); 

	/* set the mib for hw.ncpu */
	mib[0] = CTL_HW;
	mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;

	/* get the number of CPUs from the system */
	sysctl(mib, 2, &numCPU, &len, NULL, 0);

	if( numCPU < 1 ) 
	{
		mib[1] = HW_NCPU;
		sysctl( mib, 2, &numCPU, &len, NULL, 0 );
	}
#endif
	if ( numCPU < 1 )
	{
		cout << "No CPU information available. Assuming single core." << endl;
		numCPU = 1;
	}
	return numCPU;
}

