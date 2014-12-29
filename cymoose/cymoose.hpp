#ifndef  MAIN_CYTHON_INC
#define  MAIN_CYTHON_INC
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
#include "../scheduling/Clock.h"
#include "../msg/DiagonalMsg.h"
//#include "../msg/SparseMsg.h"
#include "../mpi/PostMaster.h"
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "../shell/Shell.h"
#ifdef MACOSX
#include <sys/sysctl.h>
#endif // MACOSX

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


Id init( int argc, char** argv, Shell* pShell)
{
	unsigned int numCores = getNumCores();
	int numNodes = 1;
	int myNode = 0;
	Cinfo::rebuildOpIndex();

#ifdef USE_MPI
	/*
	// OpenMPI does not use argc or argv.
	// unsigned int temp_argc = 1;
	int provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	*/
	MPI_Init( &argc, &argv );

	MPI_Comm_size( MPI_COMM_WORLD, &numNodes );
	MPI_Comm_rank( MPI_COMM_WORLD, &myNode );

	if ( myNode == 0 ) 
        {
            cout << "++ On node " << myNode << ", numNodes = " 
                << numNodes << ", numCores = " << numCores << endl;
        }

#endif

	Id shellId;
	Element* shelle = new GlobalDataElement( shellId, Shell::initCinfo(), "root", 1 );

	Id clockId = Id::nextId();
	assert( clockId.value() == 1 );
	Id classMasterId = Id::nextId();
	Id postMasterId = Id::nextId();

	pShell = reinterpret_cast< Shell* >( shellId.eref().data() );
	pShell->setShellElement( shelle );
	pShell->setHardware( numCores, numNodes, myNode );
	pShell->loadBalance();

	/// Sets up the Elements that represent each class of Msg.
	unsigned int numMsg = Msg::initMsgManagers();

	new GlobalDataElement( clockId, Clock::initCinfo(), "clock", 1 );
	new GlobalDataElement( classMasterId, Neutral::initCinfo(), "classes", 1);
	new GlobalDataElement( postMasterId, PostMaster::initCinfo(), "postmaster", 1 );

	assert ( shellId == Id() );
	assert( clockId == Id( 1 ) );
	assert( classMasterId == Id( 2 ) );
	assert( postMasterId == Id( 3 ) );

	Shell::adopt( shellId, clockId, numMsg++ );
	Shell::adopt( shellId, classMasterId, numMsg++ );
	Shell::adopt( shellId, postMasterId, numMsg++ );

	assert( numMsg == 10 ); // Must be the same on all nodes.

	Cinfo::makeCinfoElements( classMasterId );
	return shellId;
}


/**
 * @brief Initialize moose and return handle to Shell.
 *
 * @param argc
 * @param argv
 * @param s
 *
 * @return Pointer to shell.
 */
Shell* initMoose( int argc, char** argv, Shell* s)
{
        Id shellId = init( argc, argv,  s);
        return s;
}

#endif   /* ----- #ifndef MAIN_CYTHON_INC  ----- */
