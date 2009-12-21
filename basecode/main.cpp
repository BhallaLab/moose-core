#include "header.h"
#include "Shell.h"
#include <sys/time.h>
#include <math.h>
#include <queue>
#include <unistd.h> // for getopt

const FuncId ENDFUNC( -1 );

extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size, unsigned int numThreads,
	unsigned int method );
extern void testScheduling();

Id init( int argc, char** argv )
{
	int numCores = 1;
	int numNodes = 1;
	bool isSingleThreaded = 0;
	int opt;
	while ( ( opt = getopt( argc, argv, "shn:c:" ) ) != -1 ) {
		switch ( opt ) {
			case 's': // Single threaded mode
				isSingleThreaded = 1;
				break;
			case 'c': // Multiple cores per node
				// Each node handles 
				numCores = atoi( optarg );
				break;
			case 'n': // Multiple nodes
				numNodes = atoi( optarg );
				break;
			case 'h': // help
			default:
				cout << "Usage: moose -singleThreaded -help -c numCores -n numNodes\n";
				exit( 1 );
		}
	}
	Msg::initNull();
	Id shellid = Shell::initCinfo()->create( "root", 1 );
	assert ( shellid == Id() );
	SetGet::setShell();
	Shell* s = reinterpret_cast< Shell* >( shellid.eref().data() );
	s->setHardware( isSingleThreaded, numCores, numNodes );
	return shellid;
}

int main( int argc, char** argv )
{
	Id shellid = init( argc, argv );
#ifdef DO_UNIT_TESTS
	cout << "testing: ";
	testAsync();
	testScheduling();
#endif
	cout << endl;

	// Note that the main loop remains the parser loop, though it may
	// spawn a lot of other stuff.
	Element* shelle = shellid();
	Shell* s = reinterpret_cast< Shell* >( shelle->data( 0 ) );
	ProcInfo p;
	while( !s->getQuit() ) {
		Qinfo::clearQ( 0 );
		// The shell is careful not to execute any structural commands
		// during clearQ. It instead puts them onto an internal queue for
		// clearing during process.
		shelle->process( &p );
	}

	delete Id()();
	return 0;
}

