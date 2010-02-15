/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Shell.h"
#include <sys/time.h>
#include <math.h>
#include <queue>
#include <unistd.h> // for getopt
#include "../scheduling/Tick.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"

const FuncId ENDFUNC( -1 );

extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size, unsigned int numThreads,
	unsigned int method );
extern void testShell();
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
	Id clockId = Clock::initCinfo()->create( "clock", 1 );
	Element* clocke = clockId();
	// Should put this initialization stuff within the Clock creation
	// step. This means I need to add an optional init func into the Cinfo
	// constructor, or to add the init func as a virtual func in Data.
	FieldElement< Tick, Clock, &Clock::getTick >* ticke =
		new FieldElement< Tick, Clock, &Clock::getTick >
		( 
			Tick::initCinfo(), clocke,
			&Clock::getNumTicks, &Clock::setNumTicks 
		);
	Id tickId = Id::create( ticke );

	assert ( shellid == Id() );
	assert( clockId == Id( 1, 0 ) );
	assert( tickId == Id( 2, 0 ) );
	SetGet::setShell();
	Shell* s = reinterpret_cast< Shell* >( shellid.eref().data() );
	s->setHardware( isSingleThreaded, numCores, numNodes );
	s->loadBalance();
	return shellid;
}

int main( int argc, char** argv )
{
	Id shellid = init( argc, argv );
#ifdef DO_UNIT_TESTS
	cout << "testing: ";
	testAsync();
	testShell();
	testScheduling();
#endif
	cout << endl;

	// Note that the main loop remains the parser loop, though it may
	// spawn a lot of other stuff.
	Element* shelle = shellid();
	Shell* s = reinterpret_cast< Shell* >( shelle->data( 0 ) );
	ProcInfo p;
	while( !s->getQuit() ) {
		Qinfo::clearQ( Shell::procInfo() );
		// The shell is careful not to execute any structural commands
		// during clearQ. It instead puts them onto an internal queue for
		// clearing during process.
		shelle->process( &p );
	}

	delete Id()();
	delete Id( 1, 0 )();
	delete Id( 2, 0 )();
	return 0;
}

