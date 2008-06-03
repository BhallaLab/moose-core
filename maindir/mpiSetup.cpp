/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <utility/utility.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../basecode/moose.h"
#include "../element/Neutral.h"

extern void testPostMaster();

using namespace std;

static Element* pj = 0;
static const Finfo* stepFinfo;

void initMPI( int argc, char** argv )
{
#ifdef USE_MPI
	MPI::Init( argc, argv );
	unsigned int totalnodes = MPI::COMM_WORLD.Get_size();
	unsigned int mynode = MPI::COMM_WORLD.Get_rank();
	bool ret;

	Element* postmasters =
			Neutral::create( "Neutral", "postmasters", Id(), Id::scratchId());
	vector< Element* > post;
	post.reserve( totalnodes );
	for ( unsigned int i = 0; i < totalnodes; i++ ) {
		char name[10];
		if ( i == mynode ) {
			post[i] = 0;
		} else {
			sprintf( name, "node%d", i );
			Element* p = Neutral::create(
					"PostMaster", name, postmasters->id(), Id::scratchId());
			assert( p != 0 );
			set< unsigned int >( p, "remoteNode", i );
			post[i] = p;
		}
	}
	Id::setNodes( mynode, totalnodes, post );

	// This one handles parser and postmaster scheduling.
	Id sched( "/sched" );
	Id cj( "/sched/cj" );
	Id t0id( "/sched/cj/t0" );
	if ( t0id.good() ) {
		set( t0id(), "destroy" );
	}
	pj =
		Neutral::create( "ClockJob", "pj", sched, Id::scratchId() );
	Element* t0 =
			Neutral::create( "ParTick", "t0", cj, Id::scratchId() );
	Element* pt0 =
			Neutral::create( "ParTick", "t0", pj->id(), Id::scratchId() );

	///////////////////////////////////////////////////////////////////
	//	Here we connect up the postmasters to the shell and the ParTick.
	///////////////////////////////////////////////////////////////////
	Id shellId( "/shell" );
	Element* shell = shellId();
	const Finfo* serialFinfo = shell->findFinfo( "serial" );
	assert( serialFinfo != 0 );
	const Finfo* masterFinfo = shell->findFinfo( "master" );
	assert( masterFinfo != 0 );
	const Finfo* slaveFinfo = shell->findFinfo( "slave" );
	assert( slaveFinfo != 0 );
	const Finfo* pollFinfo = shell->findFinfo( "pollSrc" );
	assert( pollFinfo != 0 );
	const Finfo* tickFinfo = t0->findFinfo( "parTick" );
	assert( tickFinfo != 0 );

	stepFinfo = pj->findFinfo( "step" );
	assert( stepFinfo != 0 );

	// Breakpoint for parallel debugging
	/*
	bool glug = (argc == 2 && 
		strcmp( argv[1], "-debug" ) == 0 );
	while ( glug );
	*/
	Eref shellE = shellId.eref();
	vector< Element* >::iterator j;
	for ( j = post.begin(); j != post.end(); j++ ) {
		if ( *j == 0 )
			continue;
		ret = shellE.add( "serial", *j, "data");
		assert( ret );
		ret = 0;

		assert( ret );
		ret = Eref( t0 ).add( "parTick", *j, "parTick" );
		assert( ret );
		ret = Eref( pt0 ).add( "parTick", *j, "parTick" );
		assert( ret );
	
		/*
		bool ret = serialFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
		bool ret = (*j)->findFinfo( "data" )->add( *j, shell, serialFinfo );

		ret = tickFinfo->add( t0, *j, (*j)->findFinfo( "parTick" ) );
		assert( ret );
		ret = tickFinfo->add( pt0, *j, (*j)->findFinfo( "parTick" ) );
		assert( ret );
		*/

		if ( mynode == 0 ) {
			ret = shellE.add( "master", *j, "data" );
			// ret = masterFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
			assert( ret );
		} else {
			ret = shellE.add( "slave", *j, "data" );
			// ret = slaveFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
			assert( ret );
		}
	}
	ret = shellE.add( "pollSrc", pj, "step" );
	// ret = pollFinfo->add( shell, pj, pj->findFinfo( "step" ) );
	assert( ret );

	// cout << "On " << mynode << ", shell: " << shell->name() << endl;
	// shell->dumpMsgInfo();
	set( cj.eref(), "resched" );
	set( pj, "resched" );
	set( cj.eref(), "reinit" );
	set( pj, "reinit" );
#ifdef DO_UNIT_TESTS
	MPI::COMM_WORLD.Barrier();
	if ( mynode == 0 )
		cout << "\nInitialized " << totalnodes << " nodes\n";
	MPI::COMM_WORLD.Barrier();
	testPostMaster();
#endif // DO_UNIT_TESTS
#endif // USE_MPI
}

void terminateMPI( unsigned int mynode )
{
#ifdef USE_MPI
	Eref shell = Id::shellId().eref();
	if ( mynode != 0 ) {
		bool ret = set( shell, "poll" );
		assert( ret );
	}
	MPI::Finalize();
#endif // USE_MPI
}

void pollPostmaster()
{
	if ( pj != 0 ) {
		bool ret = set< int >( pj, stepFinfo, 1 );
		assert( ret );
	}
}
