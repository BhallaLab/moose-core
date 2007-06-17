/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <iostream>
#include "header.h"
#include "moose.h"
#include "../element/Neutral.h"
#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef DO_UNIT_TESTS
	extern void testBasecode();
	extern void testNeutral();
	extern void testShell();
	extern void testInterpol();
	extern void testTable();
	extern void testSched();
	extern void testSchedProcess();
	extern void testWildcard();
	extern void testBiophysics();
	extern void testKinetics();
#ifdef USE_MPI
	extern void testPostMaster();
#endif
#endif

#ifdef USE_GENESIS_PARSER
	extern void makeGenesisParser( const string& s );
#endif

int main(int argc, char** argv)
{
	unsigned int mynode = 0;
#ifdef USE_MPI
	MPI::Init( argc, argv );
	unsigned int totalnodes = MPI::COMM_WORLD.Get_size();
	mynode = MPI::COMM_WORLD.Get_rank();

	Element* postmasters =
			Neutral::create( "Neutral", "postmasters", Element::root());
	vector< Element* > post;
	post.reserve( totalnodes );
	for ( unsigned int i = 0; i < totalnodes; i++ ) {
		char name[10];
		if ( i != mynode ) {
			sprintf( name, "node%d", i );
			Element* p = Neutral::create(
					"PostMaster", name, postmasters );
			assert( p != 0 );
			set< unsigned int >( p, "remoteNode", i );
			post.push_back( p );
		}
	}
	// Perhaps we will soon want to also connect up the clock ticks.
	// How do we handle different timesteps?
#endif
	
	Element* sched =
			Neutral::create( "Neutral", "sched", Element::root() );
	Element* cj =
			Neutral::create( "ClockJob", "cj", sched );

#ifdef USE_MPI
	Element* shell =
			Neutral::create( "Shell", "shell", Element::root() );
	Element* t0 =
			Neutral::create( "ParTick", "t0", cj );
#else
	Neutral::create( "Tick", "t0", cj );
	Neutral::create( "Shell", "shell", Element::root() );
#endif

#ifdef DO_UNIT_TESTS
	// if ( mynode == 0 )
	if ( 1 )
	{
		testBasecode();
		testNeutral();
		testShell();
		testInterpol();
		testTable();
		testSched();
		testSchedProcess();
		testWildcard();
		testBiophysics();
		testKinetics();
	}
#endif


#ifdef USE_MPI
	///////////////////////////////////////////////////////////////////
	//	Here we connect up the postmasters to the shell and the ParTick.
	///////////////////////////////////////////////////////////////////
	const Finfo* serialFinfo = shell->findFinfo( "serial" );
	const Finfo* masterFinfo = shell->findFinfo( "master" );
	const Finfo* slaveFinfo = shell->findFinfo( "slave" );
	assert( serialFinfo != 0 );

	const Finfo* tickFinfo = t0->findFinfo( "parTick" );
	assert( tickFinfo != 0 );
	bool glug = 0; // Breakpoint for parallel debugging
	while ( glug );
	for ( vector< Element* >::iterator j = post.begin();
		j != post.end(); j++ ) {
		bool ret = serialFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
		// bool ret = (*j)->findFinfo( "data" )->add( *j, shell, serialFinfo );
		assert( ret );
		ret = tickFinfo->add( t0, *j, (*j)->findFinfo( "parTick" ) );
		assert( ret );
		if ( mynode == 0 ) {
			ret = masterFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
		} else {
			ret = slaveFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
		}
		/*
		cout << "On " << mynode << ", post: " << (*j)->name() << endl;
		(*j)->dumpMsgInfo();
		assert( ret );
		*/
	}

	cout << "On " << mynode << ", shell: " << shell->name() << endl;
	shell->dumpMsgInfo();
	set( cj, "resched" );
	set( cj, "reinit" );
#ifdef DO_UNIT_TESTS
	MPI::COMM_WORLD.Barrier();
	if ( mynode == 0 )
		cout << "\nInitialized " << totalnodes << " nodes\n";
	MPI::COMM_WORLD.Barrier();
	testPostMaster();
#endif
#endif


#ifdef USE_GENESIS_PARSER
	if ( mynode == 0 ) {
		string line = "";
		if ( argc > 1 ) {
			int len = strlen( argv[1] );
			if ( len > 3 && strcmp( argv[1] + len - 2, ".g" ) == 0 )
				line = "include";
			else if ( len > 4 && 
							strcmp( argv[1] + len - 3, ".mu" ) == 0 )
				line = "include";
			for ( int i = 1; i < argc; i++ )
				line = line + " " + argv[ i ];
		}
		makeGenesisParser( line );
	}
#endif
#ifdef USE_MPI
	MPI::Finalize();
#endif
	cout << "done" << endl;
}
