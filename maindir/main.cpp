/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * \mainpage MOOSE Code documentation: Generated by Doxygen.
 * 
 * \section intro_sec Introduction
 * MOOSE is the Multiscale Object Oriented Simulation Environment.
 * This Doxygen-generated set of pages documents the source code of
 * MOOSE.
 *
 */

#include <iostream>
#include <basecode/header.h>
#include <basecode/moose.h>
#include <element/Neutral.h>
#include <basecode/IdManager.h>
#include <utility/utility.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef CRL_MPI
#include <mpi.h>
#endif

extern int mooseInit();

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
	extern void testAverage();
#ifdef USE_MPI
	extern void testPostMaster();
#endif
#endif

#ifdef USE_GENESIS_PARSER
	extern Element* makeGenesisParser( );
//	extern void nonblock( int state );
	extern bool nonBlockingGetLine( string& s );
#endif

void setupDefaultSchedule( 
	Element* t0, Element* t1, Element* cj)
{
	set< double >( t0, "dt", 1e-2 );
	set< double >( t1, "dt", 1e-2 );
	set< int >( t1, "stage", 1 );
	set( cj, "resched" );
	set( cj, "reinit" );
}

int main(int argc, char** argv)
{
	unsigned int mynode = 0;
        // TODO : check the repurcussions of MPI command line
        ArgParser::parseArguments(argc, argv);
        
        Property::initialize(ArgParser::getConfigFile(),Property::PROP_FORMAT);
        PathUtility simpathHandler(ArgParser::getSimPath());
        simpathHandler.addPath(Property::getProperty(Property::SIMPATH)); // merge the SIMPATH from command line and property file
        Property::setProperty(Property::SIMPATH, simpathHandler.getAllPaths()); // put the updated path list in Property
        cout << "SIMPATH = " << Property::getProperty(Property::SIMPATH) << endl;
        
        mooseInit();
        
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
		testAverage();
	}
#endif

#ifdef CRL_MPI
	int iMyRank;
	int iProvidedThreadSupport;
	int iRequiredThreadSupport = MPI_THREAD_SINGLE;

	MPI_Init_thread(&argc, &argv, iRequiredThreadSupport, &iProvidedThreadSupport);
	if(iProvidedThreadSupport != iRequiredThreadSupport)
	{
		cout<<endl<<"Error: Expected thread support not provided"<<endl<<flush;
		MPI_Finalize();
		exit(1);
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
#endif

#ifdef USE_MPI
	///////////////////////////////////////////////////////////////////
	//	Here we connect up the postmasters to the shell and the ParTick.
	///////////////////////////////////////////////////////////////////
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
	const Finfo* stepFinfo = pj->findFinfo( "step" );
	assert( stepFinfo != 0 );

	bool glug = (argc == 2 && 
		strcmp( argv[1], "-debug" ) == 0 );
	// Breakpoint for parallel debugging
	while ( glug );
	for ( vector< Element* >::iterator j = post.begin();
		j != post.end(); j++ ) {
		bool ret = serialFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
		// bool ret = (*j)->findFinfo( "data" )->add( *j, shell, serialFinfo );
		assert( ret );
		ret = tickFinfo->add( t0, *j, (*j)->findFinfo( "parTick" ) );
		assert( ret );
		ret = tickFinfo->add( pt0, *j, (*j)->findFinfo( "parTick" ) );
		assert( ret );

		if ( mynode == 0 ) {
			ret = masterFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
			assert( ret );
		} else {
			ret = slaveFinfo->add( shell, *j, (*j)->findFinfo( "data" ) );
			assert( ret );
		}
		/*
		cout << "On " << mynode << ", post: " << (*j)->name() << endl;
		(*j)->dumpMsgInfo();
		assert( ret );
		*/
	}
	ret = pollFinfo->add( shell, pj, pj->findFinfo( "step" ) );
	assert( ret );

	// cout << "On " << mynode << ", shell: " << shell->name() << endl;
	// shell->dumpMsgInfo();
	set( cj, "resched" );
	set( pj, "resched" );
	set( cj, "reinit" );
	set( pj, "reinit" );
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
                vector<string> scriptArgs = ArgParser::getScriptArgs();
                
                if ( scriptArgs.size() > 0 )
                {
                    line = "include";
                    for ( unsigned int i = 0; i < scriptArgs.size(); ++i )
                    {
                        line = line + " " + scriptArgs[i];
                    }
                    line.push_back('\n');
                }
                
		Element* sli = makeGenesisParser();
		assert( sli != 0 );
		// Need to do this before the first script is loaded, but
		// after the unit test for the parser.
                Id cj("/sched/cj");
                Id t0("/sched/cj/t0");
                Id t1("/sched/cj/t1");
                
		setupDefaultSchedule( t0(), t1(), cj() );

		const Finfo* parseFinfo = sli->findFinfo( "parse" );
		assert ( parseFinfo != 0 );

#ifdef CRL_MPI
	if(iMyRank == 0)
	{
#endif
		set< string >( sli, parseFinfo, line );
		set< string >( sli, parseFinfo, "\n" );

		/**
		 * Here is the key infinite loop for getting terminal input,
		 * parsing it, polling postmaster, managing GUI and other 
		 * good things.
		 */
		string s = "";
		unsigned int lineNum = 0;
		cout << "moose #" << lineNum << " > " << flush;
		while( 1 ) {
			if ( nonBlockingGetLine( s ) ) {
				set< string >( sli, parseFinfo, s );
				if ( s.find_first_not_of( " \t\n" ) != s.npos )
					lineNum++;
				s = "";
				cout << "moose #" << lineNum << " > " << flush;
			}
#ifdef USE_MPI
			// Here we poll the postmaster
			ret = set< int >( pj, stepFinfo, 1 );
#endif
			// gui stuff here maybe.
		}
#ifdef CRL_MPI
	}
	else
	{
		set< string >( sli, parseFinfo, "nonroot" );
	}
#endif


	}
#endif
#ifdef USE_MPI
	if ( mynode != 0 ) {
		ret = set( shell, "poll" );
		assert( ret );
	}
	MPI::Finalize();
#endif

#ifdef USE_MPI
	MPI_Finalize();
#endif

	cout << "done" << endl;
}
