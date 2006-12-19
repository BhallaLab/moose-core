/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef DO_UNIT_TESTS
#include "TestField.h"
#include "TestMsg.h"
	extern void testBasecode();
	extern void testScheduling();
	extern void testTable();
#endif

int main(int argc, char** argv)
{
	int mynode = 0;
	int totalnodes = 1;
	Element* sli = 0;
#ifdef USE_MPI
	MPI::Init( argc, argv );
	totalnodes = MPI::COMM_WORLD.Get_size();
	mynode = MPI::COMM_WORLD.Get_rank();
#endif

	Cinfo::initialize();

	Element* shell = Cinfo::find("Shell")->
		create( "sli_shell", Element::root() );
	Ftype1< int >::set( shell, "totalnodes", totalnodes );
	Ftype1< int >::set( shell, "mynode", mynode );

	if ( mynode == 0 ) {
		sli = Cinfo::find("GenesisParser")->
			create( "sli", shell );
	
		sli->field( "shell" ).set( "/sli_shell" );
		shell->field( "parser" ).set( "/sli_shell/sli" );
	}

	Element* sched = Cinfo::find("Sched")->
		create( "sched", Element::root() );
	Element* cj = Cinfo::find("ClockJob")->create( "cj", sched );

	// Set up a link from the shell object so that all newly created
	// objects get scheduled or put into suitable solvers.
	Field createMsgSrc( shell, "schedNewObjectOut" );
	Field createMsgDest( cj, "schedNewObjectIn" );
	createMsgSrc.add( createMsgDest );

#ifdef USE_MPI
	// Create the postmasters and tie them to the shell
	Element* postmasters = Cinfo::find("Neutral")->
		create( "postmasters", Element::root() );
	Field remoteCommand( shell, "remoteCommand" );
	Element* ct;
	// Field process;

	for ( int i = 0; i < totalnodes; i++ ) {
		if ( i != mynode ) {
			char name[10];
			sprintf( name, "node%d", i );
			Element* p = Cinfo::find("PostMaster")->
				create( name, postmasters );
			Ftype1< int >::set( p, "remoteNode", i );

			Field temp( p, "remoteCommand" );
			remoteCommand.add( temp );
			if ( mynode != -1 ) {
				Field procTarget( p, "parProcess" );
				// process.add( procTarget );
			}
		}
	}

	if ( mynode != -1 ) {
		ct = Cinfo::find("ParTick")->create( "ct0", cj );
		// process = Field( ct, "parProcess" );
		Ftype1< int >::set( ct, "handleAsync", 1 );
	}

#endif
	
#ifdef DO_UNIT_TESTS
	if ( mynode == -1 ) {
		testBasecode();
		testScheduling();
		testTable();
		cout << "passed unit tests on node " << mynode << endl;
	}
#endif

	shell->field( "isInteractive" ).set( "1" );

	// We create the parser only on the master node.
	if ( mynode == 0 ) {
		shell->field( "isInteractive" ).set( "1" );

		if ( argc > 1 ) {
			string line = "";
			int len = strlen( argv[ 1 ] );
			if ( len > 3 && strcmp( argv[ 1 ] + len - 2, ".g" ) == 0 )
				line = "include";
			if ( len > 4 && strcmp( argv[ 1 ] + len - 3, ".mu" ) == 0 )
				line = "include";
			for (int i = 1; i < argc; i++)
				line = line + " " + argv[ i ];
	
			sli->field( "parse" ).set( line );
		}
		Field f = sli->field( "process" ) ;
		f.set( "" );
	}

#ifdef USE_MPI
	if ( mynode > 0 ) {
		while( 1 )
			Ftype0::set( ct, "pollAsyncIn" );
	}
	cout << "reached finalize on node " << mynode << endl;
	MPI::Finalize();
#endif
}
