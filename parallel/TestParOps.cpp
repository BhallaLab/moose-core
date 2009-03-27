/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifdef USE_MPI
#ifdef DO_UNIT_TESTS
#include "moose.h"
#include <math.h>
#include <mpi.h>
#include "maindir/MuMPI.h"
#include "PostMaster.h"
#include "ProxyElement.h"
#include "AsyncDestFinfo.h"
#include "../element/Neutral.h"
#include "../element/Wildcard.h"
#include "Ftype2.h"
#include "setget.h"
#include "../builtins/Interpol.h"
#include "../builtins/Table.h"
#include "../shell/Shell.h"

#ifdef _MSC_VER       // True for MS VC++ compilers
#include <Windows.h>  // The Windows Sleep function has a granularity of 1 ms.
#define usleep( x ) Sleep( ( x ) / 1000.0 )
#else
#include <unistd.h>   // Used for the usleep definition
#endif

extern void pollPostmaster(); // Defined in maindir/mpiSetup.cpp

/**
 * Creates objects on remote nodes.
 * Also tests that objects on /library get created on all nodes.
 * \todo: Needs to be extended by creating objects on remote nodes with
 * parents on different remote nodes.
 */
Id testParCreate( vector< Id >& testIds )
{
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
	if ( myNode == 0 )
		cout << flush << "\nTest ParCreate";
	MuMPI::INTRA_COMM().Barrier();
	cout << "b" << myNode << flush;
	MuMPI::INTRA_COMM().Barrier();
	Eref shellE = Id::shellId().eref();
	assert( shellE.e != 0 );
	if ( myNode == 0 ) {
		Slot remoteCreateSlot = 
			initShellCinfo()->getSlot( "parallel.createSrc" );
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			char name[20];
			sprintf( name, "tn%d", i );
			string sname = name;
			unsigned int tgt = ( i < myNode ) ? i : i - 1;
			Id newId = Id::makeIdOnNode( i );
			testIds.push_back( newId );
			// cout << "Create op: sendTo4( shellId, slot, " << tgt << ", " << "Neutral, " << sname << ", root, " << newId << endl;
			sendTo4< string, string, Nid, Nid >(
				shellE, remoteCreateSlot, tgt,
				"Neutral", sname, 
				Id(), newId
			);
		}
		Id libid = Id::localId( "/library" ); // a global
		ASSERT( libid.good(), "create libkids" );
		ASSERT( libid.isGlobal(), "create libkids" );
		SetConn c( shellE );
		Shell::staticCreate( &c, "Neutral", "foo", Id::UnknownNode, libid );
	}
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster(); // There is a barrier in the polling operation itself
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster();
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster();
	MuMPI::INTRA_COMM().Barrier();
	char name[20];
	sprintf( name, "/tn%d", myNode );
	string sname = name;
	Id kidid = Id::localId( "/library/foo" );
	ASSERT( kidid.good(), "create libkids" );
	ASSERT( kidid.isGlobal(), "create libkids" );
	bool ret = set( kidid.eref(), "destroy" );
	ASSERT( ret, "destroy libkids" );
	if ( myNode != 0 ) {
		Id tnId = Id::localId( sname );
		ASSERT( tnId.good(), "postmaster created obj on remote node" );
		return tnId;
	}
	return Id();
}

/**
 * Copies objects on remote nodes.
 * - Create original in /library              : All nodes
 * - Create target{i} on individual nodes.    : One at a time
 * - Copy /library/orig /library/dup          : Automagically on all nodes
 * - Copy /library/dup /target{i}             : One at a time
 * - Copy /target{i}/dup /target{i}/zung      : One at a time.
 *   Cannot test this last one yet because I can't get Ids from remote node
 *
 * Also tests that objects on /library get copied on all nodes.
 * \todo: Stil don't have capability to copy across nodes.
 */
void testParCopy()
{
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
	if ( myNode == 0 )
		cout << flush << "\nTest ParCopy";
	MuMPI::INTRA_COMM().Barrier();
	Eref shellE = Id::shellId().eref();
	assert( shellE.e != 0 );
	if ( myNode == 0 ) {
		vector< Id > targets;
		SetConn c( shellE );
		Shell::staticCreate( &c, "Neutral", "target0", 0, Id() );
		Id temp( "/target0" );
		ASSERT( temp.good(), "Testng copy" );
		targets.push_back( temp );
		Slot remoteCreateSlot = 
			initShellCinfo()->getSlot( "parallel.createSrc" );
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			char name[20];
			sprintf( name, "target%d", i );
			string sname = name;
			unsigned int tgt = ( i < myNode ) ? i : i - 1;
			Id newId = Id::makeIdOnNode( i );
			targets.push_back( newId );
			sendTo4< string, string, Nid, Nid >(
				shellE, remoteCreateSlot, tgt,
				"Neutral", sname, 
				Id(), newId
			);
		}
		Id libid = Id::localId( "/library" ); // a global
		ASSERT( libid.good(), "create libkids" );
		ASSERT( libid.isGlobal(), "create libkids" );
		Shell::staticCreate( &c, "Neutral", "orig", Id::UnknownNode, libid );
		Id origId = Id::localId( "/library/orig" );
		ASSERT( origId != Id(), "Testing copy" );
		Shell::copy( &c, origId, libid, "dup" );
		assert( targets.size() == numNodes );
		for ( unsigned int i = 0; i < numNodes; i++ ) {
			Shell::copy( &c, origId, targets[i], "dup" );
		}
	}
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster(); // There is a barrier in the polling operation itself
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster();
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster();
	MuMPI::INTRA_COMM().Barrier();
	char name[20];
	sprintf( name, "/tn%d", myNode );
	string sname = name;
	Id kidid = Id::localId( "/library/orig" );
	ASSERT( kidid.good(), "copy libkids" );
	ASSERT( kidid.isGlobal(), "copy libkids" );
	bool ret = set( kidid.eref(), "destroy" );
	ASSERT( ret, "destroy libkids" );
	kidid = Id::localId( "/library/dup" );
	ASSERT( kidid.good(), "copy libkids" );
	// cout << " on " << myNode << ": kidid = " << kidid << ", node = " << kidid.node() << endl << flush;
	ASSERT( kidid.isGlobal(), "copy libkids" );
	ret = set( kidid.eref(), "destroy" );
	ASSERT( ret, "destroy libkids" );
	sprintf( name, "/target%d/dup", myNode );
	sname = name;
	kidid = Id::localId( sname );
	ASSERT( kidid.good(), "copy libkids to node" );
	ASSERT( !kidid.isGlobal(), "copy libkids to node" );
	sprintf( name, "/target%d", myNode );
	sname = name;
	kidid = Id::localId( sname );
	ret = set( kidid.eref(), "destroy" );
	ASSERT( ret, "destroy libkids" );
	// Should have a global delete for library objects too.
}


/////////////////////////////////////////////////////////////////
// Now test 'get' across nodes.
// Normally the 'get' call is invoked by the parser, which expects a
// value to come back. Note that the return must be asynchronous:
// the parser cannot block since we need to execute MPI polling
// operations on either side.
/////////////////////////////////////////////////////////////////
void testParGet( Id tnId, vector< Id >& testIds )
{
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
	Slot parGetSlot = initShellCinfo()->getSlot( "parallel.getSrc" );
	char name[20];
	string sname;
	if ( myNode == 0 ) {
		cout << "\ntesting parallel get" << flush;
	} else {
		sprintf( name, "foo%d", myNode * 2 );
		sname = name;
		set< string >( tnId.eref(), "name", sname );
	}
	MuMPI::INTRA_COMM().Barrier();
	Eref e = Id::shellId().eref();
	Shell* sh = static_cast< Shell* >( e.data() );
	vector< unsigned int > rids( numNodes, 0 );
	vector< string > ret( numNodes );
	unsigned int origSize = sh->freeRidStack_.size();
	ASSERT( origSize > 0 , "Stack initialized properly" );
	if ( myNode == 0 ) {
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			rids[i] = 
				openOffNodeValueRequest< string >( sh, &ret[i], 1 );
			ASSERT( sh->freeRidStack_.size() == origSize - i, "stack in use" );
			sendTo3< Id, string, unsigned int >(
				Id::shellId().eref(), parGetSlot, i - 1,
				testIds[i - 1], "name", rids[i]
			);
		}
	}
	// Here we explicitly do what the closeOffNodeValueRequest handles.
	MuMPI::INTRA_COMM().Barrier();
	// Cycle a few times to make sure all data gets back to node 0
	for ( unsigned int i = 0; i < 5; i++ ) {
		pollPostmaster();
		MuMPI::INTRA_COMM().Barrier();
	}
	
	// Now go through to check all values have come back.
	if ( myNode == 0 ) {
		ASSERT( sh->freeRidStack_.size() == 1 + origSize - numNodes, 
			"Stack still waiting" );
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			sprintf( name, "foo%d", i * 2 );
			sname = name;
			ASSERT( sh->offNodeData_[ rids[i] ].numPending == 0,
				"Pending requests cleared" );
			ASSERT( sh->offNodeData_[ rids[i] ].data == 
				static_cast< void* >( &ret[i] ), "Pointing to strings" );
			ASSERT( ret[ i ] == sname, "All values returned correctly" );
			// Clean up the debris
			sh->offNodeData_[ rids[i] ].data = 0;
			sh->freeRidStack_.push_back( rids[i] );
		}
	}
}

void testParSet( vector< Id >& testIds )
{
	//////////////////////////////////////////////////////////////////
	// Now test 'set' across nodes. This fits nicely as a unit test.
	//////////////////////////////////////////////////////////////////
	char name[20];
	string sname;
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
	Eref shellE = Id::shellId().eref();
	assert( shellE.e != 0 );
	MuMPI::INTRA_COMM().Barrier();
	if ( myNode == 0 ) {
		cout << "\ntesting parallel set" << flush;
		Slot parSetSlot = 
			initShellCinfo()->getSlot( "parallel.setSrc" );
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			sprintf( name, "bar%d", i * 10 );
			sname = name;
			unsigned int tgt = ( i < myNode ) ? i : i - 1;
			// objId, field, value
			sendTo3< Id, string, string >(
				shellE, parSetSlot, tgt,
				testIds[i - 1], "name", sname
			);
		}
		Id libid = Id::localId( "/library" ); // a global
		ASSERT( libid.good(), "create libkids" );
		ASSERT( libid.isGlobal(), "create libkids" );
		SetConn c( shellE );
		Shell::staticCreate( &c, "Neutral", "foo", Id::UnknownNode, libid );
	}
	
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster();
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster();

	if ( myNode != 0 ) {
		sprintf( name, "/bar%d", myNode * 10 );
		sname = name;
		Id checkId = Id::localId( sname );
		// cout << "On " << myNode << ", checking id for " << sname << ": " << checkId << endl;
		ASSERT( checkId.good(), "parallel set" );
		cout << flush;
	}

	////////////////////////////////////////////////////////////////
	// Here we check for assignment on globals.
	////////////////////////////////////////////////////////////////
	Id kidid = Id::localId( "/library/foo" );
	ASSERT( kidid.good(), "setting libkids" );
	ASSERT( kidid.isGlobal(), "setting libkids" );
	MuMPI::INTRA_COMM().Barrier();
	if ( myNode == 0 ) {
		cout << "\ntesting global set" << flush;
		SetConn c( shellE );
		Shell::setField( &c, kidid, "name", "bar" );
	}
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster();
	MuMPI::INTRA_COMM().Barrier();
	pollPostmaster();
	
	Id newKidid = Id::localId( "/library/bar" );
	ASSERT( newKidid == kidid, "setting libkids" );
	ASSERT( newKidid.good(), "setting libkids" );
	ASSERT( newKidid.isGlobal(), "setting libkids" );
	
	bool ret = set( kidid.eref(), "destroy" );
	ASSERT( ret, "destroy libkids" );
	cout << flush;
	MuMPI::INTRA_COMM().Barrier();
}

void testParDelete( vector< Id >& testIds )
{
	//////////////////////////////////////////////////////////////////
	// Now test 'set' across nodes. This fits nicely as a unit test.
	//////////////////////////////////////////////////////////////////
	char name[20];
	string sname;
	vector< Id > kids;
	Id victim;
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	MuMPI::INTRA_COMM().Barrier();
	// First check the list of children, using node-local commands.
	// If we later alter childList, must fix.
	if ( myNode != 0 ) {
		victim = Id();
		get< vector< Id > >( Eref::root(), "childList", kids );
		sprintf( name, "bar%d", myNode * 10 );
		sname = name;
		vector< Id >::iterator i;
		for ( i = kids.begin(); i != kids.end(); i++ )
			if ( (*i).eref()->name() == sname )
				victim = *i;
		ASSERT( victim != Id(), "testParDelete: found victim" );
	}

	if ( myNode == 0 ) {
		Slot parDeleteSlot = 
			initShellCinfo()->getSlot( "parallel.deleteSrc" );
		cout << "\ntesting parallel delete" << flush;
		vector< Id >::iterator i;
		for ( i = testIds.begin(); i != testIds.end(); i++ ) {
			unsigned int node = i->node();
			ASSERT( node != 0, "testParDelete" );
			sendTo1< Id >( Id::shellId().eref(), parDeleteSlot, node - 1,
				*i );
		}
	}
	for ( unsigned int i = 0; i < 5; i++ ) {
		pollPostmaster();
		MuMPI::INTRA_COMM().Barrier();
	}
	// Now check on the carnage. Must use local-node commands here.
	if ( myNode != 0 ) {
		vector< Id > remainingKids;
		get< vector< Id > >( Eref::root(), "childList", remainingKids );
		ASSERT( remainingKids.size() == kids.size() - 1, "testParDelete");
		remainingKids.push_back( Id() );
		for ( unsigned int j = 0; j < kids.size(); j++ )
			if ( kids[j] != remainingKids[j] )
				ASSERT( kids[j] == victim, "testParDelete" );
		cout << flush;
	}
}

/**
 * This routine tests sending many packets over in one go. Tests
 * how many polls are needed for everything to execute, and whether
 * the execution order is clean.
 */
void testParCommandSequence()
{
	const unsigned int numSeq = 10;
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
	Eref shellE = Id::shellId().eref();
	char name[20];
	string sname;
	if ( myNode == 0 ) {
		cout << "\ntesting parallel command sequence" << flush;
	}
	if ( myNode == 0 ) {
		Slot remoteCreateSlot = 
			initShellCinfo()->getSlot( "parallel.createSrc" );
		Slot parSetSlot = 
			initShellCinfo()->getSlot( "parallel.setSrc" );
		for ( unsigned int j = 0; j < numSeq; j++ ) {
			for ( unsigned int i = 1; i < numNodes; i++ ) {
				char name[20];
				sprintf( name, "zug%d.%d", i, j );
				string sname = name;
				unsigned int tgt = ( i < myNode ) ? i : i - 1;
				Id newId = Id::makeIdOnNode( i );
				// cout << "Create op: sendTo4( shellId, slot, " << tgt << ", " << "Neutral, " << sname << ", root, " << newId << endl;
				sendTo4< string, string, Nid, Nid >(
					Id::shellId().eref(), remoteCreateSlot, tgt,
					"Table", sname, 
					Id(), newId
				);
				sname = sname + ".extra";
				sendTo3< Id, string, string >(
					Id::shellId().eref(), parSetSlot, tgt,
					newId, "name", sname
				);
			}
		}
	}
	for ( unsigned int i = 0 ; i < 5; i++ ) {
		pollPostmaster();
		MuMPI::INTRA_COMM().Barrier();
	}
	if ( myNode != 0 ) {
		for ( unsigned int j = 0; j < numSeq; j++ ) {
			sprintf( name, "/zug%d.%d.extra", myNode, j );
			sname = name;
			Id checkId = Id::localId( sname );
			// cout << "On " << myNode << ", checking id for " << sname << ": " << checkId << endl;
			ASSERT( checkId.good(), "parallel command sequencing" );

			// Clean up.
			set( checkId.eref(), "destroy" );
		}
		cout << myNode << flush;
	}
}

/**
//////////////////////////////////////////////////////////////////
// Now test message creation across nodes. 
// 	Use parallel commands to create tables on each node
// 	Use parallel commands to configure these tables.
// 		Each table has no lookup, just does a sum of inputs.
// 		First table is set to always generate 1 at the output
// 	Use parallel commands to connect up these tables to each of
// 		the two previous nodes: a Fibonacci series.
// 	Step for a certain # of cycles to let data traverse all nodes.
// 	Check that the output is OK.
//////////////////////////////////////////////////////////////////
**/
void testParMsg()
{
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
	Eref shellE = Id::shellId().eref();
	vector< Id > tabIds;
	char name[20];
	string sname;
	if ( myNode == 0 ) {
		cout << "\ntesting parallel msgs" << flush;
		Slot parSetSlot = 
			initShellCinfo()->getSlot( "parallel.setSrc" );
		Slot remoteCreateSlot = 
			initShellCinfo()->getSlot( "parallel.createSrc" );
		Element* tab0 = Neutral::create( "Table", "tab0", Id(), Id::makeIdOnNode( 0 ) );
		tabIds.push_back( tab0->id() );
		set< int >( tab0, "stepmode", 0 );
		set< int >( tab0, "xdivs", 1 );
		set< double >( tab0, "xmin", 0 );
		set< double >( tab0, "xmax", 1 );
		lookupSet< double, unsigned int >( tab0, "table", 1, 0 );
		lookupSet< double, unsigned int >( tab0, "table", 1, 1 );
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			sprintf( name, "tab%d", i );
			sname = name;
			unsigned int tgt = ( i < myNode ) ? i : i - 1;
			Id newId = Id::makeIdOnNode( i );
			tabIds.push_back( newId );
			sendTo4< string, string, Nid, Nid >(
				Id::shellId().eref(), remoteCreateSlot, tgt,
				"Table", sname, 
				Id(), newId
			);
			sendTo3< Id, string, string >(
				Id::shellId().eref(), parSetSlot, tgt,
				newId, "stepmode", "0"
			); // LOOKUP
			// We must do message addition via the Shell to be able
			// to send stuff across nodes.
			/*
			set< Id, string, Id, string >( shellE, "add", 
				tabIds[i - 1], "outputSrc", tabIds[i], "sum" );
			if ( i >= 2 ) {
				set< Id, string, Id, string >( shellE, "add", 
					tabIds[i - 2], "outputSrc", tabIds[i], "sum" );
			}
			*/
		}
	}
	for ( unsigned int i = 0; i < 5; i++ ) {
		pollPostmaster();
		MuMPI::INTRA_COMM().Barrier();
	}
	sprintf( name, "/tab%d", myNode );
	sname = name;
	Id checkId = Id::localId( sname );
	ASSERT( checkId.good(), "parallel msgs" );
	int stepmode;
	get< int >( checkId.eref(), "stepmode", stepmode );
	ASSERT( stepmode == 0, "parallel msgs" );
	cout << flush;

	if ( myNode == 0 ) {
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			set< Id, string, Id, string >( shellE, "add", 
				tabIds[i - 1], "outputSrc", tabIds[i], "sum" );
			if ( i >= 2 ) {
				set< Id, string, Id, string >( shellE, "add", 
					tabIds[i - 2], "outputSrc", tabIds[i], "sum" );
			}
		}
	}
	for ( unsigned int i = 0; i < 5; i++ ) {
		pollPostmaster();
		MuMPI::INTRA_COMM().Barrier();
	}
	cout << flush;

	// Check that the messages were made. Node 0 and 1 are special
	unsigned int numOutputSrc = numNodes - myNode - 1;
	if ( numOutputSrc >= 2 ) numOutputSrc = 2;
	unsigned int numSum = ( myNode < 2 ) ? myNode : 2;

	// cout << "On node= " << myNode << ", numOutputSrc: " << checkId.eref()->numTargets( "outputSrc" ) << ", " << numOutputSrc << endl << flush;

	ASSERT( checkId.eref()->numTargets( "outputSrc" ) == numOutputSrc, 
		"par msg" );

	// cout << "numSum: " << checkId.eref()->numTargets( "sum" ) << ", " << numSum << endl << flush;
	ASSERT( checkId.eref()->numTargets( "sum" ) == numSum, "par msg" );
	MuMPI::INTRA_COMM().Barrier();

	// True for all but the last node
	if ( myNode < numNodes - 1 )
		ASSERT( checkId.eref()->isTarget( Id::postId( 0 ).eref().e ), "isTarget" );

	// Check that the right number of messages are set up.
	for ( unsigned int i = 0; i < numNodes; i++ ) {
		PostMaster* pm = static_cast< PostMaster* >( Id::postId( i ).eref().data() );
		unsigned int numOut = 0;
		unsigned int numIn = 0;
		if ( i == myNode + 1 || i == myNode + 2 )
			numOut = 1;

		if ( i == myNode - 1 || i == myNode - 2 )
			numIn = 1;
		ASSERT( pm->numAsyncOut_ == numOut, "parallel messaging: numAsyncOut" );
		ASSERT( pm->numAsyncIn_ == numIn, "parallel messaging: numAsyncIn" );
	}

	// Now try to send data through this beast.
	Id cjId( "/sched/cj" );
	assert( cjId.good() );
	set( cjId.eref(), "resched" );
	ASSERT( checkId.eref()->numTargets( "process" ) == 1, "sched par msg");
	// set< int >( cjId.eref(), "step", static_cast< int >( numNodes ) );
	for ( unsigned int i = 0; i < numNodes * 5; i++ ) {
		// set< int >( cjId.eref(), "step", numNodes * 2 + 3 );
		set< int >( cjId.eref(), "step", 1 );
		usleep( 10000 );
		MuMPI::INTRA_COMM().Barrier();
	}
	double f1 = 0.0;
	double f2 = 1.0;
	double f = 1.0;
	// Compute Fibonacci number for the current node
	for ( unsigned int i = 0; i < myNode; i++ ) {
		f = f1 + f2;
		f1 = f2;
		f2 = f;
	}
	double x;
	get< double >( checkId.eref(), "output", x );
	// cout << "on " << myNode << ", f = " << f << ", x = " << x << endl << flush;
	ASSERT( x == f, "par msg fibonacci" );

	set( checkId.eref(), "destroy" );

	///\todo: need to set up parallel message delete.
	for ( unsigned int i = 0; i < numNodes; i++ ) {
		PostMaster* pm = static_cast< PostMaster* >( Id::postId( i ).eref().data() );
		pm->numAsyncOut_ = 0;
		pm->numAsyncIn_ = 0;
	}
}

/**
 * This routine tests traversal of nodes following a string path to
 * find an id. We first create objects on remote nodes, and then
 * try to find them by their paths. Later extend to having children
 * on different nodes than parents.
 *
 * This test scrambles the synchrony between nodes. Don't use Barrier from
 * here on.
 */
void testParTraversePath()
{
	const unsigned int numPoll = 20;
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
	vector< Id > parents;
	vector< Id > children;
	if ( myNode == 0 ) {
		cout << "\ntesting parallel path traversal" << flush;
	}
	if ( myNode == 0 ) {
		Slot remoteCreateSlot = 
			initShellCinfo()->getSlot( "parallel.createSrc" );
		Slot parSetSlot = 
			initShellCinfo()->getSlot( "parallel.setSrc" );
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			char name[20];
			sprintf( name, "zug%d", i );
			string sname = name;
			unsigned int tgt = ( i < myNode ) ? i : i - 1;
			Id newId = Id::makeIdOnNode( i );
			parents.push_back( newId );
			// cout << "Create op: sendTo4( shellId, slot, " << tgt << ", " << "Neutral, " << sname << ", root, " << newId << endl;
			sendTo4< string, string, Nid, Nid >(
				Id::shellId().eref(), remoteCreateSlot, tgt,
				"Neutral", sname, 
				Id(), newId
			);
			Id newId2 = Id::makeIdOnNode( i );
			children.push_back( newId2 );
			sname = "child";
			sendTo4< string, string, Nid, Nid >(
				Id::shellId().eref(), remoteCreateSlot, tgt,
				"Neutral", sname, 
				newId, newId2
			);
		}
	}
	for ( unsigned int i = 0; i < numPoll; i++ ) {
		pollPostmaster();
		usleep( 10000 );
		MuMPI::INTRA_COMM().Barrier();
	}

	if ( myNode == 0 ) {
		for ( unsigned int i = 1; i < numNodes; i++ ) {
			char name[20];
			string sname;
			sprintf( name, "/zug%d/child", i );
			sname = name;
			Id checkId( sname );
			// cout << "On " << myNode << ", checking id for " << sname << ": " << checkId << endl;
			ASSERT( checkId.good(), "parallel obj lookup by path" );
			ASSERT( checkId == children[i - 1], "Parallel obj lookup by path");
		}
	} 
	// Get everyone past the point where ids are checked remotely,
	// and clean up.
	// Note that we cannot put a barrier in this loop because the remote
	// nodes need to poll till they can respond to the off-node 
	// checkId request.
	Id cjId = Id::localId( "/sched/cj" );
	assert( cjId.good() );
	for ( unsigned int i = 0; i < numNodes * 3; i++ ) {
		// set< int >( cjId.eref(), "step", 1 );
		pollPostmaster();
		usleep( 10000 );
	}
	// cout << "Past Id check loop on " << myNode << flush;
	MuMPI::INTRA_COMM().Barrier();

	if ( myNode != 0 ) {
		char name[20];
		sprintf( name, "/zug%d", myNode );
		string sname = name;
		Id checkId( sname );
		ASSERT( checkId.good(), "parallel obj lookup by path" );
		set( checkId.eref(), "destroy" );
	}
}

#endif // DO_UNIT_TESTS
#endif // USE_MPI
