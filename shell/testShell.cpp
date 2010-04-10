/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Shell.h"
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "../scheduling/Tick.h"
#include "../scheduling/testScheduling.h"

void testCreateDelete()
{
	
	Eref ser = Id().eref();
	Id testId = Id::nextId();
	vector< unsigned int > dimensions;
	dimensions.push_back( 1 );
	// Need to get the id back so that I can delete it later.
	bool ret = SetGet5< string, Id, Id, string, vector< unsigned int > >::set( ser, "create", "Neutral", Id(), testId , "testCreate", dimensions );
	assert( ret );

	ret = SetGet1< Id >::set( ser, "delete", testId );
	assert( ret );

	cout << "." << flush;
}


/**
 * Tests Create and Delete calls issued through the parser interface,
 * which internally sets up blocking messaging calls.
 */
void testShellParserCreateDelete()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

#ifdef USE_MPI
	// cout << shell->myNode() << " testShellParserCreateDelete: before barrier\n";
//	MPI_Barrier( MPI_COMM_WORLD );
	// cout << shell->myNode() << " testShellParserCreateDelete: after barrier\n";
#endif

	if ( shell->myNode() != 0 ) {
		Id child = Id::nextId();
		cout << shell->myNode() << " testShellParserCreateDelete: child=" << child << endl;
		while ( !child() ) // Wait till it is created
			shell->passThroughMsgQs( sheller.element() );
		while ( child() ) // Wait till it is destroyed
			shell->passThroughMsgQs( sheller.element() );
		return;
	}
//	sheller.element()->showFields();
//	sheller.element()->showMsg();

	vector< unsigned int > dimensions;
	dimensions.push_back( 1 );
	Id child = shell->doCreate( "Neutral", Id(), "test", dimensions );
	cout << shell->myNode() << " testShellParserCreateDelete: child=" << child << endl;

	shell->doDelete( child );
//	shell->doQuit( );
	cout << "." << flush;
}

// Here we create the element independently on each node, and connect
// it up independently. Using the doAddMsg we will be able to do this
// automatically on all nodes.
void testShellParserStart()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	shell->setclock( 0, 5.0, 0 );
	shell->setclock( 1, 2.0, 0 );
	shell->setclock( 2, 2.0, 1 );
	shell->setclock( 3, 1.0, 0 );
	shell->setclock( 4, 3.0, 5 );
	shell->setclock( 5, 5.0, 1 );

	testThreadSchedElement tse;
	Eref ts( &tse, 0 );
	Element* ticke = Id( 2 )();
	Eref er0( ticke, DataId( 0, 0 ) );
	Eref er1( ticke, DataId( 0, 1 ) );
	Eref er2( ticke, DataId( 0, 2 ) );
	Eref er3( ticke, DataId( 0, 3 ) );
	Eref er4( ticke, DataId( 0, 4 ) );
	Eref er5( ticke, DataId( 0, 5 ) );

	// No idea what FuncId to use here. Assume 0.
	FuncId f( 0 );
	SingleMsg m0( er0, ts ); er0.element()->addMsgAndFunc( m0.mid(), f, 0 );
	SingleMsg m1( er1, ts ); er1.element()->addMsgAndFunc( m1.mid(), f, 1 );
	SingleMsg m2( er2, ts ); er2.element()->addMsgAndFunc( m2.mid(), f, 2 );
	SingleMsg m3( er3, ts ); er3.element()->addMsgAndFunc( m3.mid(), f, 3 );
	SingleMsg m4( er4, ts ); er4.element()->addMsgAndFunc( m4.mid(), f, 4 );
	SingleMsg m5( er5, ts ); er5.element()->addMsgAndFunc( m5.mid(), f, 5 );

	if ( shell->myNode() != 0 )
		return;

	shell->doStart( 10 );
	cout << "." << flush;
}

/**
 * Tests Shell operations carried out on multiple nodes
 */
void testInterNodeOps()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	Id child;
	if ( shell->myNode() == 0 ) {
		vector< unsigned int > dimensions;
		// dimensions.push_back( shell->numNodes() + 1 );
		dimensions.push_back( 6139 );
		child = shell->doCreate( "Neutral", Id(), "test", dimensions );
	} else {
		child = Id::nextId();
		while ( !child() )
			shell->passThroughMsgQs( sheller.element() );
		shell->passThroughMsgQs( sheller.element() );
	}
	cout << shell->myNode() << ": testInterNodeOps: #entries = " <<
		child()->numData() << endl;

	child.destroy();
	cout << "." << flush;
}

void testShellAddMsg()
{
}

void testShellParserQuit()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	if ( shell->myNode() != 0 )
		return;
	shell->doQuit( );
	cout << "." << flush;
}

void testShell( )
{
	testCreateDelete();
	testShellParserCreateDelete();
	testInterNodeOps();
	testShellParserStart();
	testShellAddMsg();
	testShellParserQuit();
}
