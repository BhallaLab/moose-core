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

#include "../builtins/Arith.h"
#include "MsgManager.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "SingleMsg.h"

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

	vector< unsigned int > dimensions;
	dimensions.push_back( 1 );
	Id child = shell->doCreate( "Neutral", Id(), "test", dimensions );

	shell->doDelete( child );
	cout << "." << flush;
}

/**
 * Tests traversal through parents and siblings.
 */
void testTreeTraversal()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	vector< unsigned int > dimensions;
	dimensions.push_back( 1 );
	Id f1 = shell->doCreate( "Neutral", Id(), "f1", dimensions );
	Id f2a = shell->doCreate( "Neutral", f1, "f2a", dimensions );
	Id f2b = shell->doCreate( "Neutral", f1, "f2b", dimensions );
	Id f2c = shell->doCreate( "Neutral", f1, "f2c", dimensions );
	Id f3aa = shell->doCreate( "Neutral", f2a, "f3aa", dimensions );
	Id f3ab = shell->doCreate( "Neutral", f2a, "f3ab", dimensions );
	Id f3ba = shell->doCreate( "Neutral", f2b, "f3ba", dimensions );

	FullId pa = Field< FullId >::get( f3aa.eref(), "parent" );
	assert( pa == FullId( f2a, 0 ) );
	pa = Field< FullId >::get( f3ab.eref(), "parent" );
	assert( pa == FullId( f2a, 0 ) );
	pa = Field< FullId >::get( f2b.eref(), "parent" );
	assert( pa == FullId( f1, 0 ) );
	pa = Field< FullId >::get( f1.eref(), "parent" );
	assert( pa == FullId( Id(), 0 ) );

	vector< Id > kids = Field< vector< Id > >::get( f1.eref(), "children" );
	assert( kids.size() == 3 );
	assert( kids[0] == f2a );
	assert( kids[1] == f2b );
	assert( kids[2] == f2c );

	kids = Field< vector< Id > >::get( f2a.eref(), "children" );
	assert( kids.size() == 2 );
	assert( kids[0] == f3aa );
	assert( kids[1] == f3ab );
	
	kids = Field< vector< Id > >::get( f2b.eref(), "children" );
	assert( kids.size() == 1 );
	assert( kids[0] == f3ba );

	kids = Field< vector< Id > >::get( f2c.eref(), "children" );
	assert( kids.size() == 0 );

	shell->doDelete( f1 );
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


	const Cinfo* testSchedCinfo = TestSched::initCinfo();
	vector< unsigned int > dims;
	Id tsid = Id::nextId();
	Element* tse = new Element( tsid, testSchedCinfo, "tse", dims );

	// testThreadSchedElement tse;
	Eref ts( tse, 0 );
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

	tsid.destroy();
	cout << "." << flush;
}

/**
 * Tests Shell operations carried out on multiple nodes
 */
void testInterNodeOps() // redundant.
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	Id child;
	if ( shell->myNode() == 0 ) {
		vector< unsigned int > dimensions;
		dimensions.push_back( 6139 );
		child = shell->doCreate( "Neutral", Id(), "test", dimensions );
	} 
	// cout << shell->myNode() << ": testInterNodeOps: #entries = " << child()->dataHandler()->numData() << endl;

	shell->doDelete( child );
	// child.destroy();
	cout << "." << flush;
}

void testShellSetGet()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	const unsigned int size = 100;
	vector< unsigned int > dimensions;
	dimensions.push_back( size );
	vector< double > val;

	// Set up the objects.
	Id a1 = shell->doCreate( "Arith", Id(), "a1", dimensions );

	// cout << Shell::myNode() << ": testShellSetGet: data here = (" << a1()->dataHandler()->begin() << " to " << a1()->dataHandler()->end() << ")" << endl;
	for ( unsigned int i = 0; i < size; ++i ) {
		val.push_back( i * i * i ); // use i^3 as a simple test.
		bool ret = SetGet1< double >::set( Eref( a1(), i ), "set_outputValue", i * i );
		assert( ret );
	}
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = Field< double >::get( Eref( a1(), i ), "outputValue" );
		assert( fabs( x - i * i ) < 1e-6 );
	}
	bool ret = SetGet1< double >::setVec( Eref( a1(), 0 ), "set_outputValue", val );
	assert( ret );
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = Field< double >::get( Eref( a1(), i ), "outputValue" );
		assert( fabs( x - i * i * i ) < 1e-6 );
	}

	shell->doDelete( a1 );
	cout << "." << flush;
}

void testShellSetGetVec()
{
	/*
	unsigned int numCells = 65535;
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< unsigned int > dimensions;
	dimensions.push_back( numCells );
	Id child = shell->doCreate( "IntFire", Id(), "test", dimensions );
	for ( unsigned int i = 0; i < numCells; ++i )
		numSyn.push_back( i % 32 );
	shell->doSetUintVec( child, DataId( i ), "numSynapses", numSyn );
	numSyn.clear();
	shell->doGetUintVec( child, DataId( i ), "numSynapses", numSyn );
	for ( unsigned int i = 0; i < numCells; ++i )
		assert( numSyn[i] == i % 32 );
	// Also want to assign each of the synapses to a different value.
	
	shell->doDelete( child );
	cout << "." << flush;
	*/
}

bool setupSched( Shell* shell, FullId& tick, Id dest )
{
	MsgId ret = shell->doAddMsg( "OneToAll", 
		tick, "process0", FullId( dest, 0 ), "process" );
	return ( ret != Msg::badMsg );
}

bool checkOutput( Id e, 
	double v0, double v1, double v2, double v3, double v4 )
{
	bool ret = 1;
	Eref e0( e(), 0 );
	double val = Field< double >::get( e0, "outputValue" );
	ret = ret && ( fabs( val - v0 ) < 1e-6 );
	// cout << "( " << v0 << ", " << val << " ) ";

	Eref e1( e(), 1 );
	val = Field< double >::get( e1, "outputValue" );
	ret = ret && ( fabs( val - v1 ) < 1e-6 );
	// cout << "( " << v1 << ", " << val << " ) ";

	Eref e2( e(), 2 );
	val = Field< double >::get( e2, "outputValue" );
	ret = ret && ( fabs( val - v2 ) < 1e-6 );
	// cout << "( " << v2 << ", " << val << " ) ";

	Eref e3( e(), 3 );
	val = Field< double >::get( e3, "outputValue" );
	ret = ret && ( fabs( val - v3 ) < 1e-6 );
	// cout << "( " << v3 << ", " << val << " ) ";

	Eref e4( e(), 4 );
	val = Field< double >::get( e4, "outputValue" );
	ret = ret && ( fabs( val - v4 ) < 1e-6 );
	// cout << "( " << v4 << ", " << val << " )\n";

	return ret;
}

void testShellAddMsg()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< unsigned int > dimensions;
	dimensions.push_back( 5 );


	///////////////////////////////////////////////////////////
	// Set up the objects.
	///////////////////////////////////////////////////////////
	Id a1 = shell->doCreate( "Arith", Id(), "a1", dimensions );
	Id a2 = shell->doCreate( "Arith", Id(), "a2", dimensions );

	Id b1 = shell->doCreate( "Arith", Id(), "b1", dimensions );
	Id b2 = shell->doCreate( "Arith", Id(), "b2", dimensions );

	Id c1 = shell->doCreate( "Arith", Id(), "c1", dimensions );
	Id c2 = shell->doCreate( "Arith", Id(), "c2", dimensions );

	Id d1 = shell->doCreate( "Arith", Id(), "d1", dimensions );
	Id d2 = shell->doCreate( "Arith", Id(), "d2", dimensions );

	Id e1 = shell->doCreate( "Arith", Id(), "e1", dimensions );
	Id e2 = shell->doCreate( "Arith", Id(), "e2", dimensions );

	///////////////////////////////////////////////////////////
	// Set up initial conditions
	///////////////////////////////////////////////////////////
	bool ret = 0;
	vector< double > init; // 12345
	for ( unsigned int i = 1; i < 6; ++i )
		init.push_back( i );
	ret = SetGet1< double >::setVec( a1.eref(), "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( b1.eref(), "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( c1.eref(), "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( d1.eref(), "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( e1.eref(), "arg1", init ); // 12345
	assert( ret );

	///////////////////////////////////////////////////////////
	// Set up messaging
	///////////////////////////////////////////////////////////
	// Should give 04000
	MsgId m1 = shell->doAddMsg( "Single", 
		FullId( a1, 3 ), "output", FullId( a2, 1 ), "arg1" );
	assert( m1 != Msg::badMsg );

	// Should give 33333
	MsgId m2 = shell->doAddMsg( "OneToAll", 
		FullId( b1, 2 ), "output", FullId( b2, 0 ), "arg1" );
	assert( m2 != Msg::badMsg );

	// Should give 12345
	MsgId m3 = shell->doAddMsg( "OneToOne", 
		FullId( c1, 0 ), "output", FullId( c2, 0 ), "arg1" );
	assert( m3 != Msg::badMsg );

	// Should give 01234
	MsgId m4 = shell->doAddMsg( "Diagonal", 
		FullId( d1, 0 ), "output", FullId( d2, 0 ), "arg1" );
	assert( m4 != Msg::badMsg );

	// Should give 54321
	MsgId m5 = shell->doAddMsg( "Sparse", 
		FullId( e1, 0 ), "output", FullId( e2, 0 ), "arg1" );
	assert( m5 != Msg::badMsg );

	const Msg* m5p = Msg::getMsg( m5 );
	Eref m5er = m5p->manager( m5p->id() );

	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5er, "setEntry", 0, 4, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5er, "setEntry", 1, 3, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5er, "setEntry", 2, 2, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5er, "setEntry", 3, 1, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5er, "setEntry", 4, 0, 0 );
	assert( ret );

	ret = SetGet1< unsigned int >::set(
		m5er, "loadBalance", Shell::numCores() );
	assert( ret );
	
	///////////////////////////////////////////////////////////
	// Set up scheduling
	///////////////////////////////////////////////////////////
	shell->setclock( 0, 1.0, 0 );

	FullId tick( Id( 2 ), 0 );
	ret = setupSched( shell, tick, a1 ); assert( ret );
	ret = setupSched( shell, tick, a2 ); assert( ret );
	ret = setupSched( shell, tick, b1 ); assert( ret );
	ret = setupSched( shell, tick, b2 ); assert( ret );
	ret = setupSched( shell, tick, c1 ); assert( ret );
	ret = setupSched( shell, tick, c2 ); assert( ret );
	ret = setupSched( shell, tick, d1 ); assert( ret );
	ret = setupSched( shell, tick, d2 ); assert( ret );
	ret = setupSched( shell, tick, e1 ); assert( ret );
	ret = setupSched( shell, tick, e2 ); assert( ret );

	///////////////////////////////////////////////////////////
	// Run it
	///////////////////////////////////////////////////////////
	
	shell->doStart( 2 );

	///////////////////////////////////////////////////////////
	// Check output.
	///////////////////////////////////////////////////////////
	
	ret = checkOutput( a2, 0, 4, 0, 0, 0 );
	assert( ret );
	ret = checkOutput( b1, 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( b2, 3, 3, 3, 3, 3 );
	assert( ret );
	ret = checkOutput( c2, 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( d2, 0, 1, 2, 3, 4 );
	assert( ret );
	ret = checkOutput( e2, 5, 4, 3, 2, 1 );
	assert( ret );

	///////////////////////////////////////////////////////////
	// Clean up.
	///////////////////////////////////////////////////////////
	shell->doDelete( a1 );
	shell->doDelete( a2 );
	shell->doDelete( b1 );
	shell->doDelete( b2 );
	shell->doDelete( c1 );
	shell->doDelete( c2 );
	shell->doDelete( d1 );
	shell->doDelete( d2 );
	shell->doDelete( e1 );
	shell->doDelete( e2 );

	cout << "." << flush;
}

void testShellParserQuit()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	shell->doQuit( );
	cout << "." << flush;
}

void testShell( )
{
//	testCreateDelete();
	// testShellParserQuit();
}

void testMpiShell( )
{
	testShellParserCreateDelete();
	testTreeTraversal();
	testShellSetGet();
	testInterNodeOps();
	testShellAddMsg();
}
