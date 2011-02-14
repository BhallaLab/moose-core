/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../builtins/Arith.h"

#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"

void testAssortedMsg()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< unsigned int > dimensions;
	Id pa = shell->doCreate( "Neutral", Id(), "pa", dimensions );
	dimensions.push_back( 5 );


	///////////////////////////////////////////////////////////
	// Set up the objects.
	///////////////////////////////////////////////////////////
	Id a1 = shell->doCreate( "Arith", pa, "a1", dimensions );
	Id a2 = shell->doCreate( "Arith", pa, "a2", dimensions );

	Id b1 = shell->doCreate( "Arith", pa, "b1", dimensions );
	Id b2 = shell->doCreate( "Arith", pa, "b2", dimensions );

	Id c1 = shell->doCreate( "Arith", pa, "c1", dimensions );
	Id c2 = shell->doCreate( "Arith", pa, "c2", dimensions );

	Id d1 = shell->doCreate( "Arith", pa, "d1", dimensions );
	Id d2 = shell->doCreate( "Arith", pa, "d2", dimensions );

	Id e1 = shell->doCreate( "Arith", pa, "e1", dimensions );
	Id e2 = shell->doCreate( "Arith", pa, "e2", dimensions );

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
	Eref m5er = m5p->manager();

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

	/*
	ret = SetGet1< unsigned int >::set(
		m5er, "loadBalance", Shell::numCores() );
		*/
	assert( ret );

	///////////////////////////////////////////////////////////
	// Test traversal
	///////////////////////////////////////////////////////////
	// Single
	FullId f = Msg::getMsg( m1 )->findOtherEnd( FullId( a1, 3 ) );
	assert( f == FullId( a2, 1 ) );

	f = Msg::getMsg( m1 )->findOtherEnd( FullId( a2, 1 ) );
	assert( f == FullId( a1, 3 ) );

	f = Msg::getMsg( m1 )->findOtherEnd( FullId( a1, 0 ) );
	assert( f == FullId( a2, DataId::bad() ) );

	f = Msg::getMsg( m1 )->findOtherEnd( FullId( a2, 0 ) );
	assert( f == FullId( a1, DataId::bad() ) );

	f = Msg::getMsg( m1 )->findOtherEnd( FullId( b2, 1 ) );
	assert( f == FullId::bad() );

	// OneToAll
	f = Msg::getMsg( m2 )->findOtherEnd( FullId( b1, 2 ) );
	assert( f == FullId( b2, 0 ) );

	f = Msg::getMsg( m2 )->findOtherEnd( FullId( b2, 0 ) );
	assert( f == FullId( b1, 2 ) );
	f = Msg::getMsg( m2 )->findOtherEnd( FullId( b2, 1 ) );
	assert( f == FullId( b1, 2 ) );
	f = Msg::getMsg( m2 )->findOtherEnd( FullId( b2, 2 ) );
	assert( f == FullId( b1, 2 ) );
	f = Msg::getMsg( m2 )->findOtherEnd( FullId( b2, 3 ) );
	assert( f == FullId( b1, 2 ) );
	f = Msg::getMsg( m2 )->findOtherEnd( FullId( b2, 4 ) );
	assert( f == FullId( b1, 2 ) );

	f = Msg::getMsg( m2 )->findOtherEnd( FullId( b1, 0 ) );
	assert( f == FullId( b2, DataId::bad() ) );

	f = Msg::getMsg( m2 )->findOtherEnd( FullId( a2, 1 ) );
	assert( f == FullId::bad() );

	// OneToOne
	for ( unsigned int i = 0; i < 5; ++i ) {
		f = Msg::getMsg( m3 )->findOtherEnd( FullId( c1, i ) );
		assert( f == FullId( c2, i ) );
		f = Msg::getMsg( m3 )->findOtherEnd( FullId( c2, i ) );
		assert( f == FullId( c1, i ) );
	}
	f = Msg::getMsg( m3 )->findOtherEnd( FullId( a2, 1 ) );
	assert( f == FullId::bad() );

	// Diagonal
	for ( unsigned int i = 0; i < 4; ++i ) {
		f = Msg::getMsg( m4 )->findOtherEnd( FullId( d1, i ) );
		assert( f == FullId( d2, i + 1 ) );
		f = Msg::getMsg( m4 )->findOtherEnd( FullId( d2, i + 1 ) );
		assert( f == FullId( d1, i ) );
	}
	f = Msg::getMsg( m4 )->findOtherEnd( FullId( d1, 4 ) );
	assert( f == FullId( d2, DataId::bad() ) );
	f = Msg::getMsg( m4 )->findOtherEnd( FullId( d2, 0 ) );
	assert( f == FullId( d1, DataId::bad() ) );

	f = Msg::getMsg( m4 )->findOtherEnd( FullId( a2, 1 ) );
	assert( f == FullId::bad() );

	// Sparse
	for ( unsigned int i = 0; i < 5; ++i ) {
		f = Msg::getMsg( m5 )->findOtherEnd( FullId( e1, i ) );
		assert( f == FullId( e2, 4 - i ) );
		f = Msg::getMsg( m5 )->findOtherEnd( FullId( e2, i ) );
		assert( f == FullId( e1, 4 - i ) );
	}

	f = Msg::getMsg( m5 )->findOtherEnd( FullId( a2, 1 ) );
	assert( f == FullId::bad() );

	cout << "." << flush;

	///////////////////////////////////////////////////////////
	// Check lookup by funcId.
	///////////////////////////////////////////////////////////
	const Finfo* aFinfo = Arith::initCinfo()->findFinfo( "arg1" );
	FuncId afid = dynamic_cast< const DestFinfo* >( aFinfo )->getFid();

	MsgId m = a2()->findCaller( afid );
	assert ( m == m1 );
	m = b2()->findCaller( afid );
	assert ( m == m2 );
	m = c2()->findCaller( afid );
	assert ( m == m3 );
	m = d2()->findCaller( afid );
	assert ( m == m4 );
	m = e2()->findCaller( afid );
	assert ( m == m5 );

	///////////////////////////////////////////////////////////
	// Clean up.
	///////////////////////////////////////////////////////////
	shell->doDelete( pa );
	/*
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
	*/

	cout << "." << flush;
}



void testMsg()
{
	testAssortedMsg();
}

void testMpiMsg( )
{
	;
}
