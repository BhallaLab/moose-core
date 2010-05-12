/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DiagonalMsg.h"
#include "OneToAllMsg.h"
#include "../scheduling/Tick.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"
#include "Arith.h"

void testArith()
{
	Id a1id = Id::nextId();
	vector< unsigned int > dims( 1, 10 );
	Element* a1 = new Element( a1id, Arith::initCinfo(), "a1", dims, 1 );

	Eref a1_0( a1, 0 );
	Eref a1_1( a1, 1 );

	Arith* data1_0 = reinterpret_cast< Arith* >( a1->dataHandler()->data1( 0 ) );
//	Arith* data1_1 = reinterpret_cast< Arith* >( a1->data1( 1 ) );

	data1_0->arg1( 1 );
	data1_0->arg2( 0 );

	ProcInfo p;
	data1_0->process( &p, a1_0 );

	assert( data1_0->getOutput() == 1 );

	data1_0->arg1( 1 );
	data1_0->arg2( 2 );

	data1_0->process( &p, a1_0 );

	assert( data1_0->getOutput() == 3 );

	a1id.destroy();

	cout << "." << flush;
}

/** 
 * This test uses the Diagonal Msg and summing in the Arith element to
 * generate a Fibonacci series.
 */
void testFibonacci()
{
	unsigned int numFib = 20;
	vector< unsigned int > dims( 1, numFib );

	Id a1id = Id::nextId();
	Element* a1 = new Element( a1id, Arith::initCinfo(), "a1", dims );

	Arith* data = reinterpret_cast< Arith* >( a1->dataHandler()->data1( 0 ) );
	if ( data ) {
		data->arg1( 0 );
		data->arg2( 1 );
	}

	const Finfo* outFinfo = Arith::initCinfo()->findFinfo( "output" );
	const Finfo* arg1Finfo = Arith::initCinfo()->findFinfo( "arg1" );
	const Finfo* arg2Finfo = Arith::initCinfo()->findFinfo( "arg2" );
	const Finfo* procFinfo = Arith::initCinfo()->findFinfo( "process" );
	DiagonalMsg* dm1 = new DiagonalMsg( a1, a1 );
	bool ret = outFinfo->addMsg( arg1Finfo, dm1->mid(), a1 );
	assert( ret );
	dm1->setStride( 1 );

	DiagonalMsg* dm2 = new DiagonalMsg( a1, a1 );
	ret = outFinfo->addMsg( arg2Finfo, dm2->mid(), a1 );
	assert( ret );
	dm1->setStride( 2 );

	/*
	bool ret = DiagonalMsg::add( a1, "output", a1, "arg1", 1 );
	assert( ret );
	ret = DiagonalMsg::add( a1, "output", a1, "arg2", 2 );
	assert( ret );
	*/

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	shell->setclock( 0, 1.0, 0 );
	Eref ticker = Id( 2 ).eref();

	const Finfo* proc0Finfo = Tick::initCinfo()->findFinfo( "process0" );
	OneToAllMsg* otam = new OneToAllMsg( ticker, a1 );
	ret = proc0Finfo->addMsg( procFinfo, otam->mid(), ticker.element() );

	// ret = OneToAllMsg::add( ticker, "process0", a1, "process" );
	assert( ret );

	shell->doStart( numFib );
	unsigned int f1 = 1;
	unsigned int f2 = 0;
	for ( unsigned int i = 0; i < numFib; ++i ) {
		if ( a1->dataHandler()->isDataHere( i ) ) {
			Arith* data = reinterpret_cast< Arith* >( a1->dataHandler()->data1( i ) );
			// cout << Shell::myNode() << ": i = " << i << ", " << data->getOutput() << ", " << f1 << endl;
			assert( data->getOutput() == f1 );
		}
		unsigned int temp = f1;
		f1 = temp + f2;
		f2 = temp;
	}

	a1id.destroy();
	cout << "." << flush;
}

/** 
 * This test uses the Diagonal Msg and summing in the Arith element to
 * generate a Fibonacci series.
 */
void testMpiFibonacci()
{
	unsigned int numFib = 20;
	vector< unsigned int > dims( 1, numFib );

	Id a1id = Id::nextId();
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );

	Element* a1 = new Element( a1id, Arith::initCinfo(), "a1", dims );

	Arith* data = reinterpret_cast< Arith* >( a1->dataHandler()->data1( 0 ) );

	if ( data ) {
		data->arg1( 0 );
		data->arg2( 1 );
	}

	MsgId mid1 = shell->doAddMsg( "Diagonal", 
		FullId( a1id, 0 ), "output", FullId( a1id, 0 ), "arg1" );
	const Msg* m1 = Msg::getMsg( mid1 );
	Eref er1 = m1->manager( m1->id() );
	bool ret = Field< int >::set( er1, "stride", 1 );
	assert( ret );

	MsgId mid2 = shell->doAddMsg( "Diagonal", 
		FullId( a1id, 0 ), "output", FullId( a1id, 0 ), "arg2" );
	const Msg* m2 = Msg::getMsg( mid2 );
	Eref er2 = m2->manager( m2->id() );
	ret = Field< int >::set( er2, "stride", 2 );
	assert( ret );
	
	/*
	bool ret = DiagonalMsg::add( a1, "output", a1, "arg1", 1 );
	assert( ret );
	ret = DiagonalMsg::add( a1, "output", a1, "arg2", 2 );
	assert( ret );
	*/

	shell->setclock( 0, 1.0, 0 );
	Eref ticker = Id( 2 ).eref();
//	ret = OneToAllMsg::add( ticker, "process0", a1, "process" );
//	assert( ret );

	shell->doStart( numFib );

	unsigned int f1 = 1;
	unsigned int f2 = 0;
	for ( unsigned int i = 0; i < numFib; ++i ) {
		if ( a1->dataHandler()->isDataHere( i ) ) {
			Arith* data = reinterpret_cast< Arith* >( a1->dataHandler()->data1( i ) );
			// cout << Shell::myNode() << ": i = " << i << ", " << data->getOutput() << ", " << f1 << endl;
			assert( data->getOutput() == f1 );
		}
		unsigned int temp = f1;
		f1 = temp + f2;
		f2 = temp;
	}

	a1id.destroy();
	cout << "." << flush;
}

void testBuiltins()
{
	testArith();
	testFibonacci();
}

void testMpiBuiltins( )
{
	//Need to update
// 	testMpiFibonacci();
}
