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
#include "Arith.h"

void testArith()
{
	Id a1id = Id::nextId();
	Element* a1 = new Element( a1id, Arith::initCinfo(), "a1", 10 );

	Eref a1_0( a1, 0 );
	Eref a1_1( a1, 1 );

	Arith* data1_0 = reinterpret_cast< Arith* >( a1->data1( 0 ) );
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
	unsigned int NumFib = 10;

	Id a1id = Id::nextId();
	Element* a1 = new Element( a1id, Arith::initCinfo(), "a1", NumFib );

	Arith* data = reinterpret_cast< Arith* >( a1->data1( 0 ) );
	data->arg1( 0 );
	data->arg2( 1 );

	bool ret = DiagonalMsg::add( a1, "output", a1, "arg1", 1 );
	assert( ret );
	ret = DiagonalMsg::add( a1, "output", a1, "arg2", 2 );
	assert( ret );


	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	shell->setclock( 0, 1.0, 0 );
	Eref ticker = Id( 2 ).eref();
	ret = OneToAllMsg::add( ticker, "process0", a1, "process" );
	assert( ret );

	shell->doStart( NumFib );

	unsigned int f1 = 1;
	unsigned int f2 = 0;
	for ( unsigned int i = 0; i < NumFib; ++i ) {
		Arith* data = reinterpret_cast< Arith* >( a1->data1( i ) );
		assert( data->getOutput() == f1 );
		// cout << i << ", " << data->getOutput() << ", " << f1 << endl;
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
