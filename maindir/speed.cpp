/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <fstream>
#include "header.h"
#include "../builtins/String.h"
#include "../builtins/Int.h"
#include "TestMsg.h"
#include <sys/time.h>

static const long NMSGS = 100000;
static const long NREPS = 10;

class timer {
	public:
		void init() {
			gettimeofday( &tv, &tz );
		}

		double dt() {
			gettimeofday( &tv2, &tz );
			double ret = tv2.tv_sec - tv.tv_sec;
			ret += double( tv2.tv_usec - tv.tv_usec ) / 1.0e6;
			return ret;
		}

	private:

		struct timeval tv;
		struct timeval tv2;
		struct timezone tz;
};


// Here we check how fast it is to call a singlemsgsrc->singlemsgdest
// 1 million times
// Equivalent would be to call a function 1 million times.
void single_single()
{
	timer t;
	Element* src = Cinfo::find( "TestMsg" )->create( "src" );
	Element* dest = Cinfo::find( "TestMsg" )->create( "dest" );
	// TestMsg* tm = static_cast< TestMsg* >( src );
	Field sf = src->field( "oneout" );
	Field df = dest->field( "onein" );

	long i;
	int k;
	cout << "Test: Msg: single_single: ";

	SynConn< int > temp( src );
	t.init();
	for (k = 0; k < NREPS; k++ )
		for (i = 0; i < NMSGS; i++)
			TestMsg::inOneFunc( &temp, 1 );
			// tm->testInOneFunc( 1 );
	double dt1 = t.dt();
	setField< int >( src, "j", 0 );
	setField< int >( dest, "j", 0 );

	if ( sf.add( df ) ) {
		t.init();
		for (k = 0; k < NREPS; k++ )
			for (i = 0; i < NMSGS; i++)
				TestMsg::inOneFunc( &temp, 1 );
				// tm->testInOneFunc( 1 );
		double dt2 = t.dt();
		int j;
		int q;
		getField< int >( src, "j", q );
		getField< int >( dest, "i", k );
		getField< int >( dest, "j", j );

		if ( q == NMSGS * NREPS && j == NMSGS * NREPS && k == 3 ) {
		// To complete test need to do drop. if ( sf.drop( df ) )
			cout << "Passed.	" << 
				dt2 << " sec, ratio = " << ( dt2 - dt1 )/dt1  << "\n";
			return;
		}
	}
	cout << "Failed\n";
}

// Here we check how fast it is to ripple across NMSGS single msgs,
// in to out.
// Equivalent would be to call a function NMSGS times.
void single_ripple()
{
	cout << "Test: Msg: single_ripple: ";

	int i, j, k;
	timer t;
	TestMsg src( "src" );
	vector< TestMsg* > dest( NMSGS );
	for (i = 0; i < NMSGS; i++) {
		dest[ i ] = new TestMsg( "dest" );
	}

	SynConn< int > temp( &src );
	t.init();
	for (k = 0; k < NREPS; k++ )
		for (i = 0; i < NMSGS; i++)
			dest[ i ]->testInOneFunc( 1 );
	double dt1 = t.dt();
	setField< int >( &src, "i", 0 );
	setField< int >( &src, "j", 0 );

	Element* last = &src;
	int ngood = 0;

	for (i = 0; i < NMSGS; i++) {
		Field sf = last->field( "oneout" );
		last = dest[ i ];
		Field df = last->field( "onein" );
		ngood += sf.add( df );
	}

	if ( ngood != NMSGS ) {
		cout << "Failed to add messages.\n";
		return;
	}
	
	t.init();
	for (k = 0; k < NREPS; k++ )
		TestMsg::inOneFunc( &temp , 1 );
	double dt2 = t.dt();
	getField< int >( dest[ NMSGS - 1 ], "i", i );
	getField< int >( dest[ NMSGS - 1 ], "j", j );
	for (k = 0; k < NMSGS; k++) {
		delete dest[ k ];
	}
	if ( i == NMSGS + 2 && j == NREPS * 2 ) {
		// To complete test need to do drop. if ( sf.drop( df ) )
			cout << "Passed.	" << 
				dt2 << " sec, ratio = " << dt2 / dt1  << "\n";
			return;
	}
	cout << "Failed.\n";
}

// Here we check how fast it is to make a message with 1 million
// single targets, and call it once.

void multi_single()
{
	return;
	cout << "Test: Msg: multi_single: ";

	int i, j, k;
	timer t;
	TestMsg src( "src" );
	vector< TestMsg* > dest( NMSGS );
	for (i = 0; i < NMSGS; i++) {
		dest[ i ] = new TestMsg( "dest" );
	}

	SynConn< int > temp( &src );
	t.init();
	for (k = 0; k < NREPS; k++ )
		for (i = 0; i < NMSGS; i++)
			dest[ i ]->testInOneFunc( 1 );
	double dt1 = t.dt();
	setField< int >( &src, "i", 0 );
	setField< int >( &src, "j", 0 );

	int ngood = 0;

	Field sf = src.field( "out" );
	for (i = 0; i < NMSGS; i++) {
		Field df = dest[ i ]->field( "onein" );
		ngood += sf.add( df );
	}

	if ( ngood != NMSGS ) {
		cout << "Failed to add messages.\n";
		return;
	}
	
	t.init();
	for (k = 0; k < NREPS; k++ )
		TestMsg::inFunc( &temp , 10 );
	double dt2 = t.dt();
	getField< int >( dest[ NMSGS - 1 ], "i", i );
	getField< int >( dest[ NMSGS - 1 ], "j", j );
	ngood = 0;
	// for (k = NMSGS - 1; k >= 0; k-- )
	for (k = 0; k < NMSGS; k++)
	{
		Field df = dest[ k ]->field( "onein" );
		ngood += sf.drop( df );
		delete dest[ k ];
	}

	if ( ngood != NMSGS ) {
		cout << "Failed to drop messages. ngood = " << ngood << "\n";
		// return;
	}
	// cout << " i = " << i << ", j = " << j << " \n";
	if ( i == 101 && j == NREPS * 2 ) {
		// To complete test need to do drop. if ( sf.drop( df ) )
			cout << "Passed.	" << 
				dt2 << " sec, ratio = " << dt2 / dt1  << "\n";
			return;
	}
	cout << "Failed.\n";
}

// Here we check how fast it is to run synapse messages. 
// We have a multisrc and 1 synaptic target.

void multi_synapse()
{
	cout << "Test: Msg: multi-> syn: ";

	int i, j, k;
	timer t;
	TestMsg syn( "syn" );
	TestMsg src( "src" );

	SynConn< int > temp( &src );
	temp.value_ = 1;
	t.init();
	for (k = 0; k < NREPS; k++ )
		for (i = 0; i < NMSGS; i++)
			TestMsg::synFunc( &temp, 1 );
	double dt1 = t.dt();
	setField< int >( &src, "i", 0 );
	setField< int >( &src, "j", 0 );
	setField< int >( &syn, "i", 0 );
	setField< int >( &syn, "j", 0 );

	int ngood = 0;

	Field sf = src.field( "out" );
	Field df = syn.field( "syn" );
	for (i = 0; i < NMSGS; i++) {
		ngood += sf.add( df );
	}

	if ( ngood != NMSGS ) {
		cout << "Failed to add messages.\n";
		return;
	}
	
	t.init();
	for (k = 0; k < NREPS; k++ )
		TestMsg::inFunc( &temp , 1 );
	double dt2 = t.dt();
	/*
	getField< int >( &src, "i", i );
	getField< int >( &src, "j", j );
	cout << " i = " << i << ", j = " << j << " \n";
	*/
	getField< int >( &syn, "i", i );
	getField< int >( &syn, "j", j );
	ngood = 0;
	/*
	// for (k = NMSGS - 1; k >= 0; k-- )
	for (k = 0; k < NMSGS; k++)
	{
		ngood += sf.drop( df );
	}

	if ( ngood != NMSGS ) {
		cout << "Failed to drop messages. ngood = " << ngood << "\n";
		// return;
	}
	*/
	// cout << " i = " << i << ", j = " << j << " \n";
	if ( i == 0 && j == ( (NREPS * ( NREPS + 1 )) / 2 ) * NMSGS ) {
		// To complete test need to do drop. if ( sf.drop( df ) )
			cout << "Passed.	" << 
				dt2 << " sec, ratio = " << dt2 / dt1  << "\n";
			return;
	}
	cout << "Failed.\n";
}
