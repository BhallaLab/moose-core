#include "header.h"
#include <math.h>
#include <queue>
#include "Reac.h"
#include "Mol.h"
#include "Tab.h"
#include "IntFire.h"

const FuncId ENDFUNC( -1 );

void testSync()
{
	// Make objects
	Mol m1( 1.0 );
	Element* e1 = new Element( &m1 );

	Mol m2( 0.0 );
	Element* e2 = new Element( &m2 );

	Reac r1( 0.2, 0.1 );
	Element* e3 = new Element( &r1 );

	Tab t1;
	Element* e4 = new Element( &t1 );

	/////////////////////////////////////////////////////////////////////
	// Set up messaging
	/////////////////////////////////////////////////////////////////////
//	Edge sub( e1, e3 );
//	Edge prd( e3, e2 );
//
	// Here is the buffer.
	// Entry 0: n of m1
	// Entry 1: n of m2
	// Entry 2: frate of r1
	// Entry 3: brate of r1
	
	vector< double > buffer( 4, 0.0 );
	e1->procBuf_.push_back( &buffer[3] );	// A
	e1->procBuf_.push_back( &buffer[2] );	// B
	e1->procBuf_.push_back( &buffer[0] );	// n
	e1->procBufRange_.push_back( 0 );
	e1->procBufRange_.push_back( 1 );
	e1->procBufRange_.push_back( 2 );
	e1->procBufRange_.push_back( 3 );

	e2->procBuf_.push_back( &buffer[2] );	// A
	e2->procBuf_.push_back( &buffer[3] );	// B
	e2->procBuf_.push_back( &buffer[1] );	// n
	e2->procBufRange_.push_back( 0 );
	e2->procBufRange_.push_back( 1 );
	e2->procBufRange_.push_back( 2 );
	e2->procBufRange_.push_back( 3 );

	e3->procBuf_.push_back( &buffer[0] );	// n of m1
	e3->procBuf_.push_back( &buffer[1] );	// n of m2
	e3->procBuf_.push_back( &buffer[2] );	// Aout
	e3->procBuf_.push_back( &buffer[3] );	// Bout
	e3->procBufRange_.push_back( 0 );
	e3->procBufRange_.push_back( 1 );
	e3->procBufRange_.push_back( 2 );
	e3->procBufRange_.push_back( 4 );

	e4->procBuf_.push_back( &buffer[0] );	// n of m1
	e4->procBufRange_.push_back( 0 );
	e4->procBufRange_.push_back( 1 );

	
	/////////////////////////////////////////////////////////////////////
	// Process
	/////////////////////////////////////////////////////////////////////
	
	e1->reinit();
	e2->reinit();
	e3->reinit();
	e4->reinit();

	double dt = 0.01;
	double plotdt = 1.0;
	double maxt = 100.0;
	ProcInfo p;
	p.dt = dt;
	for ( double pt = 0.0; pt < maxt; pt += plotdt ) {
		e4->process( &p );
		for( double t = 0.0; t < plotdt; t += dt ) {
			p.currTime = t + pt;
			e3->process( &p );
			e1->process( &p );
			e2->process( &p );
		}
	}
	e4->process( &p );

	// Dump data
	// t1.print();
	vector< double > other;
	for ( double x = 0.0; x < maxt; x += plotdt ) {
		other.push_back( 0.333333 + 0.666666 * ( exp( -x / 3.333333 ) ) );
		// cout << other.back() << endl;
	}
	assert( t1.equal( other, 0.001 ) );
	delete e1;
	delete e2;
	delete e3;
	delete e4;
	cout << "sync..." << flush;
}

void testAsync( )
{
	/*
	static Finfo* reacFinfos[] = {
		new Finfo f1( setKf ),
		new Finfo f2( setKb ),
	};
	*/

	// Make objects
	Reac r1( 0.2, 0.1 );
	Element* e1 = new Element( &r1 );

	vector< char > buffer( 100 );
	char* buf = &( buffer[0] );
	FuncId assignKf( 0 );
	FuncId assignKb( 1 );

	*static_cast< FuncId* >( static_cast< void* >( buf ) ) = assignKf;
	buf += sizeof( FuncId );
	*static_cast< double* >( static_cast< void* >( buf ) ) = 1234.0;
	buf += sizeof( double );

	*static_cast< FuncId* >( static_cast< void* >( buf ) ) = assignKb;
	buf += sizeof( FuncId );
	*static_cast< double* >( static_cast< void* >( buf ) ) = 4321.0;
	buf += sizeof( double );

	*static_cast< FuncId* >( static_cast< void* >( buf ) ) = ENDFUNC;
	buf += sizeof( FuncId );

	assert( r1.kf_ == 0.2 );
	assert( r1.kb_ == 0.1 );
	e1->clearQ( &( buffer[0] ) );
	assert( r1.kf_ == 1234.0 );
	assert( r1.kb_ == 4321.0 );

	delete e1;
	cout << "async..." << flush;
}

void testSynapse( )
{
	// Make objects. f1 and f2 connect into f3 and f4
	// IntFire f1( thresh, tau );
	IntFire f1( 1, 0.005 );
	Element* e1 = new Element( &f1 );

	IntFire f2( 1, 0.005 );
	Element* e2 = new Element( &f2 );

	IntFire f3( 1, 0.005 );
	Element* e3 = new Element( &f3 );

	IntFire f4( 1, 0.005 );
	Element* e4 = new Element( &f4 );

	// SynInfo( weight, delay )
	f3.synapses_.push_back( SynInfo( 0.5, 0.001 ) );
	f3.synapses_.push_back( SynInfo( 2.0, 0.005 ) );

	cout << "synapse..." << flush;
	cout << endl;

	double dt = 0.001;
	double maxt = 0.03;
	ProcInfo p;
	p.dt = dt;
	// addSpike( id, time )
	f3.addSpike( 0, 0.005 );
	f3.addSpike( 1, 0.010 );

	f3.addSpike( 0, 0.02 );
	f3.addSpike( 0, 0.022 );
	f3.addSpike( 0, 0.024 );
	e3->msg_.resize( 2 );
	for( double t = 0.0; t < maxt; t += dt ) {
		p.currTime = t;
		e3->process( &p );
		cout << f3.Vm_ << endl;
	}
}

int main()
{
	cout << "testing: ";
	testSync();
	testAsync();
	testSynapse();

	cout << endl;
	return 0;
}

