#include "header.h"
#include <sys/time.h>
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
	vector< Data* > v1( 1, &m1 );
	Element* e1 = new Element( v1, 1, 1 );

	Mol m2( 0.0 );
	vector< Data* > v2( 1, &m2 );
	Element* e2 = new Element( v2, 1, 1 );

	Reac r1( 0.2, 0.1 );
	vector< Data* > v3( 1, &r1 );
	Element* e3 = new Element( v3, 1, 1 );

	Tab t1;
	vector< Data* > v4( 1, &t1 );
	Element* e4 = new Element( v4, 1, 1 );

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

void checkVal( double time, const Mol* m, unsigned int size )
{
	static const double EPS = 1.0e-4;
	static const double THRESH = 1.0e-3;
	for ( unsigned int i = 0; i < size; ++i ) {
		double y = 0.333333 * ( i + i + i * exp ( -time / 3.333333 ) );
		if ( y > THRESH )
			assert( fabs( 1.0 - m[i].n_ / y ) < EPS );
		else
			assert( fabs( m[i].n_ - y ) < EPS );
		// cout << time << "	" << y << "	" << m[i].n_ << endl;
	}
}

void testSyncArray( unsigned int size )
{
	// Make objects
	Mol* m1 = new Mol[ size ];
	vector< Data* > v1( size );
	for ( unsigned int i = 0; i != size; ++i ) {
		m1[i].nInit_ = i; // This means that each calculation is unique.
		v1[i] = &( m1[ i ] );
	}
	Element* e1 = new Element( v1, 1, 2 );
	e1->sendBuf_ = new double[ size ]; // One double goes out for each molecule

	Mol* m2 = new Mol[ size ]; // Default initializer: nInit = 0.0
	vector< Data* > v2( size );
	for ( unsigned int i = 0; i != size; ++i )
		v2[i] = &( m2[ i ] );
	Element* e2 = new Element( v2, 1, 2 );
	e2->sendBuf_ = new double[ size ]; // One double goes out for each molecule

	Reac* r1 = new Reac[ size ]; // Default constructor: kf = 0.1, kb = 0.2
	vector< Data* > v3( size );
	for ( unsigned int i = 0; i != size; ++i )
		v3[i] = &( r1[ i ] );
	Element* e3 = new Element( v3, 2, 2 ); // Uses 2 outgoing msg slots
	e3->sendBuf_ = new double[ size * 2 ]; // One double each for A and B.

	// Unlike the previous test, here we will need to check the results
	// of the calculations on the
	// fly because we don't want to limit ourselves with space for
	// storing all the output.

	/////////////////////////////////////////////////////////////////////
	// Set up messaging
	/////////////////////////////////////////////////////////////////////
	
	e1->procBuf_.resize( size * 2 ); // Has A & B inputs for each mol.
	e1->procBufRange_.resize( size * 2 + 1, 0 ); // Only 2 msg types
	e2->procBuf_.resize( size * 2 ); // Has A & B inputs for each mol.
	e2->procBufRange_.resize( size * 2 + 1, 0 ); // Only 2 msg types
	e3->procBuf_.resize( size * 2 ); // Has sub and prd inputs for each reac
	e3->procBufRange_.resize( size * 2 + 1, 0 ); // Only 2 msg types
	for ( unsigned int i = 0; i != size; ++i ) {
		unsigned int j = i * 2; // index of B in send buf in reac.
		unsigned int k = i * 2; // Index of procBuf.
		unsigned int p = i * 2; // counter for procBufRange index.
		// By a coincidence all three indices match. Since this
		// is a test run, I retain all for clarity.

		e1->procBuf_[k] = &e3->sendBuf_[ j + 1 ]; // B: sum input
		e1->procBuf_[k + 1] =  &e3->sendBuf_[ j ]; // A: subtract input

		e1->procBufRange_[p] = k; // start index for procBuf sum input
		e1->procBufRange_[p + 1] = k + 1; // start index procBuf subtract
		e1->procBufRange_[p + 2] = k + 2; // End index procBuf subtract

		e2->procBuf_[k] = &e3->sendBuf_[ j ]; // A: sum input
		e2->procBuf_[k + 1] =  &e3->sendBuf_[ j + 1 ]; // B: subtract input

		e2->procBufRange_[p] = k; // start index for procBuf sum input
		e2->procBufRange_[p + 1] = k + 1; // start index procBuf subtract
		e2->procBufRange_[p + 2] = k + 2; // End index procBuf subtract

		e3->procBuf_[k] = &e1->sendBuf_[ i ]; // substrate input.
		e3->procBuf_[k + 1] =  &e2->sendBuf_[ i ]; // product input

		e3->procBufRange_[p] = k; // start index for procBuf sum input
		e3->procBufRange_[p + 1] = k + 1; // start index procBuf subtract
		e3->procBufRange_[p + 2] = k + 2; // End index procBuf subtract
	}
	
	/////////////////////////////////////////////////////////////////////
	// Process
	/////////////////////////////////////////////////////////////////////
	
	e1->reinit();
	e2->reinit();
	e3->reinit();

	double dt = 0.001;
	double plotdt = 1.0;
	double maxt = 10.0;
	ProcInfo p;
	p.dt = dt;

	struct timeval tv_start;
	struct timeval tv_end;
	gettimeofday( &tv_start, 0 );
	for ( double pt = 0.0; pt < maxt; pt += plotdt ) {
		checkVal( pt, m1, size );
		for( double t = 0.0; t < plotdt; t += dt ) {
			p.currTime = t + pt;
			e3->process( &p );
			e1->process( &p );
			e2->process( &p );
		}
	}
	gettimeofday( &tv_end, 0 );

	delete e1;
	delete e2;
	delete e3;
	delete[] m1;
	delete[] m2;
	delete[] r1;
	cout << "syncArray" << size << "	ops=" << size * maxt / dt <<
		"	time=" <<
		( tv_end.tv_sec - tv_start.tv_sec ) +
		1.0e-6 * ( tv_end.tv_usec - tv_start.tv_usec ) <<
		"	" <<
		endl;
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
	vector< Data* > v1( 1, &r1 );
	Element* e1 = new Element( v1, 1, 1 );

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

void testStandaloneIntFire( )
{
	const double EPSILON = 1e-6;
	IntFire f3( 1, 0.005 );
	vector< Data* > v3( 1, &f3 );
	Element* e3 = new Element( v3, 1, 1 );

	// SynInfo( weight, delay )
	f3.synapses_.push_back( SynInfo( 0.5, 0.001 ) );
	f3.synapses_.push_back( SynInfo( 2.0, 0.005 ) );

	cout << "IntFire..." << flush;
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
		if ( fabs( t - 0.006 ) < EPSILON ) // just after first input
			assert( fabs( f3.Vm_ - 0.4 ) < EPSILON );
		if ( fabs( t - 0.015 ) < EPSILON ) // just after second spike: fires
			assert( fabs( f3.Vm_ ) < EPSILON );
		if ( fabs( t - 0.021 ) < EPSILON ) // just after third input
			assert( fabs( f3.Vm_ - 0.4 ) < EPSILON );
		if ( fabs( t - 0.025 ) < EPSILON ) // just after last spike: fires
			assert( fabs( f3.Vm_ ) < EPSILON );
		// cout << f3.Vm_ << endl;
	}
}

void testSynapse( )
{
	testStandaloneIntFire();
	// Make objects. f1 and f2 connect into f3 and f4
	// IntFire f1( thresh, tau );
	IntFire f1( 1, 0.005 );
	vector< Data* > v1( 1, &f1 );
	Element* e1 = new Element( v1, 1, 1 );
	// SynInfo( weight, delay )
	f1.synapses_.push_back( SynInfo( 2, 0.001 ) );
	e1->msg_.resize( 2 );


	IntFire f2( 1, 0.005 );
	vector< Data* > v2( 1, &f2 );
	Element* e2 = new Element( v2, 1, 1 );
	f2.synapses_.push_back( SynInfo( 2, 0.003 ) );
	e2->msg_.resize( 2 );

	One2OneMsg m1( e1, e2 );
	One2OneMsg m2( e2, e1 );

	e1->msg_[0].push_back( &m1 );
	e1->msg_[1].push_back( &m2 );

	e2->msg_[0].push_back( &m2 );
	e2->msg_[1].push_back( &m1 );

	// Here we set up a spike to get the system rolling...
	// addSpike( id, time )
	f1.addSpike( 0, 0.005 );

	// Now off we go:
	double dt = 0.001;
	double maxt = 0.03;
	ProcInfo p;
	p.dt = dt;
	for( double t = 0.0; t < maxt; t += dt ) {
		p.currTime = t;
		e1->process( &p );
		e2->process( &p );
		cout << t << "	" << f1.Vm_ << "	" << f2.Vm_ << endl;
	}
}

int main()
{
	cout << "testing: ";
//	testSync();
	testAsync();
	testSynapse();

	for ( unsigned int size = 10; size < 10001; size *= 10 )
	// Put timing stuff here.
		testSyncArray( size );

	cout << endl;
	return 0;
}

