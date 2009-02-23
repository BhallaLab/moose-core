#include "header.h"
#include <sys/time.h>
#include <math.h>
#include <queue>
#include "Reac.h"
#include "Mol.h"
#include "Tab.h"
#include "IntFire.h"

const FuncId ENDFUNC( -1 );

extern void testSync();
extern void testSyncArray( unsigned int size, unsigned int numThreads );

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
		testSyncArray( size, 1 );

	for ( unsigned int size = 10; size < 10001; size *= 10 )
		testSyncArray( size, 2 );
	for ( unsigned int size = 10; size < 10001; size *= 10 )
		testSyncArray( size, 4 );

	cout << endl;
	return 0;
}

