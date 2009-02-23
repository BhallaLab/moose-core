#include "header.h"
#include <sys/time.h>
#include <math.h>
#include "Reac.h"
#include "Mol.h"
#include "Tab.h"

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

void forceCheckVal( double time, Element* e, unsigned int size )
{
	static const double EPS = 1.0e-4;
	static const double THRESH = 1.0e-3;
	for ( unsigned int i = 0; i < size; ++i ) {
		const Mol* m = static_cast< const Mol* >( e->data( i ) );
		double y = 0.333333 * ( i + i + i * exp ( -time / 3.333333 ) );
		if ( y > THRESH ) {
			if ( fabs( 1.0 - m->n_ / y ) > EPS ) {
				cout << "forceCheckVal error: i = " << i << 
					", time = " << time << 
					" Terminating\n";
				exit( 1 );
			}
		} else {
			if ( fabs( m->n_ - y ) > EPS ) {
				cout << "forceCheckVal error: i = " << i << 
					", time = " << time << 
					" Terminating\n";
				exit( 1 );
			}
		}
		// cout << time << "	" << y << "	" << m[i].n_ << endl;
	}
}

double process( Element* e1, Element* e2, Element* e3, 
	const Mol* m1, unsigned int size, unsigned int numThreads );

void testSyncArray( unsigned int size, unsigned int numThreads )
{
	/////////////////////////////////////////////////////////////////////
	// Make objects
	/////////////////////////////////////////////////////////////////////
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
	// Do the calculations.
	/////////////////////////////////////////////////////////////////////
	double elapsedTime = process( e1, e2, e3, m1, size, numThreads );

	/////////////////////////////////////////////////////////////////////
	// Clean up and report.
	/////////////////////////////////////////////////////////////////////
	delete e1;
	delete e2;
	delete e3;
	delete[] m1;
	delete[] m2;
	delete[] r1;
	cout << "syncArray" << size << "	ops=" << size * 10000 <<
		"	time=" << elapsedTime <<
		"	" <<
		endl;
}

void *BusyWork( void* t )
{
	long tid;
	tid = *static_cast< long* >( t );
	cout << "t" << tid << " " << flush;
	return 0;
}

class ThreadInfo
{
	public:
		ThreadInfo()
			: e1( 0 ), e2( 0 ), e3( 0 ), 
			dt( 0.001 ), plotdt( 1 ), maxt( 0.0 ),
			threadNum( 0 ), numThreads( 1 ),
			barrier( 0 )
		{;}

		ThreadInfo( Element* a1, Element* a2, Element* a3, 
		double adt, double pdt, double maxtime,
		unsigned int tn, unsigned int nt,
		pthread_barrier_t* barr )
			: e1( a1 ), e2( a2 ), e3( a3 ), 
			dt( adt ), plotdt( pdt ), maxt( maxtime ),
			threadNum( tn ), numThreads( nt ),
			barrier( barr )
		{;}
		
		Element* e1;
		Element* e2;
		Element* e3;
		double dt;
		double plotdt;
		double maxt;
		unsigned int threadNum;
		unsigned int numThreads;
		pthread_barrier_t* barrier;
};

void* processThread( void* t )
{
	ThreadInfo* ti = static_cast< ThreadInfo* >( t );
	Element* e1 = ti->e1;
	Element* e2 = ti->e2;
	Element* e3 = ti->e3;
	unsigned int threadNum = ti->threadNum;
	ProcInfo p;
	p.dt = ti->dt;
	p.numThreads = ti->numThreads;

	for ( double pt = 0.0; pt < ti->maxt; pt += ti->plotdt ) {
		forceCheckVal( pt, e1, e1->numEntries() );
		for( double t = 0.0; t < ti->plotdt; t += ti->dt ) {
			p.currTime = t + pt;
			e3->process( &p, threadNum );
			pthread_barrier_wait( ti->barrier );
			e1->process( &p, threadNum );
			e2->process( &p, threadNum );
			pthread_barrier_wait( ti->barrier );
		}
	//	cout << "on thread " << threadNum << " time = " << pt << endl;
	}
	return 0;
}

double process( Element* e1, Element* e2, Element* e3, 
	const Mol* m1, unsigned int size, unsigned int numThreads )
{	
	e1->reinit();
	e2->reinit();
	e3->reinit();

	double dt = 0.001;
	double plotdt = 1.0;
	double maxt = 10.0;
	ProcInfo p;
	p.dt = dt;
	p.numThreads = numThreads;

	struct timeval tv_start;
	struct timeval tv_end;
	gettimeofday( &tv_start, 0 );
	if ( numThreads < 2 ) {
		for ( double pt = 0.0; pt < maxt; pt += plotdt ) {
			checkVal( pt, m1, size );
			for( double t = 0.0; t < plotdt; t += dt ) {
				p.currTime = t + pt;
				e3->process( &p );
				e1->process( &p );
				e2->process( &p );
			}
		}
	} else {
		vector< pthread_t > thread( numThreads );
		pthread_attr_t attr;
		pthread_barrier_t barrier;
		if ( pthread_barrier_init( &barrier, 0, numThreads ) ) {
			cout << "Could not create barrier\n";
			return 0.0;
		}
		int rc;
		long t;
		void *status;

		/* Initialize and set thread detached attribute */
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		cout << "Main: creating " << numThreads << " threads\n";
		vector< ThreadInfo > threadInfoVec( numThreads );
		for( t = 0; t < numThreads; t++) {
			cout << "c" << t << " " << flush;
			// ThreadInfo *ti = new ThreadInfo( e1, e2, e3, dt, t, numthreads);
			threadInfoVec[ t ] = 
				ThreadInfo( e1, e2, e3, dt, plotdt, maxt,
				t, numThreads,
				&barrier );
			rc = pthread_create(&thread[t], &attr, &processThread,
				static_cast< void* >( &( threadInfoVec[ t ] ) ) );
			if (rc) {
				cout << "ERROR; return code from pthread_create() is " <<
					rc << endl;
				exit(-1);
			}
		}
		cout << "Main: waiting for threads\n";
		pthread_attr_destroy(&attr);

		/* Wait on the other threads */
		for( t = 0; t < numThreads; t++) {
			rc = pthread_join(thread[t], &status);
			if (rc) {
				cout << "ERROR; return code from pthread_join() is " <<
					rc << endl;
				exit(-1);
			}
			cout << "j" << t << " " << flush;
		}
		cout << "Main: joined threads\n";
	}
	gettimeofday( &tv_end, 0 );
	return ( tv_end.tv_sec - tv_start.tv_sec ) +
		1.0e-6 * ( tv_end.tv_usec - tv_start.tv_usec );
}
