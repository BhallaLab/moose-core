#include "header.h"
#include "Reac.h"
#include "Mol.h"
#include "Tab.h"

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
	t1.print();

	delete e1;
	delete e2;
	delete e3;
	delete e4;
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
}

void testSynapse( )
{
}

int main()
{
	testSync();
	testAsync();
	testSynapse();

	return 0;
}

