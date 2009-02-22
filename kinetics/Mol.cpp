#include <math.h>
#include "header.h"
#include "Mol.h"

#define EPSILON 1e-15

const Slot nSlot = 0;
const Slot aSlot = 0;
const Slot bSlot = 1;

/*
const Cinfo* initMolCinfo()
{
	static Finfo* molFinfos[] = 
	{
		new ValueFinfo( "n" ); 
	};
}

template< class T, class A, ( void T::F* )( A ) > 
	unsigned int adapt( Eref e, A arg )
{
	static_cast< T* >( e.data() )->F( *static_cast< A* >( arg ) );
	return sizeof( A );
}

unsigned int qFuncDouble( Eref e, char* arg )
{
	static_cast< Mol* >( e.data() )->setKf( 
		*static_cast< double* >( arg ) );
	return sizeof( double );
}

*/

Finfo** Mol::initClassInfo()
{
	/*
	static Finfo* molFinfos[] = {
		new Finfo( setN ),
		new Finfo( setNinit ),
	}

	return molFinfos;
	*/

	return 0;
}


void Mol::process( const ProcInfo* p, Eref e )
{
/*
	unsigned int synSize = sizeof( SynInfo );
	BufferInfo subInfo = e.processBuffer( substrateSlot );
	double frate = kf_;

	for ( const double* i = static_cast< const double* >( subinfo.begin ); 
		i != static_cast< const double* >( subinfo.end ); ++i)
		rate *= *i;

	for ( char* i = subInfo.begin; i != subInfo.end; i += sizeof( double ) )
		rate *= *static_cast< double* >( i );
	
	double brate = kb_;
	BufferInfo prdInfo = e.recvBuf( productSlot );
	for ( char* i = prdInfo.begin; i != prdInfo.end; i += sizeof( double ) )
		brate *= *static_cast< double* >( i );
	
	send< double, double >( e, reacSrcSlot, rate, brate );
	// send< double, double >( e, subPrdSlot, brate, rate );
	*/
	double A = e.sumBuf( aSlot );
	double B = e.sumBuf( bSlot );
	if ( n_ > EPSILON && B > EPSILON ) {
		double C = exp( -B * p->dt / n_ );
		n_ *= C + (A / B ) * ( 1.0 - C );
	} else {
		n_ += ( A - B ) * p->dt;
	}
	// send n out
	e.send1( nSlot, n_ );
	// send1< double >( e, nSlot, n_ );
}

void Mol::reinit( Eref e )
{
	n_ = nInit_;
	e.send1( nSlot, n_ );
	// send1< double >( e, nSlot, n_ );
}

/*
void Reac::clearQ( Eref e )
{
	BufferInfo q = e.asyncBuffer();
	const char* i = q.begin;
	while( i != q.end ) {
		
	}


	const char* i = e.generalQ.begin();
	while ( i != e.generalQ.end() ) {
		// FuncId* fi = static_cast< FuncId* >( i );
		// i += sizeof( FuncId );
		// i += fi->doOperation( e, i );
		// i += doOperation( *fi, e, i );
		unsigned int op = *static_cast< const unsigned int* >( i );
		i += sizeof( unsigned int );
		i += this->opVec_[ op ]( e, i );
			// opVec is set up statically, has the function ptrs.
			// All are of the form f( Eref e, const char* i ).
			// Probably need a functional static initialization.
	}
}
	*/

/*
unsigned int FuncId::doOperation( Eref e, char* i )
{
	unsigned int op = *static_cast< unsigned int* >( i );
	i += sizeof( unsigned int );
	return opVec_[ op ]( i ) + sizeof( unsigned int );
}
*/

/**
 * Problem here is that it isn't really like the functions defined for
 * the existing moose. Those do a static cast for the object, which
 * is eliminated here. Instead this does a static cast for arg data.
unsigned int Reac::setKf( Eref e, const char* i )
{
	kf_ = *static_cast< double* >( i );
	return sizeof( double );
}

 */
