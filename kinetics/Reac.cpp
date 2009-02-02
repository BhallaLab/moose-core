#include "header.h"
#include "Reac.h"

const Slot subSlot = 0;
const Slot prdSlot = 1;
const Slot reacSlot = 2;

Reac::Reac( double kf, double kb )
	: kf_( kf ), kb_( kb )
{
	;
}

void Reac::process( const ProcInfo* p, Eref e )
{
	double frate = e.prdBuf( subSlot, kf_ );
	double brate = e.prdBuf( prdSlot, kb_ );
	send2< double, double >( e, reacSlot, frate, brate );
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
}

void Reac::reinit( Eref e )
{
	;
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
