#include "header.h"
#include <math.h>
#include "Tab.h"

const Slot vSlot = 0;

Finfo** Tab::initClassInfo()
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

void Tab::process( const ProcInfo* p, Eref e )
{
	double v = e.oneBuf( vSlot );
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
	vec_.push_back( v );
}

void Tab::reinit( Eref e )
{
	vec_.resize( 0 );
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

void Tab::print()
{
	for ( vector< double >::iterator i = vec_.begin(); i != vec_.end(); ++i)
		cout << *i << endl;
}

/**
 * Compares 'other' with self, point-to-point. If vector sizes
 * differ, does as much as it can.
 * epsilon is the tolerance. If the 'other' is greater than 1e-9,
 * it does a ratio comparison self/other, and epsilon is the difference of
 * this ratio from 1. If the 'other' is smaller than 1e-9 then
 * epsilon is the absolute limit.
 * returns 0 if vectors differ.
 */
bool Tab::equal( const vector< double >& other, double eps ) const
{
	static const double MINIMUM = 1e-9;
	unsigned int max = other.size();
	if ( max > vec_.size() )
		max = vec_.size();
	for ( unsigned int i = 0; i < max; ++i ) {
		if ( other[i] > MINIMUM ) {
			if ( fabs( 1.0 - vec_[i] / other[i] ) > eps )
				return 0;
		} else {
			if ( fabs( vec_[i] - other[i] ) > eps )
				return 0;
		}
	}
	return 1;
}
